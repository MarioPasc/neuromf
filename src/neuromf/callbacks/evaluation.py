"""Two-tier evaluation callback: SWD (fast) + 2.5D FID (thorough).

Tier 1 (SWD): Every validation epoch — generates latents via 1-NFE EMA model,
computes Sliced Wasserstein Distance vs cached real latents. Fast (~2s).

Tier 2 (2.5D FID): Every ``fid_every_n_val_epochs`` validation epochs — decodes
latents through frozen MAISI VAE, extracts RadImageNet features from 3
orthogonal planes, computes FID. Enables FID-based early stopping.

Both tiers always run on the **first** validation epoch to establish a
lower-bound baseline (random-model performance).

All metrics are recorded in ``_eval_history`` and written to
``eval_summary.json`` at end of training (including early stop).

All computation is rank-0 only (DDP safe).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from neuromf.metrics.swd import compute_swd
from neuromf.sampling.one_step import sample_one_step

logger = logging.getLogger(__name__)


class EvaluationCallback(pl.Callback):
    """Two-tier evaluation: SWD every val epoch, 2.5D FID periodically.

    Args:
        n_swd_samples: Number of latents to generate for SWD.
        n_swd_projections: Number of random projections for SWD.
        n_real_cache: Number of real validation latents to cache.
        n_fid_samples: Number of latents to generate for FID.
        n_fid_real_samples: Number of real latents to decode for FID reference.
        fid_every_n_val_epochs: Tier 2 frequency (in validation epochs).
        center_slices_ratio: Fraction of center slices for 2.5D extraction.
        fid_weights_path: Path to RadImageNet ResNet-50 state dict.
        vae_config: Dict of VAE config params (for lazy loading).
        prediction_type: ``"u"`` or ``"x"`` prediction mode.
        cache_dir: Directory for caching real FID features to disk.
        early_stop_patience: Tier-2 evals without improvement before stopping.
        seed: Random seed for noise generation.
    """

    def __init__(
        self,
        n_swd_samples: int = 64,
        n_swd_projections: int = 128,
        n_real_cache: int = 200,
        n_fid_samples: int = 100,
        n_fid_real_samples: int = 200,
        fid_every_n_val_epochs: int = 2,
        center_slices_ratio: float = 0.6,
        fid_weights_path: str = "",
        vae_config: dict | None = None,
        prediction_type: str = "u",
        cache_dir: str = "",
        early_stop_patience: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._n_swd_samples = n_swd_samples
        self._n_swd_projections = n_swd_projections
        self._n_real_cache = n_real_cache
        self._n_fid_samples = n_fid_samples
        self._n_fid_real_samples = n_fid_real_samples
        self._fid_every_n_val = fid_every_n_val_epochs
        self._center_slices_ratio = center_slices_ratio
        self._fid_weights_path = fid_weights_path
        self._vae_config = vae_config
        self._prediction_type = prediction_type
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._early_stop_patience = early_stop_patience
        self._seed = seed

        # State populated in on_fit_start
        self._real_latents: Tensor | None = None
        self._swd_noise: Tensor | None = None
        self._fid_noise: Tensor | None = None

        # Lazy-loaded models
        self._vae: nn.Module | None = None
        self._feature_net: nn.Module | None = None

        # FID tracking
        self._val_epoch_count: int = 0
        self._best_fid: float = float("inf")
        self._patience_counter: int = 0
        self._real_features_cached: tuple[Tensor, Tensor, Tensor] | None = None

        # Per-epoch metrics history (written to JSON at end of training)
        self._eval_history: list[dict[str, Any]] = []

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate fixed noise tensors (rank 0 only).

        Real latent caching is deferred to the first ``on_validation_epoch_end``
        because ``trainer.val_dataloaders`` is not yet available during
        ``on_fit_start`` in DDP.
        """
        if not trainer.is_global_zero:
            return

        S = int(getattr(pl_module, "_latent_spatial", 48))
        C = int(getattr(pl_module, "_in_channels", 4))

        gen = torch.Generator().manual_seed(self._seed)
        self._swd_noise = torch.randn(self._n_swd_samples, C, S, S, S, generator=gen)
        self._fid_noise = torch.randn(self._n_fid_samples, C, S, S, S, generator=gen)

        logger.info(
            "EvaluationCallback: SWD noise=%d, FID noise=%d (real latents cached on first val)",
            self._n_swd_samples,
            self._n_fid_samples,
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run Tier 1 (SWD) every val epoch; Tier 2 (FID) periodically.

        Both tiers always run on the first validation epoch (baseline).
        Real latent caching happens lazily on the first call (deferred from
        ``on_fit_start`` for DDP compatibility).
        """
        if not trainer.is_global_zero:
            return

        # Skip during sanity check — no meaningful eval before any training
        if trainer.sanity_checking:
            return

        # Lazy-init: cache real latents on first val epoch
        if self._real_latents is None:
            self._cache_real_latents(trainer, pl_module)
            if self._real_latents is None:
                logger.warning("Still no real latents after caching attempt; skipping eval")
                return

        self._val_epoch_count += 1
        is_first = self._val_epoch_count == 1
        epoch_record: dict[str, Any] = {
            "train_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "val_epoch": self._val_epoch_count,
        }

        # Tier 1: SWD (every val epoch)
        swd_val = self._compute_swd(pl_module)
        pl_module.log("val/swd", swd_val, rank_zero_only=True, prog_bar=False)
        epoch_record["swd"] = swd_val
        logger.info("Tier 1 SWD: %.6f (val_epoch %d)", swd_val, self._val_epoch_count)

        # Tier 2: FID (every N-th val epoch, OR first epoch for baseline)
        is_fid_epoch = (self._val_epoch_count % self._fid_every_n_val == 0) or is_first
        if is_fid_epoch:
            fid_results = self._compute_fid(pl_module)
            if fid_results is not None:
                for key, val in fid_results.items():
                    pl_module.log(
                        f"val/{key}", val, rank_zero_only=True, prog_bar=(key == "fid_avg")
                    )
                epoch_record.update(
                    {
                        f"fid_{k}" if not k.startswith("fid_") else k: v
                        for k, v in fid_results.items()
                    }
                )
                logger.info(
                    "Tier 2 FID: xy=%.2f yz=%.2f zx=%.2f avg=%.2f%s",
                    fid_results["fid_xy"],
                    fid_results["fid_yz"],
                    fid_results["fid_zx"],
                    fid_results["fid_avg"],
                    " [BASELINE]" if is_first else "",
                )

                # Early stopping check (skip first epoch — it's baseline)
                if not is_first:
                    fid_avg = fid_results["fid_avg"]
                    if fid_avg < self._best_fid:
                        self._best_fid = fid_avg
                        self._patience_counter = 0
                    else:
                        self._patience_counter += 1
                        logger.info(
                            "FID not improved: %.2f >= best %.2f (patience %d/%d)",
                            fid_avg,
                            self._best_fid,
                            self._patience_counter,
                            self._early_stop_patience,
                        )
                        if self._patience_counter >= self._early_stop_patience:
                            logger.warning(
                                "Early stopping: FID patience %d exceeded (best=%.2f)",
                                self._early_stop_patience,
                                self._best_fid,
                            )
                            trainer.should_stop = True
                else:
                    # First FID sets the initial best
                    self._best_fid = fid_results["fid_avg"]

                epoch_record["best_fid"] = self._best_fid
                epoch_record["patience"] = self._patience_counter

        self._eval_history.append(epoch_record)

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Write eval summary JSON and log final metrics (handles early stop)."""
        if not trainer.is_global_zero:
            return
        if not self._eval_history:
            return

        # Build aggregate summary
        swd_values = [r["swd"] for r in self._eval_history if "swd" in r]
        fid_values = [r["fid_avg"] for r in self._eval_history if "fid_avg" in r]

        summary: dict[str, Any] = {
            "n_val_epochs": self._val_epoch_count,
            "early_stopped": trainer.should_stop,
            "final_train_epoch": trainer.current_epoch,
            "per_epoch": self._eval_history,
        }

        if swd_values:
            summary["swd_first"] = swd_values[0]
            summary["swd_last"] = swd_values[-1]
            summary["swd_best"] = min(swd_values)

        if fid_values:
            summary["fid_first"] = fid_values[0]
            summary["fid_last"] = fid_values[-1]
            summary["fid_best"] = min(fid_values)
            summary["best_fid_val_epoch"] = fid_values.index(min(fid_values)) + 1

        # Write to disk
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._cache_dir / "eval_summary.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Evaluation summary saved to %s", out_path)

        # Log summary
        if swd_values:
            logger.info(
                "Eval summary — SWD: first=%.4f, best=%.4f, last=%.4f",
                swd_values[0],
                min(swd_values),
                swd_values[-1],
            )
        if fid_values:
            logger.info(
                "Eval summary — FID: first=%.2f (baseline), best=%.2f, last=%.2f",
                fid_values[0],
                min(fid_values),
                fid_values[-1],
            )

    def _cache_real_latents(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Iterate validation dataloader, collect first N real latents."""
        val_dl = trainer.val_dataloaders
        if val_dl is None:
            logger.warning("No validation dataloader; skipping real latent cache")
            return

        # Lightning may wrap in a list
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        collected: list[Tensor] = []
        n_needed = max(self._n_real_cache, self._n_fid_real_samples)

        for batch in val_dl:
            if isinstance(batch, dict):
                z = batch["z"]
            elif isinstance(batch, (list, tuple)):
                z = batch[0]
            else:
                z = batch
            collected.append(z.cpu())
            if sum(t.shape[0] for t in collected) >= n_needed:
                break

        if collected:
            self._real_latents = torch.cat(collected, dim=0)[:n_needed]
        else:
            logger.warning("No real latents collected from validation dataloader")

    @torch.no_grad()
    def _generate_latents(self, pl_module: pl.LightningModule, noise: Tensor) -> Tensor:
        """Generate latents via 1-NFE with EMA weights.

        Args:
            pl_module: Lightning module with ``net`` and ``ema``.
            noise: Noise tensor to generate from.

        Returns:
            Generated latent tensor.
        """
        noise = noise.to(pl_module.device)
        net = pl_module.net
        ema = pl_module.ema

        ema.apply_shadow(net)
        try:
            z = sample_one_step(net, noise, prediction_type=self._prediction_type)
        finally:
            ema.restore(net)

        return z

    def _compute_swd(self, pl_module: pl.LightningModule) -> float:
        """Generate latents and compute SWD vs cached real."""
        if self._swd_noise is None or self._real_latents is None:
            return 0.0

        fake_z = self._generate_latents(pl_module, self._swd_noise)

        # Flatten to (N, D) for SWD
        n_real = min(self._n_swd_samples, self._real_latents.shape[0])
        real_flat = self._real_latents[:n_real].reshape(n_real, -1).to(fake_z.device)
        fake_flat = fake_z.reshape(fake_z.shape[0], -1)

        return compute_swd(
            real_flat,
            fake_flat,
            n_projections=self._n_swd_projections,
            seed=self._seed,
        )

    def _compute_fid(self, pl_module: pl.LightningModule) -> dict[str, float] | None:
        """Decode latents, extract features, compute 2.5D FID.

        Returns:
            FID results dict, or None if VAE/feature net not available.
        """
        if not self._fid_weights_path or self._fid_noise is None:
            logger.info("FID skipped: no weights path or noise")
            return None

        from neuromf.metrics.fid import compute_fid_2d5

        device = pl_module.device

        # Lazy-load VAE and feature network
        self._ensure_vae_loaded(device)
        self._ensure_feature_net_loaded(device)

        if self._vae is None or self._feature_net is None:
            return None

        # Get/compute real features
        real_feats = self._load_or_compute_real_features(device)

        # Generate fake latents and extract features
        fake_z = self._generate_latents(pl_module, self._fid_noise)
        fake_feats = self._extract_volume_features(fake_z, device)

        return compute_fid_2d5(real_feats, fake_feats)

    def _ensure_vae_loaded(self, device: torch.device) -> None:
        """Lazy-load the MAISI VAE on first Tier 2 call."""
        if self._vae is not None:
            return
        if self._vae_config is None:
            logger.warning("No VAE config; FID computation disabled")
            return

        from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

        vae_cfg = MAISIVAEConfig(**self._vae_config)
        self._vae = MAISIVAEWrapper(vae_cfg, device=device)
        logger.info("Loaded MAISI VAE for FID evaluation")

    def _ensure_feature_net_loaded(self, device: torch.device) -> None:
        """Lazy-load the RadImageNet feature network on first Tier 2 call."""
        if self._feature_net is not None:
            return

        from neuromf.metrics.fid import load_radimagenet_resnet50

        self._feature_net = load_radimagenet_resnet50(self._fid_weights_path)
        self._feature_net = self._feature_net.to(device)
        self._feature_net.eval()
        logger.info("Loaded RadImageNet ResNet-50 for FID evaluation")

    def _load_or_compute_real_features(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """Load real features from cache or compute from real latents.

        On first call, decodes real latents through VAE and extracts
        features. Saves to disk for subsequent calls.
        """
        # Check in-memory cache
        if self._real_features_cached is not None:
            return self._real_features_cached

        # Check disk cache
        if self._cache_dir is not None:
            cache_path = self._cache_dir / "real_features.pt"
            if cache_path.exists():
                logger.info("Loading cached real features from %s", cache_path)
                cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                self._real_features_cached = (cached["xy"], cached["yz"], cached["zx"])
                return self._real_features_cached

        # Compute from real latents
        assert self._real_latents is not None
        n_use = min(self._n_fid_real_samples, self._real_latents.shape[0])
        real_z = self._real_latents[:n_use]

        feats = self._extract_volume_features(real_z, device)
        self._real_features_cached = feats

        # Save to disk
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / "real_features.pt"
            torch.save(
                {"xy": feats[0], "yz": feats[1], "zx": feats[2]},
                str(cache_path),
            )
            logger.info("Cached real features to %s", cache_path)

        return feats

    @torch.no_grad()
    def _extract_volume_features(
        self,
        latents: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode latents through VAE and extract 2.5D features.

        Processes one volume at a time to limit peak memory.

        Args:
            latents: Latent tensor ``(N, 4, 48, 48, 48)``.
            device: Compute device.

        Returns:
            Tuple of concatenated features per plane.
        """
        from neuromf.metrics.fid import extract_2d5_features

        assert self._vae is not None
        assert self._feature_net is not None

        all_xy: list[Tensor] = []
        all_yz: list[Tensor] = []
        all_zx: list[Tensor] = []

        for i in range(latents.shape[0]):
            z_i = latents[i : i + 1].to(device)

            # Decode to pixel space
            x_hat = self._vae.decode(z_i)

            # Extract 2.5D features
            xy, yz, zx = extract_2d5_features(
                x_hat,
                self._feature_net,
                center_slices_ratio=self._center_slices_ratio,
            )
            all_xy.append(xy)
            all_yz.append(yz)
            all_zx.append(zx)

        return (
            torch.cat(all_xy, dim=0),
            torch.cat(all_yz, dim=0),
            torch.cat(all_zx, dim=0),
        )
