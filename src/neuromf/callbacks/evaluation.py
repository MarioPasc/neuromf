"""Two-tier evaluation callback: SWD (fast) + FID (thorough).

Tier 1 (SWD): Every validation epoch — generates latents via 1-NFE EMA model,
computes Sliced Wasserstein Distance vs cached real latents. Fast (~2s).

Tier 2 (FID): Every ``fid_every_n_val_epochs`` validation epochs — decodes
latents through frozen MAISI VAE, extracts features, and computes FID.
Two modes are supported:

- ``"2d5"`` (default): RadImageNet ResNet-50 on 3 orthogonal 2D planes.
- ``"3d"``: R3D-18 on full 3D volumes (matches MOTFM evaluation protocol).

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
from neuromf.sampling.multi_step import sample_euler  # noqa: F401
from neuromf.sampling.one_step import sample_one_step

logger = logging.getLogger(__name__)

_VALID_FID_MODES = ("2d5", "3d")


class EvaluationCallback(pl.Callback):
    """Two-tier evaluation: SWD every val epoch, FID periodically.

    Args:
        n_swd_samples: Number of latents to generate for SWD.
        n_swd_projections: Number of random projections for SWD.
        n_real_cache: Number of real validation latents to cache.
        n_fid_samples: Number of latents to generate for FID.
        n_fid_real_samples: Number of real latents to decode for FID reference.
        fid_every_n_val_epochs: Tier 2 frequency (in validation epochs).
        center_slices_ratio: Fraction of center slices for 2.5D extraction.
        fid_weights_path: Path to RadImageNet ResNet-50 state dict (2d5 mode).
        fid_mode: ``"2d5"`` for RadImageNet 2.5D or ``"3d"`` for Med3D 3D.
        fid_3d_weights_path: Deprecated (R3D-18 uses torchvision built-in weights).
        vae_config: Dict of VAE config params (for lazy loading).
        prediction_type: ``"u"`` or ``"x"`` prediction mode.
        cache_dir: Directory for caching real FID features to disk.
        early_stop_patience: Tier-2 evals without improvement before stopping.
        seed: Random seed for noise generation.
        eval_nfe: Number of function evaluations for sample generation.
            1 = one-step MeanFlow (default). >1 = multi-step Euler.
            Use NFE>1 for FM-only pretraining (Stage 1) where 1-NFE
            quality is meaningless.
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
        fid_mode: str = "2d5",
        fid_3d_weights_path: str = "",
        vae_config: dict | None = None,
        prediction_type: str = "u",
        cache_dir: str = "",
        early_stop_patience: int = 5,
        seed: int = 42,
        eval_nfe: int = 1,
    ) -> None:
        super().__init__()
        if fid_mode not in _VALID_FID_MODES:
            raise ValueError(f"fid_mode must be one of {_VALID_FID_MODES}, got '{fid_mode}'")

        self._n_swd_samples = n_swd_samples
        self._n_swd_projections = n_swd_projections
        self._n_real_cache = n_real_cache
        self._n_fid_samples = n_fid_samples
        self._n_fid_real_samples = n_fid_real_samples
        self._fid_every_n_val = fid_every_n_val_epochs
        self._center_slices_ratio = center_slices_ratio
        self._fid_weights_path = fid_weights_path
        self._fid_mode = fid_mode
        self._fid_3d_weights_path = fid_3d_weights_path
        self._vae_config = vae_config
        self._prediction_type = prediction_type
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._early_stop_patience = early_stop_patience
        self._seed = seed
        self._eval_nfe = eval_nfe

        # State populated in on_fit_start
        self._real_latents: Tensor | None = None
        self._swd_noise: Tensor | None = None
        self._fid_noise: Tensor | None = None

        # Lazy-loaded models
        self._vae: nn.Module | None = None
        self._feature_net: nn.Module | None = None

        # Latent denormalization stats (lazy-init from pl_module)
        self._latent_mean: Tensor | None = None
        self._latent_std: Tensor | None = None

        # FID tracking
        self._val_epoch_count: int = 0
        self._best_fid: float = float("inf")
        self._patience_counter: int = 0
        # Cache type depends on mode: tuple of 3 tensors (2d5) or single tensor (3d)
        self._real_features_cached: tuple[Tensor, Tensor, Tensor] | Tensor | None = None

        # Per-epoch metrics history (written to JSON at end of training)
        self._eval_history: list[dict[str, Any]] = []

    @property
    def _fid_key(self) -> str:
        """Primary FID metric key used for logging and early stopping."""
        return "fid_3d" if self._fid_mode == "3d" else "fid_avg"

    @property
    def _active_weights_path(self) -> str:
        """Weights path for the active FID mode."""
        if self._fid_mode == "3d":
            return self._fid_3d_weights_path
        return self._fid_weights_path

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
            "EvaluationCallback: SWD noise=%d, FID noise=%d, mode=%s "
            "(real latents cached on first val)",
            self._n_swd_samples,
            self._n_fid_samples,
            self._fid_mode,
        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Tier 1 (SWD): computed every training epoch (~2s).

        Skipped until real latents are cached (first val epoch).
        """
        if not trainer.is_global_zero:
            return
        if self._real_latents is None:
            return

        swd_val = self._compute_swd(pl_module)
        pl_module.log("train/swd", swd_val, rank_zero_only=True, prog_bar=False)

        epoch_record: dict[str, Any] = {
            "train_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "swd": swd_val,
        }
        self._eval_history.append(epoch_record)
        logger.info("Tier 1 SWD: %.6f (epoch %d)", swd_val, trainer.current_epoch)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Tier 2 (FID): computed periodically on validation epochs.

        Always runs on the first validation epoch (baseline). Real latent
        caching happens lazily on the first call (deferred from
        ``on_fit_start`` for DDP compatibility).

        All ranks must participate in the ``pl_module.log`` call so that
        ``ModelCheckpoint(monitor='val/fid_3d')`` finds the metric on every
        rank. FID computation itself runs only on rank 0; the result is
        broadcast to all other ranks before logging.
        """
        # Skip during sanity check — no meaningful eval before any training
        if trainer.sanity_checking:
            return

        # --- Rank-0 gated: caching and scheduling logic ---
        fid_results: dict[str, float] | None = None

        if trainer.is_global_zero:
            # Lazy-init: cache real latents on first val epoch
            if self._real_latents is None:
                self._cache_real_latents(trainer, pl_module)

            self._val_epoch_count += 1
            is_first = self._val_epoch_count == 1

            # Tier 2: FID (every N-th val epoch, OR first epoch for baseline)
            is_fid_epoch = (self._val_epoch_count % self._fid_every_n_val == 0) or is_first

            if is_fid_epoch and self._real_latents is not None:
                fid_results = self._compute_fid(pl_module)

        # --- Broadcast FID results to all ranks (DDP) ---
        fid_results = self._broadcast_fid_results(trainer, fid_results)

        if fid_results is None:
            return

        fid_key = self._fid_key
        for key, val in fid_results.items():
            pl_module.log(f"val/{key}", val, sync_dist=False, prog_bar=(key == fid_key))

        # --- Rank-0 only: history, logging, early stopping ---
        if not trainer.is_global_zero:
            return

        is_first = self._val_epoch_count == 1

        # Attach FID to the most recent eval_history record (from on_train_epoch_end)
        fid_record = {
            f"fid_{k}" if not k.startswith("fid_") else k: v for k, v in fid_results.items()
        }
        if self._eval_history:
            self._eval_history[-1].update(fid_record)
            self._eval_history[-1]["val_epoch"] = self._val_epoch_count
        else:
            # Edge case: val fires before train_epoch_end
            fid_record.update(
                {
                    "train_epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "val_epoch": self._val_epoch_count,
                }
            )
            self._eval_history.append(fid_record)

        # Mode-specific logging
        if self._fid_mode == "3d":
            logger.info(
                "Tier 2 FID (3D): %.2f%s",
                fid_results["fid_3d"],
                " [BASELINE]" if is_first else "",
            )
        else:
            logger.info(
                "Tier 2 FID (2.5D): xy=%.2f yz=%.2f zx=%.2f avg=%.2f%s",
                fid_results["fid_xy"],
                fid_results["fid_yz"],
                fid_results["fid_zx"],
                fid_results["fid_avg"],
                " [BASELINE]" if is_first else "",
            )

        # Early stopping check (skip first epoch — it's baseline)
        fid_primary = fid_results[fid_key]
        if not is_first:
            if fid_primary < self._best_fid:
                self._best_fid = fid_primary
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                logger.info(
                    "FID not improved: %.2f >= best %.2f (patience %d/%d)",
                    fid_primary,
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
            self._best_fid = fid_primary

        if self._eval_history:
            self._eval_history[-1]["best_fid"] = self._best_fid
            self._eval_history[-1]["patience"] = self._patience_counter

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

        fid_key = self._fid_key

        # Build aggregate summary
        swd_values = [r["swd"] for r in self._eval_history if "swd" in r]
        fid_values = [r[fid_key] for r in self._eval_history if fid_key in r]

        summary: dict[str, Any] = {
            "fid_mode": self._fid_mode,
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
                "Eval summary — FID (%s): first=%.2f (baseline), best=%.2f, last=%.2f",
                self._fid_mode,
                fid_values[0],
                min(fid_values),
                fid_values[-1],
            )

    @staticmethod
    def _broadcast_fid_results(
        trainer: pl.Trainer,
        fid_results: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """Broadcast FID results from rank 0 to all ranks.

        In single-GPU mode this is a no-op. In DDP, rank 0 sends the number
        of result keys (0 = None) followed by key-value pairs so that every
        rank ends up with an identical ``fid_results`` dict.

        Args:
            trainer: Lightning trainer (for world_size check).
            fid_results: FID dict on rank 0, ``None`` on other ranks.

        Returns:
            FID dict on all ranks, or ``None`` if rank 0 had no results.
        """
        if trainer.world_size <= 1:
            return fid_results

        import torch.distributed as dist

        device = trainer.strategy.root_device

        # Broadcast n_keys first so non-zero ranks know the buffer size
        if trainer.is_global_zero:
            n_keys = 0 if fid_results is None else len(fid_results)
        else:
            n_keys = 0
        n_keys_t = torch.tensor([n_keys], device=device, dtype=torch.float64)
        dist.broadcast(n_keys_t, src=0)
        n_keys = int(n_keys_t.item())

        if n_keys == 0:
            return None

        # Broadcast the values
        if trainer.is_global_zero:
            keys = sorted(fid_results.keys())  # type: ignore[union-attr]
            buf = torch.tensor(
                [fid_results[k] for k in keys],  # type: ignore[union-attr]
                device=device,
                dtype=torch.float64,
            )
        else:
            buf = torch.zeros(n_keys, device=device, dtype=torch.float64)
        dist.broadcast(buf, src=0)

        # Unpack — keys must match what rank 0 used (sorted order)
        if trainer.is_global_zero:
            return fid_results
        # Non-zero ranks reconstruct the dict
        # Keys are deterministic: "3d" → ["fid_3d"], "2d5" → ["fid_avg","fid_xy","fid_yz","fid_zx"]
        vals = buf.tolist()
        if n_keys == 1:
            keys = ["fid_3d"]
        else:
            keys = ["fid_avg", "fid_xy", "fid_yz", "fid_zx"]  # sorted
        return dict(zip(keys, vals))

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
        """Generate latents via 1-NFE with EMA weights, one sample at a time.

        Processes samples individually to avoid OOM — a single 178M-param
        UNet forward on ``(1, 4, 48, 48, 48)`` uses ~8 GB, so batching
        all N samples would exceed GPU memory.

        Args:
            pl_module: Lightning module with ``net`` and ``ema``.
            noise: Noise tensor ``(N, C, D, H, W)`` to generate from.

        Returns:
            Generated latent tensor ``(N, C, D, H, W)`` on CPU.
        """
        device = pl_module.device
        net = pl_module.net
        ema = pl_module.ema

        ema.apply_shadow(net)
        try:
            chunks: list[Tensor] = []
            for i in range(noise.shape[0]):
                noise_i = noise[i : i + 1].to(device)
                if self._eval_nfe <= 1:
                    z_i = sample_one_step(
                        net,
                        noise_i,
                        prediction_type=self._prediction_type,
                    )
                else:
                    from neuromf.sampling.multi_step import sample_euler

                    z_i = sample_euler(
                        net,
                        noise_i,
                        n_steps=self._eval_nfe,
                        prediction_type=self._prediction_type,
                    )
                chunks.append(z_i.cpu())
        finally:
            ema.restore(net)

        return torch.cat(chunks, dim=0)

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
        """Decode latents, extract features, compute FID.

        Dispatches to 2.5D or 3D path based on ``fid_mode``.

        Returns:
            FID results dict, or None if weights/noise not available.
        """
        # R3D-18 (3d mode) needs no external weights; 2d5 mode needs RadImageNet path
        if self._fid_mode != "3d" and not self._active_weights_path:
            logger.info("FID skipped: no weights path configured")
            return None
        if self._fid_noise is None:
            logger.info("FID skipped: no noise tensor")
            return None

        device = pl_module.device

        # Lazy-load VAE, feature network, and latent stats
        self._ensure_vae_loaded(device)
        self._ensure_feature_net_loaded(device)
        self._ensure_latent_stats(pl_module)

        if self._vae is None or self._feature_net is None:
            return None

        if self._fid_mode == "3d":
            return self._compute_fid_3d(pl_module, device)
        return self._compute_fid_2d5(pl_module, device)

    def _compute_fid_2d5(
        self,
        pl_module: pl.LightningModule,
        device: torch.device,
    ) -> dict[str, float]:
        """2.5D FID path: RadImageNet features from orthogonal slices."""
        from neuromf.metrics.fid import compute_fid_2d5

        real_feats = self._load_or_compute_real_features_2d5(device)
        fake_z = self._generate_latents(pl_module, self._fid_noise)
        fake_feats = self._extract_volume_features_2d5(fake_z, device)
        return compute_fid_2d5(real_feats, fake_feats)

    def _compute_fid_3d(
        self,
        pl_module: pl.LightningModule,
        device: torch.device,
    ) -> dict[str, float]:
        """3D FID path: R3D-18 features from full volumes."""
        from neuromf.metrics.fid_3d import compute_fid_3d

        real_feats = self._load_or_compute_real_features_3d(device)
        fake_z = self._generate_latents(pl_module, self._fid_noise)
        fake_feats = self._extract_volume_features_3d(fake_z, device)
        return {"fid_3d": compute_fid_3d(real_feats, fake_feats)}

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
        """Lazy-load the feature network based on FID mode."""
        if self._feature_net is not None:
            return

        if self._fid_mode == "3d":
            from neuromf.metrics.fid_3d import load_fid3d_feature_net

            self._feature_net = load_fid3d_feature_net(
                device=device,
                weights_path=self._fid_3d_weights_path or None,
            )
            logger.info("Loaded R3D-18 for 3D-FID evaluation (MOTFM protocol)")
        else:
            from neuromf.metrics.fid import load_radimagenet_resnet50

            self._feature_net = load_radimagenet_resnet50(self._fid_weights_path)
            self._feature_net = self._feature_net.to(device)
            self._feature_net.eval()
            logger.info("Loaded RadImageNet ResNet-50 for 2.5D-FID evaluation")

    def _ensure_latent_stats(self, pl_module: pl.LightningModule) -> None:
        """Lazy-init latent denormalization stats from the Lightning module.

        Args:
            pl_module: Lightning module with ``latent_mean`` and ``latent_std`` buffers.
        """
        if self._latent_mean is not None:
            return
        self._latent_mean = pl_module.latent_mean.detach().clone()
        self._latent_std = pl_module.latent_std.detach().clone()
        logger.debug(
            "Latent stats cached for denormalization: mean=%s, std=%s",
            self._latent_mean.flatten().tolist(),
            self._latent_std.flatten().tolist(),
        )

    def _denormalize_latent(self, z: Tensor) -> Tensor:
        """Denormalize latents from standardised to original VAE space.

        Args:
            z: Normalised latent ``(B, C, D, H, W)``.

        Returns:
            Denormalised latent ``z_0 = z * std + mean``.
        """
        assert self._latent_mean is not None and self._latent_std is not None
        mean = self._latent_mean.to(z.device)
        std = self._latent_std.to(z.device)
        return z * std + mean

    # ------------------------------------------------------------------
    # 2.5D feature extraction (existing path)
    # ------------------------------------------------------------------

    def _load_or_compute_real_features_2d5(
        self,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load or compute real 2.5D features (3-plane tuple)."""
        if self._real_features_cached is not None:
            assert isinstance(self._real_features_cached, tuple)
            return self._real_features_cached

        if self._cache_dir is not None:
            cache_path = self._cache_dir / "real_features.pt"
            if cache_path.exists():
                logger.info("Loading cached 2.5D real features from %s", cache_path)
                cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                self._real_features_cached = (cached["xy"], cached["yz"], cached["zx"])
                return self._real_features_cached

        assert self._real_latents is not None
        n_use = min(self._n_fid_real_samples, self._real_latents.shape[0])
        feats = self._extract_volume_features_2d5(self._real_latents[:n_use], device)
        self._real_features_cached = feats

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / "real_features.pt"
            torch.save(
                {"xy": feats[0], "yz": feats[1], "zx": feats[2]},
                str(cache_path),
            )
            logger.info("Cached 2.5D real features to %s", cache_path)

        return feats

    @torch.no_grad()
    def _extract_volume_features_2d5(
        self,
        latents: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode latents through VAE and extract 2.5D features.

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
            z_i = self._denormalize_latent(z_i)
            x_hat = self._vae.decode(z_i)
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

    # ------------------------------------------------------------------
    # 3D feature extraction (new path)
    # ------------------------------------------------------------------

    def _load_or_compute_real_features_3d(self, device: torch.device) -> Tensor:
        """Load or compute real 3D features (single tensor)."""
        if self._real_features_cached is not None:
            assert isinstance(self._real_features_cached, Tensor)
            return self._real_features_cached

        if self._cache_dir is not None:
            cache_path = self._cache_dir / "real_features_3d.pt"
            if cache_path.exists():
                logger.info("Loading cached 3D real features from %s", cache_path)
                cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                self._real_features_cached = cached
                return self._real_features_cached

        assert self._real_latents is not None
        n_use = min(self._n_fid_real_samples, self._real_latents.shape[0])
        feats = self._extract_volume_features_3d(self._real_latents[:n_use], device)
        self._real_features_cached = feats

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / "real_features_3d.pt"
            torch.save(feats, str(cache_path))
            logger.info("Cached 3D real features to %s", cache_path)

        return feats

    @torch.no_grad()
    def _extract_volume_features_3d(
        self,
        latents: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Decode latents through VAE and extract 3D features.

        Decodes one volume at a time (VAE memory limit), then extracts
        R3D-18 features in a single ``extract_3d_features`` call so that
        per-set min-max normalisation is applied across all decoded
        volumes jointly (matching the MOTFM protocol).

        Args:
            latents: Latent tensor ``(N, 4, 48, 48, 48)``.
            device: Compute device.

        Returns:
            Feature tensor ``(N, 512)``.
        """
        from neuromf.metrics.fid_3d import extract_3d_features

        assert self._vae is not None
        assert self._feature_net is not None

        # Decode all latents first (one at a time for VAE memory)
        decoded: list[Tensor] = []
        for i in range(latents.shape[0]):
            z_i = latents[i : i + 1].to(device)
            z_i = self._denormalize_latent(z_i)
            x_hat = self._vae.decode(z_i)
            decoded.append(x_hat.cpu())

        # Stack and extract features with per-set normalisation
        all_volumes = torch.cat(decoded, dim=0)  # (N, 1, D, H, W)
        return extract_3d_features(all_volumes, self._feature_net, normalize=True)
