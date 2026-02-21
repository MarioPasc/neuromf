"""Sample collection callback for multi-NFE latent archive during training.

Replaces in-training VAE decode with a lightweight latent-only approach:
saves raw generated latents at multiple NFE steps using fixed noise seeds,
computes per-epoch latent statistics, and defers VAE decoding to
``on_fit_end`` where all figures are auto-generated.

All epochs are stored in a single ``sample_archive.pt`` file that grows
incrementally. Structure::

    {
        "metadata": {"noise_seed", "nfe_steps", "n_samples"},
        "noise": Tensor(N, C, D, H, W),
        "latent_mean": Tensor(C,),
        "latent_std": Tensor(C,),
        "epochs": [24, 49, 74, ...],
        "epoch_0024": {"global_step", "nfe_1", ..., "stats", "nfe_consistency"},
        ...
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor

from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step
from neuromf.utils.latent_diagnostics import compute_latent_stats, compute_nfe_consistency

logger = logging.getLogger(__name__)


class SampleCollectorCallback(pl.Callback):
    """Collects generated latents at multiple NFE steps during training.

    Saves raw ``.pt`` archives per epoch with fixed noise seeds for
    deterministic evolution tracking. At the end of training, auto-generates
    all evolution plots and optionally decodes samples through the MAISI VAE.

    Args:
        samples_dir: Root directory for sample output. Archives are saved
            under ``{samples_dir}/``.
        collect_every_n_epochs: How often to collect samples.
        n_samples: Number of samples to generate per collection.
        nfe_steps: List of NFE step counts to generate (e.g. ``[1, 2, 5, 10]``).
        seed: Random seed for fixed noise generation.
        prediction_type: ``"x"`` for x-prediction, ``"u"`` for u-prediction.
        vae_config: Optional VAE config dict for lazy-loading the MAISI VAE
            at end of training (for decoding samples). If ``None``, decoded
            figures are skipped.
        figures_dir: Output directory for all generated figures. If empty,
            figure generation is skipped.
    """

    def __init__(
        self,
        samples_dir: str,
        collect_every_n_epochs: int = 25,
        n_samples: int = 8,
        nfe_steps: list[int] | None = None,
        seed: int = 42,
        prediction_type: str = "x",
        vae_config: dict[str, Any] | None = None,
        figures_dir: str = "",
    ) -> None:
        super().__init__()
        self._samples_dir = Path(samples_dir)
        self._collect_every = collect_every_n_epochs
        self._n_samples = n_samples
        self._nfe_steps = nfe_steps if nfe_steps is not None else [1, 2, 5, 10]
        self._seed = seed
        self._prediction_type = prediction_type
        self._vae_config = vae_config
        self._figures_dir = Path(figures_dir) if figures_dir else None

        # Cached fixed noise — generated in on_fit_start
        self._fixed_noise: Tensor | None = None
        # Lazy-loaded VAE — only created in on_fit_end if vae_config is set
        self._vae: Any | None = None

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate and cache fixed noise tensor; signal model that collection is active."""
        S = int(getattr(pl_module, "_latent_spatial", 48))
        C = int(getattr(pl_module, "_in_channels", 4))

        gen = torch.Generator().manual_seed(self._seed)
        self._fixed_noise = torch.randn(self._n_samples, C, S, S, S, generator=gen)

        # Signal to the model that sample collection is handled externally
        pl_module._sample_collector_active = True  # type: ignore[attr-defined]

        logger.info(
            "SampleCollectorCallback: %d samples, NFE=%s, every %d epochs, seed=%d",
            self._n_samples,
            self._nfe_steps,
            self._collect_every,
            self._seed,
        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Collect multi-NFE samples at configured epochs (rank 0 only)."""
        epoch = trainer.current_epoch
        if (epoch + 1) % self._collect_every != 0:
            return
        if not trainer.is_global_zero:
            return

        self._collect_samples(pl_module, epoch, trainer.global_step)
        self._last_collected_epoch = epoch

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Force final collection, then generate all figures and decoded samples.

        Ensures the archive includes the final model state even when
        training ends via early stopping. Then auto-generates all evolution
        plots and decoded NFE comparison figures to ``figures_dir``.
        """
        if not trainer.is_global_zero:
            return

        # 1. Force final sample collection
        epoch = trainer.current_epoch
        last = getattr(self, "_last_collected_epoch", -1)
        if epoch != last:
            logger.info("Final sample collection at epoch %d (early stop or end)", epoch)
            self._collect_samples(pl_module, epoch, trainer.global_step)

        # 2. Generate all figures
        if not self._figures_dir:
            return

        archive_path = self._samples_dir / "sample_archive.pt"
        if not archive_path.exists():
            logger.warning("No sample_archive.pt found; skipping figure generation.")
            return

        try:
            archive = torch.load(archive_path, map_location="cpu", weights_only=False)
            n_epochs = len(archive.get("epochs", []))
            logger.info("Generating figures from archive (%d epochs)...", n_epochs)

            # Decode last epoch's samples through VAE (if configured)
            decoded_nfe: dict[int, np.ndarray] = {}
            decoded_1nfe_by_epoch: dict[int, np.ndarray] = {}
            if self._vae_config:
                device = pl_module.device
                decoded_nfe, decoded_1nfe_by_epoch = self._decode_archive_samples(
                    archive,
                    device,
                )

            self._generate_all_figures(archive, decoded_nfe, decoded_1nfe_by_epoch)

        except Exception as e:
            logger.error("Figure generation failed: %s", e, exc_info=True)
        finally:
            self._cleanup_vae()

    @torch.no_grad()
    def _collect_samples(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        global_step: int,
    ) -> None:
        """Generate samples at multiple NFE steps and append to archive.

        All epochs are stored in a single ``sample_archive.pt`` that grows
        incrementally. The file is loaded, extended, and re-saved each
        collection epoch.

        Args:
            pl_module: The Lightning module with ``net``, ``ema``,
                ``latent_mean``, ``latent_std``.
            epoch: Current epoch number.
            global_step: Current global step.
        """
        noise = self._fixed_noise
        if noise is None:
            logger.warning("Fixed noise not initialised; skipping collection")
            return

        noise = noise.to(pl_module.device)
        net = pl_module.net
        ema = pl_module.ema

        # Apply EMA shadow weights for generation
        ema.apply_shadow(net)
        try:
            nfe_samples = self._generate_multi_nfe(net, noise)
        finally:
            ema.restore(net)

        # Compute per-NFE statistics
        stats: dict[str, dict] = {}
        for key, z in nfe_samples.items():
            s = compute_latent_stats(z)
            stats[key] = {
                "mean": s["mean"].tolist(),
                "std": s["std"].tolist(),
                "min": s["global_min"],
                "max": s["global_max"],
            }

        # NFE consistency (1-NFE vs multi-step)
        nfe_consistency = compute_nfe_consistency(nfe_samples)

        # Load or create the archive
        self._samples_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._samples_dir / "sample_archive.pt"

        if archive_path.exists():
            archive = torch.load(archive_path, map_location="cpu", weights_only=False)
        else:
            archive = {
                "metadata": {
                    "noise_seed": self._seed,
                    "nfe_steps": self._nfe_steps,
                    "n_samples": self._n_samples,
                },
                "noise": noise.cpu(),
                "latent_mean": pl_module.latent_mean.cpu().squeeze(),
                "latent_std": pl_module.latent_std.cpu().squeeze(),
                "epochs": [],
            }

        # Add this epoch's data
        epoch_key = f"epoch_{epoch:04d}"
        epoch_data: dict = {
            "global_step": global_step,
            "stats": stats,
            "nfe_consistency": nfe_consistency,
        }
        for key, z in nfe_samples.items():
            epoch_data[key] = z.cpu()

        archive[epoch_key] = epoch_data
        if epoch not in archive["epochs"]:
            archive["epochs"].append(epoch)
            archive["epochs"].sort()

        torch.save(archive, archive_path)

        logger.info(
            "Updated sample archive: epoch=%d, NFEs=%s, total_epochs=%d, path=%s",
            epoch,
            list(nfe_samples.keys()),
            len(archive["epochs"]),
            archive_path,
        )

    def _generate_multi_nfe(
        self,
        net: torch.nn.Module,
        noise: Tensor,
    ) -> dict[str, Tensor]:
        """Generate samples at all configured NFE step counts.

        Args:
            net: UNet model (with EMA weights already applied).
            noise: Fixed noise tensor ``(N, C, D, H, W)``.

        Returns:
            Dict mapping ``"nfe_{N}"`` to generated latent tensors.
        """
        results: dict[str, Tensor] = {}
        for nfe in self._nfe_steps:
            if nfe == 1:
                z = sample_one_step(net, noise, prediction_type=self._prediction_type)
            else:
                z = sample_euler(net, noise, n_steps=nfe, prediction_type=self._prediction_type)
            results[f"nfe_{nfe}"] = z
        return results

    # ------------------------------------------------------------------
    # End-of-training figure generation
    # ------------------------------------------------------------------

    def _ensure_vae_loaded(self, device: torch.device) -> None:
        """Lazy-load the MAISI VAE from ``vae_config``.

        Only called once in ``on_fit_end``. The VAE stays on device until
        ``_cleanup_vae()`` is called.

        Args:
            device: Device to load the VAE onto.
        """
        if self._vae is not None:
            return
        if not self._vae_config:
            return

        from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

        cfg = MAISIVAEConfig(
            weights_path=str(self._vae_config.get("weights_path", "")),
            scale_factor=float(self._vae_config.get("scale_factor", 0.96240234375)),
            spatial_dims=int(self._vae_config.get("spatial_dims", 3)),
            in_channels=int(self._vae_config.get("in_channels", 1)),
            out_channels=int(self._vae_config.get("out_channels", 1)),
            latent_channels=int(self._vae_config.get("latent_channels", 4)),
            num_channels=list(self._vae_config.get("num_channels", [64, 128, 256])),
            num_res_blocks=list(self._vae_config.get("num_res_blocks", [2, 2, 2])),
            norm_num_groups=int(self._vae_config.get("norm_num_groups", 32)),
            norm_eps=float(self._vae_config.get("norm_eps", 1e-6)),
            attention_levels=list(self._vae_config.get("attention_levels", [False, False, False])),
            with_encoder_nonlocal_attn=bool(
                self._vae_config.get("with_encoder_nonlocal_attn", False)
            ),
            with_decoder_nonlocal_attn=bool(
                self._vae_config.get("with_decoder_nonlocal_attn", False)
            ),
            use_checkpointing=bool(self._vae_config.get("use_checkpointing", False)),
            use_convtranspose=bool(self._vae_config.get("use_convtranspose", False)),
            norm_float16=bool(self._vae_config.get("norm_float16", True)),
            num_splits=int(self._vae_config.get("num_splits", 4)),
            dim_split=int(self._vae_config.get("dim_split", 1)),
            downsample_factor=int(self._vae_config.get("downsample_factor", 4)),
        )

        weights_path = cfg.weights_path
        if not weights_path or not Path(weights_path).exists():
            logger.warning("VAE weights not found at %s; skipping decode.", weights_path)
            return

        self._vae = MAISIVAEWrapper(cfg, device=device)
        logger.info("Loaded MAISI VAE on %s for end-of-training decode", device)

    def _cleanup_vae(self) -> None:
        """Release VAE memory after figure generation."""
        if self._vae is not None:
            del self._vae
            self._vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _decode_archive_samples(
        self,
        archive: dict,
        device: torch.device,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Decode samples from the archive through the MAISI VAE.

        Decodes:
        1. All NFE levels from the **last epoch** (for decoded NFE grid).
        2. 1-NFE sample #0 from **all epochs** (for decoded row in NFE
           comparison grid).

        Processes one sample at a time to limit peak GPU memory.

        Args:
            archive: Loaded ``sample_archive.pt``.
            device: Torch device for VAE decode.

        Returns:
            Tuple of:
            - ``decoded_nfe``: ``{nfe_level: volume_array (D,H,W)}``
              for the last epoch's sample #0.
            - ``decoded_1nfe_by_epoch``: ``{epoch: volume_array (D,H,W)}``
              for 1-NFE sample #0 across all epochs.
        """
        self._ensure_vae_loaded(device)
        if self._vae is None:
            return {}, {}

        latent_mean = archive["latent_mean"]
        latent_std = archive["latent_std"]
        if latent_mean.ndim == 1:
            latent_mean = latent_mean.view(1, -1, 1, 1, 1)
            latent_std = latent_std.view(1, -1, 1, 1, 1)

        epochs = sorted(archive["epochs"])
        last_epoch = epochs[-1]
        last_key = f"epoch_{last_epoch:04d}"

        # 1. Decode all NFE levels for last epoch (sample #0)
        decoded_nfe: dict[int, np.ndarray] = {}
        for nfe in self._nfe_steps:
            nfe_key = f"nfe_{nfe}"
            if nfe_key not in archive[last_key]:
                continue
            z_norm = archive[last_key][nfe_key]  # (N, C, D, H, W)
            z_denorm = z_norm[0:1] * latent_std + latent_mean
            z_denorm = z_denorm.to(device)
            decoded = self._vae.decode(z_denorm).cpu().float()
            decoded_nfe[nfe] = decoded[0, 0].numpy()  # (D, H, W)
            del z_denorm, decoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Decoded last epoch NFE=%d", nfe)

        # 2. Decode 1-NFE sample #0 for all epochs (for comparison grid row)
        decoded_1nfe_by_epoch: dict[int, np.ndarray] = {}
        for ep in epochs:
            epoch_key = f"epoch_{ep:04d}"
            if "nfe_1" not in archive[epoch_key]:
                continue
            z_norm = archive[epoch_key]["nfe_1"]
            z_denorm = z_norm[0:1] * latent_std + latent_mean
            z_denorm = z_denorm.to(device)
            decoded = self._vae.decode(z_denorm).cpu().float()
            decoded_1nfe_by_epoch[ep] = decoded[0, 0].numpy()
            del z_denorm, decoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "Decoded %d NFE levels (last epoch) + %d epochs (1-NFE)",
            len(decoded_nfe),
            len(decoded_1nfe_by_epoch),
        )
        return decoded_nfe, decoded_1nfe_by_epoch

    def _generate_all_figures(
        self,
        archive: dict,
        decoded_nfe: dict[int, np.ndarray],
        decoded_1nfe_by_epoch: dict[int, np.ndarray],
    ) -> None:
        """Generate all evolution plots and decoded figures.

        Produces 6 latent evolution plots plus optional decoded figures
        (NFE grid and decoded row in comparison grid).

        Args:
            archive: Loaded ``sample_archive.pt``.
            decoded_nfe: Decoded volumes per NFE level (last epoch).
            decoded_1nfe_by_epoch: Decoded 1-NFE volumes per epoch.
        """
        assert self._figures_dir is not None
        figures_dir = self._figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        from neuromf.utils.sample_plots import (
            plot_channel_stats_evolution,
            plot_decoded_nfe_grid,
            plot_inter_epoch_delta,
            plot_nfe_comparison_grid,
            plot_nfe_consistency_evolution,
            plot_sample_evolution_grid,
            plot_spectral_evolution,
        )

        # 6 latent evolution plots
        plot_sample_evolution_grid(archive, figures_dir)
        plot_channel_stats_evolution(archive, figures_dir)
        plot_spectral_evolution(archive, figures_dir)
        plot_nfe_comparison_grid(
            archive,
            figures_dir,
            decoded_volumes=decoded_1nfe_by_epoch if decoded_1nfe_by_epoch else None,
        )
        plot_nfe_consistency_evolution(archive, figures_dir)
        plot_inter_epoch_delta(archive, figures_dir)

        # Decoded NFE grid (if VAE decode succeeded)
        if decoded_nfe:
            plot_decoded_nfe_grid(decoded_nfe, figures_dir)

        logger.info("All figures saved to %s", figures_dir)
