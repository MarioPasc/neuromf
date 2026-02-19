"""Sample collection callback for multi-NFE latent archive during training.

Replaces in-training VAE decode with a lightweight latent-only approach:
saves raw generated latents at multiple NFE steps using fixed noise seeds,
computes per-epoch latent statistics, and defers all VAE decoding and
evolution plotting to post-training (``generate_sample_plots``).

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
    deterministic evolution tracking. No VAE loading during training.

    Args:
        samples_dir: Root directory for sample output. Archives are saved
            under ``{samples_dir}/generated_samples/``.
        collect_every_n_epochs: How often to collect samples.
        n_samples: Number of samples to generate per collection.
        nfe_steps: List of NFE step counts to generate (e.g. ``[1, 2, 5, 10]``).
        seed: Random seed for fixed noise generation.
        prediction_type: ``"x"`` for x-prediction, ``"u"`` for u-prediction.
    """

    def __init__(
        self,
        samples_dir: str,
        collect_every_n_epochs: int = 25,
        n_samples: int = 8,
        nfe_steps: list[int] | None = None,
        seed: int = 42,
        prediction_type: str = "x",
    ) -> None:
        super().__init__()
        self._samples_dir = Path(samples_dir)
        self._collect_every = collect_every_n_epochs
        self._n_samples = n_samples
        self._nfe_steps = nfe_steps if nfe_steps is not None else [1, 2, 5, 10]
        self._seed = seed
        self._prediction_type = prediction_type

        # Cached fixed noise — generated in on_fit_start
        self._fixed_noise: Tensor | None = None

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
        """Force a final sample collection if the last epoch was not collected.

        Ensures the archive includes the final model state even when
        training ends via early stopping at a non-collection epoch.
        """
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        last = getattr(self, "_last_collected_epoch", -1)
        if epoch != last:
            logger.info("Final sample collection at epoch %d (early stop or end)", epoch)
            self._collect_samples(pl_module, epoch, trainer.global_step)

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
