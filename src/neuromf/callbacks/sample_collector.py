"""Sample collection callback for multi-NFE latent archive during training.

Replaces in-training VAE decode with a lightweight latent-only approach:
saves raw generated latents at multiple NFE steps using fixed noise seeds,
computes per-epoch latent statistics, and defers all VAE decoding to a
post-training CLI (``decode_samples.py``).

Benefits: zero VAE VRAM during training, multi-NFE comparison, crash-safe
.pt archives per epoch, richer scientific analysis of generation quality.
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

    @torch.no_grad()
    def _collect_samples(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        global_step: int,
    ) -> None:
        """Generate samples at multiple NFE steps and save archive.

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

        # Move all tensors to CPU for saving
        archive: dict = {
            "epoch": epoch,
            "global_step": global_step,
            "noise_seed": self._seed,
            "noise": noise.cpu(),
            "latent_mean": pl_module.latent_mean.cpu().squeeze(),
            "latent_std": pl_module.latent_std.cpu().squeeze(),
            "stats": stats,
            "nfe_consistency": nfe_consistency,
        }
        for key, z in nfe_samples.items():
            archive[key] = z.cpu()

        # Save archive
        out_dir = self._samples_dir / "generated_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / f"epoch_{epoch:04d}.pt"
        torch.save(archive, archive_path)

        logger.info(
            "Saved sample archive: epoch=%d, NFEs=%s, path=%s",
            epoch,
            list(nfe_samples.keys()),
            archive_path,
        )

        # Save latent channel visualization PNG
        if "nfe_1" in nfe_samples:
            self._save_latent_channels(
                nfe_samples["nfe_1"].cpu(),
                epoch,
                global_step,
                out_dir.parent,
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

    def _save_latent_channels(
        self,
        z_0_hat: Tensor,
        epoch: int,
        global_step: int,
        parent_dir: Path,
    ) -> None:
        """Save 4-channel latent visualization for the first sample.

        Shows each latent channel's mid-slices in a 4x3 grid with per-channel
        statistics annotated.

        Args:
            z_0_hat: Generated latents ``(B, 4, D, H, W)`` (normalised space).
            epoch: Current epoch number.
            global_step: Current global step.
            parent_dir: Parent directory (``samples_dir``).
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping latent channels")
            return

        epoch_dir = parent_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        lat = z_0_hat[0].float()  # (4, D, H, W)
        C = lat.shape[0]
        mid = [s // 2 for s in lat.shape[1:]]
        view_names = ["Sagittal", "Coronal", "Axial"]

        global_min = lat.min().item()
        global_max = lat.max().item()
        abs_lim = max(abs(global_min), abs(global_max))

        fig, axes = plt.subplots(
            C,
            3,
            figsize=(9, 3 * C + 0.6),
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )

        im = None
        for ch in range(C):
            ch_vol = lat[ch]
            slices = [
                ch_vol[mid[0], :, :],
                ch_vol[:, mid[1], :],
                ch_vol[:, :, mid[2]],
            ]
            ch_mean = ch_vol.mean().item()
            ch_std = ch_vol.std().item()
            ch_min = ch_vol.min().item()
            ch_max = ch_vol.max().item()

            for j, sl in enumerate(slices):
                im = axes[ch, j].imshow(
                    sl.numpy().T,
                    cmap="RdBu_r",
                    origin="lower",
                    vmin=-abs_lim,
                    vmax=abs_lim,
                )
                axes[ch, j].set_axis_off()
                if ch == 0:
                    axes[ch, j].set_title(view_names[j], fontsize=11)

            stats_str = (
                f"Ch {ch}  \u03bc={ch_mean:+.3f}  \u03c3={ch_std:.3f}  [{ch_min:.2f}, {ch_max:.2f}]"
            )
            axes[ch, 0].text(
                0.02,
                0.97,
                stats_str,
                transform=axes[ch, 0].transAxes,
                fontsize=7,
                va="top",
                ha="left",
                family="monospace",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
            )

        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.04,
            aspect=40,
        )
        cbar.set_label("Latent value", fontsize=10)

        fig.suptitle(
            f"Latent Channels (1-NFE, sample #0) \u2014 Epoch {epoch}, Step {global_step}",
            fontsize=12,
            y=0.99,
        )
        fig.savefig(epoch_dir / "latent_channels.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved latent channel grid to %s", epoch_dir)
