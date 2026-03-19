"""Training callbacks for DDPM baseline.

Tracks loss history, generates multi-step DDIM samples during training,
and produces end-of-training figures. Mirrors MOTFM's MOTFMTrainingMonitor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class DDPMTrainingMonitor(pl.Callback):
    """Track DDPM training: loss history, multi-step samples, figures.

    Args:
        output_dir: Directory for loss JSON, sample PNGs, figures.
        nfe_list: DDIM step counts for sample generation (maps to NFE).
        n_samples: Number of volumes to generate per step count.
        seed: Fixed noise seed for reproducible evolution tracking.
        val_freq: Expected validation frequency (epochs).
        config: Full DDPM config dict.
    """

    def __init__(
        self,
        output_dir: str | Path,
        nfe_list: list[int] | None = None,
        n_samples: int = 4,
        seed: int = 42,
        val_freq: int = 10,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._output_dir = Path(output_dir)
        self._nfe_list = nfe_list if nfe_list is not None else [1, 10, 50]
        self._n_samples = n_samples
        self._seed = seed
        self._val_freq = val_freq
        self._config = config or {}

        self._loss_history: list[dict[str, Any]] = []
        self._best_val_loss: float = float("inf")
        self._best_val_epoch: int = -1
        self._fixed_noise: Tensor | None = None
        self._slice_cache: dict[int, dict[int, dict[str, np.ndarray]]] = {}

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Create output dirs and generate fixed noise."""
        if not trainer.is_global_zero:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "samples").mkdir(exist_ok=True)

        # Resume history if available
        history_path = self._output_dir / "training_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    self._loss_history = json.load(f)
                logger.info("Resumed loss history: %d entries", len(self._loss_history))
                for entry in self._loss_history:
                    vl = entry.get("val_loss")
                    if vl is not None and vl < self._best_val_loss:
                        self._best_val_loss = vl
                        self._best_val_epoch = entry["epoch"]
            except (json.JSONDecodeError, KeyError):
                self._loss_history = []

        vol_s = int(self._config.get("data_args", {}).get("volume_size", 192))
        gen = torch.Generator().manual_seed(self._seed)
        self._fixed_noise = torch.randn(self._n_samples, 1, vol_s, vol_s, vol_s, generator=gen)
        logger.info(
            "DDPMTrainingMonitor: output=%s, steps=%s, %d samples, vol=%d^3",
            self._output_dir,
            self._nfe_list,
            self._n_samples,
            vol_s,
        )

    def _get_or_create_epoch_entry(self, epoch: int) -> dict[str, Any]:
        """Get or create a loss history entry for the given epoch."""
        for entry in self._loss_history:
            if entry.get("epoch") == epoch:
                return entry
        entry: dict[str, Any] = {"epoch": epoch}
        self._loss_history.append(entry)
        return entry

    def _save_history(self) -> None:
        """Write loss history to JSON."""
        path = self._output_dir / "training_history.json"
        try:
            with open(path, "w") as f:
                json.dump(self._loss_history, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save history: %s", e)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record training loss."""
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train/loss")
        entry = self._get_or_create_epoch_entry(epoch)
        if train_loss is not None:
            entry["train_loss"] = float(train_loss)
        self._save_history()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record val loss and generate multi-step samples."""
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get("val/loss")
        entry = self._get_or_create_epoch_entry(epoch)
        if val_loss is not None:
            val_loss_f = float(val_loss)
            entry["val_loss"] = val_loss_f
            if val_loss_f < self._best_val_loss:
                self._best_val_loss = val_loss_f
                self._best_val_epoch = epoch
                logger.info("New best val_loss=%.6f at epoch %d", val_loss_f, epoch)
        self._save_history()
        self._generate_samples(pl_module, epoch)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, pl_module: pl.LightningModule, epoch: int) -> None:
        """Generate volumes at multiple DDIM step counts from fixed noise."""
        if self._fixed_noise is None:
            return

        model = pl_module
        model.eval()
        device = pl_module.device

        epoch_dir = self._output_dir / "samples" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        epoch_slices: dict[int, dict[str, np.ndarray]] = {}

        for nfe in self._nfe_list:
            nfe_slices_axial: list[np.ndarray] = []
            nfe_slices_coronal: list[np.ndarray] = []

            for s_idx in range(self._n_samples):
                noise_s = self._fixed_noise[s_idx : s_idx + 1].to(device)

                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
                ):
                    sample = pl_module.ddim_sample(noise_s, num_steps=nfe, device=device)

                sample = sample[0, 0].float().clamp(0, 1).cpu().numpy()
                d, h, w = sample.shape
                axial = sample[d // 2, :, :]
                coronal = sample[:, h // 2, :]

                nfe_slices_axial.append(axial)
                nfe_slices_coronal.append(coronal)

                self._save_slice_png(axial, epoch_dir / f"nfe_{nfe:03d}_sample_{s_idx}_axial.png")
                self._save_slice_png(
                    coronal, epoch_dir / f"nfe_{nfe:03d}_sample_{s_idx}_coronal.png"
                )

                del noise_s, sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_slices[nfe] = {
                "axial": nfe_slices_axial[0],
                "coronal": nfe_slices_coronal[0],
            }

        self._slice_cache[epoch] = epoch_slices
        logger.info("Generated DDPM samples epoch %d: steps=%s", epoch, self._nfe_list)

    @staticmethod
    def _save_slice_png(arr: np.ndarray, path: Path) -> None:
        """Save a 2D slice as grayscale PNG."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0.02, dpi=100)
        plt.close(fig)

    # ------------------------------------------------------------------
    # End of training
    # ------------------------------------------------------------------

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate end-of-training figures."""
        if not trainer.is_global_zero:
            return

        self._plot_loss_curves()
        self._plot_nfe_evolution_grid()
        self._save_archive(trainer.current_epoch)

    def _plot_loss_curves(self) -> None:
        """Plot train/val loss curves."""
        if not self._loss_history:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = [e["epoch"] for e in self._loss_history]
        train = [e.get("train_loss") for e in self._loss_history]
        val = [e.get("val_loss") for e in self._loss_history]

        t_ep = [e for e, v in zip(epochs, train) if v is not None]
        t_val = [v for v in train if v is not None]
        v_ep = [e for e, v in zip(epochs, val) if v is not None]
        v_val = [v for v in val if v is not None]

        if t_ep:
            ax.plot(t_ep, t_val, label="Train", alpha=0.8)
        if v_ep:
            ax.plot(v_ep, v_val, "o-", label="Val", markersize=3)
        if self._best_val_epoch >= 0:
            ax.axvline(
                self._best_val_epoch,
                color="green",
                ls="--",
                alpha=0.5,
                label=f"Best (ep {self._best_val_epoch})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("DDPM Training Loss")

        path = self._output_dir / "loss_curves.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curves saved to %s", path)

    def _plot_nfe_evolution_grid(self) -> None:
        """Plot NFE evolution grid: epochs (rows) x steps (cols)."""
        if not self._slice_cache:
            return

        sorted_epochs = sorted(self._slice_cache.keys())
        n_epochs = len(sorted_epochs)
        n_nfe = len(self._nfe_list)
        n_cols = n_nfe * 2

        fig, axes = plt.subplots(
            n_epochs, n_cols, figsize=(3 * n_cols, 2.5 * n_epochs), squeeze=False
        )

        for row, epoch in enumerate(sorted_epochs):
            for nfe_idx, nfe in enumerate(self._nfe_list):
                col_ax = nfe_idx * 2
                col_cor = nfe_idx * 2 + 1
                if nfe in self._slice_cache[epoch]:
                    slices = self._slice_cache[epoch][nfe]
                    axes[row, col_ax].imshow(slices["axial"], cmap="gray", vmin=0, vmax=1)
                    axes[row, col_cor].imshow(slices["coronal"], cmap="gray", vmin=0, vmax=1)
                for c in [col_ax, col_cor]:
                    axes[row, c].set_xticks([])
                    axes[row, c].set_yticks([])
                if row == 0:
                    axes[0, col_ax].set_title(f"Steps={nfe} ax")
                    axes[0, col_cor].set_title(f"Steps={nfe} cor")
            axes[row, 0].set_ylabel(f"Ep {epoch}", fontsize=8)

        fig.suptitle("DDPM Sample Evolution (DDIM sampling)", fontsize=12)
        fig.tight_layout()
        path = self._output_dir / "nfe_evolution_grid.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Evolution grid saved to %s", path)

    def _save_archive(self, final_epoch: int) -> None:
        """Save training archive (.pt) with history and fixed noise."""
        archive = {
            "loss_history": self._loss_history,
            "best_val_loss": self._best_val_loss,
            "best_val_epoch": self._best_val_epoch,
            "fixed_noise": self._fixed_noise,
            "nfe_list": self._nfe_list,
            "final_epoch": final_epoch,
        }
        path = self._output_dir / "training_archive.pt"
        torch.save(archive, str(path))
        logger.info("Training archive saved to %s", path)
