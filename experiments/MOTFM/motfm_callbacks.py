"""MOTFM training monitor callback.

Provides loss logging to JSON, multi-NFE sample generation with fixed noise,
and end-of-training figure generation (loss curves, NFE evolution grid, NFE
convergence plot).

Designed to run alongside MOTFM's existing TensorBoard logging and
ModelCheckpoint callbacks without modifying any vendored code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class MOTFMTrainingMonitor(pl.Callback):
    """Comprehensive training monitor for MOTFM pixel-space flow matching.

    Tracks three concerns:
    1. **Loss history**: Writes ``training_history.json`` every epoch with
       train/val loss.
    2. **Multi-NFE samples**: Generates volumes at multiple NFE steps from
       fixed noise on validation epochs, saving mid-slice PNGs.
    3. **End-of-training figures**: Loss curves, NFE evolution grid,
       NFE convergence plot, and a ``sample_archive.pt`` file.

    Args:
        output_dir: Root directory for all monitor outputs.
        nfe_list: NFE step counts for sample generation.
        n_samples: Number of samples to generate per NFE level.
        seed: Random seed for fixed noise.
        val_freq: Expected validation frequency (epochs). Used to decide
            when to generate samples.
        config: Full MOTFM config dict (needed for ``build_solver_config``).
    """

    def __init__(
        self,
        output_dir: str,
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

        # State
        self._loss_history: list[dict[str, Any]] = []
        self._best_val_loss: float = float("inf")
        self._best_val_epoch: int = -1
        self._fixed_noise: Tensor | None = None

        # Mid-slice cache for end-of-training grid: {epoch: {nfe: {"axial": arr, "coronal": arr}}}
        self._slice_cache: dict[int, dict[int, dict[str, np.ndarray]]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Create output dirs and generate fixed noise."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "samples").mkdir(exist_ok=True)

        # Load existing history if resuming
        history_path = self._output_dir / "training_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    self._loss_history = json.load(f)
                logger.info(
                    "Resumed loss history: %d entries", len(self._loss_history)
                )
                # Restore best val tracking
                for entry in self._loss_history:
                    vl = entry.get("val_loss")
                    if vl is not None and vl < self._best_val_loss:
                        self._best_val_loss = vl
                        self._best_val_epoch = entry["epoch"]
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse existing history; starting fresh")
                self._loss_history = []

        # Fixed noise: (n_samples, 1, 192, 192, 192) — pixel-space MOTFM
        gen = torch.Generator().manual_seed(self._seed)
        self._fixed_noise = torch.randn(
            self._n_samples, 1, 192, 192, 192, generator=gen
        )
        logger.info(
            "MOTFMTrainingMonitor: output=%s, NFE=%s, %d samples, seed=%d",
            self._output_dir,
            self._nfe_list,
            self._n_samples,
            self._seed,
        )

    # ------------------------------------------------------------------
    # Loss tracking
    # ------------------------------------------------------------------

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record train loss for this epoch."""
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train/loss_epoch")
        if train_loss is None:
            train_loss = trainer.callback_metrics.get("train/loss")

        entry = self._get_or_create_epoch_entry(epoch)
        if train_loss is not None:
            entry["train_loss"] = float(train_loss)
            logger.info("[Epoch %d] train_loss=%.6f", epoch, float(train_loss))
        self._save_history()

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record val loss and generate NFE samples."""
        if not trainer.is_global_zero:
            return

        # Skip sanity check
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get("val/loss")

        entry = self._get_or_create_epoch_entry(epoch)
        if val_loss is not None:
            val_loss_f = float(val_loss)
            entry["val_loss"] = val_loss_f
            logger.info("[Epoch %d] val_loss=%.6f", epoch, val_loss_f)
            if val_loss_f < self._best_val_loss:
                self._best_val_loss = val_loss_f
                self._best_val_epoch = epoch
                logger.info(
                    "New best val_loss=%.6f at epoch %d", val_loss_f, epoch
                )
        self._save_history()

        # Generate multi-NFE samples
        self._generate_samples(pl_module, epoch)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        """Generate volumes at multiple NFE steps from fixed noise."""
        if self._fixed_noise is None:
            return

        try:
            from utils.utils_fm import sample_with_solver
            from inferer import build_solver_config
        except ImportError:
            logger.warning("Cannot import MOTFM sampling utils; skipping sample generation")
            return

        model = pl_module.model
        model.eval()
        device = pl_module.device

        # Determine conditioning from the Lightning module
        mask_cond = getattr(pl_module, "mask_conditioning", False)
        class_cond = getattr(pl_module, "class_conditioning", False)

        epoch_dir = self._output_dir / "samples" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        epoch_slices: dict[int, dict[str, np.ndarray]] = {}

        for nfe in self._nfe_list:
            solver_config = build_solver_config(self._config, num_inference_steps=nfe)
            nfe_slices_axial: list[np.ndarray] = []
            nfe_slices_coronal: list[np.ndarray] = []

            for s_idx in range(self._n_samples):
                noise_s = self._fixed_noise[s_idx : s_idx + 1].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    sol = sample_with_solver(
                        model=model,
                        x_init=noise_s,
                        solver_config=solver_config,
                        cond=None if not class_cond else None,
                        masks=None if not mask_cond else None,
                    )

                # Extract final sample: sol is [time_points, B, C, D, H, W] or [B, C, D, H, W]
                if sol.dim() == noise_s.dim() + 1:
                    sample = sol[-1]
                else:
                    sample = sol
                sample = sample[0, 0].float().cpu().numpy()  # (D, H, W)

                # Mid-slices
                d, h, w = sample.shape
                axial_slice = sample[d // 2, :, :]
                coronal_slice = sample[:, h // 2, :]

                nfe_slices_axial.append(axial_slice)
                nfe_slices_coronal.append(coronal_slice)

                # Save individual PNGs
                self._save_slice_png(
                    axial_slice,
                    epoch_dir / f"nfe_{nfe:03d}_sample_{s_idx}_axial.png",
                )
                self._save_slice_png(
                    coronal_slice,
                    epoch_dir / f"nfe_{nfe:03d}_sample_{s_idx}_coronal.png",
                )

                # Free VRAM
                del noise_s, sol, sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Cache sample #0 slices for end-of-training grid
            epoch_slices[nfe] = {
                "axial": nfe_slices_axial[0],
                "coronal": nfe_slices_coronal[0],
            }

        self._slice_cache[epoch] = epoch_slices
        logger.info(
            "Generated samples for epoch %d: NFE=%s, %d samples each",
            epoch,
            self._nfe_list,
            self._n_samples,
        )

    # ------------------------------------------------------------------
    # End of training
    # ------------------------------------------------------------------

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate figures and save sample archive."""
        if not trainer.is_global_zero:
            return

        logger.info(
            "Training complete. Best val_loss=%.6f at epoch %d",
            self._best_val_loss,
            self._best_val_epoch,
        )

        # Report best checkpoint
        ckpt_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                ckpt_cb = cb
                break
        if ckpt_cb is not None and ckpt_cb.best_model_path:
            logger.info("Best checkpoint: %s", ckpt_cb.best_model_path)

        # Save sample archive
        self._save_sample_archive()

        # Generate figures
        try:
            self._plot_loss_curves()
            self._plot_nfe_evolution_grid()
            self._plot_nfe_convergence()
        except Exception as e:
            logger.error("Figure generation failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Figure 1: Loss curves
    # ------------------------------------------------------------------

    def _plot_loss_curves(self) -> None:
        """Plot train and validation loss curves."""
        if not self._loss_history:
            logger.warning("No loss history; skipping loss curves")
            return

        epochs = [e["epoch"] for e in self._loss_history]
        train_losses = [e.get("train_loss") for e in self._loss_history]
        val_epochs = [e["epoch"] for e in self._loss_history if e.get("val_loss") is not None]
        val_losses = [e["val_loss"] for e in self._loss_history if e.get("val_loss") is not None]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Filter out None values for train
        valid_train = [(ep, tl) for ep, tl in zip(epochs, train_losses) if tl is not None]
        if valid_train:
            t_ep, t_loss = zip(*valid_train)
            ax.plot(t_ep, t_loss, "b-", alpha=0.7, linewidth=1, label="Train loss")

        if val_losses:
            ax.plot(val_epochs, val_losses, "r-o", markersize=4, linewidth=1.5, label="Val loss")

        # Best val epoch marker
        if self._best_val_epoch >= 0:
            ax.axvline(
                self._best_val_epoch,
                color="green",
                linestyle="--",
                alpha=0.6,
                label=f"Best val (ep {self._best_val_epoch}, {self._best_val_loss:.6f})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale("log")
        ax.set_title("MOTFM Training Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self._output_dir / f"loss_curves.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved loss_curves.{png,pdf}")

    # ------------------------------------------------------------------
    # Figure 2: NFE evolution grid
    # ------------------------------------------------------------------

    def _plot_nfe_evolution_grid(self) -> None:
        """Plot NFE evolution grid: rows=epochs, cols=NFE x {axial, coronal}."""
        if not self._slice_cache:
            logger.warning("No cached slices; skipping NFE evolution grid")
            return

        sorted_epochs = sorted(self._slice_cache.keys())
        n_epochs = len(sorted_epochs)
        n_nfe = len(self._nfe_list)
        n_cols = n_nfe * 2  # axial + coronal per NFE

        fig, axes = plt.subplots(
            n_epochs, n_cols, figsize=(3 * n_cols, 2.5 * n_epochs),
            squeeze=False,
        )

        # Compute global vmin/vmax for consistent colormap
        all_values: list[float] = []
        for epoch_data in self._slice_cache.values():
            for nfe_data in epoch_data.values():
                all_values.append(float(nfe_data["axial"].min()))
                all_values.append(float(nfe_data["axial"].max()))
                all_values.append(float(nfe_data["coronal"].min()))
                all_values.append(float(nfe_data["coronal"].max()))
        vmin, vmax = min(all_values), max(all_values)

        for row, epoch in enumerate(sorted_epochs):
            for nfe_idx, nfe in enumerate(self._nfe_list):
                col_ax = nfe_idx * 2
                col_cor = nfe_idx * 2 + 1

                if nfe in self._slice_cache[epoch]:
                    slices = self._slice_cache[epoch][nfe]
                    axes[row, col_ax].imshow(
                        slices["axial"], cmap="gray", vmin=vmin, vmax=vmax
                    )
                    axes[row, col_cor].imshow(
                        slices["coronal"], cmap="gray", vmin=vmin, vmax=vmax
                    )
                else:
                    axes[row, col_ax].text(0.5, 0.5, "N/A", ha="center", va="center")
                    axes[row, col_cor].text(0.5, 0.5, "N/A", ha="center", va="center")

                axes[row, col_ax].set_xticks([])
                axes[row, col_ax].set_yticks([])
                axes[row, col_cor].set_xticks([])
                axes[row, col_cor].set_yticks([])

                # Column headers
                if row == 0:
                    axes[0, col_ax].set_title(f"NFE={nfe} Axial", fontsize=9)
                    axes[0, col_cor].set_title(f"NFE={nfe} Coronal", fontsize=9)

            # Row labels
            axes[row, 0].set_ylabel(f"Ep {epoch}", fontsize=9, rotation=0, labelpad=35)

        fig.suptitle("NFE Evolution (Sample #0, Fixed Noise)", fontsize=12, y=1.01)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                self._output_dir / f"nfe_evolution_grid.{ext}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
        logger.info("Saved nfe_evolution_grid.{png,pdf}")

    # ------------------------------------------------------------------
    # Figure 3: NFE convergence
    # ------------------------------------------------------------------

    def _plot_nfe_convergence(self) -> None:
        """Plot MSE between NFE=1/10 and NFE=50 over epochs."""
        if not self._slice_cache:
            logger.warning("No cached slices; skipping NFE convergence plot")
            return

        max_nfe = max(self._nfe_list)
        sorted_epochs = sorted(self._slice_cache.keys())

        # Build per-NFE MSE vs max_nfe
        nfe_mse: dict[int, list[tuple[int, float]]] = {}
        for nfe in self._nfe_list:
            if nfe == max_nfe:
                continue
            nfe_mse[nfe] = []
            for epoch in sorted_epochs:
                if nfe not in self._slice_cache[epoch] or max_nfe not in self._slice_cache[epoch]:
                    continue
                ax_nfe = self._slice_cache[epoch][nfe]["axial"]
                ax_ref = self._slice_cache[epoch][max_nfe]["axial"]
                mse = float(np.mean((ax_nfe - ax_ref) ** 2))
                nfe_mse[nfe].append((epoch, mse))

        if not any(nfe_mse.values()):
            logger.warning("Not enough NFE data for convergence plot")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, (nfe, data) in enumerate(sorted(nfe_mse.items())):
            if not data:
                continue
            ep_list, mse_list = zip(*data)
            ax.plot(
                ep_list, mse_list, "-o",
                color=colors[i % len(colors)],
                markersize=4,
                linewidth=1.5,
                label=f"MSE(NFE={nfe}, NFE={max_nfe})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (axial mid-slice)")
        ax.set_yscale("log")
        ax.set_title(f"NFE Convergence (vs NFE={max_nfe})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                self._output_dir / f"nfe_convergence.{ext}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
        logger.info("Saved nfe_convergence.{png,pdf}")

    # ------------------------------------------------------------------
    # Sample archive
    # ------------------------------------------------------------------

    def _save_sample_archive(self) -> None:
        """Save slice cache and loss history as a .pt archive."""
        if not self._slice_cache and not self._loss_history:
            return

        archive: dict[str, Any] = {
            "metadata": {
                "seed": self._seed,
                "nfe_list": self._nfe_list,
                "n_samples": self._n_samples,
            },
            "loss_history": self._loss_history,
            "epochs": sorted(self._slice_cache.keys()),
        }

        # Store fixed noise
        if self._fixed_noise is not None:
            archive["noise"] = self._fixed_noise

        # Store slices per epoch
        for epoch in sorted(self._slice_cache.keys()):
            epoch_key = f"epoch_{epoch:03d}"
            epoch_data: dict[str, Any] = {}
            for nfe, slices in self._slice_cache[epoch].items():
                epoch_data[f"nfe_{nfe}"] = {
                    "slices": {
                        "axial": torch.from_numpy(slices["axial"]),
                        "coronal": torch.from_numpy(slices["coronal"]),
                    }
                }
            archive[epoch_key] = epoch_data

        archive_path = self._output_dir / "sample_archive.pt"
        torch.save(archive, archive_path)
        logger.info("Saved sample archive: %s", archive_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_epoch_entry(self, epoch: int) -> dict[str, Any]:
        """Get or create the loss history entry for a given epoch."""
        for entry in self._loss_history:
            if entry["epoch"] == epoch:
                return entry
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": None, "val_loss": None}
        self._loss_history.append(entry)
        self._loss_history.sort(key=lambda e: e["epoch"])
        return entry

    def _save_history(self) -> None:
        """Incrementally save loss history to JSON."""
        history_path = self._output_dir / "training_history.json"
        try:
            with open(history_path, "w") as f:
                json.dump(self._loss_history, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save training history: %s", e)

    @staticmethod
    def _save_slice_png(arr: np.ndarray, path: Path) -> None:
        """Save a 2D numpy array as a grayscale PNG."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
