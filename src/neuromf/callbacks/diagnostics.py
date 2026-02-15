"""Training diagnostics callback for MeanFlow convergence monitoring.

Three-tier logging:
- Tier 1 (step): loss ratio, velocity norms, adaptive weights, grad norm
- Tier 2 (epoch): loss std, grad clip fraction, per-block grad norms,
  relative update norm, EMA divergence, sampling stats, JSON summary
- Tier 3 (periodic): per-channel loss, latent histograms
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


def _get_block_name(param_name: str) -> str:
    """Map a parameter name to its architectural block.

    Args:
        param_name: Full dotted parameter name from ``named_parameters()``.

    Returns:
        Short block label (e.g. ``"down_block_2"``, ``"r_embed"``).
    """
    parts = param_name.split(".")
    # r_embed / time_embed at the wrapper level
    if parts[0] in ("r_embed", "time_embed"):
        return parts[0]
    # UNet sub-blocks
    if "unet" in parts:
        idx = parts.index("unet")
        sub = parts[idx + 1] if idx + 1 < len(parts) else "other"
        if sub == "down_blocks" and idx + 2 < len(parts):
            return f"down_block_{parts[idx + 2]}"
        if sub == "up_blocks" and idx + 2 < len(parts):
            return f"up_block_{parts[idx + 2]}"
        if sub == "middle_block":
            return "middle_block"
        if sub == "out":
            return "out"
        if sub == "conv_in":
            return "conv_in"
        return sub
    return "other"


class TrainingDiagnosticsCallback(pl.Callback):
    """Comprehensive training diagnostics via three-tier logging.

    Args:
        diag_every_n_epochs: Frequency for Tier 3 (periodic) diagnostics.
        diagnostics_dir: Directory for JSON summaries. Empty string disables
            file output.
        gradient_clip_val: Gradient clip threshold for computing clip fraction.
    """

    def __init__(
        self,
        diag_every_n_epochs: int = 25,
        diagnostics_dir: str = "",
        gradient_clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        self._diag_every = diag_every_n_epochs
        self._diag_dir = Path(diagnostics_dir) if diagnostics_dir else None
        self._clip_val = gradient_clip_val

        # Epoch-level accumulators (reset each epoch)
        self._epoch_losses: list[float] = []
        self._epoch_grad_norms: list[float] = []
        self._epoch_block_grad_sums: dict[str, float] = {}
        self._epoch_block_grad_counts: dict[str, int] = {}
        self._epoch_t_values: list[float] = []
        self._epoch_r_values: list[float] = []
        self._epoch_channel_losses: list[torch.Tensor] = []

        # Parameter snapshot for relative update norm
        self._param_snapshot: dict[str, torch.Tensor] = {}

        # Full training history for training_summary.json
        self._training_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Enable diagnostics on the Lightning module."""
        pl_module._diag_enabled = True  # type: ignore[attr-defined]
        logger.info("TrainingDiagnosticsCallback: diagnostics enabled")

    # ------------------------------------------------------------------
    # Tier 1 — Step-level
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Read step diagnostics and log Tier 1 metrics."""
        diag = getattr(pl_module, "_step_diagnostics", None)
        if diag is None:
            return

        loss_total = diag.get("loss")

        # Accumulate for epoch stats
        if loss_total is not None:
            self._epoch_losses.append(float(loss_total.detach()))

        # Velocity norms
        for key in (
            "diag_jvp_norm",
            "diag_u_norm",
            "diag_compound_v_norm",
            "diag_target_v_norm",
        ):
            if key in diag:
                short = key.replace("diag_", "")
                pl_module.log(f"train/meanflow/{short}", float(diag[key]))

        # Adaptive weight stats and FM fraction
        for key in ("diag_adaptive_weight_mean", "diag_adaptive_weight_std", "diag_fm_fraction"):
            if key in diag:
                short = key.replace("diag_", "")
                pl_module.log(f"train/meanflow/{short}", float(diag[key]))

        # Accumulate t/r values for epoch sampling stats
        if "t" in diag:
            self._epoch_t_values.extend(diag["t"].detach().cpu().tolist())
        if "r" in diag:
            self._epoch_r_values.extend(diag["r"].detach().cpu().tolist())

        # Accumulate per-channel loss for periodic logging
        if "diag_loss_per_channel" in diag:
            self._epoch_channel_losses.append(diag["diag_loss_per_channel"].detach().cpu())

        # Clear step diagnostics to avoid memory accumulation
        pl_module._step_diagnostics = None  # type: ignore[attr-defined]

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Compute and accumulate per-block gradient norms."""
        total_norm_sq = 0.0
        for name, param in pl_module.net.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm_sq += grad_norm**2

                block = _get_block_name(name)
                self._epoch_block_grad_sums[block] = (
                    self._epoch_block_grad_sums.get(block, 0.0) + grad_norm**2
                )
                self._epoch_block_grad_counts[block] = (
                    self._epoch_block_grad_counts.get(block, 0) + 1
                )

        global_norm = total_norm_sq**0.5
        self._epoch_grad_norms.append(global_norm)
        pl_module.log("train/gradients/global_norm", global_norm)

    # ------------------------------------------------------------------
    # Tier 2 — Epoch-level
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Snapshot parameters and reset accumulators."""
        self._param_snapshot = {
            name: p.data.detach().clone()
            for name, p in pl_module.net.named_parameters()
            if p.requires_grad
        }

        # Reset epoch accumulators
        self._epoch_losses.clear()
        self._epoch_grad_norms.clear()
        self._epoch_block_grad_sums.clear()
        self._epoch_block_grad_counts.clear()
        self._epoch_t_values.clear()
        self._epoch_r_values.clear()
        self._epoch_channel_losses.clear()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log Tier 2 (epoch) and Tier 3 (periodic) metrics."""
        epoch = trainer.current_epoch
        summary: dict[str, Any] = {"epoch": epoch, "global_step": trainer.global_step}

        # --- Loss std ---
        if len(self._epoch_losses) > 1:
            loss_t = torch.tensor(self._epoch_losses)
            loss_std = float(loss_t.std())
            pl_module.log("train/loss_std", loss_std, sync_dist=True)
            summary["loss_mean"] = float(loss_t.mean())
            summary["loss_std"] = loss_std

        # --- Gradient clip fraction ---
        if self._epoch_grad_norms:
            clipped = sum(1 for g in self._epoch_grad_norms if g > self._clip_val)
            clip_frac = clipped / len(self._epoch_grad_norms)
            pl_module.log("train/gradients/clip_fraction", clip_frac, sync_dist=True)
            summary["grad_clip_fraction"] = clip_frac

        # --- Per-block gradient norms (RMS over epoch) ---
        block_norms: dict[str, float] = {}
        for block, sq_sum in self._epoch_block_grad_sums.items():
            count = self._epoch_block_grad_counts.get(block, 1)
            rms = (sq_sum / count) ** 0.5
            block_norms[block] = rms
            pl_module.log(f"train/gradients/{block}", rms, sync_dist=True)
        if block_norms:
            summary["block_grad_norms"] = block_norms

        # --- Relative update norm ---
        if self._param_snapshot:
            update_sq = 0.0
            theta_sq = 0.0
            for name, p in pl_module.net.named_parameters():
                if name in self._param_snapshot:
                    diff = p.data - self._param_snapshot[name].to(p.device)
                    update_sq += diff.norm().item() ** 2
                    theta_sq += p.data.norm().item() ** 2
            rel_update = update_sq**0.5 / (theta_sq**0.5 + 1e-8)
            pl_module.log("train/updates/relative_update_norm", rel_update, sync_dist=True)
            summary["relative_update_norm"] = rel_update
            self._param_snapshot.clear()

        # --- EMA divergence ---
        ema = getattr(pl_module, "ema", None)
        if ema is not None and hasattr(ema, "shadow"):
            diff_sq = 0.0
            theta_sq = 0.0
            for name, p in pl_module.net.named_parameters():
                if name in ema.shadow:
                    diff_sq += (p.data - ema.shadow[name].to(p.device)).norm().item() ** 2
                    theta_sq += p.data.norm().item() ** 2
            ema_div = diff_sq**0.5 / (theta_sq**0.5 + 1e-8)
            pl_module.log("ema/param_divergence", ema_div, sync_dist=True)
            summary["ema_divergence"] = ema_div

        # --- Sampling stats ---
        if self._epoch_t_values:
            t_tensor = torch.tensor(self._epoch_t_values)
            r_tensor = torch.tensor(self._epoch_r_values)
            h_tensor = t_tensor - r_tensor
            pl_module.log("train/sampling/t_mean", float(t_tensor.mean()), sync_dist=True)
            pl_module.log("train/sampling/t_std", float(t_tensor.std()), sync_dist=True)
            pl_module.log("train/sampling/h_mean", float(h_tensor.mean()), sync_dist=True)
            h_zero_frac = float((h_tensor < 1e-6).float().mean())
            pl_module.log("train/sampling/h_zero_frac", h_zero_frac, sync_dist=True)
            summary["sampling"] = {
                "t_mean": float(t_tensor.mean()),
                "t_std": float(t_tensor.std()),
                "h_mean": float(h_tensor.mean()),
                "h_zero_frac": h_zero_frac,
            }

        # --- Tier 3: Periodic diagnostics ---
        is_periodic = (epoch + 1) % self._diag_every == 0
        if is_periodic and self._epoch_channel_losses:
            stacked = torch.stack(self._epoch_channel_losses)
            ch_mean = stacked.mean(dim=0)
            for c in range(ch_mean.shape[0]):
                pl_module.log(f"train/channel/loss_ch{c}", float(ch_mean[c]), sync_dist=True)
            summary["per_channel_loss"] = ch_mean.tolist()

        # --- Write JSON summary (rank 0 only in DDP) ---
        self._training_history.append(summary)
        if self._diag_dir is not None and trainer.is_global_zero:
            self._write_json_summary(epoch, summary)

    def _write_json_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Write per-epoch and cumulative JSON summaries.

        Args:
            epoch: Current epoch number.
            summary: Epoch summary dict.
        """
        try:
            epoch_dir = self._diag_dir / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            with open(epoch_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            with open(self._diag_dir / "training_summary.json", "w") as f:
                json.dump(self._training_history, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write diagnostics JSON: %s", e)
