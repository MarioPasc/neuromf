"""Training diagnostics callback for MeanFlow convergence monitoring.

Three-tier logging:
- Tier 1 (step): loss ratio, velocity norms, adaptive weights, grad norm
- Tier 2 (epoch): loss std, grad clip fraction, per-block grad norms,
  relative update norm, EMA divergence, sampling stats, velocity norms,
  raw loss, FM/MF split, cosine similarities, prediction stats, JSON summary
- Tier 3 (periodic): per-channel loss, latent histograms
"""

from __future__ import annotations

import json
import logging
import time  # noqa: F401 - used in on_train_epoch_start/end
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
    # r_embed / time_embed / v_out at the wrapper level
    if parts[0] in ("r_embed", "time_embed", "v_out"):
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
        self._epoch_raw_losses: list[float] = []
        self._epoch_grad_norms: list[float] = []
        self._epoch_block_grad_sums: dict[str, float] = {}
        self._epoch_block_grad_counts: dict[str, int] = {}
        self._epoch_t_values: list[float] = []
        self._epoch_r_values: list[float] = []
        self._epoch_channel_losses: list[torch.Tensor] = []

        # Velocity norm accumulators
        self._epoch_u_norms: list[float] = []
        self._epoch_compound_v_norms: list[float] = []
        self._epoch_target_v_norms: list[float] = []
        self._epoch_v_tilde_norms: list[float] = []
        self._epoch_jvp_norms: list[float] = []

        # Adaptive weight accumulators
        self._epoch_adp_weight_means: list[float] = []
        self._epoch_adp_weight_stds: list[float] = []

        # FM/MF loss split accumulators
        self._epoch_raw_loss_fm: list[float] = []
        self._epoch_raw_loss_mf: list[float] = []

        # Cosine similarity accumulators
        self._epoch_cosine_V_vc: list[float] = []
        self._epoch_cosine_vtilde_vc: list[float] = []

        # Relative error accumulator
        self._epoch_relative_error: list[float] = []

        # x-hat stats accumulators (x-prediction mode)
        self._epoch_x_hat_means: list[float] = []
        self._epoch_x_hat_stds: list[float] = []
        self._epoch_x_hat_mins: list[float] = []
        self._epoch_x_hat_maxs: list[float] = []

        # u-pred stats accumulators (u-prediction mode)
        self._epoch_u_pred_means: list[float] = []
        self._epoch_u_pred_stds: list[float] = []
        self._epoch_u_pred_mins: list[float] = []
        self._epoch_u_pred_maxs: list[float] = []

        # v-head accumulators (dual-head mode)
        self._epoch_raw_loss_u: list[float] = []
        self._epoch_raw_loss_v: list[float] = []
        self._epoch_v_head_norms: list[float] = []
        self._epoch_cosine_v_vc: list[float] = []

        # Parameter snapshot for relative update norm
        self._param_snapshot: dict[str, torch.Tensor] = {}

        # Epoch wall time
        self._epoch_start_time: float = 0.0

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
        raw_loss = diag.get("raw_loss")

        # Accumulate for epoch stats
        if loss_total is not None:
            self._epoch_losses.append(float(loss_total.detach()))
        if raw_loss is not None:
            self._epoch_raw_losses.append(float(raw_loss.detach()))

        # Velocity norms — log to TensorBoard AND accumulate for JSON
        for key in (
            "diag_jvp_norm",
            "diag_u_norm",
            "diag_compound_v_norm",
            "diag_target_v_norm",
            "diag_v_tilde_norm",
        ):
            if key in diag:
                short = key.replace("diag_", "")
                val = float(diag[key])
                pl_module.log(f"train/meanflow/{short}", val)

        # Accumulate velocity norms for epoch-level JSON
        if "diag_u_norm" in diag:
            self._epoch_u_norms.append(float(diag["diag_u_norm"]))
        if "diag_compound_v_norm" in diag:
            self._epoch_compound_v_norms.append(float(diag["diag_compound_v_norm"]))
        if "diag_target_v_norm" in diag:
            self._epoch_target_v_norms.append(float(diag["diag_target_v_norm"]))
        if "diag_v_tilde_norm" in diag:
            self._epoch_v_tilde_norms.append(float(diag["diag_v_tilde_norm"]))
        if "diag_jvp_norm" in diag:
            self._epoch_jvp_norms.append(float(diag["diag_jvp_norm"]))

        # Adaptive weight stats and FM fraction — log and accumulate
        for key in ("diag_adaptive_weight_mean", "diag_adaptive_weight_std", "diag_fm_fraction"):
            if key in diag:
                short = key.replace("diag_", "")
                pl_module.log(f"train/meanflow/{short}", float(diag[key]))

        if "diag_adaptive_weight_mean" in diag:
            self._epoch_adp_weight_means.append(float(diag["diag_adaptive_weight_mean"]))
        if "diag_adaptive_weight_std" in diag:
            self._epoch_adp_weight_stds.append(float(diag["diag_adaptive_weight_std"]))

        # FM/MF split — log and accumulate
        if "diag_raw_loss_fm" in diag:
            val = float(diag["diag_raw_loss_fm"])
            pl_module.log("train/meanflow/raw_loss_fm", val)
            self._epoch_raw_loss_fm.append(val)
        if "diag_raw_loss_mf" in diag:
            val = float(diag["diag_raw_loss_mf"])
            pl_module.log("train/meanflow/raw_loss_mf", val)
            self._epoch_raw_loss_mf.append(val)

        # Cosine similarities — log and accumulate
        if "diag_cosine_sim_V_vc" in diag:
            val = float(diag["diag_cosine_sim_V_vc"])
            pl_module.log("train/meanflow/cosine_sim_V_vc", val)
            self._epoch_cosine_V_vc.append(val)
        if "diag_cosine_sim_vtilde_vc" in diag:
            val = float(diag["diag_cosine_sim_vtilde_vc"])
            pl_module.log("train/meanflow/cosine_sim_vtilde_vc", val)
            self._epoch_cosine_vtilde_vc.append(val)

        # Relative error — log and accumulate
        if "diag_relative_error" in diag:
            val = float(diag["diag_relative_error"])
            pl_module.log("train/meanflow/relative_error", val)
            self._epoch_relative_error.append(val)

        # x-hat / u-pred stats — log and accumulate (mutually exclusive)
        for key, acc in (
            ("diag_x_hat_mean", self._epoch_x_hat_means),
            ("diag_x_hat_std", self._epoch_x_hat_stds),
            ("diag_x_hat_min", self._epoch_x_hat_mins),
            ("diag_x_hat_max", self._epoch_x_hat_maxs),
            ("diag_u_pred_mean", self._epoch_u_pred_means),
            ("diag_u_pred_std", self._epoch_u_pred_stds),
            ("diag_u_pred_min", self._epoch_u_pred_mins),
            ("diag_u_pred_max", self._epoch_u_pred_maxs),
        ):
            if key in diag:
                val = float(diag[key])
                short = key.replace("diag_", "")
                pl_module.log(f"train/meanflow/{short}", val)
                acc.append(val)

        # v-head diagnostics — log and accumulate
        if "raw_loss_u" in diag:
            val_u = float(diag["raw_loss_u"])
            val_v = float(diag["raw_loss_v"])
            pl_module.log("train/meanflow/raw_loss_u", val_u)
            pl_module.log("train/meanflow/raw_loss_v", val_v)
            self._epoch_raw_loss_u.append(val_u)
            self._epoch_raw_loss_v.append(val_v)
        if "diag_v_head_norm" in diag:
            val = float(diag["diag_v_head_norm"])
            pl_module.log("train/meanflow/v_head_norm", val)
            self._epoch_v_head_norms.append(val)
        if "diag_cosine_sim_v_vc" in diag:
            val = float(diag["diag_cosine_sim_v_vc"])
            pl_module.log("train/meanflow/cosine_sim_v_vc", val)
            self._epoch_cosine_v_vc.append(val)

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
        """Snapshot parameters, record start time, and reset accumulators."""
        self._epoch_start_time = time.monotonic()
        self._param_snapshot = {
            name: p.data.detach().clone()
            for name, p in pl_module.net.named_parameters()
            if p.requires_grad
        }

        # Reset ALL epoch accumulators
        self._epoch_losses.clear()
        self._epoch_raw_losses.clear()
        self._epoch_grad_norms.clear()
        self._epoch_block_grad_sums.clear()
        self._epoch_block_grad_counts.clear()
        self._epoch_t_values.clear()
        self._epoch_r_values.clear()
        self._epoch_channel_losses.clear()
        self._epoch_u_norms.clear()
        self._epoch_compound_v_norms.clear()
        self._epoch_target_v_norms.clear()
        self._epoch_v_tilde_norms.clear()
        self._epoch_jvp_norms.clear()
        self._epoch_adp_weight_means.clear()
        self._epoch_adp_weight_stds.clear()
        self._epoch_raw_loss_fm.clear()
        self._epoch_raw_loss_mf.clear()
        self._epoch_cosine_V_vc.clear()
        self._epoch_cosine_vtilde_vc.clear()
        self._epoch_relative_error.clear()
        self._epoch_x_hat_means.clear()
        self._epoch_x_hat_stds.clear()
        self._epoch_x_hat_mins.clear()
        self._epoch_x_hat_maxs.clear()
        self._epoch_u_pred_means.clear()
        self._epoch_u_pred_stds.clear()
        self._epoch_u_pred_mins.clear()
        self._epoch_u_pred_maxs.clear()
        self._epoch_raw_loss_u.clear()
        self._epoch_raw_loss_v.clear()
        self._epoch_v_head_norms.clear()
        self._epoch_cosine_v_vc.clear()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log Tier 2 (epoch) and Tier 3 (periodic) metrics."""
        epoch = trainer.current_epoch
        summary: dict[str, Any] = {"epoch": epoch, "global_step": trainer.global_step}

        # --- Adaptive-weighted loss stats ---
        if len(self._epoch_losses) > 1:
            loss_t = torch.tensor(self._epoch_losses)
            loss_std = float(loss_t.std())
            pl_module.log("train/loss_std", loss_std, sync_dist=True)
            summary["loss_mean"] = float(loss_t.mean())
            summary["loss_std"] = loss_std

        # --- Raw (pre-adaptive) loss stats ---
        if self._epoch_raw_losses:
            raw_t = torch.tensor(self._epoch_raw_losses)
            summary["raw_loss_mean"] = float(raw_t.mean())
            summary["raw_loss_std"] = float(raw_t.std())
            pl_module.log("train/raw_loss_mean", float(raw_t.mean()), sync_dist=True)
            pl_module.log("train/raw_loss_std", float(raw_t.std()), sync_dist=True)

        # --- Gradient clip fraction ---
        if self._epoch_grad_norms:
            clipped = sum(1 for g in self._epoch_grad_norms if g > self._clip_val)
            clip_frac = clipped / len(self._epoch_grad_norms)
            grad_norm_t = torch.tensor(self._epoch_grad_norms)
            pl_module.log("train/gradients/clip_fraction", clip_frac, sync_dist=True)
            summary["grad_clip_fraction"] = clip_frac
            summary["grad_norm_mean"] = float(grad_norm_t.mean())
            summary["grad_norm_std"] = float(grad_norm_t.std())

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

        # --- Velocity norms (epoch means) ---
        velocity_norms: dict[str, float] = {}
        for key, acc in (
            ("u_norm", self._epoch_u_norms),
            ("compound_v_norm", self._epoch_compound_v_norms),
            ("target_v_norm", self._epoch_target_v_norms),
            ("v_tilde_norm", self._epoch_v_tilde_norms),
            ("jvp_norm", self._epoch_jvp_norms),
        ):
            if acc:
                velocity_norms[key] = sum(acc) / len(acc)
        if velocity_norms:
            summary["velocity_norms"] = velocity_norms

        # --- Adaptive weight stats (epoch means) ---
        if self._epoch_adp_weight_means:
            summary["adaptive_weight"] = {
                "mean": sum(self._epoch_adp_weight_means) / len(self._epoch_adp_weight_means),
                "std": sum(self._epoch_adp_weight_stds) / len(self._epoch_adp_weight_stds),
            }

        # --- FM/MF raw loss split ---
        if self._epoch_raw_loss_fm:
            summary["raw_loss_fm_mean"] = sum(self._epoch_raw_loss_fm) / len(
                self._epoch_raw_loss_fm
            )
        if self._epoch_raw_loss_mf:
            summary["raw_loss_mf_mean"] = sum(self._epoch_raw_loss_mf) / len(
                self._epoch_raw_loss_mf
            )

        # --- Cosine similarities ---
        if self._epoch_cosine_V_vc:
            summary["cosine_sim_V_vc"] = sum(self._epoch_cosine_V_vc) / len(self._epoch_cosine_V_vc)
        if self._epoch_cosine_vtilde_vc:
            summary["cosine_sim_vtilde_vc"] = sum(self._epoch_cosine_vtilde_vc) / len(
                self._epoch_cosine_vtilde_vc
            )

        # --- Relative prediction error ---
        if self._epoch_relative_error:
            summary["relative_error"] = sum(self._epoch_relative_error) / len(
                self._epoch_relative_error
            )

        # --- Prediction-specific statistics ---
        if self._epoch_x_hat_means:
            summary["x_hat_stats"] = {
                "mean": sum(self._epoch_x_hat_means) / len(self._epoch_x_hat_means),
                "std": sum(self._epoch_x_hat_stds) / len(self._epoch_x_hat_stds),
                "min": min(self._epoch_x_hat_mins),
                "max": max(self._epoch_x_hat_maxs),
            }
        if self._epoch_u_pred_means:
            summary["u_pred_stats"] = {
                "mean": sum(self._epoch_u_pred_means) / len(self._epoch_u_pred_means),
                "std": sum(self._epoch_u_pred_stds) / len(self._epoch_u_pred_stds),
                "min": min(self._epoch_u_pred_mins),
                "max": max(self._epoch_u_pred_maxs),
            }

        # --- v-head metrics (dual-head mode) ---
        if self._epoch_raw_loss_u:
            summary["raw_loss_u_mean"] = sum(self._epoch_raw_loss_u) / len(self._epoch_raw_loss_u)
        if self._epoch_raw_loss_v:
            summary["raw_loss_v_mean"] = sum(self._epoch_raw_loss_v) / len(self._epoch_raw_loss_v)
        if self._epoch_v_head_norms:
            summary["v_head_norm"] = sum(self._epoch_v_head_norms) / len(self._epoch_v_head_norms)
        if self._epoch_cosine_v_vc:
            summary["cosine_sim_v_vc"] = sum(self._epoch_cosine_v_vc) / len(self._epoch_cosine_v_vc)

        # --- Learning rate ---
        lr_sched = trainer.lr_scheduler_configs
        if lr_sched:
            current_lr = lr_sched[0].scheduler.get_last_lr()[0]
            summary["learning_rate"] = current_lr

        # --- Validation loss (from callback_metrics if available) ---
        val_loss = trainer.callback_metrics.get("val/loss")
        val_raw = trainer.callback_metrics.get("val/raw_loss")
        if val_loss is not None:
            summary["val_loss"] = float(val_loss)
        if val_raw is not None:
            summary["val_raw_loss"] = float(val_raw)

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

        # --- Epoch wall time ---
        if self._epoch_start_time > 0:
            summary["epoch_time_sec"] = time.monotonic() - self._epoch_start_time

        # --- Merge sample collector stats if archive exists ---
        samples_dir = getattr(pl_module, "cfg", {})
        if hasattr(samples_dir, "paths"):
            sd = str(samples_dir.paths.get("samples_dir", ""))
            if sd:
                archive_path = Path(sd) / "generated_samples" / f"epoch_{epoch:04d}.pt"
                if archive_path.exists():
                    try:
                        archive = torch.load(archive_path, map_location="cpu", weights_only=False)
                        summary["generated_samples"] = {
                            "stats": archive.get("stats", {}),
                            "nfe_consistency": archive.get("nfe_consistency", {}),
                        }
                    except Exception as e:
                        logger.warning("Failed to load sample archive for diagnostics: %s", e)

        # --- Write JSON summary (rank 0 only in DDP) ---
        self._training_history.append(summary)
        if self._diag_dir is not None and trainer.is_global_zero:
            self._write_json_summary(epoch, summary)

    def _write_json_summary(self, epoch: int, summary: dict[str, Any]) -> None:
        """Write per-epoch and cumulative JSON summaries.

        Folder structure::

            diagnostics/
                per_epoch_metrics/
                    epoch_000/summary.json
                    epoch_001/summary.json
                    ...
                aggregate_results/
                    training_summary.json

        Args:
            epoch: Current epoch number.
            summary: Epoch summary dict.
        """
        try:
            per_epoch_dir = self._diag_dir / "per_epoch_metrics" / f"epoch_{epoch:03d}"
            per_epoch_dir.mkdir(parents=True, exist_ok=True)
            with open(per_epoch_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            agg_dir = self._diag_dir / "aggregate_results"
            agg_dir.mkdir(parents=True, exist_ok=True)
            with open(agg_dir / "training_summary.json", "w") as f:
                json.dump(self._training_history, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write diagnostics JSON: %s", e)
