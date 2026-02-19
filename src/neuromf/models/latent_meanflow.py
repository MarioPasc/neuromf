"""PyTorch Lightning module for Latent MeanFlow training.

Implements Algorithm 1 from the methodology: MeanFlow on pre-computed 3D
brain MRI latents with EMA tracking and TensorBoard logging. Sample
generation is handled by ``SampleCollectorCallback``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from neuromf.utils.ema import EMAModel
from neuromf.utils.time_sampler import sample_t_and_r
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

logger = logging.getLogger(__name__)


class LatentMeanFlow(pl.LightningModule):
    """Latent MeanFlow training module.

    Orchestrates the full training loop: time sampling, unified MeanFlow
    loss (single ||V - v_c||^p with adaptive weighting), and EMA tracking.
    Sample generation is delegated to ``SampleCollectorCallback``.

    Args:
        config: Merged OmegaConf config with ``unet``, ``training``,
            ``meanflow``, ``time_sampling``, ``ema``, and ``paths`` sections.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": OmegaConf.to_container(config, resolve=True)})
        self.cfg = config

        # Build model (use_v_head and conditioning_mode read by from_omegaconf)
        unet_config = MAISIUNetConfig.from_omegaconf(config)
        self.net = MAISIUNetWrapper(unet_config)
        self._use_v_head = unet_config.use_v_head

        # Build loss pipeline
        mf = config.meanflow
        pipeline_config = MeanFlowPipelineConfig(
            p=float(mf.p),
            adaptive=bool(mf.adaptive),
            norm_eps=float(mf.norm_eps),
            lambda_mf=float(mf.lambda_mf),
            prediction_type=str(mf.prediction_type),
            t_min=float(mf.t_min),
            jvp_strategy=str(mf.jvp_strategy),
            fd_step_size=float(mf.fd_step_size),
            channel_weights=(list(mf.channel_weights) if mf.channel_weights else None),
            norm_p=float(mf.get("norm_p", 1.0)),
            spatial_mask_ratio=float(mf.get("spatial_mask_ratio", 0.0)),
            use_v_head=bool(mf.get("use_v_head", False)),
        )
        self.loss_pipeline = MeanFlowPipeline(pipeline_config)

        # EMA
        self.ema = EMAModel(self.net, decay=float(config.ema.decay))

        # Time sampling params
        ts = config.time_sampling
        self._ts_mu = float(ts.mu)
        self._ts_sigma = float(ts.sigma)
        self._ts_t_min = float(ts.t_min)
        self._ts_data_proportion = float(ts.data_proportion)

        # Latent spatial size for sample generation
        self._latent_spatial = int(config.get("latent_spatial_size", 48))
        self._in_channels = int(config.unet.in_channels)

        # Latent stats for denormalization (registered as buffers)
        self._load_latent_stats(config)

        # Loss history for checkpoint serialization
        self._train_loss_history: list[float] = []
        self._val_loss_history: list[float] = []

        # Diagnostics (enabled by TrainingDiagnosticsCallback)
        self._diag_enabled: bool = False
        self._step_diagnostics: dict | None = None

        # EMA-smoothed divergence guard
        # Tracks an EMA of the raw loss to smooth per-step variance (which is
        # huge in MeanFlow due to t/r sampling). Compares EMA to its running
        # minimum. This catches sustained divergence (like v2_baseline's 4808x
        # growth over 225 epochs) but ignores single-step spikes from unlucky
        # time samples.
        self._divergence_threshold = float(config.training.get("divergence_threshold", 0.0))
        self._divergence_grace_steps = int(config.training.get("divergence_grace_steps", 500))
        self._ema_raw_loss: float | None = None
        self._min_ema_raw_loss: float | None = None
        self._ema_decay: float = 0.99  # half-life ≈ 69 steps
        self._divergence_warned_3x: bool = False
        self._divergence_warned_5x: bool = False

        # Set by SampleCollectorCallback to disable legacy sample generation
        self._sample_collector_active: bool = False

        n_params = sum(p.numel() for p in self.net.parameters())
        logger.info("LatentMeanFlow: %d trainable params", n_params)

    def _load_latent_stats(self, config: DictConfig) -> None:
        """Load latent statistics for denormalization.

        Falls back to identity normalization (mean=0, std=1) if the
        stats file does not exist (e.g. in tests).

        Args:
            config: Config with ``paths.latents_dir``.
        """
        latents_dir = config.paths.get("latents_dir", "")
        stats_path = Path(latents_dir) / "latent_stats.json" if latents_dir else None

        if stats_path and stats_path.exists():
            from neuromf.utils.latent_stats import load_latent_stats

            stats = load_latent_stats(stats_path)
            per_ch = stats["per_channel"]
            n_ch = len(per_ch)
            means = [per_ch[f"channel_{c}"]["mean"] for c in range(n_ch)]
            stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)]
            self.register_buffer(
                "latent_mean",
                torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1, 1),
            )
            self.register_buffer(
                "latent_std",
                torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1, 1),
            )
            logger.info("Loaded latent stats from %s", stats_path)
        else:
            self.register_buffer(
                "latent_mean",
                torch.zeros(1, self._in_channels, 1, 1, 1),
            )
            self.register_buffer(
                "latent_std",
                torch.ones(1, self._in_channels, 1, 1, 1),
            )
            logger.warning("Latent stats not found; using identity normalization (mean=0, std=1)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute one training step (Algorithm 1).

        Args:
            batch: Dict with ``"z"`` of shape ``(B, C, D, H, W)``.
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor for backpropagation.
        """
        z_0 = batch["z"]
        B = z_0.shape[0]
        eps = torch.randn_like(z_0)

        t, r = sample_t_and_r(
            B,
            mu=self._ts_mu,
            sigma=self._ts_sigma,
            t_min=self._ts_t_min,
            data_proportion=self._ts_data_proportion,
            device=z_0.device,
        )

        result = self.loss_pipeline(
            self.net,
            z_0,
            eps,
            t,
            r,
            return_diagnostics=self._diag_enabled,
        )
        loss = result["loss"]

        # NaN guard: skip step if loss is non-finite
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss at step %d, skipping", self.global_step)
            self._step_diagnostics = None
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train/loss", loss, prog_bar=True)

        # Always log raw (pre-adaptive) loss — essential for convergence monitoring
        raw_loss = result["raw_loss"]
        self.log("train/raw_loss", raw_loss, prog_bar=True)

        # Dual-head loss components (v-head)
        if "raw_loss_u" in result:
            self.log("train/raw_loss_u", result["raw_loss_u"])
            self.log("train/raw_loss_v", result["raw_loss_v"])

        # EMA-smoothed divergence guard: compares EMA of loss to its running minimum.
        # MeanFlow has huge per-step variance from t/r sampling (100x between easy
        # and hard batches is normal), so raw step-level min-tracking false-triggers.
        # EMA smoothing filters spikes while catching sustained divergence.
        if self._divergence_threshold > 0:
            raw_val = raw_loss.item()

            # Update EMA (decay=0.99, half-life ≈ 69 steps)
            if self._ema_raw_loss is None:
                self._ema_raw_loss = raw_val
            else:
                self._ema_raw_loss = (
                    self._ema_decay * self._ema_raw_loss + (1.0 - self._ema_decay) * raw_val
                )

            # Track minimum of the EMA (not of individual steps)
            if self._min_ema_raw_loss is None or self._ema_raw_loss < self._min_ema_raw_loss:
                self._min_ema_raw_loss = self._ema_raw_loss

            # Check after grace period
            if self.global_step >= self._divergence_grace_steps and self._min_ema_raw_loss > 0:
                ratio = self._ema_raw_loss / self._min_ema_raw_loss
                if ratio > 3.0 and not self._divergence_warned_3x:
                    logger.warning(
                        "Step %d: EMA raw_loss=%.1f is 3x EMA min (%.1f)",
                        self.global_step,
                        self._ema_raw_loss,
                        self._min_ema_raw_loss,
                    )
                    self._divergence_warned_3x = True
                if ratio > 5.0 and not self._divergence_warned_5x:
                    logger.warning(
                        "Step %d: EMA raw_loss=%.1f is 5x EMA min (%.1f)",
                        self.global_step,
                        self._ema_raw_loss,
                        self._min_ema_raw_loss,
                    )
                    self._divergence_warned_5x = True
                if ratio > self._divergence_threshold:
                    raise RuntimeError(
                        f"Divergence detected at step {self.global_step}: "
                        f"EMA raw_loss={self._ema_raw_loss:.1f} > "
                        f"{self._divergence_threshold}x EMA min "
                        f"({self._min_ema_raw_loss:.1f})"
                    )
            self.log("train/ema_raw_loss", self._ema_raw_loss)
            self.log("train/min_ema_raw_loss", self._min_ema_raw_loss)

        if self._diag_enabled:
            self._step_diagnostics = result
            self._step_diagnostics["t"] = t.detach()
            self._step_diagnostics["r"] = r.detach()

        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log pre-clip gradient norm.

        Lightning clips via ``Trainer(gradient_clip_val=...)``. Here we
        compute the norm with ``max_norm=inf`` (no clipping) for logging.

        Args:
            optimizer: Current optimizer.
        """
        total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=float("inf"))
        self.log("train/grad_norm", total_norm)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update EMA after each training step.

        Args:
            outputs: Training step output.
            batch: Current batch.
            batch_idx: Batch index.
        """
        self.ema.update(self.net)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss (no backprop).

        Args:
            batch: Dict with ``"z"`` of shape ``(B, C, D, H, W)``.
            batch_idx: Batch index.
        """
        z_0 = batch["z"]
        B = z_0.shape[0]
        eps = torch.randn_like(z_0)

        t, r = sample_t_and_r(
            B,
            mu=self._ts_mu,
            sigma=self._ts_sigma,
            t_min=self._ts_t_min,
            data_proportion=self._ts_data_proportion,
            device=z_0.device,
        )

        result = self.loss_pipeline(self.net, z_0, eps, t, r)

        self.log("val/loss", result["loss"], sync_dist=True)
        self.log("val/raw_loss", result["raw_loss"], sync_dist=True)

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        """Record training loss history."""
        avg_loss = self.trainer.callback_metrics.get("train/loss")
        if avg_loss is not None:
            self._train_loss_history.append(float(avg_loss))

    def on_validation_epoch_end(self) -> None:
        """Record validation loss."""
        avg_loss = self.trainer.callback_metrics.get("val/loss")
        if avg_loss is not None:
            self._val_loss_history.append(float(avg_loss))

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """Build AdamW optimizer with configurable LR schedule.

        Supports ``"constant"`` (warmup then flat), ``"cosine"`` (warmup then
        cosine decay), and ``"linear"`` (warmup then linear decay).

        Returns:
            Dict with ``optimizer`` and ``lr_scheduler``.
        """
        tr = self.cfg.training
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(tr.lr),
            weight_decay=float(tr.weight_decay),
            betas=tuple(tr.betas),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(tr.warmup_steps)
        lr_schedule = str(tr.get("lr_schedule", "cosine"))

        if lr_schedule == "constant":

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                return 1.0

        elif lr_schedule == "cosine":

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        elif lr_schedule == "linear":

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return max(0.0, 1.0 - progress)

        else:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add EMA state and loss history to checkpoint.

        Args:
            checkpoint: Lightning checkpoint dict (mutated in-place).
        """
        checkpoint["ema_state_dict"] = self.ema.state_dict()
        checkpoint["loss_history"] = {
            "train": self._train_loss_history,
            "val": self._val_loss_history,
        }
        checkpoint["divergence_state"] = {
            "ema_raw_loss": self._ema_raw_loss,
            "min_ema_raw_loss": self._min_ema_raw_loss,
            "warned_3x": self._divergence_warned_3x,
            "warned_5x": self._divergence_warned_5x,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA state and loss history from checkpoint.

        Args:
            checkpoint: Lightning checkpoint dict.
        """
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if "loss_history" in checkpoint:
            self._train_loss_history = checkpoint["loss_history"].get("train", [])
            self._val_loss_history = checkpoint["loss_history"].get("val", [])
        if "divergence_state" in checkpoint:
            ds = checkpoint["divergence_state"]
            self._ema_raw_loss = ds.get("ema_raw_loss")
            self._min_ema_raw_loss = ds.get("min_ema_raw_loss")
            self._divergence_warned_3x = ds.get("warned_3x", False)
            self._divergence_warned_5x = ds.get("warned_5x", False)
