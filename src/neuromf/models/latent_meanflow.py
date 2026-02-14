"""PyTorch Lightning module for Latent MeanFlow training.

Implements Algorithm 1 from the methodology: MeanFlow on pre-computed 3D
brain MRI latents with EMA tracking, periodic 1-NFE sample generation,
and TensorBoard logging.
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

from neuromf.sampling.one_step import sample_one_step
from neuromf.utils.ema import EMAModel
from neuromf.utils.time_sampler import sample_t_and_r
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

logger = logging.getLogger(__name__)


class LatentMeanFlow(pl.LightningModule):
    """Latent MeanFlow training module.

    Orchestrates the full training loop: time sampling, MeanFlow loss
    computation (FM + MF with adaptive weighting), EMA tracking, and
    periodic 1-NFE sample generation decoded through the frozen MAISI VAE.

    Args:
        config: Merged OmegaConf config with ``unet``, ``training``,
            ``meanflow``, ``time_sampling``, ``ema``, and ``paths`` sections.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": OmegaConf.to_container(config, resolve=True)})
        self.cfg = config

        # Build model
        unet_config = MAISIUNetConfig.from_omegaconf(config)
        self.net = MAISIUNetWrapper(unet_config)

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

        # Lazy-loaded VAE for sample decoding
        self._vae: Any = None

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

        result = self.loss_pipeline(self.net, z_0, eps, t, r)
        loss = result["loss"]

        # NaN guard: skip step if loss is non-finite
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss at step %d, skipping", self.global_step)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_fm", result["loss_fm"])
        self.log("train/loss_mf", result["loss_mf"])

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

        self.log("val/loss_total", result["loss"], sync_dist=True)
        self.log("val/loss_fm", result["loss_fm"], sync_dist=True)
        self.log("val/loss_mf", result["loss_mf"], sync_dist=True)

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        """Record training loss and generate periodic samples."""
        avg_loss = self.trainer.callback_metrics.get("train/loss_total")
        if avg_loss is not None:
            self._train_loss_history.append(float(avg_loss))

        sample_every = int(self.cfg.get("sample_every_n_epochs", 25))
        if (self.current_epoch + 1) % sample_every == 0:
            n_samples = int(self.cfg.get("n_samples_per_log", 8))
            self._generate_samples(n_samples)

    def on_validation_epoch_end(self) -> None:
        """Record validation loss."""
        avg_loss = self.trainer.callback_metrics.get("val/loss_total")
        if avg_loss is not None:
            self._val_loss_history.append(float(avg_loss))

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """Build AdamW optimizer with linear warmup + cosine decay.

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

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

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

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, n_samples: int = 8) -> None:
        """Generate 1-NFE samples and optionally decode through frozen VAE.

        Uses EMA shadow weights for generation. Decodes through the MAISI
        VAE if weights are available, saving mid-sagittal/axial/coronal
        slices as PNG.

        Args:
            n_samples: Number of samples to generate.
        """
        S = self._latent_spatial
        noise = torch.randn(n_samples, self._in_channels, S, S, S, device=self.device)

        # Apply EMA weights for sampling
        self.ema.apply_shadow(self.net)
        try:
            z_0_hat = sample_one_step(
                self.net, noise, prediction_type=self.cfg.unet.prediction_type
            )
        finally:
            self.ema.restore(self.net)

        # Denormalize back to original latent statistics
        z_0_denorm = z_0_hat * self.latent_std + self.latent_mean

        logger.info(
            "Generated %d samples at epoch %d (latent range: [%.3f, %.3f])",
            n_samples,
            self.current_epoch,
            z_0_hat.min().item(),
            z_0_hat.max().item(),
        )

        # Try to decode through VAE if weights are available
        vae_weights = self.cfg.paths.get("maisi_vae_weights", "")
        samples_dir_str = self.cfg.paths.get("samples_dir", "")
        if vae_weights and Path(vae_weights).exists() and samples_dir_str:
            try:
                vae = self._load_vae()
                decoded = vae.decode(z_0_denorm)

                epoch_dir = Path(samples_dir_str) / f"epoch_{self.current_epoch:04d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)

                self._save_slices(decoded, epoch_dir)
                self._log_images(decoded)
            except Exception as e:
                logger.warning("Sample decoding failed: %s", e)

    def _load_vae(self) -> Any:
        """Lazy-load the frozen MAISI VAE for sample decoding.

        Returns:
            ``MAISIVAEWrapper`` instance on the current device.
        """
        if self._vae is not None:
            return self._vae

        from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

        vae_cfg = self.cfg.vae
        vae_config = MAISIVAEConfig(
            weights_path=str(self.cfg.paths.maisi_vae_weights),
            scale_factor=float(vae_cfg.scale_factor),
            spatial_dims=int(vae_cfg.spatial_dims),
            in_channels=int(vae_cfg.in_channels),
            out_channels=int(vae_cfg.out_channels),
            latent_channels=int(vae_cfg.latent_channels),
            num_channels=list(vae_cfg.num_channels),
            num_res_blocks=list(vae_cfg.num_res_blocks),
            norm_num_groups=int(vae_cfg.norm_num_groups),
            norm_eps=float(vae_cfg.norm_eps),
            attention_levels=list(vae_cfg.attention_levels),
            with_encoder_nonlocal_attn=bool(vae_cfg.with_encoder_nonlocal_attn),
            with_decoder_nonlocal_attn=bool(vae_cfg.with_decoder_nonlocal_attn),
            use_checkpointing=bool(vae_cfg.use_checkpointing),
            use_convtranspose=bool(vae_cfg.use_convtranspose),
            norm_float16=bool(vae_cfg.norm_float16),
            num_splits=int(vae_cfg.num_splits),
            dim_split=int(vae_cfg.dim_split),
            downsample_factor=int(vae_cfg.downsample_factor),
        )
        self._vae = MAISIVAEWrapper(vae_config, device=self.device)
        logger.info("Lazy-loaded MAISI VAE for sample decoding")
        return self._vae

    def _save_slices(self, decoded: Tensor, epoch_dir: Path) -> None:
        """Save mid-sagittal, axial, and coronal slices as PNG.

        Args:
            decoded: Decoded volumes ``(B, 1, H, W, D)``.
            epoch_dir: Directory to save slices.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping slice saving")
            return

        n = decoded.shape[0]
        vol = decoded.cpu().float()
        mid = [s // 2 for s in vol.shape[2:]]

        for i in range(n):
            v = vol[i, 0]  # (H, W, D)
            slices = {
                "sagittal": v[mid[0], :, :],
                "coronal": v[:, mid[1], :],
                "axial": v[:, :, mid[2]],
            }
            for name, sl in slices.items():
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(sl.numpy().T, cmap="gray", origin="lower")
                ax.set_axis_off()
                ax.set_title(f"Sample {i} — {name}")
                fig.savefig(
                    epoch_dir / f"sample_{i:02d}_{name}.png",
                    bbox_inches="tight",
                    dpi=100,
                )
                plt.close(fig)

        logger.info("Saved %d sample slices to %s", n * 3, epoch_dir)

    def _log_images(self, decoded: Tensor) -> None:
        """Log mid-axial slices to TensorBoard.

        Args:
            decoded: Decoded volumes ``(B, 1, H, W, D)``.
        """
        if self.logger is None:
            return

        tb = getattr(self.logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_image"):
            return

        vol = decoded.cpu().float()
        mid_ax = vol.shape[4] // 2

        for i in range(min(vol.shape[0], 4)):
            sl = vol[i, 0, :, :, mid_ax]
            # Normalize to [0, 1] for display
            sl_min, sl_max = sl.min(), sl.max()
            if sl_max > sl_min:
                sl = (sl - sl_min) / (sl_max - sl_min)
            tb.add_image(
                f"samples/sample_{i}",
                sl.unsqueeze(0),
                global_step=self.global_step,
            )
