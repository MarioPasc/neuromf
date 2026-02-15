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

    Orchestrates the full training loop: time sampling, unified MeanFlow
    loss (single ||V - v_c||^p with adaptive weighting), EMA tracking, and
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
            norm_p=float(mf.get("norm_p", 1.0)),
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

        # EMA-based divergence guard
        self._divergence_threshold = float(config.training.get("divergence_threshold", 0.0))
        self._raw_loss_ema: float | None = None

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

        # EMA-based divergence guard: compares step loss to running average
        if self._divergence_threshold > 0:
            raw_val = raw_loss.item()
            if self._raw_loss_ema is None:
                self._raw_loss_ema = raw_val
            else:
                self._raw_loss_ema = 0.99 * self._raw_loss_ema + 0.01 * raw_val
                if raw_val > self._divergence_threshold * self._raw_loss_ema:
                    raise RuntimeError(
                        f"Divergence detected at step {self.global_step}: "
                        f"raw_loss={raw_val:.1f} > "
                        f"{self._divergence_threshold}x EMA ({self._raw_loss_ema:.1f})"
                    )

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
        """Record training loss and generate periodic samples.

        Sample generation runs on rank 0 only to avoid redundant VAE
        loading and filesystem races in multi-GPU DDP.
        """
        avg_loss = self.trainer.callback_metrics.get("train/loss")
        if avg_loss is not None:
            self._train_loss_history.append(float(avg_loss))

        sample_every = int(self.cfg.get("sample_every_n_epochs", 25))
        if (self.current_epoch + 1) % sample_every == 0 and self.global_rank == 0:
            n_samples = int(self.cfg.get("n_samples_per_log", 8))
            self._generate_samples(n_samples)

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

        # Save latent channel visualization (no VAE needed)
        samples_dir_str = self.cfg.paths.get("samples_dir", "")
        if samples_dir_str:
            epoch_dir = Path(samples_dir_str) / f"epoch_{self.current_epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            self._save_latent_channels(z_0_hat, epoch_dir)

        # Try to decode through VAE if weights are available
        vae_weights = self.cfg.paths.get("maisi_vae_weights", "")
        if vae_weights and Path(vae_weights).exists() and samples_dir_str:
            try:
                vae = self._load_vae()

                # Decode one sample at a time to avoid OOM — the VAE
                # decoder activations for 192³ are ~3GB per sample.
                decoded_list = []
                for i in range(n_samples):
                    decoded_i = vae.decode(z_0_denorm[i : i + 1])
                    decoded_list.append(decoded_i.cpu())
                    del decoded_i
                torch.cuda.empty_cache()
                decoded = torch.cat(decoded_list, dim=0)

                self._save_sample_grid(decoded, epoch_dir)
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

    def _save_sample_grid(self, decoded: Tensor, epoch_dir: Path) -> None:
        """Save all decoded samples as one N x 3 grid (rows=samples, cols=views).

        Args:
            decoded: Decoded volumes ``(B, 1, H, W, D)``.
            epoch_dir: Directory to save the grid PNG.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping sample grid")
            return

        n = decoded.shape[0]
        vol = decoded.cpu().float()
        mid = [s // 2 for s in vol.shape[2:]]
        view_names = ["Sagittal", "Coronal", "Axial"]

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None, :]

        # Compute global intensity range across all samples for consistent contrast
        vmin = vol[:, 0].min().item()
        vmax = vol[:, 0].max().item()

        for i in range(n):
            v = vol[i, 0]
            slices = [
                v[mid[0], :, :],
                v[:, mid[1], :],
                v[:, :, mid[2]],
            ]
            for j, sl in enumerate(slices):
                axes[i, j].imshow(sl.numpy().T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                axes[i, j].set_axis_off()
                if i == 0:
                    axes[i, j].set_title(view_names[j], fontsize=11)
            axes[i, 0].set_ylabel(f"#{i}", fontsize=10, rotation=0, labelpad=20, va="center")

        fig.suptitle(
            f"Generated Samples — Epoch {self.current_epoch}, Step {self.global_step}",
            fontsize=12,
            y=0.99,
        )
        fig.tight_layout(rect=[0.03, 0, 1, 0.97])
        fig.savefig(epoch_dir / "samples_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sample grid (%d samples) to %s", n, epoch_dir)

    def _save_latent_channels(self, z_0_hat: Tensor, epoch_dir: Path) -> None:
        """Save 4-channel latent visualization for the first sample.

        Shows each latent channel's mid-slices in a 4x3 grid with per-channel
        statistics annotated — useful for diagnosing what the model learns
        in latent space before VAE decoding.

        Args:
            z_0_hat: Generated latents ``(B, 4, D, H, W)`` (normalised space).
            epoch_dir: Directory to save the grid PNG.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping latent channels")
            return

        lat = z_0_hat[0].cpu().float()  # (4, D, H, W)
        C = lat.shape[0]
        mid = [s // 2 for s in lat.shape[1:]]
        view_names = ["Sagittal", "Coronal", "Axial"]

        fig, axes = plt.subplots(C, 3, figsize=(9, 3 * C))

        for ch in range(C):
            ch_vol = lat[ch]  # (D, H, W)
            slices = [
                ch_vol[mid[0], :, :],
                ch_vol[:, mid[1], :],
                ch_vol[:, :, mid[2]],
            ]
            ch_min = ch_vol.min().item()
            ch_max = ch_vol.max().item()
            ch_mean = ch_vol.mean().item()
            ch_std = ch_vol.std().item()

            for j, sl in enumerate(slices):
                im = axes[ch, j].imshow(
                    sl.numpy().T, cmap="RdBu_r", origin="lower", vmin=ch_min, vmax=ch_max
                )
                axes[ch, j].set_axis_off()
                if ch == 0:
                    axes[ch, j].set_title(view_names[j], fontsize=11)

            axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=10, rotation=0, labelpad=25, va="center")
            # Annotate per-channel stats on rightmost panel
            axes[ch, 2].text(
                1.05,
                0.5,
                f"\u03bc={ch_mean:.3f}\n\u03c3={ch_std:.3f}\n[{ch_min:.2f}, {ch_max:.2f}]",
                transform=axes[ch, 2].transAxes,
                fontsize=8,
                va="center",
                ha="left",
                family="monospace",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
            fig.colorbar(im, ax=axes[ch, 2], fraction=0.046, pad=0.08)

        fig.suptitle(
            f"Latent Channels (sample #0) — Epoch {self.current_epoch}, Step {self.global_step}",
            fontsize=12,
            y=0.99,
        )
        fig.tight_layout(rect=[0.04, 0, 0.96, 0.97])
        fig.savefig(epoch_dir / "latent_channels.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved latent channel grid to %s", epoch_dir)

    def _log_images(self, decoded: Tensor) -> None:
        """Log a combined sample grid to TensorBoard.

        Creates a single grid image with all samples showing 3 orthogonal
        views, rather than logging individual slices.

        Args:
            decoded: Decoded volumes ``(B, 1, H, W, D)``.
        """
        if self.logger is None:
            return

        tb = getattr(self.logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_image"):
            return

        vol = decoded.cpu().float()
        n = min(vol.shape[0], 8)
        mid = [s // 2 for s in vol.shape[2:]]

        # Build a list of normalized slices: n_samples x 3 views
        rows = []
        for i in range(n):
            v = vol[i, 0]
            slices = [
                v[mid[0], :, :].T,
                v[:, mid[1], :].T,
                v[:, :, mid[2]].T,
            ]
            # Pad to same height/width for grid assembly
            max_h = max(s.shape[0] for s in slices)
            max_w = max(s.shape[1] for s in slices)
            padded = []
            for sl in slices:
                ph = max_h - sl.shape[0]
                pw = max_w - sl.shape[1]
                padded.append(torch.nn.functional.pad(sl, (0, pw, 0, ph)))
            rows.append(torch.cat(padded, dim=1))  # (H, W*3)

        grid = torch.stack(rows, dim=0)  # (N, H, W*3)

        # Normalize to [0, 1]
        g_min, g_max = grid.min(), grid.max()
        if g_max > g_min:
            grid = (grid - g_min) / (g_max - g_min)

        # Stack rows vertically into one image
        full_grid = torch.cat(list(grid), dim=0)  # (N*H, W*3)
        tb.add_image(
            "samples/grid",
            full_grid.unsqueeze(0),
            global_step=self.global_step,
        )
