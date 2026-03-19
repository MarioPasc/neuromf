"""DDPM Lightning module using MOTFM's MergedModel architecture.

Implements standard DDPM (epsilon-prediction) with DDIM sampling for
variable-step inference. Uses the same UNet as MOTFM for a fair
apples-to-apples comparison: identical capacity, identical data,
only the generative paradigm differs.

Requires MOTFM on sys.path (for ``build_model``).
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from torch import Tensor

logger = logging.getLogger(__name__)


class _MergedModel(nn.Module):
    """Thin wrapper matching MOTFM's MergedModel interface.

    Converts continuous t in [0, 1] to discrete timesteps [0, max_timestep-1]
    before calling the UNet. This ensures checkpoint compatibility with MOTFM.
    """

    def __init__(self, unet: nn.Module, max_timestep: int = 1000) -> None:
        super().__init__()
        self.unet = unet
        self.max_timestep = max_timestep

    def forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass with time conversion.

        Args:
            x: Input tensor ``(B, C, D, H, W)``.
            t: Continuous time in ``[0, 1]``.

        Returns:
            Model output (predicted noise).
        """
        t_discrete = (t * (self.max_timestep - 1)).floor().long()
        if t_discrete.dim() == 0:
            t_discrete = t_discrete.expand(x.shape[0])
        return self.unet(x=x, timesteps=t_discrete, context=None)


def _build_unet(model_config: dict) -> _MergedModel:
    """Build UNet wrapped in MergedModel, using monai.networks.nets.

    Maps MOTFM config keys to MONAI constructor params (``num_channels``
    -> ``channels``) and strips keys not accepted by MONAI.
    """
    from monai.networks.nets import DiffusionModelUNet

    mc = model_config.copy()
    # Keys used by MOTFM but not by MONAI's DiffusionModelUNet
    mc.pop("mask_conditioning", None)
    max_timestep = mc.pop("max_timestep", 1000)
    mc.pop("conditioning_embedding_num_channels", None)
    mc.pop("use_flash_attention", None)

    # MOTFM uses 'num_channels'; MONAI uses 'channels'
    if "num_channels" in mc and "channels" not in mc:
        mc["channels"] = mc.pop("num_channels")

    unet = DiffusionModelUNet(**mc)
    model = _MergedModel(unet=unet, max_timestep=max_timestep)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built UNet: %d params (%.1f MB)", n_params, n_params * 4 / 1e6)
    return model


class DDPMLightningModule(pl.LightningModule):
    """DDPM with epsilon-prediction and DDIM sampling.

    Uses MOTFM's ``build_model()`` to construct the same 132.5M-param
    MergedModel UNet, but trains with noise-prediction MSE loss instead
    of velocity matching.

    Args:
        config: Full config dict with ``model_args``, ``diffusion_args``,
            ``train_args``, and ``data_args`` sections.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)

        self.model: nn.Module = _build_unet(config["model_args"])
        self.config = config

        # Diffusion schedule
        diff = config.get("diffusion_args", {})
        self.num_train_timesteps = int(diff.get("num_train_timesteps", 1000))

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=float(diff.get("beta_start", 1e-4)),
            beta_end=float(diff.get("beta_end", 0.02)),
            schedule=str(diff.get("beta_schedule", "linear_beta")),
        )
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=float(diff.get("beta_start", 1e-4)),
            beta_end=float(diff.get("beta_end", 0.02)),
            schedule=str(diff.get("beta_schedule", "linear_beta")),
        )

        # Conditioning flags (from model_args, same as MOTFM)
        self.mask_conditioning = config["model_args"].get("mask_conditioning", False)
        self.class_conditioning = config["model_args"].get("with_conditioning", False)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("DDPMLightningModule: %d trainable params", n_params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Standard DDPM training: predict noise from noisy input.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, 1, D, H, W)``.
            batch_idx: Batch index.

        Returns:
            Scalar MSE loss.
        """
        x_0 = batch["images"]
        B = x_0.shape[0]

        # Sample noise and timesteps
        eps = torch.randn_like(x_0)
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (B,),
            device=x_0.device,
            dtype=torch.long,
        )

        # Forward diffusion: x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
        x_t = self.noise_scheduler.add_noise(x_0, eps, t)

        # Convert discrete timestep to continuous [0, 1] for MergedModel
        t_continuous = t.float() / (self.num_train_timesteps - 1)

        # Predict noise
        eps_pred = self.model(x=x_t, t=t_continuous)

        loss = F.mse_loss(eps_pred, eps)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, 1, D, H, W)``.
            batch_idx: Batch index.
        """
        x_0 = batch["images"]
        B = x_0.shape[0]

        eps = torch.randn_like(x_0)
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (B,),
            device=x_0.device,
            dtype=torch.long,
        )

        x_t = self.noise_scheduler.add_noise(x_0, eps, t)
        t_continuous = t.float() / (self.num_train_timesteps - 1)
        eps_pred = self.model(x=x_t, t=t_continuous)

        loss = F.mse_loss(eps_pred, eps)
        self.log("val/loss", loss, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Build Adam optimizer (matches MOTFM training setup)."""
        tr = self.config.get("train_args", {})
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(tr.get("lr", 1e-4)),
        )
        return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # DDIM Sampling
    # ------------------------------------------------------------------

    def _set_ddim_timesteps(self, num_steps: int) -> None:
        """Set DDIM timesteps, fixing the 1-NFE edge case.

        MONAI's ``DDIMScheduler.set_timesteps(1)`` produces ``[0]`` (degenerate).
        For true 1-NFE, we need ``[T-1]`` to step from pure noise to x_0.

        Args:
            num_steps: Number of DDIM reverse steps.
        """
        if num_steps == 1:
            self.ddim_scheduler.timesteps = torch.tensor(
                [self.num_train_timesteps - 1],
                dtype=torch.long,
            )
        else:
            self.ddim_scheduler.set_timesteps(num_steps)

    @torch.no_grad()
    def ddim_sample(
        self,
        x_T: Tensor,
        num_steps: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """DDIM reverse process (deterministic, eta=0).

        Args:
            x_T: Initial noise of shape ``(B, C, D, H, W)``.
            num_steps: Number of reverse steps (= NFE).
            device: Target device.

        Returns:
            Denoised ``x_0`` with same shape as ``x_T``.
        """
        self._set_ddim_timesteps(num_steps)
        x_t = x_T.to(device)

        for t in self.ddim_scheduler.timesteps:
            t_int = int(t.item())
            t_continuous = torch.full(
                (x_t.shape[0],),
                t_int / (self.num_train_timesteps - 1),
                device=device,
            )
            eps_pred = self.model(x=x_t, t=t_continuous)
            x_t = self.ddim_scheduler.step(eps_pred, t_int, x_t)[0]

        return x_t
