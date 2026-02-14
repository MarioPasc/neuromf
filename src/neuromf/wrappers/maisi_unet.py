"""MAISI 3D UNet adapted for MeanFlow dual time conditioning.

Wraps MONAI's ``DiffusionModelUNet`` with dual ``(r, t)`` sinusoidal embeddings
for MeanFlow training. The UNet blocks are called directly with the combined
embedding, bypassing the original single-timestep forward method.

We use the **same architecture** as the MAISI diffusion UNet but with random
initialisation and our custom dual-time conditioning (see Phase 3 spec §2.3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Scale continuous [0, 1] times to improve sinusoidal embedding resolution.
# The MONAI embedding uses max_period=10000; with t in [0,1] most frequencies
# would show negligible variation. Scaling by 1000 gives informative features.
_TIME_SCALE = 1000.0


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (Ho et al. 2020).

    Args:
        timesteps: 1-D tensor of N timestep values.
        embedding_dim: Output embedding dimension.
        max_period: Controls minimum frequency.

    Returns:
        Tensor of shape ``(N, embedding_dim)``.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    freqs = torch.exp(exponent / half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))
    return embedding


@dataclass
class MAISIUNetConfig:
    """Configuration for the MAISI UNet wrapper.

    Architecture defaults match ``config_network_rflow.json`` from
    NV-Generate-CTMR, except ``use_flash_attention=False`` (required for
    ``torch.func.jvp``) and ``with_conditioning=False`` (unconditional).
    """

    spatial_dims: int = 3
    in_channels: int = 4
    out_channels: int = 4
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    attention_levels: list[bool] = field(default_factory=lambda: [False, False, True, True])
    num_res_blocks: int = 2
    num_head_channels: list[int] = field(default_factory=lambda: [0, 0, 32, 32])
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    resblock_updown: bool = True
    transformer_num_layers: int = 1
    use_flash_attention: bool = False
    with_conditioning: bool = False
    use_checkpointing: bool = False
    prediction_type: str = "x"
    t_min: float = 0.05

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> MAISIUNetConfig:
        """Build config from OmegaConf.

        Reads UNet architecture from ``cfg.unet``. For ``gradient_checkpointing``,
        checks both ``cfg.unet.gradient_checkpointing`` and
        ``cfg.training.gradient_checkpointing`` (the latter takes precedence if set).

        Args:
            cfg: OmegaConf config with a ``unet`` section and optionally a
                ``training`` section.

        Returns:
            Populated ``MAISIUNetConfig``.
        """
        u = cfg.unet
        # Check both unet and training sections for gradient checkpointing
        use_ckpt = bool(u.get("gradient_checkpointing", False))
        training_cfg = cfg.get("training", {})
        if training_cfg and bool(training_cfg.get("gradient_checkpointing", False)):
            use_ckpt = True
        return cls(
            spatial_dims=int(u.spatial_dims),
            in_channels=int(u.in_channels),
            out_channels=int(u.out_channels),
            channels=list(u.channels),
            attention_levels=list(u.attention_levels),
            num_res_blocks=int(u.num_res_blocks),
            num_head_channels=list(u.num_head_channels),
            norm_num_groups=int(u.norm_num_groups),
            resblock_updown=bool(u.resblock_updown),
            transformer_num_layers=int(u.transformer_num_layers),
            use_flash_attention=bool(u.get("use_flash_attention", False)),
            with_conditioning=bool(u.get("with_conditioning", False)),
            use_checkpointing=use_ckpt,
            prediction_type=str(u.get("prediction_type", "x")),
            t_min=float(u.get("t_min", 0.05)),
        )


def patch_inplace_ops(module: nn.Module) -> int:
    """Recursively replace in-place activations with out-of-place variants.

    Required for ``torch.func.jvp`` compatibility since forward-mode AD
    cannot handle in-place mutations.

    Args:
        module: Root module to patch.

    Returns:
        Number of in-place ops patched.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
            count += 1
        elif isinstance(child, nn.SiLU) and getattr(child, "inplace", False):
            setattr(module, name, nn.SiLU(inplace=False))
            count += 1
        count += patch_inplace_ops(child)
    return count


# ---------------------------------------------------------------------------
# Helpers for gradient checkpointing
# ---------------------------------------------------------------------------
# These are module-level functions (not closures) to avoid issues with
# checkpoint re-execution during backward when loop variables change.


def _ckpt_block_forward(
    block: nn.Module, h: torch.Tensor, emb: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Run a down/middle block for use with ``gradient_checkpoint``.

    Args:
        block: UNet down-block or middle-block.
        h: Hidden state tensor.
        emb: Time embedding tensor.

    Returns:
        Block output (varies by block type).
    """
    return block(hidden_states=h, temb=emb, context=None)


def _ckpt_up_block_forward(
    block: nn.Module, h: torch.Tensor, emb: torch.Tensor, *res_samples: torch.Tensor
) -> torch.Tensor:
    """Run an up block for use with ``gradient_checkpoint``.

    Residual tensors are passed as ``*args`` so each is individually tracked
    by ``checkpoint`` for proper gradient computation.

    Args:
        block: UNet up-block.
        h: Hidden state tensor.
        emb: Time embedding tensor.
        *res_samples: Residual tensors from the down path.

    Returns:
        Up-block output tensor.
    """
    return block(
        hidden_states=h,
        res_hidden_states_list=list(res_samples),
        temb=emb,
        context=None,
    )


class MAISIUNetWrapper(nn.Module):
    """MAISI 3D UNet adapted for MeanFlow dual time conditioning.

    Creates a ``DiffusionModelUNet`` and adds a parallel r-embedding MLP.
    The forward method bypasses the UNet's original ``forward()`` to inject
    the combined ``(t + r)`` embedding directly into the UNet blocks.

    Args:
        config: UNet configuration.
    """

    def __init__(self, config: MAISIUNetConfig) -> None:
        super().__init__()
        self.config = config
        self._use_checkpointing = config.use_checkpointing

        # Build MONAI UNet (provides t-embedding and all conv/attention blocks)
        self.unet = DiffusionModelUNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels,
            attention_levels=config.attention_levels,
            num_res_blocks=config.num_res_blocks,
            num_head_channels=config.num_head_channels,
            norm_num_groups=config.norm_num_groups,
            resblock_updown=config.resblock_updown,
            transformer_num_layers=config.transformer_num_layers,
            use_flash_attention=config.use_flash_attention,
            with_conditioning=config.with_conditioning,
        )

        # Parallel r-embedding MLP (same architecture as unet.time_embed)
        time_embed_dim = config.channels[0] * 4
        self.r_embed = nn.Sequential(
            nn.Linear(config.channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.prediction_type = config.prediction_type
        self.t_min = config.t_min
        self._sinusoidal_dim = config.channels[0]

        # Enable gradient checkpointing if requested
        if config.use_checkpointing:
            n_ckpt = self._count_checkpointed_blocks()
            logger.info(
                "Gradient checkpointing enabled — %d blocks will be checkpointed",
                n_ckpt,
            )

        # Patch in-place ops for JVP compatibility
        n_patched = patch_inplace_ops(self)
        if n_patched > 0:
            logger.info("Patched %d in-place operations for JVP compatibility", n_patched)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MAISIUNetWrapper: %d params, prediction=%s, t_min=%.3f",
            n_params,
            config.prediction_type,
            config.t_min,
        )

    def _count_checkpointed_blocks(self) -> int:
        """Count UNet blocks that will be wrapped with gradient checkpointing.

        Returns:
            Number of blocks (down + middle + up).
        """
        return len(self.unet.down_blocks) + 1 + len(self.unet.up_blocks)

    def _forward_with_dual_emb(self, z_t: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward through UNet blocks with pre-computed dual embedding.

        Replicates ``DiffusionModelUNet.forward()`` logic but skips the
        built-in timestep embedding computation. When ``_use_checkpointing``
        is enabled and the model is in training mode, each block is wrapped
        with ``torch.utils.checkpoint.checkpoint`` to trade compute for memory.

        Args:
            z_t: Input tensor ``(B, C, D, H, W)``.
            emb: Combined time embedding ``(B, time_embed_dim)``.

        Returns:
            Output tensor ``(B, C, D, H, W)``.
        """
        use_ckpt = self._use_checkpointing and self.training
        h = self.unet.conv_in(z_t)

        # Down path
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.unet.down_blocks:
            if use_ckpt:
                h, res_samples = gradient_checkpoint(
                    _ckpt_block_forward,
                    downsample_block,
                    h,
                    emb,
                    use_reentrant=False,
                )
            else:
                h, res_samples = downsample_block(hidden_states=h, temb=emb, context=None)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Middle
        if use_ckpt:
            h = gradient_checkpoint(
                _ckpt_block_forward,
                self.unet.middle_block,
                h,
                emb,
                use_reentrant=False,
            )
        else:
            h = self.unet.middle_block(hidden_states=h, temb=emb, context=None)

        # Up path
        for upsample_block in self.unet.up_blocks:
            idx: int = -len(upsample_block.resnets)
            res_samples = down_block_res_samples[idx:]
            down_block_res_samples = down_block_res_samples[:idx]
            if use_ckpt:
                h = gradient_checkpoint(
                    _ckpt_up_block_forward,
                    upsample_block,
                    h,
                    emb,
                    *res_samples,
                    use_reentrant=False,
                )
            else:
                h = upsample_block(
                    hidden_states=h,
                    res_hidden_states_list=res_samples,
                    temb=emb,
                    context=None,
                )

        return self.unet.out(h)

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with dual ``(r, t)`` time conditioning.

        Args:
            z_t: Noisy latent ``(B, C, D, H, W)``.
            r: Interval start time ``(B,)``, values in ``[0, 1]``.
            t: Interval end time ``(B,)``, values in ``[0, 1]``.

        Returns:
            If ``prediction_type="x"``: x_hat of same shape as ``z_t``.
            If ``prediction_type="u"``: u (average velocity) of same shape.
        """
        # Scale to improve sinusoidal embedding resolution
        t_scaled = t * _TIME_SCALE
        r_scaled = r * _TIME_SCALE

        # Sinusoidal embeddings
        t_sin = get_timestep_embedding(t_scaled, self._sinusoidal_dim)
        r_sin = get_timestep_embedding(r_scaled, self._sinusoidal_dim)

        # Cast to input dtype
        t_sin = t_sin.to(dtype=z_t.dtype)
        r_sin = r_sin.to(dtype=z_t.dtype)

        # MLP projections: t uses UNet's time_embed, r uses our parallel MLP
        t_emb = self.unet.time_embed(t_sin)
        r_emb = self.r_embed(r_sin)

        # Sum embeddings (following pMF convention)
        emb = t_emb + r_emb

        return self._forward_with_dual_emb(z_t, emb)

    def u_from_x(self, z_t: torch.Tensor, x_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert x-prediction to average velocity u.

        ``u = (z_t - x_hat) / max(t, t_min)``

        Args:
            z_t: Noisy latent ``(B, C, ...)``.
            x_pred: Predicted clean data ``(B, C, ...)``.
            t: Time values ``(B,)``.

        Returns:
            Average velocity ``(B, C, ...)``.
        """
        t_safe = t.clamp(min=self.t_min)
        shape = (-1,) + (1,) * (z_t.ndim - 1)
        t_safe = t_safe.view(*shape)
        return (z_t - x_pred) / t_safe
