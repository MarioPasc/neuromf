"""Frozen MAISI VAE encoder-decoder wrapper for 3D medical volumes.

Wraps ``monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi``
with convenience methods for encoding, decoding, and round-trip reconstruction.
All parameters are frozen at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class MAISIVAEConfig:
    """Configuration for the MAISI VAE wrapper.

    Defaults match ``config_network_rflow.json`` from NV-Generate-CTMR.
    """

    weights_path: str = ""
    scale_factor: float = 0.96240234375

    # Architecture args
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    latent_channels: int = 4
    num_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    num_res_blocks: list[int] = field(default_factory=lambda: [2, 2, 2])
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    attention_levels: list[bool] = field(default_factory=lambda: [False, False, False])
    with_encoder_nonlocal_attn: bool = False
    with_decoder_nonlocal_attn: bool = False
    use_checkpointing: bool = False
    use_convtranspose: bool = False
    norm_float16: bool = True
    num_splits: int = 4
    dim_split: int = 1
    downsample_factor: int = 4

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> MAISIVAEConfig:
        """Build config from a merged OmegaConf config.

        Reads VAE architecture params from ``cfg.vae`` and the weights path
        from ``cfg.paths.maisi_vae_weights``.

        Args:
            cfg: Merged OmegaConf config (base + vae_validation).

        Returns:
            Populated ``MAISIVAEConfig`` instance.
        """
        vae_cfg = cfg.vae
        return cls(
            weights_path=str(cfg.paths.maisi_vae_weights),
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


class MAISIVAEWrapper(nn.Module):
    """Frozen MAISI VAE wrapper with encode/decode/reconstruct methods.

    Args:
        config: VAE configuration dataclass.
        device: Target device for the model.
    """

    def __init__(self, config: MAISIVAEConfig, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.config = config
        self._device = torch.device(device)

        self.model = AutoencoderKlMaisi(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            latent_channels=config.latent_channels,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
            norm_num_groups=config.norm_num_groups,
            norm_eps=config.norm_eps,
            attention_levels=config.attention_levels,
            with_encoder_nonlocal_attn=config.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=config.with_decoder_nonlocal_attn,
            use_checkpointing=config.use_checkpointing,
            use_convtranspose=config.use_convtranspose,
            norm_float16=config.norm_float16,
            num_splits=config.num_splits,
            dim_split=config.dim_split,
        )

        self._load_weights(config.weights_path)
        self.model.to(self._device)
        self.model.requires_grad_(False)
        self.model.eval()

        self.register_buffer(
            "scale_factor", torch.tensor(config.scale_factor, dtype=torch.float32)
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("MAISI VAE loaded: %d params, scale_factor=%.10f", n_params, config.scale_factor)

    def _load_weights(self, weights_path: str) -> None:
        """Load and unwrap VAE checkpoint weights.

        Args:
            weights_path: Path to ``autoencoder_v2.pt``.
        """
        if not weights_path:
            logger.warning("No weights path provided; using random initialisation")
            return

        logger.info("Loading VAE weights from %s", weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Unwrap "unet_state_dict" key if present
        if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
            state_dict = checkpoint["unet_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        logger.info("VAE weights loaded successfully (%d entries)", len(state_dict))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a volume to latent space.

        Args:
            x: Input tensor of shape ``(B, 1, H, W, D)`` where each spatial
               dimension is divisible by ``downsample_factor``.

        Returns:
            Latent tensor of shape ``(B, 4, H/4, W/4, D/4)``.
        """
        df = self.config.downsample_factor
        assert x.ndim == 5, f"Expected 5D input (B,C,H,W,D), got {x.ndim}D"
        assert x.shape[1] == self.config.in_channels, (
            f"Expected {self.config.in_channels} input channels, got {x.shape[1]}"
        )
        for i, s in enumerate(x.shape[2:]):
            assert s % df == 0, (
                f"Spatial dim {i} ({s}) not divisible by downsample_factor ({df})"
            )

        # norm_float16=True requires autocast to avoid fp16/fp32 type mismatches
        use_autocast = self.config.norm_float16 and x.is_cuda
        with torch.amp.autocast(device_type=x.device.type, enabled=use_autocast):
            z = self.model.encode_stage_2_inputs(x)
        z = z.float()
        logger.debug("Encoded %s -> %s", x.shape, z.shape)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents back to image space.

        Divides by ``scale_factor`` before calling the decoder, following the
        MAISI convention: ``x_hat = decode(z / scale_factor)``.

        Args:
            z: Latent tensor of shape ``(B, 4, h, w, d)``.

        Returns:
            Reconstructed tensor of shape ``(B, 1, h*4, w*4, d*4)``.
        """
        assert z.ndim == 5, f"Expected 5D latent (B,C,h,w,d), got {z.ndim}D"
        assert z.shape[1] == self.config.latent_channels, (
            f"Expected {self.config.latent_channels} latent channels, got {z.shape[1]}"
        )

        z_scaled = z / self.scale_factor
        # norm_float16=True requires autocast to avoid fp16/fp32 type mismatches
        use_autocast = self.config.norm_float16 and z.is_cuda
        with torch.amp.autocast(device_type=z.device.type, enabled=use_autocast):
            x_hat = self.model.decode_stage_2_outputs(z_scaled)
        x_hat = x_hat.float()
        logger.debug("Decoded %s -> %s", z.shape, x_hat.shape)
        return x_hat

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (round-trip reconstruction).

        Args:
            x: Input tensor of shape ``(B, 1, H, W, D)``.

        Returns:
            Reconstructed tensor of shape ``(B, 1, H, W, D)``.
        """
        z = self.encode(x)
        return self.decode(z)
