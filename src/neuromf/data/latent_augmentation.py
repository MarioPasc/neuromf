"""Latent-space data augmentation for MeanFlow training.

Provides a factory function to build a MONAI-based augmentation pipeline
for 3D latent tensors ``(C, D, H, W)``. Only safe transforms are
supported: depth-axis flip, per-channel Gaussian noise, and intensity
scaling. Unsafe transforms (H/W flips, 90-degree rotations) are
excluded based on empirical analysis showing they produce artifacts
after VAE decoding.

Config format (nested ``transforms`` dict)::

    augmentation:
      enabled: true
      transforms:
        flip_d:
          prob: 0.5
        gaussian_noise:
          prob: 0.2
          std_fraction: 0.05
        intensity_scale:
          prob: 0.2
          factors: 0.05
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
from monai.transforms import Compose, RandFlip, RandScaleIntensity

logger = logging.getLogger(__name__)

# Keys from the old flat config format that trigger a deprecation error
_DEPRECATED_FLAT_KEYS = {
    "flip_prob",
    "flip_axes",
    "rotate90_prob",
    "rotate90_axes",
    "gaussian_noise_prob",
    "gaussian_noise_std_fraction",
    "intensity_scale_prob",
    "intensity_scale_factors",
}


class PerChannelGaussianNoise:
    """Add Gaussian noise scaled per-channel by latent statistics.

    When ``channel_stds`` are provided, noise for channel ``c`` is sampled
    as ``N(0, std_fraction * channel_stds[c])``. Otherwise, a uniform
    ``std_fraction`` is used for all channels.

    Args:
        prob: Probability of applying noise.
        std_fraction: Fraction of channel std to use as noise std.
        channel_stds: Per-channel standard deviations from latent stats.
    """

    def __init__(
        self,
        prob: float = 0.2,
        std_fraction: float = 0.05,
        channel_stds: list[float] | None = None,
    ) -> None:
        self.prob = prob
        self.std_fraction = std_fraction
        self.channel_stds = channel_stds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel Gaussian noise.

        Args:
            x: Tensor of shape ``(C, D, H, W)``.

        Returns:
            Noisy tensor (same shape).
        """
        if torch.rand(1).item() > self.prob:
            return x
        if self.channel_stds is not None:
            # Per-channel noise: (C, 1, 1, 1) broadcast
            stds = torch.tensor(
                [self.std_fraction * s for s in self.channel_stds],
                dtype=x.dtype,
                device=x.device,
            ).view(-1, 1, 1, 1)
            noise = torch.randn_like(x) * stds
        else:
            noise = torch.randn_like(x) * self.std_fraction
        return x + noise


def _load_channel_stds(latent_stats_path: Path | None) -> list[float] | None:
    """Load per-channel standard deviations from latent stats JSON.

    Args:
        latent_stats_path: Path to ``latent_stats.json``.

    Returns:
        List of per-channel stds, or None if unavailable.
    """
    if latent_stats_path is None:
        return None
    stats_path = Path(latent_stats_path)
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text())
    per_ch = stats.get("per_channel", {})
    n_channels = len(per_ch)
    if n_channels == 0:
        return None
    channel_stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_channels)]
    logger.info("Augmentation: loaded per-channel stds from %s", stats_path)
    return channel_stds


def build_latent_augmentation(
    config: dict,
    latent_stats_path: Path | None = None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Build a latent augmentation pipeline from config.

    Uses the nested ``transforms`` dict format. Only safe transforms
    are supported: ``flip_d``, ``gaussian_noise``, ``intensity_scale``.

    Args:
        config: Augmentation config dict with ``enabled`` and ``transforms``
            keys. The ``transforms`` dict maps transform names to their
            parameters (each must include ``prob``).
        latent_stats_path: Path to ``latent_stats.json`` for per-channel
            noise calibration. If None or not found, uses uniform noise.

    Returns:
        A ``Compose`` pipeline, or ``None`` if augmentation is disabled.

    Raises:
        ValueError: If old flat config format is detected.
    """
    if not config.get("enabled", False):
        return None

    # Detect deprecated flat config format
    flat_keys_found = _DEPRECATED_FLAT_KEYS & set(config.keys())
    if flat_keys_found:
        raise ValueError(
            f"Deprecated flat augmentation config keys found: {flat_keys_found}. "
            "Migrate to the nested 'transforms' dict format:\n"
            "  augmentation:\n"
            "    enabled: true\n"
            "    transforms:\n"
            "      flip_d:\n"
            "        prob: 0.5\n"
            "      gaussian_noise:\n"
            "        prob: 0.2\n"
            "        std_fraction: 0.05\n"
            "      intensity_scale:\n"
            "        prob: 0.2\n"
            "        factors: 0.05"
        )

    transforms_cfg = config.get("transforms", {})
    if not transforms_cfg:
        logger.warning("Augmentation enabled but no transforms configured")
        return None

    channel_stds = _load_channel_stds(latent_stats_path)
    transforms: list = []

    # Depth-axis flip (axis 0 = D in (C, D, H, W))
    flip_d_cfg = transforms_cfg.get("flip_d", {})
    flip_d_prob = float(flip_d_cfg.get("prob", 0.0))
    if flip_d_prob > 0:
        transforms.append(RandFlip(prob=flip_d_prob, spatial_axis=0))

    # Per-channel Gaussian noise
    noise_cfg = transforms_cfg.get("gaussian_noise", {})
    noise_prob = float(noise_cfg.get("prob", 0.0))
    if noise_prob > 0:
        transforms.append(
            PerChannelGaussianNoise(
                prob=noise_prob,
                std_fraction=float(noise_cfg.get("std_fraction", 0.05)),
                channel_stds=channel_stds,
            )
        )

    # Intensity scaling
    scale_cfg = transforms_cfg.get("intensity_scale", {})
    scale_prob = float(scale_cfg.get("prob", 0.0))
    if scale_prob > 0:
        transforms.append(
            RandScaleIntensity(
                factors=float(scale_cfg.get("factors", 0.05)),
                prob=scale_prob,
            )
        )

    if not transforms:
        logger.warning("All transform probs are 0; augmentation effectively disabled")
        return None

    logger.info("Latent augmentation pipeline: %d transforms", len(transforms))
    return Compose(transforms)
