"""Latent-space data augmentation for MeanFlow training.

Provides a factory function to build a MONAI-based augmentation pipeline
for 3D latent tensors ``(C, D, H, W)``. All transforms are toggleable via
config and designed to be injected into ``LatentDataset`` via its
``transform`` parameter.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
from monai.transforms import Compose, RandFlip, RandRotate90, RandScaleIntensity

logger = logging.getLogger(__name__)


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
        prob: float = 0.3,
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


def build_latent_augmentation(
    config: dict,
    latent_stats_path: Path | None = None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Build a latent augmentation pipeline from config.

    Args:
        config: Augmentation config dict with keys like ``enabled``,
            ``flip_prob``, ``flip_axes``, ``rotate90_prob``, etc.
        latent_stats_path: Path to ``latent_stats.json`` for per-channel
            noise calibration. If None or not found, uses uniform noise.

    Returns:
        A ``Compose`` pipeline, or ``None`` if augmentation is disabled.
    """
    if not config.get("enabled", False):
        return None

    transforms: list = []

    # Spatial flips
    flip_prob = config.get("flip_prob", 0.5)
    flip_axes = config.get("flip_axes", [0, 1, 2])
    for axis in flip_axes:
        transforms.append(RandFlip(prob=flip_prob, spatial_axis=int(axis)))

    # 90-degree rotations
    rotate90_prob = config.get("rotate90_prob", 0.5)
    rotate90_axes = config.get("rotate90_axes", [[1, 2]])
    for pair in rotate90_axes:
        transforms.append(RandRotate90(prob=rotate90_prob, max_k=3, spatial_axes=tuple(pair)))

    # Per-channel Gaussian noise
    noise_prob = config.get("gaussian_noise_prob", 0.3)
    noise_std_fraction = config.get("gaussian_noise_std_fraction", 0.05)
    channel_stds: list[float] | None = None
    if latent_stats_path is not None:
        stats_path = Path(latent_stats_path)
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            per_ch = stats.get("per_channel", {})
            n_channels = len(per_ch)
            if n_channels > 0:
                channel_stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_channels)]
                logger.info("Augmentation: loaded per-channel stds from %s", stats_path)
    transforms.append(
        PerChannelGaussianNoise(
            prob=noise_prob,
            std_fraction=noise_std_fraction,
            channel_stds=channel_stds,
        )
    )

    # Intensity scaling
    scale_prob = config.get("intensity_scale_prob", 0.3)
    scale_factors = config.get("intensity_scale_factors", 0.05)
    transforms.append(RandScaleIntensity(factors=scale_factors, prob=scale_prob))

    logger.info("Latent augmentation pipeline: %d transforms", len(transforms))
    return Compose(transforms)
