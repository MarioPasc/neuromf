"""MRI preprocessing pipeline using MONAI transforms.

Provides composable transforms for resampling, intensity-normalising, and
cropping/padding 3D brain MRI volumes to a fixed target shape suitable
for the MAISI VAE. Uses brain-centered cropping (CropForegroundd) to ensure
the brain is centered in the output volume before padding/cropping to the
target shape. Designed for FOMO-60K data which is already skull-stripped
and RAS-oriented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_mri_preprocessing_transform(
    target_shape: tuple[int, int, int] = (192, 192, 192),
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lower_percentile: float = 0.0,
    upper_percentile: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> Compose:
    """Build a MONAI preprocessing pipeline for brain MRI volumes.

    Pipeline: Load -> Resample(1mm) -> ScaleIntensity -> CropForeground
    (brain-centered) -> ResizeWithPadOrCrop(target) -> EnsureType.

    CropForegroundd centres the crop on brain tissue before padding to the
    target shape, preventing systematic loss of frontal/occipital cortex
    that occurs with naive volume-centred cropping.

    Args:
        target_shape: Spatial dimensions after crop/pad.
        target_spacing: Isotropic voxel spacing in mm.
        lower_percentile: Lower intensity percentile for normalisation.
        upper_percentile: Upper intensity percentile for normalisation.
        b_min: Output intensity minimum.
        b_max: Output intensity maximum.

    Returns:
        MONAI ``Compose`` transform operating on dict with key ``"image"``.
    """
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            # No Orientationd — FOMO-60K data is already RAS-oriented
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=lower_percentile,
                upper=upper_percentile,
                b_min=b_min,
                b_max=b_max,
                clip=False,
            ),
            # Brain-centered crop: tight bounding box around foreground tissue,
            # then pad/crop to exact target shape. margin=4 adds a small buffer.
            CropForegroundd(keys=["image"], source_key="image", margin=4),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=target_shape),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


def build_mri_preprocessing_from_config(config: DictConfig) -> Compose:
    """Build preprocessing transform from an OmegaConf config.

    Expects ``config.data`` to contain ``target_shape``, ``target_spacing``,
    ``intensity_lower_percentile``, ``intensity_upper_percentile``,
    ``intensity_b_min``, and ``intensity_b_max``.

    Args:
        config: Merged OmegaConf config (base + vae_validation).

    Returns:
        MONAI ``Compose`` transform.
    """
    data_cfg = config.data
    return build_mri_preprocessing_transform(
        target_shape=tuple(data_cfg.target_shape),
        target_spacing=tuple(data_cfg.target_spacing),
        lower_percentile=data_cfg.intensity_lower_percentile,
        upper_percentile=data_cfg.intensity_upper_percentile,
        b_min=data_cfg.intensity_b_min,
        b_max=data_cfg.intensity_b_max,
    )


def preprocess_single_volume(
    nifti_path: str | Path,
    target_shape: tuple[int, int, int] = (192, 192, 192),
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Preprocess a single NIfTI volume and return as a tensor.

    Args:
        nifti_path: Path to a ``.nii.gz`` file.
        target_shape: Spatial dimensions after crop/pad.
        target_spacing: Isotropic voxel spacing in mm.

    Returns:
        Float32 tensor of shape ``(1, H, W, D)`` with intensity in [0, 1].
    """
    transform = build_mri_preprocessing_transform(
        target_shape=target_shape, target_spacing=target_spacing
    )
    data = transform({"image": str(nifti_path)})
    return data["image"]
