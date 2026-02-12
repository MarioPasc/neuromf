"""MRI preprocessing pipeline using MONAI transforms.

Provides composable transforms for loading, reorienting, resampling,
intensity-normalising, and cropping/padding 3D brain MRI volumes to a
fixed 128^3 target shape suitable for the MAISI VAE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_mri_preprocessing_transform(
    target_shape: tuple[int, int, int] = (128, 128, 128),
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lower_percentile: float = 0.0,
    upper_percentile: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> Compose:
    """Build a MONAI preprocessing pipeline for brain MRI volumes.

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
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=lower_percentile,
                upper=upper_percentile,
                b_min=b_min,
                b_max=b_max,
                clip=False,
            ),
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
    target_shape: tuple[int, int, int] = (128, 128, 128),
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


def get_ixi_file_list(
    dataset_root: str | Path,
    n_volumes: int | None = None,
) -> list[dict[str, str]]:
    """Glob IXI T1 NIfTI files and return MONAI-format dicts.

    Args:
        dataset_root: Root directory containing ``*.nii.gz`` files.
        n_volumes: If set, return only the first *n_volumes* (sorted by name).

    Returns:
        List of dicts ``[{"image": "/path/to/file.nii.gz"}, ...]``.
    """
    root = Path(dataset_root)
    files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {root}")
    logger.info("Found %d NIfTI files in %s", len(files), root)

    if n_volumes is not None:
        files = files[:n_volumes]
        logger.info("Using first %d volumes", n_volumes)

    return [{"image": str(f)} for f in files]
