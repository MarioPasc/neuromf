"""Shared SSIM and PSNR computation for 3D medical volumes.

Extracted from validate_vae.py for reuse across validation scripts and tests.
"""

from __future__ import annotations

import math

import torch


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio using the actual data range.

    Args:
        x: Original tensor.
        x_hat: Reconstructed tensor.

    Returns:
        PSNR in dB.
    """
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse < 1e-10:
        return 100.0
    data_range = (x.max() - x.min()).item()
    return 10.0 * math.log10(data_range**2 / mse)


def compute_ssim_3d(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Compute volumetric 3D SSIM using MONAI.

    Uses ``monai.metrics.SSIMMetric`` with ``spatial_dims=3`` and the actual
    data range for correct normalisation.

    Args:
        x: Original tensor of shape ``(1, 1, H, W, D)``.
        x_hat: Reconstructed tensor of shape ``(1, 1, H, W, D)``.

    Returns:
        SSIM value.
    """
    from monai.metrics import SSIMMetric

    data_range = (x.max() - x.min()).item()
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=data_range)
    return ssim_metric(x, x_hat).item()
