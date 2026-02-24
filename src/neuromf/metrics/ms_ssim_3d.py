"""3D Multi-Scale Structural Similarity (MS-SSIM).

Computes MS-SSIM on 3D medical volumes using a multi-level downsampling
approach with MONAI's SSIMMetric at each scale level.

Reference: Wang, Simoncelli & Bovik, "Multiscale Structural Similarity
for Image Quality Assessment", Asilomar 2003.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from neuromf.metrics.ssim_psnr import compute_ssim_3d


def compute_ms_ssim_3d(
    x: Tensor,
    y: Tensor,
    data_range: float = 1.0,
    n_levels: int = 3,
    weights: list[float] | None = None,
) -> float:
    """Compute 3D Multi-Scale SSIM.

    Computes SSIM at ``n_levels`` resolution scales (original + downsampled)
    and combines them with the given weights.

    Args:
        x: Reference volume, shape ``(1, 1, D, H, W)``.
        y: Comparison volume, shape ``(1, 1, D, H, W)``.
        data_range: Dynamic range of the data (default 1.0 for [0,1] images).
        n_levels: Number of scale levels (default 3).
        weights: Per-level weights. Default: ``[0.0448, 0.2856, 0.3001]``
            (normalised from Wang et al., 2003).

    Returns:
        MS-SSIM value in [0, 1]. Higher is better.
    """
    if weights is None:
        # Standard 3-level weights (normalised from the 5-level MS-SSIM paper)
        weights = [0.0448, 0.2856, 0.3001]
    assert len(weights) == n_levels

    # Normalise weights to sum to 1
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(0).unsqueeze(0)

    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    ssim_values: list[float] = []

    x_level = x.float()
    y_level = y.float()

    for level in range(n_levels):
        # Check minimum spatial size for SSIM (needs at least 11 per dim)
        min_dim = min(x_level.shape[2:])
        if min_dim < 11:
            # Volume too small for SSIM at this level, use last valid
            break

        ssim_val = compute_ssim_3d(x_level, y_level)
        ssim_values.append(ssim_val)

        # Downsample 2x for next level (except last)
        if level < n_levels - 1:
            x_level = F.avg_pool3d(x_level, kernel_size=2, stride=2)
            y_level = F.avg_pool3d(y_level, kernel_size=2, stride=2)

    # If not enough levels computed, pad with last value
    while len(ssim_values) < n_levels:
        ssim_values.append(ssim_values[-1])

    # Weighted geometric mean (MS-SSIM formula)
    ms_ssim = 1.0
    for w, s in zip(weights, ssim_values):
        ms_ssim *= max(s, 1e-10) ** w

    return ms_ssim
