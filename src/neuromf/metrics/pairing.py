"""Nearest-neighbour pairing for per-volume metric computation.

For unconditional generation, there is no natural real-generated pairing.
This module implements NN pairing in feature space (e.g. Med3D 2048-d)
so that per-volume metrics (MS-SSIM, PSNR) can be computed on matched pairs.

Reference: MOTFM evaluation protocol (Yazdani et al., 2025).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_nn_pairs(
    real_features: Tensor,
    gen_features: Tensor,
) -> Tensor:
    """Find nearest real neighbour for each generated sample.

    Args:
        real_features: Real feature vectors, shape ``(N_real, D)``.
        gen_features: Generated feature vectors, shape ``(N_gen, D)``.

    Returns:
        Index tensor of shape ``(N_gen,)`` where entry ``j`` is the index
        of the nearest real sample for generated sample ``j``.
    """
    real_features = real_features.float()
    gen_features = gen_features.float()

    # Pairwise distances: (N_gen, N_real)
    dists = torch.cdist(gen_features, real_features)

    # Nearest real neighbour for each generated sample
    nn_indices = dists.argmin(dim=1)

    return nn_indices
