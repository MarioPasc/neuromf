"""Sliced Wasserstein Distance (SWD) for fast distribution comparison.

Computes the SWD by projecting high-dimensional samples onto random 1D
directions and comparing the sorted projections. Runs in ~2s on CPU for
typical latent dimensions (N=200, D=442368).
"""

import torch
from torch import Tensor


def compute_swd(
    x: Tensor,
    y: Tensor,
    n_projections: int = 128,
    seed: int | None = None,
) -> float:
    """Compute Sliced Wasserstein Distance between two sample sets.

    Projects both sets onto ``n_projections`` random unit directions,
    computes the 1D Wasserstein-1 distance for each projection, and
    returns the average.

    Args:
        x: Real samples of shape ``(N, D)``.
        y: Generated samples of shape ``(M, D)``.
        n_projections: Number of random projections.
        seed: Random seed for reproducibility.

    Returns:
        Mean SWD across all projections.
    """
    x = x.detach().float()
    y = y.detach().float()

    assert x.ndim == 2 and y.ndim == 2, "Expected 2D tensors (N, D)"
    assert x.shape[1] == y.shape[1], "Feature dimensions must match"

    D = x.shape[1]

    # Random unit directions on the D-sphere
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    directions = torch.randn(n_projections, D, generator=gen, device=x.device)
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Project both sets: (N, n_proj) and (M, n_proj)
    proj_x = x @ directions.T
    proj_y = y @ directions.T

    # Subsample larger set to match smaller for equal-length 1D Wasserstein
    n, m = proj_x.shape[0], proj_y.shape[0]
    if n != m:
        min_size = min(n, m)
        perm_gen = torch.Generator(device=x.device)
        if seed is not None:
            perm_gen.manual_seed(seed + 1)
        if n > min_size:
            idx = torch.randperm(n, generator=perm_gen, device=x.device)[:min_size]
            proj_x = proj_x[idx]
        else:
            idx = torch.randperm(m, generator=perm_gen, device=x.device)[:min_size]
            proj_y = proj_y[idx]

    # Sort and compute mean absolute difference per projection
    proj_x_sorted, _ = proj_x.sort(dim=0)
    proj_y_sorted, _ = proj_y.sort(dim=0)

    # W1 = mean(|sorted_x - sorted_y|) per projection, then average
    swd = (proj_x_sorted - proj_y_sorted).abs().mean().item()

    return swd
