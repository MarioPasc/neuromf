"""Maximum Mean Discrepancy (MMD) with RBF kernel.

Two-sample test between generated and real distributions using MMD with
multiple RBF kernel bandwidths. Reports the maximum MMD^2 across bandwidths
(conservative estimate).

Reference: Gretton et al. (2012), JMLR.
"""

import torch
from torch import Tensor


def _pairwise_sq_distances(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distances between x and y.

    Uses the identity ||a-b||^2 = ||a||^2 - 2*a.b + ||b||^2.

    Args:
        x: Tensor of shape (N, D).
        y: Tensor of shape (M, D).

    Returns:
        Distance matrix of shape (N, M).
    """
    x_sq = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    y_sq = (y * y).sum(dim=1, keepdim=True)  # (M, 1)
    xy = x @ y.t()  # (N, M)
    return (x_sq - 2 * xy + y_sq.t()).clamp(min=0.0)


def compute_mmd(
    x: Tensor,
    y: Tensor,
    bandwidths: list[float] | None = None,
) -> float:
    """Compute MMD^2 with RBF kernel using multiple bandwidths.

    Uses the unbiased estimator:
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Reports the maximum MMD^2 over all bandwidths (conservative).
    If bandwidths are not provided, uses the median heuristic plus
    a range of scales.

    Args:
        x: Real samples of shape (N, D).
        y: Generated samples of shape (M, D).
        bandwidths: List of RBF kernel bandwidths (sigma values).
            If None, uses median heuristic to set bandwidths.

    Returns:
        Maximum MMD^2 value across all bandwidths.
    """
    x = x.detach().float()
    y = y.detach().float()

    d_xx = _pairwise_sq_distances(x, x)
    d_yy = _pairwise_sq_distances(y, y)
    d_xy = _pairwise_sq_distances(x, y)

    if bandwidths is None:
        all_dists = torch.cat([d_xx.view(-1), d_yy.view(-1), d_xy.view(-1)])
        median_dist = all_dists.median().item()
        sigma_med = max(median_dist**0.5, 1e-6)
        bandwidths = [sigma_med * s for s in [0.2, 0.5, 1.0, 2.0, 5.0]]

    max_mmd2 = -float("inf")
    n = x.shape[0]
    m = y.shape[0]

    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * sigma * sigma)
        k_xx = torch.exp(-gamma * d_xx)
        k_yy = torch.exp(-gamma * d_yy)
        k_xy = torch.exp(-gamma * d_xy)

        # Unbiased estimator: zero out diagonal for xx and yy
        k_xx.fill_diagonal_(0.0)
        k_yy.fill_diagonal_(0.0)

        mmd2 = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.sum() / (n * m)
        max_mmd2 = max(max_mmd2, mmd2.item())

    return max_mmd2
