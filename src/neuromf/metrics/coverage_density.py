"""Coverage and density metrics using k-nearest neighbours.

Coverage measures mode dropping: fraction of real samples with at least one
generated neighbour within its k-NN ball radius.

Density measures hallucination: average number of generated samples falling
within real k-NN balls, normalised by k.

Reference: Naeem et al. (2020), "Reliable Fidelity and Diversity Metrics
for Generative Models", ICML.
"""

import torch
from torch import Tensor


def _kth_nearest_distance(x: Tensor, k: int) -> Tensor:
    """Compute the distance to the k-th nearest neighbour for each point in x.

    Args:
        x: Tensor of shape (N, D).
        k: Number of nearest neighbours.

    Returns:
        Tensor of shape (N,) with k-th NN distances.
    """
    dists = torch.cdist(x, x)  # (N, N)
    dists.fill_diagonal_(float("inf"))
    kth_dists, _ = dists.topk(k, dim=1, largest=False)
    return kth_dists[:, -1]  # (N,) — distance to k-th nearest


def compute_coverage(
    real: Tensor,
    gen: Tensor,
    k: int = 5,
) -> float:
    """Compute coverage: fraction of real samples with a gen neighbour in k-NN ball.

    Args:
        real: Real samples of shape (N, D).
        gen: Generated samples of shape (M, D).
        k: Number of nearest neighbours for ball radius.

    Returns:
        Coverage score in [0, 1]. Higher is better (less mode dropping).
    """
    real = real.detach().float()
    gen = gen.detach().float()

    radii = _kth_nearest_distance(real, k)  # (N,)
    dists_real_gen = torch.cdist(real, gen)  # (N, M)
    min_dist_to_gen = dists_real_gen.min(dim=1).values  # (N,)
    covered = (min_dist_to_gen <= radii).float()

    return covered.mean().item()


def compute_density(
    real: Tensor,
    gen: Tensor,
    k: int = 5,
) -> float:
    """Compute density: average gen samples within real k-NN balls, normalised by k.

    Args:
        real: Real samples of shape (N, D).
        gen: Generated samples of shape (M, D).
        k: Number of nearest neighbours for ball radius.

    Returns:
        Density score. Values around 1.0 indicate good match;
        >1 indicates over-concentration, <1 indicates under-coverage.
    """
    real = real.detach().float()
    gen = gen.detach().float()

    radii = _kth_nearest_distance(real, k)  # (N,)
    dists_gen_real = torch.cdist(gen, real)  # (M, N)

    # For each gen sample, count how many real k-NN balls it falls into
    in_ball = (dists_gen_real <= radii.unsqueeze(0)).float()  # (M, N)
    count_per_gen = in_ball.sum(dim=1)  # (M,)

    return (count_per_gen.sum() / (k * gen.shape[0])).item()
