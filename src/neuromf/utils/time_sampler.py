"""Logit-normal time sampling for MeanFlow training.

Implements the (t, r) pair sampling from the MeanFlow-PyTorch reference:
- Both t and r sampled independently from logit-normal distribution
- Enforce t >= r via torch.maximum/minimum
- data_proportion fraction of batch has r = t (reduces to standard FM loss)
"""

import torch


def sample_logit_normal(
    batch_size: int,
    mu: float = -0.4,
    sigma: float = 1.0,
    t_min: float = 0.001,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample times from logit-normal distribution: t = sigmoid(N(mu, sigma^2)).

    Args:
        batch_size: Number of samples.
        mu: Mean of the underlying normal distribution.
        sigma: Std of the underlying normal distribution.
        t_min: Minimum clamp value to avoid t=0 singularity.
        device: Target device.

    Returns:
        Tensor of shape (batch_size,) with values in [t_min, 1).
    """
    normal_samples = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(normal_samples)
    t = t.clamp(min=t_min)
    return t


def sample_t_and_r(
    batch_size: int,
    mu: float = -0.4,
    sigma: float = 1.0,
    t_min: float = 0.001,
    data_proportion: float = 0.5,
    boundary_fraction: float = 0.0,
    boundary_delta: float = 0.05,
    max_gap: float | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) pairs following MeanFlow-PyTorch convention.

    Both t and r are sampled from the same logit-normal distribution,
    then swapped so t >= r. For ``data_proportion`` fraction of the batch,
    r is set equal to t (JVP term vanishes, becoming standard FM loss).

    When ``boundary_fraction > 0``, that fraction of the batch is sampled
    from the 1-NFE boundary region: t ~ Uniform([1-delta, 1]), r ~ Uniform([t_min, delta]).
    This injects training signal at the (t=1, r=0) operating point that the
    logit-normal distribution assigns negligible mass to.

    Args:
        batch_size: Number of samples.
        mu: Mean of the underlying normal.
        sigma: Std of the underlying normal.
        t_min: Minimum clamp value.
        data_proportion: Fraction of standard batch where r = t.
        boundary_fraction: Fraction of batch sampled from boundary region.
        boundary_delta: Width of the boundary region around (t=1, r=0).
        max_gap: Maximum allowed gap h = t - r. If None, no clamping.
            Used by progressive gap curriculum to start FM-heavy and
            gradually widen to full 1-NFE regime.
        device: Target device.

    Returns:
        Tuple of (t, r), each of shape (batch_size,).
    """
    n_boundary = int(batch_size * boundary_fraction)
    n_standard = batch_size - n_boundary

    # Standard logit-normal sampling
    t_raw = sample_logit_normal(n_standard, mu=mu, sigma=sigma, t_min=t_min, device=device)
    r_raw = sample_logit_normal(n_standard, mu=mu, sigma=sigma, t_min=t_min, device=device)

    # Enforce t >= r
    t_std = torch.maximum(t_raw, r_raw)
    r_std = torch.minimum(t_raw, r_raw)

    # Clamp gap h = t - r to max_gap (progressive curriculum)
    if max_gap is not None and max_gap < 1.0:
        h = t_std - r_std
        excess = (h - max_gap).clamp(min=0.0)
        r_std = r_std + excess  # shrink gap by pushing r toward t

    # For data_proportion of the standard batch, set r = t
    data_size = int(n_standard * data_proportion)
    mask = torch.arange(n_standard, device=device) < data_size
    r_std = torch.where(mask, t_std, r_std)

    if n_boundary > 0:
        # Boundary sampling: (t, r) near (1, 0) for 1-NFE signal
        t_bnd = torch.empty(n_boundary, device=device).uniform_(1.0 - boundary_delta, 1.0)
        r_bnd = torch.empty(n_boundary, device=device).uniform_(t_min, boundary_delta)

        t = torch.cat([t_std, t_bnd])
        r = torch.cat([r_std, r_bnd])

        # Shuffle to avoid systematic ordering
        perm = torch.randperm(batch_size, device=device)
        return t[perm], r[perm]

    return t_std, r_std


def sample_alpha_flow(
    batch_size: int,
    alpha: float,
    data_proportion: float = 0.5,
    mu: float = -0.4,
    sigma: float = 1.0,
    t_min: float = 0.001,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) pairs using the α-Flow curriculum.

    When α=0, all samples have r=t (pure FM).
    When α>0, MF samples have gap h = α * h_sampled, progressively
    widening the consistency constraint.

    Reference: Zhang et al., "AlphaFlow," arXiv:2510.20771, Alg. 2.

    Args:
        batch_size: Number of samples.
        alpha: Current curriculum parameter in [0, η].
        data_proportion: Fraction of batch that are FM samples (r=t).
        mu: Mean for logit-normal t distribution.
        sigma: Std for logit-normal t distribution.
        t_min: Minimum time value.
        device: Target device.

    Returns:
        Tuple (t, r) of shape (B,) each, with t >= r.
    """
    # Sample t from logit-normal
    t = sample_logit_normal(batch_size, mu=mu, sigma=sigma, t_min=t_min, device=device)

    # Determine FM vs MF split
    n_fm = int(batch_size * data_proportion)
    n_mf = batch_size - n_fm

    r = t.clone()  # default: r = t (pure FM)

    # MF samples: scale gap by α
    if n_mf > 0 and alpha > 0:
        # Sample r_base uniformly in [0, t] for MF samples
        r_base = torch.rand(n_mf, device=device) * t[n_fm:]
        # Scale gap: r = t - α * (t - r_base)
        gap = t[n_fm:] - r_base
        r[n_fm:] = t[n_fm:] - alpha * gap

    # Clamp for numerical safety
    r = r.clamp(min=0.0, max=1.0)

    return t, r
