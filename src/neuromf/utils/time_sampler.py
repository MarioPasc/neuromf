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
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) pairs following MeanFlow-PyTorch convention.

    Both t and r are sampled from the same logit-normal distribution,
    then swapped so t >= r. For ``data_proportion`` fraction of the batch,
    r is set equal to t (JVP term vanishes, becoming standard FM loss).

    Args:
        batch_size: Number of samples.
        mu: Mean of the underlying normal.
        sigma: Std of the underlying normal.
        t_min: Minimum clamp value.
        data_proportion: Fraction of batch where r = t.
        device: Target device.

    Returns:
        Tuple of (t, r), each of shape (batch_size,).
    """
    t_raw = sample_logit_normal(batch_size, mu=mu, sigma=sigma, t_min=t_min, device=device)
    r_raw = sample_logit_normal(batch_size, mu=mu, sigma=sigma, t_min=t_min, device=device)

    # Enforce t >= r
    t = torch.maximum(t_raw, r_raw)
    r = torch.minimum(t_raw, r_raw)

    # For data_proportion of the batch, set r = t
    data_size = int(batch_size * data_proportion)
    mask = torch.arange(batch_size, device=device) < data_size
    r = torch.where(mask, t, r)

    return t, r
