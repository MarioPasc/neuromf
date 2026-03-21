"""Per-channel variance rescaling and stochastic perturbation for generated latents.

Provides two post-hoc corrections for 1-NFE MeanFlow generation:
- ``variance_rescale``: affine transform to match training data statistics.
- ``stochastic_perturb``: per-channel Gaussian noise injection to break mode collapse.
"""

import torch
from torch import Tensor


def variance_rescale(
    z_gen: Tensor,
    mu_data: Tensor,
    sigma_data: Tensor,
    mu_gen: Tensor | None = None,
    sigma_gen: Tensor | None = None,
) -> Tensor:
    """Per-channel affine rescaling to match data statistics.

    Args:
        z_gen: Generated latents of shape ``(B, C, ...)``.
        mu_data: Per-channel data mean, broadcastable to ``z_gen``.
        sigma_data: Per-channel data std, broadcastable to ``z_gen``.
        mu_gen: Per-channel generated mean. If None, computed from ``z_gen``.
        sigma_gen: Per-channel generated std. If None, computed from ``z_gen``.

    Returns:
        Rescaled tensor with same shape as ``z_gen``.
    """
    reduce_dims = tuple(range(2, z_gen.ndim))  # spatial dims only

    if mu_gen is None:
        mu_gen = z_gen.mean(dim=(0, *reduce_dims), keepdim=True)
    if sigma_gen is None:
        sigma_gen = z_gen.std(dim=(0, *reduce_dims), keepdim=True)

    # Avoid division by zero
    sigma_gen = sigma_gen.clamp(min=1e-8)

    return mu_data + (sigma_data / sigma_gen) * (z_gen - mu_gen)


def stochastic_perturb(
    z_0: Tensor,
    sigma_inject: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add per-channel Gaussian noise to break 1-NFE mode collapse.

    Injects noise scaled per-channel: ``z_corrected = z_0 + sigma_inject * eta``
    where ``eta ~ N(0, I)``. The injected variance approximates the missing
    conditional variance ``Var[z_0 | z_1]`` that the deterministic 1-NFE map
    collapses.

    Args:
        z_0: Generated latents of shape ``(B, C, ...)``.
        sigma_inject: Per-channel noise std of shape ``(C,)`` or broadcastable.
        generator: Optional torch Generator for reproducibility.

    Returns:
        Perturbed tensor with same shape as ``z_0``.
    """
    sigma = sigma_inject.view(1, -1, *([1] * (z_0.ndim - 2)))
    if generator is not None:
        eta = torch.randn(z_0.shape, generator=generator, device=z_0.device, dtype=z_0.dtype)
    else:
        eta = torch.randn_like(z_0)
    return z_0 + sigma * eta
