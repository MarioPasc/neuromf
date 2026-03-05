"""Per-channel variance rescaling for generated latents.

Applies an affine transform so that the generated latent distribution matches
the training data distribution in mean and standard deviation per channel.
"""

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
