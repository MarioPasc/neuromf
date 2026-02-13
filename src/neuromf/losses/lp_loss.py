"""Per-channel Lp loss for MeanFlow training.

Extends SLIM-Diff per-channel loss to latent space. Supports arbitrary p-norms,
optional per-channel weighting, and multiple reduction modes.
"""

import torch


def lp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
    channel_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-channel Lp loss.

    Computes ``sum_over_spatial(|pred - target|^p)`` per sample, optionally
    weighted by channel, then reduces over the batch.

    Args:
        pred: Predicted tensor of shape ``(B, C, ...)`` or ``(B, D)``.
        target: Target tensor, same shape as pred.
        p: Norm exponent (e.g. 1.0 for L1, 2.0 for L2).
        channel_weights: Optional weights of shape ``(C,)`` to multiply per-channel
            losses before summing over channels. If None, channels are unweighted.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss if reduction is "mean" or "sum", per-sample loss of shape
        ``(B,)`` if reduction is "none".
    """
    diff = (pred - target).abs()
    if p != 1.0:
        diff = diff.pow(p)

    if pred.ndim >= 3 and channel_weights is not None:
        # pred is (B, C, ...) — sum over spatial dims first, weight by channel
        spatial_dims = list(range(2, pred.ndim))
        per_channel = diff.sum(dim=spatial_dims)  # (B, C)
        # Broadcast channel_weights to (1, C)
        per_channel = per_channel * channel_weights.unsqueeze(0)
        per_sample = per_channel.sum(dim=1)  # (B,)
    else:
        # Flat vectors (B, D) or no channel weights: sum all non-batch dims
        non_batch_dims = list(range(1, pred.ndim))
        per_sample = diff.sum(dim=non_batch_dims)  # (B,)

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    elif reduction == "none":
        return per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
