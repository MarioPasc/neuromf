"""Tests for per-channel Lp loss — Phase 3 verification.

P3-T6: Per-channel Lp correct for p in {1.0, 1.5, 2.0, 3.0}
"""

import numpy as np
import pytest
import torch

from neuromf.losses.lp_loss import lp_loss


def _numpy_lp_loss(
    pred: np.ndarray,
    target: np.ndarray,
    p: float,
    channel_weights: np.ndarray | None = None,
) -> float:
    """Reference numpy implementation of per-channel Lp loss."""
    diff = np.abs(pred - target) ** p

    if channel_weights is not None and pred.ndim >= 3:
        # Sum over spatial dims (axes 2, 3, 4 for 5D)
        spatial_axes = tuple(range(2, pred.ndim))
        per_channel = diff.sum(axis=spatial_axes)  # (B, C)
        # Weight by channel
        per_channel = per_channel * channel_weights[np.newaxis, :]
        per_sample = per_channel.sum(axis=1)  # (B,)
    else:
        # Sum over all non-batch dims
        non_batch_axes = tuple(range(1, pred.ndim))
        per_sample = diff.sum(axis=non_batch_axes)  # (B,)

    return per_sample.mean()


@pytest.mark.phase3
@pytest.mark.critical
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_P3_T6_perchannel_lp_loss(p: float) -> None:
    """P3-T6: Per-channel Lp loss matches numpy reference for various p."""
    torch.manual_seed(42)
    B, C, D, H, W = 2, 4, 8, 8, 8
    pred = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)

    # Without channel weights
    result = lp_loss(pred, target, p=p, reduction="mean")
    expected = _numpy_lp_loss(pred.numpy(), target.numpy(), p=p)
    assert abs(result.item() - expected) < 1e-3, (
        f"p={p} without weights: got {result.item():.6f}, expected {expected:.6f}"
    )


@pytest.mark.phase3
@pytest.mark.critical
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_P3_T6_perchannel_lp_loss_with_weights(p: float) -> None:
    """P3-T6: Per-channel Lp loss with channel weights matches numpy."""
    torch.manual_seed(42)
    B, C, D, H, W = 2, 4, 8, 8, 8
    pred = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    channel_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

    result = lp_loss(pred, target, p=p, channel_weights=channel_weights, reduction="mean")
    expected = _numpy_lp_loss(
        pred.numpy(), target.numpy(), p=p, channel_weights=channel_weights.numpy()
    )
    assert abs(result.item() - expected) < 1e-3, (
        f"p={p} with weights: got {result.item():.6f}, expected {expected:.6f}"
    )


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_lp_loss_reduction_none() -> None:
    """Per-sample loss with reduction='none' returns (B,) tensor."""
    B, C, D, H, W = 3, 4, 8, 8, 8
    pred = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)

    result = lp_loss(pred, target, p=2.0, reduction="none")
    assert result.shape == (B,), f"Expected shape ({B},), got {result.shape}"
    assert (result >= 0).all(), "Per-sample loss should be non-negative"


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_lp_loss_gradient_flows() -> None:
    """Gradient flows through lp_loss."""
    B, C, D, H, W = 2, 4, 8, 8, 8
    pred = torch.randn(B, C, D, H, W, requires_grad=True)
    target = torch.randn(B, C, D, H, W)

    loss = lp_loss(pred, target, p=2.0, reduction="mean")
    loss.backward()

    assert pred.grad is not None, "No gradient on pred"
    assert pred.grad.shape == pred.shape
