"""Tests for per-channel Lp loss."""

import torch

from neuromf.losses.lp_loss import lp_loss


class TestLpLoss:
    """Tests for lp_loss function."""

    def test_l2_matches_mse_times_numel(self) -> None:
        """L2 loss (p=2) should equal sum of squared differences."""
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = lp_loss(pred, target, p=2.0, reduction="none")
        expected = ((pred - target) ** 2).sum(dim=1)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_l1_loss(self) -> None:
        """L1 loss (p=1) should equal sum of absolute differences."""
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = lp_loss(pred, target, p=1.0, reduction="none")
        expected = (pred - target).abs().sum(dim=1)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_zero_loss_on_identical_inputs(self) -> None:
        x = torch.randn(8, 4)
        loss = lp_loss(x, x, p=2.0)
        assert loss.item() == 0.0

    def test_reduction_mean(self) -> None:
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        per_sample = lp_loss(pred, target, p=2.0, reduction="none")
        mean_loss = lp_loss(pred, target, p=2.0, reduction="mean")
        assert torch.allclose(mean_loss, per_sample.mean(), atol=1e-6)

    def test_reduction_sum(self) -> None:
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        per_sample = lp_loss(pred, target, p=2.0, reduction="none")
        sum_loss = lp_loss(pred, target, p=2.0, reduction="sum")
        assert torch.allclose(sum_loss, per_sample.sum(), atol=1e-6)

    def test_channel_weights_3d(self) -> None:
        """Channel weights should scale per-channel contributions."""
        pred = torch.randn(4, 3, 8, 8, 8)
        target = torch.zeros_like(pred)
        w = torch.tensor([1.0, 0.0, 0.0])
        loss_weighted = lp_loss(pred, target, p=2.0, channel_weights=w, reduction="none")
        # Only channel 0 contributes
        expected = (pred[:, 0] ** 2).sum(dim=(1, 2, 3))
        assert torch.allclose(loss_weighted, expected, atol=1e-5)

    def test_fractional_p(self) -> None:
        """Should work with fractional p (e.g. p=1.5)."""
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = lp_loss(pred, target, p=1.5, reduction="none")
        expected = (pred - target).abs().pow(1.5).sum(dim=1)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_output_is_finite(self) -> None:
        pred = torch.randn(16, 4)
        target = torch.randn(16, 4)
        loss = lp_loss(pred, target, p=2.0)
        assert torch.isfinite(loss)

    def test_gradients_flow(self) -> None:
        pred = torch.randn(8, 4, requires_grad=True)
        target = torch.randn(8, 4)
        loss = lp_loss(pred, target, p=2.0)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
