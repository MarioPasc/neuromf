"""Tests for MMD metric."""

import torch

from neuromf.metrics.mmd import compute_mmd


class TestMMD:
    """Unit tests for MMD computation."""

    def test_identical_distributions_near_zero(self) -> None:
        """MMD between identical samples should be approximately 0."""
        torch.manual_seed(42)
        x = torch.randn(200, 4)
        mmd = compute_mmd(x, x.clone())
        assert abs(mmd) < 0.01, f"MMD of identical distributions: {mmd}"

    def test_different_distributions_positive(self) -> None:
        """MMD between different distributions should be positive."""
        torch.manual_seed(42)
        x = torch.randn(200, 4)
        y = torch.randn(200, 4) + 3.0  # shifted mean
        mmd = compute_mmd(x, y)
        assert mmd > 0.01, f"MMD of different distributions should be positive: {mmd}"

    def test_symmetric(self) -> None:
        """MMD(x, y) should approximately equal MMD(y, x)."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        y = torch.randn(100, 4) + 1.0
        mmd_xy = compute_mmd(x, y)
        mmd_yx = compute_mmd(y, x)
        assert abs(mmd_xy - mmd_yx) < 0.01, f"MMD not symmetric: {mmd_xy} vs {mmd_yx}"

    def test_custom_bandwidths(self) -> None:
        """Should work with user-specified bandwidths."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        y = torch.randn(100, 4) + 2.0
        mmd = compute_mmd(x, y, bandwidths=[0.5, 1.0, 2.0])
        assert mmd > 0.0
