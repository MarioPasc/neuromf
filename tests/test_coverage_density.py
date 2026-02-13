"""Tests for coverage and density metrics."""

import torch

from neuromf.metrics.coverage_density import compute_coverage, compute_density


class TestCoverage:
    """Unit tests for coverage metric."""

    def test_identical_near_one(self) -> None:
        """Coverage of identical distributions should be near 1.0."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        cov = compute_coverage(x, x.clone(), k=5)
        assert cov > 0.9, f"Coverage of identical distributions: {cov}"

    def test_disjoint_near_zero(self) -> None:
        """Coverage of disjoint distributions should be near 0.0."""
        torch.manual_seed(42)
        real = torch.randn(100, 4)
        gen = torch.randn(100, 4) + 100.0  # far away
        cov = compute_coverage(real, gen, k=5)
        assert cov < 0.1, f"Coverage of disjoint distributions: {cov}"


class TestDensity:
    """Unit tests for density metric."""

    def test_identical_near_one(self) -> None:
        """Density of identical distributions should be near 1.0."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        den = compute_density(x, x.clone(), k=5)
        assert 0.5 < den < 2.0, f"Density of identical distributions: {den}"

    def test_disjoint_near_zero(self) -> None:
        """Density of disjoint distributions should be near 0.0."""
        torch.manual_seed(42)
        real = torch.randn(100, 4)
        gen = torch.randn(100, 4) + 100.0
        den = compute_density(real, gen, k=5)
        assert den < 0.1, f"Density of disjoint distributions: {den}"
