"""Tests for α-Flow curriculum scheduler and alpha-flow time sampling.

V2-T1: AlphaScheduler sigmoid correctness
V2-T2: alpha=0 → all r ≈ t
"""

from __future__ import annotations

import pytest
import torch

from neuromf.utils.alpha_scheduler import AlphaScheduler, AlphaSchedulerConfig
from neuromf.utils.time_sampler import sample_alpha_flow

# ======================================================================
# V2-T1: AlphaScheduler sigmoid correctness
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
class TestAlphaSchedulerSigmoid:
    """V2-T1: α(start) ≈ 0, α(end) ≈ η, monotonically increasing."""

    def test_alpha_zero_before_start(self) -> None:
        """α should be 0.0 before start_step."""
        sched = AlphaScheduler(AlphaSchedulerConfig(start_step=1000, end_step=5000, eta=1.0))
        assert sched.get_alpha(0) == 0.0
        assert sched.get_alpha(500) == 0.0
        assert sched.get_alpha(999) == 0.0

    def test_alpha_eta_after_end(self) -> None:
        """α should be η after end_step."""
        sched = AlphaScheduler(AlphaSchedulerConfig(start_step=1000, end_step=5000, eta=1.0))
        assert sched.get_alpha(5000) == 1.0
        assert sched.get_alpha(10000) == 1.0

    def test_alpha_eta_custom(self) -> None:
        """α should converge to custom η value."""
        sched = AlphaScheduler(AlphaSchedulerConfig(start_step=0, end_step=1000, eta=0.005))
        assert sched.get_alpha(1000) == pytest.approx(0.005)

    def test_alpha_monotonically_increasing(self) -> None:
        """α must be monotonically increasing between start and end."""
        sched = AlphaScheduler(
            AlphaSchedulerConfig(start_step=0, end_step=10000, gamma=25.0, eta=1.0)
        )
        prev = 0.0
        for step in range(0, 10001, 100):
            alpha = sched.get_alpha(step)
            assert alpha >= prev, f"α decreased at step {step}: {alpha} < {prev}"
            prev = alpha

    def test_sigmoid_midpoint_near_half_eta(self) -> None:
        """At midpoint, sigmoid should give ~η/2."""
        sched = AlphaScheduler(
            AlphaSchedulerConfig(start_step=0, end_step=10000, gamma=25.0, eta=1.0)
        )
        mid_alpha = sched.get_alpha(5000)
        assert mid_alpha == pytest.approx(0.5, abs=0.01)

    def test_sigmoid_sharp_transition(self) -> None:
        """With gamma=25, transition should be sharp (~5% window)."""
        sched = AlphaScheduler(
            AlphaSchedulerConfig(start_step=0, end_step=10000, gamma=25.0, eta=1.0)
        )
        # At 40% progress, should still be near 0
        alpha_40 = sched.get_alpha(4000)
        assert alpha_40 < 0.1

        # At 60% progress, should be near 1
        alpha_60 = sched.get_alpha(6000)
        assert alpha_60 > 0.9

    def test_linear_mode(self) -> None:
        """Linear mode: α = η * progress."""
        sched = AlphaScheduler(
            AlphaSchedulerConfig(start_step=0, end_step=1000, eta=1.0, mode="linear")
        )
        assert sched.get_alpha(0) == pytest.approx(0.0, abs=1e-6)
        assert sched.get_alpha(500) == pytest.approx(0.5, abs=1e-6)
        assert sched.get_alpha(1000) == pytest.approx(1.0, abs=1e-6)

    def test_eta_zero_always_returns_zero(self) -> None:
        """When η=0, α is always 0 regardless of step."""
        sched = AlphaScheduler(AlphaSchedulerConfig(start_step=0, end_step=1000, eta=0.0))
        for step in [0, 500, 1000, 5000]:
            assert sched.get_alpha(step) == 0.0


@pytest.mark.phase6
@pytest.mark.critical
class TestAlphaSchedulerConfig:
    """Config validation tests."""

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            AlphaSchedulerConfig(mode="invalid")

    def test_negative_eta_raises(self) -> None:
        with pytest.raises(ValueError, match="eta must be"):
            AlphaSchedulerConfig(eta=-0.1)

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end_step"):
            AlphaSchedulerConfig(start_step=1000, end_step=500)


# ======================================================================
# V2-T2: alpha=0 → all r ≈ t
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
class TestAlphaFlowSampling:
    """V2-T2: When α=0, all returned (t, r) pairs have r ≈ t."""

    def test_alpha_zero_all_r_equal_t(self) -> None:
        """With α=0, max |r - t| < 1e-6 for B=1000."""
        t, r = sample_alpha_flow(
            batch_size=1000,
            alpha=0.0,
            data_proportion=0.5,  # even MF samples should have r=t
        )
        max_diff = (t - r).abs().max().item()
        assert max_diff < 1e-6, f"max |r - t| = {max_diff} (expected < 1e-6)"

    def test_alpha_one_has_gap(self) -> None:
        """With α=1, MF samples should have r < t."""
        t, r = sample_alpha_flow(
            batch_size=1000,
            alpha=1.0,
            data_proportion=0.0,  # all MF
        )
        # At least some samples should have meaningful gap
        gaps = (t - r).abs()
        assert gaps.mean() > 0.1, f"Mean gap too small: {gaps.mean()}"

    def test_alpha_partial_scales_gap(self) -> None:
        """With α=0.5, gaps should be ~half of full MF."""
        torch.manual_seed(42)
        t_full, r_full = sample_alpha_flow(batch_size=10000, alpha=1.0, data_proportion=0.0)
        torch.manual_seed(42)
        t_half, r_half = sample_alpha_flow(batch_size=10000, alpha=0.5, data_proportion=0.0)
        gap_full = (t_full - r_full).mean().item()
        gap_half = (t_half - r_half).mean().item()
        ratio = gap_half / max(gap_full, 1e-10)
        assert 0.3 < ratio < 0.7, f"Gap ratio {ratio} not near 0.5"

    def test_output_shapes(self) -> None:
        """Output shapes should be (B,)."""
        t, r = sample_alpha_flow(batch_size=64, alpha=0.5)
        assert t.shape == (64,)
        assert r.shape == (64,)

    def test_t_geq_r(self) -> None:
        """t >= r for all samples."""
        t, r = sample_alpha_flow(batch_size=10000, alpha=1.0, data_proportion=0.0)
        assert (t >= r - 1e-7).all(), "Found samples with t < r"

    def test_values_in_range(self) -> None:
        """All values in [0, 1]."""
        t, r = sample_alpha_flow(batch_size=10000, alpha=1.0)
        assert (t >= 0).all() and (t <= 1).all()
        assert (r >= 0).all() and (r <= 1).all()

    def test_fm_proportion_respected(self) -> None:
        """First data_proportion fraction has r=t."""
        t, r = sample_alpha_flow(batch_size=100, alpha=1.0, data_proportion=0.3)
        n_fm = int(100 * 0.3)
        assert torch.allclose(t[:n_fm], r[:n_fm])
