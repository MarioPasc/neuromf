"""Tests for logit-normal time sampling utilities."""

import torch

from neuromf.utils.time_sampler import sample_logit_normal, sample_t_and_r


class TestSampleLogitNormal:
    """Tests for sample_logit_normal."""

    def test_output_shape(self) -> None:
        t = sample_logit_normal(batch_size=128)
        assert t.shape == (128,)

    def test_values_in_range(self) -> None:
        t = sample_logit_normal(batch_size=10_000, t_min=0.001)
        assert (t >= 0.001).all()
        assert (t <= 1.0).all()

    def test_no_nans(self) -> None:
        t = sample_logit_normal(batch_size=10_000)
        assert torch.isfinite(t).all()

    def test_distribution_is_unimodal_near_sigmoid_mu(self) -> None:
        """Median should be near sigmoid(mu) for large samples."""
        mu, sigma = -0.4, 1.0
        t = sample_logit_normal(batch_size=50_000, mu=mu, sigma=sigma)
        expected_median = torch.sigmoid(torch.tensor(mu))
        actual_median = t.median()
        assert abs(actual_median - expected_median) < 0.05

    def test_t_min_clamp(self) -> None:
        t = sample_logit_normal(batch_size=10_000, t_min=0.1)
        assert (t >= 0.1).all()


class TestSampleTandR:
    """Tests for sample_t_and_r."""

    def test_output_shapes(self) -> None:
        t, r = sample_t_and_r(batch_size=128)
        assert t.shape == (128,)
        assert r.shape == (128,)

    def test_t_geq_r(self) -> None:
        t, r = sample_t_and_r(batch_size=10_000)
        assert (t >= r).all()

    def test_data_proportion_sets_r_equals_t(self) -> None:
        """First data_proportion fraction should have r == t."""
        bs = 100
        t, r = sample_t_and_r(batch_size=bs, data_proportion=0.5)
        data_size = int(bs * 0.5)
        assert torch.allclose(r[:data_size], t[:data_size])
        # Not all remaining elements should have r == t
        remaining_diff = (t[data_size:] - r[data_size:]).abs()
        assert remaining_diff.sum() > 0

    def test_data_proportion_zero(self) -> None:
        """With data_proportion=0, no elements forced to r=t."""
        t, r = sample_t_and_r(batch_size=1000, data_proportion=0.0)
        assert not torch.allclose(t, r)

    def test_data_proportion_one(self) -> None:
        """With data_proportion=1.0, all r == t."""
        t, r = sample_t_and_r(batch_size=100, data_proportion=1.0)
        assert torch.allclose(t, r)

    def test_values_finite(self) -> None:
        t, r = sample_t_and_r(batch_size=10_000)
        assert torch.isfinite(t).all()
        assert torch.isfinite(r).all()


import pytest
from scipy import stats


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T7_logit_normal_distribution() -> None:
    """P3-T7: Logit-normal samples match theoretical CDF (KS test p > 0.05).

    The logit-normal is defined as t = sigmoid(X) where X ~ N(mu, sigma^2).
    We verify via KS test that logit(t) follows the correct normal.
    """
    mu, sigma = -0.4, 1.0
    n = 10_000
    t = sample_logit_normal(batch_size=n, mu=mu, sigma=sigma, t_min=0.0)

    # Transform back: logit(t) should be N(mu, sigma^2)
    logit_t = torch.log(t / (1.0 - t))
    logit_np = logit_t.numpy()

    # KS test against N(mu, sigma)
    ks_stat, p_value = stats.kstest(logit_np, "norm", args=(mu, sigma))

    assert p_value > 0.05, (
        f"KS test failed: p={p_value:.4f} (stat={ks_stat:.4f}), "
        f"logit(t) may not follow N({mu}, {sigma}^2)"
    )
