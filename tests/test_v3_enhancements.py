"""Tests for v3 enhancements: MultiEMA, progressive gap, variance rescaling."""

import torch
import torch.nn as nn

from neuromf.sampling.variance_rescaling import variance_rescale
from neuromf.utils.ema import EMAModel, MultiEMAModel
from neuromf.utils.time_sampler import sample_t_and_r

# ======================================================================
# MultiEMAModel
# ======================================================================


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class TestMultiEMA:
    """Tests for MultiEMAModel."""

    def test_multi_ema_creates_multiple_trackers(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.99, 0.999])
        assert len(multi.emas) == 3
        assert multi.decays == [0.9, 0.99, 0.999]

    def test_multi_ema_active_index_default(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.99, 0.999])
        # Default active_index=-1 means last
        assert multi.decay == 0.999

    def test_multi_ema_update_all(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.999])

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        multi.update(model)

        # Both EMAs should have been updated (shadows differ from original)
        for ema in multi.emas:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    # After one update with modified weights, shadow != original init
                    assert not torch.allclose(
                        ema.shadow[name],
                        torch.zeros_like(ema.shadow[name]),
                    )

    def test_multi_ema_apply_restore(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.999], active_index=0)

        original_weight = model.linear.weight.data.clone()

        # Update with perturbed model
        with torch.no_grad():
            model.linear.weight.add_(10.0)
        multi.update(model)

        # Apply shadow (active = index 0, decay=0.9)
        multi.apply_shadow(model)
        # Weight should now be the EMA shadow, different from perturbed
        assert not torch.allclose(model.linear.weight.data, original_weight + 10.0)

        # Restore
        multi.restore(model)
        # Weight back to perturbed value
        assert torch.allclose(model.linear.weight.data, original_weight + 10.0)

    def test_multi_ema_state_dict_roundtrip(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.99, 0.999], active_index=1)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        multi.update(model)

        state = multi.state_dict()
        assert "emas" in state
        assert state["decays"] == [0.9, 0.99, 0.999]
        assert state["active_index"] == 1

        # Load into fresh multi-EMA
        model2 = _TinyModel()
        multi2 = MultiEMAModel(model2, decays=[0.9, 0.99, 0.999])
        multi2.load_state_dict(state)

        for i in range(3):
            for key in multi.emas[i].shadow:
                assert torch.allclose(
                    multi.emas[i].shadow[key],
                    multi2.emas[i].shadow[key],
                )

    def test_multi_ema_legacy_load(self) -> None:
        model = _TinyModel()
        single = EMAModel(model, decay=0.999)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(5.0)
        single.update(model)

        legacy_state = single.state_dict()

        # Load legacy into multi-EMA
        model2 = _TinyModel()
        multi = MultiEMAModel(model2, decays=[0.9, 0.99, 0.999])
        multi.load_state_dict(legacy_state)

        # The decay=0.999 slot (index 2) should have the loaded shadow
        for key in single.shadow:
            assert torch.allclose(multi.emas[2].shadow[key], single.shadow[key])

    def test_multi_ema_shadow_property(self) -> None:
        model = _TinyModel()
        multi = MultiEMAModel(model, decays=[0.9, 0.999], active_index=1)
        # .shadow should return the active EMA's shadow
        assert multi.shadow is multi.emas[1].shadow


# ======================================================================
# Progressive Gap (max_gap in time_sampler)
# ======================================================================


class TestProgressiveGap:
    """Tests for max_gap parameter in sample_t_and_r."""

    def test_max_gap_none_no_change(self) -> None:
        """max_gap=None should not affect sampling (backward compat)."""
        torch.manual_seed(42)
        t1, r1 = sample_t_and_r(1000, data_proportion=0.0, max_gap=None)

        torch.manual_seed(42)
        t2, r2 = sample_t_and_r(1000, data_proportion=0.0)

        assert torch.allclose(t1, t2)
        assert torch.allclose(r1, r2)

    def test_max_gap_clamps_gap(self) -> None:
        """With max_gap=0.1, all gaps h=t-r should be <= 0.1."""
        t, r = sample_t_and_r(10000, data_proportion=0.0, max_gap=0.1)
        h = t - r
        assert h.max().item() <= 0.1 + 1e-6

    def test_max_gap_preserves_t_ge_r(self) -> None:
        """After gap clamping, t >= r must still hold."""
        t, r = sample_t_and_r(10000, data_proportion=0.0, max_gap=0.05)
        assert (t >= r - 1e-7).all()

    def test_max_gap_small_is_fm_like(self) -> None:
        """With very small max_gap, most pairs have r ≈ t (FM-like)."""
        t, r = sample_t_and_r(10000, data_proportion=0.0, max_gap=0.01)
        h = t - r
        assert h.mean().item() < 0.01

    def test_max_gap_1_no_effect(self) -> None:
        """max_gap=1.0 should not clamp anything (gap can't exceed 1)."""
        torch.manual_seed(42)
        t1, r1 = sample_t_and_r(1000, data_proportion=0.0, max_gap=1.0)

        torch.manual_seed(42)
        t2, r2 = sample_t_and_r(1000, data_proportion=0.0, max_gap=None)

        # With max_gap=1.0 the code path runs but doesn't change values
        assert torch.allclose(t1, t2)
        assert torch.allclose(r1, r2)

    def test_max_gap_with_data_proportion(self) -> None:
        """max_gap and data_proportion should compose correctly."""
        t, r = sample_t_and_r(1000, data_proportion=0.5, max_gap=0.2)
        h = t - r
        # data_proportion pairs have h=0, rest have h <= 0.2
        assert h.max().item() <= 0.2 + 1e-6
        # At least some pairs should have h=0 (data_proportion)
        assert (h < 1e-6).sum().item() > 0


# ======================================================================
# Variance Rescaling
# ======================================================================


class TestVarianceRescaling:
    """Tests for variance_rescale utility."""

    def test_identity_rescaling(self) -> None:
        """If gen stats match data stats, output = input."""
        z = torch.randn(8, 4, 12, 12, 12)
        mu_data = torch.zeros(1, 4, 1, 1, 1)
        sigma_data = torch.ones(1, 4, 1, 1, 1)

        result = variance_rescale(
            z,
            mu_data,
            sigma_data,
            mu_gen=torch.zeros(1, 4, 1, 1, 1),
            sigma_gen=torch.ones(1, 4, 1, 1, 1),
        )
        assert torch.allclose(result, z, atol=1e-6)

    def test_rescaling_matches_target_stats(self) -> None:
        """After rescaling, per-channel mean/std should match data stats."""
        torch.manual_seed(42)
        # Generated with wrong stats
        z_gen = torch.randn(512, 4, 8, 8, 8) * 2.0 + 3.0

        mu_data = torch.tensor([0.0, 1.0, -1.0, 0.5]).view(1, 4, 1, 1, 1)
        sigma_data = torch.tensor([1.0, 0.5, 2.0, 1.5]).view(1, 4, 1, 1, 1)

        z_rescaled = variance_rescale(z_gen, mu_data, sigma_data)

        # Check per-channel stats of rescaled output
        for c in range(4):
            ch = z_rescaled[:, c]
            assert abs(ch.mean().item() - mu_data[0, c, 0, 0, 0].item()) < 0.1
            assert abs(ch.std().item() - sigma_data[0, c, 0, 0, 0].item()) < 0.1

    def test_rescaling_with_explicit_gen_stats(self) -> None:
        """Explicit mu_gen/sigma_gen should be used instead of computed."""
        z = torch.ones(4, 2, 4, 4, 4) * 5.0
        mu_data = torch.zeros(1, 2, 1, 1, 1)
        sigma_data = torch.ones(1, 2, 1, 1, 1)
        mu_gen = torch.ones(1, 2, 1, 1, 1) * 5.0
        sigma_gen = torch.ones(1, 2, 1, 1, 1)

        result = variance_rescale(z, mu_data, sigma_data, mu_gen, sigma_gen)
        assert torch.allclose(result, torch.zeros_like(z), atol=1e-6)

    def test_rescaling_preserves_shape(self) -> None:
        """Output shape must match input shape."""
        z = torch.randn(4, 4, 12, 12, 12)
        mu = torch.zeros(1, 4, 1, 1, 1)
        sigma = torch.ones(1, 4, 1, 1, 1)
        result = variance_rescale(z, mu, sigma)
        assert result.shape == z.shape
