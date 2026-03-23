"""Tests for MultiEMA tracker.

V2-T5: Different decay rates produce different shadows
V2-T6: Memory footprint verification
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuromf.utils.ema import MultiEMAModel


def _tiny_model() -> nn.Module:
    """Small linear model for fast EMA tests."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )


# ======================================================================
# V2-T5: Different shadows after updates
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
class TestMultiEMADifferentShadows:
    """V2-T5: After 100 updates, shadow[0] != shadow[2]."""

    def test_different_decays_produce_different_shadows(self) -> None:
        """Shadows at different decay rates should diverge."""
        model = _tiny_model()
        multi_ema = MultiEMAModel(model, decays=[0.999, 0.9995, 0.9999])

        # Simulate 100 training updates with random weight changes
        for _ in range(100):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.1)
            multi_ema.update(model)

        # Compare shadow[0] (fastest decay) vs shadow[2] (slowest decay)
        for s0, s2 in zip(
            multi_ema.emas[0].shadow.values(),
            multi_ema.emas[2].shadow.values(),
        ):
            if not torch.allclose(s0, s2, atol=1e-6):
                return  # Found at least one difference — pass
        pytest.fail("All shadow params identical across decays after 100 updates")

    def test_faster_decay_tracks_model_more_closely(self) -> None:
        """Lower decay (0.999) should be closer to current model weights."""
        model = _tiny_model()
        multi_ema = MultiEMAModel(model, decays=[0.9, 0.9999])

        for _ in range(50):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
            multi_ema.update(model)

        # Compute L2 distance from model to each shadow
        dist_fast = sum(
            (s - p.data).pow(2).sum().item()
            for s, p in zip(multi_ema.emas[0].shadow.values(), model.parameters())
        )
        dist_slow = sum(
            (s - p.data).pow(2).sum().item()
            for s, p in zip(multi_ema.emas[1].shadow.values(), model.parameters())
        )
        assert dist_fast < dist_slow, (
            f"Fast decay (0.9) should track more closely: "
            f"dist_fast={dist_fast:.4f} vs dist_slow={dist_slow:.4f}"
        )


# ======================================================================
# V2-T6: Memory footprint
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
class TestMultiEMAMemory:
    """V2-T6: Memory footprint scales linearly with number of EMA copies."""

    def test_memory_scales_linearly(self) -> None:
        """3 EMA copies should use ~3x the memory of model params."""
        model = _tiny_model()
        n_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4  # float32

        multi_ema = MultiEMAModel(model, decays=[0.999, 0.9995, 0.9999])

        # Count total shadow elements
        total_shadow_elements = sum(
            sum(s.numel() for s in ema.shadow.values()) for ema in multi_ema.emas
        )

        # Should be 3x model params
        assert total_shadow_elements == 3 * n_params, (
            f"Expected {3 * n_params} shadow elements, got {total_shadow_elements}"
        )

        # Memory estimate for 178M model: 3 × 178M × 4 bytes = ~2.1 GB
        # Our tiny model: 3 × n_params × 4 bytes
        total_bytes = total_shadow_elements * bytes_per_param
        total_gb = total_bytes / (1024**3)
        # For the 178M param model, ensure < 3 GB
        projected_gb = 3 * 178_000_000 * bytes_per_param / (1024**3)
        assert projected_gb < 3.0, f"Projected EMA memory {projected_gb:.1f} GB > 3 GB limit"


# ======================================================================
# Additional: apply/restore correctness
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
class TestMultiEMAApplyRestore:
    """Verify apply_shadow and restore work correctly."""

    def test_apply_and_restore(self) -> None:
        """Applying shadow changes params, restoring reverts them."""
        model = _tiny_model()
        multi_ema = MultiEMAModel(model, decays=[0.999, 0.9999], active_index=1)

        # Get original params
        originals = [p.data.clone() for p in model.parameters()]

        # Update EMA a few times with changed model
        for _ in range(10):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.1)
            multi_ema.update(model)

        # Save current params before apply
        before_apply = [p.data.clone() for p in model.parameters()]

        # Apply shadow
        multi_ema.apply_shadow(model)

        # Model params should now differ from before_apply
        any_diff = any(
            not torch.allclose(p.data, ba) for p, ba in zip(model.parameters(), before_apply)
        )
        assert any_diff, "apply_shadow should change model params"

        # Restore
        multi_ema.restore(model)

        # Model params should match before_apply
        for p, ba in zip(model.parameters(), before_apply):
            assert torch.allclose(p.data, ba), "restore should revert params"

    def test_state_dict_roundtrip(self) -> None:
        """state_dict save/load roundtrip preserves shadows."""
        model = _tiny_model()
        multi_ema = MultiEMAModel(model, decays=[0.999, 0.9999])

        for _ in range(10):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.1)
            multi_ema.update(model)

        state = multi_ema.state_dict()

        # Create fresh EMA and load
        model2 = _tiny_model()
        multi_ema2 = MultiEMAModel(model2, decays=[0.999, 0.9999])
        multi_ema2.load_state_dict(state)

        # Compare shadows
        for ema1, ema2 in zip(multi_ema.emas, multi_ema2.emas):
            for (k1, s1), (k2, s2) in zip(ema1.shadow.items(), ema2.shadow.items()):
                assert torch.allclose(s1, s2), f"Shadow mismatch at {k1}"
