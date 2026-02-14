"""Tests for EMA model utility.

P3-T8: EMA updates correctly with UNet wrapper.
"""

import torch
import torch.nn as nn

from neuromf.utils.ema import EMAModel


class TestEMAModel:
    """Unit tests for EMAModel."""

    def _make_model(self) -> nn.Module:
        torch.manual_seed(42)
        return nn.Linear(4, 4)

    def test_init_matches_model(self) -> None:
        """Shadow parameters should match model parameters after init."""
        model = self._make_model()
        ema = EMAModel(model, decay=0.999)

        for name, param in model.named_parameters():
            assert name in ema.shadow
            assert torch.allclose(ema.shadow[name], param.data)

    def test_update_moves_toward_model(self) -> None:
        """After update, shadow should move toward current model params."""
        model = self._make_model()
        ema = EMAModel(model, decay=0.9)
        shadow_before = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        ema.update(model)

        for name, param in model.named_parameters():
            if name in ema.shadow:
                # Shadow should have moved toward new param
                dist_before = (shadow_before[name] - param.data).norm()
                dist_after = (ema.shadow[name] - param.data).norm()
                assert dist_after < dist_before, "Shadow did not move toward model"

    def test_apply_restore_roundtrip(self) -> None:
        """Apply shadow then restore should recover original params."""
        model = self._make_model()
        ema = EMAModel(model, decay=0.9)

        # Update EMA a few times with modified model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)

        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply_shadow(model)
        # Params should now be shadow (different from original)
        for name, param in model.named_parameters():
            assert not torch.allclose(param.data, original_params[name]), (
                "Shadow should differ from updated params"
            )

        ema.restore(model)
        # Params should be restored
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_params[name]), (
                "Restore did not recover original params"
            )

    def test_state_dict_roundtrip(self) -> None:
        """Save and load state dict should preserve shadow params."""
        model = self._make_model()
        ema = EMAModel(model, decay=0.995)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))
        ema.update(model)

        state = ema.state_dict()

        # Create new EMA and load
        ema2 = EMAModel(model, decay=0.5)  # different decay
        ema2.load_state_dict(state)

        assert ema2.decay == 0.995
        for name in ema.shadow:
            assert torch.allclose(ema.shadow[name], ema2.shadow[name])


import pytest

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T8_ema_with_unet_wrapper() -> None:
    """P3-T8: EMA updates correctly with MAISIUNetWrapper.

    After multiple updates with modified parameters, EMA shadow should
    differ from the initial values.
    """
    torch.manual_seed(42)
    config = MAISIUNetConfig()
    model = MAISIUNetWrapper(config)

    ema = EMAModel(model, decay=0.9)
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # Simulate a few training steps
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

    # Shadow should have moved away from initial values
    n_changed = 0
    for name in initial_shadow:
        if not torch.allclose(ema.shadow[name], initial_shadow[name], atol=1e-6):
            n_changed += 1

    assert n_changed > 0, "EMA shadow did not change after updates"
    assert n_changed == len(initial_shadow), (
        f"Only {n_changed}/{len(initial_shadow)} shadow params changed"
    )
