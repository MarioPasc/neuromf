"""Tests for EMA model utility."""

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
