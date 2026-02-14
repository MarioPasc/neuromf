"""Exponential Moving Average (EMA) of model weights.

Maintains shadow copies of model parameters that are updated as an
exponential moving average during training. At inference time, the shadow
weights can be temporarily applied to the model.

Reference: Polyak & Juditsky (1992), widely used in diffusion models.
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor


class EMAModel:
    """Exponential moving average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. Higher = slower update. Typical: 0.999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _sync_devices(self, model: nn.Module) -> None:
        """Move shadow tensors to match model device if needed.

        Called lazily on first ``update()`` after Lightning moves the model
        to GPU, or after ``load_state_dict()`` loads CPU tensors.

        Args:
            model: Model whose device to match.
        """
        # Determine target device from first parameter
        target_device = next(model.parameters()).device
        for name in list(self.shadow.keys()):
            if self.shadow[name].device != target_device:
                self.shadow[name] = self.shadow[name].to(target_device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters.

        Args:
            model: Model with current parameters.
        """
        self._sync_devices(model)
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with shadow (EMA) parameters.

        Call ``restore()`` afterwards to revert.

        Args:
            model: Model to update in-place.
        """
        self._sync_devices(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore model parameters from backup (undo ``apply_shadow``).

        Args:
            model: Model to restore in-place.
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Return shadow parameters for serialisation."""
        return {"decay": self.decay, "shadow": copy.deepcopy(self.shadow)}

    def load_state_dict(self, state: dict) -> None:
        """Load shadow parameters from a state dict.

        Shadows are stored on CPU by ``torch.save`` default. Device sync
        happens lazily on the next ``update()`` or ``apply_shadow()`` call.

        Args:
            state: Dict with "decay" and "shadow" keys.
        """
        self.decay = state["decay"]
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}
