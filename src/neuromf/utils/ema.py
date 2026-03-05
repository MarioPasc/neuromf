"""Exponential Moving Average (EMA) of model weights.

Maintains shadow copies of model parameters that are updated as an
exponential moving average during training. At inference time, the shadow
weights can be temporarily applied to the model.

Reference: Polyak & Juditsky (1992), widely used in diffusion models.
"""

import copy
import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


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


class MultiEMAModel:
    """Multiple EMA trackers with a single-EMA-compatible interface.

    Maintains several EMA copies at different decay rates. All copies are
    updated on every ``update()`` call. ``apply_shadow`` / ``restore`` delegate
    to the *active* EMA (selected by ``active_index``).

    Args:
        model: The model whose parameters to track.
        decays: List of decay rates (e.g. ``[0.999, 0.9995, 0.9999]``).
        active_index: Index into ``decays`` for the primary EMA used at
            inference time. Defaults to the last (highest decay).
    """

    def __init__(
        self,
        model: nn.Module,
        decays: list[float] | None = None,
        active_index: int = -1,
    ) -> None:
        if decays is None:
            decays = [0.999, 0.9995, 0.9999]
        self.decays = list(decays)
        self.active_index = active_index
        self.emas = [EMAModel(model, decay=d) for d in self.decays]
        logger.info(
            "MultiEMAModel: %d trackers, decays=%s, active=%d",
            len(self.decays),
            self.decays,
            self.active_index,
        )

    @property
    def shadow(self) -> dict[str, Tensor]:
        """Shadow parameters of the active EMA."""
        return self.emas[self.active_index].shadow

    @property
    def decay(self) -> float:
        """Decay rate of the active EMA."""
        return self.emas[self.active_index].decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update all EMA trackers with current model parameters.

        Args:
            model: Model with current parameters.
        """
        for ema in self.emas:
            ema.update(model)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with active EMA shadow.

        Args:
            model: Model to update in-place.
        """
        self.emas[self.active_index].apply_shadow(model)

    def restore(self, model: nn.Module) -> None:
        """Restore model parameters from backup (undo ``apply_shadow``).

        Args:
            model: Model to restore in-place.
        """
        self.emas[self.active_index].restore(model)

    def state_dict(self) -> dict:
        """Serialize all EMA states."""
        return {
            "decays": self.decays,
            "active_index": self.active_index,
            "emas": [ema.state_dict() for ema in self.emas],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EMA state, supporting both multi-EMA and legacy formats.

        Legacy format (single EMA): ``{"decay": float, "shadow": {...}}``.
        Multi-EMA format: ``{"decays": [...], "emas": [...]}``.

        For legacy checkpoints, the shadow is loaded into the EMA slot
        whose decay matches (or the active slot if no match).

        Args:
            state: Saved state dict.
        """
        if "emas" in state:
            # Multi-EMA format
            self.decays = state["decays"]
            self.active_index = state.get("active_index", self.active_index)
            for ema, ema_state in zip(self.emas, state["emas"]):
                ema.load_state_dict(ema_state)
        elif "decay" in state and "shadow" in state:
            # Legacy single-EMA format: load into matching decay slot
            legacy_decay = state["decay"]
            loaded = False
            for ema in self.emas:
                if abs(ema.decay - legacy_decay) < 1e-8:
                    ema.load_state_dict(state)
                    loaded = True
                    break
            if not loaded:
                # No exact match — load into active slot
                self.emas[self.active_index].load_state_dict(state)
            logger.info(
                "Loaded legacy single-EMA checkpoint (decay=%.6f) into MultiEMAModel",
                legacy_decay,
            )
