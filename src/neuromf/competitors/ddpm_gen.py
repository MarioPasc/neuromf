"""DDPM competitor generator — DDIM deterministic sampling.

Wraps the custom ``DDPMLightningModule`` (in ``experiments/DDPM/``) and
the vendored MOTFM UNet builder behind the :class:`BaseCompetitorGenerator`
interface.

Requires both ``src/external/MOTFM`` (for ``build_model``) and
``experiments/DDPM`` (for ``ddpm_module``) on ``sys.path`` — handled
automatically by :meth:`_load_model`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import Tensor

from neuromf.competitors.base import BaseCompetitorGenerator
from neuromf.competitors.registry import register_competitor

logger = logging.getLogger(__name__)


def _ensure_paths_on_sys(repo_root: Path | None = None) -> None:
    """Add MOTFM + DDPM + MOTFM-wrapper code to sys.path."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    for sub in ["src/external/MOTFM", "experiments/DDPM", "experiments/MOTFM"]:
        p = str(repo_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


@register_competitor("ddpm")
class DDPMCompetitorGenerator(BaseCompetitorGenerator):
    """Generate volumes using a trained DDPM model via DDIM sampling.

    Inherits the shared generation loop from
    :class:`BaseCompetitorGenerator` and implements DDPM-specific
    checkpoint loading and DDIM batch sampling.
    """

    model_name: str = "DDPM"

    def _load_model(self, config_path: Path, checkpoint_path: Path) -> None:
        _ensure_paths_on_sys()

        from utils.general_utils import load_config

        self.config = load_config(str(config_path))

        from ddpm_module import DDPMLightningModule

        logger.info("Loading DDPM checkpoint: %s", checkpoint_path)
        self._module = DDPMLightningModule(self.config)
        ckpt = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        self._module.load_state_dict(ckpt["state_dict"], strict=True)
        self._module.to(self.device)
        self._module.eval()

        self.metadata = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
        }
        del ckpt
        logger.info(
            "DDPM model loaded (epoch=%s, step=%s)",
            self.metadata.get("epoch"),
            self.metadata.get("global_step"),
        )

    def _generate_batch(self, x_init: Tensor, nfe: int) -> Tensor:
        return self._module.ddim_sample(
            x_init, num_steps=nfe, device=self.device
        )
