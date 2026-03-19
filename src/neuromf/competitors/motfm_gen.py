"""MOTFM competitor generator — ODE-based flow matching.

Wraps the vendored MOTFM code (``src/external/MOTFM/``) behind the
:class:`BaseCompetitorGenerator` interface.

Requires the MOTFM vendored repo on ``sys.path`` — handled automatically
by :meth:`_load_model`.
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


def _ensure_motfm_on_path(repo_root: Path | None = None) -> None:
    """Add MOTFM vendored code to sys.path if not already present."""
    if repo_root is None:
        # Walk up from src/neuromf/competitors/ to project root
        repo_root = Path(__file__).resolve().parents[3]
    motfm_dir = str(repo_root / "src" / "external" / "MOTFM")
    if motfm_dir not in sys.path:
        sys.path.insert(0, motfm_dir)


def _fix_solver_time_points(solver_config: dict) -> dict:
    """Ensure at least 2 time grid points for a valid integration interval.

    ``torch.linspace(0, 1, 1)`` → ``[0.0]`` (no interval), so the ODE
    solver returns the initial noise unchanged.  Clamping to ≥ 2 fixes this.

    Args:
        solver_config: Solver config dict (mutated in place).

    Returns:
        The same dict with ``time_points`` ≥ 2.
    """
    if solver_config.get("time_points", 0) < 2:
        solver_config["time_points"] = 2
    return solver_config


@register_competitor("motfm")
class MOTFMCompetitorGenerator(BaseCompetitorGenerator):
    """Generate volumes using a trained MOTFM model via ODE integration.

    Inherits the shared generation loop from
    :class:`BaseCompetitorGenerator` and implements MOTFM-specific
    checkpoint loading and batch sampling.
    """

    model_name: str = "MOTFM"

    def _load_model(self, config_path: Path, checkpoint_path: Path) -> None:
        _ensure_motfm_on_path()

        from inferer import build_solver_config, load_model_from_checkpoint
        from utils.general_utils import load_config

        self.config = load_config(str(config_path))

        logger.info("Loading MOTFM checkpoint: %s", checkpoint_path)
        self._model, self.metadata = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config=self.config,
            device=self.device,
        )
        logger.info(
            "MOTFM model loaded (epoch=%s, step=%s)",
            self.metadata.get("epoch"),
            self.metadata.get("global_step"),
        )

        self._build_solver_config = build_solver_config

    def _make_solver_config(self, nfe: int) -> dict:
        """Build and fix solver config for a given NFE."""
        cfg = self._build_solver_config(self.config, num_inference_steps=nfe)
        return _fix_solver_time_points(cfg)

    def _generate_batch(self, x_init: Tensor, nfe: int) -> Tensor:
        from utils.utils_fm import sample_with_solver

        solver_config = self._make_solver_config(nfe)

        sol = sample_with_solver(
            model=self._model,
            x_init=x_init,
            solver_config=solver_config,
            cond=None,
            masks=None,
        )

        # sol may be [time_points, B, C, D, H, W] (trajectory) — take final
        expected_ndim = x_init.dim() + 1  # batch + trajectory dim
        if sol.dim() == expected_ndim:
            return sol[-1]
        return sol
