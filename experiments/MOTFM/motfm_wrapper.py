"""MOTFM inference wrapper — load any MOTFM checkpoint and generate volumes.

Works with both our FOMO-60K-trained checkpoints and (future) author-provided ones.
Generates from pure noise — no dataloader required.

Requires MOTFM on sys.path:
    export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${PYTHONPATH}"
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _ensure_motfm_on_path(repo_root: Path | None = None) -> None:
    """Add MOTFM external code to sys.path if not already present."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    motfm_dir = str(repo_root / "src" / "external" / "MOTFM")
    if motfm_dir not in sys.path:
        sys.path.insert(0, motfm_dir)


def _fix_solver_time_points(solver_config: dict) -> dict:
    """Ensure at least 2 time grid points for a valid integration interval.

    The vendored ``build_solver_config()`` sets ``time_points = nfe``, but
    ``torch.linspace(0, 1, 1)`` produces ``[0.0]`` — a single point with no
    interval.  The ODE solver returns the initial noise unchanged (0 steps).
    With ``time_points >= 2``, ``linspace`` gives ``[0.0, ..., 1.0]`` and the
    solver integrates across that interval (actual NFE controlled by
    ``step_size``).

    Args:
        solver_config: Solver config dict (mutated in-place and returned).

    Returns:
        The same dict with ``time_points`` clamped to >= 2.
    """
    if solver_config.get("time_points", 0) < 2:
        solver_config["time_points"] = 2
    return solver_config


class MOTFMGenerator:
    """Generate volumes using a trained MOTFM model.

    Works with any MOTFM checkpoint (our FOMO-60K or author-provided).
    Does NOT require a dataloader — generates from pure noise.

    Args:
        config_path: Path to MOTFM YAML config.
        checkpoint_path: Path to Lightning .ckpt file.
        device: Torch device for inference.
        repo_root: Project root (to locate src/external/MOTFM).
        volume_size: Spatial size per axis for generation. If None, read from
            ``data_args.volume_size`` in config (default 192 for FOMO-60K,
            96 for pretrained MSD Brain checkpoint).
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        repo_root: Path | None = None,
        volume_size: int | None = None,
    ) -> None:
        _ensure_motfm_on_path(repo_root)

        from inferer import build_solver_config, load_model_from_checkpoint
        from utils.general_utils import load_config

        self.config = load_config(str(config_path))
        self.device = device

        logger.info("Loading MOTFM checkpoint: %s", checkpoint_path)
        self.model, self.metadata = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config=self.config,
            device=device,
        )
        logger.info(
            "MOTFM model loaded (epoch=%s, step=%s)",
            self.metadata.get("epoch"),
            self.metadata.get("global_step"),
        )

        # Cache builder for solver configs
        self._build_solver_config = build_solver_config

        # Detect output data range from config for correct post-processing
        norm_cfg = self.config.get("data_args", {}).get("image_norm", {})
        if isinstance(norm_cfg, dict):
            self._data_range = tuple(norm_cfg.get("range", [0.0, 1.0]))
        elif norm_cfg == "minmax_0_1":
            self._data_range = (0.0, 1.0)
        else:
            self._data_range = (-1.0, 1.0)

        # Spatial size: CLI override > config > default 192
        if volume_size is not None:
            self._volume_size = volume_size
        else:
            self._volume_size = int(self.config.get("data_args", {}).get("volume_size", 192))
        logger.info("Data range: %s, volume_size: %d", self._data_range, self._volume_size)

    def _make_solver_config(self, nfe: int) -> dict:
        """Build solver config for a given number of function evaluations."""
        solver_config = self._build_solver_config(self.config, num_inference_steps=nfe)
        return _fix_solver_time_points(solver_config)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        nfe: int,
        batch_size: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate volumes at specified NFE.

        Args:
            n_samples: Number of volumes to generate.
            nfe: Number of function evaluations (ODE steps).
            batch_size: Generation batch size.
            seed: Base random seed (per-sample seeds = seed, seed+1, ...).

        Returns:
            Tensor of shape ``(n_samples, S, S, S)`` float32 in [0, 1],
            where S = ``self._volume_size``.
        """
        from utils.utils_fm import sample_with_solver

        solver_config = self._make_solver_config(nfe)
        spatial_dims = self.config["model_args"].get("spatial_dims", 3)
        in_channels = self.config["model_args"].get("in_channels", 1)
        s = self._volume_size

        if spatial_dims == 3:
            vol_shape = (in_channels, s, s, s)
        else:
            vol_shape = (in_channels, s, s)

        logger.info(
            "Generating %d volumes at NFE=%d (batch_size=%d, solver=%s)",
            n_samples,
            nfe,
            batch_size,
            solver_config.get("method", "midpoint"),
        )

        all_volumes = []
        n_generated = 0
        t_start = time.time()

        while n_generated < n_samples:
            bs = min(batch_size, n_samples - n_generated)

            # Deterministic per-sample seeding
            batch_seed = seed + n_generated
            gen = torch.Generator(device="cpu").manual_seed(batch_seed)
            x_init = torch.randn(bs, *vol_shape, generator=gen, device="cpu")
            x_init = x_init.to(self.device)

            sol = sample_with_solver(
                model=self.model,
                x_init=x_init,
                solver_config=solver_config,
                cond=None,
                masks=None,
            )

            # sol shape: [time_points, B, C, D, H, W] or [B, C, D, H, W]
            if sol.dim() == len(vol_shape) + 2:
                # Trajectory: take final state
                final = sol[-1]
            else:
                final = sol

            # Remove channel dim (1-channel MRI) and normalize to [0, 1]
            if final.shape[1] == 1:
                final = final.squeeze(1)
            lo, hi = self._data_range
            final = final.clamp(lo, hi).float()
            final = (final - lo) / (hi - lo)
            final = final.cpu()

            all_volumes.append(final)
            n_generated += bs

            if n_generated % max(1, n_samples // 10) == 0 or n_generated == n_samples:
                elapsed = time.time() - t_start
                rate = n_generated / elapsed if elapsed > 0 else 0
                logger.info(
                    "  %d/%d generated (%.1f s, %.2f vol/s)",
                    n_generated,
                    n_samples,
                    elapsed,
                    rate,
                )

        result = torch.cat(all_volumes, dim=0)[:n_samples]
        elapsed = time.time() - t_start
        logger.info(
            "Generation complete: %d volumes in %.1f s (%.2f s/vol)",
            n_samples,
            elapsed,
            elapsed / n_samples,
        )
        return result
