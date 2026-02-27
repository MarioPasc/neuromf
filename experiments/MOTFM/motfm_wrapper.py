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
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _ensure_motfm_on_path(repo_root: Optional[Path] = None) -> None:
    """Add MOTFM external code to sys.path if not already present."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    motfm_dir = str(repo_root / "src" / "external" / "MOTFM")
    if motfm_dir not in sys.path:
        sys.path.insert(0, motfm_dir)


class MOTFMGenerator:
    """Generate volumes using a trained MOTFM model.

    Works with any MOTFM checkpoint (our FOMO-60K or author-provided).
    Does NOT require a dataloader — generates from pure noise.

    Args:
        config_path: Path to MOTFM YAML config.
        checkpoint_path: Path to Lightning .ckpt file.
        device: Torch device for inference.
        repo_root: Project root (to locate src/external/MOTFM).
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        repo_root: Optional[Path] = None,
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

    def _make_solver_config(self, nfe: int) -> dict:
        """Build solver config for a given number of function evaluations."""
        return self._build_solver_config(self.config, num_inference_steps=nfe)

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
            Tensor of shape (n_samples, 192, 192, 192) float32 in [0, 1].
        """
        from utils.utils_fm import sample_with_solver

        solver_config = self._make_solver_config(nfe)
        spatial_dims = self.config["model_args"].get("spatial_dims", 3)
        in_channels = self.config["model_args"].get("in_channels", 1)

        # Determine volume shape from config or default 192^3
        if spatial_dims == 3:
            vol_shape = (in_channels, 192, 192, 192)
        else:
            vol_shape = (in_channels, 192, 192)

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

            # Remove channel dim (1-channel MRI) and clip to [0, 1]
            if final.shape[1] == 1:
                final = final.squeeze(1)
            final = final.clamp(0.0, 1.0).float().cpu()

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
