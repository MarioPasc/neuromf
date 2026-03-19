"""DDPM inference wrapper — load checkpoint and generate volumes via DDIM.

Mirrors ``motfm_wrapper.py`` but uses DDPMLightningModule with DDIM sampling.

Requires on sys.path:
    src/external/MOTFM (for build_model)
    experiments/DDPM (for ddpm_module)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _ensure_paths_on_sys(repo_root: Path | None = None) -> None:
    """Add required code to sys.path."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    for sub in ["src/external/MOTFM", "experiments/DDPM", "experiments/MOTFM"]:
        p = str(repo_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


class DDPMGenerator:
    """Generate volumes using a trained DDPM model via DDIM sampling.

    Args:
        config_path: Path to DDPM YAML config.
        checkpoint_path: Path to Lightning .ckpt file.
        device: Torch device for inference.
        volume_size: Spatial size per axis (overrides config).
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        volume_size: int | None = None,
    ) -> None:
        _ensure_paths_on_sys()

        from utils.general_utils import load_config

        self.config = load_config(str(config_path))
        self.device = device

        from ddpm_module import DDPMLightningModule

        logger.info("Loading DDPM checkpoint: %s", checkpoint_path)
        self.module = DDPMLightningModule(self.config)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.module.load_state_dict(ckpt["state_dict"], strict=True)
        self.module.to(device)
        self.module.eval()

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

        # Data range
        norm_cfg = self.config.get("data_args", {}).get("image_norm", {})
        if isinstance(norm_cfg, dict):
            self._data_range = tuple(norm_cfg.get("range", [0.0, 1.0]))
        elif norm_cfg == "minmax_0_1":
            self._data_range = (0.0, 1.0)
        else:
            self._data_range = (-1.0, 1.0)

        # Volume size
        if volume_size is not None:
            self._volume_size = volume_size
        else:
            self._volume_size = int(self.config.get("data_args", {}).get("volume_size", 192))
        logger.info("Data range: %s, volume_size: %d", self._data_range, self._volume_size)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        nfe: int,
        batch_size: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate volumes via DDIM sampling.

        Args:
            n_samples: Number of volumes to generate.
            nfe: Number of DDIM reverse steps (= NFE).
            batch_size: Generation batch size.
            seed: Base random seed.

        Returns:
            Tensor of shape ``(n_samples, S, S, S)`` float32 in [0, 1].
        """
        s = self._volume_size
        vol_shape = (1, s, s, s)

        logger.info("Generating %d volumes at %d DDIM steps (batch=%d)", n_samples, nfe, batch_size)

        all_volumes = []
        n_generated = 0
        t_start = time.time()

        while n_generated < n_samples:
            bs = min(batch_size, n_samples - n_generated)
            gen = torch.Generator(device="cpu").manual_seed(seed + n_generated)
            x_T = torch.randn(bs, *vol_shape, generator=gen, device="cpu").to(self.device)

            x_0 = self.module.ddim_sample(x_T, num_steps=nfe, device=self.device)

            # Squeeze channel dim and normalize to [0, 1]
            final = x_0.squeeze(1)
            lo, hi = self._data_range
            final = final.clamp(lo, hi).float()
            final = (final - lo) / (hi - lo)
            all_volumes.append(final.cpu())
            n_generated += bs

            if n_generated % max(1, n_samples // 10) == 0 or n_generated == n_samples:
                elapsed = time.time() - t_start
                logger.info("  %d/%d generated (%.1f s)", n_generated, n_samples, elapsed)

        result = torch.cat(all_volumes, dim=0)[:n_samples]
        logger.info("Generation complete: %d volumes in %.1f s", n_samples, time.time() - t_start)
        return result
