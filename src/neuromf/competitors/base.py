"""Abstract base class for pixel-space competitor generators.

All pixel-space competitor models (MOTFM, DDPM, etc.) share an identical
generation loop: create noise, call the model in batches, squeeze the
channel dim, clamp, and normalize to [0, 1].  The only model-specific
parts are:

1. **Loading** the checkpoint (framework, config format, model class).
2. **Running one batch** through the sampler.

Subclasses implement :meth:`_load_model` and :meth:`_generate_batch`.
Everything else is handled by the base.

.. note::

    NeuroiMF is *not* a competitor — it operates in latent space with a
    separate VAE decode step.  This base class is only for models that
    generate full-resolution volumes directly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseCompetitorGenerator(ABC):
    """Abstract base for pixel-space generative model inference.

    Subclasses must set :attr:`model_name` (class-level str) and implement
    :meth:`_load_model` and :meth:`_generate_batch`.

    Args:
        config_path: Path to model-specific YAML config.
        checkpoint_path: Path to checkpoint file.
        device: Torch device for inference.
        volume_size: Spatial size per axis.  If None, read from config
            (default 192).
    """

    #: Human-readable model name (e.g. ``"MOTFM"``).  Set by subclass.
    model_name: str = "BaseCompetitor"

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
        volume_size: int | None = None,
    ) -> None:
        self.device = device
        self.config: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

        # Subclass loads the model and populates self.config / self.metadata
        self._load_model(Path(config_path), Path(checkpoint_path))

        self._data_range = self._detect_data_range()
        self._volume_size = self._resolve_volume_size(volume_size)

        logger.info(
            "%s ready — data_range=%s, volume_size=%d",
            self.model_name,
            self._data_range,
            self._volume_size,
        )

    # ------------------------------------------------------------------
    # Abstract methods — subclass MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(
        self,
        config_path: Path,
        checkpoint_path: Path,
    ) -> None:
        """Load the model checkpoint and populate ``self.config`` and
        ``self.metadata``.

        Called once from ``__init__``.  Subclasses should handle any
        ``sys.path`` manipulation needed for vendored dependencies.

        Args:
            config_path: Path to model-specific YAML config.
            checkpoint_path: Path to checkpoint file.
        """

    @abstractmethod
    def _generate_batch(
        self,
        x_init: Tensor,
        nfe: int,
    ) -> Tensor:
        """Run the model on a single batch of initial noise.

        Args:
            x_init: Noise tensor ``(B, C, D, H, W)`` on ``self.device``.
            nfe: Number of function evaluations / sampling steps.

        Returns:
            Generated volumes ``(B, C, D, H, W)`` on ``self.device``.
            The channel dim will be squeezed by :meth:`generate`.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _detect_data_range(self) -> tuple[float, float]:
        """Detect output data range from config (shared across wrappers).

        Returns:
            ``(lo, hi)`` tuple for clamping and normalization.
        """
        norm_cfg = self.config.get("data_args", {}).get("image_norm", {})
        if isinstance(norm_cfg, dict):
            return tuple(norm_cfg.get("range", [0.0, 1.0]))  # type: ignore[return-value]
        if norm_cfg == "minmax_0_1":
            return (0.0, 1.0)
        return (-1.0, 1.0)

    def _resolve_volume_size(self, volume_size: int | None) -> int:
        """Resolve spatial size: CLI override > config > default 192.

        Args:
            volume_size: Explicit override, or None.

        Returns:
            Spatial size per axis.
        """
        if volume_size is not None:
            return volume_size
        return int(
            self.config.get("data_args", {}).get("volume_size", 192)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        nfe: int,
        batch_size: int = 1,
        seed: int = 42,
    ) -> Tensor:
        """Generate volumes at specified NFE.

        Handles noise creation, per-sample seeding, batched inference,
        channel squeeze, clamp/normalize to [0, 1], and progress logging.
        Subclasses do NOT override this — they implement
        :meth:`_generate_batch` instead.

        Args:
            n_samples: Number of volumes to generate.
            nfe: Number of function evaluations (sampling steps).
            batch_size: Volumes per forward pass.
            seed: Base random seed (per-sample: ``seed + i``).

        Returns:
            ``(n_samples, S, S, S)`` float32 in [0, 1].
        """
        s = self._volume_size
        in_channels = int(
            self.config.get("model_args", {}).get("in_channels", 1)
        )
        spatial_dims = int(
            self.config.get("model_args", {}).get("spatial_dims", 3)
        )

        if spatial_dims == 3:
            vol_shape = (in_channels, s, s, s)
        else:
            vol_shape = (in_channels, s, s)

        logger.info(
            "%s: generating %d volumes at NFE=%d (batch=%d)",
            self.model_name,
            n_samples,
            nfe,
            batch_size,
        )

        all_volumes: list[Tensor] = []
        n_generated = 0
        t_start = time.time()

        while n_generated < n_samples:
            bs = min(batch_size, n_samples - n_generated)

            # Deterministic per-sample seeding
            gen = torch.Generator(device="cpu").manual_seed(seed + n_generated)
            x_init = torch.randn(bs, *vol_shape, generator=gen, device="cpu")
            x_init = x_init.to(self.device)

            # Model-specific forward pass
            output = self._generate_batch(x_init, nfe)

            # Squeeze channel dim (1-channel MRI) and normalize to [0, 1]
            if output.shape[1] == 1:
                output = output.squeeze(1)

            lo, hi = self._data_range
            output = output.clamp(lo, hi).float()
            output = (output - lo) / (hi - lo)
            all_volumes.append(output.cpu())

            n_generated += bs

            # Progress logging (~10 updates per run)
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
            "%s generation complete: %d volumes in %.1f s (%.2f s/vol)",
            self.model_name,
            n_samples,
            elapsed,
            elapsed / max(n_samples, 1),
        )
        return result
