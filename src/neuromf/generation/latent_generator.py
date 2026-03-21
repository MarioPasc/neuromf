"""Batch generation of latent tensors at specified NFE levels.

Generates normalised latents from Gaussian noise via a trained MeanFlow model,
storing results in chunked HDF5 archives with per-sample timing and seed
metadata. Supports shared noise protocol across NFE levels for fair comparison.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from neuromf.generation.h5_manager import H5LatentConfig, H5Manager
from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step

logger = logging.getLogger(__name__)


class LatentGenerator:
    """Batch generation of latent tensors at specified NFE levels.

    Args:
        model: Trained MeanFlow UNet (EMA weights loaded).
        prediction_type: ``"u"`` or ``"x"`` prediction parameterisation.
        device: CUDA device for generation.
        norm_correction: Scalar gamma dividing the velocity to compensate
            for compound velocity norm overshoot. Default 1.0 (no correction).
    """

    def __init__(
        self,
        model: nn.Module,
        prediction_type: str,
        device: torch.device,
        norm_correction: float = 1.0,
    ) -> None:
        self.model = model
        self.prediction_type = prediction_type
        self.device = device
        self.norm_correction = norm_correction

    def _pre_generate_noise(
        self,
        n_samples: int,
        latent_shape: tuple[int, ...],
        seed: int,
    ) -> Tensor:
        """Pre-generate all noise tensors for reproducibility.

        Args:
            n_samples: Number of samples.
            latent_shape: Shape per latent (e.g. ``(4, 48, 48, 48)``).
            seed: Random seed for the generator.

        Returns:
            Noise tensor of shape ``(n_samples, *latent_shape)``.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(n_samples, *latent_shape, generator=gen)
        return noise

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        nfe: int,
        output_path: Path,
        batch_size: int = 8,
        base_seed: int = 42,
        latent_stats: dict | None = None,
        metadata: dict | None = None,
        latent_shape: tuple[int, ...] = (4, 48, 48, 48),
        shared_noise: Tensor | None = None,
        post_process_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Generate n_samples latents and write to HDF5.

        Args:
            n_samples: Total number of latents to generate.
            nfe: Number of function evaluations (1 = one-step).
            output_path: Path to output HDF5 file.
            batch_size: Generation batch size (limited by VRAM).
            base_seed: Random seed. If ``shared_noise`` is None, noise is
                generated with ``seed = base_seed``.
            latent_stats: Dict with ``"mean"`` and ``"std"`` arrays for metadata.
            metadata: Additional metadata to attach to the archive.
            latent_shape: Shape per latent.
            shared_noise: Pre-generated noise tensor ``(n_samples, *latent_shape)``.
                If provided, the same noise is used regardless of NFE level
                (shared noise protocol for NFE-consistency analysis).
            post_process_fn: Optional callable applied to each batch of latents
                after sampling (e.g. stochastic perturbation, variance rescaling).
                Receives float32 GPU tensor, must return same shape.
        """
        output_path = Path(output_path)

        logger.info(
            "Generating %d latents at NFE=%d (seed=%d, batch=%d)",
            n_samples,
            nfe,
            base_seed,
            batch_size,
        )

        # Use shared noise if provided, otherwise generate from seed
        if shared_noise is not None:
            all_noise = shared_noise[:n_samples]
        else:
            all_noise = self._pre_generate_noise(n_samples, latent_shape, base_seed)

        # Build archive metadata
        archive_meta = {
            "nfe": nfe,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "base_seed": base_seed,
            "prediction_type": self.prediction_type,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        if latent_stats is not None:
            per_ch = latent_stats.get("per_channel", {})
            n_ch = len(per_ch)
            if n_ch > 0:
                archive_meta["latent_mean"] = [per_ch[f"channel_{c}"]["mean"] for c in range(n_ch)]
                archive_meta["latent_std"] = [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)]
        if metadata:
            archive_meta.update(metadata)

        config = H5LatentConfig(n_samples=n_samples, latent_shape=latent_shape)
        h5file = H5Manager.create_latent_archive(output_path, config, archive_meta)

        try:
            self.model.eval()
            idx = 0
            while idx < n_samples:
                b = min(batch_size, n_samples - idx)
                noise_batch = all_noise[idx : idx + b].to(self.device)

                # Time with CUDA events
                if self.device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                if nfe == 1:
                    latents = sample_one_step(
                        self.model,
                        noise_batch,
                        self.prediction_type,
                        norm_correction=self.norm_correction,
                    )
                else:
                    latents = sample_euler(
                        self.model,
                        noise_batch,
                        nfe,
                        self.prediction_type,
                        norm_correction=self.norm_correction,
                    )

                if post_process_fn is not None:
                    latents = post_process_fn(latents)

                if self.device.type == "cuda":
                    end_event.record()
                    torch.cuda.synchronize()
                    batch_time_ms = start_event.elapsed_time(end_event)
                    per_sample_ms = [batch_time_ms / b] * b
                else:
                    per_sample_ms = [0.0] * b

                seeds = list(range(base_seed + idx, base_seed + idx + b))

                H5Manager.write_latent_batch(
                    h5file, idx, latents.cpu().half(), seeds, per_sample_ms
                )

                idx += b
                if idx % (batch_size * 10) == 0 or idx >= n_samples:
                    logger.info("  Generated %d / %d", idx, n_samples)

            # Flush to ensure n_written attr is persisted on network filesystems
            # (protects against SIGTERM/SIGKILL leaving stale metadata)
            h5file.flush()
        finally:
            h5file.close()

        logger.info("Latent archive saved: %s", output_path)
