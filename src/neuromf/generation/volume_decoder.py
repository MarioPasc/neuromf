"""Batch VAE decoding from latent HDF5 to volume HDF5.

Reads normalised latents, denormalises using per-channel statistics,
decodes through the frozen MAISI VAE, and writes decoded volumes
to a volume HDF5 archive.

Note: VAE ``decode()`` already divides by ``scale_factor`` internally.
Do NOT divide before calling.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

from neuromf.generation.h5_manager import H5Manager, H5VolumeConfig
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper

logger = logging.getLogger(__name__)


class VolumeDecoder:
    """Decode latents from HDF5 through frozen MAISI VAE.

    Args:
        vae: MAISI VAE wrapper instance.
        latent_mean: Per-channel latent mean, shape ``(4,)``.
        latent_std: Per-channel latent std, shape ``(4,)``.
        device: CUDA device for decoding.
    """

    def __init__(
        self,
        vae: MAISIVAEWrapper,
        latent_mean: Tensor,
        latent_std: Tensor,
        device: torch.device,
    ) -> None:
        self.vae = vae
        self.latent_mean = latent_mean.to(device).reshape(1, -1, 1, 1, 1)
        self.latent_std = latent_std.to(device).reshape(1, -1, 1, 1, 1)
        self.device = device

    def _denormalise(self, z_hat: Tensor) -> Tensor:
        """Denormalise latents from standardised to original space.

        Args:
            z_hat: Normalised latent, shape ``(B, C, D, H, W)``.

        Returns:
            Denormalised latent ``z_0 = z_hat * std + mean``.
        """
        return z_hat * self.latent_std + self.latent_mean

    @torch.no_grad()
    def decode(
        self,
        latent_h5_path: Path,
        output_h5_path: Path,
        batch_size: int = 1,
        volume_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Decode all latents in an archive and write volumes to HDF5.

        Args:
            latent_h5_path: Path to latent HDF5 file.
            output_h5_path: Path to output volume HDF5 file.
            batch_size: Decode batch size (1 recommended for 192^3 volumes).
            volume_shape: Output volume shape ``(D, H, W)``. If None, inferred
                from latent shape as ``latent_spatial * downsample_factor``.
        """
        latent_h5_path = Path(latent_h5_path)
        output_h5_path = Path(output_h5_path)

        # Read latent archive metadata
        src_meta = H5Manager.read_metadata(latent_h5_path)
        n_samples = src_meta.get("n_samples", 0)
        nfe = src_meta.get("nfe", 0)

        logger.info(
            "Decoding %d latents from %s (NFE=%d, batch=%d)",
            n_samples,
            latent_h5_path.name,
            nfe,
            batch_size,
        )

        # Infer volume shape from latent shape if not provided
        if volume_shape is None:
            with h5py.File(str(latent_h5_path), "r") as lf:
                latent_spatial = lf["latents"].shape[2:]  # (D, H, W) of latent
            # VAE upsamples by 4x per axis
            volume_shape = tuple(s * 4 for s in latent_spatial)

        vol_config = H5VolumeConfig(n_samples=n_samples, volume_shape=volume_shape)
        vol_meta = {
            "nfe": nfe,
            "n_samples": n_samples,
            "voxel_size_mm": [1.0, 1.0, 1.0],
            "volume_shape": list(vol_config.volume_shape),
            "intensity_range": [0.0, 1.0],
            "source_latent_h5": str(latent_h5_path.name),
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        vol_h5 = H5Manager.create_volume_archive(output_h5_path, vol_config, vol_meta)

        try:
            src_h5 = h5py.File(str(latent_h5_path), "r")
            idx = 0
            while idx < n_samples:
                b = min(batch_size, n_samples - idx)

                # Read latents (stored as float16) -> float32
                z_hat = torch.from_numpy(src_h5["latents"][idx : idx + b].astype(np.float32)).to(
                    self.device
                )

                # Denormalise: z_0 = z_hat * std + mean
                z_0 = self._denormalise(z_hat)

                # Time with CUDA events
                if self.device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                # VAE decode (internally divides by scale_factor)
                x_hat = self.vae.decode(z_0)

                if self.device.type == "cuda":
                    end_event.record()
                    torch.cuda.synchronize()
                    batch_time_ms = start_event.elapsed_time(end_event)
                    per_sample_ms = [batch_time_ms / b] * b
                else:
                    per_sample_ms = [0.0] * b

                # Clamp to [0, 1]
                x_hat = x_hat.clamp(0.0, 1.0)

                H5Manager.write_volume_batch(vol_h5, idx, x_hat.cpu(), per_sample_ms)

                idx += b
                if idx % 10 == 0 or idx >= n_samples:
                    logger.info("  Decoded %d / %d", idx, n_samples)

            src_h5.close()
            vol_h5.flush()
        finally:
            vol_h5.close()

        logger.info("Volume archive saved: %s", output_h5_path)
