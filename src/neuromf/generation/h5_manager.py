"""HDF5 I/O utilities for generation archives (latents and decoded volumes).

Distinct from Phase 1 latent shards (``latent_hdf5.py``), these archives store
generated outputs with per-sample timing, seed metadata, and gzip compression.

Schemas follow Phase 5 specification sections 3.3 and 3.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class H5LatentConfig:
    """Configuration for latent HDF5 archives."""

    n_samples: int = 2000
    latent_shape: tuple[int, ...] = (4, 48, 48, 48)
    dtype: str = "float16"
    compression: str = "gzip"
    compression_opts: int = 4


@dataclass
class H5VolumeConfig:
    """Configuration for volume HDF5 archives."""

    n_samples: int = 2000
    volume_shape: tuple[int, ...] = (192, 192, 192)
    dtype: str = "float32"
    compression: str = "gzip"
    compression_opts: int = 4


class H5Manager:
    """Unified HDF5 read/write manager for generation archives."""

    @staticmethod
    def create_latent_archive(
        path: Path,
        config: H5LatentConfig,
        metadata: dict,
    ) -> h5py.File:
        """Create an HDF5 archive for generated latents.

        Args:
            path: Output ``.h5`` file path.
            config: Archive shape/dtype/compression config.
            metadata: Dict of attributes to attach (nfe, seed, checkpoint, etc.).

        Returns:
            Open ``h5py.File`` in read-write mode.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        f = h5py.File(str(path), "w")

        np_dtype = np.float16 if config.dtype == "float16" else np.float32
        chunk_shape = (1, *config.latent_shape)

        f.create_dataset(
            "latents",
            shape=(config.n_samples, *config.latent_shape),
            dtype=np_dtype,
            chunks=chunk_shape,
            compression=config.compression,
            compression_opts=config.compression_opts,
        )
        f.create_dataset("noise_seeds", shape=(config.n_samples,), dtype=np.int64)
        f.create_dataset("timing_ms", shape=(config.n_samples,), dtype=np.float32)

        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                f.attrs[key] = list(value)
            elif value is not None:
                f.attrs[key] = value

        f.attrs["n_written"] = 0

        logger.info(
            "Created latent archive %s: %d samples, shape=%s, dtype=%s",
            path.name,
            config.n_samples,
            config.latent_shape,
            config.dtype,
        )
        return f

    @staticmethod
    def create_volume_archive(
        path: Path,
        config: H5VolumeConfig,
        metadata: dict,
    ) -> h5py.File:
        """Create an HDF5 archive for decoded volumes.

        Args:
            path: Output ``.h5`` file path.
            config: Archive shape/dtype/compression config.
            metadata: Dict of attributes to attach.

        Returns:
            Open ``h5py.File`` in read-write mode.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        f = h5py.File(str(path), "w")

        np_dtype = np.float32 if config.dtype == "float32" else np.float16
        chunk_shape = (1, *config.volume_shape)

        f.create_dataset(
            "volumes",
            shape=(config.n_samples, *config.volume_shape),
            dtype=np_dtype,
            chunks=chunk_shape,
            compression=config.compression,
            compression_opts=config.compression_opts,
        )
        f.create_dataset("decode_timing_ms", shape=(config.n_samples,), dtype=np.float32)

        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                f.attrs[key] = list(value)
            elif value is not None:
                f.attrs[key] = value

        f.attrs["n_written"] = 0

        logger.info(
            "Created volume archive %s: %d samples, shape=%s",
            path.name,
            config.n_samples,
            config.volume_shape,
        )
        return f

    @staticmethod
    def write_latent_batch(
        h5file: h5py.File,
        start_idx: int,
        latents: Tensor,
        seeds: list[int],
        timing_ms: list[float],
    ) -> None:
        """Write a batch of latents with metadata.

        Args:
            h5file: Open HDF5 file in write mode.
            start_idx: Starting index in the archive.
            latents: Tensor of shape ``(B, C, D, H, W)``.
            seeds: Per-sample random seeds.
            timing_ms: Per-sample generation time in milliseconds.
        """
        B = latents.shape[0]
        end_idx = start_idx + B
        h5file["latents"][start_idx:end_idx] = latents.cpu().numpy()
        h5file["noise_seeds"][start_idx:end_idx] = seeds
        h5file["timing_ms"][start_idx:end_idx] = timing_ms
        h5file.attrs["n_written"] = int(h5file.attrs.get("n_written", 0)) + B

    @staticmethod
    def write_volume_batch(
        h5file: h5py.File,
        start_idx: int,
        volumes: Tensor,
        timing_ms: list[float],
    ) -> None:
        """Write a batch of decoded volumes with timing.

        Args:
            h5file: Open HDF5 file in write mode.
            start_idx: Starting index in the archive.
            volumes: Tensor of shape ``(B, D, H, W)`` or ``(B, 1, D, H, W)``.
            timing_ms: Per-sample decode time in milliseconds.
        """
        B = volumes.shape[0]
        end_idx = start_idx + B
        # Squeeze channel dim if present
        vols_np = volumes.cpu().numpy()
        if vols_np.ndim == 5 and vols_np.shape[1] == 1:
            vols_np = vols_np[:, 0]
        h5file["volumes"][start_idx:end_idx] = vols_np
        h5file["decode_timing_ms"][start_idx:end_idx] = timing_ms
        h5file.attrs["n_written"] = int(h5file.attrs.get("n_written", 0)) + B

    @staticmethod
    def is_complete(path: Path) -> bool:
        """Check if an archive has all expected samples written.

        Archives created before ``n_written`` tracking was added will lack
        the attribute and are treated as incomplete (returns False).

        Args:
            path: Path to an HDF5 archive (latent or volume).

        Returns:
            True if ``n_written >= dataset leading dim``.
        """
        try:
            with h5py.File(str(path), "r") as f:
                n_written = int(f.attrs.get("n_written", 0))
                for key in ("volumes", "latents"):
                    if key in f:
                        return n_written >= f[key].shape[0]
        except Exception:
            return False
        return False

    @staticmethod
    def read_latent(path: Path, index: int) -> Tensor:
        """Read a single latent from an archive.

        Args:
            path: Path to latent HDF5 file.
            index: Sample index.

        Returns:
            Float32 tensor of shape ``(C, D, H, W)``.
        """
        with h5py.File(str(path), "r") as f:
            data = f["latents"][index]
        return torch.from_numpy(data.astype(np.float32))

    @staticmethod
    def read_volume(path: Path, index: int) -> Tensor:
        """Read a single decoded volume from an archive.

        Args:
            path: Path to volume HDF5 file.
            index: Sample index.

        Returns:
            Float32 tensor of shape ``(D, H, W)``.
        """
        with h5py.File(str(path), "r") as f:
            data = f["volumes"][index]
        return torch.from_numpy(data.astype(np.float32))

    @staticmethod
    def read_metadata(path: Path) -> dict:
        """Read all attributes from an HDF5 archive.

        Args:
            path: Path to HDF5 file.

        Returns:
            Dict of archive attributes.
        """
        with h5py.File(str(path), "r") as f:
            metadata = {}
            for key, value in f.attrs.items():
                if isinstance(value, np.ndarray):
                    metadata[key] = value.tolist()
                elif isinstance(value, bytes):
                    metadata[key] = value.decode()
                else:
                    metadata[key] = value
        return metadata
