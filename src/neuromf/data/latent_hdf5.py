"""Low-level HDF5 I/O for per-dataset latent shard files.

Each shard (e.g. ``PT005_IXI.h5``) stores all encoded latents for one
FOMO-60K dataset. Provides create/read/write primitives and index building
for ``LatentDataset``.

Shard layout::

    /latents        float32, shape (N, 4, 48, 48, 48), chunks=(1, 4, 48, 48, 48)
    /written        bool,    shape (N,) — True if slot contains valid data
    /subject_id     vlen-string, shape (N,)
    /session_id     vlen-string, shape (N,)
    /source_path    vlen-string, shape (N,)

    Attributes:
      dataset_name   str
      n_volumes      int        total allocated slots
      n_written      int        count of valid entries (updated on close)
      latent_shape   (4, 48, 48, 48)
      dtype          "float32"
      scale_factor   0.96240234375
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Scale factor extracted from diffusion checkpoint (see CLAUDE.md §2)
_DEFAULT_SCALE_FACTOR = 0.96240234375
_DEFAULT_LATENT_SHAPE = (4, 48, 48, 48)


def create_shard(
    path: Path,
    dataset_name: str,
    n_volumes: int,
    latent_shape: tuple[int, ...] = _DEFAULT_LATENT_SHAPE,
    scale_factor: float = _DEFAULT_SCALE_FACTOR,
) -> h5py.File:
    """Pre-allocate an HDF5 shard file for one dataset.

    Args:
        path: Output ``.h5`` file path.
        dataset_name: FOMO-60K dataset identifier (e.g. ``"PT005_IXI"``).
        n_volumes: Number of slots to allocate.
        latent_shape: Shape per latent (default ``(4, 48, 48, 48)``).
        scale_factor: VAE scale factor stored as attribute.

    Returns:
        Open ``h5py.File`` in read-write mode.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(str(path), "w")

    # Main latent data — chunked per-sample for efficient random access
    f.create_dataset(
        "latents",
        shape=(n_volumes, *latent_shape),
        dtype="float32",
        chunks=(1, *latent_shape),
    )

    # Written mask for resume support
    f.create_dataset("written", shape=(n_volumes,), dtype=bool)

    # Variable-length string datasets for metadata
    vlen_str = h5py.string_dtype()
    f.create_dataset("subject_id", shape=(n_volumes,), dtype=vlen_str)
    f.create_dataset("session_id", shape=(n_volumes,), dtype=vlen_str)
    f.create_dataset("source_path", shape=(n_volumes,), dtype=vlen_str)

    # Attributes
    f.attrs["dataset_name"] = dataset_name
    f.attrs["n_volumes"] = n_volumes
    f.attrs["n_written"] = 0
    f.attrs["latent_shape"] = latent_shape
    f.attrs["dtype"] = "float32"
    f.attrs["scale_factor"] = scale_factor

    logger.info("Created shard %s: %d slots, latent_shape=%s", path.name, n_volumes, latent_shape)
    return f


def write_sample(
    h5file: h5py.File,
    index: int,
    z: torch.Tensor,
    subject_id: str,
    session_id: str,
    source_path: str,
) -> None:
    """Write one latent and its metadata into a shard at the given index.

    Args:
        h5file: Open HDF5 file (read-write).
        index: Slot index within the shard.
        z: Latent tensor, shape ``(C, D, H, W)``.
        subject_id: Subject identifier string.
        session_id: Session identifier string.
        source_path: Original NIfTI file path.
    """
    h5file["latents"][index] = z.cpu().numpy().astype(np.float32)
    h5file["subject_id"][index] = subject_id
    h5file["session_id"][index] = session_id
    h5file["source_path"][index] = source_path
    h5file["written"][index] = True


def read_sample(h5file: h5py.File, index: int) -> tuple[torch.Tensor, dict[str, str]]:
    """Read one latent and metadata from an open shard.

    Args:
        h5file: Open HDF5 file (read mode).
        index: Slot index within the shard.

    Returns:
        Tuple of ``(z, metadata)`` where ``z`` is a float32 tensor of shape
        ``(C, D, H, W)`` and ``metadata`` is a dict with ``subject_id``,
        ``session_id``, ``source_path``, and ``dataset``.
    """
    z = torch.from_numpy(h5file["latents"][index].astype(np.float32))
    metadata = {
        "subject_id": h5file["subject_id"][index].decode()
        if isinstance(h5file["subject_id"][index], bytes)
        else str(h5file["subject_id"][index]),
        "session_id": h5file["session_id"][index].decode()
        if isinstance(h5file["session_id"][index], bytes)
        else str(h5file["session_id"][index]),
        "source_path": h5file["source_path"][index].decode()
        if isinstance(h5file["source_path"][index], bytes)
        else str(h5file["source_path"][index]),
        "dataset": str(h5file.attrs["dataset_name"]),
    }
    return z, metadata


def get_written_mask(h5file: h5py.File) -> np.ndarray:
    """Return boolean mask indicating which shard slots have been written.

    Args:
        h5file: Open HDF5 file.

    Returns:
        Boolean numpy array of shape ``(n_volumes,)``.
    """
    return np.array(h5file["written"], dtype=bool)


def discover_shards(latent_dir: Path) -> list[Path]:
    """Find all ``.h5`` shard files in a directory, sorted by name.

    Args:
        latent_dir: Directory to search.

    Returns:
        Sorted list of ``.h5`` file paths.
    """
    return sorted(Path(latent_dir).glob("*.h5"))


def build_global_index(shard_paths: list[Path]) -> list[tuple[Path, int]]:
    """Build a flat index over all written samples across shards.

    Opens each shard in read-only mode, reads the ``written`` mask, and
    builds a list of ``(shard_path, local_index)`` tuples for all valid
    entries.

    Args:
        shard_paths: Ordered list of shard file paths.

    Returns:
        List of ``(shard_path, local_idx)`` for every written sample.
    """
    index: list[tuple[Path, int]] = []
    for shard_path in shard_paths:
        with h5py.File(str(shard_path), "r") as f:
            mask = get_written_mask(f)
            for local_idx in np.where(mask)[0]:
                index.append((shard_path, int(local_idx)))
    logger.info(
        "Global index: %d samples across %d shards",
        len(index),
        len(shard_paths),
    )
    return index
