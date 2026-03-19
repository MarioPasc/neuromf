"""Shared HDF5 I/O for competitor-generated volumes.

Provides a single ``write_volumes_h5`` function used by all competitor
generation scripts.  Output format is identical to NeuroiMF's decoded
volume archives, enabling unified downstream evaluation.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


def write_volumes_h5(
    output_path: Path | str,
    volumes: torch.Tensor,
    nfe: int,
    checkpoint_path: str,
    seed: int,
    model_name: str = "unknown",
) -> None:
    """Write generated volumes to HDF5 in the NeuroiMF-compatible format.

    Args:
        output_path: Path to write the ``.h5`` file.
        volumes: Tensor ``(N, D, H, W)`` float32.
        nfe: Number of function evaluations used.
        checkpoint_path: Source checkpoint path for metadata.
        seed: Random seed used for generation.
        model_name: Model identifier stored in HDF5 attrs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = volumes.shape[0]
    vol_shape = volumes.shape[1:]  # (D, H, W)

    with h5py.File(str(output_path), "w") as f:
        f.create_dataset(
            "volumes",
            shape=(n_samples, *vol_shape),
            dtype=np.float32,
            chunks=(1, *vol_shape),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "decode_timing_ms",
            shape=(n_samples,),
            dtype=np.float32,
        )

        f["volumes"][:] = volumes.numpy()
        f["decode_timing_ms"][:] = 0.0  # No separate decode for pixel-space models

        f.attrs["n_samples"] = n_samples
        f.attrs["n_written"] = n_samples
        f.attrs["nfe"] = nfe
        f.attrs["model"] = model_name
        f.attrs["checkpoint_path"] = str(checkpoint_path)
        f.attrs["seed"] = seed
        f.attrs["timestamp"] = datetime.datetime.now(tz=datetime.UTC).isoformat()

    size_gb = output_path.stat().st_size / 1e9
    logger.info(
        "Saved %s: %d volumes (%.2f GB), model=%s",
        output_path.name,
        n_samples,
        size_gb,
        model_name,
    )
