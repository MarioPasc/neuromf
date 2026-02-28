"""HDF5-backed lazy-loading dataset for MOTFM training.

Reads volumes on-demand from an HDF5 file instead of loading everything
into RAM.  Compatible with the same HDF5 format produced by
``prepare_data.py`` (Phase 1 temp file kept as permanent output).

Expected HDF5 layout::

    /train    → dataset  (N_train, 1, 192, 192, 192)  float32
    /valid    → dataset  (N_val,   1, 192, 192, 192)  float32

Optional pre-computed normalization attributes (set by ``prepare_data.py``):

    /train.attrs["global_min"], /train.attrs["global_max"]

If not present, a full scan is performed on first access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class H5VolumeDataset(Dataset):
    """Lazy-loading dataset backed by an HDF5 file.

    Each ``__getitem__`` reads a single volume from disk and applies
    global min-max normalisation to [0, 1].

    Args:
        h5_path: Path to the HDF5 file.
        split: Dataset name inside the HDF5 file (``"train"`` or ``"valid"``).
        global_min: Pre-computed global minimum (avoids full scan).
        global_max: Pre-computed global maximum (avoids full scan).
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        global_min: Optional[float] = None,
        global_max: Optional[float] = None,
    ) -> None:
        self.h5_path = str(h5_path)
        self.split = split

        # Open once just to read metadata, then close
        with h5py.File(self.h5_path, "r") as f:
            if split not in f:
                raise KeyError(
                    f"Split '{split}' not found in {h5_path}. "
                    f"Available: {list(f.keys())}"
                )
            dset = f[split]
            self.n_samples: int = dset.shape[0]
            self.volume_shape: tuple[int, ...] = dset.shape[1:]  # (1, D, H, W)

            # Try to load pre-computed stats from attributes
            if global_min is None:
                global_min = dset.attrs.get("global_min", None)
            if global_max is None:
                global_max = dset.attrs.get("global_max", None)

        # Compute stats if needed (single-pass streaming)
        if global_min is None or global_max is None:
            logger.info(
                "Computing global min/max for split '%s' (%d volumes)...",
                split,
                self.n_samples,
            )
            global_min, global_max = self._compute_global_stats()
            logger.info("  global_min=%.6f, global_max=%.6f", global_min, global_max)

        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self.scale = self.global_max - self.global_min
        if self.scale < 1e-8:
            logger.warning("global_max ≈ global_min (%.6f); normalisation is identity", self.global_min)
            self.scale = 1.0

        # Per-worker file handle (lazy-opened in __getitem__)
        self._h5_file: Optional[h5py.File] = None

        logger.info(
            "H5VolumeDataset: split='%s', n=%d, shape=%s, "
            "min=%.4f, max=%.4f",
            split,
            self.n_samples,
            self.volume_shape,
            self.global_min,
            self.global_max,
        )

    def _compute_global_stats(self) -> tuple[float, float]:
        """Stream through HDF5 to find global min/max without loading all data."""
        g_min = float("inf")
        g_max = float("-inf")
        with h5py.File(self.h5_path, "r") as f:
            dset = f[self.split]
            for i in range(self.n_samples):
                chunk = dset[i]  # (1, D, H, W) numpy
                g_min = min(g_min, float(np.min(chunk)))
                g_max = max(g_max, float(np.max(chunk)))
        return g_min, g_max

    def _get_h5(self) -> h5py.File:
        """Lazy-open the HDF5 file (one handle per DataLoader worker)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5 = self._get_h5()
        vol = h5[self.split][idx].astype(np.float32)  # (1, D, H, W)

        # Global min-max normalisation to [0, 1]
        vol = (vol - self.global_min) / self.scale
        vol = np.clip(vol, 0.0, 1.0)

        return {"images": torch.from_numpy(vol)}

    def __del__(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
