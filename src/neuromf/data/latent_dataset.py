"""PyTorch Dataset for pre-computed latents stored in HDF5 shard files.

Loads latent tensors lazily from per-dataset HDF5 shards with optional
per-channel normalisation using pre-computed statistics. Designed for
MeanFlow training where the VAE is frozen and all latents are pre-encoded.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from neuromf.data.latent_hdf5 import build_global_index, discover_shards, read_sample
from neuromf.utils.latent_stats import load_latent_stats

logger = logging.getLogger(__name__)


def _extract_subject_key(dataset: str, subject_id: str) -> str:
    """Build subject grouping key: ``'{dataset}_{subject_id}'``.

    Includes dataset name to avoid cross-dataset collisions
    (e.g. ``sub_001`` in IXI vs OASIS).

    Args:
        dataset: Dataset identifier, e.g. ``"PT005_IXI"``.
        subject_id: Subject identifier, e.g. ``"sub_3"``.

    Returns:
        Combined key string.
    """
    return f"{dataset}_{subject_id}"


def _build_subject_index(
    entries: list[tuple[Path, int]],
) -> dict[str, list[int]]:
    """Group entry indices by subject key.

    Opens each shard once to read the full ``subject_id`` array and
    ``dataset_name`` attribute, then maps each entry to its subject key.

    Args:
        entries: Global index from :func:`build_global_index` — list of
            ``(shard_path, local_idx)`` tuples.

    Returns:
        Dict mapping subject keys to lists of entry indices in ``entries``.
    """
    # Collect unique shard paths and which entry indices reference them
    shard_to_entries: dict[Path, list[tuple[int, int]]] = {}
    for entry_idx, (shard_path, local_idx) in enumerate(entries):
        shard_to_entries.setdefault(shard_path, []).append((entry_idx, local_idx))

    subject_index: dict[str, list[int]] = {}
    for shard_path, pairs in shard_to_entries.items():
        with h5py.File(str(shard_path), "r") as f:
            dataset_name = str(f.attrs["dataset_name"])
            subject_ids = f["subject_id"][:]
            for entry_idx, local_idx in pairs:
                raw_sid = subject_ids[local_idx]
                sid = raw_sid.decode() if isinstance(raw_sid, bytes) else str(raw_sid)
                key = _extract_subject_key(dataset_name, sid)
                subject_index.setdefault(key, []).append(entry_idx)

    return subject_index


def latent_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function that only stacks ``z`` tensors.

    Metadata contains strings that break the default collate. This
    function stacks only the latent tensors into a batch.

    Args:
        batch: List of dataset items, each with ``"z"`` and ``"metadata"``.

    Returns:
        Dict with ``"z"`` tensor of shape ``(B, C, D, H, W)``.
    """
    return {"z": torch.stack([item["z"] for item in batch])}


def hdf5_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init function that clears stale HDF5 handles.

    HDF5 file handles opened before ``fork()`` are unsafe in child
    processes. This function is passed to ``DataLoader(worker_init_fn=...)``
    to ensure each worker opens its own handles lazily on first access.

    Args:
        worker_id: Worker index (unused, required by DataLoader API).
    """
    # Nothing to do — LatentDataset._h5_handles starts empty and handles
    # are opened lazily in __getitem__. This function exists as a safety
    # hook for any future cleanup needs and as documentation of the pattern.
    pass


class LatentDataset(Dataset):
    """PyTorch Dataset backed by HDF5 latent shard files.

    Each shard is a ``.h5`` file containing latents for one FOMO-60K
    dataset. A flat global index maps dataset indices to
    ``(shard_path, local_idx)`` pairs.

    Args:
        latent_dir: Directory containing ``.h5`` shard files.
        normalise: If True, apply per-channel ``(z - mean) / std``
            normalisation using statistics from ``stats_path``.
        stats_path: Path to ``latent_stats.json``. Required if
            ``normalise=True``. If None, defaults to
            ``latent_dir / "latent_stats.json"``.
        split: If ``"train"`` or ``"val"``, return only the corresponding
            subset. If None, return all data.
        split_ratio: Fraction of data in the train split.
        split_seed: Random seed for deterministic splitting.
        transform: Optional callable applied to the ``z`` tensor after
            normalisation. Receives ``(C, D, H, W)`` tensor, must return
            same shape.
    """

    def __init__(
        self,
        latent_dir: Path,
        normalise: bool = True,
        stats_path: Path | None = None,
        split: str | None = None,
        split_ratio: float = 0.9,
        split_seed: int = 42,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.latent_dir = Path(latent_dir)
        self.normalise = normalise
        self.transform = transform

        # Discover shards and build global index
        shard_paths = discover_shards(self.latent_dir)
        if not shard_paths:
            raise FileNotFoundError(f"No .h5 shard files found in {self.latent_dir}")

        all_entries = build_global_index(shard_paths)
        if not all_entries:
            raise FileNotFoundError(f"No written samples found in shards at {self.latent_dir}")

        # Apply train/val split if requested — split by SUBJECT to avoid
        # data leakage (same subject's multiple scans in both train and val)
        if split is not None:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train', 'val', or None; got '{split}'")
            subject_to_indices = _build_subject_index(all_entries)
            subjects = sorted(subject_to_indices.keys())
            random.Random(split_seed).shuffle(subjects)
            n_train_subj = int(len(subjects) * split_ratio)
            if split == "train":
                chosen = set(subjects[:n_train_subj])
            else:
                chosen = set(subjects[n_train_subj:])
            selected = sorted(idx for subj in chosen for idx in subject_to_indices[subj])
            self._entries = [all_entries[i] for i in selected]
            logger.info(
                "LatentDataset: %d files from %d subjects (%s split, ratio=%.2f, seed=%d)",
                len(self._entries),
                len(chosen),
                split,
                split_ratio,
                split_seed,
            )
        else:
            self._entries = all_entries
            logger.info(
                "LatentDataset: %d samples from %d shards in %s",
                len(self._entries),
                len(shard_paths),
                self.latent_dir,
            )

        # Lazy per-worker HDF5 file handle cache
        self._h5_handles: dict[Path, h5py.File] = {}

        # Load normalisation stats if needed
        self._norm_mean: torch.Tensor | None = None
        self._norm_std: torch.Tensor | None = None

        if self.normalise:
            if stats_path is None:
                stats_path = self.latent_dir / "latent_stats.json"
            stats = load_latent_stats(stats_path)
            per_ch = stats["per_channel"]
            n_channels = len(per_ch)

            means = [per_ch[f"channel_{c}"]["mean"] for c in range(n_channels)]
            stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_channels)]

            # Shape (C, 1, 1, 1) for broadcasting over (C, D, H, W)
            self._norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1, 1)
            self._norm_std = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1, 1)
            logger.info(
                "LatentDataset normalisation: mean=%s, std=%s",
                [f"{m:.4f}" for m in means],
                [f"{s:.4f}" for s in stds],
            )

    @property
    def norm_mean(self) -> torch.Tensor | None:
        """Per-channel mean used for normalisation, shape ``(C, 1, 1, 1)``."""
        return self._norm_mean

    @property
    def norm_std(self) -> torch.Tensor | None:
        """Per-channel std used for normalisation, shape ``(C, 1, 1, 1)``."""
        return self._norm_std

    def _get_h5_handle(self, shard_path: Path) -> h5py.File:
        """Get or lazily open an HDF5 file handle for a shard.

        Handles are cached per shard path and opened in read-only mode.
        In multi-worker DataLoaders, each worker process opens its own
        handles after fork (handles are NOT shared across processes).

        Args:
            shard_path: Path to the ``.h5`` shard file.

        Returns:
            Open ``h5py.File`` in read-only mode.
        """
        if shard_path not in self._h5_handles:
            self._h5_handles[shard_path] = h5py.File(str(shard_path), "r")
        return self._h5_handles[shard_path]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single latent from HDF5 and optionally normalise.

        Args:
            idx: Dataset index.

        Returns:
            Dict with ``"z"`` (float32 tensor of shape ``(C, D, H, W)``)
            and ``"metadata"`` dict.
        """
        shard_path, local_idx = self._entries[idx]
        h5f = self._get_h5_handle(shard_path)
        z, metadata = read_sample(h5f, local_idx)
        z = z.float()

        if self.normalise and self._norm_mean is not None and self._norm_std is not None:
            z = (z - self._norm_mean) / self._norm_std

        if self.transform is not None:
            z = self.transform(z)

        return {"z": z, "metadata": metadata}

    def __del__(self) -> None:
        """Close all cached HDF5 file handles."""
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles.clear()
