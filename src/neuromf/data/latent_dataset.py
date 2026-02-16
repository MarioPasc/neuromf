"""PyTorch Dataset for pre-computed ``.pt`` latent representations.

Loads latent files lazily from disk with optional per-channel normalisation
using pre-computed statistics (mean, std). Designed for MeanFlow training
where the VAE is frozen and all latents are pre-encoded.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from neuromf.utils.latent_stats import load_latent_stats

logger = logging.getLogger(__name__)


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


class LatentDataset(Dataset):
    """PyTorch Dataset of pre-computed ``.pt`` latents with optional normalisation.

    Each ``.pt`` file is expected to contain a dict with at least
    ``{"z": Tensor(4, 48, 48, 48), "metadata": {...}}``.

    Args:
        latent_dir: Directory containing ``.pt`` latent files.
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

        # Glob and sort .pt files for reproducible indexing
        all_files = sorted(self.latent_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {self.latent_dir}")

        # Apply train/val split if requested
        if split is not None:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train', 'val', or None; got '{split}'")
            indices = list(range(len(all_files)))
            random.Random(split_seed).shuffle(indices)
            n_train = int(len(indices) * split_ratio)
            if split == "train":
                selected = sorted(indices[:n_train])
            else:
                selected = sorted(indices[n_train:])
            self.file_paths = [all_files[i] for i in selected]
            logger.info(
                "LatentDataset: %d/%d files (%s split, ratio=%.2f, seed=%d)",
                len(self.file_paths),
                len(all_files),
                split,
                split_ratio,
                split_seed,
            )
        else:
            self.file_paths = all_files
            logger.info("LatentDataset: %d files from %s", len(self.file_paths), self.latent_dir)

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

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single latent file and optionally normalise.

        Args:
            idx: Dataset index.

        Returns:
            Dict with ``"z"`` (float32 tensor of shape ``(C, D, H, W)``)
            and ``"metadata"`` dict.
        """
        data = torch.load(self.file_paths[idx], map_location="cpu", weights_only=True)
        z = data["z"].float()
        metadata = data.get("metadata", {})

        if self.normalise and self._norm_mean is not None and self._norm_std is not None:
            z = (z - self._norm_mean) / self._norm_std

        if self.transform is not None:
            z = self.transform(z)

        return {"z": z, "metadata": metadata}
