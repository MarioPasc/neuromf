"""Per-channel latent statistics via sum-of-powers accumulation.

Computes mean, std, skewness, kurtosis, min, max per channel, and
cross-channel Pearson correlation from a directory of ``.pt`` latent files.
Single-pass, constant-memory implementation suitable for large datasets.
Uses float64 accumulators — numerically stable for latent values in [-10, 10].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LatentStatsAccumulator:
    """Online accumulator for per-channel latent statistics.

    Accumulates sum-of-powers (S1..S4), min, max, and cross-channel
    products in a single pass over all latent files.

    Args:
        n_channels: Number of latent channels.
    """

    def __init__(self, n_channels: int = 4) -> None:
        self.n_channels = n_channels
        self.n_voxels = 0

        # Per-channel power sums (float64 for precision)
        self.s1 = np.zeros(n_channels, dtype=np.float64)  # sum(x)
        self.s2 = np.zeros(n_channels, dtype=np.float64)  # sum(x^2)
        self.s3 = np.zeros(n_channels, dtype=np.float64)  # sum(x^3)
        self.s4 = np.zeros(n_channels, dtype=np.float64)  # sum(x^4)
        self.ch_min = np.full(n_channels, np.inf, dtype=np.float64)
        self.ch_max = np.full(n_channels, -np.inf, dtype=np.float64)

        # Cross-channel product: sum(z_i * z_j) for correlation
        self.cross_sum = np.zeros((n_channels, n_channels), dtype=np.float64)

    def update(self, z: torch.Tensor) -> None:
        """Update statistics with a new latent tensor.

        Args:
            z: Latent tensor of shape ``(C, D, H, W)`` (no batch dim).
        """
        assert z.ndim == 4, f"Expected 4D (C,D,H,W), got {z.ndim}D"
        assert z.shape[0] == self.n_channels, (
            f"Expected {self.n_channels} channels, got {z.shape[0]}"
        )

        z_np = z.cpu().to(torch.float64).numpy()
        n_spatial = z_np[0].size  # D * H * W

        flat_channels = z_np.reshape(self.n_channels, -1)  # (C, D*H*W)

        for c in range(self.n_channels):
            flat = flat_channels[c]
            self.s1[c] += flat.sum()
            self.s2[c] += (flat * flat).sum()
            self.s3[c] += (flat * flat * flat).sum()
            self.s4[c] += (flat * flat * flat * flat).sum()
            self.ch_min[c] = min(self.ch_min[c], flat.min())
            self.ch_max[c] = max(self.ch_max[c], flat.max())

        # Cross-channel outer product for correlation
        self.cross_sum += flat_channels @ flat_channels.T

        self.n_voxels += n_spatial

    def finalize(self) -> dict:
        """Compute final statistics from accumulated sums.

        Returns:
            Dict with ``per_channel`` and ``cross_channel_correlation`` keys.
        """
        if self.n_voxels == 0:
            raise ValueError("No data accumulated")

        n = self.n_voxels
        stats: dict = {
            "n_files": 0,  # caller sets this
            "n_voxels_per_channel": int(n),
            "per_channel": {},
            "cross_channel_correlation": [],
        }

        means = self.s1 / n

        for c in range(self.n_channels):
            mu = means[c]
            var = self.s2[c] / n - mu * mu
            std = np.sqrt(max(var, 0.0))

            if std > 1e-10:
                # Central moments from raw moments:
                # E[(x-mu)^3] = E[x^3] - 3*mu*E[x^2] + 2*mu^3
                m3 = self.s3[c] / n - 3 * mu * self.s2[c] / n + 2 * mu**3
                # E[(x-mu)^4] = E[x^4] - 4*mu*E[x^3] + 6*mu^2*E[x^2] - 3*mu^4
                m4 = (
                    self.s4[c] / n
                    - 4 * mu * self.s3[c] / n
                    + 6 * mu**2 * self.s2[c] / n
                    - 3 * mu**4
                )
                skewness = m3 / std**3
                kurtosis = m4 / std**4 - 3.0  # excess kurtosis
            else:
                skewness = 0.0
                kurtosis = 0.0

            stats["per_channel"][f"channel_{c}"] = {
                "mean": float(mu),
                "std": float(std),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "min": float(self.ch_min[c]),
                "max": float(self.ch_max[c]),
            }

        # Pearson correlation matrix
        stds = np.array(
            [stats["per_channel"][f"channel_{c}"]["std"] for c in range(self.n_channels)]
        )
        corr = np.zeros((self.n_channels, self.n_channels), dtype=np.float64)
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                cov_ij = self.cross_sum[i, j] / n - means[i] * means[j]
                if stds[i] > 1e-10 and stds[j] > 1e-10:
                    corr[i, j] = cov_ij / (stds[i] * stds[j])
                else:
                    corr[i, j] = 1.0 if i == j else 0.0

        stats["cross_channel_correlation"] = corr.tolist()
        return stats


def compute_latent_stats(latent_dir: Path) -> dict:
    """Compute per-channel statistics from all HDF5 latent shards.

    Single-pass, constant-memory computation of mean, std, skewness,
    kurtosis, min, max, and cross-channel Pearson correlation.

    Args:
        latent_dir: Directory containing ``.h5`` shard files.

    Returns:
        Dict with ``per_channel`` and ``cross_channel_correlation`` keys.
    """
    import h5py

    from neuromf.data.latent_hdf5 import discover_shards, get_written_mask, read_sample

    latent_dir = Path(latent_dir)
    shard_paths = discover_shards(latent_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No .h5 shard files found in {latent_dir}")

    # Count total samples
    total_samples = 0
    for sp in shard_paths:
        with h5py.File(str(sp), "r") as f:
            total_samples += int(get_written_mask(f).sum())

    logger.info(
        "Computing latent stats from %d samples across %d shards in %s",
        total_samples,
        len(shard_paths),
        latent_dir,
    )

    acc: LatentStatsAccumulator | None = None
    n_processed = 0

    for shard_path in shard_paths:
        with h5py.File(str(shard_path), "r") as f:
            mask = get_written_mask(f)
            for idx in np.where(mask)[0]:
                z, _ = read_sample(f, int(idx))
                if acc is None:
                    acc = LatentStatsAccumulator(n_channels=z.shape[0])
                acc.update(z)
                n_processed += 1
                if n_processed % 100 == 0:
                    logger.info("  Processed %d / %d samples", n_processed, total_samples)

    if acc is None:
        raise FileNotFoundError(f"No written samples found in shards at {latent_dir}")

    stats = acc.finalize()
    stats["n_files"] = total_samples
    logger.info(
        "Latent stats computed: %d samples, %d voxels/channel",
        total_samples,
        stats["n_voxels_per_channel"],
    )
    return stats


def save_latent_stats(stats: dict, output_path: Path) -> None:
    """Write latent statistics to a JSON file.

    Args:
        stats: Statistics dict from ``compute_latent_stats``.
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    logger.info("Latent stats saved to %s", output_path)


def load_latent_stats(stats_path: Path) -> dict:
    """Load latent statistics from a JSON file.

    Args:
        stats_path: Path to ``latent_stats.json``.

    Returns:
        Statistics dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Latent stats not found at {stats_path}")
    return json.loads(stats_path.read_text())
