"""Tests for sample evolution plots.

Uses synthetic archive data (small spatial dims) to verify that all 3 plot
functions produce PDF+PNG output without errors. No GPU required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neuromf.utils.sample_plots import (
    _radial_average_3d,
    _subsample_epochs,
    plot_channel_stats_evolution,
    plot_sample_evolution_grid,
    plot_spectral_evolution,
)


def _make_synthetic_archive(
    n_epochs: int = 4,
    n_samples: int = 2,
    spatial: int = 8,
    channels: int = 4,
) -> dict:
    """Build a minimal synthetic archive matching sample_archive.pt structure.

    Args:
        n_epochs: Number of logged epochs.
        n_samples: Number of samples per epoch.
        spatial: Spatial dimension of latent volumes.
        channels: Number of latent channels.

    Returns:
        Archive dict matching the SampleCollectorCallback format.
    """
    gen = torch.Generator().manual_seed(42)
    noise = torch.randn(n_samples, channels, spatial, spatial, spatial, generator=gen)

    archive: dict = {
        "metadata": {"noise_seed": 42, "nfe_steps": [1, 5], "n_samples": n_samples},
        "noise": noise,
        "latent_mean": torch.zeros(channels),
        "latent_std": torch.ones(channels),
        "epochs": [],
    }

    for i in range(n_epochs):
        epoch = i * 25
        archive["epochs"].append(epoch)
        # Simulate training: early epochs are noisy, later epochs have structure
        scale = 1.0 - 0.5 * (i / max(n_epochs - 1, 1))
        z_1 = scale * torch.randn(n_samples, channels, spatial, spatial, spatial)
        z_5 = z_1 + 0.1 * torch.randn_like(z_1)

        archive[f"epoch_{epoch:04d}"] = {
            "global_step": epoch * 10,
            "nfe_1": z_1,
            "nfe_5": z_5,
            "stats": {
                "nfe_1": {
                    "mean": [0.0] * channels,
                    "std": [scale] * channels,
                    "min": -3.0,
                    "max": 3.0,
                },
            },
            "nfe_consistency": {"mse_1vs5": 0.01 * (i + 1), "cosine_1vs5": 0.99},
        }

    return archive


# ======================================================================
# Unit tests for helpers
# ======================================================================


@pytest.mark.phase4
@pytest.mark.informational
class TestSamplePlotHelpers:
    """Tests for helper functions in sample_plots."""

    def test_subsample_epochs_passthrough(self) -> None:
        """Short epoch list passes through unchanged."""
        epochs = [0, 25, 50]
        result = _subsample_epochs(epochs, max_cols=12)
        assert result == epochs

    def test_subsample_epochs_reduces(self) -> None:
        """Long epoch list is reduced to max_cols."""
        epochs = list(range(0, 500, 10))  # 50 entries
        result = _subsample_epochs(epochs, max_cols=8)
        assert len(result) <= 8
        assert result[0] == epochs[0]
        assert result[-1] == epochs[-1]

    def test_radial_average_shape(self) -> None:
        """Radial average returns correct number of bins."""
        vol = np.random.randn(16, 16, 16)
        freqs, psd = _radial_average_3d(vol)
        assert freqs.shape == psd.shape
        assert len(freqs) == 8  # min(16,16,16) // 2
        assert (freqs >= 0).all()
        assert (freqs <= 0.5).all()

    def test_radial_average_white_noise_flat(self) -> None:
        """White noise should have approximately flat spectrum."""
        rng = np.random.default_rng(123)
        vol = rng.standard_normal((32, 32, 32))
        freqs, psd = _radial_average_3d(vol)

        # For white noise, PSD should be roughly constant
        # Check that max/min ratio is < 5 (generous bound)
        ratio = psd.max() / (psd.min() + 1e-30)
        assert ratio < 5.0, f"White noise PSD ratio {ratio:.1f} too large"


# ======================================================================
# Integration tests for plot functions
# ======================================================================


@pytest.mark.phase4
@pytest.mark.informational
class TestSamplePlots:
    """Integration tests verifying plot output files."""

    def test_sample_evolution_grid_outputs(self, tmp_path: Path) -> None:
        """plot_sample_evolution_grid produces PDF+PNG files."""
        archive = _make_synthetic_archive(n_epochs=3, spatial=8)
        plot_sample_evolution_grid(archive, tmp_path)

        assert (tmp_path / "sample_evolution_grid.pdf").exists()
        assert (tmp_path / "sample_evolution_grid.png").exists()
        assert (tmp_path / "sample_evolution_grid.png").stat().st_size > 0

    def test_channel_stats_evolution_outputs(self, tmp_path: Path) -> None:
        """plot_channel_stats_evolution produces PDF+PNG files."""
        archive = _make_synthetic_archive(n_epochs=5, spatial=8)
        plot_channel_stats_evolution(archive, tmp_path)

        assert (tmp_path / "channel_stats_evolution.pdf").exists()
        assert (tmp_path / "channel_stats_evolution.png").exists()
        assert (tmp_path / "channel_stats_evolution.png").stat().st_size > 0

    def test_spectral_evolution_outputs(self, tmp_path: Path) -> None:
        """plot_spectral_evolution produces PDF+PNG files."""
        archive = _make_synthetic_archive(n_epochs=4, spatial=16)
        plot_spectral_evolution(archive, tmp_path)

        assert (tmp_path / "spectral_evolution.pdf").exists()
        assert (tmp_path / "spectral_evolution.png").exists()
        assert (tmp_path / "spectral_evolution.png").stat().st_size > 0

    def test_empty_archive_no_crash(self, tmp_path: Path) -> None:
        """All plot functions handle empty archives gracefully."""
        archive: dict = {
            "metadata": {},
            "noise": torch.randn(2, 4, 8, 8, 8),
            "latent_mean": torch.zeros(4),
            "latent_std": torch.ones(4),
            "epochs": [],
        }
        # Should log warnings but not raise
        plot_sample_evolution_grid(archive, tmp_path)
        plot_channel_stats_evolution(archive, tmp_path)
        plot_spectral_evolution(archive, tmp_path)

        # No output files expected for empty archive
        assert not (tmp_path / "sample_evolution_grid.pdf").exists()

    def test_single_epoch_no_crash(self, tmp_path: Path) -> None:
        """All plot functions handle single-epoch archives."""
        archive = _make_synthetic_archive(n_epochs=1, spatial=8)
        plot_sample_evolution_grid(archive, tmp_path)
        plot_channel_stats_evolution(archive, tmp_path)
        plot_spectral_evolution(archive, tmp_path)

        assert (tmp_path / "sample_evolution_grid.pdf").exists()
        assert (tmp_path / "channel_stats_evolution.pdf").exists()
        assert (tmp_path / "spectral_evolution.pdf").exists()
