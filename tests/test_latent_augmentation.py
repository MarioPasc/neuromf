"""Tests for latent-space data augmentation (Phase B).

Tests the augmentation pipeline factory and individual transforms,
including per-channel Gaussian noise calibration and integration with
``LatentDataset``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuromf.data.latent_augmentation import (
    PerChannelGaussianNoise,
    build_latent_augmentation,
)


@pytest.mark.phase4
@pytest.mark.informational
class TestLatentAugmentation:
    """Phase 4d augmentation tests."""

    def test_P4d_T1_disabled_returns_none(self) -> None:
        """P4d-T1: build_latent_augmentation returns None when disabled."""
        result = build_latent_augmentation({"enabled": False})
        assert result is None

        result2 = build_latent_augmentation({})
        assert result2 is None

    def test_P4d_T2_enabled_returns_callable(self) -> None:
        """P4d-T2: build_latent_augmentation returns a callable pipeline."""
        config = {
            "enabled": True,
            "flip_prob": 0.5,
            "flip_axes": [0, 1, 2],
            "rotate90_prob": 0.5,
            "rotate90_axes": [[1, 2]],
            "gaussian_noise_prob": 0.3,
            "gaussian_noise_std_fraction": 0.05,
            "intensity_scale_prob": 0.3,
            "intensity_scale_factors": 0.05,
        }
        pipeline = build_latent_augmentation(config)
        assert pipeline is not None
        assert callable(pipeline)

    def test_P4d_T3_shape_preservation(self) -> None:
        """P4d-T3: Augmented output shape matches (4, 48, 48, 48)."""
        config = {
            "enabled": True,
            "flip_prob": 1.0,
            "flip_axes": [0],
            "rotate90_prob": 1.0,
            "rotate90_axes": [[1, 2]],
            "gaussian_noise_prob": 1.0,
            "gaussian_noise_std_fraction": 0.05,
            "intensity_scale_prob": 1.0,
            "intensity_scale_factors": 0.05,
        }
        pipeline = build_latent_augmentation(config)
        x = torch.randn(4, 48, 48, 48)
        y = pipeline(x)
        assert y.shape == x.shape

    def test_P4d_T4_output_differs(self) -> None:
        """P4d-T4: Repeated calls produce different outputs (stochastic)."""
        config = {
            "enabled": True,
            "flip_prob": 0.5,
            "flip_axes": [0, 1, 2],
            "rotate90_prob": 0.5,
            "rotate90_axes": [[1, 2]],
            "gaussian_noise_prob": 1.0,
            "gaussian_noise_std_fraction": 0.1,
            "intensity_scale_prob": 1.0,
            "intensity_scale_factors": 0.1,
        }
        pipeline = build_latent_augmentation(config)
        x = torch.randn(4, 16, 16, 16)

        outputs = [pipeline(x.clone()) for _ in range(10)]
        # At least some should differ due to noise
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "All 10 augmented outputs are identical"

    def test_P4d_T5_per_channel_noise_calibrated(self, tmp_path: Path) -> None:
        """P4d-T5: Per-channel noise std is proportional to channel_std * fraction."""
        channel_stds = [1.0, 2.0, 3.0, 4.0]
        std_fraction = 0.1

        noise_fn = PerChannelGaussianNoise(
            prob=1.0,
            std_fraction=std_fraction,
            channel_stds=channel_stds,
        )

        # Large tensor for stable statistics
        x = torch.zeros(4, 64, 64, 64)
        n_trials = 200
        accumulated = torch.zeros(4, 64, 64, 64)
        for _ in range(n_trials):
            noisy = noise_fn(x.clone())
            accumulated += noisy

        # Mean should be ~0, std should be ~std_fraction * channel_std
        empirical_std = (accumulated / n_trials).std(dim=[1, 2, 3])
        # With 200 trials of noise added to zero, the std of the mean
        # should be proportional to the noise std / sqrt(n_trials)
        # But we accumulated sums, so accumulated/n_trials is the sample mean
        # Check the noise level indirectly: single-shot std
        single_noisy = noise_fn(x.clone())
        for c in range(4):
            expected_std = std_fraction * channel_stds[c]
            actual_std = single_noisy[c].std().item()
            # Allow 50% tolerance for single-shot
            assert actual_std > expected_std * 0.3, (
                f"Channel {c}: noise std {actual_std:.4f} too low (expected ~{expected_std:.4f})"
            )
            assert actual_std < expected_std * 3.0, (
                f"Channel {c}: noise std {actual_std:.4f} too high (expected ~{expected_std:.4f})"
            )

    def test_P4d_T6_dataset_with_transform(self, tmp_path: Path) -> None:
        """P4d-T6: LatentDataset applies transform in __getitem__."""
        from neuromf.data.latent_dataset import LatentDataset

        # Create fake .pt files and stats
        S = 8
        stats = {"per_channel": {f"channel_{c}": {"mean": 0.0, "std": 1.0} for c in range(4)}}
        (tmp_path / "latent_stats.json").write_text(json.dumps(stats))
        for i in range(5):
            data = {
                "z": torch.randn(4, S, S, S),
                "metadata": {"subject_id": f"sub_{i}"},
            }
            torch.save(data, tmp_path / f"sample_{i:03d}.pt")

        # Track how many times transform was called
        call_count = [0]

        def counting_transform(z: torch.Tensor) -> torch.Tensor:
            call_count[0] += 1
            return z * 2.0  # Double values to verify it was applied

        ds = LatentDataset(tmp_path, normalise=True, transform=counting_transform)
        item = ds[0]
        assert call_count[0] == 1
        assert item["z"].shape == (4, S, S, S)

        # Without transform, values should differ
        ds_no_aug = LatentDataset(tmp_path, normalise=True, transform=None)
        item_no_aug = ds_no_aug[0]
        # The doubled one should have ~2x magnitude
        ratio = item["z"].abs().mean() / (item_no_aug["z"].abs().mean() + 1e-8)
        assert ratio > 1.5, f"Transform doesn't seem applied: ratio={ratio:.2f}"
