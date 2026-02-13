"""Tests for toroid dataset (Phase 2).

P2-T1: Toroid dataset generates valid samples with correct shapes.
Updated for unnormalised torus (||z||=sqrt(2), pair norms=1.0) and ambient_dim support.
"""

import math

import pytest
import torch

from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T1ToroidDataset:
    """P2-T1: Toroid dataset generates valid samples."""

    def test_P2_T1_r4_shape(self) -> None:
        """R^4 mode produces (4,) vectors."""
        cfg = ToroidConfig(n_samples=100, mode="r4")
        ds = ToroidDataset(cfg)
        assert len(ds) == 100
        sample = ds[0]
        assert sample.shape == (4,)

    def test_P2_T1_r4_finite(self) -> None:
        """All R^4 samples are finite."""
        cfg = ToroidConfig(n_samples=1000, mode="r4")
        ds = ToroidDataset(cfg)
        for i in range(len(ds)):
            assert torch.isfinite(ds[i]).all(), f"Sample {i} has NaN/Inf"

    def test_P2_T1_r4_norm_constraint(self) -> None:
        """All R^4 samples have ||z||_2 = sqrt(2) (unnormalised torus)."""
        cfg = ToroidConfig(n_samples=1000, mode="r4")
        ds = ToroidDataset(cfg)
        target_norm = math.sqrt(2.0)
        for i in range(len(ds)):
            z = ds[i]
            norm = z.norm()
            assert abs(norm - target_norm) < 1e-5, (
                f"Sample {i}: norm={norm:.6f} != {target_norm:.6f}"
            )

    def test_P2_T1_r4_pairwise_norm(self) -> None:
        """For unnormalised flat torus: z1^2 + z2^2 = 1.0 and z3^2 + z4^2 = 1.0."""
        cfg = ToroidConfig(n_samples=1000, mode="r4")
        ds = ToroidDataset(cfg)
        for i in range(0, len(ds), 100):
            z = ds[i]
            pair1 = z[0] ** 2 + z[1] ** 2
            pair2 = z[2] ** 2 + z[3] ** 2
            assert abs(pair1 - 1.0) < 1e-5, f"pair1={pair1:.6f}"
            assert abs(pair2 - 1.0) < 1e-5, f"pair2={pair2:.6f}"

    def test_P2_T1_r4_angular_coverage(self) -> None:
        """Angles should cover [-pi, pi] range."""
        cfg = ToroidConfig(n_samples=10_000, mode="r4")
        ds = ToroidDataset(cfg)
        all_samples = ds.data  # (N, 4)
        theta1 = torch.atan2(all_samples[:, 1], all_samples[:, 0])
        theta2 = torch.atan2(all_samples[:, 3], all_samples[:, 2])
        assert theta1.min() < -2.5
        assert theta1.max() > 2.5
        assert theta2.min() < -2.5
        assert theta2.max() > 2.5

    def test_P2_T1_volumetric_shape(self) -> None:
        """Volumetric mode produces (4, 32, 32, 32) tensors."""
        cfg = ToroidConfig(n_samples=10, mode="volumetric", spatial_size=32, n_channels=4)
        ds = ToroidDataset(cfg)
        sample = ds[0]
        assert sample.shape == (4, 32, 32, 32)

    def test_P2_T1_volumetric_finite(self) -> None:
        """All volumetric samples are finite."""
        cfg = ToroidConfig(n_samples=10, mode="volumetric", spatial_size=16, n_channels=4)
        ds = ToroidDataset(cfg)
        for i in range(len(ds)):
            assert torch.isfinite(ds[i]).all(), f"Sample {i} has NaN/Inf"

    def test_P2_T1_ambient_dim_shape(self) -> None:
        """Ambient dim D=16 produces (16,) vectors."""
        cfg = ToroidConfig(n_samples=100, mode="r4", ambient_dim=16)
        ds = ToroidDataset(cfg)
        sample = ds[0]
        assert sample.shape == (16,)

    def test_P2_T1_ambient_dim_projection_roundtrip(self) -> None:
        """Project to R^D then back to R^4 should approximately recover original points."""
        cfg_4 = ToroidConfig(n_samples=100, mode="r4", ambient_dim=4)
        ds_4 = ToroidDataset(cfg_4, seed=42)

        cfg_16 = ToroidConfig(n_samples=100, mode="r4", ambient_dim=16)
        ds_16 = ToroidDataset(cfg_16, seed=42)

        # Project D=16 samples back to R^4
        recovered = ds_16.project_to_r4(ds_16.data)
        assert recovered.shape == (100, 4)

        # Should approximately match the D=4 dataset (same seed, same angles)
        max_err = (recovered - ds_4.data).abs().max()
        assert max_err < 1e-4, f"Projection roundtrip error: {max_err:.6f}"

    def test_P2_T1_ambient_dim_pairwise_distances_preserved(self) -> None:
        """Pairwise distances should be approximately preserved by orthogonal projection."""
        cfg_4 = ToroidConfig(n_samples=50, mode="r4", ambient_dim=4)
        ds_4 = ToroidDataset(cfg_4, seed=42)

        cfg_16 = ToroidConfig(n_samples=50, mode="r4", ambient_dim=16)
        ds_16 = ToroidDataset(cfg_16, seed=42)

        dists_4 = torch.cdist(ds_4.data, ds_4.data)
        dists_16 = torch.cdist(ds_16.data, ds_16.data)

        max_err = (dists_4 - dists_16).abs().max()
        assert max_err < 1e-2, f"Pairwise distance error: {max_err:.6f}"
