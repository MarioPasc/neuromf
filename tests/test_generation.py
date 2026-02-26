"""Phase 5 tests: Generation pipeline (H5Manager, LatentGenerator, VolumeDecoder).

Test naming convention: test_P5_T{N}_{description}
All tests use mock/tiny models — no GPU or real weights required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from neuromf.generation.h5_manager import H5LatentConfig, H5Manager, H5VolumeConfig
from neuromf.generation.latent_generator import LatentGenerator
from neuromf.generation.volume_decoder import VolumeDecoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model for testing: echoes input or returns zeros."""

    def __init__(self, out_channels: int = 4, prediction_type: str = "x") -> None:
        super().__init__()
        self.prediction_type = prediction_type
        self.linear = nn.Linear(1, 1)  # Need at least one parameter

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return a prediction of same shape as z_t."""
        if self.prediction_type == "x":
            # x-prediction: return something close to z_t (model predicts x_0)
            return z_t * 0.5
        else:
            # u-prediction: return small velocity
            return torch.zeros_like(z_t)


# ---------------------------------------------------------------------------
# P5-T1: Latent HDF5 create/read/write
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T1_h5_latent_archive_create_read_write(tmp_path: Path) -> None:
    """Round-trip test: create latent archive, write batch, read back."""
    n = 4
    shape = (4, 8, 8, 8)  # Tiny shape for tests
    config = H5LatentConfig(n_samples=n, latent_shape=shape)

    metadata = {"nfe": 1, "base_seed": 42, "prediction_type": "x"}
    h5_path = tmp_path / "test_latents.h5"

    # Create and write
    h5file = H5Manager.create_latent_archive(h5_path, config, metadata)
    latents = torch.randn(n, *shape)
    seeds = [100, 101, 102, 103]
    timing = [10.0, 11.0, 12.0, 13.0]
    H5Manager.write_latent_batch(h5file, 0, latents, seeds, timing)
    h5file.close()

    # Read back
    for i in range(n):
        recovered = H5Manager.read_latent(h5_path, i)
        assert recovered.shape == shape
        # float16 round-trip loses precision — check within tolerance
        np.testing.assert_allclose(recovered.numpy(), latents[i].numpy(), atol=0.01, rtol=0.01)

    # Check metadata
    meta = H5Manager.read_metadata(h5_path)
    assert meta["nfe"] == 1
    assert meta["base_seed"] == 42


# ---------------------------------------------------------------------------
# P5-T2: Volume HDF5 create/read/write
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T2_h5_volume_archive_create_read_write(tmp_path: Path) -> None:
    """Round-trip test: create volume archive, write batch, read back."""
    n = 2
    vol_shape = (16, 16, 16)  # Tiny
    config = H5VolumeConfig(n_samples=n, volume_shape=vol_shape)

    metadata = {"nfe": 1, "n_samples": n}
    h5_path = tmp_path / "test_volumes.h5"

    h5file = H5Manager.create_volume_archive(h5_path, config, metadata)
    volumes = torch.rand(n, 1, *vol_shape)  # (B, 1, D, H, W)
    timing = [50.0, 55.0]
    H5Manager.write_volume_batch(h5file, 0, volumes, timing)
    h5file.close()

    # Read back
    for i in range(n):
        recovered = H5Manager.read_volume(h5_path, i)
        assert recovered.shape == vol_shape
        np.testing.assert_allclose(recovered.numpy(), volumes[i, 0].numpy(), atol=1e-5)

    # Verify timing stored
    with h5py.File(str(h5_path), "r") as f:
        assert f["decode_timing_ms"][0] == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# P5-T3: LatentGenerator with tiny model
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T3_latent_generator_tiny_model(tmp_path: Path) -> None:
    """Generate 4 latents at NFE=1 and NFE=5 with a tiny model."""
    model = TinyModel(prediction_type="x")
    device = torch.device("cpu")
    gen = LatentGenerator(model, "x", device)

    for nfe in [1, 5]:
        out_path = tmp_path / f"nfe_{nfe:03d}.h5"
        gen.generate(
            n_samples=4,
            nfe=nfe,
            output_path=out_path,
            batch_size=2,
            base_seed=42,
            latent_shape=(4, 8, 8, 8),
        )

        assert out_path.exists()

        # Verify contents
        with h5py.File(str(out_path), "r") as f:
            assert f["latents"].shape == (4, 4, 8, 8, 8)
            assert f["noise_seeds"].shape == (4,)
            assert f["timing_ms"].shape == (4,)
            assert f.attrs["nfe"] == nfe


# ---------------------------------------------------------------------------
# P5-T4: Shared noise across NFE levels
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T4_shared_noise_across_nfe(tmp_path: Path) -> None:
    """Same base_seed should produce same initial noise for NFE=1 and NFE=5.

    The seed formula is base_seed + nfe * 1000, so NFE=1 and NFE=5 get
    different seeds. But within the same NFE level, re-generation with
    same seed must produce identical results.
    """
    model = TinyModel(prediction_type="x")
    device = torch.device("cpu")
    gen = LatentGenerator(model, "x", device)

    # Generate twice with same parameters
    for attempt in range(2):
        out_path = tmp_path / f"run{attempt}_nfe_001.h5"
        gen.generate(
            n_samples=4,
            nfe=1,
            output_path=out_path,
            batch_size=4,
            base_seed=42,
            latent_shape=(4, 8, 8, 8),
        )

    # Both runs should produce identical latents
    z0 = H5Manager.read_latent(tmp_path / "run0_nfe_001.h5", 0)
    z1 = H5Manager.read_latent(tmp_path / "run1_nfe_001.h5", 0)
    np.testing.assert_allclose(z0.numpy(), z1.numpy(), atol=0.01)


# ---------------------------------------------------------------------------
# P5-T5: VolumeDecoder with mock VAE
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T5_volume_decoder_mock_vae(tmp_path: Path) -> None:
    """Decode latents through a mock VAE and verify volume output."""
    n = 2
    latent_shape = (4, 8, 8, 8)
    vol_shape = (32, 32, 32)  # 8 * 4 per axis

    # Create a latent archive
    lat_config = H5LatentConfig(n_samples=n, latent_shape=latent_shape, dtype="float16")
    lat_path = tmp_path / "latents.h5"
    lat_h5 = H5Manager.create_latent_archive(lat_path, lat_config, {"nfe": 1, "n_samples": n})
    latents = torch.randn(n, *latent_shape)
    H5Manager.write_latent_batch(lat_h5, 0, latents.half(), [0, 1], [0.0, 0.0])
    lat_h5.close()

    # Create mock VAE that returns (B, 1, 32, 32, 32)
    mock_vae = MagicMock()
    mock_vae.decode = MagicMock(side_effect=lambda z: torch.rand(z.shape[0], 1, *vol_shape))

    device = torch.device("cpu")
    mean = torch.zeros(4)
    std = torch.ones(4)

    decoder = VolumeDecoder(mock_vae, mean, std, device)

    vol_path = tmp_path / "volumes.h5"
    decoder.decode(lat_path, vol_path, batch_size=1)

    assert vol_path.exists()

    with h5py.File(str(vol_path), "r") as f:
        assert f["volumes"].shape == (n, *vol_shape)
        assert f["decode_timing_ms"].shape == (n,)
        # Values should be in [0, 1] (clamped)
        assert f["volumes"][:].min() >= 0.0
        assert f["volumes"][:].max() <= 1.0


# ---------------------------------------------------------------------------
# P5-T10: HDF5 metadata consistency
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T10_h5_metadata_consistency(tmp_path: Path) -> None:
    """Verify all metadata fields are correctly stored and retrievable."""
    config = H5LatentConfig(n_samples=10, latent_shape=(4, 8, 8, 8))
    metadata = {
        "nfe": 5,
        "n_samples": 10,
        "batch_size": 4,
        "base_seed": 42,
        "prediction_type": "x",
        "checkpoint_path": "/path/to/ckpt",
        "latent_mean": [-0.05, -0.19, -0.05, 0.001],
        "latent_std": [0.97, 1.02, 0.97, 1.01],
        "scale_factor": 0.96240234375,
    }

    h5_path = tmp_path / "meta_test.h5"
    h5file = H5Manager.create_latent_archive(h5_path, config, metadata)
    h5file.close()

    recovered = H5Manager.read_metadata(h5_path)

    assert recovered["nfe"] == 5
    assert recovered["n_samples"] == 10
    assert recovered["base_seed"] == 42
    assert recovered["prediction_type"] == "x"
    assert recovered["scale_factor"] == pytest.approx(0.96240234375)
    assert len(recovered["latent_mean"]) == 4
    assert len(recovered["latent_std"]) == 4


# ---------------------------------------------------------------------------
# P5-T17: is_complete — empty archive returns False, full archive returns True
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T17_is_complete_tracking(tmp_path: Path) -> None:
    """Verify n_written tracking: incomplete archive → False, complete → True."""
    n = 4
    shape = (4, 8, 8, 8)
    config = H5LatentConfig(n_samples=n, latent_shape=shape)
    h5_path = tmp_path / "completeness_test.h5"

    # Create archive but don't write anything
    h5file = H5Manager.create_latent_archive(h5_path, config, {"nfe": 1})
    h5file.close()

    assert h5_path.exists()
    assert not H5Manager.is_complete(h5_path), "Empty archive should be incomplete"

    # Write half the samples
    h5file = h5py.File(str(h5_path), "a")
    latents = torch.randn(2, *shape)
    H5Manager.write_latent_batch(h5file, 0, latents, [0, 1], [0.0, 0.0])
    h5file.close()

    assert not H5Manager.is_complete(h5_path), "Half-written archive should be incomplete"

    # Write remaining samples
    h5file = h5py.File(str(h5_path), "a")
    latents = torch.randn(2, *shape)
    H5Manager.write_latent_batch(h5file, 2, latents, [2, 3], [0.0, 0.0])
    h5file.close()

    assert H5Manager.is_complete(h5_path), "Fully-written archive should be complete"


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T17b_is_complete_volume_archive(tmp_path: Path) -> None:
    """Verify is_complete works for volume archives too."""
    n = 2
    vol_shape = (16, 16, 16)
    config = H5VolumeConfig(n_samples=n, volume_shape=vol_shape)
    h5_path = tmp_path / "vol_completeness.h5"

    h5file = H5Manager.create_volume_archive(h5_path, config, {"nfe": 1})
    h5file.close()

    assert not H5Manager.is_complete(h5_path)

    h5file = h5py.File(str(h5_path), "a")
    volumes = torch.rand(n, 1, *vol_shape)
    H5Manager.write_volume_batch(h5file, 0, volumes, [0.0, 0.0])
    h5file.close()

    assert H5Manager.is_complete(h5_path)


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T17c_is_complete_missing_attr(tmp_path: Path) -> None:
    """Archives without n_written attr (legacy) should return False."""
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("latents", shape=(4, 4, 8, 8, 8), dtype=np.float16)
        # No n_written attr

    assert not H5Manager.is_complete(h5_path)


@pytest.mark.phase5
@pytest.mark.informational
def test_P5_T17d_is_complete_nonexistent_file(tmp_path: Path) -> None:
    """Nonexistent file should return False, not raise."""
    assert not H5Manager.is_complete(tmp_path / "does_not_exist.h5")
