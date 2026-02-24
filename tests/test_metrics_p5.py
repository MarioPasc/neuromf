"""Phase 5 tests: Evaluation metrics (spectral, MS-SSIM, pairing, feature_extractor).

Test naming convention: test_P5_T{N}_{description}
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from neuromf.metrics.ms_ssim_3d import compute_ms_ssim_3d
from neuromf.metrics.pairing import compute_nn_pairs
from neuromf.metrics.spectral import compute_hf_energy_ratio

# ---------------------------------------------------------------------------
# P5-T6: Spectral HF energy ratio — known signals
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T6_spectral_hf_ratio_known_signals() -> None:
    """Low-freq signal should have low HF ratio; white noise should have high."""
    # Low-frequency signal: a smooth Gaussian blob
    D, H, W = 32, 32, 32
    coords = torch.linspace(-1, 1, D)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    low_freq = torch.exp(-(x**2 + y**2 + z**2) / 0.5)

    # White noise
    torch.manual_seed(42)
    white_noise = torch.randn(D, H, W)

    hf_low = compute_hf_energy_ratio(low_freq, cutoff_fraction=0.5)
    hf_noise = compute_hf_energy_ratio(white_noise, cutoff_fraction=0.5)

    # Smooth signal: most energy at low frequencies
    assert hf_low < 0.3, f"Low-freq signal HF ratio too high: {hf_low}"
    # White noise: energy spread across all frequencies
    assert hf_noise > 0.3, f"White noise HF ratio too low: {hf_noise}"
    # Noise should have more HF energy than smooth signal
    assert hf_noise > hf_low

    # Test 5D input shape
    hf_5d = compute_hf_energy_ratio(low_freq.unsqueeze(0).unsqueeze(0))
    assert abs(hf_5d - hf_low) < 1e-5


# ---------------------------------------------------------------------------
# P5-T7: MS-SSIM 3D — identical volumes
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T7_ms_ssim_3d_identical() -> None:
    """Identical volumes should produce MS-SSIM close to 1.0."""
    torch.manual_seed(42)
    # Need minimum 48x48x48 for 3-level MS-SSIM (48 -> 24 -> 12, all >= 11)
    vol = torch.rand(1, 1, 48, 48, 48)

    ms_ssim = compute_ms_ssim_3d(vol, vol)
    assert ms_ssim > 0.99, f"MS-SSIM for identical volumes: {ms_ssim}"


@pytest.mark.phase5
@pytest.mark.informational
def test_P5_T7b_ms_ssim_3d_different_volumes() -> None:
    """Different volumes should produce MS-SSIM < 1.0."""
    torch.manual_seed(42)
    vol_a = torch.rand(1, 1, 48, 48, 48)
    vol_b = torch.rand(1, 1, 48, 48, 48)

    ms_ssim = compute_ms_ssim_3d(vol_a, vol_b)
    assert ms_ssim < 0.99, f"MS-SSIM for different volumes too high: {ms_ssim}"
    assert ms_ssim > 0.0, f"MS-SSIM should be non-negative: {ms_ssim}"


# ---------------------------------------------------------------------------
# P5-T8: NN pairing correctness
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T8_nn_pairing_correctness() -> None:
    """Verify NN pairing returns correct nearest neighbour indices."""
    # Construct features where the answer is known
    real_features = torch.tensor(
        [
            [0.0, 0.0],  # idx 0
            [10.0, 0.0],  # idx 1
            [0.0, 10.0],  # idx 2
        ]
    )

    gen_features = torch.tensor(
        [
            [0.1, 0.1],  # nearest to real[0]
            [9.9, 0.1],  # nearest to real[1]
            [0.1, 9.9],  # nearest to real[2]
            [5.0, 5.0],  # equidistant-ish, should pick one consistently
        ]
    )

    nn_indices = compute_nn_pairs(real_features, gen_features)

    assert nn_indices.shape == (4,)
    assert nn_indices[0].item() == 0  # closest to [0, 0]
    assert nn_indices[1].item() == 1  # closest to [10, 0]
    assert nn_indices[2].item() == 2  # closest to [0, 10]


# ---------------------------------------------------------------------------
# P5-T9: Feature extractor mock
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T9_feature_extractor_mock(tmp_path: Path) -> None:
    """Extract features with a mock backbone and verify caching."""
    from neuromf.metrics.feature_extractor import FeatureExtractor

    n = 3
    vol_shape = (32, 32, 32)

    # Create a mock volume archive
    vol_h5_path = tmp_path / "volumes.h5"
    with h5py.File(str(vol_h5_path), "w") as f:
        f.create_dataset("volumes", data=np.random.rand(n, *vol_shape).astype(np.float32))

    # Mock the Med3D loading and extraction
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])

    def mock_extract_3d(vol, model, normalize=True):
        return torch.randn(1, 2048)

    with (
        patch("neuromf.metrics.feature_extractor.FeatureExtractor.__init__", return_value=None),
        patch("neuromf.metrics.fid_3d.extract_3d_features", side_effect=mock_extract_3d),
    ):
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.backend = "med3d"
        extractor.device = torch.device("cpu")
        extractor.model = mock_model
        extractor.center_slices_ratio = 0.6

        feat_path = tmp_path / "features.h5"
        features = extractor.extract_and_cache(vol_h5_path, feat_path)

        assert features.shape == (n, 2048)
        assert feat_path.exists()

        # Verify load_cached works
        loaded = FeatureExtractor.load_cached(feat_path)
        assert loaded.shape == (n, 2048)
        np.testing.assert_allclose(features.numpy(), loaded.numpy(), atol=1e-5)
