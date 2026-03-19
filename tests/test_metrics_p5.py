"""Phase 5 tests: Evaluation metrics (spectral, MS-SSIM, pairing, feature_extractor, SynthSeg).

Test naming convention: test_P5_T{N}_{description}
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from neuromf.metrics.ms_ssim_3d import compute_ms_ssim_3d
from neuromf.metrics.pairing import compute_nn_pairs
from neuromf.metrics.spectral import compute_hf_energy_ratio
from neuromf.metrics.synthseg_metrics import (
    SynthSegConfig,
    _h5_volumes_to_nifti,
    check_synthseg_available,
    compute_dice_from_labels,
    compute_regional_correlation,
    compute_regional_kl,
    compute_success_rate,
    consolidate_nifti_to_h5,
    parse_volumes_csv,
    validate_volumes_csv,
)

# Keep linter from removing unused imports used in tests below
_ = (validate_volumes_csv, consolidate_nifti_to_h5)

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

    # Mock the R3D-18 loading and extraction (now returns 512-d features)
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.return_value = torch.randn(1, 512)

    def mock_extract_3d(volumes, model, normalize=True, batch_size=8):
        B = volumes.shape[0]
        return torch.randn(B, 512)

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

        assert features.shape == (n, 512)
        assert feat_path.exists()

        # Verify load_cached works
        loaded = FeatureExtractor.load_cached(feat_path)
        assert loaded.shape == (n, 512)
        np.testing.assert_allclose(features.numpy(), loaded.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# P5-T11: SynthSeg — HDF5 to NIfTI conversion
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T11_h5_to_nifti_conversion(tmp_path: Path) -> None:
    """Verify HDF5 volumes are correctly converted to NIfTI files."""
    n, D, H, W = 3, 16, 16, 16
    h5_path = tmp_path / "volumes.h5"
    with h5py.File(str(h5_path), "w") as f:
        data = np.random.rand(n, D, H, W).astype(np.float32)
        f.create_dataset("volumes", data=data)

    nifti_dir = tmp_path / "nifti"
    paths = _h5_volumes_to_nifti(h5_path, nifti_dir, voxel_size_mm=1.0)

    assert len(paths) == n
    for p in paths:
        assert p.exists()
        assert p.suffix == ".gz"

    # Verify round-trip: NIfTI data matches original
    nii = nib.load(str(paths[0]))
    nii_data = np.asarray(nii.dataobj, dtype=np.float32)
    np.testing.assert_allclose(nii_data, data[0], atol=1e-6)

    # Verify affine encodes 1mm isotropic
    np.testing.assert_allclose(np.diag(nii.affine), [1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# P5-T12: SynthSeg — Dice computation from label maps
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T12_dice_from_labels(tmp_path: Path) -> None:
    """Verify Dice computation between identical and different label maps."""
    shape = (16, 16, 16)
    affine = np.eye(4)

    # Create identical label maps
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:8, 2:8, 2:8] = 17  # Left-Hippocampus
    labels[8:14, 8:14, 8:14] = 53  # Right-Hippocampus

    real_path = tmp_path / "real_labels.nii.gz"
    gen_path_same = tmp_path / "gen_labels_same.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(real_path))
    nib.save(nib.Nifti1Image(labels, affine), str(gen_path_same))

    dices_same = compute_dice_from_labels(real_path, gen_path_same)
    assert dices_same["Left-Hippocampus"] == 1.0
    assert dices_same["Right-Hippocampus"] == 1.0
    assert dices_same["mean"] == pytest.approx(1.0, abs=0.01)

    # Create different label maps (shifted)
    labels_shifted = np.zeros(shape, dtype=np.int32)
    labels_shifted[4:10, 4:10, 4:10] = 17  # Shifted hippocampus
    gen_path_diff = tmp_path / "gen_labels_diff.nii.gz"
    nib.save(nib.Nifti1Image(labels_shifted, affine), str(gen_path_diff))

    dices_diff = compute_dice_from_labels(real_path, gen_path_diff)
    assert dices_diff["Left-Hippocampus"] < 1.0
    assert dices_diff["Left-Hippocampus"] > 0.0


# ---------------------------------------------------------------------------
# P5-T13: SynthSeg — Success rate computation
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T13_synthseg_success_rate(tmp_path: Path) -> None:
    """Verify success rate counts output files correctly."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create 4 input files, 3 output files (1 missing = 75% success)
    for i in range(4):
        (input_dir / f"vol_{i:04d}.nii.gz").write_bytes(b"dummy")
    for i in range(3):
        (output_dir / f"vol_{i:04d}.nii.gz").write_bytes(b"dummy")

    rate = compute_success_rate(input_dir, output_dir)
    assert rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# P5-T14: SynthSeg — Regional volume correlation and KL
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T14_regional_correlation_and_kl() -> None:
    """Verify regional correlation and KL with synthetic volume data."""
    np.random.seed(42)

    # Create correlated real/gen volume measurements (100 samples for stable histograms)
    real_volumes = {}
    gen_volumes = {}
    for i in range(100):
        base_hippo = 3000 + np.random.normal(0, 200)
        real_volumes[f"real_{i:03d}"] = {"Left-Hippocampus": base_hippo}
        gen_volumes[f"gen_{i:03d}"] = {"Left-Hippocampus": base_hippo + np.random.normal(0, 50)}

    corr = compute_regional_correlation(real_volumes, gen_volumes, regions=["Left-Hippocampus"])
    assert "Left-Hippocampus" in corr
    # Correlated data should have positive Pearson r
    assert corr["Left-Hippocampus"]["pearson_r"] > 0.5

    kl = compute_regional_kl(real_volumes, gen_volumes, regions=["Left-Hippocampus"], n_bins=20)
    assert "Left-Hippocampus" in kl
    # KL should be small for similar distributions
    assert kl["Left-Hippocampus"] < 2.0
    assert kl["Left-Hippocampus"] >= 0.0


# ---------------------------------------------------------------------------
# P5-T15: SynthSeg — Volumes CSV parsing
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T15_parse_volumes_csv(tmp_path: Path) -> None:
    """Verify parsing of SynthSeg volumes CSV output."""
    csv_path = tmp_path / "volumes.csv"
    csv_path.write_text(
        "subject,Left-Hippocampus,Right-Hippocampus,Left-Lateral-Ventricle\n"
        "vol_0000.nii.gz,3200.5,3150.2,8500.1\n"
        "vol_0001.nii.gz,3100.0,3050.8,9200.3\n"
    )

    volumes = parse_volumes_csv(csv_path)
    assert len(volumes) == 2
    assert volumes["vol_0000.nii.gz"]["Left-Hippocampus"] == pytest.approx(3200.5)
    assert volumes["vol_0001.nii.gz"]["Right-Hippocampus"] == pytest.approx(3050.8)


# ---------------------------------------------------------------------------
# P5-T16: SynthSeg — Config and availability check
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T16_synthseg_config_and_availability() -> None:
    """Verify SynthSegConfig defaults and availability check."""
    config = SynthSegConfig()
    assert config.command == ["mri_synthseg"]
    assert config.robust is True
    assert len(config.regions_of_interest) == 10

    # Custom command
    config_custom = SynthSegConfig(command=["/nonexistent/python", "/nonexistent/script.py"])
    assert not check_synthseg_available(config_custom)

    # A binary that definitely exists
    config_ls = SynthSegConfig(command=["ls"])
    assert check_synthseg_available(config_ls)


# ---------------------------------------------------------------------------
# P5-T18: validate_volumes_csv — all-zero detection
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T18_validate_volumes_csv_all_zeros() -> None:
    """All-zero CSV should be flagged as invalid."""
    volumes = {
        "vol_0000.nii.gz": {"Left-Hippocampus": 0.0, "Right-Hippocampus": 0.0},
        "vol_0001.nii.gz": {"Left-Hippocampus": 0.0, "Right-Hippocampus": 0.0},
    }
    valid, msg = validate_volumes_csv(volumes)
    assert not valid
    assert "all-zero" in msg.lower()


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T18b_validate_volumes_csv_valid_data() -> None:
    """Non-zero CSV should pass validation."""
    volumes = {
        "vol_0000.nii.gz": {"Left-Hippocampus": 3200.5, "Right-Hippocampus": 3100.0},
        "vol_0001.nii.gz": {"Left-Hippocampus": 3300.1, "Right-Hippocampus": 0.0},
    }
    valid, msg = validate_volumes_csv(volumes)
    assert valid
    assert msg == "OK"


@pytest.mark.phase5
@pytest.mark.informational
def test_P5_T18c_validate_volumes_csv_empty() -> None:
    """Empty CSV should be flagged as invalid."""
    valid, msg = validate_volumes_csv({})
    assert not valid
    assert "empty" in msg.lower()


# ---------------------------------------------------------------------------
# P5-T19: KL divergence zero-data guard
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T19_kl_zero_data_guard() -> None:
    """All-zero regional volumes should produce NaN KL, not crash."""
    real_volumes = {f"r_{i}": {"Left-Hippocampus": 0.0} for i in range(10)}
    gen_volumes = {f"g_{i}": {"Left-Hippocampus": 0.0} for i in range(10)}

    kl = compute_regional_kl(real_volumes, gen_volumes, regions=["Left-Hippocampus"])
    assert "Left-Hippocampus" in kl
    assert np.isnan(kl["Left-Hippocampus"]), "Zero-range data should produce NaN KL"


# ---------------------------------------------------------------------------
# P5-T20: NIfTI → HDF5 consolidation
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T20_consolidate_nifti_to_h5(tmp_path: Path) -> None:
    """Consolidate NIfTI files to HDF5, verify contents, and delete originals."""
    import h5py

    nifti_dir = tmp_path / "labels"
    nifti_dir.mkdir()

    shape = (16, 16, 16)
    affine = np.eye(4)
    n = 3

    # Create mock NIfTI label maps
    for i in range(n):
        labels = np.zeros(shape, dtype=np.int32)
        labels[i : i + 4, :, :] = 17
        nib.save(nib.Nifti1Image(labels, affine), str(nifti_dir / f"vol_{i:04d}.nii.gz"))

    output_h5 = tmp_path / "labels.h5"
    consolidate_nifti_to_h5(nifti_dir, output_h5, delete_nifti=True)

    # Verify HDF5 contents
    assert output_h5.exists()
    with h5py.File(str(output_h5), "r") as f:
        assert f["data"].shape == (n, *shape)
        assert f["data"].dtype == np.int16  # integer label maps
        assert len(f["filenames"]) == n
        assert f.attrs["n_files"] == n

        # Verify data round-trip for first volume
        expected = np.zeros(shape, dtype=np.int16)
        expected[0:4, :, :] = 17
        np.testing.assert_array_equal(f["data"][0], expected)

    # Verify NIfTI directory was deleted
    assert not nifti_dir.exists()


@pytest.mark.phase5
@pytest.mark.informational
def test_P5_T20b_consolidate_nifti_no_delete(tmp_path: Path) -> None:
    """Consolidate without deleting originals."""
    nifti_dir = tmp_path / "vols"
    nifti_dir.mkdir()

    shape = (8, 8, 8)
    data = np.random.rand(*shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_dir / "vol_0000.nii.gz"))

    output_h5 = tmp_path / "vols.h5"
    consolidate_nifti_to_h5(nifti_dir, output_h5, delete_nifti=False)

    assert output_h5.exists()
    assert nifti_dir.exists()  # Should NOT be deleted
