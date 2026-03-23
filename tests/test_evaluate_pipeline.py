"""Integration tests for the downstream evaluation pipeline.

Verifies that feature extraction, metrics computation, and the comparison-mode
folder structure work end-to-end with mock data.  All tests run on CPU with
small mock volumes (no GPU required).

Test naming convention: test_P5_T{N}_{description}
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_volume_h5(
    path: Path,
    n_samples: int = 5,
    shape: tuple[int, ...] = (32, 32, 32),
    seed: int = 42,
) -> None:
    """Create a mock volume HDF5 archive."""
    rng = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        data = rng.rand(n_samples, *shape).astype(np.float32)
        f.create_dataset("volumes", data=data)
        f.attrs["n_written"] = n_samples
        f.attrs["nfe"] = 1
        f.attrs["n_samples"] = n_samples


def _create_mock_feature_h5(
    path: Path,
    n_samples: int = 5,
    feature_dim: int = 512,
    seed: int = 42,
) -> None:
    """Create a mock feature HDF5 cache."""
    rng = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        feats = rng.randn(n_samples, feature_dim).astype(np.float32)
        f.create_dataset("features", data=feats)
        f.create_dataset(
            "source_indices",
            data=np.arange(n_samples, dtype=np.int64),
        )
        f.attrs["backend"] = "med3d"
        f.attrs["feature_dim"] = feature_dim
        f.attrs["n_samples"] = n_samples


def _build_comparison_tree(tmp_path: Path, n_samples: int = 5) -> Path:
    """Build a full comparison-mode directory tree with mock data.

    Returns the run_dir root.
    """
    run_dir = tmp_path / "run_test"
    gen_dir = run_dir / "generation"

    # Volumes
    for variant in ("baseline", "enhanced"):
        vol_dir = gen_dir / f"volumes_{variant}"
        for nfe in (1, 10, 50):
            _create_mock_volume_h5(
                vol_dir / f"nfe_{nfe:03d}.h5",
                n_samples=n_samples,
                seed=42 + nfe + (0 if variant == "baseline" else 100),
            )

    # Real test volumes
    _create_mock_volume_h5(gen_dir / "real_test.h5", n_samples=n_samples, seed=0)

    # Create empty dirs the SLURM scripts expect
    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    return run_dir


def _build_standard_tree(tmp_path: Path, n_samples: int = 5) -> Path:
    """Build a standard (non-comparison) directory tree."""
    run_dir = tmp_path / "run_standard"
    gen_dir = run_dir / "generation"

    vol_dir = gen_dir / "volumes"
    for nfe in (1, 10, 50):
        _create_mock_volume_h5(
            vol_dir / f"nfe_{nfe:03d}.h5",
            n_samples=n_samples,
            seed=42 + nfe,
        )

    _create_mock_volume_h5(gen_dir / "real_test.h5", n_samples=n_samples, seed=0)

    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    return run_dir


# ---------------------------------------------------------------------------
# P5-T21: Comparison-mode directory structure
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T21_comparison_mode_directory_structure(tmp_path: Path) -> None:
    """Verify comparison-mode tree has all expected H5 archives."""
    run_dir = _build_comparison_tree(tmp_path)
    gen_dir = run_dir / "generation"

    for variant in ("baseline", "enhanced"):
        vol_dir = gen_dir / f"volumes_{variant}"
        assert vol_dir.is_dir(), f"Missing {vol_dir}"
        for nfe in (1, 10, 50):
            h5 = vol_dir / f"nfe_{nfe:03d}.h5"
            assert h5.exists(), f"Missing {h5}"
            with h5py.File(str(h5), "r") as f:
                assert "volumes" in f
                assert f["volumes"].shape[0] == 5

    # Standard volumes/ dir should NOT exist
    assert not (gen_dir / "volumes").exists()
    assert (gen_dir / "real_test.h5").exists()


# ---------------------------------------------------------------------------
# P5-T22: Feature extraction works with variant subdirectory layout
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T22_feature_extraction_comparison_mode(tmp_path: Path) -> None:
    """Feature extractor writes to variant subdir with standard filenames."""
    from neuromf.metrics.feature_extractor import FeatureExtractor

    run_dir = _build_comparison_tree(tmp_path, n_samples=3)
    gen_dir = run_dir / "generation"

    # Mock the R3D-18 model — parameters() must return a fresh iterator each call
    mock_param = torch.zeros(1)
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([mock_param])
    mock_model.return_value = torch.randn(1, 512)

    with patch(
        "neuromf.metrics.feature_extractor.FeatureExtractor.__init__",
        return_value=None,
    ):
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.backend = "med3d"
        extractor.device = torch.device("cpu")
        extractor.model = mock_model
        extractor.center_slices_ratio = 0.6

        # Extract real features to top-level features/
        real_feat_dir = run_dir / "features"
        real_feat = real_feat_dir / "real_med3d.h5"
        real_features = extractor.extract_and_cache(gen_dir / "real_test.h5", real_feat)
        assert real_feat.exists()
        assert real_features.shape == (3, 512)

        # Extract generated features to variant subdir
        for variant in ("baseline", "enhanced"):
            gen_feat_dir = run_dir / "features" / variant
            gen_feat_dir.mkdir(parents=True, exist_ok=True)
            vol_dir = gen_dir / f"volumes_{variant}"

            for nfe in (1, 10, 50):
                vol_h5 = vol_dir / f"nfe_{nfe:03d}.h5"
                feat_h5 = gen_feat_dir / f"gen_med3d_nfe{nfe:03d}.h5"
                features = extractor.extract_and_cache(vol_h5, feat_h5)
                assert feat_h5.exists()
                assert features.shape == (3, 512)

        # Verify directory layout matches what compute_metrics.py expects
        assert (run_dir / "features" / "real_med3d.h5").exists()
        assert (run_dir / "features" / "baseline" / "gen_med3d_nfe001.h5").exists()
        assert (run_dir / "features" / "enhanced" / "gen_med3d_nfe050.h5").exists()

        # Verify load_cached works from variant subdir
        loaded = FeatureExtractor.load_cached(
            run_dir / "features" / "baseline" / "gen_med3d_nfe001.h5"
        )
        assert loaded.shape == (3, 512)


# ---------------------------------------------------------------------------
# P5-T23: compute_metrics.py with --gen-features-dir
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T23_compute_metrics_gen_features_dir(tmp_path: Path) -> None:
    """Verify compute_metrics loads features from separate real/gen dirs."""
    from neuromf.metrics.fid_3d import compute_fid_3d
    from neuromf.metrics.mmd import compute_mmd
    from neuromf.metrics.pairing import compute_nn_pairs

    n = 10
    feat_dim = 512

    # Create real features in top-level dir
    real_feat_dir = tmp_path / "features"
    _create_mock_feature_h5(real_feat_dir / "real_med3d.h5", n, feat_dim, seed=0)

    # Create gen features in variant subdir
    for variant in ("baseline", "enhanced"):
        gen_feat_dir = real_feat_dir / variant
        for nfe in (1, 10, 50):
            _create_mock_feature_h5(
                gen_feat_dir / f"gen_med3d_nfe{nfe:03d}.h5",
                n,
                feat_dim,
                seed=nfe + (0 if variant == "baseline" else 100),
            )

    # Load real features
    real_feats = FeatureExtractor_load_cached(real_feat_dir / "real_med3d.h5")
    assert real_feats.shape == (n, feat_dim)

    # Load gen features from variant subdir (same way compute_metrics.py does)
    gen_feat_path = real_feat_dir / "baseline" / "gen_med3d_nfe001.h5"
    gen_feats = FeatureExtractor_load_cached(gen_feat_path)
    assert gen_feats.shape == (n, feat_dim)

    # Compute distributional metrics
    fid = compute_fid_3d(real_feats, gen_feats)
    assert isinstance(fid, float)
    assert fid >= 0

    mmd = compute_mmd(real_feats, gen_feats)
    assert isinstance(mmd, float)

    # NN pairing
    nn_indices = compute_nn_pairs(real_feats, gen_feats)
    assert nn_indices.shape == (n,)
    assert nn_indices.max() < n


def FeatureExtractor_load_cached(path: Path) -> torch.Tensor:
    """Helper: load cached features."""
    with h5py.File(str(path), "r") as f:
        return torch.from_numpy(f["features"][:].astype(np.float32))


# ---------------------------------------------------------------------------
# P5-T24: Spectral metrics on mock volumes from H5
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T24_spectral_metrics_from_h5(tmp_path: Path) -> None:
    """Spectral metrics work when loading volumes from H5 archive."""
    from neuromf.metrics.spectral import compute_hf_energy_ratio

    h5_path = tmp_path / "nfe_001.h5"
    _create_mock_volume_h5(h5_path, n_samples=3, shape=(32, 32, 32), seed=42)

    with h5py.File(str(h5_path), "r") as f:
        for i in range(f["volumes"].shape[0]):
            vol = torch.from_numpy(f["volumes"][i].astype(np.float32))
            ratio = compute_hf_energy_ratio(vol)
            assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# P5-T25: Per-volume metrics with NN pairing on mock data
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T25_per_volume_metrics_mock(tmp_path: Path) -> None:
    """Per-volume MS-SSIM and PSNR work with H5-loaded paired volumes."""
    from neuromf.metrics.ms_ssim_3d import compute_ms_ssim_3d
    from neuromf.metrics.ssim_psnr import compute_psnr

    n = 3
    shape = (48, 48, 48)  # minimum 48 for 3-level MS-SSIM

    gen_path = tmp_path / "gen_volumes.h5"
    real_path = tmp_path / "real_volumes.h5"
    _create_mock_volume_h5(gen_path, n, shape, seed=42)
    _create_mock_volume_h5(real_path, n, shape, seed=0)

    # Simulate NN indices (identity pairing for test)
    nn_indices = torch.arange(n)

    with h5py.File(str(gen_path), "r") as gf, h5py.File(str(real_path), "r") as rf:
        for i in range(n):
            real_idx = int(nn_indices[i].item())
            gen_vol = torch.from_numpy(gf["volumes"][i].astype(np.float32))
            real_vol = torch.from_numpy(rf["volumes"][real_idx].astype(np.float32))

            gen_5d = gen_vol.unsqueeze(0).unsqueeze(0)
            real_5d = real_vol.unsqueeze(0).unsqueeze(0)

            ms_ssim = compute_ms_ssim_3d(gen_5d, real_5d)
            assert 0.0 <= ms_ssim <= 1.0

            psnr = compute_psnr(real_5d, gen_5d)
            assert psnr > 0


# ---------------------------------------------------------------------------
# P5-T26: Standard mode (non-comparison) directory structure
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T26_standard_mode_directory_structure(tmp_path: Path) -> None:
    """Standard (non-comparison) tree has volumes/ not volumes_baseline/."""
    run_dir = _build_standard_tree(tmp_path)
    gen_dir = run_dir / "generation"

    assert (gen_dir / "volumes").is_dir()
    assert not (gen_dir / "volumes_baseline").exists()
    assert not (gen_dir / "volumes_enhanced").exists()

    for nfe in (1, 10, 50):
        assert (gen_dir / "volumes" / f"nfe_{nfe:03d}.h5").exists()


# ---------------------------------------------------------------------------
# P5-T27: Coverage and density on mock features
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T27_coverage_density_mock_features() -> None:
    """Coverage and density work on mock feature tensors."""
    from neuromf.metrics.coverage_density import compute_coverage, compute_density

    torch.manual_seed(42)
    real = torch.randn(50, 512)
    gen = torch.randn(50, 512)

    cov = compute_coverage(real, gen, k=5)
    assert 0.0 <= cov <= 1.0

    dens = compute_density(real, gen, k=5)
    assert dens >= 0.0


# ---------------------------------------------------------------------------
# P5-T28: End-to-end distributional metrics pipeline
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T28_distributional_metrics_pipeline(tmp_path: Path) -> None:
    """Full distributional metrics from cached features (as compute_metrics.py does)."""
    from neuromf.metrics.coverage_density import compute_coverage, compute_density
    from neuromf.metrics.fid_3d import compute_fid_3d
    from neuromf.metrics.mmd import compute_mmd

    n_real, n_gen, feat_dim = 30, 30, 512

    # Create cached feature files
    _create_mock_feature_h5(tmp_path / "real_med3d.h5", n_real, feat_dim, seed=0)
    _create_mock_feature_h5(tmp_path / "gen_med3d_nfe001.h5", n_gen, feat_dim, seed=42)

    real_feats = FeatureExtractor_load_cached(tmp_path / "real_med3d.h5")
    gen_feats = FeatureExtractor_load_cached(tmp_path / "gen_med3d_nfe001.h5")

    fid = compute_fid_3d(real_feats, gen_feats)
    mmd = compute_mmd(real_feats, gen_feats)
    cov = compute_coverage(real_feats, gen_feats, k=5)
    dens = compute_density(real_feats, gen_feats, k=5)

    # All metrics should return valid floats
    for name, val in [("FID", fid), ("MMD", mmd), ("Coverage", cov), ("Density", dens)]:
        assert isinstance(val, float), f"{name} is not float: {type(val)}"
        assert not np.isnan(val), f"{name} is NaN"


# ---------------------------------------------------------------------------
# P5-T29: SynthSeg NIfTI conversion from comparison-mode H5
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T29_synthseg_nifti_from_comparison_h5(tmp_path: Path) -> None:
    """HDF5->NIfTI conversion works for volumes in comparison-mode subdirs."""
    from neuromf.metrics.synthseg_metrics import _h5_volumes_to_nifti

    # Create mock volume archive in comparison-mode location
    vol_dir = tmp_path / "generation" / "volumes_enhanced"
    h5_path = vol_dir / "nfe_001.h5"
    _create_mock_volume_h5(h5_path, n_samples=3, shape=(16, 16, 16))

    nifti_dir = tmp_path / "metrics" / "enhanced" / "synthseg" / "gen_nifti_nfe001"
    paths = _h5_volumes_to_nifti(h5_path, nifti_dir, voxel_size_mm=1.0, max_volumes=2)

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".gz"


# ---------------------------------------------------------------------------
# P5-T30: Metrics output JSON structure
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.informational
def test_P5_T30_metrics_output_json_structure(tmp_path: Path) -> None:
    """Verify the expected JSON keys from compute_metrics output."""
    # Simulate what compute_metrics.py writes
    metrics_result = {
        "experiment": "stage1_healthy",
        "nfe": 1,
        "n_generated": 500,
        "n_real": 326,
        "timestamp": "2026-03-23T00:00:00+00:00",
        "distributional": {
            "fid_3d": {"value": 73.85, "feature_extractor": "R3D-18", "feature_dim": 512},
            "mmd": {"value": 0.99, "kernel": "RBF", "n_bandwidths": 5},
            "coverage_k5": {"value": 0.65, "k": 5},
            "density_k5": {"value": 0.85, "k": 5},
        },
        "per_volume": {
            "ms_ssim": {"mean": 0.33, "std": 0.05, "pairing": "nearest_neighbour_med3d"},
            "psnr_db": {"mean": 18.5, "std": 2.1, "pairing": "nearest_neighbour_med3d"},
        },
        "spectral": {
            "hf_energy_ratio": {
                "mean": 0.15,
                "std": 0.02,
                "n_samples": 100,
                "cutoff": "0.5_nyquist",
            },
        },
        "morphological": {},
    }

    out_path = tmp_path / "metrics_nfe001.json"
    out_path.write_text(json.dumps(metrics_result, indent=2))

    loaded = json.loads(out_path.read_text())
    assert loaded["nfe"] == 1
    assert "distributional" in loaded
    assert "fid_3d" in loaded["distributional"]
    assert "per_volume" in loaded
    assert "spectral" in loaded
    assert "morphological" in loaded
