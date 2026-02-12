"""Phase 1 verification tests for the latent pre-computation pipeline.

Tests P1-T1 through P1-T7 are CRITICAL (gate Phase 2).
Tests P1-T8 and P1-T9 are INFORMATIONAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from neuromf.data.latent_dataset import LatentDataset
from neuromf.utils.latent_stats import LatentStatsAccumulator, load_latent_stats

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def merged_config() -> OmegaConf:
    """Load and merge base + fomo60k + encode_dataset configs."""
    configs_dir = Path(__file__).parent.parent / "configs"
    base = OmegaConf.load(configs_dir / "base.yaml")
    fomo60k = OmegaConf.load(configs_dir / "fomo60k.yaml")
    encode = OmegaConf.load(configs_dir / "encode_dataset.yaml")
    cfg = OmegaConf.merge(base, fomo60k, encode)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(scope="module")
def latent_dir(merged_config: OmegaConf) -> Path:
    """Return the latent output directory from config."""
    return Path(merged_config.output.latent_dir)


@pytest.fixture(scope="module")
def latent_stats(merged_config: OmegaConf) -> dict:
    """Load latent_stats.json from the output directory."""
    stats_path = Path(merged_config.output.stats_path)
    if not stats_path.exists():
        pytest.skip(f"latent_stats.json not found at {stats_path} — run encoding first")
    return load_latent_stats(stats_path)


@pytest.fixture(scope="module")
def encoding_log(merged_config: OmegaConf) -> dict:
    """Load encoding_log.json from the output directory."""
    log_path = Path(merged_config.output.encoding_log)
    if not log_path.exists():
        pytest.skip(f"encoding_log.json not found at {log_path} — run encoding first")
    return json.loads(log_path.read_text())


# ---------------------------------------------------------------------------
# CRITICAL tests (P1-T1 through P1-T7)
# ---------------------------------------------------------------------------


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T1_all_volumes_encode_without_error(encoding_log: dict) -> None:
    """All FOMO-60K volumes must encode without error."""
    n_failed = encoding_log["n_failed"]
    n_total = encoding_log["n_total"]
    assert n_failed == 0, (
        f"{n_failed}/{n_total} volumes failed encoding. Check encoding_log.json for details."
    )


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T2_latent_shape_correct(latent_dir: Path) -> None:
    """All .pt files must have z.shape == (4, 32, 32, 32)."""
    pt_files = sorted(latent_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files found — run encoding first")

    # Check 10 random files (or all if fewer)
    rng = np.random.default_rng(42)
    n_check = min(10, len(pt_files))
    indices = rng.choice(len(pt_files), size=n_check, replace=False)

    for idx in indices:
        data = torch.load(pt_files[idx], map_location="cpu", weights_only=True)
        z = data["z"]
        assert z.shape == (4, 32, 32, 32), (
            f"{pt_files[idx].name}: expected (4,32,32,32), got {z.shape}"
        )


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T3_per_channel_mean_near_zero(latent_stats: dict) -> None:
    """|mean_c| < 0.5 for all channels."""
    for ch_key, ch_stats in latent_stats["per_channel"].items():
        mean = ch_stats["mean"]
        assert abs(mean) < 0.5, f"{ch_key}: |mean| = {abs(mean):.4f} >= 0.5"


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T4_per_channel_std_in_range(latent_stats: dict) -> None:
    """std_c in [0.5, 2.0] for all channels."""
    for ch_key, ch_stats in latent_stats["per_channel"].items():
        std = ch_stats["std"]
        assert 0.5 <= std <= 2.0, f"{ch_key}: std = {std:.4f} not in [0.5, 2.0]"


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T5_no_nan_inf_in_latents(latent_dir: Path) -> None:
    """No NaN or Inf in latent tensors (spot-check 50 files)."""
    pt_files = sorted(latent_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files found — run encoding first")

    rng = np.random.default_rng(42)
    n_check = min(50, len(pt_files))
    indices = rng.choice(len(pt_files), size=n_check, replace=False)

    for idx in indices:
        data = torch.load(pt_files[idx], map_location="cpu", weights_only=True)
        z = data["z"]
        assert torch.isfinite(z).all(), f"{pt_files[idx].name} contains NaN or Inf"


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T6_latent_dataset_loads_correctly(tmp_path: Path) -> None:
    """LatentDataset loads mock .pt files with correct shape and normalisation."""
    n_channels = 4
    spatial = (32, 32, 32)
    n_mock = 3

    # Create mock .pt files
    for i in range(n_mock):
        z = torch.randn(n_channels, *spatial)
        data = {
            "z": z,
            "metadata": {"subject_id": f"test_{i:03d}", "dataset": "mock"},
        }
        torch.save(data, tmp_path / f"mock_{i:03d}.pt")

    # Create mock latent_stats.json
    stats = {
        "n_files": n_mock,
        "n_voxels_per_channel": n_mock * 32 * 32 * 32,
        "per_channel": {},
        "cross_channel_correlation": np.eye(n_channels).tolist(),
    }
    for c in range(n_channels):
        stats["per_channel"][f"channel_{c}"] = {
            "mean": 0.0,
            "std": 1.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "min": -3.0,
            "max": 3.0,
        }
    (tmp_path / "latent_stats.json").write_text(json.dumps(stats))

    # Test normalise=False
    ds_raw = LatentDataset(tmp_path, normalise=False)
    assert len(ds_raw) == n_mock
    sample = ds_raw[0]
    assert sample["z"].shape == (n_channels, *spatial)
    assert sample["z"].dtype == torch.float32
    assert "metadata" in sample

    # Test normalise=True (with mean=0, std=1 => should be identity)
    ds_norm = LatentDataset(tmp_path, normalise=True)
    sample_norm = ds_norm[0]
    assert sample_norm["z"].shape == (n_channels, *spatial)
    assert torch.allclose(sample["z"], sample_norm["z"], atol=1e-6), (
        "With mean=0, std=1, normalised output should match raw"
    )

    # Test normalise=True with non-trivial stats
    stats["per_channel"]["channel_0"]["mean"] = 2.0
    stats["per_channel"]["channel_0"]["std"] = 0.5
    (tmp_path / "latent_stats.json").write_text(json.dumps(stats))

    ds_shifted = LatentDataset(tmp_path, normalise=True)
    sample_shifted = ds_shifted[0]
    raw_ch0 = sample["z"][0]
    expected_ch0 = (raw_ch0 - 2.0) / 0.5
    assert torch.allclose(sample_shifted["z"][0], expected_ch0, atol=1e-5), (
        "Normalisation should apply (z - mean) / std per channel"
    )


@pytest.mark.phase1
@pytest.mark.critical
def test_P1_T7_round_trip_ssim(
    latent_dir: Path,
    merged_config: OmegaConf,
    device: torch.device,
) -> None:
    """Round-trip decode(load(.pt)) ~ original, SSIM > 0.89 for 5 volumes."""
    from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
    from neuromf.metrics.ssim_psnr import compute_ssim_3d
    from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

    pt_files = sorted(latent_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files found — run encoding first")

    # Pick 5 random files
    rng = np.random.default_rng(42)
    n_check = min(5, len(pt_files))
    indices = rng.choice(len(pt_files), size=n_check, replace=False)

    # Load VAE
    vae_config = MAISIVAEConfig.from_omegaconf(merged_config)
    vae = MAISIVAEWrapper(vae_config, device=device)

    # Build preprocessing
    transform = build_mri_preprocessing_from_config(merged_config)

    for idx in indices:
        data = torch.load(pt_files[idx], map_location="cpu", weights_only=True)
        z = data["z"].unsqueeze(0).to(device)  # (1, 4, 32, 32, 32)
        metadata = data.get("metadata", {})

        # Decode latent
        x_hat = vae.decode(z)  # (1, 1, 128, 128, 128)

        # Load and preprocess original
        source_path = metadata.get("source_path", "")
        if not source_path or not Path(source_path).exists():
            pytest.skip(f"Source NIfTI not found: {source_path}")

        processed = transform({"image": source_path})
        x_orig = processed["image"].unsqueeze(0)  # (1, 1, 128, 128, 128)

        ssim_val = compute_ssim_3d(x_orig, x_hat.cpu())
        assert ssim_val > 0.89, (
            f"Volume {pt_files[idx].name}: round-trip SSIM={ssim_val:.4f} < 0.89"
        )

        del z, x_hat, x_orig
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# INFORMATIONAL tests (P1-T8, P1-T9)
# ---------------------------------------------------------------------------


@pytest.mark.phase1
@pytest.mark.informational
def test_P1_T8_normalised_latents_near_standard_normal(tmp_path: Path) -> None:
    """After normalisation, per-channel mean ~ 0, std ~ 1."""
    n_channels = 4
    n_files = 20
    spatial = (32, 32, 32)

    # Generate mock latents with known non-zero mean/std
    true_means = [1.5, -0.3, 0.8, -1.2]
    true_stds = [0.7, 1.4, 0.9, 1.1]

    for i in range(n_files):
        z = torch.zeros(n_channels, *spatial)
        for c in range(n_channels):
            z[c] = torch.randn(spatial) * true_stds[c] + true_means[c]
        torch.save(
            {"z": z, "metadata": {"subject_id": f"test_{i}"}},
            tmp_path / f"test_{i}.pt",
        )

    # Compute stats with our accumulator
    acc = LatentStatsAccumulator(n_channels=n_channels)
    for i in range(n_files):
        data = torch.load(tmp_path / f"test_{i}.pt", map_location="cpu", weights_only=True)
        acc.update(data["z"])
    stats = acc.finalize()
    stats["n_files"] = n_files
    (tmp_path / "latent_stats.json").write_text(json.dumps(stats))

    # Load normalised dataset
    ds = LatentDataset(tmp_path, normalise=True)

    # Accumulate normalised stats over all files
    all_z = torch.stack([ds[i]["z"] for i in range(len(ds))])  # (N, C, D, H, W)
    for c in range(n_channels):
        ch_mean = all_z[:, c].mean().item()
        ch_std = all_z[:, c].std().item()
        assert abs(ch_mean) < 0.1, f"Channel {c} normalised mean={ch_mean:.4f} not near 0"
        assert abs(ch_std - 1.0) < 0.1, f"Channel {c} normalised std={ch_std:.4f} not near 1"


@pytest.mark.phase1
@pytest.mark.informational
def test_P1_T9_latent_file_count(latent_dir: Path) -> None:
    """Expected ~1100 .pt files (informational — not a hard gate)."""
    pt_files = list(latent_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files found — run encoding first")
    n = len(pt_files)
    assert n >= 500, f"Only {n} latent files found, expected ~1100"
    print(f"INFO: {n} latent files found in {latent_dir}")
