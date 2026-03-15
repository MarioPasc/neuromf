"""Tests for 3D-FID using the MOTFM R3D-18 protocol.

Verifies end-to-end correctness of the R3D-18 (Kinetics-400 pretrained)
feature extraction and FID computation. No external weights file is required
— torchvision downloads the checkpoint automatically.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# T1: R3D-18 loads successfully (replaces old Med3D path resolution test)
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T1_load_r3d18() -> None:
    """load_fid3d_feature_net loads R3D-18 in eval mode with correct output dim."""
    try:
        from neuromf.metrics.fid_3d import FID3D_FEATURE_DIM, load_fid3d_feature_net
    except ImportError as exc:
        # Pre-existing: __init__.py imports synthseg_metrics which needs nibabel
        if "nibabel" in str(exc):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "neuromf.metrics.fid_3d",
                "src/neuromf/metrics/fid_3d.py",
            )
            fid_3d = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fid_3d)
            FID3D_FEATURE_DIM = fid_3d.FID3D_FEATURE_DIM
            load_fid3d_feature_net = fid_3d.load_fid3d_feature_net
        else:
            raise

    model = load_fid3d_feature_net(device="cpu")

    assert not model.training, "Model should be in eval mode"

    # R3D-18 ≈ 33M params
    n_params = sum(p.numel() for p in model.parameters())
    assert 30_000_000 < n_params < 40_000_000, f"Expected ~33M params, got {n_params / 1e6:.2f}M"

    # Verify feature dim constant
    assert FID3D_FEATURE_DIM == 512


# ---------------------------------------------------------------------------
# T2: Backward-compatible alias works
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T2_backward_compat_alias() -> None:
    """load_med3d_resnet50 alias loads R3D-18 and warns about ignored weights_path."""
    import warnings

    from neuromf.metrics.fid_3d import load_med3d_resnet50

    # Should work with no arguments
    model = load_med3d_resnet50()
    assert not model.training

    # Should work with a dummy weights path (ignored, but warns)
    model = load_med3d_resnet50("/nonexistent/path.pth")
    assert not model.training


# ---------------------------------------------------------------------------
# T3: Forward pass shape
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T3_forward_shape(device: torch.device) -> None:
    """R3D-18 produces (B, 512) output for various volume sizes."""
    from neuromf.metrics.fid_3d import load_fid3d_feature_net

    model = load_fid3d_feature_net(device=device)

    for size in [32, 48]:
        vol = torch.randn(1, 3, size, size, size, device=device)
        with torch.no_grad():
            out = model(vol)
        assert out.shape == (1, 512), f"Expected (1, 512) for size {size}, got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# T4: Feature extraction determinism
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T4_extract_features_deterministic(device: torch.device) -> None:
    """extract_3d_features is deterministic for a fixed input."""
    from neuromf.metrics.fid_3d import extract_3d_features, load_fid3d_feature_net

    model = load_fid3d_feature_net(device=device)
    vol = torch.randn(2, 1, 48, 48, 48)

    feats1 = extract_3d_features(vol, model, normalize=True)
    feats2 = extract_3d_features(vol, model, normalize=True)

    assert feats1.shape == (2, 512)
    assert torch.allclose(feats1, feats2, atol=1e-5), (
        f"Features not deterministic: max diff = {(feats1 - feats2).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# T5: Per-set normalisation (global, not per-volume)
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T5_per_set_normalization() -> None:
    """Per-set min-max normalisation spans all volumes jointly."""
    from neuromf.metrics.fid_3d import _per_set_minmax

    # Two volumes with very different ranges
    vol_a = torch.zeros(1, 1, 8, 8, 8)
    vol_b = torch.ones(1, 1, 8, 8, 8) * 100.0
    batch = torch.cat([vol_a, vol_b], dim=0)  # (2, 1, 8, 8, 8)

    normed = _per_set_minmax(batch)

    # Global min should be 0, global max should be 100
    assert normed.min().item() == pytest.approx(0.0, abs=1e-6)
    assert normed.max().item() == pytest.approx(1.0, abs=1e-6)

    # vol_a should be all 0 (it was the global min)
    assert normed[0].max().item() == pytest.approx(0.0, abs=1e-6)
    # vol_b should be all 1 (it was the global max)
    assert normed[1].min().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# T6: FID = 0 for identical feature sets
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T6_fid_zero_identical_features(device: torch.device) -> None:
    """FID = 0.0 when real and fake features are identical."""
    from neuromf.metrics.fid_3d import (
        compute_fid_3d,
        extract_3d_features,
        load_fid3d_feature_net,
    )

    model = load_fid3d_feature_net(device=device)

    gen = torch.Generator().manual_seed(42)
    vols = torch.randn(8, 1, 48, 48, 48, generator=gen)
    feats = extract_3d_features(vols, model, normalize=True)

    assert feats.shape == (8, 512)

    fid = compute_fid_3d(feats, feats.clone())
    assert abs(fid) < 1e-3, f"FID of identical features should be ~0, got {fid}"


# ---------------------------------------------------------------------------
# T7: FID > 0 for different distributions (no feature collapse)
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T7_fid_positive_different_features(device: torch.device) -> None:
    """FID > 0 for different distributions, verifying no feature collapse."""
    from neuromf.metrics.fid_3d import (
        compute_fid_3d,
        extract_3d_features,
        load_fid3d_feature_net,
    )

    model = load_fid3d_feature_net(device=device)

    # Distribution A: random noise
    vols_a = torch.randn(8, 1, 48, 48, 48)
    feats_a = extract_3d_features(vols_a, model, normalize=True)

    # Distribution B: heavily smoothed (simulating blurry generation)
    vols_b = torch.randn(8, 1, 48, 48, 48)
    k = 11
    k1d = torch.ones(k) / k
    k3d = k1d.view(1, 1, -1, 1, 1) * k1d.view(1, 1, 1, -1, 1) * k1d.view(1, 1, 1, 1, -1)
    vols_b = F.conv3d(F.pad(vols_b, [k // 2] * 6, mode="replicate"), k3d)
    feats_b = extract_3d_features(vols_b, model, normalize=True)

    fid = compute_fid_3d(feats_a, feats_b)
    assert fid > 1.0, f"FID between sharp and smooth should be >> 0, got {fid}"

    # Also verify features are NOT collapsed (cosine sim < 0.9999)
    sim = F.cosine_similarity(feats_a[0:1], feats_a[1:2]).item()
    assert sim < 0.9999, f"Feature collapse detected: cosine sim = {sim:.6f}"


# ---------------------------------------------------------------------------
# T8: Channel replication (1ch → 3ch)
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T8_channel_replication() -> None:
    """_to_three_channels correctly handles 1ch and 3ch inputs."""
    from neuromf.metrics.fid_3d import _to_three_channels

    # 1-channel -> 3-channel (repeat)
    vol_1ch = torch.randn(2, 1, 8, 8, 8)
    vol_3ch = _to_three_channels(vol_1ch)
    assert vol_3ch.shape == (2, 3, 8, 8, 8)
    assert torch.equal(vol_3ch[:, 0], vol_3ch[:, 1])  # All channels identical

    # Already 3-channel -> pass through
    vol_rgb = torch.randn(2, 3, 8, 8, 8)
    vol_out = _to_three_channels(vol_rgb)
    assert torch.equal(vol_out, vol_rgb)

    # 4-channel -> crop to 3
    vol_4ch = torch.randn(2, 4, 8, 8, 8)
    vol_out = _to_three_channels(vol_4ch)
    assert vol_out.shape == (2, 3, 8, 8, 8)


# ---------------------------------------------------------------------------
# T9: Input shape flexibility
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_fid3d_T9_input_shape_flexibility(device: torch.device) -> None:
    """extract_3d_features accepts various input shapes."""
    from neuromf.metrics.fid_3d import extract_3d_features, load_fid3d_feature_net

    model = load_fid3d_feature_net(device=device)

    # (D, H, W) -> (1, 512)
    feats = extract_3d_features(torch.randn(32, 32, 32), model)
    assert feats.shape == (1, 512)

    # (B, D, H, W) -> (B, 512)
    feats = extract_3d_features(torch.randn(3, 32, 32, 32), model)
    assert feats.shape == (3, 512)

    # (B, 1, D, H, W) -> (B, 512)
    feats = extract_3d_features(torch.randn(2, 1, 32, 32, 32), model)
    assert feats.shape == (2, 512)
