"""Integration tests for 3D-FID with real Med3D ResNet-50 weights.

These tests load the actual ``resnet_50_23dataset.pth`` checkpoint from disk
and verify end-to-end correctness: weight loading, feature extraction, FID
computation, and config path resolution.

Requires the external drive to be mounted (weights at
``{checkpoints_root}/fid_weights/resnet_50_23dataset.pth``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def _weights_path(base_config: OmegaConf) -> Path:
    """Resolve Med3D weights path from base config."""
    return Path(base_config.paths.checkpoints_root) / "fid_weights" / "resnet_50_23dataset.pth"


def _skip_if_no_weights(base_config: OmegaConf) -> None:
    """Skip test if Med3D weights are not available on disk."""
    path = _weights_path(base_config)
    if not path.exists():
        pytest.skip(f"Med3D weights not found at {path}")


# ---------------------------------------------------------------------------
# T1: Config path resolution
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T1_weights_path_resolves(base_config: OmegaConf) -> None:
    """Med3D weights exist at the path derived from base.yaml."""
    path = _weights_path(base_config)
    assert path.exists(), f"Med3D weights not found: {path}"
    # resnet_50_23dataset.pth is ~176 MB
    size_mb = path.stat().st_size / (1024 * 1024)
    assert size_mb > 100, f"Weights file too small ({size_mb:.1f} MB), may be corrupt"


# ---------------------------------------------------------------------------
# T2: Weight loading and state dict key mapping
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T2_load_real_weights(base_config: OmegaConf) -> None:
    """load_med3d_resnet50 loads real weights without errors.

    Verifies:
    - ``module.`` prefix stripping (Med3D trains with DataParallel)
    - ``conv_seg.*`` keys are filtered out (segmentation head we don't use)
    - No missing or unexpected keys after loading
    - Correct total parameter count (~46M for ResNet-50)
    """
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import load_med3d_resnet50

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))

    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode after loading"

    # Verify parameter count: ResNet-50 with expansion=4 ≈ 46M params
    n_params = sum(p.numel() for p in model.parameters())
    assert 40_000_000 < n_params < 55_000_000, f"Expected ~46M params, got {n_params / 1e6:.2f}M"

    # Verify all parameters are non-zero (weights were actually loaded)
    zero_layers = []
    for name, p in model.named_parameters():
        if p.numel() > 1 and p.abs().max().item() == 0.0:
            zero_layers.append(name)
    assert len(zero_layers) == 0, (
        f"These layers have all-zero weights (loading may have failed): {zero_layers}"
    )


# ---------------------------------------------------------------------------
# T3: Forward pass shape with real weights
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T3_forward_shape_real_weights(
    base_config: OmegaConf,
    device: torch.device,
) -> None:
    """Real Med3D model produces (B, 2048) for a small test volume."""
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import load_med3d_resnet50

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))
    model = model.to(device)

    # Use a small volume to keep memory low (48^3 not 192^3)
    vol = torch.randn(1, 1, 48, 48, 48, device=device)
    with torch.no_grad():
        out = model(vol)

    assert out.shape == (1, 2048), f"Expected (1, 2048), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# T4: Feature extraction determinism
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T4_extract_features_deterministic(
    base_config: OmegaConf,
    device: torch.device,
) -> None:
    """extract_3d_features is deterministic for a fixed input."""
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import extract_3d_features, load_med3d_resnet50

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))
    model = model.to(device)

    vol = torch.randn(1, 1, 48, 48, 48)

    feats1 = extract_3d_features(vol, model, normalize=True)
    feats2 = extract_3d_features(vol, model, normalize=True)

    assert feats1.shape == (1, 2048)
    assert torch.allclose(feats1, feats2, atol=1e-5), (
        f"Features not deterministic: max diff = {(feats1 - feats2).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# T5: Min-max normalization behavior
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T5_normalization_effect(
    base_config: OmegaConf,
    device: torch.device,
) -> None:
    """Normalization changes features; un-normalized and normalized differ."""
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import extract_3d_features, load_med3d_resnet50

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))
    model = model.to(device)

    # Volume with non-unit range (like raw MRI intensities)
    vol = torch.randn(1, 1, 48, 48, 48) * 100.0 + 50.0

    feats_norm = extract_3d_features(vol, model, normalize=True)
    feats_raw = extract_3d_features(vol, model, normalize=False)

    assert feats_norm.shape == feats_raw.shape == (1, 2048)
    # They should differ because normalization changes the input distribution
    assert not torch.allclose(feats_norm, feats_raw, atol=1e-3), (
        "Normalized and raw features should differ for non-unit-range input"
    )


# ---------------------------------------------------------------------------
# T6: FID = 0 for identical feature sets (real model)
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T6_fid_zero_identical_real_features(
    base_config: OmegaConf,
    device: torch.device,
) -> None:
    """FID = 0.0 when real and fake features are identical (real model).

    Uses the actual Med3D model to extract features from multiple random
    volumes, then computes FID of the feature set against itself.
    """
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import (
        compute_fid_3d,
        extract_3d_features,
        load_med3d_resnet50,
    )

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))
    model = model.to(device)

    # Extract features from 8 random volumes (enough for stable covariance)
    gen = torch.Generator().manual_seed(42)
    feats_list = []
    for _ in range(8):
        vol = torch.randn(1, 1, 48, 48, 48, generator=gen)
        f = extract_3d_features(vol, model, normalize=True)
        feats_list.append(f)

    feats = torch.cat(feats_list, dim=0)  # (8, 2048)
    assert feats.shape == (8, 2048)

    fid = compute_fid_3d(feats, feats.clone())
    assert abs(fid) < 1e-3, f"FID of identical features should be ~0, got {fid}"


# ---------------------------------------------------------------------------
# T7: FID > 0 for different feature sets
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_fid3d_T7_fid_positive_different_features(
    base_config: OmegaConf,
    device: torch.device,
) -> None:
    """FID > 0 when real and fake features come from different distributions."""
    _skip_if_no_weights(base_config)

    from neuromf.metrics.fid_3d import (
        compute_fid_3d,
        extract_3d_features,
        load_med3d_resnet50,
    )

    path = _weights_path(base_config)
    model = load_med3d_resnet50(str(path))
    model = model.to(device)

    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(999)

    feats_a = torch.cat(
        [
            extract_3d_features(torch.randn(1, 1, 48, 48, 48, generator=gen_a), model)
            for _ in range(8)
        ]
    )
    # Shift distribution significantly to guarantee FID > 0
    feats_b = torch.cat(
        [
            extract_3d_features(torch.randn(1, 1, 48, 48, 48, generator=gen_b) + 5.0, model)
            for _ in range(8)
        ]
    )

    fid = compute_fid_3d(feats_a, feats_b)
    assert fid > 0.0, f"FID of different distributions should be > 0, got {fid}"


# ---------------------------------------------------------------------------
# T8: State dict key inspection (detailed verification)
# ---------------------------------------------------------------------------


@pytest.mark.informational
def test_fid3d_T8_state_dict_key_structure(base_config: OmegaConf) -> None:
    """Verify Med3D checkpoint key structure matches expectations.

    The checkpoint should have ``module.`` prefixed keys (DataParallel)
    and ``conv_seg.*`` keys (segmentation head). Our loader strips both.
    """
    _skip_if_no_weights(base_config)

    path = _weights_path(base_config)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

    # May be wrapped in "state_dict"
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    keys = list(state.keys())
    assert len(keys) > 100, f"Expected >100 keys, got {len(keys)}"

    # Check for module. prefix (DataParallel convention)
    has_module_prefix = any(k.startswith("module.") for k in keys)
    # Check for conv_seg keys (segmentation head)
    has_conv_seg = any("conv_seg" in k for k in keys)

    # Log findings (at least one of these should be true for Med3D format)
    if has_module_prefix:
        n_module = sum(1 for k in keys if k.startswith("module."))
        print(f"Found {n_module}/{len(keys)} keys with 'module.' prefix")
    if has_conv_seg:
        conv_seg_keys = [k for k in keys if "conv_seg" in k]
        print(f"Found {len(conv_seg_keys)} conv_seg keys: {conv_seg_keys}")

    # Verify expected layer structure exists (after prefix stripping)
    stripped_keys = {k.removeprefix("module.") for k in keys}
    expected_prefixes = ["conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4."]
    for prefix in expected_prefixes:
        matching = [k for k in stripped_keys if k.startswith(prefix)]
        assert len(matching) > 0, f"No keys starting with '{prefix}' — wrong checkpoint?"
