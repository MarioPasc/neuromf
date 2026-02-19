"""Tests for MAISIUNetWrapper — Phase 3 verification.

P3-T1: UNet accepts dual (r, t) conditioning
P3-T2: Output shape matches input latent shape
P3-T11: Full-size forward pass at (1, 4, 48, 48, 48)
"""

import pytest
import torch

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper


@pytest.fixture(scope="module")
def small_model() -> MAISIUNetWrapper:
    """Create a small UNet wrapper for testing at reduced resolution."""
    torch.manual_seed(42)
    config = MAISIUNetConfig()
    model = MAISIUNetWrapper(config)
    model.eval()
    return model


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T1_unet_dual_time_conditioning(small_model: MAISIUNetWrapper) -> None:
    """P3-T1: UNet accepts (r, t) conditioning without error."""
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_t = torch.randn(B, C, D, H, W)
    r = torch.tensor([0.1, 0.3])
    t = torch.tensor([0.5, 0.8])

    with torch.no_grad():
        out = small_model(z_t, r, t)

    assert out is not None
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T2_output_shape_matches_input(small_model: MAISIUNetWrapper) -> None:
    """P3-T2: Output shape matches input latent shape."""
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_t = torch.randn(B, C, D, H, W)
    r = torch.tensor([0.1, 0.3])
    t = torch.tensor([0.5, 0.8])

    with torch.no_grad():
        out = small_model(z_t, r, t)

    assert out.shape == (B, C, D, H, W), f"Expected {(B, C, D, H, W)}, got {out.shape}"


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_T11_full_size_forward_pass(device: torch.device) -> None:
    """P3-T11: Full-size forward pass at (1, 4, 48, 48, 48).

    Skipped if VRAM < 8GB (or CPU fallback too slow).
    """
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        if vram_gb < 7.5:
            pytest.skip(f"Insufficient VRAM: {vram_gb:.1f} GB")

    torch.manual_seed(42)
    config = MAISIUNetConfig()
    model = MAISIUNetWrapper(config).to(device)
    model.eval()

    B, C, D, H, W = 1, 4, 48, 48, 48
    z_t = torch.randn(B, C, D, H, W, device=device)
    r = torch.tensor([0.2], device=device)
    t = torch.tensor([0.7], device=device)

    with torch.no_grad():
        out = model(z_t, r, t)

    assert out.shape == (B, C, D, H, W)
    assert torch.isfinite(out).all()


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_u_from_x_conversion(small_model: MAISIUNetWrapper) -> None:
    """Verify u_from_x correctly converts x-prediction to velocity."""
    B, C, D, H, W = 2, 4, 8, 8, 8
    z_t = torch.randn(B, C, D, H, W)
    x_pred = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])

    u = small_model.u_from_x(z_t, x_pred, t)

    # Verify u = (z_t - x_pred) / t
    t_broad = t.view(-1, 1, 1, 1, 1)
    expected = (z_t - x_pred) / t_broad
    assert torch.allclose(u, expected, atol=1e-6)


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_u_from_x_t_min_clamping(small_model: MAISIUNetWrapper) -> None:
    """Verify t_min clamping prevents division by zero."""
    B, C, D, H, W = 1, 4, 8, 8, 8
    z_t = torch.randn(B, C, D, H, W)
    x_pred = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.001])  # Below t_min=0.05

    u = small_model.u_from_x(z_t, x_pred, t)

    assert torch.isfinite(u).all(), "u should be finite even at small t"
    # Should use t_min=0.05, not actual t=0.001
    t_safe = torch.tensor([0.05]).view(-1, 1, 1, 1, 1)
    expected = (z_t - x_pred) / t_safe
    assert torch.allclose(u, expected, atol=1e-6)


@pytest.mark.phase4
@pytest.mark.informational
def test_P4g_T10_h_conditioning_differs_from_dual() -> None:
    """P4g-T10: h-conditioning produces different embeddings than dual for r != t."""
    torch.manual_seed(42)

    config_h = MAISIUNetConfig(prediction_type="u", conditioning_mode="h")
    model_h = MAISIUNetWrapper(config_h)

    config_dual = MAISIUNetConfig(prediction_type="u", conditioning_mode="dual")
    model_dual = MAISIUNetWrapper(config_dual)

    # Copy UNet weights from dual to h for fair comparison
    model_h.unet.load_state_dict(model_dual.unet.state_dict())

    # Re-init zero-init output conv so outputs are non-zero
    for m in (model_h, model_dual):
        for name, module in m.named_modules():
            if isinstance(module, torch.nn.Conv3d) and module.weight.abs().sum() == 0:
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")

    model_h.eval()
    model_dual.eval()

    B, C, D, H, W = 1, 4, 16, 16, 16
    z_t = torch.randn(B, C, D, H, W)
    r = torch.tensor([0.2])
    t = torch.tensor([0.7])

    with torch.no_grad():
        out_h = model_h(z_t, r, t)
        out_dual = model_dual(z_t, r, t)

    # With different conditioning, outputs should differ
    assert not torch.allclose(out_h, out_dual, atol=1e-4), (
        "h-conditioning and dual conditioning should produce different outputs for r != t"
    )
