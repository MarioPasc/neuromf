"""Tests for JVP compatibility with MAISIUNetWrapper — Phase 3 verification.

P3-T3a: torch.func.jvp executes on UNet at small resolution
P3-T3b: JVP matches finite-difference at small resolution
P3-T12: Full-size finite-diff JVP at (1, 4, 48, 48, 48)
"""

import pytest
import torch

from neuromf.wrappers.jvp_strategies import ExactJVP, FiniteDifferenceJVP
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper

# Spatial dim must be divisible by 2^3=8 (3 downsample levels in the UNet)
_S = 16


@pytest.fixture(scope="module")
def small_model() -> MAISIUNetWrapper:
    """Create a small UNet wrapper for JVP testing."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.eval()
    return model


def _make_u_fn(model: MAISIUNetWrapper, t_min: float = 0.05):
    """Create u_fn closure from x-prediction model."""

    def u_fn(z: torch.Tensor, t_: torch.Tensor, r_: torch.Tensor) -> torch.Tensor:
        x_hat = model(z, r_, t_)
        t_safe = t_.view(-1, *([1] * (z.ndim - 1))).clamp(min=t_min)
        return (z - x_hat) / t_safe

    return u_fn


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T3a_jvp_executes_on_unet(small_model: MAISIUNetWrapper) -> None:
    """P3-T3a: torch.func.jvp executes at (1, 4, 16, 16, 16) without error."""
    B, C, D, H, W = 1, 4, _S, _S, _S
    z_t = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5])
    r = torch.tensor([0.2])
    v_tangent = torch.randn(B, C, D, H, W)

    u_fn = _make_u_fn(small_model)
    exact_jvp = ExactJVP()

    u, V = exact_jvp.compute(u_fn, z_t, t, r, v_tangent)

    assert u.shape == (B, C, D, H, W), f"u shape mismatch: {u.shape}"
    assert V.shape == (B, C, D, H, W), f"V shape mismatch: {V.shape}"
    assert torch.isfinite(u).all(), "u contains NaN/Inf"
    assert torch.isfinite(V).all(), "V contains NaN/Inf"


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T3b_jvp_matches_finite_difference(small_model: MAISIUNetWrapper) -> None:
    """P3-T3b: JVP matches finite-difference at (1, 4, 16, 16, 16).

    Relative error should be < 0.05 for h=1e-3.
    """
    B, C, D, H, W = 1, 4, _S, _S, _S
    torch.manual_seed(123)
    z_t = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5])
    r = torch.tensor([0.2])
    v_tangent = torch.randn(B, C, D, H, W)

    u_fn = _make_u_fn(small_model)

    exact_jvp = ExactJVP()
    fd_jvp = FiniteDifferenceJVP(h=1e-3)

    u_exact, V_exact = exact_jvp.compute(u_fn, z_t, t, r, v_tangent)
    u_fd, V_fd = fd_jvp.compute(u_fn, z_t, t, r, v_tangent)

    # u should be identical (same forward pass)
    assert torch.allclose(u_exact, u_fd, atol=1e-4), (
        f"u values differ: max diff = {(u_exact - u_fd).abs().max():.6f}"
    )

    # V should be close (JVP vs finite diff)
    rel_err = (V_exact - V_fd).norm() / V_exact.norm().clamp(min=1e-8)
    assert rel_err < 0.05, f"JVP vs FD relative error too large: {rel_err:.4f}"


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_T12_full_size_finite_diff_jvp(device: torch.device) -> None:
    """P3-T12: Full-size finite-diff JVP at (1, 4, 48, 48, 48).

    Skipped if insufficient VRAM.
    """
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        if vram_gb < 7.5:
            pytest.skip(f"Insufficient VRAM: {vram_gb:.1f} GB")

    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config).to(device)
    model.eval()

    B, C, D, H, W = 1, 4, 48, 48, 48
    z_t = torch.randn(B, C, D, H, W, device=device)
    t = torch.tensor([0.5], device=device)
    r = torch.tensor([0.2], device=device)
    v_tangent = torch.randn(B, C, D, H, W, device=device)

    u_fn = _make_u_fn(model)
    fd_jvp = FiniteDifferenceJVP(h=1e-3)

    with torch.no_grad():
        u, V = fd_jvp.compute(u_fn, z_t, t, r, v_tangent)

    assert u.shape == (B, C, D, H, W)
    assert V.shape == (B, C, D, H, W)
    assert torch.isfinite(u).all()
    assert torch.isfinite(V).all()


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_jvp_r_equals_t_reduces_to_fm() -> None:
    """When r == t, JVP term vanishes and V == u (standard FM)."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.eval()

    B, C, D, H, W = 1, 4, _S, _S, _S
    z_t = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5])
    r = torch.tensor([0.5])  # r == t
    v_tangent = torch.randn(B, C, D, H, W)

    u_fn = _make_u_fn(model)
    exact_jvp = ExactJVP()

    u, V = exact_jvp.compute(u_fn, z_t, t, r, v_tangent)

    # When t == r, compound velocity V should equal u
    assert torch.allclose(u, V, atol=1e-5), (
        f"V should equal u when r==t, max diff: {(u - V).abs().max():.6f}"
    )
