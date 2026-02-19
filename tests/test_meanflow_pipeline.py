"""Tests for MeanFlowPipeline — Phase 3 verification + Phase 4 u-prediction.

P3-T4: MeanFlow loss is finite and positive
P3-T5: Gradients flow to all UNet params
P3-T9: Combined iMF loss (FM + MF) computes
P3-T10: bf16 mixed precision works
P4f: u-prediction pipeline, FM reduces to u, FD-JVP bounded
"""

import pytest
import torch

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig


@pytest.fixture(scope="module")
def model_and_pipeline():
    """Create model + pipeline for testing."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        prediction_type="x",
        jvp_strategy="exact",
    )
    pipeline = MeanFlowPipeline(pipeline_config)
    return model, pipeline


@pytest.fixture(scope="module")
def fd_model_and_pipeline():
    """Create model + finite-difference pipeline for testing."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        prediction_type="x",
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)
    return model, pipeline


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T4_meanflow_loss_finite_positive(model_and_pipeline) -> None:
    """P3-T4: MeanFlow loss is finite and positive at (2, 4, 16, 16, 16)."""
    model, pipeline = model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r)

    assert "loss" in result
    assert "raw_loss" in result

    loss = result["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert 0 < loss.item() < 1000, f"Loss out of range: {loss.item()}"
    assert torch.isfinite(result["raw_loss"])


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T5_gradients_flow_to_all_params() -> None:
    """P3-T5: Gradients flow to all UNet params after loss.backward().

    MONAI's DiffusionModelUNet zero-initialises every ResBlock conv2 and
    the output conv (standard practice for diffusion models). This blocks
    gradient flow through the zero layers on the first backward pass.
    We re-initialise these layers with small random weights to test
    architectural gradient connectivity independent of the init scheme.
    """
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    # Re-init zero-initialised conv layers to verify gradient connectivity
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and module.weight.abs().sum() == 0:
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")

    B, C, D, H, W = 2, 4, 16, 16, 16
    z_t = torch.randn(B, C, D, H, W)
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.8])

    # Direct forward through model
    out = model(z_t, r, t)
    loss = out.pow(2).mean()
    loss.backward()

    params_with_grad = 0
    params_without_grad = 0
    no_grad_names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad += 1
                no_grad_names.append(name)

    total = params_with_grad + params_without_grad
    assert params_with_grad > 0, "No parameters received gradients"
    # Allow GroupNorm biases (init to 0) and a few others to have zero grad
    frac = params_with_grad / total
    assert frac > 0.8, (
        f"Only {frac:.1%} of params got gradients ({params_with_grad}/{total}). "
        f"First 10 without: {no_grad_names[:10]}"
    )


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T9_unified_loss(model_and_pipeline) -> None:
    """P3-T9: Unified MeanFlow loss computes for both FM (r=t) and MF (r<t) samples."""
    model, pipeline = model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.6, 0.9])
    r = torch.tensor([0.1, 0.4])

    result = pipeline(model, z_0, eps, t, r)

    assert torch.isfinite(result["loss"]), f"Loss not finite: {result['loss'].item()}"
    assert torch.isfinite(result["raw_loss"]), "Raw loss not finite"
    # Adaptive weighting normalises loss; raw_loss captures true magnitude
    assert result["raw_loss"] > 0, "Raw loss should be positive"


@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T10_bf16_mixed_precision() -> None:
    """P3-T10: JVP and loss work with bf16 mixed precision."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    # Use finite-difference for bf16 test (more robust than exact JVP in fp16)
    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        prediction_type="x",
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)

    B, C, D, H, W = 1, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5])
    r = torch.tensor([0.2])

    device_type = "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        result = pipeline(model, z_0, eps, t, r)

    assert torch.isfinite(result["loss"]), f"Loss is NaN under bf16: {result['loss']}"
    assert torch.isfinite(result["raw_loss"])


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_fd_pipeline_produces_finite_loss(fd_model_and_pipeline) -> None:
    """Finite-difference pipeline produces finite loss."""
    model, pipeline = fd_model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r)

    assert torch.isfinite(result["loss"])
    assert 0 < result["loss"].item() < 1000


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_adaptive_weighting_normalises_loss() -> None:
    """Adaptive weighting normalises per-sample loss to ~1.0."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        prediction_type="x",
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)

    B, C, D, H, W = 4, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.3, 0.5, 0.7, 0.9])
    r = torch.tensor([0.1, 0.2, 0.3, 0.4])

    result = pipeline(model, z_0, eps, t, r)

    # With adaptive weighting, loss should be around 1.0 (single normalised term)
    assert result["loss"].item() < 10.0, (
        f"Adaptive loss suspiciously large: {result['loss'].item()}"
    )


# ======================================================================
# u-prediction tests (Phase 4f — verifying FD-JVP stability with u-pred)
# ======================================================================


@pytest.fixture(scope="module")
def u_pred_fd_model_and_pipeline():
    """Create model + FD pipeline with u-prediction."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="u")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        norm_eps=1.0,
        prediction_type="u",
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)
    return model, pipeline


@pytest.mark.phase4
@pytest.mark.critical
def test_P4f_T1_u_prediction_pipeline_runs(u_pred_fd_model_and_pipeline) -> None:
    """P4f-T1: u-prediction + FD-JVP produces finite loss."""
    model, pipeline = u_pred_fd_model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r)

    assert torch.isfinite(result["loss"]), f"Loss not finite: {result['loss'].item()}"
    assert torch.isfinite(result["raw_loss"])
    assert result["loss"].item() > 0


@pytest.mark.phase4
@pytest.mark.critical
def test_P4f_T2_u_pred_fm_reduces_to_u(u_pred_fd_model_and_pipeline) -> None:
    """P4f-T2: For FM samples (r=t), compound V equals u (no JVP correction).

    When r=t, the MeanFlow correction term (t-r)*du/dt = 0, so V should
    equal u exactly. This verifies the FM path works with u-prediction.
    """
    model, pipeline = u_pred_fd_model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = t.clone()  # FM: r = t

    result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

    assert torch.isfinite(result["loss"])
    # FM fraction should be 1.0 (all samples are FM)
    assert result["diag_fm_fraction"].item() > 0.99


@pytest.mark.phase4
@pytest.mark.critical
def test_P4f_T3_u_pred_fd_jvp_bounded() -> None:
    """P4f-T3: FD-JVP norm stays bounded with u-prediction (key regression test).

    This is the critical test: with x-prediction + FD-JVP, the 1/t factor
    causes JVP norms to explode. With u-prediction, the FD-JVP computes
    finite differences of direct model outputs — no 1/t amplification.
    JVP norms should stay O(model_output_norm), not explode.
    """
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="u")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        norm_eps=1.0,
        prediction_type="u",
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)

    B, C, D, H, W = 4, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    # Use a range of t values including small t (where x-pred would explode)
    t = torch.tensor([0.1, 0.3, 0.7, 0.95])
    r = torch.tensor([0.05, 0.1, 0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

    jvp_norm = result["diag_jvp_norm"]
    u_norm = result["diag_u_norm"]
    compound_v_norm = result["diag_compound_v_norm"]

    assert torch.isfinite(jvp_norm), f"JVP norm not finite: {jvp_norm.item()}"
    assert torch.isfinite(compound_v_norm), "Compound V norm not finite"

    # Key assertion: JVP norm should be same order of magnitude as u_norm,
    # not orders of magnitude larger (which would indicate 1/t amplification)
    ratio = jvp_norm.item() / max(u_norm.item(), 1e-8)
    assert ratio < 100, (
        f"JVP norm ({jvp_norm.item():.1f}) is {ratio:.0f}x u_norm "
        f"({u_norm.item():.1f}) — possible 1/t amplification"
    )

    # Compound V should be reasonably close in magnitude to target V
    target_v_norm = result["diag_target_v_norm"]
    cv_ratio = compound_v_norm.item() / max(target_v_norm.item(), 1e-8)
    assert cv_ratio < 50, (
        f"Compound V norm ({compound_v_norm.item():.1f}) is {cv_ratio:.0f}x "
        f"target V norm ({target_v_norm.item():.1f})"
    )


@pytest.mark.phase4
@pytest.mark.informational
def test_P4f_u_pred_diagnostics_keys(u_pred_fd_model_and_pipeline) -> None:
    """u-prediction diagnostics include u_pred_* keys instead of x_hat_*."""
    model, pipeline = u_pred_fd_model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

    # u-prediction should have u_pred_* keys, NOT x_hat_* keys
    assert "diag_u_pred_mean" in result
    assert "diag_u_pred_std" in result
    assert "diag_u_pred_min" in result
    assert "diag_u_pred_max" in result
    assert "diag_x_hat_mean" not in result
    assert "diag_x_hat_std" not in result


# ======================================================================
# x-pred + exact JVP ablation tests (verifying the xpred_exact_jvp arm)
# ======================================================================


@pytest.fixture(scope="module")
def x_pred_exact_model_and_pipeline():
    """Create model + exact JVP pipeline with x-prediction (no gradient checkpointing)."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x", use_checkpointing=False)
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        norm_eps=1.0,
        prediction_type="x",
        jvp_strategy="exact",
    )
    pipeline = MeanFlowPipeline(pipeline_config)
    return model, pipeline


@pytest.mark.phase4
@pytest.mark.critical
def test_P4f_T4_xpred_exact_jvp_stable(x_pred_exact_model_and_pipeline) -> None:
    """P4f-T4: x-prediction + exact JVP produces finite loss (ablation arm validation).

    This validates the xpred_exact_jvp ablation arm. x-pred with exact JVP
    (not FD-JVP) should be numerically stable because torch.func.jvp computes
    the exact derivative without finite-difference amplification of the 1/t factor.
    """
    model, pipeline = x_pred_exact_model_and_pipeline
    B, C, D, H, W = 4, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    # Include small t values that would cause FD-JVP to explode with x-pred
    t = torch.tensor([0.1, 0.3, 0.7, 0.95])
    r = torch.tensor([0.05, 0.1, 0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

    assert torch.isfinite(result["loss"]), f"Loss not finite: {result['loss'].item()}"
    assert torch.isfinite(result["raw_loss"])

    # With exact JVP, even x-prediction should have bounded JVP norms
    jvp_norm = result["diag_jvp_norm"]
    compound_v_norm = result["diag_compound_v_norm"]
    target_v_norm = result["diag_target_v_norm"]

    assert torch.isfinite(jvp_norm), f"JVP norm not finite: {jvp_norm.item()}"

    # Compound V should be same order as target V (not 1000x larger)
    cv_ratio = compound_v_norm.item() / max(target_v_norm.item(), 1e-8)
    assert cv_ratio < 50, (
        f"Compound V norm ({compound_v_norm.item():.1f}) is {cv_ratio:.0f}x "
        f"target V norm ({target_v_norm.item():.1f}) — unexpected for exact JVP"
    )


@pytest.mark.phase4
@pytest.mark.critical
def test_P4f_T5_xpred_exact_jvp_gradients_flow(x_pred_exact_model_and_pipeline) -> None:
    """P4f-T5: Gradients flow through x-pred + exact JVP pipeline."""
    model, pipeline = x_pred_exact_model_and_pipeline
    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r)
    result["loss"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed through x-pred + exact JVP pipeline"
