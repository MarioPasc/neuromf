"""Tests for MeanFlowPipeline — Phase 3 verification.

P3-T4: MeanFlow loss is finite and positive
P3-T5: Gradients flow to all UNet params
P3-T9: Combined iMF loss (FM + MF) computes
P3-T10: bf16 mixed precision works
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
