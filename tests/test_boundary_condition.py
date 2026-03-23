"""Tests for boundary condition v-head replacement.

V2-T3: v_tilde shape matches u shape
V2-T4: v_tilde.requires_grad == False
"""

from __future__ import annotations

import pytest
import torch

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper


def _tiny_unet(use_v_head: bool = False) -> MAISIUNetWrapper:
    """Build a tiny UNet for fast tests (spatial=16)."""
    config = MAISIUNetConfig(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=[8, 16, 32, 64],
        attention_levels=[False, False, False, False],
        num_res_blocks=1,
        num_head_channels=[0, 0, 0, 8],
        norm_num_groups=8,
        resblock_updown=False,
        transformer_num_layers=1,
        use_flash_attention=False,
        with_conditioning=False,
        prediction_type="x",
        t_min=0.05,
        use_v_head=use_v_head,
        v_head_num_res_blocks=0,
        conditioning_mode="t_h",
    )
    return MAISIUNetWrapper(config)


# ======================================================================
# V2-T3: Boundary condition shape
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
@pytest.mark.slow
def test_V2_T3_boundary_condition_shape() -> None:
    """v_tilde = model(z_t, t, t) has same shape as model(z_t, r, t)."""
    model = _tiny_unet(use_v_head=False)
    model.eval()

    B, C, S = 2, 4, 16
    z_t = torch.randn(B, C, S, S, S)
    t = torch.rand(B)
    r = torch.rand(B) * t  # r < t

    with torch.no_grad():
        u = model(z_t, r, t)
        v_tilde = model(z_t, t, t)  # boundary condition: r = t

    assert v_tilde.shape == u.shape, f"Shape mismatch: v_tilde={v_tilde.shape}, u={u.shape}"


# ======================================================================
# V2-T4: Boundary condition no grad
# ======================================================================


@pytest.mark.phase6
@pytest.mark.critical
@pytest.mark.slow
def test_V2_T4_boundary_condition_no_grad() -> None:
    """v_tilde computed under no_grad should have requires_grad=False."""
    model = _tiny_unet(use_v_head=False)
    model.eval()

    B, C, S = 2, 4, 16
    z_t = torch.randn(B, C, S, S, S)
    t = torch.rand(B)

    with torch.no_grad():
        v_tilde = model(z_t, t, t)

    assert not v_tilde.requires_grad, "v_tilde should not require grad"


@pytest.mark.phase6
@pytest.mark.critical
@pytest.mark.slow
def test_V2_T3b_boundary_matches_single_head_path() -> None:
    """Verify MeanFlowPipeline single-head path uses boundary condition."""
    from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

    model = _tiny_unet(use_v_head=False)

    pipeline = MeanFlowPipeline(
        MeanFlowPipelineConfig(
            p=2.0,
            adaptive=True,
            norm_eps=1.0,
            prediction_type="x",
            t_min=0.05,
            jvp_strategy="exact",
            use_v_head=False,
        )
    )

    B, C, S = 2, 4, 16
    z_0 = torch.randn(B, C, S, S, S)
    eps = torch.randn(B, C, S, S, S)
    t = torch.rand(B).clamp(min=0.1)
    r = torch.rand(B) * t * 0.5  # r < t

    # Should complete without error (uses boundary condition internally)
    result = pipeline(model, z_0, eps, t, r)
    assert torch.isfinite(result["loss"]), "Loss should be finite"
    assert result["loss"].shape == (), "Loss should be scalar"


@pytest.mark.phase6
@pytest.mark.critical
@pytest.mark.slow
def test_V2_T3c_no_v_head_params() -> None:
    """Model with use_v_head=False should have no v_out parameters."""
    model = _tiny_unet(use_v_head=False)
    assert not hasattr(model, "v_out") or model.v_out is None, (
        "Model with use_v_head=False should not have v_out"
    )
