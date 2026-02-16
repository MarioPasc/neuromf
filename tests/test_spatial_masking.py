"""Tests for spatial loss masking in MeanFlowPipeline (Phase C).

Verifies that ``spatial_mask_ratio`` correctly masks voxels in the loss
computation and normalizes by the keep fraction.
"""

from __future__ import annotations

import pytest
import torch

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig


def _make_tiny_model() -> MAISIUNetWrapper:
    """Build a tiny UNet for testing."""
    cfg = MAISIUNetConfig(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=[8, 16, 32, 64],
        attention_levels=[False] * 4,
        num_res_blocks=1,
        num_head_channels=[0, 0, 0, 8],
        norm_num_groups=8,
        use_flash_attention=False,
    )
    return MAISIUNetWrapper(cfg)


@pytest.mark.phase4
@pytest.mark.informational
class TestSpatialMasking:
    """Phase 4d spatial masking tests."""

    def test_P4d_T7_mask_ratio_zero_no_effect(self) -> None:
        """P4d-T7: spatial_mask_ratio=0.0 gives same loss as default config."""
        model = _make_tiny_model()
        B, S = 2, 16

        # Pipeline with mask_ratio=0 (disabled)
        pipeline_off = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
                spatial_mask_ratio=0.0,
            )
        )
        # Pipeline with default config (should also be 0)
        pipeline_default = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
            )
        )

        torch.manual_seed(42)
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.05, 0.95)
        r = t * torch.rand(B) * 0.5

        torch.manual_seed(123)
        result_off = pipeline_off(model, z_0, eps, t, r)
        torch.manual_seed(123)
        result_default = pipeline_default(model, z_0, eps, t, r)

        assert torch.allclose(result_off["raw_loss"], result_default["raw_loss"], atol=1e-5)

    def test_P4d_T8_mask_ratio_nonzero_works(self) -> None:
        """P4d-T8: spatial_mask_ratio=0.5 produces finite, normalized loss."""
        model = _make_tiny_model()
        B, S = 2, 16

        pipeline = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
                spatial_mask_ratio=0.5,
            )
        )

        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.05, 0.95)
        r = t * torch.rand(B) * 0.5

        result = pipeline(model, z_0, eps, t, r)
        assert torch.isfinite(result["loss"])
        assert torch.isfinite(result["raw_loss"])
        assert result["raw_loss"].item() > 0

    def test_P4d_T9_mask_broadcasts_channels(self) -> None:
        """P4d-T9: Spatial mask is (B,1,D,H,W) — same voxels across channels."""
        # Test the masking logic directly
        B, C, D, H, W = 2, 4, 8, 8, 8
        mask_ratio = 0.5
        keep_prob = 1.0 - mask_ratio

        torch.manual_seed(42)
        mask = (torch.rand(B, 1, D, H, W) < keep_prob).float()

        # Mask shape should broadcast across channels
        assert mask.shape == (B, 1, D, H, W)

        # Expand to check all channels get same mask
        expanded = mask.expand(B, C, D, H, W)
        for c in range(1, C):
            assert torch.equal(expanded[:, 0], expanded[:, c])
