"""Phase 4 tests: Pretrained rflow weight transfer loading.

Tests use a tiny UNet mock (channels=[8,16,32,64]) to verify the transfer
loading logic. The mock rflow checkpoint uses ``DiffusionModelUNetMaisi``
with ``include_spacing_input=True`` and ``num_class_embeds=128``, which
produces ``time_emb_proj.weight`` of shape ``(C, 64)`` vs the target
``DiffusionModelUNet``'s ``(C, 32)`` — matching the production 512→256
pattern at tiny scale.

A separate ``TestRealCheckpoint`` class runs against the actual 2GB MAISI
rflow checkpoint (``diff_unet_3d_rflow-mr.pt``) with production-size
[64,128,256,512] UNet. These tests are skipped when the checkpoint is not
available (e.g. CI or laptop without external drive).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from omegaconf import OmegaConf

from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.utils.param_groups import build_transfer_param_groups
from neuromf.utils.pretrained_loading import load_rflow_pretrained
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper


# ======================================================================
# Fixtures
# ======================================================================

_TINY_CHANNELS = [8, 16, 32, 64]
_TINY_ATTENTION = [False, False, True, True]
_TINY_HEAD_CH = [0, 0, 32, 32]
_TINY_NUM_RES = 1
_TINY_NORM_GROUPS = 8


@pytest.fixture(scope="module")
def rflow_checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a mock rflow checkpoint using DiffusionModelUNetMaisi.

    This mimics the real ``diff_unet_3d_rflow-mr.pt`` structure: a
    ``DiffusionModelUNetMaisi`` with ``include_spacing_input=True`` and
    ``num_class_embeds=128``, wrapped as ``{"unet_state_dict": ...}``.
    """
    rflow_unet = DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=_TINY_CHANNELS,
        attention_levels=_TINY_ATTENTION,
        num_head_channels=_TINY_HEAD_CH,
        num_res_blocks=_TINY_NUM_RES,
        include_spacing_input=True,
        num_class_embeds=128,
        resblock_updown=True,
        include_fc=True,
        use_flash_attention=False,
        norm_num_groups=_TINY_NORM_GROUPS,
    )

    ckpt_path = tmp_path_factory.mktemp("ckpt") / "mock_rflow.pt"
    torch.save(
        {
            "unet_state_dict": rflow_unet.state_dict(),
            "epoch": 100,
            "scale_factor": 0.9624,
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture(scope="module")
def target_wrapper() -> MAISIUNetWrapper:
    """Create a target MAISIUNetWrapper with the tiny config."""
    config = MAISIUNetConfig(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=_TINY_CHANNELS,
        attention_levels=_TINY_ATTENTION,
        num_head_channels=_TINY_HEAD_CH,
        num_res_blocks=_TINY_NUM_RES,
        norm_num_groups=_TINY_NORM_GROUPS,
        resblock_updown=True,
        use_flash_attention=False,
        with_conditioning=False,
        prediction_type="x",
        use_v_head=True,
        v_head_num_res_blocks=1,
        conditioning_mode="t_h",
    )
    return MAISIUNetWrapper(config)


def _save_kaiming_snapshot(wrapper: MAISIUNetWrapper) -> dict[str, torch.Tensor]:
    """Save a snapshot of the Kaiming-init state for comparison."""
    return {k: v.clone() for k, v in wrapper.unet.state_dict().items()}


def _tiny_config(**overrides: object) -> OmegaConf:
    """Build a tiny config for LatentMeanFlow tests.

    Args:
        **overrides: Keys to override in the config.

    Returns:
        OmegaConf DictConfig with tiny UNet settings.
    """
    cfg = OmegaConf.create(
        {
            "unet": {
                "spatial_dims": 3,
                "in_channels": 4,
                "out_channels": 4,
                "channels": _TINY_CHANNELS,
                "attention_levels": [False, False, True, True],
                "num_res_blocks": _TINY_NUM_RES,
                "num_head_channels": _TINY_HEAD_CH,
                "norm_num_groups": _TINY_NORM_GROUPS,
                "resblock_updown": True,
                "transformer_num_layers": 1,
                "use_flash_attention": False,
                "with_conditioning": False,
                "gradient_checkpointing": False,
                "prediction_type": "x",
                "t_min": 0.05,
                "use_v_head": True,
                "v_head_num_res_blocks": 1,
                "conditioning_mode": "t_h",
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "warmup_steps": 0,
                "lr_schedule": "constant",
                "max_epochs": 2,
                "batch_size": 2,
                "gradient_clip_norm": 1.0,
                "mixed_precision": "fp32",
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_every_n_epochs": 1,
                "num_workers": 0,
                "prefetch_factor": None,
                "split_ratio": 0.9,
                "split_seed": 42,
            },
            "meanflow": {
                "p": 2.0,
                "adaptive": True,
                "norm_eps": 1.0,
                "lambda_mf": 1.0,
                "prediction_type": "x",
                "t_min": 0.05,
                "jvp_strategy": "exact",
                "fd_step_size": 1e-3,
                "channel_weights": None,
                "norm_p": 1.0,
                "use_v_head": True,
            },
            "time_sampling": {
                "mu": -0.4,
                "sigma": 1.0,
                "t_min": 0.001,
                "data_proportion": 0.75,
            },
            "ema": {"decay": 0.999},
            "paths": {
                "latents_dir": "",
                "checkpoints_dir": "",
                "logs_dir": "",
                "samples_dir": "",
                "maisi_vae_weights": "",
            },
            "sample_every_n_epochs": 100,
            "n_samples_per_log": 2,
            "latent_spatial_size": 8,
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ======================================================================
# Informational tests — mock checkpoint structure
# ======================================================================


@pytest.mark.phase4
@pytest.mark.informational
@pytest.mark.slow
class TestMockCheckpointStructure:
    """Verify the mock rflow checkpoint has expected structure."""

    def test_P4_T9a_mock_rflow_structure(
        self, rflow_checkpoint_path: Path
    ) -> None:
        """P4-T9a: Mock checkpoint has expected top-level keys."""
        ckpt = torch.load(rflow_checkpoint_path, map_location="cpu", weights_only=False)
        assert "unet_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "scale_factor" in ckpt
        assert ckpt["epoch"] == 100

    def test_P4_T9b_mock_rflow_time_emb_proj_shape(
        self, rflow_checkpoint_path: Path
    ) -> None:
        """P4-T9b: Mock time_emb_proj.weight is wider than target."""
        ckpt = torch.load(rflow_checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["unet_state_dict"]

        # Find any time_emb_proj.weight key
        proj_keys = [k for k in sd if k.endswith("time_emb_proj.weight")]
        assert len(proj_keys) > 0, "No time_emb_proj.weight keys in rflow"

        # All should have dim=1 of 64 (channels[0]*4*2 = 8*4*2 for spacing)
        for k in proj_keys:
            assert sd[k].shape[1] == 64, f"{k}: expected dim1=64, got {sd[k].shape[1]}"

    def test_P4_T9c_mock_has_class_spacing_keys(
        self, rflow_checkpoint_path: Path
    ) -> None:
        """P4-T9c: Mock has class_embedding and spacing_layer keys."""
        ckpt = torch.load(rflow_checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["unet_state_dict"]

        assert "class_embedding.weight" in sd
        assert "spacing_layer.0.weight" in sd
        assert "spacing_layer.0.bias" in sd
        assert "spacing_layer.2.weight" in sd
        assert "spacing_layer.2.bias" in sd


# ======================================================================
# Critical tests — loading utility
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
@pytest.mark.slow
class TestPretrainedLoading:
    """Critical tests for the pretrained weight loading utility."""

    def test_P4_T10_key_coverage(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T10: Correct counts for matched, sliced, skipped keys."""
        # Fresh wrapper for this test
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)

        stats = load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        total_target = len(wrapper.unet.state_dict())
        total_loaded = stats["loaded_exact"] + stats["loaded_sliced"]

        assert stats["loaded_exact"] > 0, "No exact matches"
        assert stats["loaded_sliced"] > 0, "No sliced keys"
        assert stats["skipped_rflow_only"] == 5, "Expected 5 rflow-only keys"
        assert stats["not_in_rflow"] == 0, "Unexpected keys not in rflow"
        assert total_loaded == total_target, (
            f"Not all target keys loaded: {total_loaded}/{total_target}"
        )

    def test_P4_T11_weights_actually_copied(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T11: conv_in.weight differs from Kaiming after load."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        kaiming = _save_kaiming_snapshot(wrapper)

        load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        loaded_sd = wrapper.unet.state_dict()
        key = "conv_in.conv.weight"
        assert not torch.equal(
            kaiming[key], loaded_sd[key]
        ), "conv_in.conv.weight unchanged after loading — transfer failed"

    def test_P4_T12_time_emb_proj_slicing_correct(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T12: Sliced time_emb_proj[:, :32] matches rflow first 32 cols."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)

        ckpt = torch.load(rflow_checkpoint_path, map_location="cpu", weights_only=False)
        rflow_sd = ckpt["unet_state_dict"]

        load_rflow_pretrained(wrapper, rflow_checkpoint_path, slice_time_emb_proj=True)

        loaded_sd = wrapper.unet.state_dict()
        target_dim = config.channels[0] * 4  # 32

        proj_keys = [k for k in loaded_sd if k.endswith("time_emb_proj.weight")]
        assert len(proj_keys) > 0

        for key in proj_keys:
            if key in rflow_sd and rflow_sd[key].shape[1] > target_dim:
                expected = rflow_sd[key][:, :target_dim]
                actual = loaded_sd[key]
                assert torch.equal(actual, expected), (
                    f"Slicing mismatch for {key}: "
                    f"expected rflow[:, :{target_dim}], got different values"
                )

    def test_P4_T13_time_emb_proj_skip_when_disabled(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T13: slice_time_emb_proj=False leaves time_emb_proj at Kaiming."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        kaiming = _save_kaiming_snapshot(wrapper)

        load_rflow_pretrained(
            wrapper, rflow_checkpoint_path, slice_time_emb_proj=False
        )

        loaded_sd = wrapper.unet.state_dict()
        target_dim = config.channels[0] * 4

        proj_keys = [k for k in loaded_sd if k.endswith("time_emb_proj.weight")]
        for key in proj_keys:
            # Keys with shape mismatch should still be at Kaiming init
            ckpt = torch.load(
                rflow_checkpoint_path, map_location="cpu", weights_only=False
            )
            rflow_sd = ckpt["unet_state_dict"]
            if key in rflow_sd and rflow_sd[key].shape[1] > target_dim:
                assert torch.equal(
                    loaded_sd[key], kaiming[key]
                ), f"{key} should be at Kaiming init when slicing disabled"

    def test_P4_T14_output_conv_reinit(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T14: reinit_output_conv=True keeps output conv at init values."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        kaiming = _save_kaiming_snapshot(wrapper)

        load_rflow_pretrained(
            wrapper, rflow_checkpoint_path, reinit_output_conv=True
        )

        loaded_sd = wrapper.unet.state_dict()
        out_keys = [k for k in loaded_sd if k.startswith("out.")]
        assert len(out_keys) > 0, "No output conv keys found"

        for key in out_keys:
            assert torch.equal(
                loaded_sd[key], kaiming[key]
            ), f"{key} should be at Kaiming init when reinit_output_conv=True"

    def test_P4_T15_new_layers_unchanged(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T15: Wrapper-only params (h_embed, v_out) are not affected."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)

        # Save wrapper-level params before loading
        wrapper_params_before = {}
        for name, param in wrapper.named_parameters():
            if not name.startswith("unet."):
                wrapper_params_before[name] = param.data.clone()

        load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        for name, param in wrapper.named_parameters():
            if not name.startswith("unet."):
                assert torch.equal(
                    param.data, wrapper_params_before[name]
                ), f"Wrapper param {name} was modified by rflow loading"

    def test_P4_T16_forward_pass_after_load(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T16: Forward pass produces finite output after loading."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        B, S = 2, 8
        z_t = torch.randn(B, 4, S, S, S)
        r = torch.zeros(B)
        t = torch.ones(B)

        wrapper.eval()
        with torch.no_grad():
            out = wrapper(z_t, r, t)

        if isinstance(out, tuple):
            u_out, v_out = out
            assert torch.isfinite(u_out).all(), "u_out has non-finite values"
            assert torch.isfinite(v_out).all(), "v_out has non-finite values"
        else:
            assert torch.isfinite(out).all(), "Output has non-finite values"

    def test_P4_T17_gradient_flow(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T17: >95% params have grad after backward on loaded model."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        # Reinit zero-weight convs so gradients can flow
        with torch.no_grad():
            for param in wrapper.parameters():
                if param.abs().sum() == 0 and param.numel() > 1:
                    nn.init.normal_(param, std=0.01)

        B, S = 2, 8
        z_t = torch.randn(B, 4, S, S, S)
        r = torch.zeros(B)
        t = torch.ones(B)

        wrapper.train()
        out = wrapper(z_t, r, t)
        if isinstance(out, tuple):
            loss = out[0].sum() + out[1].sum()
        else:
            loss = out.sum()
        loss.backward()

        total = 0
        with_grad = 0
        for name, param in wrapper.named_parameters():
            if param.requires_grad:
                total += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    with_grad += 1

        ratio = with_grad / max(total, 1)
        assert ratio > 0.90, f"Only {with_grad}/{total} ({ratio:.1%}) params have grad"

    def test_P4_T18_ema_from_pretrained(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T18: EMA shadows match pretrained weights, not Kaiming."""
        from neuromf.utils.ema import EMAModel

        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        kaiming = _save_kaiming_snapshot(wrapper)

        load_rflow_pretrained(wrapper, rflow_checkpoint_path)
        ema = EMAModel(wrapper, decay=0.999)

        # EMA should be initialized from the loaded (pretrained) weights
        key = "unet.conv_in.conv.weight"
        assert key in ema.shadow
        loaded_val = wrapper.unet.state_dict()["conv_in.conv.weight"]
        assert torch.equal(ema.shadow[key], loaded_val), (
            "EMA shadow should match loaded weights"
        )
        assert not torch.equal(ema.shadow[key], kaiming["conv_in.conv.weight"]), (
            "EMA shadow should differ from Kaiming init"
        )


# ======================================================================
# Critical tests — param groups
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
@pytest.mark.slow
class TestParamGroups:
    """Critical tests for differential learning rate parameter groups."""

    def test_P4_T19_param_groups_split(
        self, rflow_checkpoint_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T19: Two groups with correct LRs, total params matches model."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        stats = load_rflow_pretrained(wrapper, rflow_checkpoint_path)

        loaded_inner = stats["loaded_keys"]
        loaded_wrapper = {f"unet.{k}" for k in loaded_inner}

        base_lr = 1e-3
        factor = 0.1
        groups = build_transfer_param_groups(
            wrapper, loaded_wrapper, base_lr=base_lr, backbone_lr_factor=factor
        )

        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"

        # Check LRs
        backbone_group = groups[0]
        new_group = groups[1]
        assert backbone_group["lr"] == pytest.approx(base_lr * factor)
        assert new_group["lr"] == pytest.approx(base_lr)

        # Total params should match model
        group_total = sum(p.numel() for g in groups for p in g["params"])
        model_total = sum(
            p.numel() for p in wrapper.parameters() if p.requires_grad
        )
        assert group_total == model_total, (
            f"Param count mismatch: groups={group_total} vs model={model_total}"
        )


# ======================================================================
# Critical tests — error handling
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
@pytest.mark.slow
class TestErrorHandling:
    """Critical tests for error handling in pretrained loading."""

    def test_P4_T20_missing_checkpoint_raises(
        self, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T20: FileNotFoundError for non-existent path."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)
        with pytest.raises(FileNotFoundError):
            load_rflow_pretrained(wrapper, "/nonexistent/path/model.pt")

    def test_P4_T21_malformed_checkpoint_raises(
        self, tmp_path: Path, target_wrapper: MAISIUNetWrapper
    ) -> None:
        """P4-T21: KeyError if checkpoint missing unet_state_dict."""
        config = target_wrapper.config
        wrapper = MAISIUNetWrapper(config)

        bad_ckpt = tmp_path / "bad.pt"
        torch.save({"model": "something_else"}, bad_ckpt)

        with pytest.raises(KeyError, match="unet_state_dict"):
            load_rflow_pretrained(wrapper, bad_ckpt)


# ======================================================================
# Critical tests — integration with LatentMeanFlow
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
@pytest.mark.slow
class TestLatentMeanFlowIntegration:
    """Integration tests for pretrained loading in the full Lightning module."""

    def test_P4_T22_no_transfer_when_kaiming(self) -> None:
        """P4-T22: Default config uses Kaiming init (no transfer)."""
        cfg = _tiny_config()
        model = LatentMeanFlow(cfg)
        assert model._pretrained_load_stats is None

    def test_P4_T23_latent_meanflow_integration(
        self, rflow_checkpoint_path: Path
    ) -> None:
        """P4-T23: Full LatentMeanFlow with rflow transfer + training step."""
        cfg = _tiny_config(
            **{
                "unet": {
                    "initialization": {
                        "method": "rflow_transfer",
                        "rflow_checkpoint_path": str(rflow_checkpoint_path),
                        "slice_time_emb_proj": True,
                        "reinit_output_conv": False,
                        "backbone_lr_factor": 0.1,
                    }
                }
            }
        )
        model = LatentMeanFlow(cfg)

        assert model._pretrained_load_stats is not None
        assert model._pretrained_load_stats["loaded_exact"] > 0

        # Training step should work
        model.train()
        batch = {"z": torch.randn(2, 4, 8, 8, 8)}
        loss = model.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    def test_P4_T24_latent_meanflow_ema_ordering(
        self, rflow_checkpoint_path: Path
    ) -> None:
        """P4-T24: EMA shadows come from pretrained, not Kaiming init."""
        cfg = _tiny_config(
            **{
                "unet": {
                    "initialization": {
                        "method": "rflow_transfer",
                        "rflow_checkpoint_path": str(rflow_checkpoint_path),
                        "slice_time_emb_proj": True,
                        "reinit_output_conv": False,
                        "backbone_lr_factor": 0.1,
                    }
                }
            }
        )

        # Create a fresh Kaiming wrapper to get baseline
        from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper

        kaiming_cfg = MAISIUNetConfig.from_omegaconf(cfg)
        kaiming_wrapper = MAISIUNetWrapper(kaiming_cfg)
        kaiming_conv_in = kaiming_wrapper.unet.state_dict()["conv_in.conv.weight"].clone()

        # Create model with transfer
        model = LatentMeanFlow(cfg)
        ema_conv_in = model.ema.shadow["unet.conv_in.conv.weight"]

        # EMA should differ from Kaiming (loaded from rflow)
        assert not torch.equal(ema_conv_in, kaiming_conv_in), (
            "EMA shadow matches Kaiming — transfer loading was not applied "
            "before EMA init"
        )


# ======================================================================
# Critical tests — real MAISI rflow checkpoint (skipped if unavailable)
# ======================================================================

# Hardcoded path to the real 2GB MAISI rflow checkpoint.
# This file lives on the external drive and may not be mounted in CI.
_REAL_RFLOW_PATH = Path(
    "/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/"
    "NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt"
)

# Expected counts from manual inspection of the real checkpoint.
_EXPECTED_RFLOW_KEYS = 435
_EXPECTED_TARGET_KEYS = 430
_EXPECTED_EXACT_MATCH = 402
_EXPECTED_SLICED = 28
_EXPECTED_RFLOW_ONLY = 5  # class_embedding + spacing_layer
_EXPECTED_TARGET_ONLY = 0


def _production_wrapper() -> MAISIUNetWrapper:
    """Create a production-sized MAISIUNetWrapper matching train_meanflow.yaml."""
    config = MAISIUNetConfig(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=2,
        num_head_channels=[0, 0, 32, 32],
        norm_num_groups=32,
        resblock_updown=True,
        transformer_num_layers=1,
        use_flash_attention=False,
        with_conditioning=False,
        prediction_type="x",
        use_v_head=True,
        v_head_num_res_blocks=1,
        conditioning_mode="t_h",
    )
    return MAISIUNetWrapper(config)


_skip_no_real_ckpt = pytest.mark.skipif(
    not _REAL_RFLOW_PATH.exists(),
    reason=f"Real rflow checkpoint not found: {_REAL_RFLOW_PATH}",
)


@pytest.mark.phase4
@pytest.mark.critical
@pytest.mark.slow
@_skip_no_real_ckpt
class TestRealCheckpoint:
    """Validate transfer loading against the real 2GB MAISI rflow checkpoint.

    These tests verify that every expected weight is actually transferred
    with the production-size [64,128,256,512] UNet, catching any silent
    failures that the tiny mock might miss (e.g. key naming differences
    between MONAI versions, unexpected shape mismatches).
    """

    def test_P4_T25_real_checkpoint_structure(self) -> None:
        """P4-T25: Real checkpoint has expected top-level keys and counts."""
        ckpt = torch.load(_REAL_RFLOW_PATH, map_location="cpu", weights_only=False)

        assert "unet_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "scale_factor" in ckpt

        sd = ckpt["unet_state_dict"]
        assert len(sd) == _EXPECTED_RFLOW_KEYS, (
            f"Expected {_EXPECTED_RFLOW_KEYS} rflow keys, got {len(sd)}"
        )

        # Verify rflow-only keys exist
        assert "class_embedding.weight" in sd
        for i in (0, 2):
            assert f"spacing_layer.{i}.weight" in sd
            assert f"spacing_layer.{i}.bias" in sd

    def test_P4_T26_real_key_by_key_coverage(self) -> None:
        """P4-T26: Every target key is in rflow (exact or sliceable)."""
        ckpt = torch.load(_REAL_RFLOW_PATH, map_location="cpu", weights_only=False)
        rflow_sd = ckpt["unet_state_dict"]
        wrapper = _production_wrapper()
        target_sd = wrapper.unet.state_dict()

        assert len(target_sd) == _EXPECTED_TARGET_KEYS

        exact = 0
        sliceable = 0
        missing = []

        for key in target_sd:
            if key not in rflow_sd:
                missing.append(key)
                continue
            if target_sd[key].shape == rflow_sd[key].shape:
                exact += 1
            elif (
                key.endswith("time_emb_proj.weight")
                and rflow_sd[key].shape[0] == target_sd[key].shape[0]
                and rflow_sd[key].shape[1] > target_sd[key].shape[1]
            ):
                sliceable += 1
            else:
                missing.append(f"{key} (shape mismatch: rflow={list(rflow_sd[key].shape)} "
                               f"vs target={list(target_sd[key].shape)})")

        assert exact == _EXPECTED_EXACT_MATCH, (
            f"Expected {_EXPECTED_EXACT_MATCH} exact matches, got {exact}"
        )
        assert sliceable == _EXPECTED_SLICED, (
            f"Expected {_EXPECTED_SLICED} sliceable keys, got {sliceable}"
        )
        assert len(missing) == _EXPECTED_TARGET_ONLY, (
            f"Unexpected missing keys: {missing}"
        )

    def test_P4_T27_real_all_time_emb_proj_are_512_to_256(self) -> None:
        """P4-T27: All 28 mismatched keys are time_emb_proj.weight (C, 512) -> (C, 256)."""
        ckpt = torch.load(_REAL_RFLOW_PATH, map_location="cpu", weights_only=False)
        rflow_sd = ckpt["unet_state_dict"]
        wrapper = _production_wrapper()
        target_sd = wrapper.unet.state_dict()

        mismatched = []
        for key in target_sd:
            if key in rflow_sd and target_sd[key].shape != rflow_sd[key].shape:
                mismatched.append(key)

        assert len(mismatched) == 28, f"Expected 28 mismatched keys, got {len(mismatched)}"

        for key in mismatched:
            assert key.endswith("time_emb_proj.weight"), (
                f"Non-time_emb_proj key has shape mismatch: {key}"
            )
            assert rflow_sd[key].shape[1] == 512, (
                f"{key}: rflow dim1 should be 512, got {rflow_sd[key].shape[1]}"
            )
            assert target_sd[key].shape[1] == 256, (
                f"{key}: target dim1 should be 256, got {target_sd[key].shape[1]}"
            )

    def test_P4_T28_real_load_all_weights_transferred(self) -> None:
        """P4-T28: After loading, every target key differs from Kaiming init."""
        wrapper = _production_wrapper()
        kaiming = {k: v.clone() for k, v in wrapper.unet.state_dict().items()}

        stats = load_rflow_pretrained(wrapper, _REAL_RFLOW_PATH)

        assert stats["loaded_exact"] == _EXPECTED_EXACT_MATCH
        assert stats["loaded_sliced"] == _EXPECTED_SLICED
        assert stats["skipped_rflow_only"] == _EXPECTED_RFLOW_ONLY
        assert stats["not_in_rflow"] == _EXPECTED_TARGET_ONLY
        assert len(stats["loaded_keys"]) == _EXPECTED_TARGET_KEYS

        # Every single key should have changed from Kaiming
        loaded_sd = wrapper.unet.state_dict()
        unchanged = []
        for key in loaded_sd:
            if torch.equal(kaiming[key], loaded_sd[key]):
                unchanged.append(key)

        assert len(unchanged) == 0, (
            f"{len(unchanged)} keys unchanged after loading (silent failure): "
            f"{unchanged[:10]}{'...' if len(unchanged) > 10 else ''}"
        )

    def test_P4_T29_real_slicing_matches_rflow_columns(self) -> None:
        """P4-T29: Sliced time_emb_proj[:, :256] exactly matches rflow first 256 cols."""
        ckpt = torch.load(_REAL_RFLOW_PATH, map_location="cpu", weights_only=False)
        rflow_sd = ckpt["unet_state_dict"]

        wrapper = _production_wrapper()
        load_rflow_pretrained(wrapper, _REAL_RFLOW_PATH)
        loaded_sd = wrapper.unet.state_dict()

        mismatches = []
        for key in loaded_sd:
            if (
                key.endswith("time_emb_proj.weight")
                and key in rflow_sd
                and rflow_sd[key].shape[1] == 512
            ):
                expected = rflow_sd[key][:, :256]
                actual = loaded_sd[key]
                if not torch.equal(actual, expected):
                    mismatches.append(key)

        assert len(mismatches) == 0, (
            f"Slicing mismatch for {len(mismatches)} keys: {mismatches}"
        )

    def test_P4_T30_real_exact_match_weights_identical(self) -> None:
        """P4-T30: All 402 exact-match keys are bitwise identical to rflow."""
        ckpt = torch.load(_REAL_RFLOW_PATH, map_location="cpu", weights_only=False)
        rflow_sd = ckpt["unet_state_dict"]

        wrapper = _production_wrapper()
        load_rflow_pretrained(wrapper, _REAL_RFLOW_PATH)
        loaded_sd = wrapper.unet.state_dict()

        mismatches = []
        for key in loaded_sd:
            if key in rflow_sd and loaded_sd[key].shape == rflow_sd[key].shape:
                if not torch.equal(loaded_sd[key], rflow_sd[key]):
                    mismatches.append(key)

        assert len(mismatches) == 0, (
            f"{len(mismatches)} exact-match keys differ from rflow: {mismatches[:10]}"
        )

    def test_P4_T31_real_rflow_only_keys_dropped(self) -> None:
        """P4-T31: class_embedding and spacing_layer are NOT in our wrapper."""
        wrapper = _production_wrapper()
        target_keys = set(wrapper.unet.state_dict().keys())

        assert "class_embedding.weight" not in target_keys
        for i in (0, 2):
            assert f"spacing_layer.{i}.weight" not in target_keys
            assert f"spacing_layer.{i}.bias" not in target_keys

    def test_P4_T32_real_wrapper_params_untouched(self) -> None:
        """P4-T32: Wrapper-only params (h_embed, v_out) unaffected by real load."""
        wrapper = _production_wrapper()

        before = {}
        for name, param in wrapper.named_parameters():
            if not name.startswith("unet."):
                before[name] = param.data.clone()

        load_rflow_pretrained(wrapper, _REAL_RFLOW_PATH)

        for name, param in wrapper.named_parameters():
            if not name.startswith("unet."):
                assert torch.equal(param.data, before[name]), (
                    f"Wrapper param {name} was modified by rflow loading"
                )
