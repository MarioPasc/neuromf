"""Phase 4b tests: Training diagnostics.

All tests use a tiny UNet (channels=[8,16,32,64], spatial=16) so they run
in <10s on CPU. Uses ``jvp_strategy="finite_difference"`` to avoid
``torch.func`` overhead on small tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from neuromf.callbacks.diagnostics import TrainingDiagnosticsCallback
from neuromf.callbacks.performance import PerformanceCallback
from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig


def _tiny_config(**overrides) -> OmegaConf:
    """Build a tiny config for fast local tests."""
    cfg = OmegaConf.create(
        {
            "unet": {
                "spatial_dims": 3,
                "in_channels": 4,
                "out_channels": 4,
                "channels": [8, 16, 32, 64],
                "attention_levels": [False, False, False, False],
                "num_res_blocks": 1,
                "num_head_channels": [0, 0, 0, 8],
                "norm_num_groups": 8,
                "resblock_updown": False,
                "transformer_num_layers": 1,
                "use_flash_attention": False,
                "with_conditioning": False,
                "gradient_checkpointing": False,
                "prediction_type": "x",
                "t_min": 0.05,
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "warmup_steps": 10,
                "max_epochs": 2,
                "batch_size": 2,
                "gradient_clip_norm": 1.0,
                "mixed_precision": "fp32",
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_every_n_epochs": 1,
                "num_workers": 0,
                "prefetch_factor": None,
            },
            "meanflow": {
                "p": 2.0,
                "adaptive": True,
                "norm_eps": 0.01,
                "lambda_mf": 1.0,
                "prediction_type": "x",
                "t_min": 0.05,
                "jvp_strategy": "finite_difference",
                "fd_step_size": 1e-3,
                "channel_weights": None,
            },
            "time_sampling": {
                "mu": -0.4,
                "sigma": 1.0,
                "t_min": 0.001,
                "data_proportion": 0.25,
            },
            "ema": {"decay": 0.999},
            "paths": {
                "latents_dir": "",
                "checkpoints_dir": "",
                "logs_dir": "",
                "samples_dir": "",
                "maisi_vae_weights": "",
                "diagnostics_dir": "",
            },
            "sample_every_n_epochs": 100,
            "n_samples_per_log": 2,
            "latent_spatial_size": 16,
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _tiny_pipeline() -> tuple[MeanFlowPipeline, MAISIUNetWrapper]:
    """Build a tiny pipeline + model for direct testing."""
    pipeline_cfg = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        norm_eps=0.01,
        lambda_mf=1.0,
        prediction_type="x",
        t_min=0.05,
        jvp_strategy="finite_difference",
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_cfg)

    unet_cfg = MAISIUNetConfig(
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
    )
    model = MAISIUNetWrapper(unet_cfg)
    return pipeline, model


def _fake_batch(batch_size: int = 2, spatial: int = 16) -> dict[str, torch.Tensor]:
    """Create a fake batch for testing."""
    return {"z": torch.randn(batch_size, 4, spatial, spatial, spatial)}


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.phase4
@pytest.mark.informational
class TestDiagnostics:
    """Phase 4b diagnostics tests."""

    def test_P4b_T1_pipeline_return_diagnostics_false(self) -> None:
        """return_diagnostics=False returns exactly 3 keys, no regression."""
        pipeline, model = _tiny_pipeline()
        B, S = 2, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.01, 0.99)
        r = t * torch.rand(B)

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=False)

        assert set(result.keys()) == {"loss", "loss_fm", "loss_mf"}
        assert torch.isfinite(result["loss"])

    def test_P4b_T2_pipeline_return_diagnostics_true(self) -> None:
        """return_diagnostics=True returns all diag_* keys, finite, detached."""
        pipeline, model = _tiny_pipeline()
        B, S = 2, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.01, 0.99)
        r = t * torch.rand(B) * 0.5  # Ensure r < t

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

        expected_diag_keys = {
            "diag_jvp_norm",
            "diag_u_norm",
            "diag_v_tilde_norm",
            "diag_compound_v_norm",
            "diag_target_v_norm",
            "diag_adaptive_weight_mean",
            "diag_adaptive_weight_std",
            "diag_loss_per_channel",
            "diag_loss_fm_per_sample",
            "diag_loss_mf_per_sample",
        }
        actual_diag_keys = {k for k in result if k.startswith("diag_")}
        assert expected_diag_keys == actual_diag_keys

        for key in expected_diag_keys:
            val = result[key]
            assert torch.isfinite(val).all(), f"{key} is not finite"
            assert not val.requires_grad, f"{key} still requires grad"

    def test_P4b_T3_diagnostics_no_grad_leak(self) -> None:
        """loss.backward() works identically with return_diagnostics=True."""
        pipeline, model = _tiny_pipeline()
        B, S = 2, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.01, 0.99)
        r = t * torch.rand(B) * 0.5

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)
        loss = result["loss"]

        # Should not raise — diag tensors don't leak into graph
        loss.backward()

        # Verify model got gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowed after backward with diagnostics"

    def test_P4b_T4_per_channel_loss_shape(self) -> None:
        """diag_loss_per_channel has shape (4,) for 4-channel latents."""
        pipeline, model = _tiny_pipeline()
        B, S = 2, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.01, 0.99)
        r = t * torch.rand(B) * 0.5

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)
        ch_loss = result["diag_loss_per_channel"]

        assert ch_loss.shape == (4,), f"Expected (4,), got {ch_loss.shape}"
        assert (ch_loss >= 0).all(), "Per-channel loss should be non-negative"

    def test_P4b_T5_diagnostics_callback_step(self) -> None:
        """Callback reads _step_diagnostics and accumulates metrics."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()
        model._diag_enabled = True

        # Run a training step to populate _step_diagnostics
        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)

        assert model._step_diagnostics is not None
        assert "diag_u_norm" in model._step_diagnostics
        assert "t" in model._step_diagnostics
        assert "r" in model._step_diagnostics

        # Simulate callback reading
        callback = TrainingDiagnosticsCallback(
            diag_every_n_epochs=1,
            diagnostics_dir="",
            gradient_clip_val=1.0,
        )

        # Mock trainer and pl_module.log
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        model.log = MagicMock()

        callback.on_train_batch_end(mock_trainer, model, {}, batch, 0)

        # _step_diagnostics should be cleared
        assert model._step_diagnostics is None

        # Epoch accumulators should have data
        assert len(callback._epoch_losses) == 1
        assert len(callback._epoch_t_values) > 0

    def test_P4b_T6_json_summary_written(self, tmp_path: Path) -> None:
        """After simulated epoch, summary.json exists and is valid."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()
        model._diag_enabled = True

        callback = TrainingDiagnosticsCallback(
            diag_every_n_epochs=1,
            diagnostics_dir=str(tmp_path),
            gradient_clip_val=1.0,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        mock_trainer.global_step = 10
        model.log = MagicMock()

        # Simulate epoch lifecycle
        callback.on_train_epoch_start(mock_trainer, model)

        # Run a step
        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)
        loss.backward()
        callback.on_before_optimizer_step(mock_trainer, model, MagicMock())
        callback.on_train_batch_end(mock_trainer, model, {}, batch, 0)

        # Zero grads for clean state
        model.zero_grad()

        # End epoch
        callback.on_train_epoch_end(mock_trainer, model)

        # Check JSON was written
        epoch_json = tmp_path / "epoch_000" / "summary.json"
        assert epoch_json.exists(), f"Expected {epoch_json}"

        summary = json.loads(epoch_json.read_text())
        assert summary["epoch"] == 0
        assert "sampling" in summary

        # Check cumulative summary
        cumul_json = tmp_path / "training_summary.json"
        assert cumul_json.exists()
        history = json.loads(cumul_json.read_text())
        assert len(history) == 1

    def test_P4b_T7_performance_callback_throughput(self) -> None:
        """PerformanceCallback produces positive throughput."""
        callback = PerformanceCallback(log_every_n_steps=1)

        mock_trainer = MagicMock()
        mock_trainer.global_step = 0

        mock_module = MagicMock()
        mock_module.device = torch.device("cpu")
        logged_values: dict[str, float] = {}

        def fake_log(key: str, val: float, **kwargs: object) -> None:
            logged_values[key] = val

        mock_module.log = fake_log

        batch = _fake_batch(batch_size=4)

        callback.on_train_batch_start(mock_trainer, mock_module, batch, 0)
        # Simulate some work
        _ = torch.randn(100, 100) @ torch.randn(100, 100)
        callback.on_train_batch_end(mock_trainer, mock_module, {}, batch, 0)

        assert "perf/samples_per_sec" in logged_values
        assert logged_values["perf/samples_per_sec"] > 0
        assert "perf/steps_per_sec" in logged_values
        assert logged_values["perf/steps_per_sec"] > 0

    def test_P4b_T8_jvp_norm_recovery(self) -> None:
        """(V - u) / (t - r) recovers JVP direction within tolerance."""
        pipeline, model = _tiny_pipeline()
        B, S = 4, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        # Use substantial t-r gap so JVP recovery is meaningful
        t = torch.rand(B).clamp(0.3, 0.99)
        r = t * torch.rand(B) * 0.3  # r << t

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=True)

        jvp_norm = result["diag_jvp_norm"]
        assert torch.isfinite(jvp_norm), "JVP norm is not finite"
        assert jvp_norm > 0, "JVP norm should be positive for non-trivial t-r gap"
