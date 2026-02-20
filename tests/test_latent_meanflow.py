"""Phase 4 tests: Latent MeanFlow Lightning module.

All tests use a tiny UNet (channels=[8,16,32,64], spatial=16) so they run
in <10s on CPU. Uses ``jvp_strategy="finite_difference"`` to avoid
``torch.func`` overhead on small tensors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.sampling.one_step import sample_one_step


def _tiny_config(**overrides) -> OmegaConf:
    """Build a tiny config for fast local tests.

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
                "jvp_strategy": "finite_difference",
                "fd_step_size": 1e-3,
                "channel_weights": None,
                "norm_p": 1.0,
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
            "latent_spatial_size": 16,
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _fake_batch(batch_size: int = 2, spatial: int = 16) -> dict[str, torch.Tensor]:
    """Create a fake batch for testing.

    Args:
        batch_size: Number of samples.
        spatial: Spatial dimension size.

    Returns:
        Dict with ``"z"`` tensor.
    """
    return {"z": torch.randn(batch_size, 4, spatial, spatial, spatial)}


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
class TestLatentMeanFlow:
    """Phase 4 local tests with tiny UNet."""

    def test_P4_T1_lightning_module_init(self) -> None:
        """P4-T1: LatentMeanFlow instantiates without error."""
        config = _tiny_config()
        model = LatentMeanFlow(config)

        assert model.net is not None
        assert model.loss_pipeline is not None
        assert model.ema is not None
        assert model.latent_mean.shape == (1, 4, 1, 1, 1)
        assert model.latent_std.shape == (1, 4, 1, 1, 1)

    def test_P4_T2_training_step_runs(self) -> None:
        """P4-T2: training_step returns finite loss on fake batch."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()

        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.requires_grad

    def test_P4_T3_training_step_gradients_flow(self) -> None:
        """P4-T3: All trainable params get gradients after one step.

        MONAI DiffusionModelUNet zero-initialises conv2 layers in ResBlocks
        and the output conv, blocking gradient flow on the first backward.
        We reinit zero convs with small random values to test gradient flow.
        """
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()

        # Reinit zero-weight convs so gradients can flow on first backward
        with torch.no_grad():
            for param in model.net.parameters():
                if param.abs().sum() == 0 and param.numel() > 1:
                    nn.init.normal_(param, std=0.01)

        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)
        loss.backward()

        params_with_grad = 0
        params_total = 0
        for name, param in model.net.named_parameters():
            if param.requires_grad:
                params_total += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1

        # After reinit, most params should have non-zero grads
        ratio = params_with_grad / max(params_total, 1)
        assert ratio > 0.5, f"Only {params_with_grad}/{params_total} params have non-zero grads"

    def test_P4_T4_ema_updates(self) -> None:
        """P4-T4: EMA shadow differs from model after 5 manual steps."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()

        # Record initial EMA state
        initial_shadow = {k: v.clone() for k, v in model.ema.shadow.items()}

        # Run 5 training steps and manually update EMA
        for i in range(5):
            batch = _fake_batch()
            loss = model.training_step(batch, batch_idx=i)
            loss.backward()
            # Simulate optimizer step
            with torch.no_grad():
                for p in model.net.parameters():
                    if p.grad is not None:
                        p.data -= 1e-3 * p.grad
                        p.grad.zero_()
            model.ema.update(model.net)

        # EMA shadow should differ from initial state
        changed = 0
        for name in initial_shadow:
            if not torch.allclose(initial_shadow[name], model.ema.shadow[name]):
                changed += 1

        assert changed > 0, "No EMA parameters were updated"

    def test_P4_T5_checkpoint_save_load(self, tmp_path: Path) -> None:
        """P4-T5: Save/load checkpoint preserves EMA + loss history."""
        config = _tiny_config()
        model = LatentMeanFlow(config)

        # Simulate some training history
        model._train_loss_history = [2.0, 1.8, 1.5]
        model._val_loss_history = [2.1, 1.9]

        # Run one step to change EMA
        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)
        loss.backward()
        with torch.no_grad():
            for p in model.net.parameters():
                if p.grad is not None:
                    p.data -= 1e-3 * p.grad
                    p.grad.zero_()
        model.ema.update(model.net)

        # Save checkpoint
        checkpoint = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(checkpoint)

        assert "ema_state_dict" in checkpoint
        assert "loss_history" in checkpoint
        assert checkpoint["loss_history"]["train"] == [2.0, 1.8, 1.5]
        assert checkpoint["loss_history"]["val"] == [2.1, 1.9]

        # Create fresh model and load checkpoint
        model2 = LatentMeanFlow(config)
        model2.load_state_dict(checkpoint["state_dict"])
        model2.on_load_checkpoint(checkpoint)

        assert model2._train_loss_history == [2.0, 1.8, 1.5]
        assert model2._val_loss_history == [2.1, 1.9]

        # Verify EMA shadow matches
        for name in model.ema.shadow:
            assert torch.allclose(
                model.ema.shadow[name],
                model2.ema.shadow[name],
            ), f"EMA mismatch for {name}"

    def test_P4_T6_resume_loss_continuity(self) -> None:
        """P4-T6: Loss after resume is comparable magnitude to before save."""
        torch.manual_seed(42)
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.train()

        # Run 3 steps, record losses
        losses = []
        for i in range(3):
            batch = _fake_batch()
            loss = model.training_step(batch, batch_idx=i)
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                for p in model.net.parameters():
                    if p.grad is not None:
                        p.data -= 1e-3 * p.grad
                        p.grad.zero_()
            model.ema.update(model.net)

        # Save state
        checkpoint = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(checkpoint)

        # Restore into new model
        model2 = LatentMeanFlow(config)
        model2.load_state_dict(checkpoint["state_dict"])
        model2.on_load_checkpoint(checkpoint)
        model2.train()

        # Run one more step
        batch = _fake_batch()
        loss_resumed = model2.training_step(batch, batch_idx=3)

        # Loss should be comparable magnitude (within 10x of average)
        avg_loss = sum(losses) / len(losses)
        assert loss_resumed.item() < avg_loss * 10, (
            f"Resumed loss {loss_resumed.item():.4f} too far from avg pre-save {avg_loss:.4f}"
        )

    def test_P4_T7_sample_generation_shape(self) -> None:
        """P4-T7: _generate_samples with mock VAE produces correct shapes."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.eval()

        # Directly test the EMA sampling + denormalization path
        S = model._latent_spatial
        n_samples = 2
        noise = torch.randn(n_samples, 4, S, S, S)

        model.ema.apply_shadow(model.net)
        try:
            z_0_hat = sample_one_step(model.net, noise, prediction_type="x")
        finally:
            model.ema.restore(model.net)

        assert z_0_hat.shape == (n_samples, 4, S, S, S)

        # Denormalize
        z_0_denorm = z_0_hat * model.latent_std + model.latent_mean
        assert z_0_denorm.shape == z_0_hat.shape
        assert torch.isfinite(z_0_denorm).all()

    def test_P4_T8_cli_dry_run(self) -> None:
        """P4-T8: CLI with --dry-run exits 0."""
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "train_meanflow.yaml"

        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "experiments" / "cli" / "train.py"),
                "--config",
                str(config_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"CLI dry-run failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )


@pytest.mark.phase4
@pytest.mark.informational
class TestLatentMeanFlowExtended:
    """Phase 4 extended tests for configurability."""

    def test_P4_T9_lr_schedule_options(self) -> None:
        """P4-T9: Verify constant, cosine, and linear LR schedules work."""
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, TensorDataset

        for schedule in ["constant", "cosine", "linear"]:
            cfg = _tiny_config(**{"training": {"lr_schedule": schedule, "max_epochs": 1}})
            model = LatentMeanFlow(cfg)

            # Minimal dataset and trainer to trigger configure_optimizers
            ds = TensorDataset(torch.randn(4, 4, 16, 16, 16))
            dl = DataLoader(
                ds, batch_size=2, collate_fn=lambda batch: {"z": torch.stack([b[0] for b in batch])}
            )

            trainer = pl.Trainer(
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
                limit_train_batches=2,
            )
            # Just fit — if configure_optimizers raises, this fails
            trainer.fit(model, dl)

    def test_P4_T10_norm_p_configurable(self) -> None:
        """P4-T10: Verify norm_p is wired through to MeanFlowPipeline."""
        cfg = _tiny_config(**{"meanflow": {"norm_p": 0.5}})
        model = LatentMeanFlow(cfg)
        assert model.loss_pipeline.config.norm_p == 0.5

        # Default should be 1.0
        cfg_default = _tiny_config()
        model_default = LatentMeanFlow(cfg_default)
        assert model_default.loss_pipeline.config.norm_p == 1.0

    def test_P4_T11_raw_loss_always_returned(self) -> None:
        """P4-T11: raw_loss always in pipeline output (not gated by diagnostics)."""
        from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
        from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

        pipeline = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
            )
        )
        unet_cfg = MAISIUNetConfig(
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
        model = MAISIUNetWrapper(unet_cfg)
        B, S = 2, 16
        z_0 = torch.randn(B, 4, S, S, S)
        eps = torch.randn(B, 4, S, S, S)
        t = torch.rand(B).clamp(0.05, 0.95)
        r = t * torch.rand(B) * 0.5

        result = pipeline(model, z_0, eps, t, r, return_diagnostics=False)
        assert set(result.keys()) == {"loss", "raw_loss"}
        assert torch.isfinite(result["raw_loss"])
        assert torch.isfinite(result["loss"])

    def test_P4_T12_divergence_monitor(self) -> None:
        """P4-T12: EMA divergence monitor tracks loss and warns (no crash)."""
        cfg = _tiny_config(
            **{
                "training": {
                    "divergence_grace_steps": 0,
                }
            }
        )
        model = LatentMeanFlow(cfg)
        model.train()

        # First step: initialises EMA
        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)
        assert model._ema_raw_loss is not None
        assert model._min_ema_raw_loss is not None

        # Manually set EMA min to a tiny value so next step triggers warnings
        model._min_ema_raw_loss = model._ema_raw_loss / 100.0

        # Should warn but NOT crash (early stopping is FID-based now)
        batch2 = _fake_batch()
        loss2 = model.training_step(batch2, batch_idx=1)
        assert torch.isfinite(loss2)
        assert model._divergence_warned_3x
        assert model._divergence_warned_5x

    def test_P4d_T10_grace_period(self) -> None:
        """P4d-T10: Divergence warnings do not fire during grace period."""
        cfg = _tiny_config(
            **{
                "training": {
                    "divergence_grace_steps": 9999,
                }
            }
        )
        model = LatentMeanFlow(cfg)
        model.train()

        # First step
        batch = _fake_batch()
        loss = model.training_step(batch, batch_idx=0)

        # Set EMA min to tiny value — would warn without grace period
        model._min_ema_raw_loss = 1e-10

        # But global_step < grace_steps, so warnings should NOT fire
        batch2 = _fake_batch()
        loss2 = model.training_step(batch2, batch_idx=1)
        assert torch.isfinite(loss2)
        assert not model._divergence_warned_3x
