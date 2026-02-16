"""Phase 4 tests: SampleCollectorCallback + decode_samples CLI.

All tests use a tiny UNet (channels=[8,16,32,64], spatial=16) so they run
in <10s on CPU. Uses ``jvp_strategy="finite_difference"`` to avoid
``torch.func`` overhead on small tensors.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from neuromf.callbacks.sample_collector import SampleCollectorCallback
from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.utils.latent_diagnostics import (
    compute_inter_epoch_delta,
    compute_latent_stats,
    compute_nfe_consistency,
)


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


def _make_mock_trainer(epoch: int = 0, global_step: int = 100) -> MagicMock:
    """Create a mock trainer with is_global_zero=True.

    Args:
        epoch: Current epoch to report.
        global_step: Current global step.

    Returns:
        MagicMock trainer.
    """
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.global_step = global_step
    trainer.is_global_zero = True
    return trainer


# ======================================================================
# Tests
# ======================================================================


@pytest.mark.phase4
@pytest.mark.critical
class TestSampleCollector:
    """Phase 4 sample collector tests."""

    def test_P4_T13_fixed_noise_deterministic(self) -> None:
        """P4-T13: Same seed produces identical noise across callback instances."""
        cb1 = SampleCollectorCallback(samples_dir="/tmp/test_sc", seed=42, n_samples=4)
        cb2 = SampleCollectorCallback(samples_dir="/tmp/test_sc", seed=42, n_samples=4)

        model = LatentMeanFlow(_tiny_config())
        trainer = _make_mock_trainer()

        cb1.on_fit_start(trainer, model)
        cb2.on_fit_start(trainer, model)

        assert cb1._fixed_noise is not None
        assert cb2._fixed_noise is not None
        assert torch.allclose(cb1._fixed_noise, cb2._fixed_noise), (
            "Same seed should produce identical noise"
        )

        # Different seed should produce different noise
        cb3 = SampleCollectorCallback(samples_dir="/tmp/test_sc", seed=99, n_samples=4)
        cb3.on_fit_start(trainer, model)
        assert not torch.allclose(cb1._fixed_noise, cb3._fixed_noise), (
            "Different seed should produce different noise"
        )

    def test_P4_T14_archive_structure(self, tmp_path: Path) -> None:
        """P4-T14: Saved .pt has all expected keys and shapes."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.eval()

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=2,
            nfe_steps=[1, 5],
            seed=42,
            prediction_type="x",
        )

        trainer = _make_mock_trainer(epoch=0, global_step=10)
        cb.on_fit_start(trainer, model)

        # Trigger collection
        cb.on_train_epoch_end(trainer, model)

        # Check archive file exists
        archive_path = tmp_path / "generated_samples" / "epoch_0000.pt"
        assert archive_path.exists(), f"Archive not saved at {archive_path}"

        archive = torch.load(archive_path, map_location="cpu", weights_only=False)

        # Check all expected keys
        expected_keys = {
            "epoch",
            "global_step",
            "noise_seed",
            "noise",
            "latent_mean",
            "latent_std",
            "stats",
            "nfe_consistency",
            "nfe_1",
            "nfe_5",
        }
        assert expected_keys.issubset(set(archive.keys())), (
            f"Missing keys: {expected_keys - set(archive.keys())}"
        )

        # Check shapes
        S = 16  # tiny model spatial size
        assert archive["noise"].shape == (2, 4, S, S, S)
        assert archive["nfe_1"].shape == (2, 4, S, S, S)
        assert archive["nfe_5"].shape == (2, 4, S, S, S)
        assert archive["epoch"] == 0
        assert archive["global_step"] == 10
        assert archive["noise_seed"] == 42

    def test_P4_T15_multi_nfe_different(self, tmp_path: Path) -> None:
        """P4-T15: NFE-1 and NFE-5 produce different results.

        Requires reinit of zero-init convs (MONAI zero-initialises output conv
        and conv2 layers in ResBlocks), otherwise model outputs all zeros.
        """
        import torch.nn as nn

        config = _tiny_config()
        model = LatentMeanFlow(config)

        # Reinit zero-weight convs so model produces non-trivial output
        with torch.no_grad():
            for param in model.net.parameters():
                if param.abs().sum() == 0 and param.numel() > 1:
                    nn.init.normal_(param, std=0.01)
            # Also update EMA shadow to match
            for name, param in model.net.named_parameters():
                if name in model.ema.shadow:
                    model.ema.shadow[name] = param.data.clone()

        model.eval()

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=2,
            nfe_steps=[1, 5],
            seed=42,
            prediction_type="x",
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        archive_path = tmp_path / "generated_samples" / "epoch_0000.pt"
        archive = torch.load(archive_path, map_location="cpu", weights_only=False)

        nfe_1 = archive["nfe_1"]
        nfe_5 = archive["nfe_5"]

        # Multi-step should give different results from 1-step
        assert not torch.allclose(nfe_1, nfe_5, atol=1e-3), (
            "NFE-1 and NFE-5 should produce different results"
        )

    def test_P4_T16_ema_applied(self, tmp_path: Path) -> None:
        """P4-T16: Callback applies EMA shadow, restores after."""
        config = _tiny_config()
        model = LatentMeanFlow(config)

        # Modify model weights so EMA and online differ
        with torch.no_grad():
            for p in model.net.parameters():
                p.data += 1.0
            model.ema.update(model.net)
            for p in model.net.parameters():
                p.data += 1.0
            # Now online params differ from EMA shadow

        # Record online weights before collection
        online_before = {n: p.data.clone() for n, p in model.net.named_parameters()}

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=1,
            nfe_steps=[1],
            seed=42,
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        # Online weights should be restored to exactly what they were before
        for name, param in model.net.named_parameters():
            if name in online_before:
                assert torch.allclose(param.data, online_before[name]), (
                    f"Online weight {name} was not restored after EMA sampling"
                )


@pytest.mark.phase4
@pytest.mark.informational
class TestSampleCollectorExtended:
    """Phase 4 sample collector extended tests."""

    def test_P4_T17_epoch_skipping(self, tmp_path: Path) -> None:
        """P4-T17: Callback skips non-collection epochs."""
        config = _tiny_config()
        model = LatentMeanFlow(config)

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=5,
            n_samples=1,
            nfe_steps=[1],
            seed=42,
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)

        # Epoch 0: (0+1) % 5 != 0 -> skip
        trainer.current_epoch = 0
        cb.on_train_epoch_end(trainer, model)
        assert not (tmp_path / "generated_samples" / "epoch_0000.pt").exists()

        # Epoch 4: (4+1) % 5 == 0 -> collect
        trainer.current_epoch = 4
        cb.on_train_epoch_end(trainer, model)
        assert (tmp_path / "generated_samples" / "epoch_0004.pt").exists()

    def test_P4_T18_stats_finite(self, tmp_path: Path) -> None:
        """P4-T18: All per-NFE stats are finite."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.eval()

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=2,
            nfe_steps=[1, 2],
            seed=42,
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        archive = torch.load(
            tmp_path / "generated_samples" / "epoch_0000.pt",
            map_location="cpu",
            weights_only=False,
        )

        for nfe_key in ["nfe_1", "nfe_2"]:
            s = archive["stats"][nfe_key]
            for stat_name in ["mean", "std"]:
                vals = s[stat_name]
                assert all(abs(v) < 1e6 for v in vals), (
                    f"{nfe_key}/{stat_name} contains extreme values: {vals}"
                )

    def test_P4_T19_nfe_consistency_computed(self, tmp_path: Path) -> None:
        """P4-T19: MSE/cosine between NFE levels is finite."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.eval()

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=2,
            nfe_steps=[1, 5, 10],
            seed=42,
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        archive = torch.load(
            tmp_path / "generated_samples" / "epoch_0000.pt",
            map_location="cpu",
            weights_only=False,
        )

        nfe_con = archive["nfe_consistency"]
        assert "mse_1vs5" in nfe_con
        assert "mse_1vs10" in nfe_con
        assert "cosine_1vs5" in nfe_con
        assert "cosine_1vs10" in nfe_con

        for key, val in nfe_con.items():
            assert abs(val) < 1e6, f"{key} is extreme: {val}"

    def test_P4_T20_latent_channel_png(self, tmp_path: Path) -> None:
        """P4-T20: Latent visualization PNG is saved."""
        config = _tiny_config()
        model = LatentMeanFlow(config)
        model.eval()

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=2,
            nfe_steps=[1],
            seed=42,
        )

        trainer = _make_mock_trainer(epoch=0)
        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        png_path = tmp_path / "epoch_0000" / "latent_channels.png"
        assert png_path.exists(), f"Latent channels PNG not saved at {png_path}"

    def test_P4_T21_rank_guard(self, tmp_path: Path) -> None:
        """P4-T21: Non-rank-0 processes skip collection."""
        config = _tiny_config()
        model = LatentMeanFlow(config)

        cb = SampleCollectorCallback(
            samples_dir=str(tmp_path),
            collect_every_n_epochs=1,
            n_samples=1,
            nfe_steps=[1],
            seed=42,
        )

        # Non-rank-0 trainer
        trainer = _make_mock_trainer(epoch=0)
        trainer.is_global_zero = False

        cb.on_fit_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)

        # No archive should be saved
        assert not (tmp_path / "generated_samples").exists()


@pytest.mark.phase4
@pytest.mark.informational
class TestLatentDiagnostics:
    """Tests for latent_diagnostics utility functions."""

    def test_P4_T22_compute_latent_stats(self) -> None:
        """compute_latent_stats returns correct shapes and finite values."""
        z = torch.randn(4, 4, 8, 8, 8)
        stats = compute_latent_stats(z)

        assert stats["mean"].shape == (4,)
        assert stats["std"].shape == (4,)
        assert stats["min"].shape == (4,)
        assert stats["max"].shape == (4,)
        assert stats["skewness"].shape == (4,)
        assert stats["kurtosis"].shape == (4,)
        assert torch.isfinite(stats["mean"]).all()
        assert torch.isfinite(stats["std"]).all()
        assert (stats["std"] > 0).all()

    def test_P4_T23_nfe_consistency_basic(self) -> None:
        """compute_nfe_consistency returns MSE and cosine for each pair."""
        samples = {
            "nfe_1": torch.randn(2, 4, 8, 8, 8),
            "nfe_5": torch.randn(2, 4, 8, 8, 8),
            "nfe_10": torch.randn(2, 4, 8, 8, 8),
        }
        result = compute_nfe_consistency(samples)

        assert "mse_1vs5" in result
        assert "mse_1vs10" in result
        assert "cosine_1vs5" in result
        assert "cosine_1vs10" in result

        # MSE should be positive
        assert result["mse_1vs5"] > 0
        assert result["mse_1vs10"] > 0

    def test_P4_T24_inter_epoch_delta(self) -> None:
        """compute_inter_epoch_delta returns finite L2 and cosine."""
        current = torch.randn(2, 4, 8, 8, 8)
        previous = current + 0.1 * torch.randn_like(current)

        result = compute_inter_epoch_delta(current, previous)

        assert "l2_distance" in result
        assert "cosine_similarity" in result
        assert result["l2_distance"] > 0
        assert -1.0 <= result["cosine_similarity"] <= 1.0
        # Small perturbation should have high cosine similarity
        assert result["cosine_similarity"] > 0.9
