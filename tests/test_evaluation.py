"""Tests for the two-tier evaluation callback (SWD + 2.5D FID).

Test IDs follow the P4h convention from the phase split.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Tier 1: SWD tests
# ---------------------------------------------------------------------------


@pytest.mark.critical
def test_P4h_T1_swd_identical_distributions() -> None:
    """SWD of identical distributions should be approximately zero."""
    from neuromf.metrics.swd import compute_swd

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(200, 128, generator=gen)

    swd = compute_swd(x, x.clone(), n_projections=256, seed=0)
    assert swd < 0.01, f"SWD of identical distributions should be ~0, got {swd}"


@pytest.mark.critical
def test_P4h_T2_swd_shifted_distributions() -> None:
    """SWD of shifted distributions should be much larger than identical."""
    from neuromf.metrics.swd import compute_swd

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(200, 128, generator=gen)
    y = x + 5.0  # Large shift

    swd_identical = compute_swd(x, x.clone(), n_projections=256, seed=0)
    swd_shifted = compute_swd(x, y, n_projections=256, seed=0)

    assert swd_shifted > 10 * swd_identical, (
        f"Shifted SWD ({swd_shifted}) should be >> identical SWD ({swd_identical})"
    )
    assert swd_shifted > 1.0, f"Shifted SWD should be substantial, got {swd_shifted}"


# ---------------------------------------------------------------------------
# Tier 2: FID tests
# ---------------------------------------------------------------------------


class MockFeatureNet(nn.Module):
    """Mock feature extractor returning (B, 2048, 1, 1)."""

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        return torch.randn(B, 2048, 1, 1)


@pytest.mark.critical
def test_P4h_T3_extract_2d5_features_shapes() -> None:
    """extract_2d5_features returns (N_slices, 2048) per plane."""
    from neuromf.metrics.fid import extract_2d5_features

    volume = torch.randn(1, 1, 16, 16, 16)
    net = MockFeatureNet()

    xy, yz, zx = extract_2d5_features(volume, net, center_slices_ratio=0.6, batch_size=32)

    # With 16 voxels per axis and 0.6 ratio: int(0.2*16)=3 to int(0.8*16)=12
    # → 9 slices per plane (approximately)
    for name, feat in [("xy", xy), ("yz", yz), ("zx", zx)]:
        assert feat.ndim == 2, f"{name} should be 2D, got {feat.ndim}D"
        assert feat.shape[1] == 2048, f"{name} feat_dim should be 2048, got {feat.shape[1]}"
        assert feat.shape[0] > 0, f"{name} should have >0 slices"


@pytest.mark.critical
def test_P4h_T4_fid_identical_features() -> None:
    """FID of identical feature sets should be approximately zero."""
    from neuromf.metrics.fid import compute_fid_2d5

    gen = torch.Generator().manual_seed(42)
    feats = (
        torch.randn(100, 2048, generator=gen),
        torch.randn(100, 2048, generator=gen),
        torch.randn(100, 2048, generator=gen),
    )

    results = compute_fid_2d5(feats, feats)

    assert "fid_avg" in results
    assert "fid_xy" in results
    assert "fid_yz" in results
    assert "fid_zx" in results
    for key, val in results.items():
        assert val < 1.0, f"{key} should be ~0 for identical features, got {val}"


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


def _make_tiny_model() -> MagicMock:
    """Create a mock pl_module with net, ema, and required attributes."""
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")
    pl_module._latent_spatial = 8
    pl_module._in_channels = 4

    # Simple model that returns noise-shaped tensor
    class TinyNet(nn.Module):
        def forward(self, z_t: Tensor, r: Tensor, t: Tensor) -> Tensor:
            return torch.randn_like(z_t)

    net = TinyNet()
    pl_module.net = net

    # Mock EMA that does nothing
    ema = MagicMock()
    ema.apply_shadow = MagicMock()
    ema.restore = MagicMock()
    pl_module.ema = ema

    # Mock log
    pl_module.log = MagicMock()

    return pl_module


class _MockDataLoader:
    """Iterable that yields dict batches, mimicking a real DataLoader."""

    def __init__(self, data: Tensor, batch_size: int = 16) -> None:
        self._data = data
        self._bs = batch_size

    def __iter__(self):
        for i in range(0, self._data.shape[0], self._bs):
            yield {"z": self._data[i : i + self._bs]}


def _make_mock_trainer(val_data: Tensor) -> MagicMock:
    """Create a mock trainer with a validation dataloader."""
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.sanity_checking = False
    type(trainer).should_stop = PropertyMock(return_value=False)

    # Mock val dataloader as a list containing one DataLoader-like object
    trainer.val_dataloaders = [_MockDataLoader(val_data)]
    return trainer


@pytest.mark.critical
def test_P4h_T5_callback_logs_swd() -> None:
    """EvaluationCallback logs train/swd every training epoch."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    cb = EvaluationCallback(
        n_swd_samples=4,
        n_swd_projections=16,
        n_real_cache=8,
        n_fid_samples=4,
        fid_every_n_val_epochs=100,  # Disable FID for this test
        seed=42,
    )

    pl_module = _make_tiny_model()
    real_data = torch.randn(8, 4, 8, 8, 8)
    trainer = _make_mock_trainer(real_data)

    cb.on_fit_start(trainer, pl_module)
    # Cache real latents via val epoch first
    cb.on_validation_epoch_end(trainer, pl_module)
    # SWD fires on train epoch end
    cb.on_train_epoch_end(trainer, pl_module)

    logged_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "train/swd" in logged_keys, f"Expected train/swd in logged keys: {logged_keys}"


@pytest.mark.critical
def test_P4h_T6_callback_logs_fid_at_interval() -> None:
    """EvaluationCallback logs val/fid_avg at configured FID interval."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EvaluationCallback(
            n_swd_samples=4,
            n_swd_projections=16,
            n_real_cache=8,
            n_fid_samples=2,
            n_fid_real_samples=4,
            fid_every_n_val_epochs=1,  # FID every val epoch
            fid_weights_path="dummy",
            cache_dir=tmpdir,
            seed=42,
        )

        pl_module = _make_tiny_model()
        real_data = torch.randn(8, 4, 8, 8, 8)
        trainer = _make_mock_trainer(real_data)

        cb.on_fit_start(trainer, pl_module)

        # Mock the FID computation to avoid needing real VAE/weights
        mock_fid_results = {
            "fid_xy": 10.0,
            "fid_yz": 12.0,
            "fid_zx": 11.0,
            "fid_avg": 11.0,
        }
        # SWD on train epoch, FID on val epoch
        with patch.object(cb, "_compute_fid", return_value=mock_fid_results):
            cb.on_validation_epoch_end(trainer, pl_module)
        cb.on_train_epoch_end(trainer, pl_module)

        logged_keys = [call.args[0] for call in pl_module.log.call_args_list]
        assert "train/swd" in logged_keys
        assert "val/fid_avg" in logged_keys
        assert "val/fid_xy" in logged_keys


@pytest.mark.critical
def test_P4h_T7_early_stopping_triggers() -> None:
    """Early stopping triggers after patience exceeded with non-improving FID."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    cb = EvaluationCallback(
        n_swd_samples=4,
        n_swd_projections=16,
        n_real_cache=8,
        n_fid_samples=2,
        fid_every_n_val_epochs=1,
        early_stop_patience=3,
        fid_weights_path="dummy",
        seed=42,
    )

    pl_module = _make_tiny_model()
    real_data = torch.randn(8, 4, 8, 8, 8)
    trainer = _make_mock_trainer(real_data)

    # Use a simple namespace instead of MagicMock for should_stop tracking
    class _StopTracker:
        should_stop = False

    stop_tracker = _StopTracker()
    # Redirect trainer.should_stop to the tracker
    type(trainer).should_stop = property(
        lambda self: stop_tracker.should_stop,
        lambda self, val: setattr(stop_tracker, "should_stop", val),
    )

    cb.on_fit_start(trainer, pl_module)

    # Simulate FID evaluations: first good, then non-improving
    fid_values = [10.0, 12.0, 14.0, 16.0, 18.0]  # Only first is best

    for fid_val in fid_values:
        mock_fid = {
            "fid_xy": fid_val,
            "fid_yz": fid_val,
            "fid_zx": fid_val,
            "fid_avg": fid_val,
        }
        with patch.object(cb, "_compute_fid", return_value=mock_fid):
            cb.on_validation_epoch_end(trainer, pl_module)
        cb.on_train_epoch_end(trainer, pl_module)

    # After 1 good + 4 bad = patience 4 > 3, should have triggered stop
    assert stop_tracker.should_stop is True, "Early stopping should have triggered"


@pytest.mark.informational
def test_P4h_T8_callback_handles_vhead_model() -> None:
    """Callback handles models that return (u, v) tuples."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    cb = EvaluationCallback(
        n_swd_samples=4,
        n_swd_projections=16,
        n_real_cache=8,
        seed=42,
    )

    pl_module = _make_tiny_model()

    # Replace net with one that returns tuples
    class DualHeadNet(nn.Module):
        def forward(self, z_t: Tensor, r: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
            return torch.randn_like(z_t), torch.randn_like(z_t)

    pl_module.net = DualHeadNet()

    real_data = torch.randn(8, 4, 8, 8, 8)
    trainer = _make_mock_trainer(real_data)

    cb.on_fit_start(trainer, pl_module)
    # Cache real latents, then SWD — should not raise with dual-head model
    cb.on_validation_epoch_end(trainer, pl_module)
    cb.on_train_epoch_end(trainer, pl_module)

    logged_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "train/swd" in logged_keys


@pytest.mark.informational
def test_P4h_T9_fid_cache_reuse() -> None:
    """Real features are cached to disk and reused on second call."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EvaluationCallback(
            n_swd_samples=4,
            n_real_cache=8,
            n_fid_real_samples=4,
            cache_dir=tmpdir,
            seed=42,
        )

        # Simulate having real latents cached
        cb._real_latents = torch.randn(8, 4, 8, 8, 8)

        # Create fake features and save as cache
        fake_feats = (
            torch.randn(10, 2048),
            torch.randn(10, 2048),
            torch.randn(10, 2048),
        )
        cache_path = Path(tmpdir) / "real_features.pt"
        torch.save(
            {"xy": fake_feats[0], "yz": fake_feats[1], "zx": fake_feats[2]},
            str(cache_path),
        )

        # Load should find the cache
        device = torch.device("cpu")
        result = cb._load_or_compute_real_features(device)

        assert result[0].shape == fake_feats[0].shape
        assert torch.allclose(result[0], fake_feats[0])


@pytest.mark.critical
def test_P4h_T11_first_epoch_baseline_fid() -> None:
    """Both SWD and FID run on first val epoch regardless of fid_every_n setting."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    cb = EvaluationCallback(
        n_swd_samples=4,
        n_swd_projections=16,
        n_real_cache=8,
        n_fid_samples=2,
        fid_every_n_val_epochs=10,  # Would not trigger until val_epoch 10
        fid_weights_path="dummy",
        seed=42,
    )

    pl_module = _make_tiny_model()
    real_data = torch.randn(8, 4, 8, 8, 8)
    trainer = _make_mock_trainer(real_data)

    cb.on_fit_start(trainer, pl_module)

    mock_fid = {"fid_xy": 50.0, "fid_yz": 55.0, "fid_zx": 52.0, "fid_avg": 52.33}
    with patch.object(cb, "_compute_fid", return_value=mock_fid):
        cb.on_validation_epoch_end(trainer, pl_module)  # val_epoch 1 → baseline FID
    cb.on_train_epoch_end(trainer, pl_module)  # SWD

    logged_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "train/swd" in logged_keys, "SWD should be logged on first epoch"
    assert "val/fid_avg" in logged_keys, "FID should be logged on first epoch (baseline)"

    # Verify both SWD and FID in history
    assert len(cb._eval_history) == 2  # 1 from val (FID), 1 from train (SWD)
    fid_record = [r for r in cb._eval_history if "fid_avg" in r]
    assert len(fid_record) == 1
    assert fid_record[0]["fid_avg"] == 52.33


@pytest.mark.critical
def test_P4h_T12_on_fit_end_writes_summary() -> None:
    """on_fit_end writes eval_summary.json with aggregate metrics."""
    from neuromf.callbacks.evaluation import EvaluationCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EvaluationCallback(
            n_swd_samples=4,
            n_swd_projections=16,
            n_real_cache=8,
            n_fid_samples=2,
            fid_every_n_val_epochs=1,
            fid_weights_path="dummy",
            cache_dir=tmpdir,
            seed=42,
        )

        pl_module = _make_tiny_model()
        real_data = torch.randn(8, 4, 8, 8, 8)
        trainer = _make_mock_trainer(real_data)
        trainer.current_epoch = 10

        cb.on_fit_start(trainer, pl_module)

        # Simulate 3 val epochs with decreasing FID, each followed by SWD
        for fid_val in [100.0, 80.0, 60.0]:
            mock_fid = {
                "fid_xy": fid_val,
                "fid_yz": fid_val,
                "fid_zx": fid_val,
                "fid_avg": fid_val,
            }
            with patch.object(cb, "_compute_fid", return_value=mock_fid):
                cb.on_validation_epoch_end(trainer, pl_module)
            cb.on_train_epoch_end(trainer, pl_module)

        # Trigger on_fit_end
        cb.on_fit_end(trainer, pl_module)

        # Check JSON was written
        summary_path = Path(tmpdir) / "eval_summary.json"
        assert summary_path.exists(), "eval_summary.json should be written"

        import json

        summary = json.loads(summary_path.read_text())
        assert summary["n_val_epochs"] == 3
        assert summary["fid_first"] == 100.0
        assert summary["fid_best"] == 60.0
        assert summary["fid_last"] == 60.0
        assert "swd_first" in summary
        assert "swd_best" in summary
        # 3 val records (FID attached) + 3 train records (SWD) = 6
        swd_records = [r for r in summary["per_epoch"] if "swd" in r]
        fid_records = [r for r in summary["per_epoch"] if "fid_avg" in r]
        assert len(swd_records) == 3
        assert len(fid_records) == 3


@pytest.mark.informational
def test_P4h_T10_load_radimagenet_from_state_dict() -> None:
    """load_radimagenet_resnet50 produces (B, 2048, H', W') from local state dict."""
    from radimagenet_models.models.resnet import ResNet50

    from neuromf.metrics.fid import _spatial_average, load_radimagenet_resnet50

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        # Save a random Keras-style ResNet-50 state dict
        model = ResNet50()
        torch.save(model.state_dict(), f.name)

        # Load via our function
        feat_net = load_radimagenet_resnet50(f.name)

        # Forward pass with a dummy image
        dummy = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = feat_net(dummy)
            pooled = _spatial_average(out)

        assert out.shape[1] == 2048, f"Expected 2048 channels, got {out.shape[1]}"
        assert pooled.shape == (2, 2048), f"Expected (2, 2048), got {pooled.shape}"


@pytest.mark.informational
def test_P4h_T13_load_radimagenet_offline() -> None:
    """load_radimagenet_resnet50 works without internet access (Picasso offline)."""
    import socket

    from radimagenet_models.models.resnet import ResNet50

    from neuromf.metrics.fid import _spatial_average, load_radimagenet_resnet50

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        model = ResNet50()
        torch.save(model.state_dict(), f.name)

        # Block all network access
        orig_socket = socket.socket

        class _BlockedSocket(socket.socket):
            def connect(self, *args: object, **kwargs: object) -> None:
                raise ConnectionRefusedError("Simulated offline")

        socket.socket = _BlockedSocket  # type: ignore[misc]
        try:
            feat_net = load_radimagenet_resnet50(f.name)
            dummy = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                pooled = _spatial_average(feat_net(dummy))
            assert pooled.shape == (1, 2048)
        finally:
            socket.socket = orig_socket  # type: ignore[misc]
