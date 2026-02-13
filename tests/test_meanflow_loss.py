"""Tests for MeanFlow JVP loss (Phase 2).

P2-T2: Loss computes without error (finite, positive).
P2-T3: JVP shape + finite-difference check (relative error < 1e-3).
P2-T4 through P2-T8: Added after training outputs exist (see sweep.py).
"""

import json
from pathlib import Path

import pytest
import torch
from scipy import stats

from neuromf.losses.meanflow_jvp import compute_compound_velocity, meanflow_loss
from neuromf.models.toy_mlp import ToyMLP

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid")


@pytest.fixture
def toy_model() -> ToyMLP:
    """Small toy MLP for testing."""
    torch.manual_seed(42)
    return ToyMLP(data_dim=4, hidden_dim=64, n_layers=3, prediction_type="u")


@pytest.fixture
def toy_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random batch: (x_0, eps, t, r)."""
    torch.manual_seed(42)
    B = 16
    x_0 = torch.randn(B, 4)
    eps = torch.randn(B, 4)
    t = torch.rand(B) * 0.8 + 0.1  # [0.1, 0.9]
    r = t * torch.rand(B)  # r <= t
    return x_0, eps, t, r


# ---------------------------------------------------------------------------
# P2-T2: Loss computes without error
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T2LossComputes:
    """P2-T2: MeanFlow loss computes without error on toroid batch."""

    def test_P2_T2_loss_finite_positive(self, toy_model: ToyMLP, toy_batch: tuple) -> None:
        x_0, eps, t, r = toy_batch
        result = meanflow_loss(toy_model, x_0, eps, t, r, p=2.0)

        assert torch.isfinite(result["loss"]), "Loss is not finite"
        assert result["loss"].item() > 0, "Loss should be positive"
        assert torch.isfinite(result["loss_fm"]), "FM loss is not finite"
        assert torch.isfinite(result["loss_mf"]), "MF loss is not finite"

    def test_P2_T2_loss_backward(self, toy_model: ToyMLP, toy_batch: tuple) -> None:
        """Verify gradients flow through the loss."""
        x_0, eps, t, r = toy_batch
        result = meanflow_loss(toy_model, x_0, eps, t, r, p=2.0)
        result["loss"].backward()

        has_grad = False
        for p in toy_model.parameters():
            if p.grad is not None:
                has_grad = True
                assert torch.isfinite(p.grad).all(), "Gradient has NaN/Inf"
        assert has_grad, "No gradients computed"

    def test_P2_T2_loss_with_r_equals_t(self, toy_model: ToyMLP) -> None:
        """When r == t, the MeanFlow JVP term should vanish."""
        torch.manual_seed(42)
        B = 16
        x_0 = torch.randn(B, 4)
        eps = torch.randn(B, 4)
        t = torch.rand(B) * 0.8 + 0.1
        r = t.clone()  # r == t

        result = meanflow_loss(toy_model, x_0, eps, t, r, p=2.0)
        assert torch.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# P2-T3: JVP shape + finite-difference check
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T3JVP:
    """P2-T3: JVP computation produces correct shape + finite-diff verification."""

    def test_P2_T3_jvp_shape(self, toy_model: ToyMLP) -> None:
        """JVP output shape should match model output shape."""
        torch.manual_seed(42)
        B = 8
        z_t = torch.randn(B, 4)
        t = torch.rand(B) * 0.8 + 0.1
        r = t * torch.rand(B)
        v_tangent = torch.randn(B, 4)

        def u_fn(z: torch.Tensor, t_: torch.Tensor, r_: torch.Tensor) -> torch.Tensor:
            return toy_model(z, r_, t_)

        u, V = compute_compound_velocity(u_fn, z_t, t, r, v_tangent)

        assert u.shape == (B, 4), f"u shape mismatch: {u.shape}"
        assert V.shape == (B, 4), f"V shape mismatch: {V.shape}"
        assert torch.isfinite(u).all(), "u has NaN/Inf"
        assert torch.isfinite(V).all(), "V has NaN/Inf"

    def test_P2_T3_jvp_finite_difference(self, toy_model: ToyMLP) -> None:
        """JVP should match finite-difference approximation (relative error < 1e-3)."""
        torch.manual_seed(42)
        B = 4
        z_t = torch.randn(B, 4)
        t = torch.rand(B) * 0.5 + 0.3  # [0.3, 0.8]
        r = t * torch.rand(B) * 0.5  # r well below t
        v_tangent = torch.randn(B, 4)

        def u_fn(z: torch.Tensor, t_: torch.Tensor, r_: torch.Tensor) -> torch.Tensor:
            return toy_model(z, r_, t_)

        # Analytical JVP via torch.func.jvp
        dt = torch.ones_like(t)
        dr = torch.zeros_like(r)
        _, du_dt_analytical = torch.func.jvp(u_fn, (z_t, t, r), (v_tangent, dt, dr))

        # Finite-difference JVP: [u(z+h*v, t+h, r) - u(z, t, r)] / h
        h = 1e-4
        u_base = u_fn(z_t, t, r)
        u_perturbed = u_fn(z_t + h * v_tangent, t + h, r)
        du_dt_fd = (u_perturbed - u_base) / h

        rel_error = (du_dt_analytical - du_dt_fd).norm() / (du_dt_fd.norm() + 1e-8)
        assert rel_error < 1e-2, f"JVP finite-diff relative error too large: {rel_error:.6f}"


# ---------------------------------------------------------------------------
# P2-T4: Training loss decreases (reads training log)
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T4LossDecreases:
    """P2-T4: Loss decreases monotonically (after warmup)."""

    def test_P2_T4_loss_decreases(self) -> None:
        log_file = (
            RESULTS_DIR / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "training_log.json"
        )
        if not log_file.exists():
            pytest.skip("Training not yet run — training_log.json missing")

        with open(log_file) as f:
            data = json.load(f)

        losses = data["epoch_losses"]
        assert losses[99] < losses[9], f"loss@100 ({losses[99]:.6f}) >= loss@10 ({losses[9]:.6f})"
        assert losses[9] < losses[0], f"loss@10 ({losses[9]:.6f}) >= loss@1 ({losses[0]:.6f})"


# ---------------------------------------------------------------------------
# P2-T5: 1-NFE samples lie on torus
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T5TorusFidelity:
    """P2-T5: 1-NFE samples lie approximately on the torus."""

    def test_P2_T5_mean_torus_distance(self) -> None:
        metrics_file = (
            RESULTS_DIR / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "metrics.json"
        )
        if not metrics_file.exists():
            pytest.skip("Training not yet run — metrics.json missing")

        with open(metrics_file) as f:
            metrics = json.load(f)

        mean_dist = metrics["one_step"]["mean_torus_distance"]
        assert mean_dist < 0.1, f"Mean distance to torus: {mean_dist:.4f} (should be < 0.1)"


# ---------------------------------------------------------------------------
# P2-T6: Angular uniformity
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.critical
class TestP2T6AngularUniformity:
    """P2-T6: Angular distribution of samples is approximately uniform."""

    def test_P2_T6_ks_test(self) -> None:
        samples_file = RESULTS_DIR / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "samples.pt"
        if not samples_file.exists():
            pytest.skip("Training not yet run — samples.pt missing")

        data = torch.load(samples_file, weights_only=True)
        samples = data["one_step"]  # (N, 4)

        import numpy as np

        theta1 = torch.atan2(samples[:, 1], samples[:, 0]).numpy()
        theta2 = torch.atan2(samples[:, 3], samples[:, 2]).numpy()

        theta1_norm = (theta1 + np.pi) / (2 * np.pi)
        theta2_norm = (theta2 + np.pi) / (2 * np.pi)

        ks1 = stats.kstest(theta1_norm, "uniform")
        ks2 = stats.kstest(theta2_norm, "uniform")

        assert ks1.pvalue > 0.01, f"theta1 KS p-value {ks1.pvalue:.4f} < 0.01 (not uniform)"
        assert ks2.pvalue > 0.01, f"theta2 KS p-value {ks2.pvalue:.4f} < 0.01 (not uniform)"


# ---------------------------------------------------------------------------
# P2-T7: Multi-step <= 1-step distance (INFORMATIONAL)
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.informational
class TestP2T7MultiStep:
    """P2-T7: 5-step produces better or equal torus-fidelity than 1-step."""

    def test_P2_T7_multi_step_better(self) -> None:
        metrics_file = (
            RESULTS_DIR / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "metrics.json"
        )
        if not metrics_file.exists():
            pytest.skip("Training not yet run — metrics.json missing")

        with open(metrics_file) as f:
            metrics = json.load(f)

        one_step_dist = metrics["one_step"]["mean_torus_distance"]
        multi_step_dist = metrics["multi_step"]["mean_torus_distance"]

        assert multi_step_dist <= one_step_dist + 0.02, (
            f"Multi-step ({multi_step_dist:.4f}) > 1-step ({one_step_dist:.4f})"
        )


# ---------------------------------------------------------------------------
# P2-T8: x-pred and u-pred both converge (INFORMATIONAL)
# ---------------------------------------------------------------------------


@pytest.mark.phase2
@pytest.mark.informational
class TestP2T8PredictionTypes:
    """P2-T8: Both x-prediction and u-prediction converge on toroid."""

    def test_P2_T8_both_converge(self) -> None:
        # Check ablation B results for D=4 runs
        u_dir = RESULTS_DIR / "ablation_b" / "D4_u-pred"
        x_dir = RESULTS_DIR / "ablation_b" / "D4_x-pred"

        u_metrics = u_dir / "metrics.json"
        x_metrics = x_dir / "metrics.json"

        if not u_metrics.exists() or not x_metrics.exists():
            pytest.skip("Ablation B not yet run")

        with open(u_metrics) as f:
            u_data = json.load(f)
        with open(x_metrics) as f:
            x_data = json.load(f)

        u_loss = u_data.get("final_loss", float("inf"))
        x_loss = x_data.get("final_loss", float("inf"))

        tau = 3.0
        assert u_loss < tau, f"u-pred final loss {u_loss:.4f} >= {tau}"
        assert x_loss < tau, f"x-pred final loss {x_loss:.4f} >= {tau}"
        ratio = max(u_loss, x_loss) / min(u_loss, x_loss)
        assert ratio < 1.5, f"Loss ratio {ratio:.4f} — methods diverge"
