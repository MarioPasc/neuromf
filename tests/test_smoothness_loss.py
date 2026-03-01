"""Tests for velocity field smoothness regularizer (doc2).

9 tests covering FD and exact JVP modes, gradient flow, Hutchinson
unbiasedness, probe distributions, variance reduction, fp32 safety,
and zero-lambda baseline.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuromf.losses.smoothness_loss import velocity_smoothness_loss

# Spatial size for fast CPU tests
B, C, D, H, W = 2, 4, 6, 6, 6


class _MinimalDualHead(nn.Module):
    """Tiny dual-head model for smoothness tests.

    Uses small channels to keep tests fast on CPU.
    """

    def __init__(self, in_channels: int = C) -> None:
        super().__init__()
        self.u_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, in_channels, 3, padding=1),
        )
        self.v_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, in_channels, 3, padding=1),
        )

    def forward(
        self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.u_branch(z_t)
        v = self.v_branch(z_t)
        return u, v


class _LinearVModel(nn.Module):
    """Linear model v(z) = Az + b for Hutchinson unbiasedness test.

    The Frobenius norm of A is known analytically.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, d) * 0.1)
        self.b = nn.Parameter(torch.randn(d) * 0.01)

    def forward(
        self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten spatial dims, apply linear, reshape back
        shape = z_t.shape
        z_flat = z_t.reshape(shape[0], -1)  # (B, d)
        v_flat = z_flat @ self.A.T + self.b
        v = v_flat.reshape(shape)
        # Return dummy u and v
        u = torch.zeros_like(v)
        return u, v


@pytest.fixture
def dual_head_model() -> _MinimalDualHead:
    """Create a minimal dual-head model."""
    torch.manual_seed(42)
    return _MinimalDualHead()


@pytest.fixture
def sample_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample z_t and t tensors."""
    torch.manual_seed(42)
    z_t = torch.randn(B, C, D, H, W)
    t = torch.rand(B).clamp(0.05, 0.95)
    return z_t, t


# -----------------------------------------------------------------------
# SM-T1: FD mode returns correct keys with finite positive values
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T1SMFdReturnsCorrectKeys:
    def test_P6_T1_SM_fd_returns_correct_keys(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs
        with torch.no_grad():
            _, v_original = dual_head_model(z_t, t, t)

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=v_original.detach(),
            strategy="fd",
            fd_step=1e-3,
            n_probes=1,
            probe_distribution="gaussian",
        )

        assert "loss" in result
        assert "jac_frob_norm_est" in result
        assert torch.isfinite(result["loss"]), f"loss={result['loss']}"
        assert result["loss"].item() > 0, "Loss should be positive"
        assert torch.isfinite(result["jac_frob_norm_est"])
        assert result["jac_frob_norm_est"].item() > 0


# -----------------------------------------------------------------------
# SM-T2: FD mode — gradient flows to v-head params
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T2SMGradientFlows:
    def test_P6_T2_SM_gradient_flows_to_model(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs
        with torch.no_grad():
            _, v_original = dual_head_model(z_t, t, t)

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=v_original.detach(),
            strategy="fd",
        )

        result["loss"].backward()

        # Check that v_branch params have gradients
        v_params_with_grad = 0
        v_params_total = 0
        for p in dual_head_model.v_branch.parameters():
            v_params_total += 1
            if p.grad is not None and p.grad.norm() > 0:
                v_params_with_grad += 1

        assert v_params_total > 0
        frac = v_params_with_grad / v_params_total
        assert frac >= 0.9, f"Only {frac:.0%} of v_branch params have nonzero grad"


# -----------------------------------------------------------------------
# SM-T3: Hutchinson unbiased for known linear model
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T3SMHutchinsonUnbiasedLinear:
    def test_P6_T3_SM_hutchinson_unbiased_linear(self) -> None:
        torch.manual_seed(0)
        d = C * D * H * W  # = 4*6*6*6 = 864
        model = _LinearVModel(d)

        # Analytical Frobenius norm: ||A||_F^2 / d
        with torch.no_grad():
            A = model.A.data
            true_frob_sq_over_d = (A.norm() ** 2 / d).item()

        z_t = torch.randn(1, C, D, H, W)
        t = torch.tensor([0.5])

        with torch.no_grad():
            _, v_original = model(z_t, t, t)

        # Accumulate 500 estimates
        estimates = []
        for seed in range(500):
            torch.manual_seed(seed + 100)
            result = velocity_smoothness_loss(
                v_fn=model,
                z_t=z_t.detach(),
                t=t,
                v_original=v_original.detach(),
                strategy="fd",
                fd_step=1e-3,
                n_probes=1,
                probe_distribution="gaussian",
            )
            estimates.append(result["jac_frob_norm_est"].item())

        mean_est = sum(estimates) / len(estimates)
        rel_error = abs(mean_est - true_frob_sq_over_d) / (true_frob_sq_over_d + 1e-8)
        assert rel_error < 0.15, (
            f"Hutchinson estimate {mean_est:.4f} vs true {true_frob_sq_over_d:.4f}, "
            f"relative error {rel_error:.2%}"
        )


# -----------------------------------------------------------------------
# SM-T4: Exact JVP mode returns correct keys with finite values
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T4SMExactJvpReturnsCorrectKeys:
    def test_P6_T4_SM_exact_jvp_returns_correct_keys(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=torch.zeros_like(z_t),  # unused in exact mode
            strategy="exact",
            n_probes=1,
            probe_distribution="gaussian",
        )

        assert "loss" in result
        assert "jac_frob_norm_est" in result
        assert torch.isfinite(result["loss"]), f"loss={result['loss']}"
        assert result["loss"].item() > 0
        assert torch.isfinite(result["jac_frob_norm_est"])


# -----------------------------------------------------------------------
# SM-T5: Exact JVP mode — gradient flows to model
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T5SMExactJvpGradientFlows:
    def test_P6_T5_SM_exact_jvp_gradient_flows(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=torch.zeros_like(z_t),
            strategy="exact",
        )

        result["loss"].backward()

        v_params_with_grad = 0
        v_params_total = 0
        for p in dual_head_model.v_branch.parameters():
            v_params_total += 1
            if p.grad is not None and p.grad.norm() > 0:
                v_params_with_grad += 1

        assert v_params_total > 0
        frac = v_params_with_grad / v_params_total
        # Output bias doesn't appear in Jacobian dv/dz_t, so 75% (3/4) is expected
        assert frac >= 0.5, f"Only {frac:.0%} of v_branch params have nonzero grad"


# -----------------------------------------------------------------------
# SM-T6: Rademacher probe distribution works
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T6SMRademacherProbes:
    def test_P6_T6_SM_rademacher_probes(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs
        with torch.no_grad():
            _, v_original = dual_head_model(z_t, t, t)

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=v_original.detach(),
            strategy="fd",
            probe_distribution="rademacher",
        )

        assert torch.isfinite(result["loss"])
        assert result["loss"].item() > 0


# -----------------------------------------------------------------------
# SM-T7: More probes reduce variance
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T7SMNProbesVariance:
    def test_P6_T7_SM_n_probes_variance(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs
        with torch.no_grad():
            _, v_original = dual_head_model(z_t, t, t)

        n_trials = 30

        def run_trials(n_probes: int) -> list[float]:
            vals = []
            for seed in range(n_trials):
                torch.manual_seed(seed + 200)
                result = velocity_smoothness_loss(
                    v_fn=dual_head_model,
                    z_t=z_t.detach(),
                    t=t,
                    v_original=v_original.detach(),
                    strategy="fd",
                    n_probes=n_probes,
                )
                vals.append(result["loss"].item())
            return vals

        vals_1 = run_trials(1)
        vals_8 = run_trials(8)

        std_1 = torch.tensor(vals_1).std().item()
        std_8 = torch.tensor(vals_8).std().item()

        assert std_8 < std_1, (
            f"n_probes=8 std ({std_8:.4f}) should be < n_probes=1 std ({std_1:.4f})"
        )


# -----------------------------------------------------------------------
# SM-T8: FD fp32 subtraction avoids catastrophic cancellation with bf16
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T8SMFdFp32Subtraction:
    def test_P6_T8_SM_fd_fp32_subtraction(self) -> None:
        torch.manual_seed(42)

        # Create model and cast to bfloat16 to simulate mixed-precision
        model = _MinimalDualHead()
        if not torch.cuda.is_available():
            # bf16 on CPU requires PyTorch >= 2.0, which we have
            model = model.to(torch.bfloat16)
            z_t = torch.randn(B, C, D, H, W, dtype=torch.bfloat16)
            t = torch.rand(B).clamp(0.05, 0.95)
        else:
            model = model.cuda().to(torch.bfloat16)
            z_t = torch.randn(B, C, D, H, W, dtype=torch.bfloat16, device="cuda")
            t = torch.rand(B, device="cuda").clamp(0.05, 0.95)

        with torch.no_grad():
            _, v_original = model(z_t, t, t)

        result = velocity_smoothness_loss(
            v_fn=model,
            z_t=z_t.detach(),
            t=t,
            v_original=v_original.detach(),
            strategy="fd",
            fd_step=1e-3,
        )

        loss = result["loss"]
        assert torch.isfinite(loss), f"Non-finite loss with bf16 model: {loss}"
        assert loss.item() > 0, "Loss should be positive"


# -----------------------------------------------------------------------
# SM-T9: lambda=0 means smoothness doesn't alter loss
# -----------------------------------------------------------------------
@pytest.mark.phase6
class TestP6T9SMZeroLambdaBaseline:
    def test_P6_T9_SM_zero_lambda_baseline(
        self,
        dual_head_model: _MinimalDualHead,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z_t, t = sample_inputs
        with torch.no_grad():
            _, v_original = dual_head_model(z_t, t, t)

        result = velocity_smoothness_loss(
            v_fn=dual_head_model,
            z_t=z_t.detach(),
            t=t,
            v_original=v_original.detach(),
            strategy="fd",
        )

        # Simulate training_step: loss = base_loss + 0.0 * loss_smooth
        base_loss = torch.tensor(1.0, requires_grad=True)
        lambda_smooth = 0.0
        total_loss = base_loss + lambda_smooth * result["loss"]

        # With lambda=0, total_loss equals base_loss exactly
        assert total_loss.item() == pytest.approx(base_loss.item(), abs=1e-7)

        # Gradients should only flow to base_loss, not to model
        total_loss.backward()
        for p in dual_head_model.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() == 0.0, "Model grads should be zero with lambda=0"
