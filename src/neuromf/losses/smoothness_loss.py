"""Velocity field smoothness regularizer via Hutchinson trace estimator.

Penalizes ||dv/dz_t||_F^2 / d to encourage smoother velocity fields,
improving JVP tangent quality and ODE trajectory regularity.

Two modes:
- **FD** (finite-difference): memory-efficient, recommended for training.
- **exact** (torch.func.jvp): more accurate but ~8-10 GB extra memory.

Reference: Finlay et al., "How to Train Your Neural ODE", ICML 2020.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def velocity_smoothness_loss(
    v_fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    z_t: Tensor,
    t: Tensor,
    v_original: Tensor,
    strategy: str = "fd",
    fd_step: float = 1e-3,
    n_probes: int = 1,
    probe_distribution: str = "gaussian",
) -> dict[str, Tensor]:
    """Estimate ||dv_theta/dz_t||_F^2 / d via Hutchinson trace estimator.

    Args:
        v_fn: Network forward ``(z_t, r, t) -> (u_or_x, v)``. Called with
            ``r=t`` for instantaneous velocity.
        z_t: Noisy interpolation ``(B, C, D, H, W)``, detached.
        t: Upper time ``(B,)``.
        v_original: Pre-computed v-head output ``(B, C, D, H, W)``, detached.
            Reused from the main forward pass to avoid extra computation.
        strategy: ``"fd"`` for finite-difference or ``"exact"`` for
            ``torch.func.jvp``.
        fd_step: Step size for finite-difference mode.
        n_probes: Number of Hutchinson probes per step.
        probe_distribution: ``"gaussian"`` or ``"rademacher"``.

    Returns:
        Dict with ``"loss"`` (scalar, differentiable) and
        ``"jac_frob_norm_est"`` (scalar, detached).
    """
    if strategy == "fd":
        return _fd_smoothness(
            v_fn, z_t, t, v_original, fd_step, n_probes, probe_distribution
        )
    elif strategy == "exact":
        return _exact_smoothness(v_fn, z_t, t, n_probes, probe_distribution)
    else:
        raise ValueError(f"Unknown smoothness strategy: {strategy!r}")


def _sample_probe(z_t: Tensor, distribution: str) -> Tensor:
    """Sample a Hutchinson probe vector.

    Args:
        z_t: Reference tensor for shape and device.
        distribution: ``"gaussian"`` or ``"rademacher"``.

    Returns:
        Probe vector with same shape as ``z_t``.
    """
    if distribution == "gaussian":
        return torch.randn_like(z_t)
    elif distribution == "rademacher":
        return torch.sign(torch.randn_like(z_t))
    else:
        raise ValueError(f"Unknown probe distribution: {distribution!r}")


def _fd_smoothness(
    v_fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    z_t: Tensor,
    t: Tensor,
    v_original: Tensor,
    fd_step: float,
    n_probes: int,
    probe_distribution: str,
) -> dict[str, Tensor]:
    """Finite-difference Hutchinson estimator."""
    d = z_t[0].numel()  # C*D*H*W
    total_sq = torch.zeros(z_t.shape[0], device=z_t.device)

    for _ in range(n_probes):
        xi = _sample_probe(z_t, probe_distribution)
        z_perturbed = z_t.detach() + fd_step * xi
        _, v_perturbed = v_fn(z_perturbed, t, t)
        # fp32 subtraction to avoid catastrophic cancellation in bf16
        jvp_est = (v_perturbed.float() - v_original.detach().float()) / fd_step
        # ||Jv * xi||^2 per sample, sum over (C,D,H,W)
        total_sq = total_sq + jvp_est.flatten(1).pow(2).sum(dim=1)

    # Mean over probes and normalize by dimension
    loss_per_sample = total_sq / (n_probes * d)
    loss = loss_per_sample.mean()

    return {
        "loss": loss,
        "jac_frob_norm_est": loss.detach(),
    }


def _exact_smoothness(
    v_fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    z_t: Tensor,
    t: Tensor,
    n_probes: int,
    probe_distribution: str,
) -> dict[str, Tensor]:
    """Exact JVP Hutchinson estimator via torch.func.jvp."""
    d = z_t[0].numel()
    total_sq = torch.zeros(z_t.shape[0], device=z_t.device)

    # Closure that returns only v (second output of v_fn)
    def v_only(z: Tensor) -> Tensor:
        _, v = v_fn(z, t, t)
        return v

    for _ in range(n_probes):
        xi = _sample_probe(z_t, probe_distribution)
        _, jvp_val = torch.func.jvp(v_only, (z_t,), (xi,))
        total_sq = total_sq + jvp_val.flatten(1).pow(2).sum(dim=1)

    loss_per_sample = total_sq / (n_probes * d)
    loss = loss_per_sample.mean()

    return {
        "loss": loss,
        "jac_frob_norm_est": loss.detach(),
    }
