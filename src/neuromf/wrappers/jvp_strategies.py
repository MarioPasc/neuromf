"""JVP computation strategies for MeanFlow compound velocity.

Provides an abstraction layer over exact ``torch.func.jvp`` and
finite-difference approximation. This lets Phase 4+ swap between methods
based on available VRAM: ``ExactJVP`` on A100 40GB, ``FiniteDifferenceJVP``
on RTX 4060 8GB.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Type alias for the u_fn closure: (z_t, t, r) -> u
UFn = Callable[[Tensor, Tensor, Tensor], Tensor]

# Type alias for dual-output model: (z_t, t, r) -> (u, v)
DualFn = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


class JVPStrategy(Protocol):
    """Protocol for JVP computation strategies."""

    def compute(
        self,
        u_fn: UFn,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_tangent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute u and compound velocity V.

        Args:
            u_fn: Callable ``(z_t, t, r) -> u``.
            z_t: Noisy data ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            v_tangent: Tangent vector for z_t, same shape as z_t.

        Returns:
            Tuple ``(u, V)`` where ``V = u + (t-r) * sg[JVP]``.
        """
        ...


def _compound_velocity(u: Tensor, du_dt: Tensor, t: Tensor, r: Tensor, z_t: Tensor) -> Tensor:
    """Compute V = u + (t - r) * sg[du/dt] with proper broadcasting."""
    t_minus_r = (t - r).clamp(min=0.0)
    if z_t.ndim > 1:
        shape = (-1,) + (1,) * (z_t.ndim - 1)
        t_minus_r = t_minus_r.view(*shape)
    return u + t_minus_r * du_dt.detach()


class ExactJVP:
    """Exact JVP using ``torch.func.jvp``.

    Most accurate but requires a JVP-compatible model (no in-place ops)
    and enough VRAM for the forward-mode primal computation tape.
    """

    def compute(
        self,
        u_fn: UFn,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_tangent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute exact JVP via ``torch.func.jvp``."""
        dt = torch.ones_like(t)
        dr = torch.zeros_like(r)

        u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v_tangent, dt, dr))

        V = _compound_velocity(u, du_dt, t, r, z_t)
        return u, V

    def compute_dual(
        self,
        dual_fn: DualFn,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_tangent: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute exact JVP for dual-output model returning ``(u, v)``.

        The JVP is computed through a wrapper that extracts just ``u`` from
        the dual output. The ``v`` output is captured via ``has_aux``.

        Args:
            dual_fn: Callable ``(z_t, t, r) -> (u, v)``.
            z_t: Noisy data ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            v_tangent: Tangent vector for z_t.

        Returns:
            Tuple ``(u, V, v)`` where V is the compound velocity.
        """
        dt = torch.ones_like(t)
        dr = torch.zeros_like(r)

        def _u_with_v_aux(z: Tensor, t_: Tensor, r_: Tensor) -> tuple[Tensor, Tensor]:
            u, v = dual_fn(z, t_, r_)
            return u, v

        (u, v), du_dt = torch.func.jvp(
            _u_with_v_aux, (z_t, t, r), (v_tangent, dt, dr), has_aux=True
        )
        V = _compound_velocity(u, du_dt, t, r, z_t)
        return u, V, v


class FiniteDifferenceJVP:
    """Finite-difference JVP approximation.

    Always works regardless of model internals. Uses 2 sequential
    forward passes; peak VRAM is approximately that of 1 forward pass
    since intermediates are released between passes.

    ``JVP approx (u(z+h*v, r, t+h) - u(z, r, t)) / h``

    Args:
        h: Step size for finite differences.
    """

    def __init__(self, h: float = 1e-3) -> None:
        self.h = h

    def compute(
        self,
        u_fn: UFn,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_tangent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute finite-difference JVP approximation.

        The perturbed evaluation runs under ``torch.no_grad()`` since ``du_dt``
        is always detached in the compound velocity. This saves one full
        computation graph (~50% activation memory reduction for the JVP step).

        The subtraction is performed in fp32 to avoid catastrophic cancellation
        under bf16 mixed precision (bf16 has ~3 decimal digits of mantissa;
        with h=1e-3 the difference would lose all significant digits).
        """
        h = self.h

        u = u_fn(z_t, t, r)
        with torch.no_grad():
            u_perturbed = u_fn(
                z_t.detach() + h * v_tangent.detach(),
                t.detach() + h,
                r.detach(),
            )
        # fp32 subtraction to avoid bf16 catastrophic cancellation
        du_dt = (u_perturbed.float() - u.detach().float()) / h

        V = _compound_velocity(u, du_dt, t, r, z_t)
        return u, V

    def compute_dual(
        self,
        dual_fn: DualFn,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_tangent: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute finite-difference JVP for dual-output model.

        Args:
            dual_fn: Callable ``(z_t, t, r) -> (u, v)``.
            z_t: Noisy data ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            v_tangent: Tangent vector for z_t.

        Returns:
            Tuple ``(u, V, v)`` where V is the compound velocity.
        """
        h = self.h

        u, v = dual_fn(z_t, t, r)
        with torch.no_grad():
            u_perturbed, _ = dual_fn(
                z_t.detach() + h * v_tangent.detach(),
                t.detach() + h,
                r.detach(),
            )
        du_dt = (u_perturbed.float() - u.detach().float()) / h

        V = _compound_velocity(u, du_dt, t, r, z_t)
        return u, V, v
