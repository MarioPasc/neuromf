"""MeanFlow JVP loss computation (iMF combined objective, Eq. 13).

Core training loss for MeanFlow: uses torch.func.jvp to compute the compound
velocity V = u + (t-r) * sg[du_dt], then enforces self-consistency against the
conditional velocity v_c = eps - x.

Reference: MeanFlow-PyTorch/meanflow.py lines 125-188.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


def compute_compound_velocity(
    u_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    z_t: Tensor,
    t: Tensor,
    r: Tensor,
    v_tangent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute u and compound velocity V = u + (t-r) * sg[JVP].

    The JVP is computed with tangent vectors (v_tangent, dt=1, dr=0),
    differentiating u_fn w.r.t. z_t and t but not r.

    Args:
        u_fn: Callable(z_t, t, r) -> u. Must be a pure function suitable
            for torch.func.jvp.
        z_t: Noisy data of shape (B, D) or (B, C, ...).
        t: Upper time of shape matching z_t batch dim.
        r: Lower time of shape matching z_t batch dim.
        v_tangent: Tangent vector for z_t, same shape as z_t.

    Returns:
        Tuple of (u, V) where u is the model output and V is the
        compound velocity with stop-gradient on the JVP term.
    """
    dt = torch.ones_like(t)
    dr = torch.zeros_like(r)

    u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v_tangent, dt, dr))

    # Compound velocity: V = u + (t - r) * sg[du/dt]
    # Reshape (t - r) for broadcasting with arbitrary data shapes
    t_minus_r = (t - r).clamp(min=0.0)
    if z_t.ndim > 1:
        # Reshape to (B, 1, ...) for broadcasting
        shape = (-1,) + (1,) * (z_t.ndim - 1)
        t_minus_r = t_minus_r.view(*shape)

    V = u + t_minus_r * du_dt.detach()
    return u, V


def meanflow_loss(
    model: nn.Module,
    x_0: Tensor,
    eps: Tensor,
    t: Tensor,
    r: Tensor,
    p: float = 2.0,
    adaptive: bool = True,
    norm_eps: float = 0.01,
    lambda_mf: float = 1.0,
    prediction_type: str = "u",
) -> dict[str, Tensor]:
    """Full iMF combined loss (Eq. 13).

    Computes both the flow-matching loss (v_tilde vs v_c) and the MeanFlow
    loss (V vs v_c) with optional adaptive weighting.

    The model convention is ``model(z_t, r, t) -> output`` where:
    - For u-prediction: output is the average velocity u directly.
    - For x-prediction: output is x_hat, converted via u = (z_t - x_hat) / max(t, eps).

    Args:
        model: Neural network with forward(z_t, r, t) -> output.
        x_0: Clean data of shape (B, D) or (B, C, ...).
        eps: Noise of shape matching x_0.
        t: Upper time of shape (B,).
        r: Lower time of shape (B,).
        p: Lp norm exponent.
        adaptive: Whether to use adaptive per-sample weighting (Eq. 14).
        norm_eps: Small constant c in adaptive weight denominator.
        lambda_mf: Weight for MeanFlow loss term.
        prediction_type: "u" or "x" — how to interpret model output.

    Returns:
        Dict with keys: "loss" (total), "loss_fm" (flow matching term),
        "loss_mf" (MeanFlow term).
    """
    # Construct z_t via linear interpolation: z_t = (1 - t) * x_0 + t * eps
    if x_0.ndim > 1:
        shape = (-1,) + (1,) * (x_0.ndim - 1)
        t_broad = t.view(*shape)
        r_broad = r.view(*shape)
    else:
        t_broad = t
        r_broad = r

    z_t = (1 - t_broad) * x_0 + t_broad * eps

    # Conditional velocity
    v_c = eps - x_0

    # Define u_fn closure for JVP: wraps model to handle prediction_type
    if prediction_type == "u":

        def u_fn(z: Tensor, t_: Tensor, r_: Tensor) -> Tensor:
            return model(z, r_, t_)
    elif prediction_type == "x":

        def u_fn(z: Tensor, t_: Tensor, r_: Tensor) -> Tensor:
            x_hat = model(z, r_, t_)
            # u = (z_t - x_hat) / max(t, 0.05) to avoid division by zero
            if z.ndim > 1:
                t_safe = t_.view(-1, *([1] * (z.ndim - 1))).clamp(min=0.05)
            else:
                t_safe = t_.clamp(min=0.05)
            return (z - x_hat) / t_safe
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    # Instantaneous velocity estimate: v_tilde = u_theta(z_t, t, t)
    v_tilde = u_fn(z_t, t, t)

    # Compute compound velocity via JVP
    u, V = compute_compound_velocity(u_fn, z_t, t, r, v_tilde)

    # --- Flow matching loss: ||v_tilde - v_c||_p^p ---
    non_batch_dims = list(range(1, x_0.ndim))
    if len(non_batch_dims) == 0:
        non_batch_dims = [0]  # shouldn't happen with batched data

    fm_error = (v_tilde - v_c).abs()
    if p != 1.0:
        fm_error = fm_error.pow(p)
    loss_fm_per_sample = fm_error.sum(dim=non_batch_dims)  # (B,)

    # --- MeanFlow loss: ||V - v_c||_p^p ---
    mf_error = (V - v_c).abs()
    if p != 1.0:
        mf_error = mf_error.pow(p)
    loss_mf_per_sample = mf_error.sum(dim=non_batch_dims)  # (B,)

    # --- Adaptive weighting (Eq. 14) ---
    if adaptive:
        fm_weight = loss_fm_per_sample.detach() + norm_eps
        loss_fm_per_sample = loss_fm_per_sample / fm_weight

        mf_weight = loss_mf_per_sample.detach() + norm_eps
        loss_mf_per_sample = loss_mf_per_sample / mf_weight

    # --- Combine ---
    loss_fm = loss_fm_per_sample.mean()
    loss_mf = loss_mf_per_sample.mean()
    loss = loss_fm + lambda_mf * loss_mf

    return {
        "loss": loss,
        "loss_fm": loss_fm,
        "loss_mf": loss_mf,
    }
