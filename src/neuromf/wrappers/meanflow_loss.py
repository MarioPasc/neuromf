"""Full MeanFlow loss pipeline with configurable JVP strategy.

Wraps the JVP strategy, compound velocity computation, and iMF combined
loss (Eq. 13) into a single module. Supports both exact JVP and
finite-difference approximation for hardware flexibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neuromf.losses.lp_loss import lp_loss
from neuromf.wrappers.jvp_strategies import ExactJVP, FiniteDifferenceJVP

logger = logging.getLogger(__name__)


@dataclass
class MeanFlowPipelineConfig:
    """Configuration for MeanFlow loss pipeline."""

    p: float = 2.0
    adaptive: bool = True
    norm_eps: float = 0.01
    lambda_mf: float = 1.0
    prediction_type: str = "x"
    t_min: float = 0.05
    jvp_strategy: str = "exact"
    fd_step_size: float = 1e-3
    channel_weights: list[float] | None = None
    norm_p: float = 1.0


class MeanFlowPipeline(nn.Module):
    """Full MeanFlow loss computation pipeline.

    Orchestrates:
    1. Interpolation ``z_t = (1-t)*z_0 + t*eps``
    2. Instantaneous velocity ``v_tilde = u(z_t, t, t)``
    3. JVP-based compound velocity ``V``
    4. iMF combined loss (Eq. 13) with per-channel Lp and adaptive weighting

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: MeanFlowPipelineConfig) -> None:
        super().__init__()
        self.config = config

        if config.jvp_strategy == "exact":
            self.jvp = ExactJVP()
        elif config.jvp_strategy == "finite_difference":
            self.jvp = FiniteDifferenceJVP(h=config.fd_step_size)
        else:
            raise ValueError(f"Unknown JVP strategy: {config.jvp_strategy}")

        if config.channel_weights is not None:
            self.register_buffer(
                "_channel_weights",
                torch.tensor(config.channel_weights, dtype=torch.float32),
            )
        else:
            self._channel_weights: Tensor | None = None

        logger.info(
            "MeanFlowPipeline: p=%.1f, adaptive=%s, jvp=%s, prediction=%s",
            config.p,
            config.adaptive,
            config.jvp_strategy,
            config.prediction_type,
        )

    def _make_u_fn(self, model: nn.Module):
        """Create u_fn closure for JVP computation.

        Args:
            model: Network with ``forward(z_t, r, t) -> output``.

        Returns:
            Callable ``(z_t, t, r) -> u`` (average velocity).
        """
        prediction_type = self.config.prediction_type
        t_min = self.config.t_min

        if prediction_type == "u":

            def u_fn(z: Tensor, t_: Tensor, r_: Tensor) -> Tensor:
                return model(z, r_, t_)

        elif prediction_type == "x":

            def u_fn(z: Tensor, t_: Tensor, r_: Tensor) -> Tensor:
                x_hat = model(z, r_, t_)
                if z.ndim > 1:
                    t_safe = t_.view(-1, *([1] * (z.ndim - 1))).clamp(min=t_min)
                else:
                    t_safe = t_.clamp(min=t_min)
                return (z - x_hat) / t_safe

        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        return u_fn

    def forward(
        self,
        model: nn.Module,
        z_0: Tensor,
        eps: Tensor,
        t: Tensor,
        r: Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Compute iMF combined loss (Eq. 13).

        Args:
            model: Network with ``forward(z_t, r, t) -> output``.
            z_0: Clean data ``(B, C, ...)``.
            eps: Noise ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            return_diagnostics: If True, include detached diagnostic tensors
                in the return dict (zero overhead when False).

        Returns:
            Dict with ``"loss"``, ``"loss_fm"``, ``"loss_mf"``, and optionally
            ``"diag_*"`` keys when ``return_diagnostics=True``.
        """
        p = self.config.p
        adaptive = self.config.adaptive
        norm_eps = self.config.norm_eps
        lambda_mf = self.config.lambda_mf

        # Interpolate: z_t = (1-t)*z_0 + t*eps
        if z_0.ndim > 1:
            shape = (-1,) + (1,) * (z_0.ndim - 1)
            t_broad = t.view(*shape)
        else:
            t_broad = t

        z_t = (1 - t_broad) * z_0 + t_broad * eps

        # Conditional velocity
        v_c = eps - z_0

        # Build u_fn closure
        u_fn = self._make_u_fn(model)

        # Instantaneous velocity: v_tilde = u(z_t, t, t)
        v_tilde = u_fn(z_t, t, t)

        # Compound velocity via JVP
        u, V = self.jvp.compute(u_fn, z_t, t, r, v_tilde)

        # --- Flow matching loss ---
        cw = self._channel_weights
        raw_fm_per_sample = lp_loss(v_tilde, v_c, p=p, channel_weights=cw, reduction="none")

        # --- MeanFlow loss ---
        raw_mf_per_sample = lp_loss(V, v_c, p=p, channel_weights=cw, reduction="none")

        # --- Adaptive weighting ---
        if adaptive:
            norm_p = self.config.norm_p
            fm_weight = (raw_fm_per_sample.detach() + norm_eps) ** norm_p
            loss_fm_per_sample = raw_fm_per_sample / fm_weight

            mf_weight = (raw_mf_per_sample.detach() + norm_eps) ** norm_p
            loss_mf_per_sample = raw_mf_per_sample / mf_weight
        else:
            loss_fm_per_sample = raw_fm_per_sample
            loss_mf_per_sample = raw_mf_per_sample

        # Combine
        loss_fm = loss_fm_per_sample.mean()
        loss_mf = loss_mf_per_sample.mean()
        loss = loss_fm + lambda_mf * loss_mf

        # Raw (pre-adaptive) loss — always returned for observability
        raw_loss_fm = raw_fm_per_sample.detach().mean()
        raw_loss_mf = raw_mf_per_sample.detach().mean()

        result: dict[str, Tensor] = {
            "loss": loss,
            "loss_fm": loss_fm,
            "loss_mf": loss_mf,
            "raw_loss_fm": raw_loss_fm,
            "raw_loss_mf": raw_loss_mf,
        }

        if return_diagnostics:
            result.update(
                self._compute_diagnostics(
                    u,
                    V,
                    v_tilde,
                    v_c,
                    t,
                    r,
                    p,
                    lambda_mf,
                    raw_fm_per_sample,
                    raw_mf_per_sample,
                    fm_weight if adaptive else None,
                    mf_weight if adaptive else None,
                )
            )

        return result

    def _compute_diagnostics(
        self,
        u: Tensor,
        V: Tensor,
        v_tilde: Tensor,
        v_c: Tensor,
        t: Tensor,
        r: Tensor,
        p: float,
        lambda_mf: float,
        loss_fm_per_sample: Tensor,
        loss_mf_per_sample: Tensor,
        fm_weight: Tensor | None,
        mf_weight: Tensor | None,
    ) -> dict[str, Tensor]:
        """Compute detached diagnostic tensors from existing intermediates.

        All returned tensors are detached and do not participate in backward.

        Args:
            u: Average velocity ``(B, C, ...)``.
            V: Compound velocity ``(B, C, ...)``.
            v_tilde: Instantaneous velocity ``(B, C, ...)``.
            v_c: Conditional velocity ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            p: Norm exponent.
            lambda_mf: MeanFlow loss weight.
            loss_fm_per_sample: Per-sample raw FM loss ``(B,)`` (pre-adaptive-weighting).
            loss_mf_per_sample: Per-sample raw MF loss ``(B,)`` (pre-adaptive-weighting).
            fm_weight: Adaptive FM weights (None if not adaptive).
            mf_weight: Adaptive MF weights (None if not adaptive).

        Returns:
            Dict with ``diag_*`` keys, all detached scalars or small tensors.
        """
        diag: dict[str, Tensor] = {}

        # Norm diagnostics (batch means of L2 norms)
        diag["diag_u_norm"] = u.detach().flatten(1).norm(dim=1).mean()
        diag["diag_v_tilde_norm"] = v_tilde.detach().flatten(1).norm(dim=1).mean()
        diag["diag_compound_v_norm"] = V.detach().flatten(1).norm(dim=1).mean()
        diag["diag_target_v_norm"] = v_c.detach().flatten(1).norm(dim=1).mean()

        # JVP norm recovery: du/dt = (V - u) / (t - r) for MF samples
        h = (t - r).detach()
        mf_mask = h > 1e-6
        if mf_mask.any():
            h_broad = h[mf_mask].view(-1, *([1] * (u.ndim - 1)))
            jvp_approx = (V[mf_mask].detach() - u[mf_mask].detach()) / h_broad
            diag["diag_jvp_norm"] = jvp_approx.flatten(1).norm(dim=1).mean()
        else:
            diag["diag_jvp_norm"] = torch.tensor(0.0, device=u.device)

        # Adaptive weight stats
        if fm_weight is not None and mf_weight is not None:
            all_weights = torch.cat([fm_weight.detach(), mf_weight.detach()])
            diag["diag_adaptive_weight_mean"] = all_weights.mean()
            diag["diag_adaptive_weight_std"] = all_weights.std()
        else:
            diag["diag_adaptive_weight_mean"] = torch.tensor(0.0, device=u.device)
            diag["diag_adaptive_weight_std"] = torch.tensor(0.0, device=u.device)

        # Per-channel loss (recompute cheaply from existing tensors)
        spatial_dims = list(range(2, v_tilde.ndim))
        ch_fm = (v_tilde.detach() - v_c.detach()).abs().pow(p).mean(dim=[0] + spatial_dims)
        ch_mf = (V.detach() - v_c.detach()).abs().pow(p).mean(dim=[0] + spatial_dims)
        diag["diag_loss_per_channel"] = ch_fm + lambda_mf * ch_mf

        # Per-sample losses (pre-adaptive-weighting raw values)
        diag["diag_loss_fm_per_sample"] = loss_fm_per_sample.detach()
        diag["diag_loss_mf_per_sample"] = loss_mf_per_sample.detach()

        return diag
