"""Full MeanFlow loss pipeline with configurable JVP strategy.

Wraps the JVP strategy, compound velocity computation, and unified
MeanFlow loss into a single module. Uses the iMF formulation where
the model's own predicted velocity v_tilde = u(z_t, t, t) serves as
the JVP tangent, making V a legitimate prediction function of z_t
alone (iMF Section 4.1, Algorithm 1). Computes a single loss
||V - v_c||^p with optional adaptive weighting.
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
    lambda_mf: float = 1.0  # kept for backward compat, unused in forward
    prediction_type: str = "x"
    t_min: float = 0.05
    jvp_strategy: str = "exact"
    fd_step_size: float = 1e-3
    channel_weights: list[float] | None = None
    norm_p: float = 1.0
    spatial_mask_ratio: float = 0.0  # 0.0=disabled, 0.5=mask 50% spatial voxels
    use_v_head: bool = False


class MeanFlowPipeline(nn.Module):
    """Full MeanFlow loss computation pipeline (iMF formulation).

    Orchestrates:
    1. Interpolation ``z_t = (1-t)*z_0 + t*eps``
    2. Conditional velocity ``v_c = eps - z_0`` (target)
    3. Model predicted velocity ``v_tilde = u(z_t, t, t)`` (JVP tangent)
    4. JVP-based compound velocity ``V`` using ``v_tilde`` as tangent
    5. Single unified loss ``||V - v_c||^p`` with adaptive weighting

    Using the model's own prediction as tangent (iMF Algorithm 1) makes
    V depend only on z_t, not on the ground-truth e-x. This produces
    lower-variance gradients and stable training (iMF Section 4.1).

    For FM samples (r=t), V reduces to u (instantaneous velocity),
    recovering the standard flow matching loss. For MF samples (r<t),
    V = u + (t-r)*sg[du/dt] enforces the MeanFlow self-consistency.

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
            "MeanFlowPipeline: p=%.1f, adaptive=%s, jvp=%s, prediction=%s, v_head=%s",
            config.p,
            config.adaptive,
            config.jvp_strategy,
            config.prediction_type,
            config.use_v_head,
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

    def _make_dual_fn(self, model: nn.Module):
        """Create dual-output closure for v-head models.

        Args:
            model: Network with ``forward(z_t, r, t) -> (u_or_x, v)``.

        Returns:
            Callable ``(z_t, t, r) -> (u, v)`` where u is average velocity.
        """
        prediction_type = self.config.prediction_type
        t_min = self.config.t_min

        if prediction_type == "u":

            def dual_fn(z: Tensor, t_: Tensor, r_: Tensor) -> tuple[Tensor, Tensor]:
                u, v = model(z, r_, t_)
                return u, v

        elif prediction_type == "x":

            def dual_fn(z: Tensor, t_: Tensor, r_: Tensor) -> tuple[Tensor, Tensor]:
                x_hat, v = model(z, r_, t_)
                if z.ndim > 1:
                    t_safe = t_.view(-1, *([1] * (z.ndim - 1))).clamp(min=t_min)
                else:
                    t_safe = t_.clamp(min=t_min)
                u = (z - x_hat) / t_safe
                return u, v

        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        return dual_fn

    def forward(
        self,
        model: nn.Module,
        z_0: Tensor,
        eps: Tensor,
        t: Tensor,
        r: Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Compute unified MeanFlow loss (iMF Algorithm 1).

        Uses the model's own predicted velocity ``v_tilde = u(z_t, t, t)``
        as the JVP tangent, following the iMF formulation. This makes V a
        function of z_t alone, enabling stable training. The target remains
        ``v_c = eps - z_0``.

        Args:
            model: Network with ``forward(z_t, r, t) -> output``.
            z_0: Clean data ``(B, C, ...)``.
            eps: Noise ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            return_diagnostics: If True, include detached diagnostic tensors
                in the return dict (zero overhead when False).

        Returns:
            Dict with ``"loss"`` and ``"raw_loss"``, and optionally
            ``"diag_*"`` keys when ``return_diagnostics=True``.
        """
        p = self.config.p
        adaptive = self.config.adaptive
        norm_eps = self.config.norm_eps
        norm_p = self.config.norm_p

        # Interpolate: z_t = (1-t)*z_0 + t*eps
        if z_0.ndim > 1:
            shape = (-1,) + (1,) * (z_0.ndim - 1)
            t_broad = t.view(*shape)
        else:
            t_broad = t

        z_t = (1 - t_broad) * z_0 + t_broad * eps

        # Ground-truth conditional velocity (target, not tangent)
        v_c = eps - z_0

        cw = self._channel_weights

        if self.config.use_v_head:
            return self._forward_dual_head(
                model,
                z_t,
                v_c,
                t,
                r,
                p,
                adaptive,
                norm_eps,
                norm_p,
                cw,
                return_diagnostics,
            )
        else:
            return self._forward_single_head(
                model,
                z_t,
                v_c,
                t,
                r,
                p,
                adaptive,
                norm_eps,
                norm_p,
                cw,
                return_diagnostics,
            )

    def _forward_single_head(
        self,
        model: nn.Module,
        z_t: Tensor,
        v_c: Tensor,
        t: Tensor,
        r: Tensor,
        p: float,
        adaptive: bool,
        norm_eps: float,
        norm_p: float,
        cw: Tensor | None,
        return_diagnostics: bool,
    ) -> dict[str, Tensor]:
        """Single-head forward pass (original path, unchanged)."""
        u_fn = self._make_u_fn(model)

        with torch.no_grad():
            v_tilde = u_fn(z_t, t, t)

        u, V = self.jvp.compute(u_fn, z_t, t, r, v_tilde)

        # Spatial loss masking
        mask_ratio = self.config.spatial_mask_ratio
        if mask_ratio > 0.0:
            keep_prob = 1.0 - mask_ratio
            spatial_shape = V.shape[2:]
            mask = (torch.rand(V.shape[0], 1, *spatial_shape, device=V.device) < keep_prob).float()
            V_for_loss = V * mask
            v_c_for_loss = v_c * mask
        else:
            V_for_loss = V
            v_c_for_loss = v_c
            keep_prob = 1.0

        raw_loss_per_sample = lp_loss(
            V_for_loss, v_c_for_loss, p=p, channel_weights=cw, reduction="none"
        )

        if mask_ratio > 0.0:
            raw_loss_per_sample = raw_loss_per_sample / keep_prob

        if adaptive:
            adp_weight = (raw_loss_per_sample.detach() + norm_eps) ** norm_p
            loss_per_sample = raw_loss_per_sample / adp_weight
        else:
            loss_per_sample = raw_loss_per_sample

        loss = loss_per_sample.mean()
        raw_loss = raw_loss_per_sample.detach().mean()

        result: dict[str, Tensor] = {
            "loss": loss,
            "raw_loss": raw_loss,
        }

        if return_diagnostics:
            result.update(
                self._compute_diagnostics(
                    u,
                    V,
                    v_c,
                    v_tilde,
                    t,
                    r,
                    p,
                    raw_loss_per_sample,
                    adp_weight if adaptive else None,
                    z_t=z_t,
                )
            )

        return result

    def _forward_dual_head(
        self,
        model: nn.Module,
        z_t: Tensor,
        v_c: Tensor,
        t: Tensor,
        r: Tensor,
        p: float,
        adaptive: bool,
        norm_eps: float,
        norm_p: float,
        cw: Tensor | None,
        return_diagnostics: bool,
    ) -> dict[str, Tensor]:
        """Dual-head forward pass (iMF v-head architecture).

        Uses the v-head output as the JVP tangent instead of the u-head's
        instantaneous velocity. The v-head is directly supervised to predict
        ``v_c``, providing a high-quality tangent from early training.
        """
        dual_fn = self._make_dual_fn(model)

        # v-head tangent: directly supervised to match v_c
        with torch.no_grad():
            _, v_tangent = dual_fn(z_t, t, t)  # r=t for instantaneous

        # JVP with dual output
        u, V, v = self.jvp.compute_dual(dual_fn, z_t, t, r, v_tangent)

        # loss_u: ||V - v_c||^p with adaptive weighting
        raw_loss_u = lp_loss(V, v_c, p=p, channel_weights=cw, reduction="none")

        # loss_v: ||v - v_c||^p with independent adaptive weighting
        raw_loss_v = lp_loss(v, v_c, p=p, channel_weights=cw, reduction="none")

        if adaptive:
            adp_weight_u = (raw_loss_u.detach() + norm_eps) ** norm_p
            loss_u = raw_loss_u / adp_weight_u
            adp_weight_v = (raw_loss_v.detach() + norm_eps) ** norm_p
            loss_v = raw_loss_v / adp_weight_v
        else:
            loss_u = raw_loss_u
            loss_v = raw_loss_v

        loss = (loss_u + loss_v).mean()
        raw_loss = (raw_loss_u.detach() + raw_loss_v.detach()).mean()

        result: dict[str, Tensor] = {
            "loss": loss,
            "raw_loss": raw_loss,
            "raw_loss_u": raw_loss_u.detach().mean(),
            "raw_loss_v": raw_loss_v.detach().mean(),
        }

        if return_diagnostics:
            result.update(
                self._compute_diagnostics(
                    u,
                    V,
                    v_c,
                    v_tangent,
                    t,
                    r,
                    p,
                    raw_loss_u,
                    adp_weight_u if adaptive else None,
                    z_t=z_t,
                )
            )
            # v-head specific diagnostics
            v_norms = v.detach().flatten(1).norm(dim=1)
            result["diag_v_head_norm"] = v_norms.mean()
            vc_flat = v_c.detach().flatten(1)
            v_flat = v.detach().flatten(1)
            cos_sim_v_vc = torch.nn.functional.cosine_similarity(v_flat, vc_flat, dim=1)
            result["diag_cosine_sim_v_vc"] = cos_sim_v_vc.mean()
            result["diag_raw_loss_u_per_sample"] = raw_loss_u.detach().mean()
            result["diag_raw_loss_v_per_sample"] = raw_loss_v.detach().mean()

        return result

    def _compute_diagnostics(
        self,
        u: Tensor,
        V: Tensor,
        v_c: Tensor,
        v_tilde: Tensor,
        t: Tensor,
        r: Tensor,
        p: float,
        raw_loss_per_sample: Tensor,
        adp_weight: Tensor | None,
        z_t: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute detached diagnostic tensors from existing intermediates.

        All returned tensors are detached and do not participate in backward.

        Args:
            u: Average velocity ``(B, C, ...)``.
            V: Compound velocity ``(B, C, ...)``.
            v_c: Conditional velocity ``(B, C, ...)``.
            v_tilde: Model predicted velocity used as JVP tangent ``(B, C, ...)``.
            t: Upper time ``(B,)``.
            r: Lower time ``(B,)``.
            p: Norm exponent.
            raw_loss_per_sample: Per-sample raw loss ``(B,)`` (pre-adaptive).
            adp_weight: Adaptive weights (None if not adaptive).
            z_t: Noisy interpolation ``(B, C, ...)`` (optional, for x-hat stats).

        Returns:
            Dict with ``diag_*`` keys, all detached scalars or small tensors.
        """
        diag: dict[str, Tensor] = {}
        _zero = torch.tensor(0.0, device=u.device)

        # --- Norm diagnostics (batch means of L2 norms) ---
        u_norms = u.detach().flatten(1).norm(dim=1)
        V_norms = V.detach().flatten(1).norm(dim=1)
        vc_norms = v_c.detach().flatten(1).norm(dim=1)
        vtilde_norms = v_tilde.detach().flatten(1).norm(dim=1)

        diag["diag_u_norm"] = u_norms.mean()
        diag["diag_compound_v_norm"] = V_norms.mean()
        diag["diag_target_v_norm"] = vc_norms.mean()
        diag["diag_v_tilde_norm"] = vtilde_norms.mean()

        # --- JVP norm: du/dt = (V - u) / (t - r) for MF samples ---
        h = (t - r).detach()
        mf_mask = h > 1e-6
        fm_mask = h < 1e-6
        if mf_mask.any():
            h_broad = h[mf_mask].view(-1, *([1] * (u.ndim - 1)))
            jvp_approx = (V[mf_mask].detach() - u[mf_mask].detach()) / h_broad
            diag["diag_jvp_norm"] = jvp_approx.flatten(1).norm(dim=1).mean()
        else:
            diag["diag_jvp_norm"] = _zero

        # --- Adaptive weight stats ---
        if adp_weight is not None:
            diag["diag_adaptive_weight_mean"] = adp_weight.detach().mean()
            diag["diag_adaptive_weight_std"] = adp_weight.detach().std()
        else:
            diag["diag_adaptive_weight_mean"] = _zero
            diag["diag_adaptive_weight_std"] = _zero

        # --- Per-channel loss from (V - v_c) ---
        spatial_dims = list(range(2, V.ndim))
        ch_loss = (V.detach() - v_c.detach()).abs().pow(p).mean(dim=[0] + spatial_dims)
        diag["diag_loss_per_channel"] = ch_loss

        # --- Per-sample raw loss (pre-adaptive-weighting) ---
        diag["diag_loss_per_sample"] = raw_loss_per_sample.detach()

        # --- FM/MF split ---
        diag["diag_fm_fraction"] = fm_mask.float().mean()
        if fm_mask.any():
            diag["diag_raw_loss_fm"] = raw_loss_per_sample[fm_mask].detach().mean()
        else:
            diag["diag_raw_loss_fm"] = _zero
        if mf_mask.any():
            diag["diag_raw_loss_mf"] = raw_loss_per_sample[mf_mask].detach().mean()
        else:
            diag["diag_raw_loss_mf"] = _zero

        # --- Cosine similarity: V vs v_c (direction alignment) ---
        V_flat = V.detach().flatten(1)
        vc_flat = v_c.detach().flatten(1)
        cos_sim = torch.nn.functional.cosine_similarity(V_flat, vc_flat, dim=1)
        diag["diag_cosine_sim_V_vc"] = cos_sim.mean()

        # --- Relative prediction error: ||V - v_c|| / ||v_c|| ---
        error_norms = (V.detach() - v_c.detach()).flatten(1).norm(dim=1)
        rel_error = error_norms / (vc_norms + 1e-8)
        diag["diag_relative_error"] = rel_error.mean()

        # --- v_tilde alignment with v_c (tangent quality) ---
        vtilde_flat = v_tilde.detach().flatten(1)
        cos_sim_tangent = torch.nn.functional.cosine_similarity(vtilde_flat, vc_flat, dim=1)
        diag["diag_cosine_sim_vtilde_vc"] = cos_sim_tangent.mean()

        # --- Prediction-specific statistics ---
        if self.config.prediction_type == "x" and z_t is not None:
            t_safe = t.detach().view(-1, *([1] * (z_t.ndim - 1))).clamp(min=self.config.t_min)
            x_hat = z_t.detach() - t_safe * u.detach()
            diag["diag_x_hat_mean"] = x_hat.mean()
            diag["diag_x_hat_std"] = x_hat.std()
            diag["diag_x_hat_min"] = x_hat.min()
            diag["diag_x_hat_max"] = x_hat.max()
        elif self.config.prediction_type == "u":
            diag["diag_u_pred_mean"] = u.detach().mean()
            diag["diag_u_pred_std"] = u.detach().std()
            diag["diag_u_pred_min"] = u.detach().min()
            diag["diag_u_pred_max"] = u.detach().max()

        return diag
