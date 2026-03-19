"""Multi-step Euler sampling for MeanFlow.

Divides the integration interval [1, 0] into n_steps uniform steps and
applies the solver step z_{r} = z_t - (t - r) * u_theta(z_t, r, t) at each.
Supports both u-prediction and x-prediction parameterisations.
Reference: MeanFlow-PyTorch/meanflow.py:196-203.
"""

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    noise: Tensor,
    n_steps: int = 50,
    prediction_type: str = "u",
    norm_correction: float = 1.0,
    return_trajectory: bool = False,
) -> Tensor | list[tuple[float, Tensor]]:
    """Multi-step Euler sampling from t=1 to t=0.

    For u-prediction: z_{t-dt} = z_t - dt * u_theta(z_t, r=t-dt, t=t) / gamma.
    For x-prediction: convert x_hat to u via u = (z_t - x_hat) / max(t, eps),
    then apply the same Euler step with norm correction.

    Args:
        model: Network with forward(z_t, r, t) -> output.
        noise: Gaussian noise of shape (B, D) or (B, C, ...).
        n_steps: Number of Euler steps.
        prediction_type: "u" for u-prediction, "x" for x-prediction.
        norm_correction: Scalar correction factor gamma. Divides the
            velocity by gamma to compensate for compound velocity norm
            overshoot. Default 1.0 (no correction).
        return_trajectory: If True, return a list of ``(t, z_t)`` tuples
            at each step instead of just the final sample. Useful for
            latent space visualization.

    Returns:
        If ``return_trajectory=False``: generated samples ``(B, C, ...)``.
        If ``return_trajectory=True``: list of ``(t_value, z_t_tensor)``
        tuples from ``t=1`` (noise) to ``t=0`` (data).
    """
    B = noise.shape[0]
    device = noise.device

    t_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    z = noise.clone()
    trajectory: list[tuple[float, Tensor]] = []

    if return_trajectory:
        trajectory.append((1.0, z.clone().cpu()))

    for i in range(n_steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_curr - t_next  # positive

        t_batch = t_curr.expand(B)
        r_batch = t_next.expand(B)

        output = model(z, r_batch, t_batch)

        # Dual-head models return (u_or_x, v) — use u-head only at inference
        if isinstance(output, tuple):
            output = output[0]

        if prediction_type == "x":
            # Convert x-prediction to velocity: u = (z_t - x_hat) / max(t, eps)
            t_safe = max(t_curr.item(), 0.05)
            u = (z - output) / t_safe
        else:
            u = output

        z = z - dt * u / norm_correction

        if return_trajectory:
            trajectory.append((float(t_next.item()), z.clone().cpu()))

    if return_trajectory:
        return trajectory
    return z
