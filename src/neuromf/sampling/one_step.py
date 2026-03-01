"""1-NFE one-step MeanFlow sampling.

Generates samples via a single forward pass: z_0 = eps - u_theta(eps, r=0, t=1).
Supports both u-prediction (direct) and x-prediction (model outputs x_hat).
Reference: MeanFlow-PyTorch/meanflow.py:196-203.
"""

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def sample_one_step(
    model: nn.Module,
    noise: Tensor,
    prediction_type: str = "u",
    norm_correction: float = 1.0,
) -> Tensor:
    """Generate samples via 1-NFE MeanFlow sampling.

    For u-prediction: z_0 = noise - u_theta(noise, r=0, t=1) / gamma.
    For x-prediction: z_0 = noise - (noise - x_hat) / gamma.

    Args:
        model: Network with forward(z_t, r, t) -> output.
        noise: Gaussian noise of shape (B, D) or (B, C, ...).
        prediction_type: "u" for u-prediction, "x" for x-prediction.
        norm_correction: Scalar correction factor gamma. Divides the
            velocity by gamma to compensate for compound velocity norm
            overshoot. Default 1.0 (no correction).

    Returns:
        Generated samples of same shape as noise.
    """
    B = noise.shape[0]
    device = noise.device
    r = torch.zeros(B, device=device)
    t = torch.ones(B, device=device)

    output = model(noise, r, t)

    # Dual-head models return (u_or_x, v) — use u-head only at inference
    if isinstance(output, tuple):
        output = output[0]

    if prediction_type == "x":
        if norm_correction != 1.0:
            # x-pred: u = (noise - x_hat) / t, t=1 => u = noise - x_hat
            # corrected: z_0 = noise - u / gamma = noise - (noise - x_hat) / gamma
            return noise - (noise - output) / norm_correction
        return output  # model directly predicts x_0
    else:
        return noise - output / norm_correction  # z_0 = noise - u / gamma
