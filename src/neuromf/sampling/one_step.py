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
) -> Tensor:
    """Generate samples via 1-NFE MeanFlow sampling.

    For u-prediction: z_0 = noise - u_theta(noise, r=0, t=1).
    For x-prediction: z_0 = x_hat = model(noise, r=0, t=1).

    Args:
        model: Network with forward(z_t, r, t) -> output.
        noise: Gaussian noise of shape (B, D) or (B, C, ...).
        prediction_type: "u" for u-prediction, "x" for x-prediction.

    Returns:
        Generated samples of same shape as noise.
    """
    B = noise.shape[0]
    device = noise.device
    r = torch.zeros(B, device=device)
    t = torch.ones(B, device=device)

    output = model(noise, r, t)

    if prediction_type == "x":
        return output  # model directly predicts x_0
    else:
        return noise - output  # z_0 = noise - u
