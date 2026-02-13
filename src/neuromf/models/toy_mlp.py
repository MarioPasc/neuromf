"""Toy MLP model for MeanFlow experiments on simple manifolds.

7-layer ReLU MLP that takes (z_t, r, t) concatenated as input and outputs
the average velocity u (u-prediction) or data estimate x_hat (x-prediction).
Uses inplace=False for all ReLUs to ensure torch.func.jvp compatibility.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ToyMLP(nn.Module):
    """7-layer ReLU MLP for toy MeanFlow experiments.

    Input: [z_t, r, t] concatenated (dim = data_dim + 2).
    Output: u-prediction of shape (B, data_dim), or x_hat for x-prediction.

    Args:
        data_dim: Dimensionality of the data (e.g. 4 for R^4 torus).
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers.
        prediction_type: "u" for direct u-prediction, "x" for x-prediction.
    """

    def __init__(
        self,
        data_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 7,
        prediction_type: str = "u",
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.prediction_type = prediction_type

        # Input: z_t (data_dim) + r (1) + t (1)
        input_dim = data_dim + 2
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=False))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z_t: Noisy data of shape (B, data_dim).
            r: Lower time bound of shape (B,) or (B, 1).
            t: Upper time bound of shape (B,) or (B, 1).

        Returns:
            Output of shape (B, data_dim). Interpretation depends on
            prediction_type: "u" returns average velocity, "x" returns
            data estimate x_hat.
        """
        # Ensure r, t are (B, 1) for concatenation
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        x = torch.cat([z_t, r, t], dim=-1)
        return self.net(x)
