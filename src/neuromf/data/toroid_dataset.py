"""Synthetic toroidal manifold dataset for MeanFlow pipeline validation.

Provides two modes:
- "r4": Flat torus T^2 embedded in R^4 (unnormalised) via phi(theta1, theta2) =
  (cos(theta1), sin(theta1), cos(theta2), sin(theta2)).
  Optionally projects into higher ambient dimension D > 4 using a fixed
  orthogonal projection matrix (following pMF Section 5).
- "volumetric": 4-channel 32^3 volumes parameterised by (theta1, theta2).
"""

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class ToroidConfig:
    """Configuration for toroid dataset generation.

    Args:
        n_samples: Number of samples to generate.
        spatial_size: Spatial extent for volumetric mode.
        n_channels: Number of channels for volumetric mode.
        mode: "r4" for flat torus in R^4, "volumetric" for 3D volumes.
        ambient_dim: Ambient dimension D for the torus embedding. When D > 4,
            samples are projected via a fixed orthogonal matrix.
    """

    n_samples: int = 10_000
    spatial_size: int = 32
    n_channels: int = 4
    mode: str = "r4"
    ambient_dim: int = 4


class ToroidDataset(Dataset):
    """Flat torus dataset in R^4 (or R^D) or volumetric 3D mode.

    Args:
        config: Toroid configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(self, config: ToroidConfig, seed: int = 42) -> None:
        super().__init__()
        self.config = config
        gen = torch.Generator().manual_seed(seed)

        # Sample angles uniformly in [-pi, pi]
        self.theta1 = torch.rand(config.n_samples, generator=gen) * 2 * math.pi - math.pi
        self.theta2 = torch.rand(config.n_samples, generator=gen) * 2 * math.pi - math.pi

        # Build projection matrix for D > 4 (using a separate fixed seed)
        self.projection_matrix: torch.Tensor | None = None
        if config.mode == "r4" and config.ambient_dim > 4:
            proj_gen = torch.Generator().manual_seed(42)
            random_matrix = torch.randn(config.ambient_dim, 4, generator=proj_gen)
            Q, _ = torch.linalg.qr(random_matrix)
            self.projection_matrix = Q[:, :4]  # (D, 4) orthogonal columns

        if config.mode == "r4":
            self.data = self._build_r4()
        elif config.mode == "volumetric":
            self.data = self._build_volumetric()
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    def _build_r4(self) -> torch.Tensor:
        """Build flat torus points in R^4 (unnormalised) or R^D via projection."""
        # Unnormalised embedding: (cos t1, sin t1, cos t2, sin t2), ||z|| = sqrt(2)
        points = torch.stack(
            [
                torch.cos(self.theta1),
                torch.sin(self.theta1),
                torch.cos(self.theta2),
                torch.sin(self.theta2),
            ],
            dim=1,
        )  # (N, 4)

        if self.projection_matrix is not None:
            # Project to R^D: x = P @ z_4 -> (N, D)
            points = points @ self.projection_matrix.t()

        return points

    def _build_volumetric(self) -> torch.Tensor:
        """Build 4-channel volumetric data parameterised by (theta1, theta2)."""
        S = self.config.spatial_size
        C = self.config.n_channels
        N = self.config.n_samples

        coords = torch.linspace(0, 2 * math.pi, S)
        gi, gj, gk = torch.meshgrid(coords, coords, coords, indexing="ij")
        sin_basis = torch.sin(gi) * torch.sin(gj) * torch.sin(gk)
        cos_basis = torch.cos(gi) * torch.cos(gj) * torch.cos(gk)

        sin_basis = sin_basis / (sin_basis.norm() + 1e-8)
        cos_basis = cos_basis / (cos_basis.norm() + 1e-8)

        volumes = torch.zeros(N, C, S, S, S)
        for c in range(C):
            phase = 2 * math.pi * c / C
            a = torch.cos(self.theta1 + phase).view(N, 1, 1, 1)
            b = torch.sin(self.theta2 + phase).view(N, 1, 1, 1)
            volumes[:, c] = a * sin_basis + b * cos_basis

        return volumes

    def project_to_r4(self, samples: torch.Tensor) -> torch.Tensor:
        """Project D-dimensional samples back to R^4.

        Args:
            samples: Tensor of shape (N, D).

        Returns:
            Tensor of shape (N, 4) in the original torus coordinate system.
        """
        if self.projection_matrix is None:
            return samples
        # P is (D, 4), so P^T @ x maps R^D -> R^4
        return samples @ self.projection_matrix

    def __len__(self) -> int:
        return self.config.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
