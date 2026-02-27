"""Unified feature extraction with HDF5 caching.

Wraps R3D-18 (3D, MOTFM protocol) and RadImageNet (2.5D) feature extractors
with a common interface. Extracts features from volume HDF5 archives and
caches results to feature HDF5 files for reuse across evaluation runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Unified feature extraction from 3D volumes.

    Args:
        backend: ``"med3d"`` (R3D-18, MOTFM protocol) or ``"radimagenet"`` (2.5D).
        weights_path: Path to pretrained weights. Ignored for ``"med3d"``
            backend (R3D-18 uses torchvision built-in weights).
        device: CUDA device.
        center_slices_ratio: For radimagenet, fraction of center slices.
    """

    def __init__(
        self,
        backend: str,
        weights_path: str | Path,
        device: torch.device,
        center_slices_ratio: float = 0.6,
    ) -> None:
        self.backend = backend
        self.device = device
        self.center_slices_ratio = center_slices_ratio

        if backend == "med3d":
            from neuromf.metrics.fid_3d import load_fid3d_feature_net

            wp = str(weights_path) if weights_path else None
            self.model = load_fid3d_feature_net(device=device, weights_path=wp)
        elif backend == "radimagenet":
            from neuromf.metrics.fid import load_radimagenet_resnet50

            self.model = load_radimagenet_resnet50(weights_path).to(device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'med3d' or 'radimagenet'.")

    @torch.no_grad()
    def extract_and_cache(
        self,
        volume_h5_path: Path,
        output_h5_path: Path,
        batch_size: int = 1,
    ) -> Tensor:
        """Extract features from all volumes and cache to HDF5.

        Args:
            volume_h5_path: Path to volume HDF5 archive.
            output_h5_path: Path to output feature HDF5 file.
            batch_size: Feature extraction batch size.

        Returns:
            Feature matrix of shape ``(N, D)`` where D is the feature dim.
        """
        volume_h5_path = Path(volume_h5_path)
        output_h5_path = Path(output_h5_path)
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(volume_h5_path), "r") as vf:
            n_samples = vf["volumes"].shape[0]

        logger.info(
            "Extracting %s features from %d volumes (%s)",
            self.backend,
            n_samples,
            volume_h5_path.name,
        )

        if self.backend == "med3d":
            features = self._extract_med3d_features(volume_h5_path, n_samples)
        else:
            features = self._extract_radimagenet_features(
                volume_h5_path, n_samples
            )

        # Cache to HDF5
        with h5py.File(str(output_h5_path), "w") as ff:
            ff.create_dataset("features", data=features.numpy())
            ff.create_dataset(
                "source_indices",
                data=np.arange(n_samples, dtype=np.int64),
            )
            ff.attrs["backend"] = self.backend
            ff.attrs["feature_dim"] = features.shape[1]
            ff.attrs["n_samples"] = n_samples
            ff.attrs["source_h5"] = str(volume_h5_path.name)

        logger.info(
            "Features cached: %s (%d, %d)",
            output_h5_path.name,
            *features.shape,
        )
        return features

    def _extract_med3d_features(
        self,
        volume_h5_path: Path,
        n_samples: int,
    ) -> Tensor:
        """Load all volumes, then extract R3D-18 features with per-set normalisation."""
        from neuromf.metrics.fid_3d import extract_3d_features

        volumes: list[Tensor] = []
        with h5py.File(str(volume_h5_path), "r") as vf:
            for i in range(n_samples):
                vol = torch.from_numpy(vf["volumes"][i].astype(np.float32))
                if vol.ndim == 3:
                    vol = vol.unsqueeze(0).unsqueeze(0)
                elif vol.ndim == 4:
                    vol = vol.unsqueeze(0)
                volumes.append(vol)
                if (i + 1) % 50 == 0 or i == n_samples - 1:
                    logger.info("  Loaded %d / %d volumes", i + 1, n_samples)

        all_volumes = torch.cat(volumes, dim=0)  # (N, 1, D, H, W)
        return extract_3d_features(all_volumes, self.model, normalize=True)

    def _extract_radimagenet_features(
        self,
        volume_h5_path: Path,
        n_samples: int,
    ) -> Tensor:
        """Extract RadImageNet 2.5D features per-volume, averaged across planes."""
        from neuromf.metrics.fid import extract_2d5_features

        all_features: list[Tensor] = []
        with h5py.File(str(volume_h5_path), "r") as vf:
            for i in range(n_samples):
                vol = torch.from_numpy(vf["volumes"][i].astype(np.float32))
                if vol.ndim == 3:
                    vol = vol.unsqueeze(0).unsqueeze(0)

                xy, yz, zx = extract_2d5_features(
                    vol, self.model, self.center_slices_ratio, batch_size=16
                )
                feat = torch.cat(
                    [
                        xy.mean(0, keepdim=True),
                        yz.mean(0, keepdim=True),
                        zx.mean(0, keepdim=True),
                    ],
                    dim=0,
                ).mean(0, keepdim=True)
                all_features.append(feat)

                if (i + 1) % 50 == 0 or i == n_samples - 1:
                    logger.info("  Extracted %d / %d", i + 1, n_samples)

        return torch.cat(all_features, dim=0)

    @staticmethod
    def load_cached(feature_h5_path: Path) -> Tensor:
        """Load previously cached features from HDF5.

        Args:
            feature_h5_path: Path to feature HDF5 file.

        Returns:
            Feature matrix of shape ``(N, D)``.
        """
        with h5py.File(str(feature_h5_path), "r") as f:
            features = torch.from_numpy(f["features"][:].astype(np.float32))
        return features
