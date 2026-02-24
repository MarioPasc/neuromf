"""3D-FID computation using Med3D ResNet-50 features.

Extracts 2048-d volumetric features from 3D medical volumes using a
Med3D-pretrained ResNet-50 (Chen et al., 2019), then computes FID
between real and generated feature distributions.

Matches the 3D-FID protocol used by MOTFM (Table 2) and HA-GAN (Sun et al., 2022):
feature extractor = ``resnet_50_23dataset.pth`` (23-dataset pretraining),
preprocessing = per-volume min-max normalisation to [0, 1],
FID = standard Frechet distance on 2048-d vectors.

Standalone usage::

    from neuromf.metrics.fid_3d import load_med3d_resnet50, extract_3d_features, compute_fid_3d

    net = load_med3d_resnet50("/path/to/resnet_50_23dataset.pth")
    net = net.to(device).eval()

    real_feats = torch.cat([extract_3d_features(v, net) for v in real_vols])
    fake_feats = torch.cat([extract_3d_features(v, net) for v in fake_vols])
    fid = compute_fid_3d(real_feats, fake_feats)

Reference:
    Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis",
    arXiv:1904.00625, 2019.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from neuromf.metrics.resnet3d import load_med3d_weights, resnet3d_50

logger = logging.getLogger(__name__)


def load_med3d_resnet50(weights_path: str | Path) -> nn.Module:
    """Load Med3D ResNet-50 as a 3D feature extractor (2048-d output).

    Args:
        weights_path: Path to ``resnet_50_23dataset.pth``.

    Returns:
        ResNet3D model in eval mode with pretrained weights.
    """
    model = resnet3d_50(shortcut_type="B", in_channels=1)
    load_med3d_weights(model, weights_path)
    model.eval()
    logger.info("Loaded Med3D ResNet-50 from %s (2048-d features)", weights_path)
    return model


@torch.no_grad()
def extract_3d_features(
    volume: Tensor,
    feature_network: nn.Module,
    normalize: bool = True,
) -> Tensor:
    """Extract 3D features from a single volume.

    Args:
        volume: Single volume ``(1, 1, D, H, W)`` or ``(1, D, H, W)``.
        feature_network: Med3D ResNet-50 feature extractor.
        normalize: If True, apply per-volume min-max normalisation to [0, 1].

    Returns:
        Feature vector ``(1, 2048)``.
    """
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    assert volume.ndim == 5 and volume.shape[0] == 1, (
        f"Expected (1, C, D, H, W), got {volume.shape}"
    )

    # Ensure single-channel input
    if volume.shape[1] != 1:
        volume = volume[:, :1]

    device = next(feature_network.parameters()).device
    vol = volume.to(device).float()

    if normalize:
        vmin = vol.min()
        vmax = vol.max()
        vol = (vol - vmin) / (vmax - vmin + 1e-10)

    return feature_network(vol).cpu()


def compute_fid_3d(
    real_features: Tensor,
    fake_features: Tensor,
) -> float:
    """Compute FID between two sets of 3D features.

    Uses MONAI's ``FIDMetric`` for the Frechet distance computation,
    consistent with the 2.5D-FID path.

    Args:
        real_features: Real feature vectors ``(N_real, 2048)``.
        fake_features: Generated feature vectors ``(N_fake, 2048)``.

    Returns:
        FID score (lower is better). Returns 0.0 for identical distributions.
    """
    from monai.metrics.fid import FIDMetric

    fid_metric = FIDMetric()
    fid_val = fid_metric(fake_features, real_features)
    return float(fid_val)
