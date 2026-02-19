"""2.5D FID computation using RadImageNet ResNet-50 features.

Extracts features from 3 orthogonal planes (XY, YZ, ZX) of 3D volumes
using a RadImageNet-pretrained ResNet-50, then computes per-plane FID via
MONAI's ``FIDMetric``. Follows the protocol from NV-Generate-CTMR.

Reference:
    ``src/external/NV-Generate-CTMR/scripts/compute_fid_2-5d_ct.py``
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50

logger = logging.getLogger(__name__)

# ImageNet BGR channel means (matching MAISI reference: subtract_mean())
_IMAGENET_BGR_MEAN = [0.406, 0.456, 0.485]


def load_radimagenet_resnet50(weights_path: str | Path) -> nn.Module:
    """Load RadImageNet ResNet-50 as a 2048-d feature extractor.

    Creates a ``torchvision.models.resnet50``, loads the state dict from
    disk, and replaces the final FC layer with ``nn.Identity()`` so the
    output is a 2048-d feature vector after global average pooling.

    Args:
        weights_path: Path to the saved ``.pt`` state dict file.

    Returns:
        Feature extractor model in eval mode.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"RadImageNet weights not found: {weights_path}")

    model = resnet50(weights=None)
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Replace FC with identity to get 2048-d features
    model.fc = nn.Identity()
    model.eval()

    logger.info("Loaded RadImageNet ResNet-50 from %s (2048-d features)", weights_path)
    return model


def _radimagenet_normalise(slices_4d: Tensor) -> Tensor:
    """Normalise 2D slices for RadImageNet feature extraction.

    Applies global min-max normalisation to [0, 1], then subtracts
    ImageNet BGR channel means. Matches the MAISI reference:
    ``radimagenet_intensity_normalisation()`` with ``norm2d=False``.

    Args:
        slices_4d: Tensor of shape ``(B, 3, H, W)`` in BGR channel order.

    Returns:
        Normalised tensor of same shape.
    """
    # Global min-max to [0, 1]
    minval = slices_4d.min()
    maxval = slices_4d.max()
    slices_4d = (slices_4d - minval) / (maxval - minval + 1e-10)

    # Subtract per-channel BGR means
    slices_4d[:, 0, ...] -= _IMAGENET_BGR_MEAN[0]
    slices_4d[:, 1, ...] -= _IMAGENET_BGR_MEAN[1]
    slices_4d[:, 2, ...] -= _IMAGENET_BGR_MEAN[2]

    return slices_4d


def _spatial_average(x: Tensor) -> Tensor:
    """Global average pooling over spatial dimensions.

    Args:
        x: Feature map of shape ``(B, C, H, W)`` or ``(B, C, H, W, D)``.

    Returns:
        Pooled tensor of shape ``(B, C)``.
    """
    spatial_dims = list(range(2, x.ndim))
    return x.mean(dim=spatial_dims)


def _extract_plane_features(
    slices_list: list[Tensor],
    feature_network: nn.Module,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Extract features from a list of 2D slices through a feature network.

    Args:
        slices_list: List of ``(1, 3, H, W)`` tensors (already BGR, normalised).
        feature_network: ResNet-50 feature extractor.
        batch_size: Mini-batch size for forward passes.
        device: Compute device.

    Returns:
        Features of shape ``(N_slices, 2048)``.
    """
    # Stack all slices: (N_slices, 3, H, W)
    all_slices = torch.cat(slices_list, dim=0)
    all_slices = _radimagenet_normalise(all_slices)

    features_list: list[Tensor] = []
    for i in range(0, all_slices.shape[0], batch_size):
        batch = all_slices[i : i + batch_size].to(device)
        feat = feature_network(batch)
        feat = _spatial_average(feat) if feat.ndim > 2 else feat
        features_list.append(feat.cpu())

    return torch.cat(features_list, dim=0)


def extract_2d5_features(
    volume: Tensor,
    feature_network: nn.Module,
    center_slices_ratio: float = 0.6,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract 2.5D features from 3 orthogonal planes of a 3D volume.

    Per plane:
    1. Compute center slice range from ``center_slices_ratio``
    2. Unbind slices along that axis
    3. Replicate 1ch to 3ch, swap channels 0 and 2 (RGB to BGR)
    4. Normalise via ``_radimagenet_normalise``
    5. Forward through feature network in mini-batches
    6. Spatial average to ``(N_slices, 2048)``

    Args:
        volume: Single volume of shape ``(1, 1, H, W, D)``.
        feature_network: RadImageNet feature extractor.
        center_slices_ratio: Fraction of center slices to use per axis.
        batch_size: Mini-batch size for feature extraction.

    Returns:
        Tuple of ``(xy_feats, yz_feats, zx_feats)``, each
        ``(N_slices, feat_dim)``.
    """
    assert volume.ndim == 5 and volume.shape[0] == 1, (
        f"Expected (1, C, H, W, D), got {volume.shape}"
    )

    # Infer device from network parameters (fall back to CPU)
    try:
        device = next(feature_network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Replicate to 3 channels if single-channel
    if volume.shape[1] == 1:
        volume = volume.repeat(1, 3, 1, 1, 1)

    # RGB -> BGR (matching MAISI reference)
    volume = volume[:, [2, 1, 0], ...]

    _, C, H, W, D = volume.shape

    results: list[Tensor] = []
    dims_sizes = [(4, D), (2, H), (3, W)]  # (unbind_dim, size) for XY, YZ, ZX

    for unbind_dim, size in dims_sizes:
        start = int((1.0 - center_slices_ratio) / 2.0 * size)
        end = int((1.0 + center_slices_ratio) / 2.0 * size)

        # Narrow along the axis, then unbind
        narrowed = volume.narrow(unbind_dim, start, end - start)
        slices = list(torch.unbind(narrowed, dim=unbind_dim))

        feats = _extract_plane_features(slices, feature_network, batch_size, device)
        results.append(feats)

    return results[0], results[1], results[2]


def compute_fid_2d5(
    real_features: tuple[Tensor, Tensor, Tensor],
    fake_features: tuple[Tensor, Tensor, Tensor],
) -> dict[str, float]:
    """Compute 2.5D FID (per-plane and average).

    Uses ``monai.metrics.fid.FIDMetric`` for Frechet distance computation,
    matching the MAISI evaluation protocol.

    Args:
        real_features: ``(xy_feats, yz_feats, zx_feats)`` from real data.
        fake_features: ``(xy_feats, yz_feats, zx_feats)`` from generated data.

    Returns:
        Dict with keys ``"fid_xy"``, ``"fid_yz"``, ``"fid_zx"``, ``"fid_avg"``.
    """
    from monai.metrics.fid import FIDMetric

    fid_metric = FIDMetric()
    plane_names = ["fid_xy", "fid_yz", "fid_zx"]

    results: dict[str, float] = {}
    total = 0.0

    for name, real_f, fake_f in zip(plane_names, real_features, fake_features):
        fid_val = fid_metric(fake_f, real_f)
        fid_float = float(fid_val)
        results[name] = fid_float
        total += fid_float

    results["fid_avg"] = total / 3.0
    return results
