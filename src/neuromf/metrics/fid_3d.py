"""3D-FID computation matching the MOTFM evaluation protocol.

Computes volumetric FID using a **torchvision R3D-18** backbone
(Kinetics-400 pretrained, FC head replaced with ``Identity``) as the 3D
feature extractor, producing **512-d** feature vectors per volume.

This matches the protocol released by the MOTFM authors in
``src/external/MOTFM/evaluation_3d/evaluate_3d.py`` (Feb 2026):

- Feature extractor: ``torchvision.models.video.r3d_18`` (Kinetics-400)
- FC head replaced with ``torch.nn.Identity()`` → 512-d output
- Single-channel volumes replicated to 3 channels
- **Per-set** min-max normalisation to [0, 1] (global across all
  volumes in each set, NOT per-volume)
- FID via standard Fréchet distance (MONAI ``FIDMetric``)

.. note::

   Despite being called "3D-FID", this is a *pseudo*-3D metric in the
   sense that the R3D-18 backbone was pretrained on Kinetics-400 video
   clips (natural RGB sequences), not on volumetric medical images.
   Nevertheless, it provides a standardised distributional comparison
   that is directly comparable with MOTFM's published numbers.

   The previous Med3D ResNet-50 implementation was found to produce
   collapsed features (cosine similarity ≈ 1.0 for all inputs),
   rendering FID meaningless. R3D-18 does not suffer from this issue.

Standalone usage::

    from neuromf.metrics.fid_3d import load_fid3d_feature_net, extract_3d_features, compute_fid_3d

    net = load_fid3d_feature_net(device=torch.device("cuda"))

    real_feats = extract_3d_features(real_vols, net)   # (N_real, 512)
    fake_feats = extract_3d_features(fake_vols, net)   # (N_fake, 512)
    fid = compute_fid_3d(real_feats, fake_feats)

Reference:
    Tresckow et al., "MOTFM: Medical OT Flow Matching", 2025.
    evaluation_3d/evaluate_3d.py — ``_load_r3d18_backbone``, ``_extract_r3d_features``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Feature dimension for R3D-18 (after removing FC head)
FID3D_FEATURE_DIM = 512


def load_fid3d_feature_net(
    device: torch.device | str = "cpu",
    weights_path: str | None = None,
) -> nn.Module:
    """Load R3D-18 as a 3D feature extractor (512-d output).

    Uses ``torchvision.models.video.r3d_18`` with Kinetics-400 pretrained
    weights. The final FC layer is replaced with ``Identity()`` to expose
    the 512-d feature representation.

    Args:
        device: Device to load the model onto.
        weights_path: Optional path to a local ``.pth`` file with R3D-18
            weights (for offline compute nodes). If ``None`` or empty,
            torchvision downloads the checkpoint automatically.

    Returns:
        R3D-18 model in eval mode with 512-d output.
    """
    import torchvision
    from pathlib import Path

    if weights_path and Path(weights_path).is_file():
        # Offline mode: build model without pretrained weights, load from file
        model = torchvision.models.video.r3d_18(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded R3D-18 weights from %s", weights_path)
    else:
        model = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT,
        )
        logger.info("Loaded R3D-18 weights via torchvision (Kinetics-400)")

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    logger.info("R3D-18 feature extractor ready (512-d, device=%s)", device)
    return model


# Backward-compatible alias (old API used Med3D weights path argument)
def load_med3d_resnet50(weights_path: str | None = None) -> nn.Module:
    """Backward-compatible alias for :func:`load_fid3d_feature_net`.

    .. deprecated::
        The Med3D ResNet-50 backend has been replaced by R3D-18 to match
        the MOTFM evaluation protocol. The ``weights_path`` argument is
        ignored — R3D-18 uses torchvision's built-in Kinetics-400 weights.

    Args:
        weights_path: Ignored (kept for API compatibility).

    Returns:
        R3D-18 feature extractor (512-d output).
    """
    if weights_path:
        logger.warning(
            "load_med3d_resnet50: weights_path is ignored — using R3D-18 "
            "(Kinetics-400) instead of Med3D ResNet-50 to match MOTFM protocol."
        )
    return load_fid3d_feature_net()


def _to_three_channels(volumes: Tensor) -> Tensor:
    """Replicate single-channel volumes to 3 channels for R3D-18.

    Matches MOTFM ``_to_three_channels()`` — repeats channels if < 3,
    crops if > 3.

    Args:
        volumes: Tensor of shape ``(B, C, D, H, W)``.

    Returns:
        Tensor of shape ``(B, 3, D, H, W)``.
    """
    c = volumes.shape[1]
    if c == 3:
        return volumes
    if c < 3:
        repeats = (3 + c - 1) // c
        return volumes.repeat(1, repeats, 1, 1, 1)[:, :3]
    return volumes[:, :3]


def _per_set_minmax(volumes: Tensor) -> Tensor:
    """Per-set min-max normalisation to [0, 1].

    Normalises across ALL volumes in the set jointly (global min/max),
    matching MOTFM's ``_minmax()`` and their default ``per_set_minmax`` mode.

    Args:
        volumes: Tensor of shape ``(B, C, D, H, W)``.

    Returns:
        Normalised tensor in [0, 1], same shape.
    """
    x = torch.nan_to_num(volumes.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_min = x.amin()
    x_max = x.amax()
    denom = (x_max - x_min).clamp_min(1e-8)
    return (x - x_min) / denom


@torch.no_grad()
def extract_3d_features(
    volumes: Tensor,
    feature_network: nn.Module,
    normalize: bool = True,
    batch_size: int = 8,
) -> Tensor:
    """Extract 3D features from a batch of volumes using R3D-18.

    Follows the MOTFM protocol: per-set min-max normalisation,
    single→3 channel replication, batch-wise feature extraction.

    Args:
        volumes: Volumes of shape ``(B, 1, D, H, W)`` or ``(1, 1, D, H, W)``
            for a single volume. Also accepts ``(B, D, H, W)`` or ``(D, H, W)``.
        feature_network: R3D-18 feature extractor from :func:`load_fid3d_feature_net`.
        normalize: If True, apply per-set min-max normalisation to [0, 1].
        batch_size: Mini-batch size for feature extraction forward passes.

    Returns:
        Feature tensor ``(B, 512)`` on CPU.
    """
    # Handle various input shapes
    if volumes.ndim == 3:
        volumes = volumes.unsqueeze(0).unsqueeze(0)  # (D,H,W) -> (1,1,D,H,W)
    elif volumes.ndim == 4:
        volumes = volumes.unsqueeze(1)  # (B,D,H,W) -> (B,1,D,H,W)
    assert volumes.ndim == 5, f"Expected 5D tensor, got shape {volumes.shape}"

    if normalize:
        volumes = _per_set_minmax(volumes)

    volumes = _to_three_channels(volumes)

    device = next(feature_network.parameters()).device
    feats: list[Tensor] = []
    for i in range(0, volumes.shape[0], batch_size):
        batch = volumes[i : i + batch_size].to(device)
        feats.append(feature_network(batch).detach().cpu())

    return torch.cat(feats, dim=0)


def compute_fid_3d(
    real_features: Tensor,
    fake_features: Tensor,
) -> float:
    """Compute FID between two sets of 3D features.

    Uses MONAI's ``FIDMetric`` for the Fréchet distance computation,
    matching both the 2.5D-FID and the MOTFM 3D-FID paths.

    Args:
        real_features: Real feature vectors ``(N_real, 512)``.
        fake_features: Generated feature vectors ``(N_fake, 512)``.

    Returns:
        FID score (lower is better). Returns 0.0 for identical distributions.
    """
    from monai.metrics.fid import FIDMetric

    fid_metric = FIDMetric()
    fid_val = fid_metric(fake_features, real_features)
    return float(fid_val)
