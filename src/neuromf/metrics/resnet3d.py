"""3D ResNet-50 architecture matching Tencent/MedicalNet (Med3D).

Self-contained implementation — no external dependency on MedicalNet repo.
Matches ``models/resnet.py`` from https://github.com/Tencent/MedicalNet exactly:

- Conv1: ``Conv3d(1, 64, 7, stride=2, padding=3)`` + BN + ReLU + MaxPool
- Layer1: 3x Bottleneck(64→256), stride=1, dilation=1
- Layer2: 4x Bottleneck(128→512), stride=2, dilation=1
- Layer3: 6x Bottleneck(256→1024), stride=1, dilation=2 (dilated, no downsample)
- Layer4: 3x Bottleneck(512→2048), stride=1, dilation=4 (dilated, no downsample)

For FID: the segmentation head (``conv_seg``) is replaced with
``AdaptiveAvgPool3d(1,1,1) + Flatten()`` → 2048-d feature vector.

Reference:
    Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis",
    arXiv:1904.00625, 2019.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ResNet3DBottleneck(nn.Module):
    """3D Bottleneck block with expansion=4 and shortcut type B.

    Shortcut type B uses a 1x1x1 Conv3d + BN for dimension matching
    whenever input and output dimensions differ (stride != 1 or
    channel mismatch).

    Args:
        inplanes: Number of input channels.
        planes: Number of intermediate channels (output = planes * 4).
        stride: Stride for the 3x3x3 convolution.
        dilation: Dilation for the 3x3x3 convolution.
        downsample: Optional projection shortcut module.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        # Dilation replaces stride for spatial dimensions when dilation > 1
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet for volumetric medical image feature extraction.

    Matches the MedicalNet architecture with dilated layers 3-4
    (segmentation-style, preserves spatial resolution after layer2).

    For FID evaluation, uses ``AdaptiveAvgPool3d`` + ``Flatten`` as
    the output head, producing a 2048-d feature vector per volume.

    Args:
        layers: Number of blocks per layer (e.g. ``[3, 4, 6, 3]`` for ResNet-50).
        shortcut_type: Shortcut strategy. Only ``"B"`` (projection) is supported.
        in_channels: Number of input channels (1 for single-channel MRI).
    """

    def __init__(
        self,
        layers: list[int],
        shortcut_type: str = "B",
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if shortcut_type != "B":
            raise ValueError(f"Only shortcut_type='B' is supported, got '{shortcut_type}'")

        self.inplanes = 64

        # Stem: conv1 + bn + relu + maxpool
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Layers with MedicalNet dilation pattern: [1, 1, 2, 4]
        self.layer1 = self._make_layer(64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4)

        # Feature head (replaces conv_seg segmentation head)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        """Build a residual layer with ``blocks`` bottleneck blocks.

        Args:
            planes: Intermediate channel count.
            blocks: Number of bottleneck blocks.
            stride: Stride for the first block (downsamples if > 1).
            dilation: Dilation for all blocks in this layer.

        Returns:
            Sequential module of bottleneck blocks.
        """
        downsample = None
        outplanes = planes * ResNet3DBottleneck.expansion

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    outplanes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(outplanes),
            )

        layers = [ResNet3DBottleneck(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(ResNet3DBottleneck(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming initialization matching MedicalNet."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Extract 2048-d feature vector from a 3D volume.

        Args:
            x: Input volume ``(B, C, D, H, W)`` where ``C=1``.

        Returns:
            Feature vector ``(B, 2048)``.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


def resnet3d_50(shortcut_type: str = "B", in_channels: int = 1) -> ResNet3D:
    """Construct a 3D ResNet-50 (layers = [3, 4, 6, 3]).

    Args:
        shortcut_type: Shortcut strategy (only ``"B"`` supported).
        in_channels: Number of input channels.

    Returns:
        ResNet3D model with 2048-d output.
    """
    return ResNet3D([3, 4, 6, 3], shortcut_type=shortcut_type, in_channels=in_channels)


def load_med3d_weights(
    model: ResNet3D,
    weights_path: str | Path,
) -> ResNet3D:
    """Load Med3D pretrained weights into a ResNet3D model.

    Handles Med3D state dict conventions:
    - Strips ``module.`` prefix (from DataParallel training)
    - Ignores ``conv_seg.*`` keys (segmentation head we don't use)

    Args:
        model: ResNet3D model to load weights into.
        weights_path: Path to ``resnet_50_23dataset.pth``.

    Returns:
        Model with loaded weights.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Med3D weights not found: {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    # Med3D checkpoints may be wrapped in a "state_dict" key
    if "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    else:
        raw_state = checkpoint

    # Strip "module." prefix and filter out conv_seg keys
    cleaned: OrderedDict[str, Tensor] = OrderedDict()
    skipped: list[str] = []
    for key, value in raw_state.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("conv_seg"):
            skipped.append(clean_key)
            continue
        cleaned[clean_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    n_loaded = len(cleaned) - len(unexpected)
    logger.info(
        "Loaded Med3D weights: %d params loaded, %d conv_seg skipped, %d missing, %d unexpected",
        n_loaded,
        len(skipped),
        len(missing),
        len(unexpected),
    )

    if missing:
        # avgpool/flatten have no parameters, so missing is expected to be empty
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    return model
