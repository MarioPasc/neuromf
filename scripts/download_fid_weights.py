"""Download RadImageNet ResNet-50 weights for offline FID computation.

Fetches the pretrained model via torch.hub and saves the state dict
to a local file for use on compute nodes without internet access.

Usage:
    python scripts/download_fid_weights.py \
        --output /path/to/checkpoints/fid_weights/radimagenet_resnet50.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Download and save RadImageNet ResNet-50 state dict."""
    parser = argparse.ArgumentParser(
        description="Download RadImageNet ResNet-50 weights for offline FID."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the state dict .pt file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading radimagenet_resnet50 via torch.hub...")
    model = torch.hub.load(
        "Warvito/radimagenet-models",
        model="radimagenet_resnet50",
        verbose=True,
        trust_repo=True,
    )

    state_dict = model.state_dict()
    torch.save(state_dict, str(output_path))

    n_keys = len(state_dict)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved: %s (%d keys, %.1f MB)", output_path, n_keys, size_mb)
    logger.info("Verification: load with load_radimagenet_resnet50('%s')", output_path)


if __name__ == "__main__":
    main()
