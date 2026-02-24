"""Download Med3D ResNet-50 weights from HuggingFace.

Usage:
    python scripts/download_med3d_weights.py --output /path/to/fid_weights/
    python scripts/download_med3d_weights.py --output /path/to/fid_weights/ --token hf_xxxxx

Downloads ``resnet_50_23dataset.pth`` (~178 MB) from the TencentMedicalNet
HuggingFace repository. The file contains weights for a 3D ResNet-50 pretrained
on 23 medical segmentation datasets (Chen et al., 2019).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "TencentMedicalNet/MedicalNet-Resnet50"
FILENAME = "resnet_50_23dataset.pth"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Med3D ResNet-50 weights from HuggingFace."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the downloaded weights.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated/private model access.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID}).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=FILENAME,
        help=f"Filename within the repo (default: {FILENAME}).",
    )
    return parser.parse_args()


def main() -> None:
    """Download Med3D weights."""
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename

    if output_path.exists():
        logger.info("Weights already exist at %s — skipping download", output_path)
        return

    logger.info("Downloading %s from %s ...", args.filename, args.repo_id)

    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(output_dir),
        token=args.token,
    )

    logger.info("Downloaded to %s", downloaded_path)

    # Verify the file exists at expected location
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Verified: %s (%.1f MB)", output_path, size_mb)
    else:
        logger.info("File saved at: %s", downloaded_path)


if __name__ == "__main__":
    main()
