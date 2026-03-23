#!/usr/bin/env python3
"""Find the best FID checkpoint from a training run.

Scans the most recent run directory for FID-based checkpoints and returns
the path to the one with the lowest FID score. If no FID checkpoints exist,
falls back to the best raw loss checkpoint, then to last.ckpt.

Usage:
    python slurm/train_v2/find_best_checkpoint.py /path/to/results

Output (stdout):
    /absolute/path/to/best_checkpoint.ckpt

Exit codes:
    0: Found checkpoint
    1: No checkpoints found
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_best_fid_checkpoint(results_root: Path) -> Path | None:
    """Find best FID checkpoint from the most recent run.

    Args:
        results_root: Results root directory containing runs/.

    Returns:
        Path to best checkpoint, or None if not found.
    """
    runs_dir = results_root / "runs"
    if not runs_dir.exists():
        return None

    # Find most recent run directory
    run_dirs = sorted(runs_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    if not run_dirs:
        return None

    ckpt_dir = run_dirs[0] / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Strategy 1: Best FID checkpoint (filename: best_fid_NNN_val/fid_3d=XX.XX.ckpt)
    fid_ckpts = list(ckpt_dir.glob("best_fid_*.ckpt"))
    if fid_ckpts:
        # Parse FID value from filename and find lowest
        best_ckpt = None
        best_fid = float("inf")
        for ckpt in fid_ckpts:
            # Extract FID value from filename patterns like:
            # best_fid_epoch=123_val/fid_3d=7.34.ckpt
            # best_fid_123_val/fid_3d=7.34.ckpt
            match = re.search(r"fid[_3d]*=(\d+\.?\d*)", ckpt.name)
            if match:
                fid_val = float(match.group(1))
                if fid_val < best_fid:
                    best_fid = fid_val
                    best_ckpt = ckpt
        if best_ckpt is not None:
            return best_ckpt
        # If parsing failed, return most recent FID checkpoint
        return max(fid_ckpts, key=lambda p: p.stat().st_mtime)

    # Strategy 2: Best raw loss checkpoint
    raw_ckpts = list(ckpt_dir.glob("best_raw_*.ckpt"))
    if raw_ckpts:
        return max(raw_ckpts, key=lambda p: p.stat().st_mtime)

    # Strategy 3: Last checkpoint
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    # Strategy 4: Any checkpoint
    any_ckpts = list(ckpt_dir.glob("*.ckpt"))
    if any_ckpts:
        return max(any_ckpts, key=lambda p: p.stat().st_mtime)

    return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python find_best_checkpoint.py <results_root>", file=sys.stderr)
        sys.exit(1)

    results_root = Path(sys.argv[1])
    ckpt = find_best_fid_checkpoint(results_root)

    if ckpt is None:
        print("ERROR: No checkpoint found", file=sys.stderr)
        sys.exit(1)

    # Print absolute path to stdout (for capture by shell scripts)
    print(str(ckpt.resolve()))


if __name__ == "__main__":
    main()
