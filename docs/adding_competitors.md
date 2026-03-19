# Adding a New Competitor Model to NeuroMF

This guide explains how to add a new pixel-space competitor model (e.g., ScoreSDE, EDM, Consistency Models) to the NeuroMF evaluation pipeline. The architecture is designed so that adding a competitor requires **only implementing 2 abstract methods** — everything else (generation loop, HDF5 I/O, SLURM infrastructure, evaluation, comparison figures, tables) is handled by the framework.

## Prerequisites

- The competitor model must generate 3D brain MRI volumes directly in pixel space (shape `(1, S, S, S)`, typically `S=192`).
- You have a trained checkpoint and a config file for the model.
- Any vendored code the model depends on is in `src/external/` (read-only).

## Step 1: Create the Generator Class

Create a new file in `src/neuromf/competitors/`, e.g. `src/neuromf/competitors/scoresde_gen.py`:

```python
"""ScoreSDE competitor generator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from neuromf.competitors.base import BaseCompetitorGenerator
from neuromf.competitors.registry import register_competitor


def _ensure_deps_on_path() -> None:
    """Add vendored ScoreSDE code to sys.path."""
    repo_root = Path(__file__).resolve().parents[3]
    scoresde_dir = str(repo_root / "src" / "external" / "ScoreSDE")
    if scoresde_dir not in sys.path:
        sys.path.insert(0, scoresde_dir)


@register_competitor("scoresde")
class ScoreSDECompetitorGenerator(BaseCompetitorGenerator):
    """Generate volumes using a trained ScoreSDE model."""

    model_name: str = "ScoreSDE"

    def _load_model(self, config_path: Path, checkpoint_path: Path) -> None:
        """Load the ScoreSDE model.

        Must populate:
          - self.config: dict with at least data_args.volume_size and
            data_args.image_norm (for data range detection)
          - self.metadata: dict with epoch/global_step (optional)
          - Any model-specific attributes needed by _generate_batch()
        """
        _ensure_deps_on_path()

        # Load your config
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load your model
        from scoresde import ScoreModel
        self._model = ScoreModel.load_from_checkpoint(str(checkpoint_path))
        self._model.to(self.device)
        self._model.eval()

        self.metadata = {"source": str(checkpoint_path)}

    def _generate_batch(self, x_init: Tensor, nfe: int) -> Tensor:
        """Run ScoreSDE reverse SDE sampling on one batch.

        Args:
            x_init: Noise tensor (B, C, D, H, W) on self.device.
            nfe: Number of function evaluations.

        Returns:
            Generated volumes (B, C, D, H, W) on self.device.
            (Channel dim will be squeezed by the base class.)
        """
        return self._model.sample(x_init, n_steps=nfe)
```

### What you must implement:

| Method | Purpose | Populates |
|--------|---------|-----------|
| `_load_model(config_path, checkpoint_path)` | Load checkpoint, set up model | `self.config`, `self.metadata`, model attrs |
| `_generate_batch(x_init, nfe)` | Run ONE batch through the sampler | Returns `(B, C, D, H, W)` tensor |

### What the base class handles for you:
- Noise creation with deterministic per-sample seeding
- Batched generation loop with progress logging
- Channel dim squeeze (1→0 for single-channel MRI)
- Clamp + normalize to [0, 1] based on `self.config["data_args"]["image_norm"]`
- Volume size resolution (CLI override > config > default 192)

## Step 2: Register the Module

Add a single import line to `src/neuromf/competitors/__init__.py`:

```python
import neuromf.competitors.scoresde_gen  # noqa: F401  ← ADD THIS
```

That's it — the `@register_competitor("scoresde")` decorator handles registration.

## Step 3 (Optional): Register a Visual Style

In `src/neuromf/competitors/styles.py`, add an entry to `DEFAULT_METHOD_STYLES`:

```python
DEFAULT_METHOD_STYLES["ScoreSDE"] = {
    "color": "#228833",  # Paul Tol green
    "marker": "D",
    "ls": "-.",
}
```

If you skip this, the system auto-assigns a color from the fallback palette.

## Step 4 (Optional): Add Paper Baselines

In `experiments/analysis/data_loader.py`, add to `COMPETITOR_BASELINES`:

```python
COMPETITOR_BASELINES["ScoreSDE"] = {
    1:  {"fid_3d": 45.0, "ms_ssim": 0.55, "mmd": 1.2},
    10: {"fid_3d": 12.0, "ms_ssim": 0.72, "mmd": 0.30},
    50: {"fid_3d": 8.5,  "ms_ssim": 0.75, "mmd": 0.20},
}
```

These are only used by the single-model analysis (`run_analysis.py`). The multi-model comparison (`run_comparison.py`) uses actual computed metrics.

## Step 5: Generate Volumes

```bash
# Unified CLI — no per-model script needed:
python experiments/cli/generate_competitor.py \
    --model scoresde \
    --config configs/picasso/scoresde/my_config.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --nfe 1 10 50 \
    --n-samples 500 \
    --output-dir /path/to/ScoreSDE/generation/volumes/

# Or via SLURM:
bash slurm/generate_competitor/launch.sh \
    --model scoresde \
    --config configs/picasso/scoresde/my_config.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --run-dir /path/to/ScoreSDE/
```

## Step 6: Evaluate and Compare

The evaluation and comparison pipelines are already model-agnostic:

```bash
# Evaluate (works for any model):
bash slurm/evaluate/launch.sh --run-dir /path/to/ScoreSDE/

# Compare (just add --competitor):
python experiments/cli/run_comparison.py \
    --neuroimf-dir /path/to/NeuroiMF/ \
    --competitor MOTFM:/path/to/MOTFM/ \
    --competitor DDPM:/path/to/DDPM/ \
    --competitor ScoreSDE:/path/to/ScoreSDE/ \
    --output-dir /path/to/comparison/
```

## Step 7: Full SLURM Orchestration

```bash
bash slurm/orchestrate_evaluation/launch.sh \
    --neuroimf-checkpoint /path/to/neuromf.ckpt \
    --competitor motfm:/path/to/motfm.ckpt \
    --competitor ddpm:/path/to/ddpm.ckpt \
    --competitor scoresde:/path/to/scoresde.ckpt \
    --output-root /path/to/comparison_20260320/
```

## Summary: What Changes, What Doesn't

| Task | Files to change |
|------|----------------|
| **Core generator** | `src/neuromf/competitors/NEW_gen.py` (new) |
| **Registration** | `src/neuromf/competitors/__init__.py` (1 import line) |
| **Visual style** | `src/neuromf/competitors/styles.py` (optional, 3 lines) |
| **Paper baselines** | `experiments/analysis/data_loader.py` (optional, 4 lines) |

**Nothing else changes.** The generation CLI, SLURM scripts, evaluation pipeline, comparison figures, tables, and statistical tests all work automatically with any registered competitor.

## Architecture Overview

```
src/neuromf/competitors/
├── __init__.py          # Auto-imports all generators
├── base.py              # BaseCompetitorGenerator (ABC)
├── registry.py          # CompetitorRegistry + @register_competitor
├── io.py                # write_volumes_h5() (shared HDF5 writer)
├── styles.py            # Visual style registry (colors/markers)
├── motfm_gen.py         # MOTFM implementation
├── ddpm_gen.py          # DDPM implementation
└── scoresde_gen.py      # ← Your new competitor
```
