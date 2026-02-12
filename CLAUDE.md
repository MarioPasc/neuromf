# NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis

## 1. Project Overview

NeuroMF trains a MeanFlow model in the latent space of a frozen MAISI 3D VAE to achieve 1-step (1-NFE) generation of 128^3 brain MRI volumes. The project introduces per-channel Lp loss (extending SLIM-Diff to latent space) and LoRA fine-tuning for joint synthesis of rare epilepsy pathology (FCD). Target venues: Medical Image Analysis, IEEE TMI, or MICCAI 2026.

## 2. Architecture Summary

```
Pipeline: Input MRI (1x128^3) -> Frozen MAISI VAE Encoder -> Latent (4x32^3)
          -> Train MeanFlow in Latent Space (1-NFE) -> Decode -> Synthetic MRI (1x128^3)

Key shapes:
  - Pixel space:  (B, 1, 128, 128, 128)
  - Latent space:  (B, 4, 32, 32, 32)
  - Compression: 16x overall (spatial 4x per axis, 1->4 channels)

MeanFlow core idea:
  - Learn average velocity u(z_t, r, t) instead of instantaneous velocity v(z_t, t)
  - MeanFlow Identity enforces self-consistency via JVP (Eq. 8)
  - 1-NFE sampling: z_0 = eps - u_theta(eps, 0, 1)
  - Training uses iMF combined loss (Eq. 13) with adaptive weighting (Eq. 14)
  - x-prediction reparameterisation (Eqs. 15-16) for manifold structure
```

## 3. Repository Map

| Directory | Purpose |
|---|---|
| `src/neuromf/` | Core Python package: wrappers, models, data, losses, sampling, metrics, utils, errors |
| `src/external/` | **READ-ONLY** vendored repos: `MeanFlow/`, `MeanFlow-PyTorch/`, `NV-Generate-CTMR/`, `MOTFM/`, `pmf/` |
| `configs/` | OmegaConf YAML configurations for all experiments |
| `experiments/cli/` | CLI entry points for each phase |
| `experiments/` | Experiment directories with results, figures, verification reports |
| `tests/` | pytest test files, one per phase group |
| `docs/main/` | **READ-ONLY** master documents: `technical_guide.md`, `methodology_expanded.md` |
| `docs/splits/` | Phase split documents (one per phase, self-contained for subagents) |
| `docs/papers/` | Paper PDFs and insight documents from `/review-external` |
| `docs/misc/` | Miscellaneous documentation |
| `.claude/` | Agent configs: settings, agents, commands, skills, hooks |

## 4. Coding Standards

- **Type hints** on ALL function signatures and return types.
- **Google-style docstrings** on all public functions and classes. No usage examples needed.
- **Brief inline comments** on non-obvious code only. Do not comment obvious lines.
- **Logging:** Python `logging` module with `rich` handler. INFO for training events, DEBUG for shapes/values.
- **No magic numbers.** All hyperparameters from YAML configs via OmegaConf/Hydra.
- **Prefer library functions.** MONAI transforms over custom preprocessing. `einops.rearrange` over manual reshapes. `F.scaled_dot_product_attention` over manual QKV matmuls.
- **Tests use pytest.** Each test file runnable independently: `pytest tests/test_xxx.py -v`.
- **Keep functions atomic.** One conceptual task per function.
- **OOP with dataclasses** for configuration containers.
- **Custom exceptions** in `src/neuromf/errors/` for domain-specific errors.
- **OmegaConf** for hierarchical YAML config management.
- **Scientific claims** must reference sources (article titles, not just "see paper").
- **Test IDs** from phase splits: name tests `test_P{N}_T{M}_<description>`.
- **Leverage reference codebases.** Start from PyTorch MeanFlow reference, do not reimplement tested patterns.

## 5. Key Paths

| Resource | Path |
|---|---|
| Project root | `/home/mpascual/research/code/neuromf/` |
| Conda environment | `~/.conda/envs/neuromf/` (Python 3.11, PyTorch >=2.1, MONAI >=1.3) |
| External: MeanFlow (JAX) | `src/external/MeanFlow/` |
| External: MeanFlow (PyTorch) | `src/external/MeanFlow-PyTorch/` |
| External: MAISI / NV-Generate | `src/external/NV-Generate-CTMR/` |
| External: MOTFM | `src/external/MOTFM/` |
| External: pMF | `src/external/pmf/` |
| Configs | `configs/` |
| Results output | `experiments/` |
| Checkpoints | Configure via `configs/base.yaml` `checkpoint_dir` field |
| Datasets | Configure via `configs/base.yaml` `data.dataset_root` field |

## 6. Phase System

The project is implemented in 9 gated phases (Phase 0 through Phase 8). Each phase has a self-contained split document in `docs/splits/phase_{N}.md`.

**Before starting any phase:**
1. Read `CLAUDE.md` (this file) for project context.
2. Read the phase split document `docs/splits/phase_{N}.md`.
3. Read any insight documents referenced in the split (in `docs/papers/*/insights.md`).

**Gating rule:** Phase N+1 cannot start until Phase N's verification tests all pass. Use `/check-gate N` to verify.

| Phase | Title | Key Output |
|---|---|---|
| 0 | Environment Bootstrap and VAE Validation | `maisi_vae.py` wrapper, VAE reconstruction metrics |
| 1 | Latent Pre-computation Pipeline | `.pt` latent files, per-channel stats |
| 2 | Toy Experiment — MeanFlow on Toroidal Manifold | Validated MeanFlow pipeline on known manifold |
| 3 | MeanFlow Loss Integration with 3D UNet | JVP-compatible UNet wrapper, MeanFlow loss |
| 4 | Training on Brain MRI Latents | Trained MeanFlow model, EMA checkpoints |
| 5 | Evaluation Suite | FID, SSIM, SynthSeg metrics |
| 6 | Ablation Runs | x-pred vs u-pred, Lp sweep, NFE steps, RF baseline |
| 7 | LoRA Fine-Tuning for FCD | Joint image-mask synthesis model |
| 8 | Paper Figures and Tables | Publication-ready figures (PDF+PNG) |

## 7. Testing Conventions

- Tests live in `tests/`.
- Framework: `pytest` via `~/.conda/envs/neuromf/bin/python -m pytest`.
- Naming: `test_P{N}_T{M}_<description>` where N is phase number, M is test number.
- Run phase tests: `pytest tests/ -v -k "P{N}"`.
- Run all tests: `pytest tests/ -v --tb=short`.
- Each verification test is tagged CRITICAL (blocks gate) or INFORMATIONAL.
- A phase gate is OPEN when all CRITICAL tests pass.

## 8. Forbidden Actions

- **DO NOT** modify anything in `src/external/`. These are frozen vendored repos.
- **DO NOT** delete or overwrite `docs/main/`. These are master reference documents.
- **DO NOT** retrain or fine-tune the MAISI VAE. It is a frozen foundation model.
- **DO NOT** use `diffusers` — it is 2D-only and incompatible with our 3D pipeline.
- **DO NOT** use `torchcfm` — time convention mismatch (t=0 noise vs our t=0 data).
- **DO NOT** run `rm -rf` or force-push to git.
