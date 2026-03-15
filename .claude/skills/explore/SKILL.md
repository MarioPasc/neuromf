---
name: explore
description: Deep codebase exploration in isolated context
context: fork
agent: Explore
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - WebFetch
  - WebSearch
---

# NeuroMF Codebase Explorer

Thoroughly explore the neuromf codebase to answer: $ARGUMENTS

## Project Context

NeuroMF trains iMF (improved MeanFlow) with dual-head architecture in the latent space of a frozen MAISI 3D VAE for 1-NFE brain MRI synthesis. Key architecture:

- **x-prediction mode**: model outputs denoised x_hat, converted to u = (z_t - x_hat) / t
- **Exact JVP** via `torch.func.jvp` with `has_aux=True` for compound velocity V = u + (t-r)*sg[du/dt]
- **v-head**: supervised tangent predictor for stable JVP (disabled at inference)
- **FORBIDDEN combo**: x-prediction + finite_difference JVP (1/t singularity explosion)
- **Safe combos**: x + exact, u + FD, u + exact
- **Latent shape**: (B, 4, 48, 48, 48), pixel shape: (B, 1, 192, 192, 192)
- **Scale factor**: 0.96240234375 (from diffusion checkpoint, NOT VAE)

## Key Source Locations

| Area | Path | Key Files |
|------|------|-----------|
| Core package | `src/neuromf/` | `__init__.py` |
| MeanFlow loss | `src/neuromf/wrappers/meanflow_loss.py` | `MeanFlowPipeline`, `MeanFlowPipelineConfig`, compound velocity, JVP |
| JVP strategies | `src/neuromf/wrappers/jvp_strategies.py` | `ExactJVP`, `FiniteDifferenceJVP`, Protocol pattern |
| UNet wrapper | `src/neuromf/wrappers/maisi_unet.py` | `MAISIUNetWrapper`, dual-head (u+v), conditioning modes |
| VAE wrapper | `src/neuromf/wrappers/maisi_vae.py` | `MAISIVAEWrapper`, encode/decode with scale_factor |
| Lightning module | `src/neuromf/models/latent_meanflow.py` | `LatentMeanFlow`, training_step, EMA, callbacks |
| Config loader | `src/neuromf/config/loader.py` | `load_merged_config()` (shared across all CLI scripts) |
| Losses | `src/neuromf/losses/` | `lp_loss.py`, `smoothness_loss.py` |
| Sampling | `src/neuromf/sampling/` | `one_step.py` (1-NFE), `multi_step.py` (Euler) |
| Metrics | `src/neuromf/metrics/` | `fid_3d.py` (R3D-18), `fid.py` (2.5D RadImageNet), `mmd.py`, `ms_ssim_3d.py` |
| Callbacks | `src/neuromf/callbacks/` | `evaluation.py` (FID), `diagnostics.py`, `sample_collector.py` |
| Time sampling | `src/neuromf/utils/time_sampler.py` | `sample_t_and_r()`, logit-normal, boundary sampling |
| EMA | `src/neuromf/utils/ema.py` | `EMAModel`, `MultiEMAModel` |
| CLI scripts | `experiments/cli/` | `train.py`, `generate_latents.py`, `compute_metrics.py`, `decode_samples.py` |
| External refs | `src/external/` | MeanFlow (JAX), MeanFlow-PyTorch, pmf, MOTFM, NV-Generate-CTMR |
| Configs | `configs/` | `base.yaml`, `train_meanflow.yaml`, `picasso/*.yaml` |
| Tests | `tests/` | 292 tests, markers: `slow`, `phase0-6`, `critical`, `informational` |
| Phase specs | `docs/splits/` | `phase_0.md` through `phase_8.md` |
| Technical report | `docs/technical_report/` | `technical_report.tex` (hub for section includes) |

## Current Results (v1 Best Model)

| NFE | FID-3D | MOTFM | Gap |
|-----|--------|-------|-----|
| 1   | 73.85  | 32.10 | +41.75 (we lose) |
| 10  | 7.34   | 9.27  | -1.93 (we win) |
| 50  | 6.14   | 7.93  | -1.79 (we win) |

**Known bottleneck:** 1-NFE variance under-estimation (std ~0.75 vs ~1.0 real).

## Exploration Guidelines

- When searching for how something works, start from the config key name and trace it through the code
- External repos in `src/external/` are READ-ONLY — check `docs/papers/*/code_exploration.md` for pre-computed analysis
- Time convention: t=0 is DATA, t=1 is NOISE (opposite of some papers)
- Config merge chain: `base.yaml -> train_meanflow.yaml -> [generate.yaml] -> user overlays`
