# NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis

## 1. What This Project Does

NeuroMF trains a **MeanFlow** model in the latent space of a **frozen MAISI 3D VAE** to achieve **1-step (1-NFE) generation** of 128^3 brain MRI volumes. The project introduces per-channel Lp loss (extending SLIM-Diff to latent space) and LoRA fine-tuning for joint synthesis of rare epilepsy pathology (FCD). Target venues: Medical Image Analysis, IEEE TMI, or MICCAI 2026.

### Core Pipeline

```
Input MRI (1×128³) ─► Frozen MAISI VAE Encoder ─► Latent (4×32³)
                                                      │
                                              Train MeanFlow (1-NFE)
                                                      │
                                                      ▼
Synthetic MRI (1×128³) ◄── Frozen MAISI VAE Decoder ◄─┘
```

### Key Shapes

| Space | Shape | Notes |
|-------|-------|-------|
| Pixel | `(B, 1, 128, 128, 128)` | Single-channel MRI |
| Latent | `(B, 4, 32, 32, 32)` | 4x spatial compression per axis, 1→4 channels |

### MeanFlow in One Paragraph

MeanFlow learns the **average velocity** `u(z_t, t, r)` instead of the instantaneous velocity `v(z_t, t)`. The MeanFlow Identity enforces self-consistency via a JVP (Jacobian-vector product). At inference, a single forward pass produces a sample: `z_0 = eps - u_θ(eps, 0, 1)`. Training uses the iMF combined loss with adaptive weighting. The pMF extension adds x-prediction reparameterization and perceptual auxiliary losses.

---

## 2. Critical Constants

These are verified values from checkpoint and dataset inspection. Use them directly.

| Constant | Value | Source |
|----------|-------|--------|
| **scale_factor** | **0.96240234375** | Extracted from `diff_unet_3d_rflow-mr.pt["scale_factor"]` |
| VAE latent channels | 4 | `config_network_rflow.json` |
| VAE spatial compression | 4× per axis | 3 encoder levels, `num_channels=[64,128,256]` |
| VAE total parameters | 20,944,897 (~21M) | 130 state dict entries |
| VAE attention | **None** | All `attention_levels=false`, no nonlocal attention |
| VAE memory splits | `num_splits=4, dim_split=1` | Enables 128³ on 8GB VRAM |
| VAE checkpoint format | Wrapped in `"unet_state_dict"` key | Must unwrap before `load_state_dict` |
| IXI dataset size | 581 T1-weighted volumes | Sites: Guys, HH, IOP |
| IXI native shape | `(256, 256, 150)` | Spacing: `(0.94, 0.94, 1.2)` mm |
| Hardware | RTX 4060 Laptop, 8GB VRAM | `max_batch_size_vae=1` for 128³ |

---

## 3. Forbidden Actions

- **DO NOT** modify anything in `src/external/`. These are frozen vendored repos.
- **DO NOT** delete or overwrite `docs/main/`. These are master reference documents.
- **DO NOT** retrain or fine-tune the MAISI VAE. It is a frozen foundation model.
- **DO NOT** use `diffusers` — it is 2D-only and incompatible with our 3D pipeline.
- **DO NOT** use `torchcfm` — time convention mismatch (t=0 noise vs our t=0 data).
- **DO NOT** run `rm -rf` or force-push to git.

---

## 4. Phase System

The project is implemented in **9 gated phases** (Phase 0 through Phase 8). **Phase N+1 cannot start until Phase N's CRITICAL tests all pass.** Use `/check-gate N` to verify.

| Phase | Title | Key Output |
|-------|-------|------------|
| 0 | Environment Bootstrap & VAE Validation | `maisi_vae.py` wrapper, reconstruction metrics |
| 1 | Latent Pre-computation Pipeline | `.pt` latent files, per-channel stats |
| 2 | Toy Experiment — MeanFlow on Toroid | Validated MeanFlow on known manifold |
| 3 | MeanFlow Loss + 3D UNet | JVP-compatible wrapper, MeanFlow loss |
| 4 | Training on Brain MRI Latents | Trained model, EMA checkpoints |
| 5 | Evaluation Suite | FID, SSIM, SynthSeg metrics |
| 6 | Ablation Runs | x-pred vs u-pred, Lp sweep, NFE steps |
| 7 | LoRA Fine-Tuning for FCD | Joint image-mask synthesis |
| 8 | Paper Figures and Tables | Publication-ready figures (PDF+PNG) |

**Before starting any phase**, read its split document at the path below.

---

## 5. Resource Hub

All paths are absolute. The agent environment is `~/.conda/envs/neuromf/` (Python 3.11.14, PyTorch 2.10, MONAI 1.5.2).

### 5.1 Environment & Execution

| Resource | Path / Command |
|----------|---------------|
| Conda Python | `/home/mpascual/.conda/envs/neuromf/bin/python` |
| Run pytest | `~/.conda/envs/neuromf/bin/python -m pytest tests/ -v --tb=short` |
| Run phase tests | `~/.conda/envs/neuromf/bin/python -m pytest tests/ -v -k "P{N}"` |
| Verify paths | `~/.conda/envs/neuromf/bin/python /home/mpascual/research/code/neuromf/scripts/verify_paths.py` |
| Check environment | `~/.conda/envs/neuromf/bin/python /home/mpascual/research/code/neuromf/scripts/check_env.py` |
| Activate script | `source /home/mpascual/research/code/neuromf/scripts/activate.sh` |
| Base config (all paths) | `/home/mpascual/research/code/neuromf/configs/base.yaml` |

### 5.2 Data & Checkpoints (External Drive)

| Resource | Path |
|----------|------|
| IXI T1 volumes (581 files) | `/media/mpascual/Sandisk2TB/research/neuromf/datasets/IXI/IXI-T1/` |
| MAISI VAE weights (80MB) | `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI diffusion weights (2.1GB) | `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt` |
| Results root | `/media/mpascual/Sandisk2TB/research/neuromf/results/` |
| Latent cache | `/media/mpascual/Sandisk2TB/research/neuromf/results/latents/` |
| Training checkpoints | `/media/mpascual/Sandisk2TB/research/neuromf/results/training_checkpoints/` |

### 5.3 Code

| Resource | Path |
|----------|------|
| Project root | `/home/mpascual/research/code/neuromf/` |
| Core package | `/home/mpascual/research/code/neuromf/src/neuromf/` |
| Tests | `/home/mpascual/research/code/neuromf/tests/` |
| Test fixtures | `/home/mpascual/research/code/neuromf/tests/conftest.py` |
| Configs | `/home/mpascual/research/code/neuromf/configs/` |
| Experiments/CLI | `/home/mpascual/research/code/neuromf/experiments/cli/` |

### 5.4 External Vendored Repos (READ-ONLY)

| Repo | Path | What it contains |
|------|------|-----------------|
| MeanFlow (JAX) | `src/external/MeanFlow/` | Original JAX reference: JVP loss, t/r sampling, 1-NFE |
| MeanFlow (PyTorch) | `src/external/MeanFlow-PyTorch/` | PyTorch port: `torch.func.jvp`, SiT architecture |
| NV-Generate-CTMR | `src/external/NV-Generate-CTMR/` | MAISI VAE, preprocessing, 2.5D FID evaluation |
| MOTFM | `src/external/MOTFM/` | Medical OT flow matching: trainer, inferer, UNet wrapper |
| pMF | `src/external/pmf/` | Progressive MeanFlow: x-prediction, compound V, perceptual losses |

### 5.5 Slash Commands

| Command | Usage | What it does |
|---------|-------|-------------|
| `/implement-phase` | `/implement-phase 3` | Launches phase-implementer (Opus) for end-to-end phase work |
| `/run-tests` | `/run-tests 2` | Launches test-runner (Haiku) for phase verification |
| `/check-gate` | `/check-gate 1` | Reads verification report, reports OPEN/BLOCKED |
| `/review-external` | `/review-external meanflow_2025 MeanFlow` | Launches code-reviewer (Sonnet) to produce insights doc |

### 5.6 Subagents

| Agent | Model | Purpose |
|-------|-------|---------|
| `phase-implementer` | Opus | Reads phase split, writes code + tests, runs verification |
| `test-runner` | Haiku | Runs pytest, reports pass/fail |
| `external-code-reviewer` | Sonnet | Reviews external code against paper, produces insights |
| `paper-figure-generator` | Sonnet | Generates publication figures from experiment results |

---

## 6. Documentation Index

> **IMPORTANT — Selective Reading:** The documents below range from short summaries to 1000+ line technical guides. **Do NOT read them all at once.** Before starting a task, scan the table below and pick only the 1-3 documents directly relevant to your current work. Reading everything will waste context window.

### 6.1 Master References (READ-ONLY, large files)

These are comprehensive documents. Read only the section(s) you need, not the full file.

| Document | Path | Contents | When to read |
|----------|------|----------|-------------|
| Technical Guide | `/home/mpascual/research/code/neuromf/docs/main/technical_guide.md` | Step-by-step implementation guide for all 9 phases, repo layout, agent context spec | When you need implementation details for a specific phase beyond what the split provides |
| Methodology | `/home/mpascual/research/code/neuromf/docs/main/methodology_expanded.md` | Theoretical foundations, formal derivations, Lp loss theory, x-pred vs u-pred analysis, ablation design, evaluation protocol, data strategy | When you need mathematical grounding, paper-level methodology, or ablation design rationale |

### 6.2 Phase Split Documents (one per phase, self-contained)

Read the split for the phase you are working on. Each split is self-contained with all context a subagent needs.

| Phase | Path |
|-------|------|
| 0 | `/home/mpascual/research/code/neuromf/docs/splits/phase_0.md` |
| 1 | `/home/mpascual/research/code/neuromf/docs/splits/phase_1.md` |
| 2 | `/home/mpascual/research/code/neuromf/docs/splits/phase_2.md` |
| 3 | `/home/mpascual/research/code/neuromf/docs/splits/phase_3.md` |
| 4 | `/home/mpascual/research/code/neuromf/docs/splits/phase_4.md` |
| 5 | `/home/mpascual/research/code/neuromf/docs/splits/phase_5.md` |
| 6 | `/home/mpascual/research/code/neuromf/docs/splits/phase_6.md` |
| 7 | `/home/mpascual/research/code/neuromf/docs/splits/phase_7.md` |
| 8 | `/home/mpascual/research/code/neuromf/docs/splits/phase_8.md` |

### 6.3 Code Exploration Documents (pre-computed reference for each external repo)

These capture findings from reading the vendored repos so you don't need to re-explore them.

| Topic | Path | Key content |
|-------|------|------------|
| MAISI VAE API | `/home/mpascual/research/code/neuromf/docs/papers/maisi_2024/code_exploration.md` | VAE constructor args, encode/decode API, scale_factor extraction, preprocessing transforms, num_splits memory optimization, 2.5D FID protocol |
| MeanFlow (JAX) | `/home/mpascual/research/code/neuromf/docs/papers/meanflow_2025/code_exploration.md` | JVP loss (lines 226-236), t/r sampling with data_proportion, 1-NFE formula, Algorithm 1, adaptive weighting |
| MeanFlow (PyTorch) | `/home/mpascual/research/code/neuromf/docs/papers/meanflow_2025/pytorch_code_exploration.md` | `torch.func.jvp` usage, standalone class design (not nn.Module), key diffs from JAX, sampling code |
| pMF | `/home/mpascual/research/code/neuromf/docs/papers/pmf_2026/code_exploration.md` | x-prediction reparameterization, compound V, adaptive weighting, LPIPS+ConvNeXt perceptual losses, dual-head MiT architecture |
| MOTFM | `/home/mpascual/research/code/neuromf/docs/papers/motfm_2025/code_exploration.md` | ODE solver (midpoint/rk4/euler), MergedModel UNet+ControlNet wrapper, velocity matching loss, PyTorch Lightning training |

### 6.4 Data & Checkpoint Exploration

| Topic | Path | Key content |
|-------|------|------------|
| IXI Dataset | `/home/mpascual/research/code/neuromf/docs/data/ixi_exploration.md` | 581 T1 files, shapes, spacing, intensity ranges, preprocessing pipeline (Orientation→Spacing→Percentile→Crop) |
| MAISI Checkpoints | `/home/mpascual/research/code/neuromf/docs/data/checkpoint_exploration.md` | VAE state dict structure (wrapped in `"unet_state_dict"`), diffusion checkpoint keys, **scale_factor=0.9624** extraction code |

### 6.5 Paper PDFs

| Paper | Path |
|-------|------|
| Flow Matching (2023) | `/home/mpascual/research/code/neuromf/docs/papers/flow_matching_2023/flow-matching.pdf` |
| MeanFlow (2025) | `/home/mpascual/research/code/neuromf/docs/papers/meanflow_2025/meanflow.pdf` |
| Improved MeanFlow (2025) | `/home/mpascual/research/code/neuromf/docs/papers/imf_2025/improved-mean-flows.pdf` |
| MAISI-v2 (2025) | `/home/mpascual/research/code/neuromf/docs/papers/maisi_v2_2025/maisi-v2.pdf` |
| MOTFM (2025) | `/home/mpascual/research/code/neuromf/docs/papers/motfm_2025/motfm.pdf` |
| LoRA (2022) | `/home/mpascual/research/code/neuromf/docs/papers/lora_2022/lora.pdf` |
| pMF (2026) | `/home/mpascual/research/code/neuromf/docs/papers/pmf_2026/pmf.pdf` |
| SLIM-Diff (2026) | `/home/mpascual/research/code/neuromf/docs/papers/slim_diff_2026/slim-diff.pdf` |

---

## 7. Dependency Management

All dependencies are declared in `pyproject.toml`. If you need a new package:
1. Add it to the appropriate section in `pyproject.toml` (core `dependencies` or an optional group).
2. Run: `~/.conda/envs/neuromf/bin/pip install -e "/home/mpascual/research/code/neuromf"`

Do NOT install packages with bare `pip install <pkg>` — always go through `pyproject.toml` so the dependency is tracked.

---

## 8. Coding Standards

Full standards are in `.claude/rules/coding-standards.md` (auto-loaded). The essentials:

- **Type hints** on all function signatures and return types.
- **Google-style docstrings** on all public functions/classes.
- **No magic numbers** — all hyperparameters from YAML configs via OmegaConf/Hydra.
- **Prefer library functions:** MONAI transforms, `einops.rearrange`, `F.scaled_dot_product_attention`.
- **Test naming:** `test_P{N}_T{M}_<description>` matching phase splits.
- **Leverage reference codebases.** Port from PyTorch MeanFlow reference, do not reimplement.
- **Logging:** Python `logging` with `rich` handler. INFO for events, DEBUG for shapes/values.

---

## 9. Testing

- **Framework:** pytest via `~/.conda/envs/neuromf/bin/python -m pytest`
- **Fixtures:** `tests/conftest.py` provides `base_config`, `device`, `results_root`
- **Markers:** `phase0`–`phase7`, `critical`, `informational` (defined in `pyproject.toml`)
- **Gating:** A phase gate is OPEN when all its CRITICAL tests pass
- **Run all:** `~/.conda/envs/neuromf/bin/python -m pytest tests/ -v --tb=short`
- **Run one phase:** `~/.conda/envs/neuromf/bin/python -m pytest tests/ -v -k "P3"`

---

## 10. Quick Reference: What to Read for Each Phase

| Phase | Must read | Useful if stuck |
|-------|-----------|----------------|
| 0 | `phase_0.md`, `maisi_2024/code_exploration.md`, `checkpoint_exploration.md` | `ixi_exploration.md` |
| 1 | `phase_1.md`, `ixi_exploration.md`, `maisi_2024/code_exploration.md` | — |
| 2 | `phase_2.md`, `meanflow_2025/code_exploration.md` | `pytorch_code_exploration.md` |
| 3 | `phase_3.md`, `pytorch_code_exploration.md`, `pmf_2026/code_exploration.md` | `methodology_expanded.md` §2-4 |
| 4 | `phase_4.md` | `technical_guide.md` §6 |
| 5 | `phase_5.md`, `maisi_2024/code_exploration.md` (2.5D FID section) | `motfm_2025/code_exploration.md` |
| 6 | `phase_6.md` | `methodology_expanded.md` §9 |
| 7 | `phase_7.md` | `lora_2022/lora.pdf` |
| 8 | `phase_8.md` | All previous experiment results |
