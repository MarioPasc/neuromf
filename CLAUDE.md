# NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis

## 1. What This Project Does

NeuroMF trains a **MeanFlow** model in the latent space of a **frozen MAISI 3D VAE** to achieve **1-step (1-NFE) generation** of 192^3 brain MRI volumes. The project introduces per-channel Lp loss (extending SLIM-Diff to latent space) and LoRA fine-tuning for joint synthesis of rare epilepsy pathology (FCD). Target venues: Medical Image Analysis, IEEE TMI, or MICCAI 2026.

**Layer 1 — Flow Matching (the base idea)**: You define a straight-line path between data z_0 and noise eps: z_t = (1-t)*z_0 + t*eps. A neural network learns the velocity field v(z_t, t) that transports noise to data. At inference, you integrate v over many steps (e.g., 50 Euler steps) to turn noise into data. The training loss is simple: ||v_theta(z_t, t) - (eps - z_0)||^2.

**Layer 2 — MeanFlow (the 1-step trick)**: Instead of instantaneous velocity v at time t, learn the average velocity u(z_t, r, t) over interval [r, t]. If you know the average velocity from t=0 to t=1, you can generate in 1 step: z_0 = noise - u(noise, 0, 1). But u must be self-consistent: the average over [r, t] must relate properly to averages over sub-intervals. This self-consistency is enforced via the compound velocity: V = u + (t-r) * stop_grad(du/dt), where du/dt comes from a JVP (Jacobian-vector product). Training: ||V - (eps - z_0)||^2.

**Layer 3 — Improved MeanFlow / iMF**: Original MF computes the JVP using ground-truth velocity (eps - z_0) as the tangent direction. Problem: this makes V depend on data you don't have at inference. iMF says: use the model's own prediction v_tilde = u(z_t, t, t) as the tangent instead. Now V depends only on z_t — matching what happens at inference. This gives lower-variance gradients and stable loss curves.

**Layer 4 — iMF Dual-Head (what we use):** The UNet has a shared backbone with two output heads: the u-head predicts the average velocity u (or x_hat in x-prediction mode), and a v-head (supervised with its own loss `||v - v_c||^p`) predicts the instantaneous velocity used as the JVP tangent. The v-head receives direct supervision toward the correct tangent, providing high-quality JVP tangents from early training — solving the MF loss divergence problem. The v-head is disabled at inference (zero cost). Our best configuration uses **x-prediction** (model outputs denoised data estimate x_hat, from which u is derived as `u = (z_t - x_hat) / t`) with **exact JVP** (`torch.func.jvp`) and **(t, h) conditioning** (condition on both t and h=t-r, per MF Table 1c). Dual loss: `loss = loss_u_weighted + loss_v_weighted`, each independently adaptive-weighted. **Critical rule:** x-pred + exact JVP = stable; u-pred + FD-JVP = stable; x-pred + FD-JVP = explosion (1/t singularity amplified by finite differences).

In summary, one training step does:
1. Sample z_0 (data), eps (noise), t, r; compute z_t = (1-t)z_0 + teps
2. v_tangent = v_head(z_t, t, t) [no grad — tangent from supervised v-head]
3. JVP via `torch.func.jvp` with `has_aux=True`: get (u, du/dt, v) from dual_fn
   - dual_fn returns (u, v) where u = (z_t - x_hat) / t [x-pred -> u conversion]
4. Compound velocity: V = u + (t-r)*sg[du/dt]
5. Loss: ||V - v_c||^p (MF consistency) + ||v - v_c||^p (tangent supervision), each adaptive-weighted
6. At inference (1-NFE): z_0 = model(noise, r=0, t=1) — direct x-prediction output

### Core Pipeline

```
Input MRI (1×192³) ─► Frozen MAISI VAE Encoder ─► Latent (4×48³)
                                                      │
                                              Train MeanFlow (1-NFE)
                                                      │
                                                      ▼
Synthetic MRI (1×192³) ◄── Frozen MAISI VAE Decoder ◄─┘
```

### Key Shapes

| Space | Shape | Notes |
|-------|-------|-------|
| Pixel | `(B, 1, 192, 192, 192)` | Single-channel MRI |
| Latent | `(B, 4, 48, 48, 48)` | 4x spatial compression per axis, 1→4 channels |

### MeanFlow in One Paragraph

MeanFlow learns the **average velocity** `u(z_t, t, r)` instead of the instantaneous velocity `v(z_t, t)`. The MeanFlow Identity enforces self-consistency via a JVP (Jacobian-vector product). At inference, a single forward pass produces a sample: `z_0 = eps - u_θ(eps, 0, 1)`. Training uses the iMF dual-head architecture: a u-head for the compound velocity loss and a v-head that provides the JVP tangent (disabled at inference). Both losses are independently adaptive-weighted.

### Current Status (as of 2026-03-15)

**Phases 0-5 COMPLETE.** Best v1 model trained 690 epochs (28,980 steps), early-stopped (patience=10).

| Metric | NFE=1 | NFE=10 | NFE=50 |
|--------|-------|--------|--------|
| FID-3D | 73.85 | 7.34 | 6.14 |
| MOTFM baseline | 32.10 | 9.27 | 7.93 |

**Known issues:** Variance under-estimation at 1-NFE (std ~0.75 vs ~1.0 real), over-smoothed outputs. NeuroMF loses to MOTFM at NFE=1 but wins at NFE>=10.

**Remaining:** Phase 6 PARTIAL (Lp sweep pending). Phases 7-8 NOT STARTED.

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
| VAE memory splits | `num_splits=6, dim_split=1` | Enables 192³ on 8GB VRAM |
| VAE checkpoint format | Wrapped in `"unet_state_dict"` key | Must unwrap before `load_state_dict` |
| FOMO-60K subset | 5,471 train / 674 val / 326 test T1 scans (8 datasets) | Stratified split, seed=42 |
| Best v1 model | Epoch 388 (step 16,338) | x-pred + exact JVP, v-head, (t,h) conditioning |
| Effective batch size | 132 (2/GPU × 6 GPUs × 11 accum) | Picasso A100 training config |
| FOMO-60K status | Skull-stripped, RAS, co-registered | Shapes/spacing vary by dataset |
| Hardware (local) | RTX 4060 Laptop, 8GB VRAM | CPU-only tests, stats, figures, code dev |
| Hardware (Picasso) | 4 nodes × 8×A100 40GB VRAM, 500GB RAM, 128 cores | VAE encode/decode, training, evaluation |

### Hardware Usage Policy

- **Local laptop (RTX 4060 8GB):** Code development, unit tests with mock data, statistical analysis, figure generation, git operations. **No GPU-heavy tasks** — 192³ volumes exceed 8GB VRAM even with `num_splits=6`.
- **Picasso supercomputer (A100 40GB):** All GPU workloads — VAE validation (Phase 0), latent encoding (Phase 1), MeanFlow training (Phase 4), evaluation (Phase 5), ablations (Phase 6). Atomic SLURM jobs live in `slurm/<task>/` (e.g., `slurm/train/`, `slurm/generate/`); multi-model orchestration in `experiments/slurm/phase_5/`.
- **SLURM scripts:** Each atomic job has a `launch.sh` (run from login node) and `worker.sh` (submitted by launcher). Launchers send status to stderr, job ID to stdout (for orchestration capture via `JOB_ID=$(bash slurm/X/launch.sh ...)`). All launchers accept `--depends-on JOB_ID`.

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

| Phase | Title | Key Output | Status |
|-------|-------|------------|--------|
| 0 | Environment Bootstrap & VAE Validation | `maisi_vae.py` wrapper, reconstruction metrics | COMPLETE |
| 1 | Latent Pre-computation Pipeline | `.pt` latent files, per-channel stats | COMPLETE |
| 2 | Toy Experiment — MeanFlow on Toroid | Validated MeanFlow on known manifold | COMPLETE |
| 3 | MeanFlow Loss + 3D UNet | JVP-compatible wrapper, MeanFlow loss | COMPLETE |
| 4 | Training on Brain MRI Latents | Trained model, EMA checkpoints | COMPLETE |
| 5 | Generation Pipeline + Evaluation | Latent generation, VAE decoding, FID/MMD/MS-SSIM/spectral metrics | COMPLETE |
| 6 | Ablation Runs | x-pred vs u-pred, Lp sweep, NFE steps | PARTIAL |
| 7 | LoRA Fine-Tuning for FCD | Joint image-mask synthesis | NOT STARTED |
| 8 | Paper Figures and Tables | Publication-ready figures (PDF+PNG) | NOT STARTED |

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
| FOMO-60K root | `/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K/` |
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

### 5.3b Competitor Model System

The project uses an extensible competitor framework for evaluating pixel-space generative models against NeuroiMF. **To add a new competitor, see `docs/adding_competitors.md`.**

| Resource | Path |
|----------|------|
| Competitor base class | `src/neuromf/competitors/base.py` |
| Model registry | `src/neuromf/competitors/registry.py` |
| Shared HDF5 writer | `src/neuromf/competitors/io.py` |
| Style registry | `src/neuromf/competitors/styles.py` |
| MOTFM implementation | `src/neuromf/competitors/motfm_gen.py` |
| DDPM implementation | `src/neuromf/competitors/ddpm_gen.py` |
| Unified generation CLI | `experiments/cli/generate_competitor.py` |
| Generic SLURM launcher | `slurm/generate_competitor/launch.sh` |
| Multi-model comparison CLI | `experiments/cli/run_comparison.py` |
| Comparison figures | `experiments/analysis/comparison_figures.py` |
| Adding competitors guide | `docs/adding_competitors.md` |

**Key commands:**
```bash
# List registered competitors:
python experiments/cli/generate_competitor.py --list-models

# Generate for any registered model:
python experiments/cli/generate_competitor.py --model motfm --config ... --checkpoint ...

# Multi-model comparison (extensible):
python experiments/cli/run_comparison.py --neuroimf-dir PATH --competitor MOTFM:PATH --competitor DDPM:PATH

# Full orchestration:
bash slurm/orchestrate_evaluation/launch.sh --neuroimf-checkpoint ... --competitor motfm:CKPT --competitor ddpm:CKPT
```

### 5.4 External Vendored Repos (READ-ONLY)

| Repo | Path | What it contains |
|------|------|-----------------|
| MeanFlow (JAX) | `src/external/MeanFlow/` | Original JAX reference: JVP loss, t/r sampling, 1-NFE |
| MeanFlow (PyTorch) | `src/external/MeanFlow-PyTorch/` | PyTorch port: `torch.func.jvp`, SiT architecture |
| NV-Generate-CTMR | `src/external/NV-Generate-CTMR/` | MAISI VAE, preprocessing, 2.5D FID evaluation |
| MOTFM | `src/external/MOTFM/` | Medical OT flow matching: trainer, inferer, UNet wrapper |
| pMF | `src/external/pmf/` | Progressive MeanFlow: x-prediction, compound V, perceptual losses |

### 5.5 When to Use What

| You want to... | Use | Trigger |
|----------------|-----|---------|
| Write code for a phase | `/implement-phase N` | "implement phase 3", "build the loss function" |
| Run tests after code changes | `/test` | "run tests", "check if tests pass", after any code edit |
| Validate before training submission | `/pre-flight` | "ready to train?", "check config", before any SLURM job |
| Analyze a completed training run | `/analyze-run <dir>` | "analyze results", "how did training go?", "compare to MOTFM" |
| Diagnose training dynamics | `/phase4-results-diagnoser` | "check training progress", "diagnose loss curves", "why is FID high?" |
| Understand metrics scientifically | `/dl-scientist` | "why is 1-NFE bad?", "propose improvements", "root cause analysis" |
| Check if a phase gate is open | `/check-gate N` | "is phase 3 done?", "can I start phase 4?" |
| Explore codebase | `/explore` | "how does X work?", "where is Y implemented?" |
| Review external paper code | `/review-external` | "compare MeanFlow JAX to our code" |
| Generate publication figures | Launch `paper-figure-generator` | "make figures for the paper", "plot FID comparison" |

### 5.6 Slash Commands

| Command | Usage | What it does |
|---------|-------|-------------|
| `/implement-phase` | `/implement-phase 3` | Launches phase-implementer (Opus) for end-to-end phase work |
| `/run-tests` | `/run-tests 2` | Launches test-runner (Haiku) for phase verification |
| `/check-gate` | `/check-gate 1` | Reads verification report, reports OPEN/BLOCKED |
| `/review-external` | `/review-external meanflow_2025 MeanFlow` | Launches code-reviewer (Sonnet) to produce insights doc |
| `/dl-scientist` | `/dl-scientist` | Rigorous analysis of training results with literature grounding |
| `/pre-flight` | `/pre-flight configs/picasso/train_meanflow.yaml` | Validates config + code before multi-day training submission |
| `/analyze-run` | `/analyze-run /path/to/run_dir` | Post-training analysis: compare vs MOTFM, propose next experiments |
| `/test` | `/test -m "not slow"` | Quick test runner (default: fast suite, ~40s) |

### 5.7 Subagents

| Agent | Model | Purpose |
|-------|-------|---------|
| `phase-implementer` | Opus | Reads phase split, writes code + tests, runs verification |
| `test-runner` | Haiku | Runs pytest with slow/fast awareness, reports pass/fail with gate status |
| `pre-flight-validator` | Opus | Validates config + code before expensive GPU training runs |
| `results-analyst` | Opus | Analyzes completed runs vs MOTFM baseline, proposes improvements |
| `external-code-reviewer` | Sonnet | Reviews external code against paper, produces insights |
| `paper-figure-generator` | Sonnet | Generates publication figures with MOTFM comparison baselines |

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
| FOMO-60K Dataset | `/home/mpascual/research/code/neuromf/docs/data/fomo60k_exploration.md` | 1,379 T1 sessions, 3 datasets, metadata structure, group labels, preprocessing status |
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
| 0 | `phase_0.md`, `maisi_2024/code_exploration.md`, `checkpoint_exploration.md` | `fomo60k_exploration.md` |
| 1 | `phase_1.md`, `fomo60k_exploration.md`, `maisi_2024/code_exploration.md` | — |
| 2 | `phase_2.md`, `meanflow_2025/code_exploration.md` | `pytorch_code_exploration.md` |
| 3 | `phase_3.md`, `pytorch_code_exploration.md`, `pmf_2026/code_exploration.md` | `methodology_expanded.md` §2-4 |
| 4 | `phase_4.md` | `technical_guide.md` §6 |
| 5 | `phase_5.md`, `maisi_2024/code_exploration.md` (2.5D FID section) | `motfm_2025/code_exploration.md` |
| 6 | `phase_6.md` | `methodology_expanded.md` §9 |
| 7 | `phase_7.md` | `lora_2022/lora.pdf` |
| 8 | `phase_8.md` | All previous experiment results |

### v1 Results Summary

| Metric | NFE=1 | NFE=10 | NFE=50 | MOTFM NFE=10 |
|--------|-------|--------|--------|--------------|
| FID-3D | 73.85 | 7.34 | 6.14 | 9.27 |
| MMD | 0.99 | 0.23 | 0.17 | 0.25 |
| MS-SSIM | 0.33 | 0.66 | 0.66 | 0.77 |
