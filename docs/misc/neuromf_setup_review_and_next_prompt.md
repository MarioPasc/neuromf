# NeuroMF Agent Setup — Review & Next Prompt

## Part 1: Review of the Implementation Plan

### Overall Assessment: ✅ Correct and Well-Structured

The implementation plan (7 deliverables, 44 new files, 18 modified files) is internally consistent with both `technical_guide.md` and `methodology_expanded.md`. The execution order is logically sound — skeleton first (G), then permissions (B), then CLAUDE.md (A), then agents/commands (C/D), then papers (F), then phase splits (E). Below is a detailed audit.

---

### 1. Discrepancy Table — Correct

The discrepancy table accurately identifies the gaps between the original `agent_setup_prompt.md` assumptions and the actual repository state:

- **`model-zoo/` and `monai-tutorials/` not needed:** Correct. MONAI is a pip dependency; `NV-Generate-CTMR/` contains the MAISI-specific configs and scripts that are not in the pip package.
- **`pmf/` lowercase:** Correct. The actual cloned repo uses lowercase.
- **`checkpoints/` and `datasets/` not at project root:** Correct. These are external paths that must be configured in `configs/base.yaml`, not embedded in the repo tree.
- **Stale `vMF`/`vmf` references:** Confirmed present. The grep command `grep -r "vMF\|vmf" .claude/` is the right verification.

**One minor addition to consider:** The plan notes `src/external/MeanFlow/` is from `github.com/Gsunshine/meanflow`, but the technical guide says `github.com/zhuyu-cs/MeanFlow`. Verify which is the actual cloned upstream. This does not affect the plan's correctness but should be checked during execution.

---

### 2. Deliverable-by-Deliverable Audit

#### G (Skeleton) ✅
- All 9 `__init__.py` docstrings are present and correctly summarise each submodule.
- The 4 stale-reference fixes target the right files.
- Missing directories (`docs/splits/`, `configs/overrides/`, `.claude/commands/`, experiment ablation folders) are all correctly identified.

#### B (Settings) ✅
- The new `allow` rules are sensible: `python *`, `nvidia-smi`, `find`, `cp`, `mv`, `git add/commit/checkout` are all needed for autonomous agent operation.
- Moving `pip install` from deny to allow is correct per the setup prompt.
- The new `deny` rules (`src/external/**`, `docs/main/**`, `checkpoints/**`, `datasets/**`) correctly protect read-only assets.
- **Note:** The deny rules for `checkpoints/**` and `datasets/**` reference project-root-relative paths, but these directories live on an external drive. The agent should verify whether these deny patterns match the actual symlink or mount structure, or whether they need to be absolute paths.

#### A (CLAUDE.md) ✅
- All 8 sections are specified with the right content. The forbidden actions list (no modifying `src/external/`, no retraining MAISI VAE, no `diffusers`, no `torchcfm`) is consistent with `methodology_expanded.md` §7 and the technical guide §11.

#### C (Agents) ✅
- The 3 new agents and 1 update are well-scoped: `external-code-reviewer` (sonnet), `phase-implementer` (opus), `paper-figure-generator` (sonnet), `test-runner` (haiku). Model assignments match task complexity.
- The `external-code-reviewer` producing a 7-section `insights.md` is a good idea — it creates a durable artefact that the `phase-implementer` can consume later.

#### D (Commands) ✅
- The 4 slash commands (`review-external`, `implement-phase`, `run-tests`, `check-gate`) create the right workflow interface.

#### F (Papers) ✅
- Renaming to `meanflow_2025/`, `maisi_2024/`, etc. is consistent with academic convention.
- The 5 new paper folders (`imf_2025`, `maisi_v2_2025`, `slim_diff_2026`, `lora_2022`, `flow_matching_2023`) cover all papers listed in the technical guide §11.1 that don't already have folders.
- **Note:** The plan includes `slim_diff_2026` (Pascual-González et al., 2026) — ensure the PDF is available or add a placeholder README noting it's unpublished/in-press.

#### E (Phase Splits) ✅
- All 9 phases (0–8) are present with correct equation references, external code pointers, test IDs, and file lists.
- Phase 3 is correctly flagged as the hardest engineering phase (JVP compatibility with MAISI UNet).
- The test counts across phases appear consistent with the technical guide (58 tests mentioned; the plan lists P0-T1→T7, P1-T1→T7, P2-T1→T8, P3-T1→T10, P4-T1→T8, P5-T1→T6, P6-T1→T5, P7-T1→T7 = 7+7+8+10+8+6+5+7 = 58 ✓).

---

### 3. Potential Issues to Monitor

1. **Conda env path mismatch:** The plan references `~/.conda/envs/neuromf/bin/python` in the settings, but the user specifies `~/conda/envs/neuromf/bin` (no dot). Verify the actual path on the machine.
2. **External drive paths:** The plan correctly notes that `checkpoints/` and `datasets/` are not at project root, but does not explicitly set up the path constants in `configs/base.yaml`. This is deferred to the next prompt (below).
3. **`pmf/` vs `pMF/`:** The plan correctly uses lowercase `pmf/` everywhere, but a search for stale `pMF` references (uppercase M) should also be run.

---

## Part 2: Next Agent Prompt

The following prompt is designed for the **second setup session** — focused on environment exploration, path configuration, external code analysis, and quality-of-life tooling. It assumes Deliverables A–G from the first session are complete.

---

```markdown
# NeuroMF — Phase 0 Preparation: Environment Exploration & Path Configuration

> **Context:** The first setup session created CLAUDE.md, phase splits, agent files, slash commands, and fixed all scaffolding. This session prepares the agent environment for Phase 0 implementation by exploring external code, configuring all concrete paths, and setting up testing infrastructure.

---

## 0. Pre-Flight: Read Before Doing Anything

1. Read `CLAUDE.md` at project root — understand the full project context.
2. Read `docs/splits/phase_0.md` — understand what Phase 0 will implement.
3. Read `docs/splits/phase_1.md` — understand the immediate downstream dependency.
4. Run `nvidia-smi` to confirm GPU availability and VRAM.
5. Run `~/conda/envs/neuromf/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` to confirm PyTorch sees the GPU.

---

## 1. Concrete Path Configuration

### 1.1 System Paths (Hardcode in `configs/base.yaml`)

Create or update `configs/base.yaml` with these **absolute paths**:

```yaml
paths:
  project_root: "/media/mpascual/Sandisk2TB/research/neuromf"
  conda_python: "/home/mpascual/conda/envs/neuromf/bin/python"
  conda_env: "/home/mpascual/conda/envs/neuromf"

  # Data
  datasets_root: "/media/mpascual/Sandisk2TB/research/neuromf/datasets"
  ixi_raw: "/media/mpascual/Sandisk2TB/research/neuromf/datasets/IXI"

  # Model weights
  checkpoints_root: "/media/mpascual/Sandisk2TB/research/neuromf/checkpoints"
  maisi_vae_weights: "/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models"

  # Outputs
  results_root: "/media/mpascual/Sandisk2TB/research/neuromf/results"
  latents_dir: "/media/mpascual/Sandisk2TB/research/neuromf/results/latents"
  generated_dir: "/media/mpascual/Sandisk2TB/research/neuromf/results/generated"
  figures_dir: "/media/mpascual/Sandisk2TB/research/neuromf/results/figures"
  logs_dir: "/media/mpascual/Sandisk2TB/research/neuromf/results/logs"

  # Training checkpoints (model saves during training)
  training_checkpoints: "/media/mpascual/Sandisk2TB/research/neuromf/results/training_checkpoints"
```

### 1.2 Verify All Paths Exist

Write a small Python script `scripts/verify_paths.py` that:
- Reads `configs/base.yaml` using OmegaConf
- Checks each path under `paths:` exists (or can be created for output dirs)
- Creates output directories if they don't exist (`results/`, `results/latents/`, `results/generated/`, `results/figures/`, `results/logs/`, `results/training_checkpoints/`)
- Prints a summary table: path → status (✓ exists / ✓ created / ✗ missing)
- Exits with code 1 if any input path (datasets, checkpoints) is missing

Run the script and fix any issues.

### 1.3 Update `.claude/settings.json` Deny Rules

Update the deny rules that reference `checkpoints/**` and `datasets/**` to use the actual absolute paths:
- `Edit(/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/**)` — protect pretrained weights
- `Edit(/media/mpascual/Sandisk2TB/research/neuromf/datasets/**)` — protect raw data

---

## 2. Conda Environment Audit

### 2.1 Verify Installed Packages

Run `~/conda/envs/neuromf/bin/pip list` and check that these critical packages are installed:

| Package | Required | Why |
|---|---|---|
| `torch` | ≥ 2.1 | Core framework, `torch.func.jvp` support |
| `monai` | ≥ 1.3 | MAISI VAE, medical transforms |
| `pytorch-lightning` | ≥ 2.0 | Training loop |
| `omegaconf` | any | Config management |
| `einops` | any | Tensor reshaping |
| `nibabel` | any | NIfTI I/O |
| `scikit-image` | any | SSIM/PSNR |
| `rich` | any | Logging |
| `peft` | any | LoRA (Phase 7, but install now) |
| `lpips` | any | Perceptual loss |
| `torch-fidelity` | any | FID computation |
| `pytest` | any | Testing |
| `matplotlib` | any | Plotting |
| `seaborn` | any | Statistical plots |
| `scipy` | any | Statistical tests |
| `pandas` | any | Results aggregation |

### 2.2 Install Missing Packages

If any are missing, install them:
```bash
~/conda/envs/neuromf/bin/pip install <package>
```

### 2.3 Verify `torch.func.jvp` Works

This is critical for Phase 3. Run:
```python
import torch
from torch.func import jvp

def f(x):
    return x ** 2

x = torch.randn(4, requires_grad=False)
v = torch.randn(4)
y, jvp_val = jvp(f, (x,), (v,))
print(f"jvp works: y={y.shape}, jvp={jvp_val.shape}")
```

---

## 3. External Code Exploration

### 3.1 Explore `src/external/NV-Generate-CTMR/` (MAISI VAE — Critical for Phase 0)

This is the most important external repo for Phase 0. Explore it thoroughly:

1. **List the directory tree** (2 levels deep) to understand structure.
2. **Find the VAE model definition:** Search for files containing `AutoencoderKL` or `VQModel` or `vae` in the filenames and class definitions.
3. **Find the weight loading code:** Search for `load_state_dict`, `torch.load`, or `safetensors` usage.
4. **Find preprocessing examples:** Search for MONAI transform chains (`Compose`, `LoadImage`, `Orientation`, `Spacing`, `ScaleIntensity`).
5. **Find the encode/decode API:** Identify the exact method signatures for encoding a volume to latent and decoding back.
6. **Check latent space dimensions:** Confirm the 4-channel, 4× downsampling claim (128³ → 4×32³).
7. **Document findings** in `docs/papers/maisi_2024/code_exploration.md` with:
   - Exact file paths within `src/external/NV-Generate-CTMR/` for VAE class, weight loading, preprocessing
   - The actual class name and constructor arguments
   - The encode/decode method signatures
   - Any gotchas (e.g., does encode return mean or sample? Is there a scaling factor?)

### 3.2 Explore `src/external/MeanFlow/` (JAX Reference — Critical for Phase 2–3)

1. **Find the JVP loss implementation:** The key file is `meanflow.py`. Locate the `jvp` call and document:
   - The exact loss formulation (which lines)
   - How `t` and `r` are sampled
   - How the compound prediction $V_\theta(\mathbf{z}_t, t, r)$ is defined
   - Whether it uses x-prediction or u-prediction
2. **Find Algorithm 1 (training loop):** Identify the data flow from noise + data → loss.
3. **Document findings** in `docs/papers/meanflow_2025/code_exploration.md`.

### 3.3 Explore `src/external/MeanFlow-PyTorch/` (PyTorch Reference — Critical for Phase 2–3)

1. **Find the PyTorch JVP loss:** This is the direct reference for our PyTorch implementation.
2. **Compare with JAX version:** Note any differences in API, numerical stability tricks, or time sampling.
3. **Find any in-place operation workarounds:** These are critical for `torch.func.jvp` compatibility.
4. **Document findings** in `docs/papers/meanflow_2025/pytorch_code_exploration.md`.

### 3.4 Explore `src/external/pmf/` (x-Prediction Reference — Critical for Phase 3)

1. **Find the x-prediction reparameterisation:** How does pMF convert from x-prediction to velocity?
2. **Find the adaptive weighting formula:** Equation 14 in methodology_expanded.md.
3. **Document findings** in `docs/papers/pmf_2026/code_exploration.md`.

### 3.5 Explore `src/external/MOTFM/` (Evaluation Reference — Critical for Phase 5)

1. **Find the FID computation code:** How does MOTFM compute FID on 3D volumes (slice-wise protocol)?
2. **Find the inference/sampling code:** `inferer.py` — how does it generate samples and save them?
3. **Find the evaluation metrics:** What metrics are computed and how?
4. **Document findings** in `docs/papers/motfm_2025/code_exploration.md`.

---

## 4. Dataset Exploration

### 4.1 Explore IXI Dataset

1. **List the contents** of `/media/mpascual/Sandisk2TB/research/neuromf/datasets/IXI/`:
   - How many subjects?
   - What modalities are available (T1, T2, PD, FLAIR)?
   - What is the file format (NIfTI .nii.gz)?
   - What is the naming convention?
2. **Load one volume** and report:
   - Shape (e.g., 256×256×150)
   - Voxel spacing (affine matrix)
   - Intensity range
   - Data type (float32, int16, etc.)
3. **Document findings** in `docs/data/ixi_exploration.md` with:
   - Dataset statistics (N subjects, modalities, resolution)
   - Preprocessing requirements to reach 128³ at 1mm³ isotropic
   - Any quality issues (corrupt files, missing modalities)

### 4.2 Explore MAISI Checkpoint Structure

1. **List the contents** of `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/`:
   - What files are present? (.pt, .pth, .safetensors, .ckpt?)
   - What is the file size? (sanity check: VAE weights should be ~200MB–1GB)
   - Is there a config file alongside the weights?
2. **Document findings** in `docs/data/checkpoint_exploration.md`.

---

## 5. Results Directory Structure

Create the following directory structure under the results root:

```
/media/mpascual/Sandisk2TB/research/neuromf/results/
├── phase_0/
│   ├── vae_validation/
│   │   ├── metrics/          # SSIM, PSNR, LPIPS CSV files
│   │   ├── reconstructions/  # Sample reconstruction images
│   │   └── latent_stats/     # Per-channel statistics, PCA, histograms
│   └── verification_report.md
├── phase_1/
│   ├── latent_cache/         # Pre-computed .pt latent files
│   └── verification_report.md
├── phase_2/
│   ├── toroid/
│   │   ├── training_curves/
│   │   ├── generated_samples/
│   │   └── angular_distributions/
│   └── verification_report.md
├── phase_3/
│   └── verification_report.md
├── phase_4/
│   ├── training_logs/
│   ├── samples/              # Periodic 1-NFE samples
│   └── verification_report.md
├── phase_5/
│   ├── generated_volumes/    # 2000 synthetic volumes
│   ├── metrics/
│   └── verification_report.md
├── phase_6/
│   ├── ablation_xpred_upred/
│   ├── ablation_lp_sweep/
│   ├── ablation_nfe_steps/
│   └── verification_report.md
├── phase_7/
│   ├── lora_training/
│   ├── fcd_samples/
│   └── verification_report.md
├── phase_8/
│   ├── figures/
│   └── tables/
├── latents/                  # Shared latent cache (used by multiple phases)
├── training_checkpoints/     # Model checkpoints during training
├── generated/                # Generated volumes (general)
├── figures/                  # Publication figures
└── logs/                     # W&B / TensorBoard logs
```

---

## 6. Testing Infrastructure Setup

### 6.1 Create `conftest.py`

Create `tests/conftest.py` with shared fixtures:

```python
"""Shared pytest fixtures for NeuroMF test suite."""
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def base_config():
    """Load base configuration with all paths."""
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    return OmegaConf.load(config_path)


@pytest.fixture(scope="session")
def device():
    """Return available device (GPU preferred)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def results_root(base_config):
    """Return the results root directory."""
    return Path(base_config.paths.results_root)
```

### 6.2 Create `pytest.ini` or update `pyproject.toml`

Add pytest configuration:
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "phase0: Phase 0 tests (VAE validation)",
    "phase1: Phase 1 tests (latent pre-computation)",
    "phase2: Phase 2 tests (toroid toy experiment)",
    "phase3: Phase 3 tests (MeanFlow loss + UNet)",
    "phase4: Phase 4 tests (training)",
    "phase5: Phase 5 tests (evaluation)",
    "phase6: Phase 6 tests (ablations)",
    "phase7: Phase 7 tests (LoRA FCD)",
    "critical: Must pass for phase gate",
    "informational: Nice to have, not gating",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### 6.3 Verify Test Runner Works

Run `~/conda/envs/neuromf/bin/python -m pytest tests/ --collect-only` to confirm pytest discovers test files (even if they're empty stubs).

---

## 7. Paper PDF Verification

### 7.1 Check PDFs Are Readable

For each PDF in `docs/papers/`:
1. Confirm the file exists and is non-empty
2. Attempt to extract text (e.g., first page) to verify it's not corrupted

The PDFs that should exist:
- `docs/papers/meanflow_2025/meanflow.pdf` (Geng et al., 2025a — MeanFlow)
- `docs/papers/maisi_2024/maisi.pdf` (Guo et al., 2024 — MAISI)
- `docs/papers/motfm_2025/motfm.pdf` (Yazdani et al., 2025 — MOTFM)
- `docs/papers/pmf_2026/pmf.pdf` (Lu et al., 2026 — pMF)

If the rename from Step 6/Deliverable F hasn't happened yet, they may still be at the old paths (`docs/papers/meanflow/`, etc.). Document whichever state you find.

### 7.2 Papers in Claude's Project Context

The following PDFs are also available in the Claude project context (attached to the project). Verify they match the papers in `docs/papers/` by comparing titles:
- `flowmatching.pdf` — should be Lipman et al. 2023
- `improvedmeanflows.pdf` — should be Geng et al. 2025b (iMF)
- `lora.pdf` — should be Hu et al. 2022
- `maisiv2.pdf` — should be Zhao et al. 2025
- `meanflow.pdf` — should be Geng et al. 2025a
- `motfm.pdf` — should be Yazdani et al. 2025
- `pmf.pdf` — should be Lu et al. 2026

---

## 8. Quality-of-Life: Helper Scripts

### 8.1 Create `scripts/activate.sh`

A convenience script for activating the environment:
```bash
#!/bin/bash
# Source this file: source scripts/activate.sh
export NEUROMF_ROOT="/media/mpascual/Sandisk2TB/research/neuromf"
export NEUROMF_RESULTS="/media/mpascual/Sandisk2TB/research/neuromf/results"
export NEUROMF_DATA="/media/mpascual/Sandisk2TB/research/neuromf/datasets"
export NEUROMF_CKPT="/media/mpascual/Sandisk2TB/research/neuromf/checkpoints"
conda activate neuromf
echo "NeuroMF environment activated. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
```

### 8.2 Create `scripts/check_env.py`

A comprehensive environment check that the agent can run at the start of any session:
```python
"""Check that the NeuroMF environment is correctly configured."""
# Checks: Python version, torch version, CUDA, all paths, all packages, disk space
```

---

## 9. Deliverables Checklist

After completing this session, the following should be true:

- [ ] `configs/base.yaml` has all absolute paths configured
- [ ] `scripts/verify_paths.py` runs clean (all paths exist or created)
- [ ] All critical pip packages are installed in the conda env
- [ ] `torch.func.jvp` works on the target GPU
- [ ] `docs/papers/maisi_2024/code_exploration.md` documents VAE class, encode/decode API, weight loading
- [ ] `docs/papers/meanflow_2025/code_exploration.md` documents JVP loss, time sampling, Algorithm 1
- [ ] `docs/papers/meanflow_2025/pytorch_code_exploration.md` documents PyTorch JVP differences
- [ ] `docs/papers/pmf_2026/code_exploration.md` documents x-prediction reparameterisation
- [ ] `docs/papers/motfm_2025/code_exploration.md` documents FID protocol and inference
- [ ] `docs/data/ixi_exploration.md` documents dataset stats and preprocessing needs
- [ ] `docs/data/checkpoint_exploration.md` documents weight file structure
- [ ] Results directory structure created under `/media/mpascual/Sandisk2TB/research/neuromf/results/`
- [ ] `tests/conftest.py` with shared fixtures
- [ ] pytest configuration in `pyproject.toml`
- [ ] `pytest --collect-only` works
- [ ] All paper PDFs verified readable
- [ ] `scripts/activate.sh` and `scripts/check_env.py` created

**After this session, the agent environment is fully prepared for Phase 0 implementation.**
```

---

## Part 3: Notes on Prompt Design

The prompt above is designed with the following principles:

1. **Concrete paths everywhere.** The agent needs absolute paths — no `~/` ambiguity, no relative assumptions. Every path the agent will ever need is defined in `configs/base.yaml` as a single source of truth.

2. **External code exploration produces durable artefacts.** Each `code_exploration.md` file becomes a reference that the `phase-implementer` agent can read later without re-exploring the external repos. This is critical because context windows are finite.

3. **Results directory structure mirrors the phase system.** Each phase gets its own results folder with predictable subdirectories, making it trivial for the agent to know where to write outputs.

4. **Testing infrastructure is set up before any tests exist.** The `conftest.py` fixtures and pytest markers mean that when the Phase 0 implementer starts writing tests, the infrastructure is already there.

5. **Environment verification is scriptable.** The `verify_paths.py` and `check_env.py` scripts can be run at the start of every future session, catching path or dependency issues immediately.

6. **The prompt does NOT implement anything.** It only explores, documents, and configures — consistent with the "setup, not implementation" philosophy of the first session.
