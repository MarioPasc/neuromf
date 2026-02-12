# NeuroMF — Agent Environment Setup Prompt

> **Usage:** Copy this entire file as the initial prompt to the local Claude Code Opus 4.6 instance.  
> **Working directory:** The agent should be launched from the project root (`/media/mpascual/Sandisk2TB/research/neuromf/`).

---

You are a world-class deep learning research engineer. Your sole task right now is **not** to write any model code — it is to architect the perfect agent environment so that future Claude Code sessions (and subagents) can implement this project autonomously, phase by phase, with maximum context efficiency and minimal hallucination.

---

## 0. Project Identity

| Field | Value |
|---|---|
| **Project** | NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis |
| **Goal** | Train a MeanFlow model in the latent space of a frozen MAISI 3D VAE to achieve 1-step (1-NFE) generation of 128³ brain MRI volumes, with per-channel Lp loss and LoRA fine-tuning for rare epilepsy pathology (FCD). |
| **Target venue** | Q1 journal (Medical Image Analysis / IEEE TMI) or MICCAI 2026 |
| **Author** | Mario Pascual-González |

---

## 1. Current State of the Repository — Read Before Doing Anything

**CRITICAL: Before creating any file, read the full project tree and every existing file.** The repo is mostly empty scaffolding, but the following already exist and must not be overwritten or contradicted:

### 1.1 Files You Must Read First (in this order)

1. `docs/main/technical_guide.md` — The master implementation guide. Contains the full project structure, all 9 phases (Phase 0–8), every verification test (P0-T1 through P8), the toroid toy experiment design, the agent context specification (§11), and the suggested SKILL files. **This is your bible.**
2. `docs/main/methodology_expanded.md` — The formal methodology. Contains all mathematical derivations (MeanFlow Identity, JVP computation, Lp loss transfer theory, x-pred vs. u-pred manifold argument, LoRA formulation), the literature framing, the ablation design with statistical protocols, and the evaluation metrics. **You will extract from this for each phase split.**
3. `CLAUDE.md` (if it exists at root) — Any existing project-level instructions.

### 1.2 What Already Exists on Disk

```
/media/mpascual/Sandisk2TB/research/neuromf/
├── .claude/                          # May be empty or partially set up — READ IT
├── docs/
│   ├── main/
│   │   ├── technical_guide.md        # MASTER GUIDE — 900+ lines
│   │   └── methodology_expanded.md   # FORMAL METHODS — 800+ lines
│   └── papers/                       # Will contain PDFs of key papers
├── src/
│   └── external/                     # CLONED REPOS (already done):
│       ├── MeanFlow/                 # github.com/zhuyu-cs/MeanFlow
│       ├── MeanFlow-PyTorch/         # github.com/HaoyiZhu/MeanFlow-PyTorch
│       ├── model-zoo/                # github.com/Project-MONAI/model-zoo
│       ├── monai-tutorials/          # github.com/Project-MONAI/tutorials
│       ├── MOTFM/                    # github.com/milad1378yz/MOTFM
│       └── pMF/                      # github.com/Lyy-iiis/pMF (JAX reference)
├── checkpoints/
│   └── NV-Generate-MR/
│       └── models/
│           ├── autoencoder_v2.pt     # MAISI VAE weights (MR version)
│           └── diff_unet_3d_rflow-mr.pt  # MAISI diffusion UNet (reference only)
├── datasets/
│   └── IXI/                          # Downloaded IXI dataset (NIfTI files)
└── configs/                          # May be empty — you will define the schema
```

### 1.3 Conda Environment

- **Name:** `neuromf` (already created and activated)
- **Key packages installed:** PyTorch ≥2.1, MONAI ≥1.3, PyTorch Lightning ≥2.0, OmegaConf, einops, nibabel, wandb, scikit-image, torch-fidelity, lpips
- You may install additional packages if needed via `pip install --quiet <pkg>` — log what you install.

---

## 2. Your Deliverables — What You Must Create

You will create the following, and **only** the following. Do not write any model code, training code, or data processing code. You are setting up the environment for the agents that will.

### Deliverable A: `CLAUDE.md` (project root)

The master context file that every Claude Code session loads automatically. It must contain:

1. **Project overview** — 3-sentence summary of what NeuroMF is.
2. **Architecture summary** — The pipeline: frozen MAISI VAE → pre-compute latents → train MeanFlow in latent space → decode. Include the key tensor shapes (input 1×128³ → latent 4×32³).
3. **Repository map** — A concise description of every top-level directory and what it contains, so any agent can orient itself instantly.
4. **Coding standards** — The user's preferences (see below).
5. **Key paths** — Absolute paths to: checkpoints, datasets, external repos, configs, results output.
6. **Phase system** — Explain that the project is implemented in phases (0–8), each phase has a split document in `docs/splits/`, and an agent should read the relevant split before starting work. Phases are gated: Phase N+1 cannot start until Phase N's verification tests all pass.
7. **Testing conventions** — Tests live in `tests/`. Use `pytest`. Every phase has verification tests specified in its split document. The agent should run them after implementation and report pass/fail.
8. **Forbidden actions** — Do NOT modify anything in `src/external/`. Do NOT delete or overwrite `docs/main/`. Do NOT retrain or fine-tune the MAISI VAE. Do NOT use `diffusers` (2D-only). Do NOT use `torchcfm` (time convention mismatch).

**Coding standards to embed (from user preferences):**
- Python: type hints on all functions, Google-style docstrings (no usage examples), brief inline comments, atomic function design, low cyclomatic complexity, use established libraries (don't reinvent), logging via Python `logging` module.
- Project-level: OOP with dataclasses, custom exceptions in `src/neuromf/errors/`, submodule organisation, OmegaConf for configs.
- All scientific claims must reference sources (article titles, not just "see paper").
- When writing tests: include the test ID from the phase split (e.g., `test_P0_T1_vae_loads`).

### Deliverable B: `.claude/settings.json`

Configure permissions appropriate for a research ML project:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(src/neuromf/**)",
      "Edit(tests/**)",
      "Edit(configs/**)",
      "Edit(experiments/**)",
      "Edit(scripts/**)",
      "Edit(docs/splits/**)",
      "Edit(CLAUDE.md)",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(pip install *)",
      "Bash(conda *)",
      "Bash(git *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(find *)",
      "Bash(wc *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(mkdir *)",
      "Bash(cp *)",
      "Bash(mv *)",
      "Bash(nvidia-smi)",
      "Bash(wandb *)"
    ],
    "deny": [
      "Edit(src/external/**)",
      "Edit(docs/main/**)",
      "Edit(checkpoints/**)",
      "Edit(datasets/**)",
      "Read(.env*)",
      "Bash(rm -rf *)"
    ]
  }
}
```

Adjust as needed — the key principle is: **external code and master docs are read-only; everything in `src/neuromf/`, `tests/`, `configs/`, `experiments/` is writable.**

### Deliverable C: `.claude/agents/` — Subagents

Create the following subagent files as Markdown with YAML frontmatter:

#### C1. `external-code-reviewer.md`

```yaml
---
name: external-code-reviewer
description: "Reads a paper PDF from docs/papers/ and its corresponding code in src/external/, then produces a structured insight document."
model: sonnet
allowed-tools:
  - Read
  - Bash(find *)
  - Bash(cat *)
  - Bash(head *)
  - Bash(wc *)
  - Edit(docs/papers/**/insights.md)
---
```

**Prompt body for this subagent:**

You are an expert deep learning research engineer tasked with creating a structured code-and-paper insight document. You will be given:
- A paper (PDF or text) located in `docs/papers/{paper_name}/`
- The corresponding code implementation in `src/external/{repo_name}/`

Your output is a single Markdown file: `docs/papers/{paper_name}/insights.md`

Structure the insights document as follows:

1. **Paper Summary** (5–10 sentences): Core contribution, method, key results.
2. **Architecture Details**: Model architecture, key hyperparameters, input/output shapes.
3. **Training Procedure**: Loss function (with exact formula), optimizer, learning rate schedule, data augmentation.
4. **Code Map**: For each key file in the repo, one line describing what it does and which paper section it implements.
5. **Reusable Components**: List specific functions/classes we should wrap or adapt for NeuroMF, with file paths and line numbers.
6. **Gotchas and Caveats**: Any known bugs, incompatibilities (e.g., FlashAttention + JVP), time convention mismatches, or undocumented assumptions.
7. **Key Equations → Code Mapping**: For the 3–5 most important equations in the paper, show the exact code location that implements them.

Be precise. Include file paths and line numbers. Do not speculate — if you cannot find something in the code, say so.

#### C2. `phase-implementer.md`

```yaml
---
name: phase-implementer
description: "Implements a specific project phase. Reads the phase split document, writes code, writes tests, runs verification, and reports results."
model: opus
allowed-tools:
  - Read
  - Edit(src/neuromf/**)
  - Edit(tests/**)
  - Edit(configs/**)
  - Edit(experiments/**)
  - Bash(python *)
  - Bash(pytest *)
  - Bash(pip install *)
  - Bash(nvidia-smi)
  - Bash(git *)
  - Bash(ls *)
  - Bash(find *)
  - Bash(cat *)
  - Bash(mkdir *)
  - Bash(cp *)
---
```

**Prompt body:**

You are an expert deep learning engineer implementing one phase of the NeuroMF project. Before writing any code:

1. Read `CLAUDE.md` for project context and coding standards.
2. Read the phase split document at `docs/splits/phase_{N}.md` (where N is provided in the task).
3. Read ANY insight documents referenced in the phase split (in `docs/papers/*/insights.md`).
4. Read the relevant sections of `docs/main/methodology_expanded.md` referenced in the phase split.

Then:
1. **Plan**: List every file you will create or modify, and what each will contain.
2. **Implement**: Write the code following the coding standards in CLAUDE.md. Every function must have type hints and a docstring. Every module must have a module-level docstring.
3. **Test**: Write the verification tests specified in the phase split. Name them `test_P{N}_T{M}_<description>`. Run them with `pytest tests/ -v -k "P{N}"`.
4. **Report**: Create `experiments/{phase_name}/verification_report.md` with:
   - Test ID | Status (PASS/FAIL) | Details
   - Any issues encountered and how you resolved them
   - If ANY critical test fails, STOP and report — do not proceed to the next phase.

#### C3. `test-runner.md`

```yaml
---
name: test-runner
description: "Runs verification tests for a specific phase and produces a structured pass/fail report."
model: haiku
allowed-tools:
  - Read
  - Bash(pytest *)
  - Bash(python *)
  - Bash(cat *)
  - Edit(experiments/**/verification_report.md)
---
```

**Prompt body:**

You are a test runner. Given a phase number N:
1. Run `pytest tests/ -v -k "P{N}" --tb=short 2>&1 | tee experiments/phase_{N}/test_output.txt`
2. Parse the output.
3. Create/update `experiments/phase_{N}/verification_report.md` with a table: Test ID | Status | Duration | Error (if failed).
4. Print a one-line summary: "Phase {N}: {passed}/{total} tests passed. Gate: OPEN/BLOCKED."

#### C4. `paper-figure-generator.md`

```yaml
---
name: paper-figure-generator
description: "Generates publication-quality figures and tables from experiment results."
model: sonnet
allowed-tools:
  - Read
  - Bash(python *)
  - Edit(experiments/**/figures/**)
  - Edit(experiments/**/tables/**)
---
```

**Prompt body:**

You are a scientific visualisation expert. Given experiment results (CSVs, JSONs, .pt files) in `experiments/`, generate publication-quality figures using matplotlib/seaborn with the following standards:
- Font: serif (Times New Roman or Computer Modern), size 10pt for labels, 8pt for ticks.
- Figsize: single-column (3.5 inches wide) or double-column (7 inches wide).
- Save as both PDF (vector) and PNG (300 DPI).
- Use colorblind-friendly palettes (e.g., from seaborn's "colorblind" or "Set2").
- No titles on figures (titles go in captions in the paper).
- Include error bars (mean ± std) where multiple seeds exist.

### Deliverable D: `.claude/commands/` — Slash Commands

Create these slash commands:

#### D1. `review-external.md`
```markdown
---
description: "Review an external repo's code against its paper and produce an insights document"
allowed-tools:
  - Task
---
Invoke the external-code-reviewer subagent for the paper and repo specified in $ARGUMENTS.
Format: /review-external <paper_folder_name> <external_repo_name>
Example: /review-external meanflow_2025 MeanFlow
```

#### D2. `implement-phase.md`
```markdown
---
description: "Implement a specific project phase end-to-end"
allowed-tools:
  - Task
---
Invoke the phase-implementer subagent for phase $ARGUMENTS.
Before invoking, verify that all previous phases have passing verification reports.
Format: /implement-phase <phase_number>
Example: /implement-phase 0
```

#### D3. `run-tests.md`
```markdown
---
description: "Run verification tests for a phase and report results"
allowed-tools:
  - Task
---
Invoke the test-runner subagent for phase $ARGUMENTS.
Format: /run-tests <phase_number>
```

#### D4. `check-gate.md`
```markdown
---
description: "Check if a phase gate is open (all critical tests pass)"
---
Read `experiments/phase_$ARGUMENTS/verification_report.md`.
Report: Phase $ARGUMENTS gate is OPEN (all critical tests pass) or BLOCKED (list failing tests).
```

### Deliverable E: `docs/splits/` — Phase Split Documents

This is the most important and labour-intensive deliverable. For each of the 9 phases (Phase 0 through Phase 8), create a standalone document `docs/splits/phase_{N}.md` that a `phase-implementer` subagent can consume to implement that phase **without reading the full 900-line technical guide or the 800-line methodology**.

Each phase split must follow this exact template:

```markdown
# Phase {N}: {Phase Title}

**Depends on:** Phase {N-1} (gate must be OPEN)
**Modules touched:** `src/neuromf/{...}`, `tests/{...}`, `configs/{...}`, `experiments/{...}`
**Estimated effort:** {time}

---

## 1. Objective

{2–3 sentences: what this phase achieves and why it matters.}

## 2. Theoretical Background

{Extract the RELEVANT mathematical content from methodology_expanded.md.
Include ONLY the equations and definitions needed for this phase.
For example, Phase 3 needs the MeanFlow Identity, JVP formula, compound prediction,
and loss function — but NOT the LoRA formulation or the SynthSeg evaluation.
Copy the equations verbatim with their equation numbers from the methodology doc.}

## 3. External Code to Leverage

{For each external repo relevant to this phase:
- Repo path: `src/external/{name}/`
- Insights doc: `docs/papers/{name}/insights.md` (if available)
- Specific files to study: list exact file paths
- What to extract/wrap: specific functions, classes, or patterns
- What to AVOID: known incompatibilities or things that don't apply}

## 4. Implementation Specification

{Detailed, file-by-file specification:
- For each file to create:
  - Full path
  - Purpose (1 sentence)
  - Key classes/functions to implement (with signatures)
  - Dependencies (which other project modules it imports)
  - Which external code it wraps or adapts
- For each config file:
  - Full path
  - Key fields and their types/defaults}

## 5. Data and I/O

{What data does this phase consume? Where is it?
What does it produce? Where should outputs go?
Exact paths, tensor shapes, file formats.}

## 6. Verification Tests

{Copy the verification test table from technical_guide.md for this phase.
For each test, add:
- Suggested test file path: `tests/test_phase_{N}.py::test_P{N}_T{M}_<name>`
- Implementation hint (1–2 sentences on how to check the criterion)
- Whether it is CRITICAL (blocks gate) or INFORMATIONAL}

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|

## 7. Expected Outputs

{List every file/artefact this phase should produce:
- Code files (in src/neuromf/)
- Test files (in tests/)
- Config files (in configs/)
- Result files (in experiments/phase_{N}/)
- Visualisations (if any)
- JSON/CSV metrics files (if any)}

## 8. Failure Modes and Mitigations

{Copy relevant rows from the Risk Register in technical_guide.md Appendix C.
Add phase-specific failure modes.}
```

**Now, here is what to extract from the two master documents for each split:**

#### Phase 0 — Environment Bootstrap and VAE Validation
- **From methodology §6**: VAE architecture table, reconstruction quality requirements
- **From technical_guide §2**: All P0 tests, VAE wrapper pseudocode, the preprocessing note
- **External code**: `src/external/model-zoo/` (MAISI weights), `src/external/monai-tutorials/` (VAE tutorial notebook)
- **Data**: IXI at `datasets/IXI/`, weights at `checkpoints/NV-Generate-MR/models/autoencoder_v2.pt`

#### Phase 1 — Latent Pre-computation Pipeline
- **From methodology §2.1, §2.4**: Latent normalisation rationale, channel statistics
- **From technical_guide §3**: All P1 tests, encoding script spec
- **External code**: MONAI transforms from `monai-tutorials/`
- **Data**: IXI NIfTI → `.pt` latents in a new directory

#### Phase 2 — Toy Experiment: MeanFlow on Toroidal Manifold
- **From methodology §1.2–1.5**: MeanFlow Identity, JVP, compound prediction, sampling algorithm — the FULL mathematical pipeline
- **From technical_guide §4**: Toroid construction (both R⁴ flat torus and volumetric), all P2 tests, verification mathematics
- **External code**: `src/external/MeanFlow/` (loss computation), `src/external/MeanFlow-PyTorch/` (alternative reference)
- **Data**: Synthetically generated (no external data needed)

#### Phase 3 — MeanFlow Loss Integration with 3D UNet
- **From methodology §1.3, §1.4, §2.2–2.3**: iMF combined loss, latent x-prediction, the full loss equation
- **From technical_guide §5**: All P3 tests, dual time conditioning spec, JVP computation code pattern, in-place ops workaround, FlashAttention note, memory estimation
- **External code**: `src/external/MeanFlow/` (loss logic), `src/external/model-zoo/` (UNet architecture), `src/external/pMF/` (x-prediction reference)
- **Key: this phase is the hardest engineering challenge (torch.func.jvp compatibility with UNet)**

#### Phase 4 — Training on Brain MRI Latents
- **From methodology §2.5**: Complete training algorithm
- **From technical_guide §6**: Training config YAML, all P4 tests, EMA protocol
- **External code**: None new (uses Phase 3 outputs)
- **Data**: Pre-computed latents from Phase 1

#### Phase 5 — Evaluation Suite
- **From methodology §10**: All metrics (FID, 3D-FID, SSIM, PSNR, SynthSeg, spectral analysis)
- **From technical_guide §7**: All P5 tests, generation script, evaluation script
- **External code**: `src/external/MOTFM/` (evaluation protocol, FID implementation)

#### Phase 6 — Ablation Runs
- **From methodology §9**: All 5 ablation designs with statistical protocols
- **From technical_guide §8**: Ablation matrix (24 training runs), sweep launcher, all P6 tests
- **External code**: None new

#### Phase 7 — LoRA Fine-Tuning for FCD
- **From methodology §5**: LoRA formulation (Eq. 34–36), joint synthesis strategies A and B, decision criterion
- **From methodology §3.4**: Per-channel Lp in joint synthesis (Eq. 28–29)
- **From technical_guide §9**: All P7 tests, mask encoding test
- **Data**: FCD cohort (when available) — for now, create the data pipeline that can accept it

#### Phase 8 — Paper Figures and Tables
- **From methodology §12**: Paper structure, figure/table specs
- **From technical_guide §10**: Figure list (Fig 1–8), table list (Table 1–4)

### Deliverable F: `docs/papers/` — Directory Structure for Insight Documents

Create the following empty directory structure with README placeholders. The user will drop PDFs into these folders, then invoke `/review-external` to generate insights:

```
docs/papers/
├── README.md              # Explains the folder structure
├── meanflow_2025/         # Geng et al., 2025a (MeanFlow, NeurIPS 2025)
│   └── README.md          # Paper: arXiv:2505.13447, Code: src/external/MeanFlow/
├── imf_2025/              # Geng et al., 2025b (Improved MeanFlow)
│   └── README.md          # Paper: arXiv:2512.02012
├── pmf_2026/              # Lu et al., 2026 (Pixel MeanFlow)
│   └── README.md          # Paper: arXiv:2601.22158, Code: src/external/pMF/
├── maisi_2024/            # Guo et al., 2024 (MAISI)
│   └── README.md          # Paper: arXiv:2409.11169, Code: src/external/model-zoo/
├── maisi_v2_2025/         # Zhao et al., 2025 (MAISI-v2)
│   └── README.md          # Paper: arXiv:2508.05772
├── motfm_2025/            # Yazdani et al., 2025 (MOTFM, MICCAI 2025)
│   └── README.md          # Paper: arXiv:2503.00266, Code: src/external/MOTFM/
├── slim_diff_2026/        # Pascual-González et al., 2026 (SLIM-Diff)
│   └── README.md          # Paper: arXiv:2602.03372
├── lora_2022/             # Hu et al., 2022 (LoRA)
│   └── README.md          # Paper: ICLR 2022
└── flow_matching_2023/    # Lipman et al., 2023 (Flow Matching)
    └── README.md          # Paper: ICLR 2023
```

Each README should contain: paper title, arXiv ID, corresponding external code path (if any), and a note saying "Run `/review-external {folder} {repo}` after placing the PDF here."

### Deliverable G: Skeleton Directories and `__init__.py` Files

Create the full `src/neuromf/` package skeleton with empty `__init__.py` files and module-level docstrings, following the structure in `docs/main/technical_guide.md` §1.1. Also create the `experiments/` directory structure with README placeholders.

**Do NOT write implementation code** — only create the directory tree, `__init__.py` files (with docstrings), and `README.md` files in each experiment folder.

---

## 3. Execution Order

Do these in order:
1. **Read** `docs/main/technical_guide.md` completely.
2. **Read** `docs/main/methodology_expanded.md` completely.
3. **Read** the current state of `.claude/` (if anything exists).
4. **Read** the project tree (`find . -type f | head -200`).
5. **Create** Deliverable G (skeleton directories) — so that all paths referenced later exist.
6. **Create** Deliverable B (settings.json).
7. **Create** Deliverable A (CLAUDE.md).
8. **Create** Deliverable C (subagents).
9. **Create** Deliverable D (slash commands).
10. **Create** Deliverable F (papers directory).
11. **Create** Deliverable E (phase splits) — **this is the big one; take your time.**

For Deliverable E, work through phases 0–8 sequentially. For each phase:
- Open `docs/main/technical_guide.md` and find the relevant section.
- Open `docs/main/methodology_expanded.md` and find the relevant equations/theory.
- Cross-reference to extract exactly what the implementing agent needs.
- Write the phase split following the template above.

---

## 4. Quality Criteria — How I Will Judge Your Output

After you finish, I will check:

1. **CLAUDE.md loads correctly:** A new Claude Code session in this repo should immediately understand the project, its phases, and where everything lives.
2. **Settings are correct:** Permissions protect `src/external/` and `docs/main/` from writes while allowing all necessary development operations.
3. **Subagents are well-scoped:** Each subagent has a clear purpose, appropriate model tier (haiku for cheap tasks, sonnet for medium, opus for implementation), and minimal but sufficient tool permissions.
4. **Phase splits are self-contained:** A `phase-implementer` subagent should be able to implement any phase by reading ONLY `CLAUDE.md` + that phase's split document + referenced insight docs, without needing to read the 900-line technical guide. Each split must include all necessary equations, test specifications, file paths, and external code references.
5. **Phase splits are complete:** Every verification test from the technical guide is present in exactly one split. No test is missing; no test is duplicated across phases.
6. **Slash commands work:** `/implement-phase 2` should plausibly kick off the toroid experiment implementation. `/review-external meanflow_2025 MeanFlow` should produce a useful insights document.
7. **Nothing is hallucinated:** Every file path, equation number, and test ID must match what is actually in the master documents.

---

## 5. Final Notes

- **Do not write any implementation code.** No model definitions, no training loops, no data loaders. Only environment, configuration, documentation, and scaffolding.
- **Do not modify `docs/main/`.** Those are the master reference documents.
- **Do not modify `src/external/`.** Those are frozen upstream repos.
- **Be precise with file paths.** The implementing agents will follow them literally.
- **Be thorough with phase splits.** Err on the side of including too much context rather than too little. An implementing agent with perfect context will write better code than one that has to guess.
- **Every equation you include in a phase split must be copy-pasted from methodology_expanded.md** — do not re-derive or rephrase. Consistency across documents is critical for a research project.
- Take your time. This setup will determine the quality of every subsequent implementation session.
