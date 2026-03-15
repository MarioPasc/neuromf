# NeuroMF Code Enhancement Report

**Date:** 2026-03-15
**Scope:** Source code review for modularity, quality, and readiness for Phases 6-8
**Codebase:** 11,497 lines (src/neuromf), 8,694 lines (tests), 6,359 lines (experiments/cli)

---

## A. Architecture Scorecard

### Module Hierarchy

```
src/neuromf/  (11,497 lines across 47 .py files)
├── callbacks/   2,150 lines  — diagnostics, evaluation, sample_collector, performance
├── data/        1,400 lines  — latent_dataset, latent_hdf5, fomo60k, augmentation, toroid
├── errors/          5 lines  — custom exceptions
├── generation/    614 lines  — h5_manager, latent_generator, volume_decoder
├── losses/        408 lines  — meanflow_jvp, lp_loss, combined_loss, smoothness_loss
├── metrics/     2,220 lines  — fid, fid_3d, mmd, ms_ssim, swd, spectral, coverage_density
├── models/        621 lines  — latent_meanflow (LightningModule), toy_mlp, lora, rectflow
├── sampling/      173 lines  — one_step, multi_step, variance_rescaling
├── utils/       2,371 lines  — time_sampler, ema, sample_plots, visualisation, checkpoint
└── wrappers/    1,529 lines  — meanflow_loss (591), maisi_unet (494), maisi_vae, jvp_strategies
```

### Grades

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Modularity | B+ | Clean separation of concerns. Callback-module coupling via boolean flags. |
| Code Quality | A- | Consistent type hints (95%+), Google docstrings, config-driven. |
| Testing | B- | 286 functions across 30 files, but 47% lack critical/informational markers. |
| Extensibility | B | Config-driven ablations. Config loading duplication across CLI scripts. |
| Error Handling | B+ | Good on critical paths, sparse in utility functions. |
| Performance | A- | Gradient checkpointing, lazy HDF5, batched metrics. Minor async opportunities. |

---

## B. High-Priority Enhancements (for Phases 6-8)

### B1. Config Loading Consolidation

**Problem:** At least 7 CLI scripts duplicate the OmegaConf base→main→overlay merge pattern with subtly different layer ordering. Found in:
- `experiments/cli/train.py`
- `experiments/cli/generate_latents.py`
- `experiments/cli/compute_metrics.py`
- `experiments/cli/decode_volumes.py`
- `experiments/cli/decode_samples.py`
- `experiments/cli/prepare_real_test.py`
- `experiments/cli/generate_figures.py`

Some include `generate.yaml`, some don't; some check for double-loading. This is error-prone for Phase 6 ablation scripting where new entry points will be needed.

**Solution:** Create `src/neuromf/config/loader.py`:
```python
def load_merged_config(
    base_dir: Path,
    main_config: str,
    overlays: list[str] | None = None,
) -> DictConfig:
    """Unified config loading with base.yaml → main → overlays merge chain."""
```

**Files:** NEW `src/neuromf/config/__init__.py`, `src/neuromf/config/loader.py`. MODIFY 7+ CLI scripts.
**Impact:** HIGH. Prevents config merge bugs, enables clean ablation scripting.
**Effort:** MEDIUM (2-3h).
**Status:** IMPLEMENTED (2026-03-15). Created `src/neuromf/config/loader.py` and refactored 7 CLI scripts (`train.py`, `generate_latents.py`, `compute_metrics.py`, `decode_volumes.py`, `decode_samples.py`, `generate_figures.py`, `prepare_real_test.py`).

### B2. Diagnostics Result Dataclass

**Problem:** `wrappers/meanflow_loss.py` `_compute_diagnostics()` returns a flat dict with 20+ `diag_*` keys. The dual-head path adds more. Callbacks in `callbacks/diagnostics.py` access these by string key with no type safety or autocomplete.

**Solution:** Create `@dataclass class DiagnosticsResult` with typed fields grouped by category (norms, cosine similarities, per-channel, gradients). Provide `.to_dict()` for backward compatibility with TensorBoard logging.

**Files:** NEW `src/neuromf/losses/diagnostics_types.py`. MODIFY `wrappers/meanflow_loss.py`, `callbacks/diagnostics.py`, `models/latent_meanflow.py`.
**Impact:** MEDIUM. Type safety, IDE support, self-documenting.
**Effort:** MEDIUM (2-3h).

### B3. Input Shape Validation

**Problem:** Functions like `MeanFlowPipeline.forward()`, `sample_one_step()`, `sample_multi_step()` accept tensors without shape validation. Wrong shapes produce cryptic errors deep in the UNet (e.g., spatial dimension mismatch at skip connections).

**Solution:** Create `src/neuromf/utils/validation.py`:
```python
def validate_latent_shape(x: Tensor, channels: int = 4, spatial: int = 48) -> None:
def validate_time_shape(t: Tensor, batch_size: int) -> None:
```
Add calls at entry points of loss forward, sampling functions.

**Files:** NEW `src/neuromf/utils/validation.py`. MODIFY 3-4 entry point files.
**Impact:** LOW-MEDIUM. Earlier error detection, better debugging (especially Phase 7 LoRA).
**Effort:** LOW (1h).

### B4. Callback-Module Decoupling

**Problem:** `models/latent_meanflow.py` has boolean flags:
```python
self._sample_collector_active: bool = False  # Set by SampleCollectorCallback
self._diag_enabled: bool = False             # Set by TrainingDiagnosticsCallback
```
Callbacks set these via `type: ignore[attr-defined]`, creating tight bidirectional coupling.

**Solution:** Replace with formal capability check. Option A: `LatentMeanFlow` accepts `capabilities: set[str]` in `__init__`. Option B: Store state on the Trainer object (Lightning's standard pattern for cross-callback communication).

**Files:** MODIFY `models/latent_meanflow.py`, `callbacks/diagnostics.py`, `callbacks/sample_collector.py`.
**Impact:** LOW. Cleaner architecture, no `type: ignore`, easier isolated testing.
**Effort:** MEDIUM (2h).

---

## C. Medium-Priority Enhancements

### C5. Unified Metric Interface

**Problem:** 8 metric modules (`fid.py`, `fid_3d.py`, `mmd.py`, `ms_ssim_3d.py`, `swd.py`, `spectral.py`, `coverage_density.py`, `ssim_psnr.py`) have different function signatures and return types.

**Solution:** Define a `Protocol` class `MetricComputer` with `def compute(...) -> dict[str, float]`. Adapters for each metric.

**Impact:** Cleaner evaluation callback, pluggable metrics for ablation sweeps.
**Effort:** MEDIUM (3-4h).

### C6. Feature Caching with TTL

**Problem:** Real-data FID features are recomputed each validation epoch despite never changing. `metrics/feature_extractor.py` has HDF5 caching infrastructure but the evaluation callback doesn't leverage it for real features.

**Solution:** Cache real features once at first validation; invalidate only if dataset config changes.

**Impact:** ~30s savings per validation epoch. Significant over 3000-epoch v2 runs.
**Effort:** LOW (1-2h).

### C7. Async Metric Computation

**Problem:** FID computation (feature extraction + Frechet distance) blocks the training loop during validation.

**Solution:** Use `concurrent.futures.ThreadPoolExecutor` to run metric computation in background while GPU resumes training.

**Impact:** Higher GPU utilization during long runs.
**Effort:** MEDIUM (3h, need careful DDP sync).

---

## D. Test Infrastructure Gaps

### Marker Coverage

| Category | Count | % of Total (286) |
|----------|-------|-------------------|
| Tests with `critical` marker | ~74 | 26% |
| Tests with `informational` marker | ~39 | 14% |
| **Tests with no marker** | **~134** | **47%** |
| Tests with phase marker only | ~39 | 14% |

18 of 30 test files have at least one `critical` marker. 12 files have no markers at all.

### Specific Gaps

1. **Phase 4 annotation crisis:** Many Phase 4 tests (`test_latent_meanflow.py`, `test_transfer_loading.py`, `test_sample_collector.py`, `test_diagnostics.py`) have large numbers of unmarked tests. Gate verification (`pytest -k "P4 and critical"`) may miss important tests.

2. **Phase 6 completely unmarked:** `test_v3_enhancements.py` contains Phase 6-related tests with zero markers. Cannot run gate verification.

3. **Phase 7 missing entirely:** No test file exists for LoRA fine-tuning. Phase 7 split specifies 8-10 critical tests.

4. **Empty stubs:**
   - `test_one_step_sampling.py` — 0 test functions
   - `experiments/cli/evaluate.py` — empty
   - `experiments/cli/generate.py` — empty
   - `experiments/cli/run_ablation.py` — empty
   - `experiments/cli/train_lora.py` — empty

5. **P4h fixture bug:** Evaluation callback tests (P4h_T5-T8, P4h_T11-T12) fail due to missing `trainer.world_size = 1` in MagicMock fixture.

---

## E. Known Technical Debt

| Item | Severity | Description |
|------|----------|-------------|
| SWD unreliable | MEDIUM | Sliced Wasserstein Distance diverges from FID (best SWD at epoch 9 vs best FID at epoch 388). Should be deprecated or its usage documented as unreliable before Phase 8 paper. |
| nibabel import | LOW | **FIXED (2026-03-15).** `metrics/__init__.py` now uses `__getattr__` for lazy import of `synthseg_metrics`. |
| bf16 dtype mismatch | LOW | **FIXED (2026-03-15).** `sample_collector.py` now decodes on CPU in `on_fit_end`, bypassing bf16/autocast conflicts from training precision context. |
| Empty CLI stubs | HIGH | 4 experiment scripts are empty (evaluate.py, generate.py, run_ablation.py, train_lora.py). Blocks Phases 6-7 workflow. |
| Config snapshot inconsistency | LOW | `encode_dataset.py` and `validate_vae.py` use different merge chain than training scripts (base + fomo60k vs base + train_meanflow). Intentional but undocumented. |

---

## F. Prioritized Roadmap

Ordered by impact/effort ratio for upcoming Phases 6-8 work.

| # | Enhancement | Impact | Effort | Blocks |
|---|-------------|--------|--------|--------|
| 1 | Config loader consolidation (B1) | HIGH | MEDIUM | Phase 6 ablation scripts |
| 2 | Test marker audit (D) | HIGH | LOW | Gate verification reliability |
| 3 | nibabel lazy import fix (E) | MEDIUM | LOW | Local test reliability |
| 4 | Diagnostics dataclass (B2) | MEDIUM | MEDIUM | Phase 6 analysis quality |
| 5 | Empty CLI stubs (E) | HIGH | HIGH | Phase 6-7 execution |
| 6 | Shape validation (B3) | LOW-MED | LOW | Better debugging |
| 7 | Feature caching (C6) | MEDIUM | LOW | Training throughput |
| 8 | Callback decoupling (B4) | LOW | MEDIUM | Code cleanliness |
| 9 | SWD deprecation (E) | LOW | LOW | Phase 8 paper accuracy |
| 10 | Unified metrics (C5) | MEDIUM | MEDIUM | Evaluation extensibility |

### Recommended Implementation Order

**Before Phase 6:** Items 1-3 (config loader, test markers, nibabel fix)
**During Phase 6:** Items 4, 7 (diagnostics dataclass, feature caching)
**Before Phase 7:** Item 5 (implement empty CLI stubs, especially train_lora.py)
**Before Phase 8:** Item 9 (SWD deprecation decision)

---

## G. Critical Bugs Fixed (2026-03-15)

### G1. NCCL Timeout in DDP on_fit_end (sample_collector.py)

**Root cause:** `on_fit_end` returned early on non-zero ranks while rank 0 did VAE decode + figure generation (30-60s). Non-zero ranks then hit implicit collectives and timed out after NCCL's 30-minute watchdog.

**Fix:** Restructured `on_fit_end` to call `_on_fit_end_rank0()` only on rank 0, followed by a `dist.barrier()` that ALL ranks participate in. Non-zero ranks wait at the barrier instead of returning early.

**Pattern (correct):**
```python
def on_fit_end(self, trainer, pl_module):
    if trainer.is_global_zero:
        self._on_fit_end_rank0(trainer, pl_module)
    if trainer.world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
```

### G2. bf16 Dtype Mismatch in VAE Decode (sample_collector.py)

**Root cause:** During `on_fit_end`, the VAE was loaded on `pl_module.device` (training GPU) where Lightning's bf16 mixed-precision context may still be active. This caused `c10::Half` input vs `float` bias mismatches inside the MAISI VAE.

**Fix:** Changed `_on_fit_end_rank0` to decode on CPU (`torch.device("cpu")`), completely bypassing any autocast context from the training precision plugin. Also removed `torch.cuda.empty_cache()` calls that would fail on CPU.
