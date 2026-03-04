# NeuroMF Test Suite Report
**Date:** 2026-03-04
**Total Tests:** 275
**Results:** 234 PASSED, 32 FAILED, 6 SKIPPED, 3 ERRORS

---

## Executive Summary

The test suite has **critical issues** preventing phase gates from opening:

1. **JVP Strategy Validation (25 failures):** Tests using `jvp_strategy="finite_difference"` with `prediction_type="x"` are correctly rejected by the validation gate in `MeanFlowPipelineConfig.__post_init__()`. This is **correct behavior** — the code correctly prevents the forbidden x-pred + FD-JVP combination that causes 1/t singularity explosion.

2. **VAE Output Shape Mismatch (3 failures):** P0 tests expect shapes like (1,4,32,32,32) but VAE is outputting (1,4,32,30,32). This indicates a preprocessing or cropping issue.

3. **Memory Pressure (6 OOM errors):** Tests running VAE operations on RTX 4060 (8GB) are hitting OOM. This is expected for GPU-heavy tests that should only run on Picasso.

4. **Mock Testing Issue (1 failure):** P5 test trying to pass MagicMock to real tensor operation.

---

## Detailed Failure Analysis

### Category 1: JVP Strategy Validation (25 FAILED) — **EXPECTED & CORRECT**

**Root Cause:** Tests are using `jvp_strategy="finite_difference"` with `prediction_type="x"`, which violates the critical rule documented in CLAUDE.md Layer 4:

> Critical rule: x-pred + exact JVP = stable; u-pred + FD-JVP = stable; x-pred + FD-JVP = explosion (1/t singularity amplified by finite differences).

**Validation Code** (`src/neuromf/wrappers/meanflow_loss.py:46`):
```python
if self.prediction_type == "x" and self.jvp_strategy == "finite_difference":
    raise ValueError(
        "x-prediction + finite_difference JVP is forbidden: the 1/t singularity in x-pred "
        "is amplified by finite differences, causing loss explosion. Use jvp_strategy='exact' "
        "with prediction_type='x', or prediction_type='u' with jvp_strategy='finite_difference'."
    )
```

**Affected Tests (25 total):**
- `test_diagnostics.py` (7 failures: T1-T6, T8)
- `test_meanflow_pipeline.py` (2 failures: T10, adaptive_weighting)
- `test_sample_collector.py` (9 failures: T13-T21)
- `test_spatial_masking.py` (2 failures: T7-T8)
- `test_transfer_loading.py` (3 failures: T22-T24)
- `test_real_data_augmentation.py` (1 failure: T16)
- `test_meanflow_pipeline.py::test_P3_fd_pipeline_produces_finite_loss` (1 ERROR)

**Solution:** Update test configs to use either:
- Option A: `prediction_type="x"` + `jvp_strategy="exact"` (current best)
- Option B: `prediction_type="u"` + `jvp_strategy="finite_difference"` (faster for small tensors)

Files requiring updates:
- `/home/mpascual/research/code/neuromf/tests/test_diagnostics.py`
- `/home/mpascual/research/code/neuromf/tests/test_meanflow_pipeline.py`
- `/home/mpascual/research/code/neuromf/tests/test_sample_collector.py`
- `/home/mpascual/research/code/neuromf/tests/test_spatial_masking.py`
- `/home/mpascual/research/code/neuromf/tests/test_transfer_loading.py`
- `/home/mpascual/research/code/neuromf/tests/test_real_data_augmentation.py`

---

### Category 2: VAE Output Shape Mismatch (3 FAILED) — **UNEXPECTED**

**Affected Tests:**
1. `test_maisi_vae_wrapper.py::test_P0_T2_encode_shape`
2. `test_maisi_vae_wrapper.py::test_P0_T3_decode_shape`
3. `test_maisi_vae_wrapper.py::test_P0_T8_noise_input_low_ssim`

**Details:**

Test expects: `(1, 4, 32, 32, 32)` (latent)
Actual output: `(1, 4, 32, 30, 32)` — **Y-axis reduced to 30 instead of 32**

Test expects: `(1, 1, 128, 128, 128)` (decoded pixel)
Actual output: `(1, 1, 128, 120, 128)` — **Y-axis reduced to 120 instead of 128**

**Root Cause:** Likely the 128³ → 192³ resolution change or preprocessing pipeline differences. The Y-axis is systematically being reduced.

**Impacted File:** 
- `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py` (lines with shape assertions)

---

### Category 3: GPU Out-of-Memory (6 OOM ERRORS) — **EXPECTED FOR LOCAL GPU**

These tests should only run on Picasso (A100 40GB), not on local RTX 4060 (8GB).

**Affected Tests (6 total):**
- `test_maisi_vae_wrapper.py::test_P0_T4_ssim_above_threshold` (ERROR)
- `test_maisi_vae_wrapper.py::test_P0_T5_psnr_above_threshold` (ERROR)
- `test_maisi_vae_wrapper.py::test_P0_T10_scale_factor_matters` (FAILED OOM)
- `test_maisi_vae_wrapper.py::test_P0_T11_latent_statistics` (FAILED OOM)
- `test_maisi_vae_wrapper.py::test_P0_T12_ood_blank_input` (FAILED OOM)
- `test_latent_dataset.py::test_P1_T7_round_trip_ssim` (FAILED OOM)

**Solution:** Mark these with `@pytest.mark.skipif(not have_enough_vram, ...)` or run only on Picasso via CI.

**Impacted Files:**
- `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py`
- `/home/mpascual/research/code/neuromf/tests/test_latent_dataset.py`

---

### Category 4: Mock Testing Bug (1 FAILED) — **CODE BUG**

**Test:** `test_metrics_p5.py::test_P5_T9_feature_extractor_mock`

**Error:** `TypeError: expected Tensor as element 0 in argument 0, but got MagicMock`

**Root Cause:** Test is passing `MagicMock` to `feature_extractor()` which expects real tensors.

**Impacted File:**
- `/home/mpascual/research/code/neuromf/tests/test_metrics_p5.py` (T9)

**Solution:** Either:
- Remove `.to()` call or other tensor operations from mock
- Use a real small tensor instead of mock
- Mark as `@pytest.mark.skip` if testing infrastructure not ready

---

## Summary by Test File

| Test File | Failed | Passed | Root Cause |
|-----------|--------|--------|-----------|
| `test_diagnostics.py` | 7 | 1 | x-pred + FD-JVP forbidden (correct validation) |
| `test_maisi_vae_wrapper.py` | 8 | 3 | Shape mismatch (3) + OOM (5) |
| `test_latent_dataset.py` | 1 | 6 | OOM on 128³ round-trip |
| `test_meanflow_pipeline.py` | 3 | 5 | x-pred + FD-JVP forbidden (2) + other |
| `test_sample_collector.py` | 9 | 11 | x-pred + FD-JVP forbidden (9) |
| `test_spatial_masking.py` | 2 | 2 | x-pred + FD-JVP forbidden |
| `test_transfer_loading.py` | 3 | 8 | x-pred + FD-JVP forbidden |
| `test_real_data_augmentation.py` | 1 | 5 | x-pred + FD-JVP forbidden |
| `test_metrics_p5.py` | 1 | 4 | Mock passing to real tensor op |
| **All others** | **0** | **187** | PASS |

---

## Required Actions

### IMMEDIATE (Blocking Phase Gates)

1. **Fix JVP Strategy in Test Configs (25 tests)**
   - Search for all `jvp_strategy: finite_difference` in tests
   - Update to `jvp_strategy: exact` if using `prediction_type: x`
   - Alternatively, switch to `prediction_type: u` with FD
   - Command to find:
     ```bash
     grep -r "jvp_strategy.*finite_difference" /home/mpascual/research/code/neuromf/tests/
     ```

2. **Fix VAE Shape Assertions (3 tests)**
   - Investigate preprocessing pipeline changes
   - Update test input shapes or fix preprocessing to match expected 32³/128³
   - Check if 192³ resolution change affected these tests
   - Review: `/home/mpascual/research/code/neuromf/src/neuromf/data/mri_preprocessing.py`

3. **Fix Mock Test (1 test)**
   - Update `test_metrics_p5.py::test_P5_T9_feature_extractor_mock`
   - Either use real tensor or mark skip

### OPTIONAL (Performance & CI)

4. **Skip GPU-Heavy Tests on Local RTX 4060 (6 tests)**
   - Add VRAM checks to conftest.py
   - Mark tests with `@pytest.mark.skipif(device_vram < 20GB, ...)`
   - Ensure tests run on Picasso CI

---

## Test Health Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Pass Rate | 234/275 (85.1%) | ACCEPTABLE |
| Critical Failures (blocking gates) | 25 | MUST FIX |
| Environmental Failures (OOM) | 6 | EXPECTED |
| Bugs | 1 | MUST FIX |
| Skipped (intentional) | 6 | OK |

---

## Commands to Fix Each Category

### Fix JVP Strategy (25 tests)
```bash
# Find all offending configs
grep -r "jvp_strategy.*finite_difference" /home/mpascual/research/code/neuromf/tests/

# Check which tests use x-prediction
grep -B5 "jvp_strategy.*finite_difference" /home/mpascual/research/code/neuromf/tests/*.py | grep -E "prediction_type|jvp_strategy"
```

### Fix VAE Shapes (3 tests)
```bash
# Check current preprocessing in tests
grep -A10 "torch.rand" /home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py

# Review preprocessing changes
grep -r "ResizeWithPadOrCropd\|CropForegroundd" /home/mpascual/research/code/neuromf/src/
```

### Re-run Tests (after fixes)
```bash
# All tests
~/.conda/envs/neuromf/bin/python -m pytest /home/mpascual/research/code/neuromf/tests/ -v --tb=short

# Just failing categories
~/.conda/envs/neuromf/bin/python -m pytest /home/mpascual/research/code/neuromf/tests/test_diagnostics.py -v
~/.conda/envs/neuromf/bin/python -m pytest /home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py::test_P0_T2_encode_shape -v
```

