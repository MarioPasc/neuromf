# NeuroMF Test Suite — Final Actionable Report
**Date:** 2026-03-04 | **Test Run Duration:** 4:52 | **Total Tests:** 275

---

## Quick Status

| Category | Count | Status | Action |
|----------|-------|--------|--------|
| Passed | 234 | ✓ OK | None |
| Failed (config issue) | 25 | ⚠ Fixable | Update configs |
| Failed (shape mismatch) | 3 | ⚠ Fixable | Investigate preprocessing |
| Failed (OOM) | 4 | ✓ Expected | Skip on local GPU |
| Errored (OOM) | 2 | ✓ Expected | Skip on local GPU |
| Errored (config) | 1 | ⚠ Fixable | Update config |
| Skipped | 6 | ✓ OK | None |

**Gates Status:** All phases BLOCKED due to config issues + shape mismatches

---

## Problem 1: JVP Strategy Validation (25 test failures)

### What Happened
Tests are configured with `prediction_type="x"` + `jvp_strategy="finite_difference"`. The code now validates this combination and rejects it as unsafe. This is **correct behavior**.

### Why It's Forbidden
The x-prediction formulation has a 1/t singularity (division by t). Finite differences amplify this singularity, causing loss explosion during training. This is documented in CLAUDE.md Layer 4.

### The Tests' Intent
The test comments say they want `jvp_strategy="finite_difference"` to avoid `torch.func` overhead on small tensors (for speed on CPU). This is a reasonable goal, but requires switching the prediction type.

### Solution: Two Options

**Option A: Keep x-prediction, switch to exact JVP**
```yaml
prediction_type: "x"
jvp_strategy: "exact"  # Safe combination
```
Slightly slower due to `torch.func`, but mathematically sound.

**Option B: Switch to u-prediction, keep finite_difference JVP**
```yaml
prediction_type: "u"
jvp_strategy: "finite_difference"  # Safe combination, fast
```
Achieves the speed goal while maintaining safety.

### Affected Test Files (6 files, 25+ config instances)

1. **test_diagnostics.py** — 7 failures (P4b tests)
   - Comment: "Uses `jvp_strategy="finite_difference"` to avoid `torch.func` overhead"
   - Config: `_tiny_config()` function, lines 25-95
   - Tests: test_P4b_T1 through T6, T8

2. **test_meanflow_pipeline.py** — 3 failures (P3 tests)
   - Tests: test_P3_T10_bf16, test_P3_adaptive_weighting_normalises_loss
   - Config: Multiple inline configs with `jvp_strategy="finite_difference"`

3. **test_sample_collector.py** — 9 failures (P4 tests)
   - Comment: "Uses `jvp_strategy="finite_difference"` to avoid `torch.func` overhead"
   - Config: `_tiny_config()` function
   - Tests: test_P4_T13 through T21

4. **test_spatial_masking.py** — 2 failures (P4d tests)
   - Tests: test_P4d_T7, test_P4d_T8
   - Config: Inline configs with `jvp_strategy="finite_difference"`

5. **test_transfer_loading.py** — 3 failures (P4 tests)
   - Tests: test_P4_T22, T23, T24
   - Config: Inline configs

6. **test_real_data_augmentation.py** — 1 failure (P4d test)
   - Test: test_P4d_T16_masking_loss_on_real_latents
   - Config: Inline config

### Fix Commands

**Find all instances:**
```bash
grep -n "jvp_strategy.*finite_difference" /home/mpascual/research/code/neuromf/tests/*.py
```

**Recommended fix for each file:**

File 1: `test_diagnostics.py` line 68
```python
# FROM:
"jvp_strategy": "finite_difference",
# TO:
"jvp_strategy": "exact",
```
Or switch to u-prediction:
```python
"prediction_type": "u",
"jvp_strategy": "finite_difference",
```

**Consistency rule:** Once you choose, apply consistently across all test files. Recommend **Option A (exact JVP)** since the production config already uses it.

---

## Problem 2: VAE Output Shape Mismatch (3 test failures)

### What Happened
Tests expect cube shapes (128³, 32³) but VAE outputs non-cube shapes with reduced Y-axis:
- Expected latent: `(1, 4, 32, 32, 32)` → Got: `(1, 4, 32, 30, 32)`
- Expected pixel: `(1, 1, 128, 128, 128)` → Got: `(1, 1, 128, 120, 128)`

The **Y-axis is systematically reduced by 8 voxels** (32→30, 128→120).

### Root Cause Investigation

Likely culprits:
1. Preprocessing pipeline changed (CropForeground, ResizeWithPadOrCrop)
2. Test input random shape differs from expected
3. VAE's own padding/cropping introduced asymmetry

### Affected Tests (3, all in test_maisi_vae_wrapper.py)

1. **test_P0_T2_encode_shape** — Line ~60
   - Input: `torch.rand(1, 1, 128, 128, 128)`
   - Expected latent shape: `(1, 4, 32, 32, 32)`
   - Actual: `(1, 4, 32, 30, 32)`

2. **test_P0_T3_decode_shape** — Line ~75
   - Input latent: `torch.rand(1, 4, 32, 32, 32)`
   - Expected decoded: `(1, 1, 128, 128, 128)`
   - Actual: `(1, 1, 128, 120, 128)`

3. **test_P0_T8_noise_input_low_ssim** — Line ~150
   - Cascading failure due to shape mismatch

### Investigation Steps

1. **Check preprocessing in VAE wrapper:**
   ```bash
   cat /home/mpascual/research/code/neuromf/src/neuromf/wrappers/maisi_vae.py | grep -A20 "def encode"
   ```

2. **Check test input creation:**
   ```bash
   grep -A5 "torch.rand.*128" /home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py
   ```

3. **Verify VAE config:**
   ```bash
   grep -r "num_splits\|ResizeWithPadOrCrop\|CropForeground" /home/mpascual/research/code/neuromf/configs/
   ```

4. **Check if 192³ migration affected 128³:**
   ```bash
   cat /home/mpascual/research/code/neuromf/docs/data/resolution_analysis.md
   ```

### Likely Fix

The Y-axis reduction suggests the VAE is applying asymmetric cropping. Either:
- Update test expectations to match actual VAE behavior
- Fix VAE preprocessing to output symmetric shapes
- Change test inputs to match VAE's expected shapes

**Action:** Investigate whether this is intentional (VAE design) or a bug.

---

## Problem 3: GPU Out-of-Memory (6 test failures/errors)

### What Happened
Tests are trying to encode/decode 128³ MRI volumes on RTX 4060 (8GB VRAM), exceeding memory.

### Affected Tests (6 total)

**Errors (2):**
- `test_P0_T4_ssim_above_threshold`
- `test_P0_T5_psnr_above_threshold`

**Failures (4):**
- `test_P0_T10_scale_factor_matters`
- `test_P0_T11_latent_statistics`
- `test_P0_T12_ood_blank_input`
- `test_P1_T7_round_trip_ssim`

### Why This Is Expected

The project's policy (from CLAUDE.md §9):
> Local laptop (RTX 4060 8GB): Code development, unit tests with **mock data**, statistical analysis, figure generation, git operations. **No GPU-heavy tasks** — 192³ volumes exceed 8GB VRAM even with num_splits=6.

These tests violate that policy by trying to do actual VAE operations locally.

### Solution

**Option 1: Skip tests on local GPU (recommended)**
```python
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20e9,
    reason="Requires A100 with 40GB VRAM"
)
def test_P0_T4_ssim_above_threshold():
    ...
```

**Option 2: Add CPU fallback**
```python
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20e9:
    pytest.skip("Test requires Picasso A100")
```

### Files to Update
- `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py`
- `/home/mpascual/research/code/neuromf/tests/test_latent_dataset.py`
- `/home/mpascual/research/code/neuromf/tests/conftest.py` (add helper fixture)

---

## Problem 4: Mock Testing Bug (1 test failure)

### Test
`test_metrics_p5.py::test_P5_T9_feature_extractor_mock`

### Error
```
TypeError: expected Tensor as element 0 in argument 0, but got MagicMock
```

### Root Cause
Test is passing `MagicMock` to a real tensor operation (likely `.to(device)` or similar).

### Fix

Find the test:
```bash
grep -A15 "test_P5_T9_feature_extractor_mock" /home/mpascual/research/code/neuromf/tests/test_metrics_p5.py
```

Either:
1. Use a real tensor instead of mock
2. Mark test as skip: `@pytest.mark.skip(reason="Mock infrastructure not ready")`

---

## Recommended Fix Order

### Phase 1: Critical (unblock phase gates)
1. Fix JVP strategy in all 6 test files (25+ instances)
2. Investigate VAE shape mismatches (3 tests)
3. Fix mock test (1 test)

Estimated time: 30 minutes

### Phase 2: Environmental
4. Add VRAM checks to skip GPU tests locally (6 tests)

Estimated time: 15 minutes

### Phase 3: Verification
5. Re-run full suite
6. Confirm all phase gates open

---

## Commands for Each Fix

### Fix JVP Strategy
```bash
# View current config
grep -B2 -A2 "jvp_strategy.*finite_difference" /home/mpascual/research/code/neuromf/tests/test_diagnostics.py

# Option A: Switch to exact JVP (safe with x-pred)
sed -i 's/"jvp_strategy": "finite_difference"/"jvp_strategy": "exact"/g' \
  /home/mpascual/research/code/neuromf/tests/test_diagnostics.py

# Apply to all files
for f in test_diagnostics.py test_meanflow_pipeline.py test_sample_collector.py \
         test_spatial_masking.py test_transfer_loading.py test_real_data_augmentation.py; do
  sed -i 's/jvp_strategy": "finite_difference"/jvp_strategy": "exact"/g' \
    /home/mpascual/research/code/neuromf/tests/$f
done
```

### Verify fixes
```bash
# Verify no more finite_difference + x-pred
grep -r "finite_difference" /home/mpascual/research/code/neuromf/tests/*.py | grep -v "skip\|comment"

# Re-run tests
~/.conda/envs/neuromf/bin/python -m pytest /home/mpascual/research/code/neuromf/tests/ -v --tb=short -k "P4 or P3"
```

### Investigate VAE shapes
```bash
# Check actual VAE behavior
~/.conda/envs/neuromf/bin/python -c "
import torch
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/base.yaml')
vae = MAISIVAEWrapper(cfg)

x = torch.randn(1, 1, 128, 128, 128)
latent = vae.encode(x)
print(f'Input: {x.shape} -> Latent: {latent.shape}')

x_recon = vae.decode(latent)
print(f'Latent: {latent.shape} -> Recon: {x_recon.shape}')
"
```

---

## Summary Table

| Issue | Count | Severity | Fix Time | Blocks |
|-------|-------|----------|----------|--------|
| JVP Strategy | 25 | High | 5 min | P3, P4, P4b, P4d |
| VAE Shapes | 3 | High | 20 min | P0, P1 |
| Mock Bug | 1 | Medium | 5 min | P5 |
| OOM Tests | 6 | Low | 10 min | N/A |
| **TOTAL** | **35** | — | **40 min** | **All phases** |

---

## Files to Modify

Priority order:

1. `/home/mpascual/research/code/neuromf/tests/test_diagnostics.py` (7 instances)
2. `/home/mpascual/research/code/neuromf/tests/test_sample_collector.py` (9 instances)
3. `/home/mpascual/research/code/neuromf/tests/test_meanflow_pipeline.py` (3+ instances)
4. `/home/mpascual/research/code/neuromf/tests/test_transfer_loading.py` (3 instances)
5. `/home/mpascual/research/code/neuromf/tests/test_spatial_masking.py` (2 instances)
6. `/home/mpascual/research/code/neuromf/tests/test_real_data_augmentation.py` (1 instance)
7. `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py` (shape assertions)
8. `/home/mpascual/research/code/neuromf/tests/test_metrics_p5.py` (mock test)

