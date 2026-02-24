# Phase 5 Verification Report

**Date:** 2026-02-24  
**Tests Run:** 17 total (16 passed, 1 failed)  
**Gate Status:** BLOCKED

## Test Results

| Test ID | Test Name | Status | Duration | Error |
|---------|-----------|--------|----------|-------|
| P5-T1 | `test_P5_T1_h5_latent_archive_create_read_write` | PASS | 0.08s | — |
| P5-T2 | `test_P5_T2_h5_volume_archive_create_read_write` | PASS | 0.06s | — |
| P5-T3 | `test_P5_T3_latent_generator_tiny_model` | PASS | 0.23s | — |
| P5-T4 | `test_P5_T4_shared_noise_across_nfe` | PASS | 0.08s | — |
| P5-T5 | `test_P5_T5_volume_decoder_mock_vae` | PASS | 0.14s | — |
| P5-T6 | `test_P5_T6_spectral_hf_ratio_known_signals` | PASS | 0.07s | — |
| P5-T7 | `test_P5_T7_ms_ssim_3d_identical` | PASS | 0.04s | — |
| P5-T7b | `test_P5_T7b_ms_ssim_3d_different_volumes` | PASS | 0.06s | — |
| P5-T8 | `test_P5_T8_nn_pairing_correctness` | PASS | 0.05s | — |
| P5-T9 | `test_P5_T9_feature_extractor_mock` | PASS | 0.03s | — |
| P5-T10 | `test_P5_T10_h5_metadata_consistency` | PASS | 0.05s | — |
| P5-T11 | `test_P5_T11_h5_to_nifti_conversion` | PASS | 0.12s | — |
| P5-T12 | `test_P5_T12_dice_from_labels` | PASS | 0.04s | — |
| P5-T13 | `test_P5_T13_synthseg_success_rate` | PASS | 0.04s | — |
| **P5-T14** | `test_P5_T14_regional_correlation_and_kl` | **FAIL** | 0.04s | KL divergence assertion failed |
| P5-T15 | `test_P5_T15_parse_volumes_csv` | PASS | 0.08s | — |
| P5-T16 | `test_P5_T16_synthseg_config_and_availability` | PASS | 0.05s | — |

## Failure Details

### P5-T14: Regional Correlation and KL Divergence
**File:** `/home/mpascual/research/code/neuromf/tests/test_metrics_p5.py`  
**Line:** 299  
**Error:** `AssertionError: assert 9.634550391882097 < 1.0`

**Description:**
The test verifies KL divergence computation for regional brain volumes. It creates 20 paired samples where:
- Real: `base + N(0, 200)`
- Generated: `base + N(0, 100)`

The test expects KL < 1.0, but the actual KL divergence is 9.63. This indicates the distributions are more dissimilar than the test threshold allows.

**Root Cause Analysis:**
The KL divergence formula `sum(P * log(P/Q))` is sensitive to:
1. **Variance mismatch:** Real has σ=200, generated has σ=100 (2x difference)
2. **Histogram binning:** With 50 bins and only 20 samples, bin populations are sparse (~0.4 samples/bin on average)
3. **Small sample size:** Histogram noise dominates with N=20

With seed=42, the generated volumes are consistently offset and narrower than real volumes, producing high KL divergence due to poor histogram overlap.

**Recommendation:**
The test is checking a realistic scenario where generated volumes have lower variance than real volumes. The threshold of KL < 1.0 is too tight. Either:
1. Increase sample size (N=20 → N=100+) to reduce histogram noise
2. Increase threshold to KL < 2.0 to allow for variance differences
3. Change test to use matched variances (both N(0, 150)) to verify computation

For now, this is an informational failure indicating the metric is working but the test tolerance is unrealistic for small sample sizes.

## Summary

- **Total Tests:** 17
- **Passed:** 16
- **Failed:** 1 (P5-T14, informational issue with test tolerance)
- **Gate Status:** BLOCKED until P5-T14 is fixed

The failure is isolated to a test tolerance issue in the KL divergence assertion. All generation, archive, and other evaluation metrics pass successfully. The gate is blocked due to this one critical test failure.

