# Phase 4 — MAISI UNet Wrapper Tests Verification Report

**Date:** 2026-02-21
**Test Suite:** `tests/test_maisi_unet_wrapper.py`
**Total Tests:** 8
**Status:** ALL PASS

## Test Results

| Test ID | Test Name | Status | Duration | Notes |
|---------|-----------|--------|----------|-------|
| P3-T1 | `test_P3_T1_unet_dual_time_conditioning` | PASS | ~1.7s | UNet accepts (r, t) conditioning without error |
| P3-T2 | `test_P3_T2_output_shape_matches_input` | PASS | ~1.7s | Output shape (B, C, D, H, W) matches input latent shape |
| P3-T11 | `test_P3_T11_full_size_forward_pass` | PASS | ~9.9s | Full-size forward pass at (1, 4, 48, 48, 48) validates on CPU |
| P3-Inf | `test_P3_u_from_x_conversion` | PASS | <0.1s | u_from_x correctly converts x-prediction to velocity: u = (z_t - x_pred) / t |
| P3-Inf | `test_P3_u_from_x_t_min_clamping` | PASS | <0.1s | t_min clamping prevents division by zero at small t values |
| **P4-Inf** | **`test_P4_t_h_conditioning_mode`** | **PASS** | **~1.7s** | **(t, h) conditioning mode produces valid output; h_embed MLP created, no r_embed** |
| **P4-Inf** | **`test_P4_t_h_differs_from_h_only`** | **PASS** | **~1.7s** | **(t, h) conditioning differs from h-only (adds absolute time t information)** |
| P4g-T10 | `test_P4g_T10_h_conditioning_differs_from_dual` | PASS | ~1.7s | h-conditioning produces different embeddings than dual for r ≠ t |

## Summary

- **Total:** 8 tests
- **Passed:** 8 (100%)
- **Failed:** 0
- **Total Duration:** ~18.5 seconds

## Key Validations

### New Conditioning Modes Verified
1. **`conditioning_mode="t_h"`** — Conditions on both absolute time `t` and interval width `h=t-r`
   - Creates `h_embed` MLP, no `r_embed`
   - Produces valid finite outputs
   - Distinct from `h`-only conditioning (adds absolute time information)

2. **`conditioning_mode="h"`** — Conditions on interval width `h=t-r` only
   - Produces different outputs than dual `(r, t)` conditioning
   - Valid for iMF with self-consistent field

3. **`conditioning_mode="dual"`** (legacy) — Conditions on separate `(r, t)` values
   - Still supported, produces different embeddings than h-only

### Backward Compatibility
- All existing tests (P3-T1, P3-T2, P3-T11, u-conversion) pass without modification
- No regressions detected
- Zero-init conv handling verified in new tests

## Implications for Phase 4 Training

With new conditioning modes validated:
- **v3_config:** Can safely use `conditioning_mode: "h"` with u-prediction
- **iMF dual-head:** Will condition UNet on h=t-r, enabling better self-consistency
- **Inference:** 1-NFE sampling can use h-conditioning with h=1.0 for full interval [0,1]

## Next Steps
1. Commit changes to version control
2. Prepare Phase 4 v3 training with `conditioning_mode: "h"` and dual-head enabled
3. Monitor loss curves to verify MF loss stability improvements
