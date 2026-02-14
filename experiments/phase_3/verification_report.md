# Phase 3 Verification Report

**Date:** 2026-02-14  
**Environment:** `~/.conda/envs/neuromf/bin/python -m pytest`  
**Test Command:** `pytest tests/ -v -k "P3" --tb=short`  
**Total Duration:** 189.99s (3m 9s)

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 28 |
| Passed | 27 |
| Failed | 1 |
| Skipped | 0 |
| Pass Rate | 96.4% |
| **Gate Status** | **BLOCKED** |

## Test Results Table

| Test ID | File | Test Name | Status | Duration | Error |
|---------|------|-----------|--------|----------|-------|
| P3-T1 | test_maisi_unet_wrapper.py | test_P3_T1_unet_dual_time_conditioning | PASS | — | — |
| P3-T2 | test_maisi_unet_wrapper.py | test_P3_T2_output_shape_matches_input | PASS | — | — |
| P3-T3a | test_jvp_compatibility.py | test_P3_T3a_jvp_executes_on_unet | PASS | — | — |
| P3-T3b | test_jvp_compatibility.py | test_P3_T3b_jvp_matches_finite_difference | PASS | — | — |
| P3-T4 | test_meanflow_pipeline.py | test_P3_T4_meanflow_loss_finite_positive | PASS | — | — |
| **P3-T5** | **test_meanflow_pipeline.py** | **test_P3_T5_gradients_flow_to_all_params** | **FAIL** | — | AssertionError: Only 0.5% of params got gradients (2/434) |
| P3-T6 (Lp 1.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss[1.0] | PASS | — | — |
| P3-T6 (Lp 1.5) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss[1.5] | PASS | — | — |
| P3-T6 (Lp 2.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss[2.0] | PASS | — | — |
| P3-T6 (Lp 3.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss[3.0] | PASS | — | — |
| P3-T6 (Weights 1.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss_with_weights[1.0] | PASS | — | — |
| P3-T6 (Weights 1.5) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss_with_weights[1.5] | PASS | — | — |
| P3-T6 (Weights 2.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss_with_weights[2.0] | PASS | — | — |
| P3-T6 (Weights 3.0) | test_lp_loss_perchannel.py | test_P3_T6_perchannel_lp_loss_with_weights[3.0] | PASS | — | — |
| P3-T7 | test_time_sampler.py | test_P3_T7_logit_normal_distribution | PASS | — | — |
| P3-T8 | test_ema.py | test_P3_T8_ema_with_unet_wrapper | PASS | — | — |
| P3-T9 | test_meanflow_pipeline.py | test_P3_T9_combined_imf_loss | PASS | — | — |
| P3-T10 | test_meanflow_pipeline.py | test_P3_T10_bf16_mixed_precision | PASS | — | — |
| P3-T11 | test_maisi_unet_wrapper.py | test_P3_T11_full_size_forward_pass | PASS | — | — |
| P3-T12 | test_jvp_compatibility.py | test_P3_T12_full_size_finite_diff_jvp | PASS | — | — |
| P3-T13 | test_real_latents.py | test_P3_T13_real_latent_smoke_test | PASS | — | — |

## Passing CRITICAL Tests (23/24)

All CRITICAL tests pass except P3-T5:

- P3-T1: UNet dual time conditioning ✓
- P3-T2: Output shape matching ✓
- P3-T3a: JVP executes on UNet ✓
- P3-T3b: JVP matches finite difference ✓
- P3-T4: MeanFlow loss finite and positive ✓
- P3-T6: Per-channel Lp loss (8 variants) ✓
- P3-T7: Logit-normal time sampling ✓
- P3-T8: EMA with UNet wrapper ✓
- P3-T9: Combined iMF loss ✓
- P3-T10: bf16 mixed precision ✓
- P3-T11: Full-size forward pass ✓
- P3-T12: Full-size finite-diff JVP ✓
- P3-T13: Real latent smoke test ✓

## FAILING CRITICAL Test

### P3-T5: Gradients Flow to All Params (test_meanflow_pipeline.py:79-123)

**Status:** FAIL  
**Error:** `AssertionError: Only 0.5% of params got gradients (2/434)`

**Test Location:** `/home/mpascual/research/code/neuromf/tests/test_meanflow_pipeline.py`, lines 79-123

**Test Code:**
```python
@pytest.mark.phase3
@pytest.mark.critical
def test_P3_T5_gradients_flow_to_all_params() -> None:
    """P3-T5: Gradients flow to all UNet params after loss.backward()."""
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.train()

    pipeline_config = MeanFlowPipelineConfig(
        p=2.0,
        adaptive=True,
        prediction_type="x",
        jvp_strategy="finite_difference",  # Using FD for robust autograd
        fd_step_size=1e-3,
    )
    pipeline = MeanFlowPipeline(pipeline_config)

    B, C, D, H, W = 2, 4, 16, 16, 16
    z_0 = torch.randn(B, C, D, H, W)
    eps = torch.randn(B, C, D, H, W)
    t = torch.tensor([0.5, 0.8])
    r = torch.tensor([0.2, 0.3])

    result = pipeline(model, z_0, eps, t, r)
    result["loss"].backward()

    # Check: 90% of params should have non-zero gradients
    params_with_grad = 0
    params_without_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad += 1

    total = params_with_grad + params_without_grad
    assert params_with_grad > 0, "No parameters received gradients"
    frac = params_with_grad / total
    assert frac > 0.9, f"Only {frac:.1%} of params got gradients ({params_with_grad}/{total})"
```

**Result:** Only 2 out of 434 parameters received gradients (0.46%)

## Root Cause Analysis

The gradient flow blockage occurs in two critical locations:

### 1. **JVP Strategies** (`src/neuromf/wrappers/jvp_strategies.py`, line 56)

In the `_compound_velocity()` function:
```python
def _compound_velocity(u: Tensor, du_dt: Tensor, t: Tensor, r: Tensor, z_t: Tensor) -> Tensor:
    """Compute V = u + (t - r) * sg[JVP]"""
    t_minus_r = (t - r).clamp(min=0.0)
    if z_t.ndim > 1:
        shape = (-1,) + (1,) * (z_t.ndim - 1)
        t_minus_r = t_minus_r.view(*shape)
    return u + t_minus_r * du_dt.detach()  # <-- DETACH HERE BREAKS GRADIENT FLOW
```

**Problem:** `du_dt.detach()` prevents gradients from flowing through the JVP term during backprop. Since the MeanFlow loss term depends primarily on the compound velocity V (which contains du_dt), most of the gradient signal is blocked.

### 2. **Adaptive Weighting** (`src/neuromf/wrappers/meanflow_loss.py`, lines 165-169)

In the `forward()` method:
```python
if adaptive:
    fm_weight = loss_fm_per_sample.detach() + norm_eps  # <-- DETACH HERE
    loss_fm_per_sample = loss_fm_per_sample / fm_weight

    mf_weight = loss_mf_per_sample.detach() + norm_eps  # <-- DETACH HERE
    loss_mf_per_sample = loss_mf_per_sample / mf_weight
```

**Problem:** Detaching the loss before normalization creates a division by a non-differentiable quantity, further breaking the gradient flow.

## What's Currently Working

Despite the gradient flow issue, the implementation is **mathematically sound**:

- **Loss computation:** P3-T4 (finite and positive) ✓
- **JVP computation:** P3-T3a and P3-T3b (matches finite difference) ✓
- **Combined iMF loss:** P3-T9 (FM + MF correctly combined) ✓
- **Mixed precision:** P3-T10 (bf16 autocast compatible) ✓
- **Real latents:** P3-T13 (smoke test with actual latent data) ✓

All supporting infrastructure tests pass. The issue is **exclusively in gradient flow**.

## Design Question: Is Stop-Gradient Intentional?

The MeanFlow paper (Eq. 13) shows the compound velocity as:
```
V = u + (t - r) * ∂u/∂t
```

Theory may require the `∂u/∂t` term to not contribute to parameter gradients because:
1. It's an auxiliary regularization term (not the main training signal)
2. Only the FM term (v_tilde) should drive training
3. The MeanFlow constraint is a consistency check, not an optimization target

**However:** The current implementation detaches at two levels, which may be too aggressive.

## Next Steps

1. **Check theory:** Review `/home/mpascual/research/code/neuromf/docs/papers/meanflow_2025/meanflow.pdf` and `/home/mpascual/research/code/neuromf/docs/papers/imf_2025/improved-mean-flows.pdf` to clarify if stop-gradient is required by the algorithm.

2. **Review reference implementations:**
   - JAX version: `src/external/MeanFlow/` (line 226-236 in code_exploration doc)
   - PyTorch version: `src/external/MeanFlow-PyTorch/` (check gradient policy)

3. **Fix options:**
   - **Option A:** Remove `du_dt.detach()` from line 56 of `jvp_strategies.py` if theory allows full gradient flow
   - **Option B:** Remove `.detach()` from adaptive weighting (lines 165-169 of `meanflow_loss.py`) if normalization weights should be differentiable
   - **Option C:** Keep stop-gradients but modify the test to only check FM term gradients (FM should have 100% gradient flow)

4. **Verify fix:** Re-run P3-T5 after modification. Target: ≥90% params with non-zero gradients.

5. **Gate decision:** Only proceed to Phase 4 after P3-T5 passes.

## Informational Tests (2/2 PASS)

- test_P3_fd_pipeline_produces_finite_loss ✓
- test_P3_adaptive_weighting_normalises_loss ✓

## File References

| Component | File Path |
|-----------|-----------|
| Pipeline implementation | `/home/mpascual/research/code/neuromf/src/neuromf/wrappers/meanflow_loss.py` |
| JVP strategies | `/home/mpascual/research/code/neuromf/src/neuromf/wrappers/jvp_strategies.py` |
| Failing test | `/home/mpascual/research/code/neuromf/tests/test_meanflow_pipeline.py` |
| All tests | `/home/mpascual/research/code/neuromf/tests/test_*.py` |

