# Test Report Index — 2026-03-04

Full test suite run on NeuroMF project after YAML config and SLURM script changes.

## Summary

- **Date:** 2026-03-04
- **Duration:** 4:52 (292 seconds)
- **Total Tests:** 275
- **Passed:** 234 (85.1%)
- **Failed:** 32
- **Errors:** 3
- **Skipped:** 6

**All phase gates are currently BLOCKED** due to 4 categories of failures requiring fixes.

---

## Reports Generated

### 1. Main Technical Report
**File:** `test_report_2026_03_04.md` (213 lines, 8.5 KB)

**Contains:**
- Executive summary of all issues
- Detailed failure analysis for each category
- Summary table by test file
- Health metrics
- Root cause analysis for each failure

**Read this for:** Understanding what went wrong and why

---

### 2. Actionable Fix Report
**File:** `ACTIONABLE_FIX_REPORT_2026_03_04.md` (336 lines, 11 KB)

**Contains:**
- Quick status table
- Problem 1: JVP Strategy Validation (25 failures)
  - What happened, why, solutions (Option A vs B)
  - All 6 affected test files listed
  - Exact line numbers and config sections
  - Fix commands with sed examples
- Problem 2: VAE Output Shape Mismatch (3 failures)
  - Root cause investigation steps
  - Specific test names and expected vs actual shapes
  - Files to check in source code
- Problem 3: GPU Out-of-Memory (6 failures)
  - Why expected (project policy)
  - Skip solutions with code examples
- Problem 4: Mock Testing Bug (1 failure)
  - Simple one-line fix options
- Recommended fix order and timing
- Commands for each fix
- Summary table with severity/impact

**Read this for:** Step-by-step instructions to fix all failures

---

## Quick Decision Table

| Issue | Count | Type | Severity | FIX TIME | Read |
|-------|-------|------|----------|----------|------|
| JVP Strategy | 25 | Config | HIGH | 5 min | ACTIONABLE §1 |
| VAE Shapes | 3 | Code | HIGH | 20 min | ACTIONABLE §2 |
| Mock Bug | 1 | Code | MEDIUM | 5 min | ACTIONABLE §4 |
| OOM Tests | 6 | Policy | LOW | 10 min | ACTIONABLE §3 |

---

## Failure Categories

### 1. JVP Strategy Validation (25 failures) — CRITICAL
Tests configured with forbidden `x-prediction + finite_difference JVP` combination.
- Files: 6 test files
- Root cause: Code now correctly rejects unsafe configuration
- **Solution:** Change `jvp_strategy: "finite_difference"` → `"exact"`
- **Time:** 5 minutes
- **Blocks:** Phase 3, 4, 4b, 4d gates

### 2. VAE Output Shape Mismatch (3 failures) — CRITICAL
Expected cube shapes (128³, 32³) but VAE outputs asymmetric shapes with Y-axis reduced.
- File: 1 test file (test_maisi_vae_wrapper.py)
- Root cause: Likely preprocessing pipeline change
- **Solution:** Investigate preprocessing, update test assertions
- **Time:** 20 minutes
- **Blocks:** Phase 0, 1 gates

### 3. GPU Out-of-Memory (6 failures) — EXPECTED
Local RTX 4060 (8GB) insufficient for VAE operations (project policy).
- Files: 2 test files
- Root cause: Tests violate "CPU-only" policy for local GPU
- **Solution:** Add VRAM checks, skip on local GPU
- **Time:** 10 minutes
- **Blocks:** None (environmental)

### 4. Mock Testing Bug (1 failure) — CODE BUG
Test passing MagicMock to real tensor operation.
- File: 1 test file (test_metrics_p5.py)
- Root cause: Mock infrastructure incomplete
- **Solution:** Use real tensor instead
- **Time:** 5 minutes
- **Blocks:** Phase 5 gate

---

## Files Requiring Changes

**Priority 1 (JVP configs):**
- `/home/mpascual/research/code/neuromf/tests/test_diagnostics.py`
- `/home/mpascual/research/code/neuromf/tests/test_sample_collector.py`
- `/home/mpascual/research/code/neuromf/tests/test_meanflow_pipeline.py`
- `/home/mpascual/research/code/neuromf/tests/test_transfer_loading.py`
- `/home/mpascual/research/code/neuromf/tests/test_spatial_masking.py`
- `/home/mpascual/research/code/neuromf/tests/test_real_data_augmentation.py`

**Priority 2 (VAE shapes):**
- `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py`

**Priority 3 (Mock):**
- `/home/mpascual/research/code/neuromf/tests/test_metrics_p5.py`

**Priority 4 (OOM):**
- `/home/mpascual/research/code/neuromf/tests/conftest.py`
- `/home/mpascual/research/code/neuromf/tests/test_maisi_vae_wrapper.py`
- `/home/mpascual/research/code/neuromf/tests/test_latent_dataset.py`

---

## How to Use These Reports

### For Quick Understanding
1. Read this index
2. Check the decision table above
3. Pick the report that matches your task:
   - **Investigating failures?** → `test_report_2026_03_04.md`
   - **Fixing failures?** → `ACTIONABLE_FIX_REPORT_2026_03_04.md`

### For Implementation
1. Open `ACTIONABLE_FIX_REPORT_2026_03_04.md`
2. Follow the recommended fix order (Phase 1 → 2 → 3)
3. Copy commands as-is
4. Re-run tests after each fix
5. Verify phase gates open

### For Review
1. Read both reports in order
2. Each report shows different perspectives:
   - `test_report` = What's broken and why (technical)
   - `ACTIONABLE_FIX_REPORT` = How to fix it (practical)

---

## Next Steps

1. **Immediate:** Fix JVP strategy configs (5 min) → unblocks 25 tests
2. **Then:** Investigate & fix VAE shapes (20 min) → unblocks Phases 0-1
3. **Then:** Fix mock test (5 min) → unblocks Phase 5
4. **Finally:** Add VRAM checks (10 min) → improves CI hygiene

**Total time to unblock all gates: ~40 minutes**

---

## Test Results by Phase

| Phase | Critical Tests | Status | Blocker |
|-------|----------------|--------|---------|
| P0 | 7 | BLOCKED | VAE shapes (3) + OOM (6) |
| P1 | 7 | BLOCKED | OOM (1) |
| P2 | 8 | OK | None — 6 pass, 2 skip |
| P3 | 10 | BLOCKED | JVP strategy (3) |
| P4 | 8 | BLOCKED | JVP strategy (25) |
| P4b | N/A | BLOCKED | JVP strategy (7) |
| P4d | N/A | BLOCKED | JVP strategy (3) |
| P5 | 11 | BLOCKED | Mock (1) |

---

## Document Versions

- **test_report_2026_03_04.md**: Technical analysis (comprehensive)
- **ACTIONABLE_FIX_REPORT_2026_03_04.md**: Fix instructions (practical)
- **TEST_REPORT_INDEX_2026_03_04.md**: This file (navigation)

All saved in: `/home/mpascual/research/code/neuromf/experiments/`

