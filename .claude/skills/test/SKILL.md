---
name: test
description: Run pytest tests for the neuromf project
allowed-tools:
  - Bash
argument-hint: "[test-path or pattern, e.g. 'tests/test_meanflow_loss.py' or '-m \"not slow\"' or '-k P4']"
---

# NeuroMF Test Runner

Run pytest for the neuromf project using the neuromf conda environment.

## Instructions

1. Run the tests specified by `$ARGUMENTS`. If no arguments given, run the **fast** suite (`-m "not slow"`).
2. Command format:
   ```bash
   ~/.conda/envs/neuromf/bin/python -m pytest $ARGUMENTS -v --tb=short
   ```
3. If tests fail, analyze the failures and suggest fixes.
4. If all tests pass, report the summary.

## Test Suite Structure

| Suite | Command | Tests | Time | When to use |
|-------|---------|-------|------|-------------|
| **Fast** | `-m "not slow"` | ~172 | ~40s | After any code change (default) |
| **Slow** | `-m "slow"` | ~120 | ~5 min | After model/architecture changes |
| **All** | (no marker filter) | ~292 | ~6 min | Before training submission |
| **Phase N** | `-m "phaseN"` | varies | varies | Checking specific phase gate |
| **Critical only** | `-m "critical"` | ~74 | ~3 min | Quick gate verification |

## Marker Reference

- `slow` — Tests that construct UNet/LatentMeanFlow/VAE models (1-3s each for init)
- `phase0`-`phase6` — Phase-specific tests
- `critical` — Must pass for phase gate to open
- `informational` — Nice to have, not gating

## Known Issues

- `test_P5_T9_feature_extractor_mock` — Pre-existing mock bug (MagicMock vs Tensor), does NOT indicate real code failure
- Phase 0 VAE tests (`test_maisi_vae_wrapper.py`) require real VAE weights + GPU — expected to fail locally
- Phase 1 `test_P1_T7_round_trip_ssim` requires real data + GPU

## Examples

```bash
# Default: fast suite (recommended after code changes)
/test

# Specific file
/test tests/test_meanflow_pipeline.py

# Phase 4 critical tests only
/test -m "phase4 and critical"

# All slow tests (before training submission)
/test -m "slow"

# Specific test by name
/test -k "test_P3_T4_meanflow_loss_finite_positive"
```
