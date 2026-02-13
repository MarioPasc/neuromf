# Phase 0 — VAE Validation Report

**Date:** 2026-02-13 09:31:12
**Volumes:** 20
**Total time:** 145.4s
**Status:** ALL PASS

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean SSIM | 0.9649 | > 0.90 | PASS |
| Mean PSNR | 36.79 dB | > 30.0 | PASS |

## Latent Space Statistics

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| channel_0 | 0.0053 | 1.0200 | -5.2907 | 5.4293 |
| channel_1 | -0.0633 | 1.0263 | -5.3222 | 4.5758 |
| channel_2 | -0.0128 | 0.9771 | -4.8032 | 7.0226 |
| channel_3 | 0.0370 | 1.0109 | -4.7873 | 8.3470 |

## Negative Controls

| Control | Pass | Details |
|---------|------|---------|
| gaussian_noise | PASS | ssim=0.08533000946044922, expected=< 0.5 |
| uniform_noise | PASS | ssim=0.07478873431682587, expected=< 0.5 |
| blank_zeros | PASS | latent_finite=True, recon_finite=True |
| constant_half | PASS | latent_finite=True, recon_finite=True |
| wrong_scale_factor | PASS | ssim_correct_sf=0.08709130436182022, ssim_wrong_sf=0.08507780730724335, correct_sf=0.96240234375, wrong_sf=1.0 |
| encoding_stochasticity | PASS | seeded_match=True, unseeded_differ=True, mean_posterior_sigma=0.47709977626800537 |

## Per-Volume Results

| Dataset | Volume | SSIM | PSNR (dB) |
|---------|--------|------|-----------|
| PT001_OASIS1 | t1.nii.gz | 0.9604 | 35.44 |
| PT001_OASIS1 | t1.nii.gz | 0.9771 | 37.99 |
| PT001_OASIS1 | t1.nii.gz | 0.9574 | 35.11 |
| PT001_OASIS1 | t1.nii.gz | 0.9736 | 37.27 |
| PT001_OASIS1 | t1.nii.gz | 0.9671 | 35.29 |
| PT001_OASIS1 | t1.nii.gz | 0.9584 | 35.13 |
| PT001_OASIS1 | t1.nii.gz | 0.9631 | 36.28 |
| PT001_OASIS1 | t1.nii.gz | 0.9673 | 37.47 |
| PT001_OASIS1 | t1.nii.gz | 0.9536 | 36.22 |
| PT001_OASIS1 | t1.nii.gz | 0.9584 | 35.41 |
| PT001_OASIS1 | t1.nii.gz | 0.9748 | 37.33 |
| PT001_OASIS1 | t1.nii.gz | 0.9771 | 39.35 |
| PT001_OASIS1 | t1.nii.gz | 0.9749 | 38.73 |
| PT001_OASIS1 | t1.nii.gz | 0.9651 | 37.56 |
| PT001_OASIS1 | t1.nii.gz | 0.9551 | 36.44 |
| PT001_OASIS1 | t1.nii.gz | 0.9619 | 36.79 |
| PT001_OASIS1 | t1.nii.gz | 0.9700 | 38.03 |
| PT001_OASIS1 | t1.nii.gz | 0.9447 | 34.82 |
| PT001_OASIS1 | t1.nii.gz | 0.9692 | 37.34 |
| PT001_OASIS1 | t1.nii.gz | 0.9680 | 37.76 |