# Phase 0 — VAE Validation Report

**Date:** 2026-02-12 22:10:00
**Volumes:** 20
**Total time:** 84.0s
**Status:** ALL PASS

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean SSIM | 0.9463 | > 0.90 | PASS |
| Mean PSNR | 34.66 dB | > 30.0 | PASS |

## Latent Space Statistics

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| channel_0 | 0.0977 | 1.0863 | -5.1672 | 5.0454 |
| channel_1 | 0.1197 | 1.0137 | -4.6852 | 5.0778 |
| channel_2 | 0.0231 | 0.9871 | -4.9342 | 6.2229 |
| channel_3 | 0.0590 | 1.0116 | -4.9957 | 7.3986 |

## Negative Controls

| Control | Pass | Details |
|---------|------|---------|
| gaussian_noise | PASS | ssim=0.0868455097079277, expected=< 0.5 |
| uniform_noise | PASS | ssim=0.07552602142095566, expected=< 0.5 |
| blank_zeros | PASS | latent_finite=True, recon_finite=True |
| constant_half | PASS | latent_finite=True, recon_finite=True |
| wrong_scale_factor | PASS | ssim_correct_sf=0.08759452402591705, ssim_wrong_sf=0.08561642467975616, correct_sf=0.96240234375, wrong_sf=1.0 |
| encoding_stochasticity | PASS | seeded_match=True, unseeded_differ=True, mean_posterior_sigma=0.47772371768951416 |

## Per-Volume Results

| Dataset | Volume | SSIM | PSNR (dB) |
|---------|--------|------|-----------|
| PT001_OASIS1 | t1.nii.gz | 0.9358 | 33.11 |
| PT001_OASIS1 | t1.nii.gz | 0.9625 | 35.77 |
| PT001_OASIS1 | t1.nii.gz | 0.9377 | 33.32 |
| PT001_OASIS1 | t1.nii.gz | 0.9571 | 34.76 |
| PT001_OASIS1 | t1.nii.gz | 0.9543 | 33.49 |
| PT001_OASIS1 | t1.nii.gz | 0.9319 | 32.91 |
| PT001_OASIS1 | t1.nii.gz | 0.9475 | 34.52 |
| PT001_OASIS1 | t1.nii.gz | 0.9479 | 35.16 |
| PT001_OASIS1 | t1.nii.gz | 0.9308 | 34.24 |
| PT001_OASIS1 | t1.nii.gz | 0.9367 | 33.36 |
| PT001_OASIS1 | t1.nii.gz | 0.9610 | 35.09 |
| PT001_OASIS1 | t1.nii.gz | 0.9639 | 36.74 |
| PT001_OASIS1 | t1.nii.gz | 0.9610 | 36.67 |
| PT001_OASIS1 | t1.nii.gz | 0.9492 | 35.61 |
| PT001_OASIS1 | t1.nii.gz | 0.9346 | 34.50 |
| PT001_OASIS1 | t1.nii.gz | 0.9441 | 34.72 |
| PT001_OASIS1 | t1.nii.gz | 0.9495 | 35.63 |
| PT001_OASIS1 | t1.nii.gz | 0.9184 | 33.00 |
| PT001_OASIS1 | t1.nii.gz | 0.9506 | 34.97 |
| PT001_OASIS1 | t1.nii.gz | 0.9511 | 35.73 |