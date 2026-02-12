# Phase 0 — VAE Validation Report

**Date:** 2026-02-12 22:02:09
**Volumes:** 20
**Total time:** 84.7s
**Status:** ALL PASS

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean SSIM | 0.9463 | > 0.90 | PASS |
| Mean PSNR | 34.70 dB | > 30.0 | PASS |

## Latent Space Statistics

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| channel_0 | 0.0973 | 1.0867 | -5.2410 | 5.0506 |
| channel_1 | 0.1158 | 1.0131 | -4.7756 | 5.2362 |
| channel_2 | 0.0239 | 0.9871 | -4.6222 | 6.2954 |
| channel_3 | 0.0606 | 1.0122 | -4.4369 | 7.5962 |

## Negative Controls

| Control | Pass | Details |
|---------|------|---------|
| gaussian_noise | PASS | ssim=0.08694388717412949, expected=< 0.5 |
| uniform_noise | PASS | ssim=0.07721330225467682, expected=< 0.5 |
| blank_zeros | PASS | latent_finite=True, recon_finite=True |
| constant_half | PASS | latent_finite=True, recon_finite=True |
| wrong_scale_factor | PASS | ssim_correct_sf=0.08744288235902786, ssim_wrong_sf=0.08548322319984436, correct_sf=0.96240234375, wrong_sf=1.0 |
| encoding_stochasticity | PASS | seeded_match=True, unseeded_differ=True, mean_posterior_sigma=0.4779486656188965 |

## Per-Volume Results

| Volume | SSIM | PSNR (dB) |
|--------|------|-----------|
| t1.nii.gz | 0.9356 | 33.08 |
| t1.nii.gz | 0.9627 | 35.78 |
| t1.nii.gz | 0.9383 | 33.44 |
| t1.nii.gz | 0.9575 | 34.84 |
| t1.nii.gz | 0.9544 | 33.49 |
| t1.nii.gz | 0.9315 | 32.89 |
| t1.nii.gz | 0.9476 | 34.56 |
| t1.nii.gz | 0.9477 | 35.24 |
| t1.nii.gz | 0.9305 | 34.20 |
| t1.nii.gz | 0.9367 | 33.42 |
| t1.nii.gz | 0.9608 | 35.00 |
| t1.nii.gz | 0.9642 | 37.03 |
| t1.nii.gz | 0.9607 | 36.56 |
| t1.nii.gz | 0.9497 | 35.70 |
| t1.nii.gz | 0.9341 | 34.46 |
| t1.nii.gz | 0.9446 | 34.88 |
| t1.nii.gz | 0.9494 | 35.65 |
| t1.nii.gz | 0.9186 | 33.08 |
| t1.nii.gz | 0.9507 | 35.00 |
| t1.nii.gz | 0.9508 | 35.63 |