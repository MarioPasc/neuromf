# Phase 0 — VAE Validation Report

**Date:** 2026-02-12 20:12:54
**Volumes:** 20
**Total time:** 88.8s
**Status:** ALL PASS

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean SSIM | 0.9212 | > 0.90 | PASS |
| Mean PSNR | 30.86 dB | > 30.0 | PASS |

## Latent Space Statistics

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| channel_0 | 0.2398 | 1.0920 | -4.5941 | 6.0172 |
| channel_1 | 0.3611 | 0.9513 | -4.4204 | 4.8584 |
| channel_2 | 0.0501 | 1.1120 | -5.3451 | 5.6238 |
| channel_3 | 0.0755 | 1.1686 | -6.2570 | 6.0405 |

## Negative Controls

| Control | Pass | Details |
|---------|------|---------|
| gaussian_noise | PASS | ssim=0.08816829323768616, expected=< 0.5 |
| uniform_noise | PASS | ssim=0.0764639675617218, expected=< 0.5 |
| blank_zeros | PASS | latent_finite=True, recon_finite=True |
| constant_half | PASS | latent_finite=True, recon_finite=True |
| wrong_scale_factor | PASS | ssim_correct_sf=0.08684395253658295, ssim_wrong_sf=0.08480949699878693, correct_sf=0.96240234375, wrong_sf=1.0 |
| encoding_stochasticity | PASS | seeded_match=True, unseeded_differ=True, mean_posterior_sigma=0.47782862186431885 |

## Per-Volume Results

| Volume | SSIM | PSNR (dB) |
|--------|------|-----------|
| IXI002-Guys-0828-T1.nii.gz | 0.9298 | 30.79 |
| IXI012-HH-1211-T1.nii.gz | 0.9345 | 32.26 |
| IXI013-HH-1212-T1.nii.gz | 0.9397 | 33.60 |
| IXI014-HH-1236-T1.nii.gz | 0.9500 | 34.00 |
| IXI015-HH-1258-T1.nii.gz | 0.9357 | 33.28 |
| IXI016-Guys-0697-T1.nii.gz | 0.8955 | 28.47 |
| IXI017-Guys-0698-T1.nii.gz | 0.9258 | 30.33 |
| IXI019-Guys-0702-T1.nii.gz | 0.9182 | 29.71 |
| IXI020-Guys-0700-T1.nii.gz | 0.9246 | 30.76 |
| IXI021-Guys-0703-T1.nii.gz | 0.9103 | 30.10 |
| IXI022-Guys-0701-T1.nii.gz | 0.9221 | 30.34 |
| IXI023-Guys-0699-T1.nii.gz | 0.9321 | 31.46 |
| IXI024-Guys-0705-T1.nii.gz | 0.9181 | 30.02 |
| IXI025-Guys-0852-T1.nii.gz | 0.9170 | 30.29 |
| IXI026-Guys-0696-T1.nii.gz | 0.9121 | 30.23 |
| IXI027-Guys-0710-T1.nii.gz | 0.9067 | 29.37 |
| IXI028-Guys-1038-T1.nii.gz | 0.9193 | 30.49 |
| IXI029-Guys-0829-T1.nii.gz | 0.8969 | 30.28 |
| IXI030-Guys-0708-T1.nii.gz | 0.9176 | 30.08 |
| IXI031-Guys-0797-T1.nii.gz | 0.9179 | 31.40 |