#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact
# Called by Claude Code on SessionStart[compact] events

cat <<'CONTEXT'

=== NEUROMF POST-COMPACTION CONTEXT ===

PROJECT: NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis
  Train MeanFlow in frozen MAISI VAE latent space (4×48³) for 1-NFE generation of 192³ brain MRI.

CRITICAL CONSTANTS:
  scale_factor = 0.96240234375  (from diffusion checkpoint, NOT VAE)
  VAE checkpoint wrapped in "unet_state_dict" key — must unwrap before load_state_dict
  Latent shape: (B, 4, 48, 48, 48) for 192³ input
  Local: RTX 4060 8GB (dev only). Training: Picasso A100 40GB × 3-8 GPUs (DDP)

ENVIRONMENT:
  Python:  /home/mpascual/.conda/envs/neuromf/bin/python
  pytest:  ~/.conda/envs/neuromf/bin/python -m pytest tests/ -v --tb=short
  Config:  /home/mpascual/research/code/neuromf/configs/base.yaml
  Results: /media/mpascual/Sandisk2TB/research/neuromf/results/

KEY PATHS:
  Project:     /home/mpascual/research/code/neuromf/
  Core code:   src/neuromf/
  Tests:       tests/ (fixtures in conftest.py)
  VAE weights: /media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt
  FOMO-60K:    /media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K/ (5,471 train / 674 val / 326 test)

TRAINING STATUS (v1 best model, epoch 388):
  Best 2.5D FID: 11.67 | FID-3D: NFE=1 73.85, NFE=10 7.34, NFE=50 6.14
  Config: x-pred + exact JVP, (t,h) cond, v-head, eff. batch=132 (2/GPU × 6 × 11 accum)
  Phases 0-5 COMPLETE | Phase 6 PARTIAL | Phases 7-8 NOT STARTED
  Phases:      docs/splits/phase_{N}.md (read before implementing)

FORBIDDEN:
  - DO NOT modify src/external/ or docs/main/
  - DO NOT use diffusers (2D-only) or torchcfm (time convention mismatch)
  - DO NOT retrain the MAISI VAE

COMMANDS: /implement-phase N, /run-tests N, /check-gate N, /review-external, /dl-scientist
DEPENDENCIES: If you need a new package, add it to pyproject.toml and run:
  ~/.conda/envs/neuromf/bin/pip install -e "/home/mpascual/research/code/neuromf"

Re-read CLAUDE.md for full context: /home/mpascual/research/code/neuromf/CLAUDE.md

=== END COMPACTION RECOVERY ===
CONTEXT
