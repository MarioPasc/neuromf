#!/usr/bin/env bash
# =============================================================================
# PHASE 0: VAE VALIDATION — SLURM LAUNCHER
#
# Login-node script that submits the Phase 0 VAE validation pipeline on Picasso.
# Uses a single A100 GPU to validate the frozen MAISI VAE on 20 FOMO-60K volumes
# at 192^3 resolution (requires ~15GB VRAM, too much for local RTX 4060 8GB).
#
# Expected time: ~15-30 min on A100
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#   bash experiments/slurm/phase_0/validate_vae.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PHASE 0: VAE VALIDATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_0_vae_validation"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/metrics"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/reconstructions"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/figures"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/latent_stats"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p0_vae" \
    --time=0-01:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/phase_0/vae_validation_%j.out" \
    --error="${RESULTS_DST}/phase_0/vae_validation_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/validate_vae_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_0/vae_validation_${JOB_ID}.{out,err}"
echo "Results:   ${RESULTS_DST}/phase_0/"
echo ""
echo "After completion:"
echo "  1. Check HTML report: ${RESULTS_DST}/phase_0/verification_report.html"
echo "  2. Run Phase 0 tests:  python -m pytest tests/test_maisi_vae_wrapper.py -v --tb=short"
