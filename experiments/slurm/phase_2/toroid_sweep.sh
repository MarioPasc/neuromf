#!/usr/bin/env bash
# =============================================================================
# PHASE 2: TOROID ABLATION SWEEP — SLURM LAUNCHER
#
# Login-node script that submits the full Phase 2 formal experiment on Picasso.
# Single worker runs all 18 training runs + NFE sweep + figures + report
# end-to-end without releasing resources.
#
# The experiment is CPU-only (toy MLP on R^4 data, no GPU needed), but we
# request a DGX node for its fast CPUs and large RAM.
#
# Expected wall-time: ~45-90 min (18 training runs × 500 epochs each)
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#   bash experiments/slurm/phase_2/toroid_sweep.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PHASE 2: TOROID ABLATION SWEEP LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_2_toroid_sweep"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results/phase_2/toroid"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}"
mkdir -p "${RESULTS_DST}/figures"
mkdir -p "${RESULTS_DST}/tables"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p2_toroid" \
    --time=0-03:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/sweep_%j.out" \
    --error="${RESULTS_DST}/sweep_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/toroid_sweep_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/sweep_${JOB_ID}.{out,err}"
echo "Results:   ${RESULTS_DST}/"
echo ""
echo "After completion:"
echo "  - Report:  ${RESULTS_DST}/report.html"
echo "  - Figures:  ${RESULTS_DST}/figures/"
echo "  - Tables:   ${RESULTS_DST}/tables/"
echo "  - Run tests: python -m pytest tests/ -v -k P2"
