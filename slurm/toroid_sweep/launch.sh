#!/usr/bin/env bash
# =============================================================================
# TOROID ABLATION SWEEP — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the full Phase 2 formal experiment on Picasso.
# Single worker runs all 18 training runs + NFE sweep + figures + report
# end-to-end without releasing resources.
#
# The experiment is CPU-only (toy MLP on R^4 data, no GPU needed), but we
# request a DGX node for its fast CPUs and large RAM.
#
# Expected wall-time: ~45-90 min (18 training runs x 500 epochs each)
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#   bash slurm/toroid_sweep/launch.sh
#   bash slurm/toroid_sweep/launch.sh --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/toroid_sweep/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "TOROID ABLATION SWEEP LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_2_toroid_sweep"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results/phase_2/toroid}"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "" >&2

# Create output directories
mkdir -p "${RESULTS_DST}"
mkdir -p "${RESULTS_DST}/figures"
mkdir -p "${RESULTS_DST}/tables"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p2_toroid"
    --time=0-23:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=32G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/sweep_%j.out"
    --error="${RESULTS_DST}/sweep_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RESULTS_DST}/sweep_${JOB_ID}.{out,err}" >&2
echo "Results:   ${RESULTS_DST}/" >&2
echo "" >&2
echo "After completion:" >&2
echo "  - Report:  ${RESULTS_DST}/report.html" >&2
echo "  - Figures:  ${RESULTS_DST}/figures/" >&2
echo "  - Tables:   ${RESULTS_DST}/tables/" >&2
echo "  - Run tests: python -m pytest tests/ -v -k P2" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
