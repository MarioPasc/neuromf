#!/usr/bin/env bash
# =============================================================================
# PHASE 5: EVALUATION — SLURM LAUNCHER
#
# Login-node script that submits the feature extraction + metrics job.
# Requires generate.sh to have completed first.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/evaluate.sh
#   bash experiments/slurm/phase_5/evaluate.sh --depends-on 95484
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
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/slurm/phase_5/evaluate.sh [--depends-on JOB_ID]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "PHASE 5: EVALUATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_5_evaluation"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Results:     ${RESULTS_DST}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/phase_5/features"
mkdir -p "${RESULTS_DST}/phase_5/metrics"
mkdir -p "${RESULTS_DST}/phase_5/metrics/synthseg"

# ========================================================================
# PRE-DOWNLOAD WEIGHTS (login node has internet, worker nodes do not)
# ========================================================================
R3D18_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/r3d_18_fid3d"
R3D18_FILE="${R3D18_DIR}/r3d_18-b3b3357e.pth"
R3D18_URL="https://download.pytorch.org/models/r3d_18-b3b3357e.pth"

if [ -f "${R3D18_FILE}" ]; then
    SIZE=$(stat -c%s "${R3D18_FILE}" 2>/dev/null || echo "?")
    echo "R3D-18 weights: ${R3D18_FILE} (${SIZE} bytes) [cached]"
else
    echo "Downloading R3D-18 weights (login node → ${R3D18_FILE}) ..."
    mkdir -p "${R3D18_DIR}"
    wget -q --show-progress -O "${R3D18_FILE}" "${R3D18_URL}"
    echo "R3D-18 weights downloaded: $(stat -c%s "${R3D18_FILE}") bytes"
fi
echo ""

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p5_eval"
    --time=0-23:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/phase_5/evaluate_%j.out"
    --error="${RESULTS_DST}/phase_5/evaluate_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency=afterok:${DEPENDS_ON})
    echo "Dependency:  afterok:${DEPENDS_ON}"
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/evaluate_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_5/evaluate_${JOB_ID}.{out,err}"
echo "Features:  ${RESULTS_DST}/phase_5/features/"
echo "Metrics:   ${RESULTS_DST}/phase_5/metrics/"
