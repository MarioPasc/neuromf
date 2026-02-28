#!/usr/bin/env bash
# =============================================================================
# MOTFM TRAINING — SLURM LAUNCHER
#
# Login-node script that submits the MOTFM training job on FOMO-60K.
# Requires data prep to be complete (fomo60k_3d.pkl).
#
# Usage (from login node):
#   bash experiments/slurm/motfm/train.sh
#   bash experiments/slurm/motfm/train.sh --depends-on <JOB_ID>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "MOTFM TRAINING LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# PARSE ARGS
# ========================================================================
DEPENDS_ON=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="motfm_train"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

export MOTFM_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Config:      ${MOTFM_CONFIG}"
echo "  Results:     ${RESULTS_DST}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/motfm/checkpoints"
mkdir -p "${RESULTS_DST}/motfm/logs"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="motfm_train"
    --time=3-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/motfm/train_%j.out"
    --error="${RESULTS_DST}/motfm/train_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}"
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:       ${JOB_ID}"
echo "Monitor:      squeue -j ${JOB_ID}"
echo "Logs:         ${RESULTS_DST}/motfm/train_${JOB_ID}.{out,err}"
echo "Checkpoints:  ${RESULTS_DST}/motfm/checkpoints/"
echo ""
echo "Chain with generation:"
echo "  bash experiments/slurm/phase_5/generate.sh --motfm --depends-on ${JOB_ID}"
