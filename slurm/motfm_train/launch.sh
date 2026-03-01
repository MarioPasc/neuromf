#!/usr/bin/env bash
# =============================================================================
# MOTFM TRAINING — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the MOTFM training job on FOMO-60K.
# Requires data prep to be complete (fomo60k_3d.h5).
#
# Usage (from login node):
#   bash slurm/motfm_train/launch.sh
#   bash slurm/motfm_train/launch.sh --depends-on <JOB_ID>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/motfm_train/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "MOTFM TRAINING LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="motfm_train"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export MOTFM_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Config:      ${MOTFM_CONFIG}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "" >&2

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
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:       ${JOB_ID}" >&2
echo "Monitor:      squeue -j ${JOB_ID}" >&2
echo "Logs:         ${RESULTS_DST}/motfm/train_${JOB_ID}.{out,err}" >&2
echo "Checkpoints:  ${RESULTS_DST}/motfm/checkpoints/" >&2
echo "" >&2
echo "Chain with generation:" >&2
echo "  bash experiments/slurm/phase_5/generate.sh --motfm --depends-on ${JOB_ID}" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
