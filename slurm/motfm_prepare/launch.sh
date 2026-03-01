#!/usr/bin/env bash
# =============================================================================
# MOTFM DATA PREPARATION — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the FOMO-60K -> MOTFM .h5 conversion job.
# CPU-only, high memory (loading all 1,379 volumes at 192^3).
#
# Usage (from login node):
#   bash slurm/motfm_prepare/launch.sh
#   bash slurm/motfm_prepare/launch.sh --depends-on 12345
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
            echo "Usage: bash slurm/motfm_prepare/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "MOTFM DATA PREPARATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="motfm_prepare_data"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export MOTFM_OUTPUT="${RESULTS_DST}/motfm/data/fomo60k_3d"
export SPLIT_MANIFEST="${RESULTS_DST}/latents/split_manifest.json"

echo "Configuration:" >&2
echo "  Repo:            ${REPO_SRC}" >&2
echo "  Configs:         ${CONFIGS_DIR}" >&2
echo "  Results:         ${RESULTS_DST}" >&2
echo "  Output base:     ${MOTFM_OUTPUT} (.h5)" >&2
echo "  Split manifest:  ${SPLIT_MANIFEST}" >&2
echo "" >&2

# Create output directories
mkdir -p "${RESULTS_DST}/motfm/data"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="motfm_data_prep"
    --time=0-06:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=500G
    --partition=cpu
    --output="${RESULTS_DST}/motfm/prepare_data_%j.out"
    --error="${RESULTS_DST}/motfm/prepare_data_%j.err"
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
echo "Logs:      ${RESULTS_DST}/motfm/prepare_data_${JOB_ID}.{out,err}" >&2
echo "Output:    ${MOTFM_OUTPUT}" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
