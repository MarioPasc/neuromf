#!/usr/bin/env bash
# =============================================================================
# MOTFM PRETRAINED VERIFICATION — SLURM LAUNCHER
#
# Generates a small number of volumes from the author-provided MOTFM checkpoint
# at multiple NFE levels to verify output quality.
#
# Usage (from login node):
#   bash slurm/verify_motfm_pretrained/launch.sh
#   bash slurm/verify_motfm_pretrained/launch.sh --checkpoint /path/to/ckpt
#   bash slurm/verify_motfm_pretrained/launch.sh --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
CHECKPOINT_ARG=""
CONFIG_ARG=""
OUTPUT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --config)
            CONFIG_ARG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ARG="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/verify_motfm_pretrained/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --checkpoint PATH     MOTFM pretrained checkpoint" >&2
            echo "  --config PATH         Config matching checkpoint architecture" >&2
            echo "  --output-dir PATH     Output directory for verification results" >&2
            echo "  --depends-on ID       Wait for job ID to finish" >&2
            exit 1
            ;;
    esac
done

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Pretrained checkpoint (default: author-provided weights on Picasso)
export VERIFY_CHECKPOINT="${CHECKPOINT_ARG:-${RESULTS_DST}/checkpoints/MOTFM/motfm_3d_unconditional_mri_weights/epoch119-valloss0.040524.ckpt}"
export VERIFY_CONFIG="${CONFIG_ARG:-${REPO_SRC}/configs/motfm_pretrained/config_3D.yaml}"
export VERIFY_OUTPUT="${OUTPUT_ARG:-${RESULTS_DST}/motfm_pretrained_verification}"

echo "==========================================" >&2
echo "MOTFM PRETRAINED VERIFICATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time:        $(date)" >&2
echo "Checkpoint:  ${VERIFY_CHECKPOINT}" >&2
echo "Config:      ${VERIFY_CONFIG}" >&2
echo "Output:      ${VERIFY_OUTPUT}" >&2
echo "" >&2

# Pre-flight
if [ ! -f "${VERIFY_CHECKPOINT}" ]; then
    echo "WARNING: Checkpoint not found: ${VERIFY_CHECKPOINT}" >&2
    echo "         Job will fail at runtime. Transfer the checkpoint to Picasso first." >&2
fi

mkdir -p "${VERIFY_OUTPUT}"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="motfm_verify"
    --time=01:00:00
    --ntasks=1
    --cpus-per-task=8
    --mem=64G
    --constraint=dgx
    --gres=gpu:1
    --output="${VERIFY_OUTPUT}/verify_%j.out"
    --error="${VERIFY_OUTPUT}/verify_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "VERIFICATION JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:  ${JOB_ID}" >&2
echo "Monitor: squeue -j ${JOB_ID}" >&2
echo "Logs:    ${VERIFY_OUTPUT}/verify_${JOB_ID}.{out,err}" >&2
echo "Output:  ${VERIFY_OUTPUT}/" >&2

# Job ID to stdout for orchestration
echo "${JOB_ID}"
