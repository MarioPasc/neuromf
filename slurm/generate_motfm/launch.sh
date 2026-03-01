#!/usr/bin/env bash
# =============================================================================
# MOTFM GENERATION — SLURM LAUNCHER (atomic)
#
# Submits a single MOTFM generation job. Generates volumes at multiple NFE
# levels, prepares real test volumes, and visualizes.
#
# Usage (from login node):
#   bash slurm/generate_motfm/launch.sh --run-dir /path/to/MOTFM_01032026
#   bash slurm/generate_motfm/launch.sh --run-dir /path --checkpoint /path/to/last.ckpt
#   bash slurm/generate_motfm/launch.sh --run-dir /path --n-samples 500 --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
N_SAMPLES=""
RUN_DIR_ARG=""
CHECKPOINT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/generate_motfm/launch.sh --run-dir PATH [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH        Output directory for this run (required)" >&2
            echo "  --checkpoint PATH     MOTFM checkpoint (default: from config)" >&2
            echo "  --n-samples N         Override number of samples" >&2
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
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Resolve run directory
if [ -n "${RUN_DIR_ARG}" ]; then
    export MOTFM_RUN_DIR="${RUN_DIR_ARG}"
else
    RUN_DATE=$(date +%d%m%Y)
    export MOTFM_RUN_DIR="${RESULTS_DST}/phase_5/MOTFM_${RUN_DATE}"
fi

# Resolve checkpoint
if [ -n "${CHECKPOINT_ARG}" ]; then
    export MOTFM_CHECKPOINT="${CHECKPOINT_ARG}"
else
    export MOTFM_CHECKPOINT="${RESULTS_DST}/motfm/checkpoints/fomo60k_unconditional_3d/last.ckpt"
fi

export MOTFM_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"
export GEN_N_SAMPLES="${N_SAMPLES}"

echo "==========================================" >&2
echo "MOTFM GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "Run directory: ${MOTFM_RUN_DIR}" >&2
echo "  Checkpoint:  ${MOTFM_CHECKPOINT}" >&2
echo "  Config:      ${MOTFM_CONFIG}" >&2
if [ -n "${N_SAMPLES}" ]; then
    echo "  n_samples:   ${N_SAMPLES} (override)" >&2
fi
echo "" >&2

# Pre-flight: check checkpoint exists
if [ ! -f "${MOTFM_CHECKPOINT}" ]; then
    echo "WARNING: MOTFM checkpoint not found: ${MOTFM_CHECKPOINT}" >&2
    echo "         MOTFM job will fail at runtime." >&2
fi

# Create output directories
mkdir -p "${MOTFM_RUN_DIR}/generation/volumes"
mkdir -p "${MOTFM_RUN_DIR}/features"
mkdir -p "${MOTFM_RUN_DIR}/metrics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="motfm_p5_gen"
    --time=1-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${MOTFM_RUN_DIR}/generate_%j.out"
    --error="${MOTFM_RUN_DIR}/generate_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "MOTFM JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${MOTFM_RUN_DIR}/generate_${JOB_ID}.{out,err}" >&2
echo "Volumes:   ${MOTFM_RUN_DIR}/generation/volumes/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
