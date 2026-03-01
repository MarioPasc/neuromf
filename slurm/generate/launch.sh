#!/usr/bin/env bash
# =============================================================================
# NEUROMF GENERATION — SLURM LAUNCHER (atomic)
#
# Submits a single NeuroiMF generation + decode job. Generates latents at
# multiple NFE levels, prepares real test volumes, decodes, and visualizes.
#
# Usage (from login node):
#   bash slurm/generate/launch.sh --run-dir /path/to/NeuroiMF_01032026
#   bash slurm/generate/launch.sh --run-dir /path/to/run --n-samples 500
#   bash slurm/generate/launch.sh --run-dir /path/to/run --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
N_SAMPLES=""
RUN_DIR_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="$2"
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
            echo "Usage: bash slurm/generate/launch.sh --run-dir PATH [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH      Output directory for this run (required)" >&2
            echo "  --n-samples N       Override number of samples (default: from config)" >&2
            echo "  --depends-on ID     Wait for job ID to finish" >&2
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
    export RUN_DIR="${RUN_DIR_ARG}"
else
    RUN_DATE=$(date +%d%m%Y)
    export RUN_DIR="${RESULTS_DST}/phase_5/NeuroiMF_${RUN_DATE}"
fi

export EXPERIMENT_NAME="phase_5_gen_neuromf"
export GEN_N_SAMPLES="${N_SAMPLES}"

echo "==========================================" >&2
echo "NEUROMF GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "Run directory: ${RUN_DIR}" >&2
if [ -n "${N_SAMPLES}" ]; then
    echo "n_samples:     ${N_SAMPLES} (override)" >&2
fi
echo "" >&2

# Create output directories
mkdir -p "${RUN_DIR}/generation/latents"
mkdir -p "${RUN_DIR}/generation/volumes"
mkdir -p "${RUN_DIR}/features"
mkdir -p "${RUN_DIR}/metrics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p5_gen"
    --time=1-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${RUN_DIR}/generate_%j.out"
    --error="${RUN_DIR}/generate_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "NEUROMF JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RUN_DIR}/generate_${JOB_ID}.{out,err}" >&2
echo "Latents:   ${RUN_DIR}/generation/latents/" >&2
echo "Volumes:   ${RUN_DIR}/generation/volumes/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
