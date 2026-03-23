#!/usr/bin/env bash
# =============================================================================
# EVALUATION — SLURM LAUNCHER (atomic)
#
# Submits a single evaluation job (feature extraction + metrics) for one
# model's results directory. Works for both NeuroiMF and MOTFM.
#
# Usage (from login node):
#   bash slurm/evaluate/launch.sh --run-dir /path/to/NeuroiMF_01032026
#   bash slurm/evaluate/launch.sh --run-dir /path/to/run --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
RUN_DIR_ARG=""
SYNTHSEG_MAX_VOLUMES_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --synthseg-max-volumes)
            SYNTHSEG_MAX_VOLUMES_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/evaluate/launch.sh --run-dir PATH [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH              Results directory to evaluate (required)" >&2
            echo "  --depends-on ID             Wait for job ID to finish" >&2
            echo "  --synthseg-max-volumes N    Limit SynthSeg to first N volumes (default: 100)" >&2
            exit 1
            ;;
    esac
done

if [ -z "${RUN_DIR_ARG}" ]; then
    echo "ERROR: --run-dir is required" >&2
    exit 1
fi

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RUN_DIR="${RUN_DIR_ARG}"
export SYNTHSEG_MAX_VOLUMES="${SYNTHSEG_MAX_VOLUMES_ARG:-100}"

echo "==========================================" >&2
echo "EVALUATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "Run directory: ${RUN_DIR}" >&2
echo "" >&2

# ========================================================================
# PRE-DOWNLOAD WEIGHTS (login node has internet, worker nodes do not)
# ========================================================================
R3D18_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/r3d_18_fid3d"
R3D18_FILE="${R3D18_DIR}/r3d_18-b3b3357e.pth"
R3D18_URL="https://download.pytorch.org/models/r3d_18-b3b3357e.pth"

if [ -f "${R3D18_FILE}" ]; then
    SIZE=$(stat -c%s "${R3D18_FILE}" 2>/dev/null || echo "?")
    echo "R3D-18 weights: ${R3D18_FILE} (${SIZE} bytes) [cached]" >&2
else
    echo "Downloading R3D-18 weights (login node -> ${R3D18_FILE}) ..." >&2
    mkdir -p "${R3D18_DIR}"
    wget -q --show-progress -O "${R3D18_FILE}" "${R3D18_URL}"
    echo "R3D-18 weights downloaded: $(stat -c%s "${R3D18_FILE}") bytes" >&2
fi
echo "" >&2

# Create output directories
mkdir -p "${RUN_DIR}/features"
mkdir -p "${RUN_DIR}/metrics"

# Detect comparison mode and create variant directories
GEN_DIR="${RUN_DIR}/generation"
if [ -d "${GEN_DIR}/volumes_baseline" ] && [ -d "${GEN_DIR}/volumes_enhanced" ]; then
    echo "Comparison mode detected — creating variant output dirs" >&2
    mkdir -p "${RUN_DIR}/features/baseline"
    mkdir -p "${RUN_DIR}/features/enhanced"
    mkdir -p "${RUN_DIR}/metrics/baseline"
    mkdir -p "${RUN_DIR}/metrics/enhanced"
else
    mkdir -p "${RUN_DIR}/metrics/synthseg"
fi

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
    --output="${RUN_DIR}/evaluate_%j.out"
    --error="${RUN_DIR}/evaluate_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "EVALUATION JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RUN_DIR}/evaluate_${JOB_ID}.{out,err}" >&2
echo "Features:  ${RUN_DIR}/features/" >&2
echo "Metrics:   ${RUN_DIR}/metrics/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
