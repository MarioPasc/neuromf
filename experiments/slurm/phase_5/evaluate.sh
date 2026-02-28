#!/usr/bin/env bash
# =============================================================================
# PHASE 5: EVALUATION — SLURM LAUNCHER
#
# Login-node script that submits feature extraction + metrics job(s).
# Requires generate.sh to have completed first.
# Each model's results live in its own {ModelName}_{date}/ directory.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/evaluate.sh --results-dir .../NeuroiMF_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --motfm \
#       --results-dir .../NeuroiMF_01032026 --motfm-results-dir .../MOTFM_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --motfm --skip-neuromf \
#       --motfm-results-dir .../MOTFM_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --depends-on 95484 --results-dir ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
ENABLE_MOTFM=0
SKIP_NEUROMF=0
NEUROMF_RESULTS_DIR=""
MOTFM_RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --motfm)
            ENABLE_MOTFM=1
            shift
            ;;
        --skip-neuromf)
            SKIP_NEUROMF=1
            shift
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --results-dir)
            NEUROMF_RESULTS_DIR="$2"
            shift 2
            ;;
        --motfm-results-dir)
            MOTFM_RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/slurm/phase_5/evaluate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --results-dir PATH        NeuroiMF run directory"
            echo "  --motfm                   Also evaluate MOTFM (separate job)"
            echo "  --motfm-results-dir PATH  MOTFM run directory"
            echo "  --skip-neuromf            Skip NeuroiMF evaluation"
            echo "  --depends-on JOB_ID       Wait for job to finish"
            exit 1
            ;;
    esac
done

# Validate flags
if [ "${SKIP_NEUROMF}" -eq 1 ] && [ "${ENABLE_MOTFM}" -eq 0 ]; then
    echo "ERROR: --skip-neuromf requires --motfm"
    exit 1
fi

if [ "${SKIP_NEUROMF}" -eq 0 ] && [ -z "${NEUROMF_RESULTS_DIR}" ]; then
    echo "ERROR: --results-dir is required for NeuroiMF evaluation"
    exit 1
fi

if [ "${ENABLE_MOTFM}" -eq 1 ] && [ -z "${MOTFM_RESULTS_DIR}" ]; then
    echo "ERROR: --motfm-results-dir is required when --motfm is set"
    exit 1
fi

echo "=========================================="
echo "PHASE 5: EVALUATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    echo "  NeuroiMF:    ${NEUROMF_RESULTS_DIR}"
fi
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    echo "  MOTFM:       ${MOTFM_RESULTS_DIR}"
fi
echo ""

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
# SUBMIT NEUROMF EVALUATION JOB
# ========================================================================
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    export RUN_DIR="${NEUROMF_RESULTS_DIR}"

    mkdir -p "${RUN_DIR}/features"
    mkdir -p "${RUN_DIR}/metrics"
    mkdir -p "${RUN_DIR}/metrics/synthseg"

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
        SBATCH_ARGS+=(--dependency=afterok:${DEPENDS_ON})
        echo "Dependency:  afterok:${DEPENDS_ON}"
    fi

    NEUROMF_JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/evaluate_worker.sh")

    echo "=========================================="
    echo "NEUROMF EVALUATION JOB SUBMITTED"
    echo "=========================================="
    echo "Job ID:    ${NEUROMF_JOB_ID}"
    echo "Monitor:   squeue -j ${NEUROMF_JOB_ID}"
    echo "Logs:      ${RUN_DIR}/evaluate_${NEUROMF_JOB_ID}.{out,err}"
    echo "Features:  ${RUN_DIR}/features/"
    echo "Metrics:   ${RUN_DIR}/metrics/"
    echo ""
fi

# ========================================================================
# SUBMIT MOTFM EVALUATION JOB (same worker script, different RUN_DIR)
# ========================================================================
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    export RUN_DIR="${MOTFM_RESULTS_DIR}"

    mkdir -p "${RUN_DIR}/features"
    mkdir -p "${RUN_DIR}/metrics"
    mkdir -p "${RUN_DIR}/metrics/synthseg"

    MOTFM_SBATCH_ARGS=(
        --parsable
        --job-name="motfm_p5_eval"
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
        MOTFM_SBATCH_ARGS+=(--dependency=afterok:${DEPENDS_ON})
    fi

    MOTFM_JOB_ID=$(sbatch "${MOTFM_SBATCH_ARGS[@]}" "${SCRIPT_DIR}/evaluate_worker.sh")

    echo "=========================================="
    echo "MOTFM EVALUATION JOB SUBMITTED"
    echo "=========================================="
    echo "Job ID:    ${MOTFM_JOB_ID}"
    echo "Monitor:   squeue -j ${MOTFM_JOB_ID}"
    echo "Logs:      ${RUN_DIR}/evaluate_${MOTFM_JOB_ID}.{out,err}"
    echo "Features:  ${RUN_DIR}/features/"
    echo "Metrics:   ${RUN_DIR}/metrics/"
    echo ""
fi

# ========================================================================
# SUMMARY
# ========================================================================
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    echo "  NeuroiMF:  Job ${NEUROMF_JOB_ID}  →  ${NEUROMF_RESULTS_DIR}"
fi
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    echo "  MOTFM:     Job ${MOTFM_JOB_ID}  →  ${MOTFM_RESULTS_DIR}"
fi
echo ""
echo "Next step: run analyse.sh with --depends-on and --results-dir"
