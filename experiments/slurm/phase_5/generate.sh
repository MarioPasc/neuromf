#!/usr/bin/env bash
# =============================================================================
# PHASE 5: GENERATION PIPELINE — SLURM LAUNCHER
#
# Login-node script that submits Phase 5 generation + decode job(s).
# Each model gets its own run directory: {ModelName}_{DDMMYYYY}/
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/generate.sh
#   bash experiments/slurm/phase_5/generate.sh --motfm
#   bash experiments/slurm/phase_5/generate.sh --motfm --skip-neuromf
#   bash experiments/slurm/phase_5/generate.sh --depends-on <JOB_ID>
#   bash experiments/slurm/phase_5/generate.sh --n-samples 500
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGS
# ========================================================================
ENABLE_MOTFM=0
SKIP_NEUROMF=0
DEPENDS_ON=""
N_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/slurm/phase_5/generate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --motfm             Also generate MOTFM volumes (separate job)"
            echo "  --skip-neuromf      Skip NeuroiMF generation (use with --motfm)"
            echo "  --depends-on ID     Wait for job ID to finish"
            echo "  --n-samples N       Override number of samples (default: from config)"
            exit 1
            ;;
    esac
done

# Validate flags
if [ "${SKIP_NEUROMF}" -eq 1 ] && [ "${ENABLE_MOTFM}" -eq 0 ]; then
    echo "ERROR: --skip-neuromf requires --motfm"
    exit 1
fi

echo "=========================================="
echo "PHASE 5: GENERATION PIPELINE LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

RUN_DATE=$(date +%d%m%Y)

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo "  Run date:    ${RUN_DATE}"
if [ -n "${N_SAMPLES}" ]; then
    echo "  n_samples:   ${N_SAMPLES} (override)"
fi
echo ""

# ========================================================================
# SUBMIT NEUROMF JOB
# ========================================================================
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    export RUN_DIR="${RESULTS_DST}/phase_5/NeuroiMF_${RUN_DATE}"
    export EXPERIMENT_NAME="phase_5_gen_neuromf"
    export GEN_N_SAMPLES="${N_SAMPLES}"

    # Create output directories
    mkdir -p "${RUN_DIR}/generation/latents"
    mkdir -p "${RUN_DIR}/generation/volumes"
    mkdir -p "${RUN_DIR}/features"
    mkdir -p "${RUN_DIR}/metrics"

    echo "NeuroiMF run directory: ${RUN_DIR}"

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
        echo "Dependency: afterok:${DEPENDS_ON}"
    fi

    NEUROMF_JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/generate_worker.sh")

    echo "=========================================="
    echo "NEUROMF JOB SUBMITTED"
    echo "=========================================="
    echo "Job ID:    ${NEUROMF_JOB_ID}"
    echo "Monitor:   squeue -j ${NEUROMF_JOB_ID}"
    echo "Logs:      ${RUN_DIR}/generate_${NEUROMF_JOB_ID}.{out,err}"
    echo "Latents:   ${RUN_DIR}/generation/latents/"
    echo "Volumes:   ${RUN_DIR}/generation/volumes/"
    echo ""
fi

# ========================================================================
# SUBMIT MOTFM JOB (separate from NeuroiMF)
# ========================================================================
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    export MOTFM_RUN_DIR="${RESULTS_DST}/phase_5/MOTFM_${RUN_DATE}"
    export MOTFM_CHECKPOINT="${RESULTS_DST}/motfm/checkpoints/fomo60k_unconditional_3d/last.ckpt"
    export MOTFM_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"
    export GEN_N_SAMPLES="${N_SAMPLES}"

    # Create output directories
    mkdir -p "${MOTFM_RUN_DIR}/generation/volumes"
    mkdir -p "${MOTFM_RUN_DIR}/features"
    mkdir -p "${MOTFM_RUN_DIR}/metrics"

    echo "MOTFM run directory: ${MOTFM_RUN_DIR}"
    echo "  Checkpoint:  ${MOTFM_CHECKPOINT}"
    echo "  Config:      ${MOTFM_CONFIG}"

    # Pre-flight: check MOTFM checkpoint exists
    if [ ! -f "${MOTFM_CHECKPOINT}" ]; then
        echo "WARNING: MOTFM checkpoint not found: ${MOTFM_CHECKPOINT}"
        echo "         MOTFM job will fail at runtime."
    fi

    MOTFM_SBATCH_ARGS=(
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
        MOTFM_SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    fi

    MOTFM_JOB_ID=$(sbatch "${MOTFM_SBATCH_ARGS[@]}" "${SCRIPT_DIR}/generate_motfm_worker.sh")

    echo "=========================================="
    echo "MOTFM JOB SUBMITTED"
    echo "=========================================="
    echo "Job ID:    ${MOTFM_JOB_ID}"
    echo "Monitor:   squeue -j ${MOTFM_JOB_ID}"
    echo "Logs:      ${MOTFM_RUN_DIR}/generate_${MOTFM_JOB_ID}.{out,err}"
    echo "Volumes:   ${MOTFM_RUN_DIR}/generation/volumes/"
    echo ""
fi

# ========================================================================
# SUMMARY
# ========================================================================
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    echo "  NeuroiMF:  Job ${NEUROMF_JOB_ID}  →  ${RUN_DIR}"
fi
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    echo "  MOTFM:     Job ${MOTFM_JOB_ID}  →  ${MOTFM_RUN_DIR}"
fi
echo ""
echo "Next step: run evaluate.sh with --depends-on and --results-dir"
