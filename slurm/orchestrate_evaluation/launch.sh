#!/usr/bin/env bash
# =============================================================================
# MULTI-MODEL EVALUATION ORCHESTRATION — SLURM LAUNCHER
#
# Master script that chains the full evaluation pipeline across all 3 models:
#   1. Generate volumes (NeuroiMF, MOTFM, DDPM — in parallel)
#   2. Evaluate each model (feature extraction + metrics — after generation)
#   3. Cross-model comparison (after all evaluations complete)
#
# Each step is an atomic SLURM job with proper dependencies.
#
# Usage (from login node):
#   bash slurm/orchestrate_evaluation/launch.sh \
#       --neuroimf-checkpoint /path/to/best.ckpt \
#       --motfm-checkpoint /path/to/last.ckpt \
#       --ddpm-checkpoint /path/to/last.ckpt \
#       --output-root /path/to/comparison_YYYYMMDD/ \
#       --n-samples 500 \
#       --synthseg-max-volumes 100
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
NEUROIMF_CHECKPOINT_ARG=""
MOTFM_CHECKPOINT_ARG=""
DDPM_CHECKPOINT_ARG=""
OUTPUT_ROOT_ARG=""
N_SAMPLES=500
SYNTHSEG_MAX_VOLUMES=100
N_BOOTSTRAP=1000
NFE_LEVELS="1 10 50"
SKIP_BOOTSTRAP=0
SKIP_QUALITATIVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --neuroimf-checkpoint)
            NEUROIMF_CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --motfm-checkpoint)
            MOTFM_CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --ddpm-checkpoint)
            DDPM_CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT_ARG="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --synthseg-max-volumes)
            SYNTHSEG_MAX_VOLUMES="$2"
            shift 2
            ;;
        --n-bootstrap)
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --nfe)
            NFE_LEVELS=""
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                NFE_LEVELS="${NFE_LEVELS} $1"
                shift
            done
            ;;
        --skip-bootstrap)
            SKIP_BOOTSTRAP=1
            shift
            ;;
        --skip-qualitative)
            SKIP_QUALITATIVE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "" >&2
            echo "Usage: bash slurm/orchestrate_evaluation/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Required:" >&2
            echo "  --neuroimf-checkpoint PATH   NeuroiMF model checkpoint" >&2
            echo "  --motfm-checkpoint PATH      MOTFM model checkpoint" >&2
            echo "  --ddpm-checkpoint PATH       DDPM model checkpoint" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --output-root PATH           Root output directory (default: auto-dated)" >&2
            echo "  --n-samples N                Number of samples to generate (default: 500)" >&2
            echo "  --synthseg-max-volumes N     Max volumes for SynthSeg (default: 100)" >&2
            echo "  --n-bootstrap N              Bootstrap samples for comparison (default: 1000)" >&2
            echo "  --nfe 1 10 50                NFE levels (default: 1 10 50)" >&2
            echo "  --skip-bootstrap             Skip bootstrap in comparison" >&2
            echo "  --skip-qualitative           Skip qualitative figures in comparison" >&2
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${NEUROIMF_CHECKPOINT_ARG}" ]; then
    echo "ERROR: --neuroimf-checkpoint is required" >&2
    exit 1
fi
if [ -z "${MOTFM_CHECKPOINT_ARG}" ]; then
    echo "ERROR: --motfm-checkpoint is required" >&2
    exit 1
fi
if [ -z "${DDPM_CHECKPOINT_ARG}" ]; then
    echo "ERROR: --ddpm-checkpoint is required" >&2
    exit 1
fi

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Resolve output root
if [ -n "${OUTPUT_ROOT_ARG}" ]; then
    OUTPUT_ROOT="${OUTPUT_ROOT_ARG}"
else
    ORCH_DATE=$(date +%Y%m%d)
    OUTPUT_ROOT="${RESULTS_DST}/phase_5/comparison_${ORCH_DATE}"
fi

# Model output directories
NEUROIMF_DIR="${OUTPUT_ROOT}/NeuroiMF"
MOTFM_DIR="${OUTPUT_ROOT}/MOTFM"
DDPM_DIR="${OUTPUT_ROOT}/DDPM"
COMPARE_DIR="${OUTPUT_ROOT}/comparison"

echo "==========================================" >&2
echo "MULTI-MODEL EVALUATION ORCHESTRATOR" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2
echo "Configuration:" >&2
echo "  Output root:          ${OUTPUT_ROOT}" >&2
echo "  NeuroiMF checkpoint:  ${NEUROIMF_CHECKPOINT_ARG}" >&2
echo "  MOTFM checkpoint:     ${MOTFM_CHECKPOINT_ARG}" >&2
echo "  DDPM checkpoint:      ${DDPM_CHECKPOINT_ARG}" >&2
echo "  n_samples:            ${N_SAMPLES}" >&2
echo "  synthseg_max_volumes: ${SYNTHSEG_MAX_VOLUMES}" >&2
echo "  n_bootstrap:          ${N_BOOTSTRAP}" >&2
echo "  NFE levels:           ${NFE_LEVELS}" >&2
echo "" >&2

# ========================================================================
# PRE-FLIGHT: VERIFY CHECKPOINTS EXIST
# ========================================================================
echo "Pre-flight checks:" >&2

PREFLIGHT_OK=1
if [ -f "${NEUROIMF_CHECKPOINT_ARG}" ]; then
    echo "  [OK]   NeuroiMF checkpoint: ${NEUROIMF_CHECKPOINT_ARG}" >&2
else
    echo "  [MISS] NeuroiMF checkpoint: ${NEUROIMF_CHECKPOINT_ARG}" >&2
    PREFLIGHT_OK=0
fi

if [ -f "${MOTFM_CHECKPOINT_ARG}" ]; then
    echo "  [OK]   MOTFM checkpoint: ${MOTFM_CHECKPOINT_ARG}" >&2
else
    echo "  [MISS] MOTFM checkpoint: ${MOTFM_CHECKPOINT_ARG}" >&2
    PREFLIGHT_OK=0
fi

if [ -f "${DDPM_CHECKPOINT_ARG}" ]; then
    echo "  [OK]   DDPM checkpoint: ${DDPM_CHECKPOINT_ARG}" >&2
else
    echo "  [MISS] DDPM checkpoint: ${DDPM_CHECKPOINT_ARG}" >&2
    PREFLIGHT_OK=0
fi

if [ "${PREFLIGHT_OK}" -eq 0 ]; then
    echo "" >&2
    echo "ERROR: One or more checkpoints not found. Aborting." >&2
    exit 1
fi
echo "" >&2

# Create output directory tree
mkdir -p "${NEUROIMF_DIR}" "${MOTFM_DIR}" "${DDPM_DIR}" "${COMPARE_DIR}"

# ========================================================================
# STEP 1: SUBMIT GENERATION JOBS (parallel — no dependencies on each other)
# ========================================================================
echo "==========================================" >&2
echo "STEP 1: SUBMITTING GENERATION JOBS" >&2
echo "==========================================" >&2

# --- NeuroiMF generation ---
echo "" >&2
echo "--- NeuroiMF Generation ---" >&2
NEUROMF_GEN_JOB=$(bash "${SLURM_ROOT}/generate/launch.sh" \
    --run-dir "${NEUROIMF_DIR}" \
    --n-samples "${N_SAMPLES}")
echo "  Job ID: ${NEUROMF_GEN_JOB}" >&2

# --- MOTFM generation ---
echo "" >&2
echo "--- MOTFM Generation ---" >&2
MOTFM_GEN_JOB=$(bash "${SLURM_ROOT}/generate_motfm/launch.sh" \
    --run-dir "${MOTFM_DIR}" \
    --checkpoint "${MOTFM_CHECKPOINT_ARG}" \
    --n-samples "${N_SAMPLES}")
echo "  Job ID: ${MOTFM_GEN_JOB}" >&2

# --- DDPM generation ---
echo "" >&2
echo "--- DDPM Generation ---" >&2
DDPM_GEN_JOB=$(bash "${SLURM_ROOT}/generate_ddpm/launch.sh" \
    --run-dir "${DDPM_DIR}" \
    --checkpoint "${DDPM_CHECKPOINT_ARG}" \
    --n-samples "${N_SAMPLES}")
echo "  Job ID: ${DDPM_GEN_JOB}" >&2

# ========================================================================
# STEP 2: SUBMIT EVALUATION JOBS (each depends on its generation job)
# ========================================================================
echo "" >&2
echo "==========================================" >&2
echo "STEP 2: SUBMITTING EVALUATION JOBS" >&2
echo "==========================================" >&2

# --- NeuroiMF evaluation ---
echo "" >&2
echo "--- NeuroiMF Evaluation ---" >&2
NEUROMF_EVAL_JOB=$(bash "${SLURM_ROOT}/evaluate/launch.sh" \
    --run-dir "${NEUROIMF_DIR}" \
    --depends-on "${NEUROMF_GEN_JOB}" \
    --synthseg-max-volumes "${SYNTHSEG_MAX_VOLUMES}")
echo "  Job ID: ${NEUROMF_EVAL_JOB}" >&2

# --- MOTFM evaluation ---
echo "" >&2
echo "--- MOTFM Evaluation ---" >&2
MOTFM_EVAL_JOB=$(bash "${SLURM_ROOT}/evaluate/launch.sh" \
    --run-dir "${MOTFM_DIR}" \
    --depends-on "${MOTFM_GEN_JOB}" \
    --synthseg-max-volumes "${SYNTHSEG_MAX_VOLUMES}")
echo "  Job ID: ${MOTFM_EVAL_JOB}" >&2

# --- DDPM evaluation ---
echo "" >&2
echo "--- DDPM Evaluation ---" >&2
DDPM_EVAL_JOB=$(bash "${SLURM_ROOT}/evaluate/launch.sh" \
    --run-dir "${DDPM_DIR}" \
    --depends-on "${DDPM_GEN_JOB}" \
    --synthseg-max-volumes "${SYNTHSEG_MAX_VOLUMES}")
echo "  Job ID: ${DDPM_EVAL_JOB}" >&2

# ========================================================================
# STEP 3: SUBMIT COMPARISON JOB (depends on all 3 evaluations)
# ========================================================================
echo "" >&2
echo "==========================================" >&2
echo "STEP 3: SUBMITTING COMPARISON JOB" >&2
echo "==========================================" >&2

COMPARE_ARGS=(
    --neuroimf-dir "${NEUROIMF_DIR}"
    --motfm-dir "${MOTFM_DIR}"
    --ddpm-dir "${DDPM_DIR}"
    --output-dir "${COMPARE_DIR}"
    --depends-on "${NEUROMF_EVAL_JOB}:${MOTFM_EVAL_JOB}:${DDPM_EVAL_JOB}"
    --n-bootstrap "${N_BOOTSTRAP}"
    --nfe ${NFE_LEVELS}
)

if [ "${SKIP_BOOTSTRAP}" -eq 1 ]; then
    COMPARE_ARGS+=(--skip-bootstrap)
fi

if [ "${SKIP_QUALITATIVE}" -eq 1 ]; then
    COMPARE_ARGS+=(--skip-qualitative)
fi

COMPARE_JOB=$(bash "${SLURM_ROOT}/compare/launch.sh" "${COMPARE_ARGS[@]}")
echo "  Job ID: ${COMPARE_JOB}" >&2

# ========================================================================
# SUMMARY
# ========================================================================
echo "" >&2
echo "==========================================" >&2
echo "ORCHESTRATION COMPLETE — ALL JOBS SUBMITTED" >&2
echo "==========================================" >&2
echo "" >&2
echo "Job dependency chain:" >&2
echo "" >&2
echo "  GENERATION (parallel):" >&2
echo "    NeuroiMF gen:  ${NEUROMF_GEN_JOB}" >&2
echo "    MOTFM gen:     ${MOTFM_GEN_JOB}" >&2
echo "    DDPM gen:      ${DDPM_GEN_JOB}" >&2
echo "" >&2
echo "  EVALUATION (after respective generation):" >&2
echo "    NeuroiMF eval: ${NEUROMF_EVAL_JOB}  (afterok:${NEUROMF_GEN_JOB})" >&2
echo "    MOTFM eval:    ${MOTFM_EVAL_JOB}  (afterok:${MOTFM_GEN_JOB})" >&2
echo "    DDPM eval:     ${DDPM_EVAL_JOB}  (afterok:${DDPM_GEN_JOB})" >&2
echo "" >&2
echo "  COMPARISON (after all evaluations):" >&2
echo "    Compare:       ${COMPARE_JOB}  (afterok:${NEUROMF_EVAL_JOB}:${MOTFM_EVAL_JOB}:${DDPM_EVAL_JOB})" >&2
echo "" >&2
echo "Output root: ${OUTPUT_ROOT}" >&2
echo "  NeuroiMF: ${NEUROIMF_DIR}/" >&2
echo "  MOTFM:    ${MOTFM_DIR}/" >&2
echo "  DDPM:     ${DDPM_DIR}/" >&2
echo "  Compare:  ${COMPARE_DIR}/" >&2
echo "" >&2
echo "Monitor all jobs:" >&2
echo "  squeue -j ${NEUROMF_GEN_JOB},${MOTFM_GEN_JOB},${DDPM_GEN_JOB},${NEUROMF_EVAL_JOB},${MOTFM_EVAL_JOB},${DDPM_EVAL_JOB},${COMPARE_JOB}" >&2
echo "" >&2
echo "Total jobs submitted: 7 (3 gen + 3 eval + 1 compare)" >&2
