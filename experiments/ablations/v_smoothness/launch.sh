#!/usr/bin/env bash
# =============================================================================
# VELOCITY SMOOTHNESS ABLATION LAUNCHER
#
# Submits the neuro-iMF model with the v-head Jacobian smoothness regularizer
# enabled (doc2). Inherits all settings from the best known config (x-pred,
# exact JVP, t_h conditioning, v-head, 3000 epochs) and adds:
#   L_total = L_mf + 0.01 * ||dv/dz_t||_F^2 / d
#
# The FD Hutchinson estimator adds one extra v_fn forward per step (~+2 GB),
# fitting within batch_size=2 on A100 40GB with exact JVP.
#
# Usage (from login node):
#   bash experiments/ablations/v_smoothness/launch.sh
#   bash experiments/ablations/v_smoothness/launch.sh --resume /path/to/ckpt
#   bash experiments/ablations/v_smoothness/launch.sh --depends-on 12345
#   N_GPUS=4 bash experiments/ablations/v_smoothness/launch.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/../../slurm/train/worker.sh"

# Validate worker script exists
if [ ! -f "${WORKER_SCRIPT}" ]; then
    # Fallback to phase_4 worker
    WORKER_SCRIPT="${SCRIPT_DIR}/../../slurm/phase_4/train_worker.sh"
    if [ ! -f "${WORKER_SCRIPT}" ]; then
        echo "ERROR: Worker script not found." >&2
        echo "  Tried: ${SCRIPT_DIR}/../../slurm/train/worker.sh" >&2
        echo "  Tried: ${SCRIPT_DIR}/../../slurm/phase_4/train_worker.sh" >&2
        exit 1
    fi
fi

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
export RESUME_CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            export RESUME_CKPT="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash experiments/ablations/v_smoothness/launch.sh [--resume CKPT] [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "VELOCITY SMOOTHNESS ABLATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="v_smoothness_ablation"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Config chain: Picasso overlay + smoothness ablation overlay
export TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${SCRIPT_DIR}/config.yaml"

# Number of GPUs — 6 for exact JVP (batch=2/GPU, accum=11, eff=132)
export N_GPUS="${N_GPUS:-6}"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))

# Cap at DGX limits (128 cores, ~500GB)
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${TRAIN_CONFIG}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "  CPUs:        ${CPUS}" >&2
echo "  Memory:      ${MEM}G" >&2
echo "  Resume:      ${RESUME_CKPT:-none}" >&2
echo "" >&2

# SLURM log output directory (run dir is created by train.py)
ABL_DIR="${RESULTS_DST}/ablations/v_smoothness"
mkdir -p "${ABL_DIR}"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_v_smooth"
    --time=7-00:00:00
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}G"
    --constraint=dgx
    --gres="gpu:${N_GPUS}"
    --output="${ABL_DIR}/train_%j.out"
    --error="${ABL_DIR}/train_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${WORKER_SCRIPT}")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "GPUs:      ${N_GPUS}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${ABL_DIR}/train_${JOB_ID}.{out,err}" >&2
echo "Checkpts:  ${ABL_DIR}/checkpoints/" >&2
echo "Diag:      ${ABL_DIR}/diagnostics/" >&2
echo "" >&2
echo "After completion, compare against baseline:" >&2
echo "  tensorboard --logdir_spec baseline:${RESULTS_DST}/phase_4/logs,smooth:${ABL_DIR}/logs" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
