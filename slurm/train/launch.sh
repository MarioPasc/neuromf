#!/usr/bin/env bash
# =============================================================================
# MEANFLOW TRAINING — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the training job on Picasso.
# Uses x-pred + exact JVP (best known config), requiring batch_size=2 per GPU.
#
# Expected time: ~980s/epoch on 3xA100 (8K+ volumes), 5-day wall limit.
# Early stopping (patience=150 epochs) + 3D-FID checkpointing should
# converge well within the 5-day window (model peaks ~200, stops ~350).
#
# Usage (from login node):
#   bash slurm/train/launch.sh
#   N_GPUS=4 bash slurm/train/launch.sh
#   bash slurm/train/launch.sh --resume /path/to/checkpoint.ckpt
#   bash slurm/train/launch.sh --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
            echo "Usage: bash slurm/train/launch.sh [--resume CKPT] [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "MEANFLOW TRAINING LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_4_meanflow_training"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Number of GPUs — 3 for exact JVP (batch=2/GPU, accum=22, eff=132)
# Using 3 GPUs instead of 6 to avoid idle GPU waste from rank-0-only
# evaluation callbacks (FID, SWD, sample collection). Same effective
# batch size; training takes ~2x per epoch but total epochs are bounded
# by early stopping (~350 epochs, ~4 days).
export N_GPUS="${N_GPUS:-3}"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))

# Cap at DGX limits (128 cores, ~500GB)
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${CONFIGS_DIR}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "  CPUs:        ${CPUS}" >&2
echo "  Memory:      ${MEM}G" >&2
echo "  Resume:      ${RESUME_CKPT:-none}" >&2
echo "" >&2

# Create SLURM log output directory (run dir is created by train.py)
mkdir -p "${RESULTS_DST}/phase_4"

# ========================================================================
# SUBMIT JOB
# ========================================================================
# Lightning handles multi-GPU process spawning internally — SLURM only
# needs 1 task with N GPUs allocated. No srun needed.
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p4_${N_GPUS}gpu"
    --time=5-00:00:00
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}G"
    --constraint=dgx
    --gres="gpu:${N_GPUS}"
    --output="${RESULTS_DST}/phase_4/train_%j.out"
    --error="${RESULTS_DST}/phase_4/train_%j.err"
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
echo "GPUs:      ${N_GPUS}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RESULTS_DST}/phase_4/train_${JOB_ID}.{out,err}" >&2
echo "Run dir:   ${RESULTS_DST}/runs/run_<timestamp>/" >&2
echo "" >&2
echo "After completion:" >&2
echo "  ls ${RESULTS_DST}/runs/       # find your run directory" >&2
echo "  tensorboard --logdir ${RESULTS_DST}/runs/run_*/logs" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
