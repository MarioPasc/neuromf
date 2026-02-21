#!/usr/bin/env bash
# =============================================================================
# PHASE 4: MEANFLOW TRAINING — SLURM LAUNCHER
#
# Login-node script that submits the Phase 4 training job on Picasso.
# Uses x-pred + exact JVP (best known config), requiring batch_size=2 per GPU.
#
# Expected time: ~400s/epoch x 1500 epochs = ~7 days on 6×A100
#
# Usage (from login node):
#   bash experiments/slurm/phase_4/train.sh              # 6 GPUs (default)
#   N_GPUS=4 bash experiments/slurm/phase_4/train.sh     # 4 GPUs (slower)
#   bash experiments/slurm/phase_4/train.sh --resume /path/to/checkpoint.ckpt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PHASE 4: MEANFLOW TRAINING LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_4_meanflow_training"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

# Number of GPUs — 6 for exact JVP (batch=2/GPU, accum=11, eff=132)
export N_GPUS="${N_GPUS:-6}"

# Optional: pass --resume /path/to/ckpt as argument
export RESUME_CKPT="${1:-}"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))

# Cap at DGX limits (128 cores, ~500GB)
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo "  GPUs:        ${N_GPUS}"
echo "  CPUs:        ${CPUS}"
echo "  Memory:      ${MEM}G"
echo "  Resume:      ${RESUME_CKPT:-none}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/training_checkpoints"
mkdir -p "${RESULTS_DST}/phase_4/logs"
mkdir -p "${RESULTS_DST}/phase_4/samples"
mkdir -p "${RESULTS_DST}/phase_4/diagnostics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
# Lightning handles multi-GPU process spawning internally — SLURM only
# needs 1 task with N GPUs allocated. No srun needed.
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p4_${N_GPUS}gpu" \
    --time=7-00:00:00 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}G" \
    --constraint=dgx \
    --gres="gpu:${N_GPUS}" \
    --output="${RESULTS_DST}/phase_4/train_%j.out" \
    --error="${RESULTS_DST}/phase_4/train_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/train_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "GPUs:      ${N_GPUS}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_4/train_${JOB_ID}.{out,err}"
echo "Checkpts:  ${RESULTS_DST}/training_checkpoints/"
echo "Samples:   ${RESULTS_DST}/phase_4/samples/"
echo ""
echo "After completion, check TensorBoard logs:"
echo "  tensorboard --logdir ${RESULTS_DST}/phase_4/logs"
