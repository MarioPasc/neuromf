#!/usr/bin/env bash
# =============================================================================
# PHASE 4: MEANFLOW TRAINING — SLURM LAUNCHER
#
# Login-node script that submits the Phase 4 training job on Picasso.
# Uses a single A100 GPU to train the Latent MeanFlow model on
# pre-computed latents (~1100 volumes, 500 epochs).
#
# Expected time: ~24-48h on A100
#
# Usage (from login node):
#   bash experiments/slurm/phase_4/train.sh
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

# Optional: pass --resume /path/to/ckpt as argument
export RESUME_CKPT="${1:-}"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo "  Resume:      ${RESUME_CKPT:-none}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/training_checkpoints"
mkdir -p "${RESULTS_DST}/phase_4/logs"
mkdir -p "${RESULTS_DST}/phase_4/samples"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p4_train" \
    --time=2-00:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/phase_4/train_%j.out" \
    --error="${RESULTS_DST}/phase_4/train_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/train_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_4/train_${JOB_ID}.{out,err}"
echo "Checkpts:  ${RESULTS_DST}/training_checkpoints/"
echo "Samples:   ${RESULTS_DST}/phase_4/samples/"
echo ""
echo "After completion, check TensorBoard logs:"
echo "  tensorboard --logdir ${RESULTS_DST}/phase_4/logs"
