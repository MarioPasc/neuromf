#!/usr/bin/env bash
# =============================================================================
# MODULE 0C: TOY VALIDATION — SLURM LAUNCHER
#
# Login-node script that submits the Module 0C experiment pipeline.
#
# Parallel mode (default):
#   Phase 1: SBATCH array[0-1] — Exp 1 sweep, one manifold per task (~4h each)
#   Phase 2: SBATCH dependency — merge + Exps 2-4 + figures/HTML (~2h)
#   Total: ~6h wall time (vs ~8h serial)
#
# Serial mode (--serial):
#   Runs everything in a single SBATCH job (old behavior).
#
# Usage:
#   bash experiments/slurm/toy_validation.sh              # parallel (default)
#   bash experiments/slurm/toy_validation.sh --serial     # serial fallback
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
MODE="parallel"
if [ "${1:-}" = "--serial" ]; then
    MODE="serial"
fi

echo "=========================================="
echo "MODULE 0C: TOY VALIDATION LAUNCHER"
echo "=========================================="
echo "Mode:     ${MODE}"
echo "Time:     $(date)"
echo ""

# ========================================================================
# SHARED CONFIGURATION (exported for worker/finalize scripts)
# ========================================================================
export EXPERIMENT_NAME="module_0C_toy_validation"
export CONDA_ENV_NAME="vmf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/vMF"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"

# Training hyperparameters
export MAX_EPOCHS=500
export NUM_TRAIN=10000
export NUM_VAL=2000
export BATCH_SIZE=256
export HIDDEN_DIM=256
export NUM_LAYERS=7

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo "  Max epochs:  ${MAX_EPOCHS}"
echo "  Train/Val:   ${NUM_TRAIN}/${NUM_VAL}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  MLP:         ${HIDDEN_DIM}h x ${NUM_LAYERS}L"
echo ""

# Create results directory
mkdir -p "${RESULTS_DST}"

# ========================================================================
# SERIAL MODE — Single SBATCH job (old behavior)
# ========================================================================
if [ "${MODE}" = "serial" ]; then
    echo "Submitting serial job (all experiments in one job)..."

    JOB_SERIAL=$(sbatch --parsable \
        --job-name="toy_validation_vmf_serial" \
        --time=0-10:00:00 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --constraint=dgx \
        --gres=gpu:1 \
        --output="${RESULTS_DST}/toy_serial_%j.out" \
        --error="${RESULTS_DST}/toy_serial_%j.err" \
        --export=ALL \
        --wrap="
set -euo pipefail

# Environment setup
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi \"^\${m}[[:space:]]\"; then
    module load \"\$m\" && module_loaded=1 && break
  fi
done
if [ \"\$module_loaded\" -eq 0 ]; then
  echo '[env] No conda module loaded; assuming conda already in PATH.'
fi
if command -v conda >/dev/null 2>&1; then
  source \"\$(conda info --base)/etc/profile.d/conda.sh\" || true
  conda activate \"${CONDA_ENV_NAME}\" 2>/dev/null || source activate \"${CONDA_ENV_NAME}\"
else
  source activate \"${CONDA_ENV_NAME}\"
fi

echo 'Environment: '
which python
python -c 'import torch; print(\"PyTorch\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,memory.total --format=csv

cd \"${REPO_SRC}\"
python experiments/scripts/run_toy_3d.py \\
    --output-dir \"${RESULTS_DST}\" \\
    --device cuda \\
    --max-epochs ${MAX_EPOCHS} \\
    --num-train ${NUM_TRAIN} \\
    --num-val ${NUM_VAL} \\
    --batch-size ${BATCH_SIZE} \\
    --hidden-dim ${HIDDEN_DIM} \\
    --num-layers ${NUM_LAYERS} \\
    --manifolds swiss_roll toroid
")

    echo "Serial job submitted: ${JOB_SERIAL}"
    echo "Monitor with: squeue -j ${JOB_SERIAL}"
    echo "Logs: ${RESULTS_DST}/toy_serial_${JOB_SERIAL}.{out,err}"
    exit 0
fi

# ========================================================================
# PARALLEL MODE — 2-phase pipeline
# ========================================================================
echo "Submitting parallel pipeline..."
echo ""

# Phase 1: Array job — one task per manifold
JOB1=$(sbatch --parsable \
    --output="${RESULTS_DST}/toy_sweep_%A_%a.out" \
    --error="${RESULTS_DST}/toy_sweep_%A_%a.err" \
    --export=ALL \
    "${SCRIPT_DIR}/toy_sweep_worker.sh")

echo "Phase 1 (sweep array): Job ${JOB1}"
echo "  Task 0 = swiss_roll"
echo "  Task 1 = toroid"
echo ""

# Phase 2: Finalize — runs after all array tasks complete
JOB2=$(sbatch --parsable \
    --dependency=afterok:${JOB1} \
    --output="${RESULTS_DST}/toy_finalize_%j.out" \
    --error="${RESULTS_DST}/toy_finalize_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/toy_finalize.sh")

echo "Phase 2 (merge + finalize): Job ${JOB2} (after ${JOB1})"
echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo "=========================================="
echo "PIPELINE SUBMITTED"
echo "=========================================="
echo "Phase 1 (sweep):    Job ${JOB1} (array 0-1, ~4h each)"
echo "Phase 2 (finalize): Job ${JOB2} (dependency: afterok:${JOB1}, ~2h)"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB1},${JOB2}"
echo "  tail -f ${RESULTS_DST}/toy_sweep_${JOB1}_*.out"
echo ""
echo "Results will be in: ${RESULTS_DST}/"
echo "  sweep_swiss_roll/  — Phase 1 partial (task 0)"
echo "  sweep_toroid/      — Phase 1 partial (task 1)"
echo "  figures/           — Phase 2 final figures"
echo "  toy_validation_report.json  — Combined report"
echo "  toy_validation_report.html  — Combined HTML"
