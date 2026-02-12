#!/usr/bin/env bash
#SBATCH -J toy_sweep_vmf
#SBATCH --array=0-1
#SBATCH --time=0-05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=toy_sweep_%A_%a.out
#SBATCH --error=toy_sweep_%A_%a.err

# =============================================================================
# MODULE 0C: SWEEP WORKER (Phase 1)
#
# SBATCH array job: each task runs Experiment 1 (x vs u sweep) for one manifold.
#   Task 0 → swiss_roll
#   Task 1 → toroid
#
# Produces:
#   $RESULTS_DST/sweep_{manifold}/toy_validation_report.json  (partial)
#   $RESULTS_DST/sweep_{manifold}/samples/*.npy
#   $RESULTS_DST/sweep_{manifold}/checkpoints/
#
# Expected env vars (exported by toy_validation.sh launcher):
#   REPO_SRC, RESULTS_DST, CONDA_ENV_NAME
#   MAX_EPOCHS, NUM_TRAIN, NUM_VAL, BATCH_SIZE, HIDDEN_DIM, NUM_LAYERS
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Sweep worker started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}, Array Task: ${SLURM_ARRAY_TASK_ID:-0}"

# ========================================================================
# MANIFOLD SELECTION
# ========================================================================
MANIFOLDS=(swiss_roll toroid)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MANIFOLD=${MANIFOLDS[$TASK_ID]}
echo "Assigned manifold: ${MANIFOLD} (task ${TASK_ID})"

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# ========================================================================
# ENVIRONMENT VERIFICATION
# ========================================================================
echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ========================================================================
# SMOKE TEST (only on task 0 to avoid redundancy)
# ========================================================================
if [ "$TASK_ID" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PRE-FLIGHT SMOKE TEST (task 0 only)"
    echo "=========================================="
    python -c "
import torch
from vmf.toy.toy_trainer import ToyMeanFlowTrainer, ToyTrainerConfig

cfg = ToyTrainerConfig(
    observation_dim=8, prediction_type='x', max_epochs=5,
    num_train_samples=200, num_val_samples=50, batch_size=64,
    hidden_dim=32, num_layers=2, seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
trainer = ToyMeanFlowTrainer(cfg)
result = trainer.train()
assert not result['diverged'], 'Smoke test FAILED: training diverged'
assert result['best_val_loss'] < 1e6, 'Smoke test FAILED: loss too large'
print(f'Smoke test PASSED: best_val_loss={result[\"best_val_loss\"]:.4f}')
"
    if [ $? -ne 0 ]; then
        echo "SMOKE TEST FAILED — aborting."
        exit 1
    fi
fi

# ========================================================================
# OUTPUT DIRECTORY
# ========================================================================
SWEEP_DIR="${RESULTS_DST}/sweep_${MANIFOLD}"
mkdir -p "${SWEEP_DIR}"
mkdir -p "${SWEEP_DIR}/samples"
mkdir -p "${SWEEP_DIR}/checkpoints"

# ========================================================================
# RUN EXPERIMENT 1 FOR THIS MANIFOLD
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING SWEEP: ${MANIFOLD}"
echo "=========================================="
echo "Max epochs:  ${MAX_EPOCHS}"
echo "Train/Val:   ${NUM_TRAIN}/${NUM_VAL}"
echo "Batch size:  ${BATCH_SIZE}"
echo "MLP:         ${HIDDEN_DIM}h x ${NUM_LAYERS}L"
echo "Output:      ${SWEEP_DIR}"
echo ""

cd "${REPO_SRC}"

python experiments/scripts/run_toy_3d.py \
    --output-dir "${SWEEP_DIR}" \
    --device cuda \
    --max-epochs "${MAX_EPOCHS}" \
    --num-train "${NUM_TRAIN}" \
    --num-val "${NUM_VAL}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --manifolds "${MANIFOLD}" \
    --skip-exp 2 3 4 \
    --skip-figures

SWEEP_EXIT=$?

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

EXPECTED_FILES=(
    "${SWEEP_DIR}/toy_validation_report.json"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $f (${SIZE} bytes)"
    else
        echo "[MISS] $f"
        MISSING=$((MISSING + 1))
    fi
done

# Check for .npy samples
NPY_COUNT=$(find "${SWEEP_DIR}/samples" -name "*.npy" 2>/dev/null | wc -l)
echo "Sample .npy files: ${NPY_COUNT}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "SWEEP WORKER COMPLETED"
echo "=========================================="
echo "Manifold:  ${MANIFOLD}"
echo "Finished:  $(date)"
echo "Duration:  $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code: ${SWEEP_EXIT}"

exit "${SWEEP_EXIT}"
