#!/usr/bin/env bash
#SBATCH -J competitor_gen
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# GENERIC COMPETITOR GENERATION WORKER
#
# Model-agnostic: uses the competitor registry via generate_competitor.py.
#
# Expected env vars (from launch.sh):
#   REPO_SRC, CONFIGS_DIR, CONDA_ENV_NAME,
#   COMPETITOR_MODEL, COMPETITOR_CONFIG, COMPETITOR_CHECKPOINT,
#   COMPETITOR_RUN_DIR, GEN_N_SAMPLES, GEN_VOLUME_SIZE
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Competitor generation started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Model: ${COMPETITOR_MODEL}"
echo "Run directory: ${COMPETITOR_RUN_DIR}"

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module."

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"

cd "${REPO_SRC}"

# PYTHONPATH: include external vendored repos + experiment dirs
export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${REPO_SRC}/experiments/DDPM:${REPO_SRC}/experiments/MOTFM:${PYTHONPATH:-}"

N_SAMPLES="${GEN_N_SAMPLES:-500}"
GEN_DIR="${COMPETITOR_RUN_DIR}/generation"
VOL_DIR="${GEN_DIR}/volumes"
mkdir -p "${VOL_DIR}"

# ========================================================================
# PRE-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

[ ! -f "${COMPETITOR_CHECKPOINT}" ] && echo "FATAL: Checkpoint not found: ${COMPETITOR_CHECKPOINT}" >&2 && exit 1
echo "[OK] Checkpoint: ${COMPETITOR_CHECKPOINT}"
[ ! -f "${COMPETITOR_CONFIG}" ] && echo "FATAL: Config not found: ${COMPETITOR_CONFIG}" >&2 && exit 1
echo "[OK] Config: ${COMPETITOR_CONFIG}"

# Quick import check
python -c "
from neuromf.competitors import CompetitorRegistry
names = CompetitorRegistry.list_names()
print(f'Registered competitors: {names}')
assert '${COMPETITOR_MODEL}'.lower() in [n.lower() for n in names], \
    f'Model \"${COMPETITOR_MODEL}\" not registered. Available: {names}'
print('OK')
"

# ========================================================================
# STAGE 1: PREPARE REAL TEST VOLUMES
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 1: PREPARING REAL TEST VOLUMES"
echo "=========================================="

python -u experiments/cli/prepare_real_test.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --output-dir "${GEN_DIR}" 2>&1 || echo "[WARN] prepare_real_test failed (may already exist)"

# ========================================================================
# STAGE 2: COMPETITOR GENERATION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: ${COMPETITOR_MODEL} GENERATION"
echo "=========================================="

GEN_ARGS=(
    experiments/cli/generate_competitor.py
    --model "${COMPETITOR_MODEL}"
    --config "${COMPETITOR_CONFIG}"
    --checkpoint "${COMPETITOR_CHECKPOINT}"
    --nfe 1 10 50
    --n-samples "${N_SAMPLES}"
    --batch-size 1
    --output-dir "${VOL_DIR}"
    --seed 42
)

if [ -n "${GEN_VOLUME_SIZE:-}" ]; then
    GEN_ARGS+=(--volume-size "${GEN_VOLUME_SIZE}")
fi

python -u "${GEN_ARGS[@]}"

# ========================================================================
# STAGE 3: VISUALIZATION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 3: VISUALIZATION"
echo "=========================================="

python -u experiments/cli/visualize_generation.py \
    --generation-dir "${GEN_DIR}" \
    --output-dir "${GEN_DIR}/figures" \
    --n-subjects 3 \
    --nfe 1 10 50 2>&1 || echo "[WARN] visualization failed (non-critical)"

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

for nfe in 1 10 50; do
    f="${VOL_DIR}/nfe_$(printf '%03d' $nfe).h5"
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (${SIZE} bytes)"
    else
        echo "[MISS] nfe_$(printf '%03d' $nfe).h5"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "${COMPETITOR_MODEL} GENERATION COMPLETED"
echo "=========================================="
echo "Run dir:  ${COMPETITOR_RUN_DIR}"
echo "Finished: $(date)"
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
