#!/usr/bin/env bash
# =============================================================================
# FIGURE GENERATION WORKER
#
# Generates all training figures (evolution plots + VAE-decoded NFE grids)
# from a sample archive. Lightweight job: 1 GPU, ~15 min with decode.
#
# Expected env vars (exported by generate_figures.sh):
#   REPO_SRC, CONFIGS_DIR, CONDA_ENV_NAME
#   SAMPLES_DIR:  path to samples/ containing sample_archive.pt
#   FIGURES_DIR:  output path for figures
#   SKIP_DECODE:  "--skip-decode" or "" (empty = do decode)
# =============================================================================

set -euo pipefail

echo "Figure generation started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

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

echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Devices:', torch.cuda.device_count())"

# ========================================================================
# PRE-FLIGHT
# ========================================================================
cd "${REPO_SRC}"

if [ ! -f "${SAMPLES_DIR}/sample_archive.pt" ]; then
    echo "ERROR: sample_archive.pt not found at ${SAMPLES_DIR}"
    exit 1
fi
echo "[OK] Archive: ${SAMPLES_DIR}/sample_archive.pt"

ARCHIVE_SIZE=$(stat -c%s "${SAMPLES_DIR}/sample_archive.pt" 2>/dev/null || echo "?")
echo "     Size: ${ARCHIVE_SIZE} bytes"

# ========================================================================
# RUN FIGURE GENERATION
# ========================================================================
echo ""
echo "=========================================="
echo "GENERATING FIGURES"
echo "=========================================="
echo "Samples: ${SAMPLES_DIR}"
echo "Output:  ${FIGURES_DIR}"
echo "Decode:  $([ -z "${SKIP_DECODE:-}" ] && echo 'yes' || echo 'no')"
echo ""

CMD="python experiments/cli/generate_figures.py \
    --samples-dir ${SAMPLES_DIR} \
    --output-dir ${FIGURES_DIR}"

if [ -z "${SKIP_DECODE:-}" ]; then
    CMD="${CMD} --config ${CONFIGS_DIR}/train_meanflow.yaml --configs-dir ${CONFIGS_DIR}"
else
    CMD="${CMD} --skip-decode"
fi

eval "${CMD}"
EXIT_CODE=$?

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

FIG_COUNT=$(find "${FIGURES_DIR}" -name "*.png" -o -name "*.pdf" 2>/dev/null | wc -l)
echo "Generated figures: ${FIG_COUNT}"
ls -la "${FIGURES_DIR}"/ 2>/dev/null || true

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
exit "${EXIT_CODE}"
