#!/usr/bin/env bash
#SBATCH -J neuromf_skull_strip
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# SKULL-STRIP DEFACED DATASETS — SLURM WORKER
#
# Runs HD-BET skull-stripping on defaced FOMO-60K datasets. In Phase B array
# mode, each worker processes a round-robin slice of the full work list
# (worker k handles items k, k+N, k+2N, ...).
#
# Expected env vars (exported by skull_strip.sh launcher):
#   REPO_SRC, FOMO60K_ROOT, RESULTS_DST, CONDA_ENV_NAME,
#   SKULL_STRIP_PHASE, NUM_WORKERS
#
# For array jobs, SLURM provides:
#   SLURM_ARRAY_TASK_ID — this worker's index (0-based)
# =============================================================================

set -euo pipefail

# Determine worker ID (array job or standalone)
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

START_TIME=$(date +%s)
echo "Skull-strip worker started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Phase: ${SKULL_STRIP_PHASE}"
echo "Worker: ${WORKER_ID}/${NUM_WORKERS}"

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

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count())"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify FOMO-60K root exists
if [ -d "${FOMO60K_ROOT}" ]; then
    echo "[OK]   FOMO-60K root: ${FOMO60K_ROOT}"
else
    echo "[MISS] FOMO-60K root: ${FOMO60K_ROOT}"
    echo "ERROR: FOMO-60K root not found. Aborting."
    exit 1
fi

# Verify metadata files
for f in "${FOMO60K_ROOT}/mapping.tsv" "${FOMO60K_ROOT}/participants.tsv"; do
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f"
        echo "ERROR: Required metadata file missing. Aborting."
        exit 1
    fi
done

# Verify HD-BET weights are available
python -c "
from brainles_hd_bet.utils import get_params_fname
missing = [i for i in range(5) if not get_params_fname(i).exists()]
if missing:
    print(f'[FAIL] HD-BET weights missing for folds: {missing}')
    print('       Run the launcher from the login node first to download weights.')
    exit(1)
else:
    print('[OK]   HD-BET weights: all 5 folds present')
"
if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — HD-BET weights not found."
    exit 1
fi

# Quick import check
python -c "
from brainles_preprocessing.brain_extraction.brain_extractor import HDBetExtractor
import nibabel, scipy, matplotlib, pandas
print('[OK]   All imports OK')
"

echo "Pre-flight checks PASSED"

# ========================================================================
# RUN SKULL-STRIPPING
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING SKULL-STRIP PHASE ${SKULL_STRIP_PHASE} (worker ${WORKER_ID}/${NUM_WORKERS})"
echo "=========================================="
echo "FOMO-60K root: ${FOMO60K_ROOT}"
echo ""

python scripts/skull_strip_defaced.py \
    --phase "${SKULL_STRIP_PHASE}" \
    --fomo60k-root "${FOMO60K_ROOT}" \
    --device 0 \
    --worker-id "${WORKER_ID}" \
    --num-workers "${NUM_WORKERS}"

SS_EXIT=$?

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "POST-FLIGHT (worker ${WORKER_ID})"
echo "=========================================="

if [ "${SKULL_STRIP_PHASE}" == "B" ]; then
    # Count brain masks created per dataset (only worker 0 to avoid spam)
    if [ "${WORKER_ID}" == "0" ]; then
        for ds in PT011_MBSR PT012_UCLA PT015_NKI; do
            MASK_COUNT=$(find "${FOMO60K_ROOT}/${ds}" -name "*_brainmask.nii.gz" 2>/dev/null | wc -l)
            T1_COUNT=$(find "${FOMO60K_ROOT}/${ds}" -name "t1*.nii.gz" ! -name "*_brainmask*" ! -name "*.bak" 2>/dev/null | wc -l)
            echo "  ${ds}: ${MASK_COUNT} masks / ${T1_COUNT} T1 files"
        done
    fi
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "WORKER ${WORKER_ID} COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code:  ${SS_EXIT}"

if [ "$SS_EXIT" -eq 0 ]; then
    echo "Worker ${WORKER_ID} completed successfully."
else
    echo "Worker ${WORKER_ID} FAILED with exit code ${SS_EXIT}."
    exit "${SS_EXIT}"
fi
