#!/usr/bin/env bash
#SBATCH -J motfm_data_prep
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --partition=cpu
#SBATCH --output=prepare_data_%j.out
#SBATCH --error=prepare_data_%j.err

# =============================================================================
# MOTFM DATA PREPARATION WORKER
#
# Converts FOMO-60K NIfTI volumes to MOTFM pickle format.
# Uses same splits as Phase 1 (split_manifest.json).
#
# Expected env vars (exported by prepare_data.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME,
#   MOTFM_OUTPUT, SPLIT_MANIFEST
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "MOTFM data preparation started at: $(date)"
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
python -c "import torch; print('PyTorch', torch.__version__)"

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify split manifest
if [ -f "${SPLIT_MANIFEST}" ]; then
    echo "[OK]   Split manifest: ${SPLIT_MANIFEST}"
else
    echo "[MISS] Split manifest not found: ${SPLIT_MANIFEST}"
    exit 1
fi

# Verify neuromf is importable
python -c "
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
print('neuromf imports OK')
"

echo "Memory available: $(free -h | awk '/^Mem:/{print \$2}')"

# ========================================================================
# CONVERT DATA
# ========================================================================
echo ""
echo "=========================================="
echo "CONVERTING FOMO-60K → MOTFM HDF5"
echo "=========================================="

# Output base path (no extension — prepare_data.py creates .h5 and optionally .pkl)
MOTFM_OUTPUT_BASE="${MOTFM_OUTPUT%.pkl}"
MOTFM_OUTPUT_BASE="${MOTFM_OUTPUT_BASE%.h5}"

python -u experiments/MOTFM/prepare_data.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --output-path "${MOTFM_OUTPUT_BASE}" \
    --split-manifest "${SPLIT_MANIFEST}"

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

H5_OUTPUT="${MOTFM_OUTPUT_BASE}.h5"
if [ -f "${H5_OUTPUT}" ]; then
    SIZE=$(stat -c%s "${H5_OUTPUT}" 2>/dev/null || echo "?")
    SIZE_GB=$(echo "scale=2; ${SIZE} / 1073741824" | bc 2>/dev/null || echo "?")
    echo "[OK]   ${H5_OUTPUT} (${SIZE_GB} GB)"

    # Verify HDF5 structure
    python -c "
import h5py
with h5py.File('${H5_OUTPUT}', 'r') as f:
    for key in sorted(f.keys()):
        dset = f[key]
        g_min = dset.attrs.get('global_min', '?')
        g_max = dset.attrs.get('global_max', '?')
        print(f'  /{key}: shape={dset.shape}, dtype={dset.dtype}, min={g_min}, max={g_max}')
"
else
    echo "[MISS] HDF5 output not found: ${H5_OUTPUT}"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "MOTFM DATA PREPARATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
