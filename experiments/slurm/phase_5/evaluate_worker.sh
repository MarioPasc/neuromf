#!/usr/bin/env bash
#SBATCH -J neuromf_p5_eval
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate_%j.out
#SBATCH --error=evaluate_%j.err

# =============================================================================
# PHASE 5: EVALUATION WORKER
#
# Stage 1: Extract Med3D features (real + generated)
# Stage 2: Compute metrics (FID, MMD, Coverage, Density, MS-SSIM, PSNR,
#           spectral, SynthSeg morphological)
#
# Expected env vars (exported by evaluate.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 5 evaluation started at: $(date)"
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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"

cd "${REPO_SRC}"

GEN_DIR="${RESULTS_DST}/phase_5/generation"
FEAT_DIR="${RESULTS_DST}/phase_5/features"
METRICS_DIR="${RESULTS_DST}/phase_5/metrics"
SYNTHSEG_DIR="${METRICS_DIR}/synthseg"

# ========================================================================
# PRE-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

for nfe in 1 10 50; do
    f="${GEN_DIR}/volumes/nfe_$(printf '%03d' $nfe).h5"
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f — run generate.sh first"
        exit 1
    fi
done

if [ -f "${GEN_DIR}/real_test.h5" ]; then
    echo "[OK]   ${GEN_DIR}/real_test.h5"
else
    echo "[MISS] ${GEN_DIR}/real_test.h5 — run generate.sh first"
    exit 1
fi

# ========================================================================
# STAGE 1: FEATURE EXTRACTION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 1: FEATURE EXTRACTION"
echo "=========================================="

# Extract Med3D features for real test set and generated volumes
python -c "
import torch
from pathlib import Path
from omegaconf import OmegaConf

from neuromf.metrics.feature_extractor import FeatureExtractor

# Load config to get weights path
configs_dir = Path('${CONFIGS_DIR}')
layers = []
base = configs_dir / 'base.yaml'
if base.exists():
    layers.append(OmegaConf.load(base))
gen_yaml = configs_dir / 'generate.yaml'
if gen_yaml.exists():
    layers.append(OmegaConf.load(gen_yaml))
config = OmegaConf.merge(*layers)
OmegaConf.resolve(config)

med3d_weights = config.features.med3d.weights_path
print(f'Med3D weights: {med3d_weights}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = FeatureExtractor('med3d', med3d_weights, device)

# Real features
real_vol = Path('${GEN_DIR}/real_test.h5')
real_feat = Path('${FEAT_DIR}/real_med3d.h5')
if not real_feat.exists():
    extractor.extract_and_cache(real_vol, real_feat)
    print('Real features extracted.')
else:
    print('Real features already cached.')

# Generated features for each NFE
for nfe in [1, 10, 50]:
    gen_vol = Path(f'${GEN_DIR}/volumes/nfe_{nfe:03d}.h5')
    gen_feat = Path(f'${FEAT_DIR}/gen_med3d_nfe{nfe:03d}.h5')
    if gen_vol.exists() and not gen_feat.exists():
        extractor.extract_and_cache(gen_vol, gen_feat)
        print(f'NFE={nfe} features extracted.')
    else:
        print(f'NFE={nfe} features already cached or volume missing.')
"

echo "Feature extraction complete."

# ========================================================================
# STAGE 2: METRICS COMPUTATION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: METRICS COMPUTATION"
echo "=========================================="

python experiments/cli/compute_metrics.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --volumes-dir "${GEN_DIR}/volumes" \
    --real-features-dir "${FEAT_DIR}" \
    --real-volumes-h5 "${GEN_DIR}/real_test.h5" \
    --nfe 1 10 50 \
    --output-dir "${METRICS_DIR}"

echo "Metrics computation complete."

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

for f in "${FEAT_DIR}"/*.h5; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $(basename $f) (${SIZE} bytes)"
    fi
done

for f in "${METRICS_DIR}"/*.json; do
    if [ -f "$f" ]; then
        echo "[OK]   $(basename $f)"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 5 EVALUATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
