#!/usr/bin/env bash
#SBATCH -J neuromf_p5_eval
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=evaluate_%j.out
#SBATCH --error=evaluate_%j.err

# =============================================================================
# EVALUATION WORKER (generic — works for NeuroiMF and MOTFM)
#
# Supports both standard mode (volumes/) and comparison mode
# (volumes_baseline/ + volumes_enhanced/).
#
# Stage 1: Extract R3D-18 features (real + generated) — MOTFM protocol
# Stage 2: Compute metrics (FID, MMD, Coverage, Density, MS-SSIM, PSNR,
#           spectral, SynthSeg morphological)
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONFIGS_DIR, RUN_DIR, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 5 evaluation started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Run directory: ${RUN_DIR}"

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

GEN_DIR="${RUN_DIR}/generation"
FEAT_DIR="${RUN_DIR}/features"
METRICS_DIR="${RUN_DIR}/metrics"

# ========================================================================
# DETECT COMPARISON MODE
# ========================================================================
# Same pattern as slurm/generate/worker.sh: if both volumes_baseline/ and
# volumes_enhanced/ exist, evaluate each variant separately.
if [ -d "${GEN_DIR}/volumes_baseline" ] && [ -d "${GEN_DIR}/volumes_enhanced" ]; then
    VARIANTS=("baseline" "enhanced")
    echo "Comparison mode detected — evaluating both baseline and enhanced variants"
else
    VARIANTS=("")
    echo "Standard mode — evaluating volumes/"
fi

# ========================================================================
# PRE-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

for VARIANT in "${VARIANTS[@]}"; do
    if [ -n "${VARIANT}" ]; then
        VOL_DIR="${GEN_DIR}/volumes_${VARIANT}"
        echo "--- Variant: ${VARIANT} ---"
    else
        VOL_DIR="${GEN_DIR}/volumes"
    fi

    for nfe in 1 10 50; do
        f="${VOL_DIR}/nfe_$(printf '%03d' $nfe).h5"
        if [ -f "$f" ]; then
            echo "[OK]   $f"
        else
            echo "[MISS] $f — run generate first"
            exit 1
        fi
    done
done

if [ -f "${GEN_DIR}/real_test.h5" ]; then
    echo "[OK]   ${GEN_DIR}/real_test.h5"
else
    echo "[MISS] ${GEN_DIR}/real_test.h5 — run generate first"
    exit 1
fi

# ========================================================================
# EVALUATE EACH VARIANT
# ========================================================================
for VARIANT in "${VARIANTS[@]}"; do

# Resolve paths for this variant.
# In comparison mode, generated features go into a variant subdirectory
# (features/baseline/, features/enhanced/) so that compute_metrics.py
# can find them at the standard name (gen_med3d_nfe{nfe:03d}.h5).
# Real features (real_med3d.h5) stay in the top-level features/ dir.
if [ -n "${VARIANT}" ]; then
    VOL_DIR="${GEN_DIR}/volumes_${VARIANT}"
    GEN_FEAT_DIR="${FEAT_DIR}/${VARIANT}"
    VARIANT_METRICS_DIR="${METRICS_DIR}/${VARIANT}"
    echo ""
    echo "=========================================="
    echo "EVALUATING VARIANT: ${VARIANT}"
    echo "=========================================="
else
    VOL_DIR="${GEN_DIR}/volumes"
    GEN_FEAT_DIR="${FEAT_DIR}"
    VARIANT_METRICS_DIR="${METRICS_DIR}"
fi

mkdir -p "${GEN_FEAT_DIR}"
mkdir -p "${VARIANT_METRICS_DIR}"

# ========================================================================
# STAGE 1: FEATURE EXTRACTION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 1: FEATURE EXTRACTION${VARIANT:+ (${VARIANT})}"
echo "=========================================="

python -u -c "
import time
import torch
import h5py
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from neuromf.metrics.feature_extractor import FeatureExtractor

# -- Config --
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Diagnostics --
print()
print('--- Feature Extraction Diagnostics ---')
print(f'  Device:          {device}')
if device.type == 'cuda':
    print(f'  GPU:             {torch.cuda.get_device_name(device)}')
    mem_gb = torch.cuda.get_device_properties(device).total_mem / 1e9 if hasattr(torch.cuda.get_device_properties(device), 'total_mem') else torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f'  GPU memory:      {mem_gb:.1f} GB')

# -- Load feature extractor --
weights_path = config.features.med3d.get('weights_path', '')
print(f'  Config weights:  {weights_path or \"(empty -- torchvision auto-download)\"}')

extractor = FeatureExtractor('med3d', weights_path, device)
model = extractor.model

arch = type(model).__name__
n_params = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    dummy = torch.randn(1, 3, 16, 16, 16, device=device)
    dummy_out = model(dummy)
    feature_dim = dummy_out.shape[-1]

print(f'  Architecture:    {arch}')
print(f'  Parameters:      {n_params / 1e6:.2f}M')
print(f'  Feature dim:     {feature_dim}')
variant = '${VARIANT}'
if variant:
    print(f'  Variant:         {variant}')
print('--- End Diagnostics ---')
print()

# -- Helper: log volume archive info --
def log_h5_info(path, label):
    if not path.exists():
        print(f'  [{label}] {path.name} -- NOT FOUND')
        return 0
    with h5py.File(str(path), 'r') as f:
        shape = f['volumes'].shape
        dtype = f['volumes'].dtype
        size_mb = path.stat().st_size / 1e6
    print(f'  [{label}] {path.name}: {shape}, dtype={dtype}, {size_mb:.1f} MB')
    return shape[0]

# -- Extract features --
vol_dir = Path('${VOL_DIR}')
gen_feat_dir = Path('${GEN_FEAT_DIR}')
real_feat_dir = Path('${FEAT_DIR}')
real_vol = Path('${GEN_DIR}/real_test.h5')
real_feat = real_feat_dir / 'real_med3d.h5'

print('Input volumes:')
log_h5_info(real_vol, 'real')
for nfe in [1, 10, 50]:
    log_h5_info(vol_dir / f'nfe_{nfe:03d}.h5', f'nfe={nfe}')
print()

# Real features (shared across variants — extract once, reuse)
if not real_feat.exists():
    t0 = time.time()
    feats = extractor.extract_and_cache(real_vol, real_feat)
    dt = time.time() - t0
    print(f'Real features extracted: {feats.shape} in {dt:.1f}s')
else:
    with h5py.File(str(real_feat), 'r') as f:
        cached_shape = f['features'].shape
        cached_backend = f.attrs.get('backend', 'unknown')
    print(f'Real features already cached: {cached_shape}, backend={cached_backend}')

# Generated features for each NFE (standard name in variant subdir)
gen_feat_dir.mkdir(parents=True, exist_ok=True)
for nfe in [1, 10, 50]:
    gen_vol = vol_dir / f'nfe_{nfe:03d}.h5'
    gen_feat = gen_feat_dir / f'gen_med3d_nfe{nfe:03d}.h5'
    if gen_vol.exists() and not gen_feat.exists():
        t0 = time.time()
        feats = extractor.extract_and_cache(gen_vol, gen_feat)
        dt = time.time() - t0
        print(f'NFE={nfe} features extracted: {feats.shape} in {dt:.1f}s')
    elif gen_feat.exists():
        with h5py.File(str(gen_feat), 'r') as f:
            cached_shape = f['features'].shape
            cached_backend = f.attrs.get('backend', 'unknown')
        print(f'NFE={nfe} features already cached: {cached_shape}, backend={cached_backend}')
    else:
        print(f'NFE={nfe} volume missing -- skipping.')
"

echo "Feature extraction complete${VARIANT:+ (${VARIANT})}."

# ========================================================================
# STAGE 2: METRICS COMPUTATION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: METRICS COMPUTATION${VARIANT:+ (${VARIANT})}"
echo "=========================================="

# Build metrics CLI args
METRICS_ARGS=(
    experiments/cli/compute_metrics.py
    --config "${CONFIGS_DIR}/generate.yaml"
    --configs-dir "${CONFIGS_DIR}"
    --volumes-dir "${VOL_DIR}"
    --real-features-dir "${FEAT_DIR}"
    --gen-features-dir "${GEN_FEAT_DIR}"
    --real-volumes-h5 "${GEN_DIR}/real_test.h5"
    --nfe 1 10 50
    --output-dir "${VARIANT_METRICS_DIR}"
)

# Limit SynthSeg to N volumes (default 100) to save compute
if [ -n "${SYNTHSEG_MAX_VOLUMES:-}" ] && [ "${SYNTHSEG_MAX_VOLUMES}" -gt 0 ] 2>/dev/null; then
    METRICS_ARGS+=(--synthseg-max-volumes "${SYNTHSEG_MAX_VOLUMES}")
    echo "SynthSeg limited to ${SYNTHSEG_MAX_VOLUMES} volumes per NFE level."
fi

python -u "${METRICS_ARGS[@]}"

echo "Metrics computation complete${VARIANT:+ (${VARIANT})}."

# ========================================================================
# NIFTI CONSOLIDATION CHECK
# ========================================================================
SYNTHSEG_DIR="${VARIANT_METRICS_DIR}/synthseg"
echo ""
echo "=========================================="
echo "NIFTI -> HDF5 CONSOLIDATION${VARIANT:+ (${VARIANT})}"
echo "=========================================="
if [ -d "${SYNTHSEG_DIR}" ]; then
    du -sh "${SYNTHSEG_DIR}/"
    echo "HDF5 archives:"
    ls -lh "${SYNTHSEG_DIR}/"*.h5 2>/dev/null || echo "  (no .h5 files yet)"
    echo "Remaining NIfTI dirs:"
    find "${SYNTHSEG_DIR}/" -maxdepth 1 -type d -name "*nifti*" -o -name "*labels*" 2>/dev/null | head -10 || echo "  (none -- all consolidated)"
fi

done  # END VARIANT LOOP

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

# Check feature files (top-level + variant subdirs)
for f in "${FEAT_DIR}"/*.h5; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $(basename $f) (${SIZE} bytes)"
    fi
done
for VARIANT in "${VARIANTS[@]}"; do
    if [ -n "${VARIANT}" ]; then
        for f in "${FEAT_DIR}/${VARIANT}"/*.h5; do
            if [ -f "$f" ]; then
                SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
                echo "[OK]   ${VARIANT}/$(basename $f) (${SIZE} bytes)"
            fi
        done
    fi
done

# Check metrics in all variant dirs
for VARIANT in "${VARIANTS[@]}"; do
    if [ -n "${VARIANT}" ]; then
        CHECK_DIR="${METRICS_DIR}/${VARIANT}"
    else
        CHECK_DIR="${METRICS_DIR}"
    fi
    for f in "${CHECK_DIR}"/*.json; do
        if [ -f "$f" ]; then
            echo "[OK]   ${VARIANT:+${VARIANT}/}$(basename $f)"
        fi
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 5 EVALUATION COMPLETED"
echo "=========================================="
echo "Run dir:    ${RUN_DIR}"
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
