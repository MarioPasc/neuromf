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
# PHASE 5: EVALUATION WORKER
#
# Stage 1: Extract R3D-18 features (real + generated) — MOTFM protocol
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

# Extract R3D-18 features (MOTFM protocol) for real test set and generated volumes
# Note: R3D-18 replaces Med3D ResNet-50 (which had feature collapse).
# The 'med3d' backend name is kept for backward compatibility but loads R3D-18.
python -u -c "
import time
import torch
import h5py
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from neuromf.metrics.feature_extractor import FeatureExtractor

# ── Config ──
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

# ── Diagnostics: environment ──
print()
print('--- Feature Extraction Diagnostics ---')
print(f'  Device:          {device}')
if device.type == 'cuda':
    print(f'  GPU:             {torch.cuda.get_device_name(device)}')
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f'  GPU memory:      {mem_gb:.1f} GB')

# ── Load feature extractor ──
weights_path = config.features.med3d.get('weights_path', '')
print(f'  Config weights:  {weights_path or \"(empty — torchvision auto-download)\"}')

extractor = FeatureExtractor('med3d', weights_path, device)
model = extractor.model

# ── Diagnostics: model ──
arch = type(model).__name__
n_params = sum(p.numel() for p in model.parameters())
feature_dim = None
# Probe feature dim with a dummy forward pass
with torch.no_grad():
    dummy = torch.randn(1, 3, 16, 16, 16, device=device)
    dummy_out = model(dummy)
    feature_dim = dummy_out.shape[-1]

print(f'  Architecture:    {arch}')
print(f'  Parameters:      {n_params / 1e6:.2f}M')
print(f'  Feature dim:     {feature_dim}')
print(f'  Training mode:   {model.training}')

# Check where torchvision cached the weights
import torchvision
cache_dir = Path(torch.hub.get_dir()) / 'checkpoints'
r3d_files = sorted(cache_dir.glob('r3d_18*'))
if r3d_files:
    for f in r3d_files:
        print(f'  Weights file:    {f} ({f.stat().st_size / 1e6:.1f} MB)')
else:
    print(f'  Weights file:    (not found in {cache_dir})')
print('--- End Diagnostics ---')
print()

# ── Helper: log volume archive info ──
def log_h5_info(path, label):
    if not path.exists():
        print(f'  [{label}] {path.name} — NOT FOUND')
        return 0
    with h5py.File(str(path), 'r') as f:
        shape = f['volumes'].shape
        dtype = f['volumes'].dtype
        size_mb = path.stat().st_size / 1e6
    print(f'  [{label}] {path.name}: {shape}, dtype={dtype}, {size_mb:.1f} MB')
    return shape[0]

# ── Extract features ──
real_vol = Path('${GEN_DIR}/real_test.h5')
real_feat = Path('${FEAT_DIR}/real_med3d.h5')

print('Input volumes:')
log_h5_info(real_vol, 'real')
for nfe in [1, 10, 50]:
    log_h5_info(Path(f'${GEN_DIR}/volumes/nfe_{nfe:03d}.h5'), f'nfe={nfe}')
print()

# Real features
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

# Generated features for each NFE
for nfe in [1, 10, 50]:
    gen_vol = Path(f'${GEN_DIR}/volumes/nfe_{nfe:03d}.h5')
    gen_feat = Path(f'${FEAT_DIR}/gen_med3d_nfe{nfe:03d}.h5')
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
        print(f'NFE={nfe} volume missing — skipping.')
"

echo "NeuroMF feature extraction complete."

# ── MOTFM feature extraction (same extractor, same protocol) ──
if [ -d "${GEN_DIR}/volumes/motfm" ]; then
    echo ""
    echo "--- MOTFM Feature Extraction ---"

    MOTFM_FEAT_DIR="${FEAT_DIR}/motfm"
    mkdir -p "${MOTFM_FEAT_DIR}"

    python -u -c "
import time
import torch
import h5py
from pathlib import Path
from omegaconf import OmegaConf

from neuromf.metrics.feature_extractor import FeatureExtractor

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
weights_path = config.features.med3d.get('weights_path', '')
extractor = FeatureExtractor('med3d', weights_path, device)

for nfe in [1, 10, 50]:
    gen_vol = Path(f'${GEN_DIR}/volumes/motfm/nfe_{nfe:03d}.h5')
    gen_feat = Path(f'${MOTFM_FEAT_DIR}/gen_med3d_nfe{nfe:03d}.h5')
    if gen_vol.exists() and not gen_feat.exists():
        t0 = time.time()
        feats = extractor.extract_and_cache(gen_vol, gen_feat)
        dt = time.time() - t0
        print(f'MOTFM NFE={nfe} features extracted: {feats.shape} in {dt:.1f}s')
    elif gen_feat.exists():
        with h5py.File(str(gen_feat), 'r') as f:
            cached_shape = f['features'].shape
        print(f'MOTFM NFE={nfe} features already cached: {cached_shape}')
    else:
        print(f'MOTFM NFE={nfe} volume missing — skipping.')
"
    echo "MOTFM feature extraction complete."
fi

# ========================================================================
# STAGE 2: METRICS COMPUTATION
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: METRICS COMPUTATION"
echo "=========================================="

python -u experiments/cli/compute_metrics.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --volumes-dir "${GEN_DIR}/volumes" \
    --real-features-dir "${FEAT_DIR}" \
    --real-volumes-h5 "${GEN_DIR}/real_test.h5" \
    --nfe 1 10 50 \
    --output-dir "${METRICS_DIR}"

echo "NeuroMF metrics computation complete."

# ── MOTFM metrics computation (same pipeline, different volumes dir) ──
if [ -d "${GEN_DIR}/volumes/motfm" ]; then
    echo ""
    echo "--- MOTFM Metrics Computation ---"

    MOTFM_METRICS_DIR="${METRICS_DIR}/motfm"
    MOTFM_FEAT_DIR="${FEAT_DIR}/motfm"
    mkdir -p "${MOTFM_METRICS_DIR}"

    # Symlink shared real features into MOTFM features dir so
    # compute_metrics.py finds real_med3d.h5 alongside gen features
    if [ -f "${FEAT_DIR}/real_med3d.h5" ] && [ ! -e "${MOTFM_FEAT_DIR}/real_med3d.h5" ]; then
        ln -s "${FEAT_DIR}/real_med3d.h5" "${MOTFM_FEAT_DIR}/real_med3d.h5"
    fi

    python -u experiments/cli/compute_metrics.py \
        --config "${CONFIGS_DIR}/generate.yaml" \
        --configs-dir "${CONFIGS_DIR}" \
        --volumes-dir "${GEN_DIR}/volumes/motfm" \
        --real-features-dir "${MOTFM_FEAT_DIR}" \
        --real-volumes-h5 "${GEN_DIR}/real_test.h5" \
        --nfe 1 10 50 \
        --output-dir "${MOTFM_METRICS_DIR}"

    echo "MOTFM metrics computation complete."
fi

# ========================================================================
# NIFTI CONSOLIDATION CHECK
# ========================================================================
echo ""
echo "=========================================="
echo "NIFTI → HDF5 CONSOLIDATION"
echo "=========================================="
echo "Consolidated NIfTI → HDF5, cleaned up temporary files"
if [ -d "${METRICS_DIR}/synthseg" ]; then
    du -sh "${METRICS_DIR}/synthseg/"
    echo "HDF5 archives:"
    ls -lh "${METRICS_DIR}/synthseg/"*.h5 2>/dev/null || echo "  (no .h5 files yet)"
    echo "Remaining NIfTI dirs:"
    find "${METRICS_DIR}/synthseg/" -maxdepth 1 -type d -name "*nifti*" -o -name "*labels*" 2>/dev/null | head -10 || echo "  (none — all consolidated)"
fi

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

# MOTFM outputs
if [ -d "${FEAT_DIR}/motfm" ]; then
    for f in "${FEAT_DIR}/motfm"/*.h5; do
        if [ -f "$f" ]; then
            SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
            echo "[OK]   motfm/$(basename $f) (${SIZE} bytes)"
        fi
    done
fi
if [ -d "${METRICS_DIR}/motfm" ]; then
    for f in "${METRICS_DIR}/motfm"/*.json; do
        if [ -f "$f" ]; then
            echo "[OK]   motfm/$(basename $f)"
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 5 EVALUATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
