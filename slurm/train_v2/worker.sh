#!/usr/bin/env bash
#SBATCH -J neuromf_v2_train
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:3
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# =============================================================================
# NeuroMF v2 TRAINING WORKER
#
# Trains the Latent MeanFlow model with v2 changes:
#   - α-Flow curriculum (FM pretraining → MF fine-tuning)
#   - Boundary condition (no v-head, uses u(z_t, t, t))
#   - Constant LR with warmup
#   - norm_p=0.5 adaptive weighting
#   - MultiEMA (3 decay rates)
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, RESUME_CKPT,
#   N_GPUS, TRAIN_CONFIG
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "NeuroMF v2 training started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

# ========================================================================
# NCCL CONFIGURATION
# ========================================================================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT=0

N_GPUS="${N_GPUS:-3}"
echo "GPUs requested: ${N_GPUS}"

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
python -c "import pytorch_lightning; print('Lightning', pytorch_lightning.__version__)"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "NCCL_DEBUG=${NCCL_DEBUG}"

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${CONFIGS_DIR}/train_v2_stage1.yaml}"

# Verify configs exist
for f in "${CONFIGS_DIR}/base.yaml" ${TRAIN_CONFIG}; do
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f"
        echo "ERROR: Required config file missing. Aborting."
        exit 1
    fi
done

# Verify latent dir has HDF5 shard files and stats
LATENT_DIR=$(python -c "
from omegaconf import OmegaConf; import sys
layers = [OmegaConf.load('${CONFIGS_DIR}/base.yaml')]
# Auto-load train_meanflow.yaml as base layer
main_cfg = '${REPO_SRC}/configs/train_meanflow.yaml'
if __import__('pathlib').Path(main_cfg).exists():
    layers.append(OmegaConf.load(main_cfg))
for cp in sys.argv[1:]:
    layers.append(OmegaConf.load(cp))
cfg = OmegaConf.merge(*layers); OmegaConf.resolve(cfg)
print(cfg.paths.latents_dir)
" ${TRAIN_CONFIG} 2>/dev/null) || LATENT_DIR="${RESULTS_DST}/latents"
echo "Latent dir: ${LATENT_DIR}"

if [ -d "${LATENT_DIR}" ]; then
    H5_COUNT=$(find "${LATENT_DIR}" -name "*.h5" | wc -l)
else
    H5_COUNT=0
fi
echo "Latent HDF5 shards: ${H5_COUNT}"
if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No .h5 shard files found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi

if [ ! -f "${LATENT_DIR}/latent_stats.json" ]; then
    echo "ERROR: latent_stats.json not found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi
echo "[OK]   ${LATENT_DIR}/latent_stats.json"

# Verify GPU count
VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Visible GPUs: ${VISIBLE_GPUS} (requested: ${N_GPUS})"
if [ "$VISIBLE_GPUS" -lt "$N_GPUS" ]; then
    echo "WARNING: Only ${VISIBLE_GPUS} GPUs visible but ${N_GPUS} requested."
fi

# Quick import check (includes v2 imports)
python -c "
from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.data.latent_dataset import LatentDataset, latent_collate_fn
from neuromf.wrappers.maisi_unet import MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline
from neuromf.utils.alpha_scheduler import AlphaScheduler
from neuromf.utils.time_sampler import sample_alpha_flow
print('All imports OK (including v2 modules)')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# Print resolved v2-specific settings
python -c "
from omegaconf import OmegaConf; import sys
layers = [OmegaConf.load('${CONFIGS_DIR}/base.yaml')]
main_cfg = '${REPO_SRC}/configs/train_meanflow.yaml'
if __import__('pathlib').Path(main_cfg).exists():
    layers.append(OmegaConf.load(main_cfg))
for cp in sys.argv[1:]:
    layers.append(OmegaConf.load(cp))
cfg = OmegaConf.merge(*layers); OmegaConf.resolve(cfg)

# v2 settings
print('--- v2 Settings ---')
print(f'  use_v_head:     {cfg.unet.use_v_head}')
print(f'  prediction:     {cfg.unet.prediction_type}')
print(f'  lr_schedule:    {cfg.training.lr_schedule}')
print(f'  betas:          {list(cfg.training.betas)}')
print(f'  norm_p:         {cfg.meanflow.norm_p}')
print(f'  norm_eps:       {cfg.meanflow.norm_eps}')
af = cfg.get('alpha_flow', {})
if af:
    print(f'  alpha_flow.eta: {af.eta}')
    print(f'  alpha_flow.start_step: {af.start_step}')
    print(f'  alpha_flow.end_step:   {af.end_step}')
    print(f'  alpha_flow.mode:       {af.mode}')
ts = cfg.time_sampling
print(f'  data_proportion: {ts.data_proportion}')
init = cfg.unet.get('initialization', {})
print(f'  init method:    {init.get(\"method\", \"kaiming\")}')
ema = cfg.ema
if 'decays' in ema:
    print(f'  ema decays:     {list(ema.decays)}')
else:
    print(f'  ema decay:      {ema.decay}')
print(f'  max_epochs:     {cfg.training.max_epochs}')
" ${TRAIN_CONFIG} 2>/dev/null || echo "[WARN] Could not resolve v2 settings"

# ========================================================================
# RUN TRAINING
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING NeuroMF v2 TRAINING"
echo "=========================================="
echo "Config dir:    ${CONFIGS_DIR}"
echo "Train config:  ${TRAIN_CONFIG}"
echo "GPUs:          ${N_GPUS}"
echo ""

TRAIN_CMD="python experiments/cli/train.py \
    --config ${TRAIN_CONFIG} \
    --configs-dir ${CONFIGS_DIR}"

# Add resume flag if checkpoint specified
if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
    echo "Resuming from: ${RESUME_CKPT}"
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME_CKPT}"
elif [ -n "${RESUME_CKPT:-}" ] && [ ! -f "${RESUME_CKPT}" ]; then
    echo "WARNING: Resume checkpoint not found: ${RESUME_CKPT}"
    echo "         Starting from scratch."
fi

eval "${TRAIN_CMD}" && TRAIN_EXIT=0 || TRAIN_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

RUN_DIR=$(ls -1dt "${RESULTS_DST}/runs/run_"* 2>/dev/null | head -1)
if [ -n "${RUN_DIR}" ]; then
    echo "Run directory: ${RUN_DIR}"
    CKPT_DIR="${RUN_DIR}/checkpoints"
    CKPT_COUNT=$(find "${CKPT_DIR}" -name "*.ckpt" 2>/dev/null | wc -l)
    echo "Checkpoint files: ${CKPT_COUNT}"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        SIZE=$(stat -c%s "${CKPT_DIR}/last.ckpt" 2>/dev/null || echo "?")
        echo "[OK]   last.ckpt (${SIZE} bytes)"
    else
        echo "[MISS] last.ckpt"
    fi

    # List best FID checkpoints for Stage 2 resume
    echo ""
    echo "Best FID checkpoints (use for Stage 2 --resume):"
    ls -la "${CKPT_DIR}"/best_fid_*.ckpt 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Best raw loss checkpoints:"
    ls -la "${CKPT_DIR}"/best_raw_*.ckpt 2>/dev/null || echo "  (none found)"

    DIAG_SUMMARY="${RUN_DIR}/diagnostics/aggregate_results/training_summary.json"
    if [ -f "${DIAG_SUMMARY}" ]; then
        echo "[OK]   training_summary.json"
        python -c "
import json
with open('${DIAG_SUMMARY}') as f:
    data = json.load(f)
if data and 'gpu_memory' in data[-1]:
    mem = data[-1]['gpu_memory']
    print(f\"  GPU peak allocated: {mem['peak_allocated_gb']:.2f} GB\")
    print(f\"  GPU peak reserved:  {mem['peak_reserved_gb']:.2f} GB\")
" 2>/dev/null || true
    fi
else
    echo "[MISS] No run directory found under ${RESULTS_DST}/runs/"
    CKPT_COUNT=0
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "NeuroMF v2 TRAINING COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "GPUs:       ${N_GPUS}"
echo "Run dir:    ${RUN_DIR:-not found}"
echo "Checkpts:   ${CKPT_COUNT} files"
echo "Exit code:  ${TRAIN_EXIT}"

if [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training FAILED with exit code ${TRAIN_EXIT}."
    exit "${TRAIN_EXIT}"
fi
