#!/usr/bin/env bash
#SBATCH -J neuromf_compare
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=compare_%j.out
#SBATCH --error=compare_%j.err

# =============================================================================
# COMPARISON ANALYSIS WORKER
#
# Runs the cross-model comparison pipeline: loads pre-computed features and
# metrics from all 3 models, produces figures, tables, and statistical tests.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONFIGS_DIR, CONDA_ENV_NAME,
#   NEUROIMF_DIR, MOTFM_DIR, DDPM_DIR, COMPARE_OUTPUT_DIR,
#   N_BOOTSTRAP, SKIP_BOOTSTRAP, SKIP_QUALITATIVE, NFE_LEVELS
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Comparison analysis started at: $(date)"
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

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="
echo "NeuroiMF dir: ${NEUROIMF_DIR}"
echo "MOTFM dir:    ${MOTFM_DIR}"
echo "DDPM dir:     ${DDPM_DIR}"
echo "Output dir:   ${COMPARE_OUTPUT_DIR}"

MISSING=0
for model_label in NeuroiMF MOTFM DDPM; do
    case "${model_label}" in
        NeuroiMF) model_dir="${NEUROIMF_DIR}" ;;
        MOTFM)    model_dir="${MOTFM_DIR}" ;;
        DDPM)     model_dir="${DDPM_DIR}" ;;
    esac

    # Check features directory
    if [ -d "${model_dir}/features" ]; then
        N_FEAT=$(find "${model_dir}/features" -name "*.h5" | wc -l)
        echo "[OK]   ${model_label}: features/ (${N_FEAT} .h5 files)"
    else
        echo "[FAIL] ${model_label}: features/ — not found"
        MISSING=1
    fi

    # Check metrics directory
    if [ -d "${model_dir}/metrics" ]; then
        N_JSON=$(find "${model_dir}/metrics" -maxdepth 1 -name "*.json" | wc -l)
        echo "[OK]   ${model_label}: metrics/ (${N_JSON} .json files)"
    else
        echo "[FAIL] ${model_label}: metrics/ — not found"
        MISSING=1
    fi

    # Check per-NFE metrics
    for nfe in ${NFE_LEVELS}; do
        nfe_padded=$(printf '%03d' $nfe)
        json_file="${model_dir}/metrics/metrics_nfe${nfe_padded}.json"
        if [ -f "${json_file}" ]; then
            echo "[OK]   ${model_label}: metrics_nfe${nfe_padded}.json"
        else
            echo "[WARN] ${model_label}: metrics_nfe${nfe_padded}.json — not found"
        fi
    done
done

if [ "${MISSING}" -eq 1 ]; then
    echo ""
    echo "ERROR: Required directories are missing. Run evaluation first."
    exit 1
fi

# ========================================================================
# RUN COMPARISON PIPELINE
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING COMPARISON PIPELINE"
echo "=========================================="
echo "  NFE levels:        ${NFE_LEVELS}"
echo "  Bootstrap samples: ${N_BOOTSTRAP}"
echo "  Skip bootstrap:    ${SKIP_BOOTSTRAP}"
echo "  Skip qualitative:  ${SKIP_QUALITATIVE}"
echo ""

# Build CLI arguments
CLI_ARGS=(
    experiments/cli/run_comparison.py
    --neuroimf-dir "${NEUROIMF_DIR}"
    --motfm-dir "${MOTFM_DIR}"
    --ddpm-dir "${DDPM_DIR}"
    --output-dir "${COMPARE_OUTPUT_DIR}"
    --nfe ${NFE_LEVELS}
    --n-bootstrap "${N_BOOTSTRAP}"
)

if [ "${SKIP_BOOTSTRAP}" -eq 1 ]; then
    CLI_ARGS+=(--skip-bootstrap)
fi

if [ "${SKIP_QUALITATIVE}" -eq 1 ]; then
    CLI_ARGS+=(--skip-qualitative)
fi

python -u "${CLI_ARGS[@]}"

# ========================================================================
# POST-FLIGHT: OUTPUT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

echo "Figures:"
N_PDF=$(find "${COMPARE_OUTPUT_DIR}/figures" -name "*.pdf" 2>/dev/null | wc -l)
N_PNG=$(find "${COMPARE_OUTPUT_DIR}/figures" -name "*.png" 2>/dev/null | wc -l)
echo "  ${N_PDF} PDF files, ${N_PNG} PNG files"
for f in "${COMPARE_OUTPUT_DIR}/figures"/*.pdf; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "  [OK] $(basename "$f") (${SIZE} bytes)"
    fi
done

echo ""
echo "Tables:"
N_TEX=$(find "${COMPARE_OUTPUT_DIR}/tables" -name "*.tex" 2>/dev/null | wc -l)
N_CSV=$(find "${COMPARE_OUTPUT_DIR}/tables" -name "*.csv" 2>/dev/null | wc -l)
echo "  ${N_TEX} .tex files, ${N_CSV} .csv files"
for f in "${COMPARE_OUTPUT_DIR}/tables"/*.tex; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        echo "  [OK] $(basename "$f") (${LINES} lines)"
    fi
done

echo ""
echo "Data files:"
for f in "${COMPARE_OUTPUT_DIR}/data"/*; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "  [OK] $(basename "$f") (${SIZE} bytes)"
    fi
done

# ========================================================================
# DONE
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "COMPARISON ANALYSIS COMPLETED"
echo "=========================================="
echo "Output:     ${COMPARE_OUTPUT_DIR}/"
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
