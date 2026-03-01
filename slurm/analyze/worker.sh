#!/usr/bin/env bash
#SBATCH -J neuromf_p5_analysis
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=analyse_%j.out
#SBATCH --error=analyse_%j.err

# =============================================================================
# POST-HOC ANALYSIS WORKER
#
# Runs the full analysis pipeline: bootstrap CIs, publication figures,
# LaTeX tables, statistical tests, and manifest generation.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, ANALYSIS_RESULTS_DIR, CONDA_ENV_NAME,
#   N_BOOTSTRAP, SKIP_QUALITATIVE, SKIP_BOOTSTRAP, NFE_LEVELS
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 5 analysis started at: $(date)"
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
echo "Results dir: ${ANALYSIS_RESULTS_DIR}"

MISSING=0
for nfe in ${NFE_LEVELS}; do
    nfe_padded=$(printf '%03d' $nfe)
    json_file="${ANALYSIS_RESULTS_DIR}/metrics/metrics_nfe${nfe_padded}.json"
    feat_file="${ANALYSIS_RESULTS_DIR}/features/gen_med3d_nfe${nfe_padded}.h5"

    if [ -f "${json_file}" ]; then
        echo "[OK]   metrics/metrics_nfe${nfe_padded}.json"
    else
        echo "[FAIL] metrics/metrics_nfe${nfe_padded}.json — not found"
        MISSING=1
    fi

    if [ -f "${feat_file}" ]; then
        SIZE=$(stat -c%s "${feat_file}" 2>/dev/null || echo "?")
        echo "[OK]   features/gen_med3d_nfe${nfe_padded}.h5 (${SIZE} bytes)"
    else
        echo "[WARN] features/gen_med3d_nfe${nfe_padded}.h5 — not found (bootstrap will be skipped for NFE=${nfe})"
    fi
done

if [ -f "${ANALYSIS_RESULTS_DIR}/features/real_med3d.h5" ]; then
    SIZE=$(stat -c%s "${ANALYSIS_RESULTS_DIR}/features/real_med3d.h5" 2>/dev/null || echo "?")
    echo "[OK]   features/real_med3d.h5 (${SIZE} bytes)"
else
    echo "[WARN] features/real_med3d.h5 — not found (bootstrap will be skipped)"
fi

if [ "${MISSING}" -eq 1 ]; then
    echo ""
    echo "ERROR: Required metrics JSONs are missing. Run evaluate first."
    exit 1
fi

# Check optional inputs
echo ""
echo "Optional inputs:"
if [ -d "${ANALYSIS_RESULTS_DIR}/metrics/synthseg" ]; then
    N_CSV=$(find "${ANALYSIS_RESULTS_DIR}/metrics/synthseg" -name "*.csv" | wc -l)
    echo "[OK]   SynthSeg data: ${N_CSV} CSV files"
else
    echo "[SKIP] SynthSeg data not found"
fi

if [ -f "${ANALYSIS_RESULTS_DIR}/generation/real_test.h5" ]; then
    echo "[OK]   Decoded volumes: real_test.h5"
else
    echo "[SKIP] Decoded volumes not found (qualitative figures will be skipped)"
fi

# ========================================================================
# RUN ANALYSIS PIPELINE
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING ANALYSIS PIPELINE"
echo "=========================================="
echo "  Bootstrap samples: ${N_BOOTSTRAP}"
echo "  Skip bootstrap:    ${SKIP_BOOTSTRAP}"
echo "  Skip qualitative:  ${SKIP_QUALITATIVE}"
echo "  NFE levels:        ${NFE_LEVELS}"
echo ""

# Build CLI arguments
CLI_ARGS=(
    experiments/cli/run_analysis.py
    --results-dir "${ANALYSIS_RESULTS_DIR}"
    --nfe ${NFE_LEVELS}
    --n-bootstrap "${N_BOOTSTRAP}"
)

if [ "${SKIP_QUALITATIVE}" -eq 1 ]; then
    CLI_ARGS+=(--skip-qualitative)
fi

if [ "${SKIP_BOOTSTRAP}" -eq 1 ]; then
    CLI_ARGS+=(--skip-bootstrap)
fi

python -u "${CLI_ARGS[@]}"

# ========================================================================
# POST-FLIGHT: OUTPUT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

ANALYSIS_DIR="${ANALYSIS_RESULTS_DIR}/analysis"

echo "Figures:"
N_PDF=$(find "${ANALYSIS_DIR}/figures" -name "*.pdf" 2>/dev/null | wc -l)
N_PNG=$(find "${ANALYSIS_DIR}/figures" -name "*.png" 2>/dev/null | wc -l)
echo "  ${N_PDF} PDF files, ${N_PNG} PNG files"
for f in "${ANALYSIS_DIR}/figures"/*.pdf; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "  [OK] $(basename "$f") (${SIZE} bytes)"
    fi
done

echo ""
echo "Tables:"
for f in "${ANALYSIS_DIR}/tables"/*.tex; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        echo "  [OK] $(basename "$f") (${LINES} lines)"
    fi
done

echo ""
echo "Source data (npz):"
for f in "${ANALYSIS_DIR}/data"/*.npz; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "  [OK] $(basename "$f") (${SIZE} bytes)"
    fi
done

echo ""
echo "Manifest:"
if [ -f "${ANALYSIS_DIR}/analysis_manifest.json" ]; then
    echo "  [OK] analysis_manifest.json"
    python -c "import json; m=json.load(open('${ANALYSIS_DIR}/analysis_manifest.json')); print(f'       Figures: {len(m.get(\"figures\",{}))}'); print(f'       Tables:  {len(m.get(\"tables\",{}))}'); print(f'       Data:    {len(m.get(\"data_files\",{}))}'); print(f'       Skipped: {m.get(\"skipped\",[])}'); "
else
    echo "  [FAIL] analysis_manifest.json — not found"
fi

# ========================================================================
# DONE
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 5 ANALYSIS COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Output:     ${ANALYSIS_DIR}/"
