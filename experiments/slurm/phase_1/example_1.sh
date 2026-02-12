#!/usr/bin/env bash
#SBATCH -J toy_finalize_vmf
#SBATCH --time=0-01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=toy_finalize_%j.out
#SBATCH --error=toy_finalize_%j.err

# =============================================================================
# MODULE 0C: FINALIZE (Phase 2)
#
# Runs after Phase 1 sweep array completes (--dependency=afterok).
# Merges partial sweep results, runs Exps 2-4, generates figures + HTML.
#
# Expected env vars (exported by toy_validation.sh launcher):
#   REPO_SRC, RESULTS_DST, CONDA_ENV_NAME
#   MAX_EPOCHS, NUM_TRAIN, NUM_VAL, BATCH_SIZE, HIDDEN_DIM, NUM_LAYERS
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Finalize job started at: $(date)"
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

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ========================================================================
# VERIFY PHASE 1 OUTPUTS
# ========================================================================
echo ""
echo "=========================================="
echo "VERIFYING PHASE 1 OUTPUTS"
echo "=========================================="

SWEEP_SR="${RESULTS_DST}/sweep_swiss_roll"
SWEEP_TO="${RESULTS_DST}/sweep_toroid"
PHASE1_OK=1

for d in "${SWEEP_SR}" "${SWEEP_TO}"; do
    REPORT="${d}/toy_validation_report.json"
    if [ -f "${REPORT}" ]; then
        SIZE=$(stat -c%s "${REPORT}" 2>/dev/null || echo "?")
        echo "[OK]   ${REPORT} (${SIZE} bytes)"
    else
        echo "[MISS] ${REPORT}"
        PHASE1_OK=0
    fi
done

if [ "$PHASE1_OK" -eq 0 ]; then
    echo "ERROR: Phase 1 partial results missing. Cannot finalize."
    exit 1
fi

# ========================================================================
# RUN MERGE + FINALIZE
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING MERGE + FINALIZE"
echo "=========================================="

cd "${REPO_SRC}"

python experiments/scripts/merge_toy_results.py \
    --sweep-dirs "${SWEEP_SR}" "${SWEEP_TO}" \
    --output-dir "${RESULTS_DST}" \
    --device cuda \
    --max-epochs "${MAX_EPOCHS}" \
    --num-train "${NUM_TRAIN}" \
    --num-val "${NUM_VAL}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}"

MERGE_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify expected outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

EXPECTED_FILES=(
    "${RESULTS_DST}/toy_validation_report.json"
    "${RESULTS_DST}/toy_validation_report.html"
    "${RESULTS_DST}/figures/fig1a_swiss_roll.png"
    "${RESULTS_DST}/figures/fig1b_toroid.png"
    "${RESULTS_DST}/figures/fig2_convergence_curves.png"
    "${RESULTS_DST}/figures/fig3_lp_sweep.png"
    "${RESULTS_DST}/figures/fig4_nfe_comparison.png"
    "${RESULTS_DST}/figures/fig5_multichannel.png"
    "${RESULTS_DST}/figures/fig6_gate_summary.png"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $f (${SIZE} bytes)"
    else
        echo "[MISS] $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: ${MISSING} expected files missing."
fi

# Extract gate decision
if [ -f "${RESULTS_DST}/toy_validation_report.json" ]; then
    echo ""
    echo "=========================================="
    echo "SOFT GATE DECISION"
    echo "=========================================="
    python -c "
import json
with open('${RESULTS_DST}/toy_validation_report.json') as f:
    report = json.load(f)
gate = report.get('gate_decision', {})
print(f'Verdict:   {gate.get(\"verdict\", \"N/A\")}')
print(f'p-value:   {gate.get(\"p_value\", \"N/A\")}')
print(f'Cohen d:   {gate.get(\"cohens_d\", \"N/A\")}')
print(f'Gap:       {gate.get(\"pct_gap\", \"N/A\")}%')
print(f'Rationale: {gate.get(\"rationale\", \"N/A\")}')
"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "FINALIZE JOB COMPLETED"
echo "=========================================="
echo "Finished:  $(date)"
echo "Duration:  $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Results:   ${RESULTS_DST}"
echo "Exit code: ${MERGE_EXIT}"

if [ "$MERGE_EXIT" -eq 0 ]; then
    echo "Module 0C toy validation completed successfully."
else
    echo "Module 0C finalize FAILED with exit code ${MERGE_EXIT}."
    exit "${MERGE_EXIT}"
fi
