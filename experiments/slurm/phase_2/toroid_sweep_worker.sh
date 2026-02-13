#!/usr/bin/env bash
#SBATCH -J neuromf_p2_toroid
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

# =============================================================================
# PHASE 2: TOROID ABLATION SWEEP WORKER
#
# Runs the full formal experiment end-to-end on a single node:
#   - 18 training runs (Ablations A/B/C/E)
#   - NFE inference sweep (Ablation D)
#   - Publication figures (PDF+PNG)
#   - CSV tables
#   - Self-contained HTML report
#
# CPU-only experiment (toy MLP). GPU requested only for node allocation.
# No internet access required — all code is self-contained.
#
# Expected env vars (exported by toroid_sweep.sh launcher):
#   REPO_SRC, RESULTS_DST, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 2 toroid sweep started at: $(date)"
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

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify experiment configs exist
CONFIGS_DIR="${REPO_SRC}/experiments/toy_toroid/configs"
for f in base.yaml ablation_a.yaml ablation_b.yaml ablation_c.yaml ablation_d.yaml ablation_e.yaml; do
    if [ -f "${CONFIGS_DIR}/${f}" ]; then
        echo "[OK]   ${CONFIGS_DIR}/${f}"
    else
        echo "[MISS] ${CONFIGS_DIR}/${f}"
        echo "ERROR: Required config file missing. Aborting."
        exit 1
    fi
done

# Full import check — every module that will be used during the sweep
python -c "
import sys, os
# Test all library imports
from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset
from neuromf.losses.meanflow_jvp import meanflow_loss
from neuromf.losses.lp_loss import lp_loss
from neuromf.models.toy_mlp import ToyMLP
from neuromf.sampling.one_step import sample_one_step
from neuromf.sampling.multi_step import sample_euler
from neuromf.utils.ema import EMAModel
from neuromf.utils.time_sampler import sample_t_and_r
from neuromf.metrics.mmd import compute_mmd
from neuromf.metrics.coverage_density import compute_coverage, compute_density
print('[OK] All neuromf imports')

# Test experiment imports
from experiments.toy_toroid.train import train_run
from experiments.toy_toroid.evaluate import evaluate_nfe_sweep
from experiments.toy_toroid.figures import generate_all_figures
from experiments.toy_toroid.report import generate_report
from experiments.toy_toroid.sweep import _load_config
print('[OK] All experiment imports')

# Test config loading
cfg = _load_config('a')
print(f'[OK] Config: epochs={cfg.training.epochs}, batch={cfg.training.batch_size}, n_samples={cfg.data.n_samples}')

# Test matplotlib rendering (catches font cache issues)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
fig.savefig('/tmp/_test_mpl_picasso.png', dpi=50)
plt.close()
os.remove('/tmp/_test_mpl_picasso.png')
print('[OK] Matplotlib rendering')

# Test a tiny training step (1 batch, 1 epoch) to catch runtime errors
import torch
torch.manual_seed(42)
ds = ToroidDataset(ToroidConfig(n_samples=64, mode='r4', ambient_dim=4), seed=42)
model = ToyMLP(data_dim=4, hidden_dim=32, n_layers=2, prediction_type='u')
ema = EMAModel(model, decay=0.999)
loader = torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_0 = next(iter(loader))
eps = torch.randn_like(x_0)
t, r = sample_t_and_r(32, data_proportion=0.75)
result = meanflow_loss(model, x_0, eps, t, r, p=2.0, prediction_type='u')
result['loss'].backward()
optimizer.step()
ema.update(model)
# Test EMA eval
ema.apply_shadow(model)
noise = torch.randn(16, 4)
s1 = sample_one_step(model, noise, prediction_type='u')
s5 = sample_euler(model, noise, n_steps=5, prediction_type='u')
ema.restore(model)
# Test metrics
mmd_val = compute_mmd(ds.data[:100], s1[:16].cpu())
cov_val = compute_coverage(ds.data[:100], s1[:16].cpu(), k=3)
den_val = compute_density(ds.data[:100], s1[:16].cpu(), k=3)
print(f'[OK] Dry-run: loss={result[\"loss\"].item():.4f}, mmd={mmd_val:.6f}, cov={cov_val:.3f}, den={den_val:.3f}')

from scipy import stats
import numpy as np
theta = torch.atan2(s1[:, 1], s1[:, 0]).numpy()
ks = stats.kstest((theta + np.pi) / (2 * np.pi), 'uniform')
print(f'[OK] scipy.stats KS test: p={ks.pvalue:.4f}')

print()
print('=== ALL PRE-FLIGHT CHECKS PASSED ===')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# ========================================================================
# RUN FULL SWEEP
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING PHASE 2 TOROID ABLATION SWEEP"
echo "=========================================="
echo "Results:    ${RESULTS_DST}"
echo "Ablations:  A (baseline), B (dim x pred), C (Lp), D (NFE), E (data_prop)"
echo "Total runs: 18 training + 6 inference"
echo ""

python experiments/toy_toroid/sweep.py \
    --ablation all \
    --results-dir "${RESULTS_DST}"

SWEEP_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

EXPECTED_FILES=(
    "${RESULTS_DST}/summary_metrics.json"
    "${RESULTS_DST}/report.html"
    "${RESULTS_DST}/tables/table_s1_full_results.csv"
    "${RESULTS_DST}/tables/table_1_dim_scaling.csv"
    "${RESULTS_DST}/figures/fig2a_loss_convergence.png"
    "${RESULTS_DST}/figures/fig2b_dim_scaling.png"
    "${RESULTS_DST}/figures/fig2c_nfe_tradeoff.png"
    "${RESULTS_DST}/figures/fig2d_angular_distributions.png"
    "${RESULTS_DST}/figures/fig2e_lp_impact.png"
    "${RESULTS_DST}/figures/fig2f_data_proportion.png"
    "${RESULTS_DST}/ablation_a/baseline_D4_u-pred_p2.0_dp0.75/metrics.json"
    "${RESULTS_DST}/ablation_b/D4_u-pred/metrics.json"
    "${RESULTS_DST}/ablation_b/D256_x-pred/metrics.json"
    "${RESULTS_DST}/ablation_c/p2.0/metrics.json"
    "${RESULTS_DST}/ablation_d/nfe_sweep.json"
    "${RESULTS_DST}/ablation_e/dp0.75/metrics.json"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $(basename "$f") (${SIZE} bytes)"
    else
        echo "[MISS] $f"
        MISSING=$((MISSING + 1))
    fi
done

# Count run directories
ABL_A=$(find "${RESULTS_DST}/ablation_a" -name "metrics.json" 2>/dev/null | wc -l)
ABL_B=$(find "${RESULTS_DST}/ablation_b" -name "metrics.json" 2>/dev/null | wc -l)
ABL_C=$(find "${RESULTS_DST}/ablation_c" -name "metrics.json" 2>/dev/null | wc -l)
ABL_E=$(find "${RESULTS_DST}/ablation_e" -name "metrics.json" 2>/dev/null | wc -l)
echo ""
echo "Completed runs: A=${ABL_A}/1, B=${ABL_B}/8, C=${ABL_C}/4, E=${ABL_E}/5"

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: ${MISSING} expected files missing."
fi

# Print key metrics summary
if [ -f "${RESULTS_DST}/ablation_a/baseline_D4_u-pred_p2.0_dp0.75/metrics.json" ]; then
    echo ""
    echo "KEY METRICS (Ablation A baseline):"
    python -c "
import json
with open('${RESULTS_DST}/ablation_a/baseline_D4_u-pred_p2.0_dp0.75/metrics.json') as f:
    m = json.load(f)
one = m['one_step']
multi = m['multi_step']
print(f'  1-NFE torus distance: {one[\"mean_torus_distance\"]:.4f} (target < 0.1)')
print(f'  1-NFE MMD:            {one[\"mmd\"]:.6f}')
print(f'  1-NFE coverage:       {one[\"coverage\"]:.3f}')
print(f'  KS theta1 p-value:    {one[\"theta1_ks_pvalue\"]:.4f} (target > 0.01)')
print(f'  KS theta2 p-value:    {one[\"theta2_ks_pvalue\"]:.4f} (target > 0.01)')
print(f'  10-step torus dist:   {multi[\"mean_torus_distance\"]:.4f}')
print(f'  Final loss:           {m[\"final_loss\"]:.6f}')
"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 2 TOROID SWEEP COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Report:     ${RESULTS_DST}/report.html"
echo "Exit code:  ${SWEEP_EXIT}"

if [ "$SWEEP_EXIT" -eq 0 ] && [ "$MISSING" -eq 0 ]; then
    echo "Phase 2 toroid sweep completed SUCCESSFULLY."
else
    echo "Phase 2 toroid sweep finished with issues (exit=${SWEEP_EXIT}, missing=${MISSING})."
    exit "${SWEEP_EXIT}"
fi
