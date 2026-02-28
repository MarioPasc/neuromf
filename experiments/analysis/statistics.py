"""Statistical tests and bootstrap confidence intervals for analysis.

Provides bootstrap CIs for distributional metrics (FID, MMD, coverage,
density), per-volume CIs from t-distribution, Cohen's d effect sizes,
KS tests with Bonferroni correction, and regional correlation analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import Tensor

from experiments.utils.settings import get_effect_size_interpretation

logger = logging.getLogger(__name__)


def bootstrap_metric(
    real_feats: Tensor,
    gen_feats: Tensor,
    metric_fn: Callable[[Tensor, Tensor], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap confidence interval for a distributional metric.

    Resamples gen_feats with replacement, recomputes metric_fn(real, gen_sample).

    Args:
        real_feats: Real feature vectors (N_real, D).
        gen_feats: Generated feature vectors (N_gen, D).
        metric_fn: Function(real, gen) -> float.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'point_estimate', 'ci_lower', 'ci_upper', 'samples', 'std'.
    """
    rng = np.random.RandomState(seed)
    point_estimate = metric_fn(real_feats, gen_feats)

    n_gen = gen_feats.shape[0]
    samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n_gen, size=n_gen)
        gen_sample = gen_feats[idx]
        samples[i] = metric_fn(real_feats, gen_sample)
        if (i + 1) % 100 == 0:
            logger.debug("Bootstrap %d/%d", i + 1, n_bootstrap)

    ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "samples": samples,
        "std": float(np.std(samples)),
    }


def compute_per_volume_ci(
    mean: float,
    std: float,
    n: int,
) -> tuple[float, float]:
    """95% confidence interval from t-distribution.

    Args:
        mean: Sample mean.
        std: Sample standard deviation.
        n: Sample size.

    Returns:
        Tuple (ci_lower, ci_upper).
    """
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (mean - t_crit * se, mean + t_crit * se)


def compute_cohens_d(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> dict[str, Any]:
    """Cohen's d with pooled standard deviation and CI.

    Args:
        mean1: Mean of group 1 (e.g., NeuroiMF).
        std1: Std of group 1.
        n1: Sample size of group 1.
        mean2: Mean of group 2 (e.g., MOTFM).
        std2: Std of group 2.
        n2: Sample size of group 2.

    Returns:
        Dict with 'd', 'interpretation', 'ci_lower', 'ci_upper'.
    """
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    )
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    # SE of Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d

    return {
        "d": float(d),
        "interpretation": get_effect_size_interpretation(abs(d)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def compute_ks_tests(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    regions: list[str],
    bonferroni_n: int | None = None,
) -> pd.DataFrame:
    """Two-sample KS test per region with Bonferroni correction.

    Args:
        real_df: Real SynthSeg volumes DataFrame.
        gen_df: Generated SynthSeg volumes DataFrame.
        regions: Column names to test.
        bonferroni_n: Number of comparisons for correction.
            If None, uses len(regions).

    Returns:
        DataFrame with columns: region, statistic, p_value, p_corrected, stars.
    """
    if bonferroni_n is None:
        bonferroni_n = len(regions)

    rows: list[dict[str, Any]] = []
    for region in regions:
        if region not in real_df.columns or region not in gen_df.columns:
            logger.warning("Region '%s' not in both DataFrames, skipping", region)
            continue
        real_vals = real_df[region].dropna().values
        gen_vals = gen_df[region].dropna().values
        if len(real_vals) == 0 or len(gen_vals) == 0:
            continue
        ks_stat, p_val = stats.ks_2samp(real_vals, gen_vals)
        p_corrected = min(p_val * bonferroni_n, 1.0)
        from experiments.utils.settings import get_significance_stars
        rows.append({
            "region": region,
            "statistic": ks_stat,
            "p_value": p_val,
            "p_corrected": p_corrected,
            "stars": get_significance_stars(p_corrected),
        })

    return pd.DataFrame(rows)


def compute_regional_stats(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    regions: list[str],
    n_bins: int = 50,
) -> pd.DataFrame:
    """Per-region Pearson r and KL divergence.

    Args:
        real_df: Real SynthSeg volumes DataFrame.
        gen_df: Generated SynthSeg volumes DataFrame.
        regions: Column names to analyse.
        n_bins: Number of bins for KL divergence histograms.

    Returns:
        DataFrame with columns: region, real_mean, real_std, gen_mean,
        gen_std, pearson_r, pearson_p, kl_divergence.
    """
    rows: list[dict[str, Any]] = []
    for region in regions:
        if region not in real_df.columns or region not in gen_df.columns:
            continue
        real_vals = real_df[region].dropna().values
        gen_vals = gen_df[region].dropna().values
        if len(real_vals) < 3 or len(gen_vals) < 3:
            continue

        # Pearson correlation (compare sorted distributions)
        min_n = min(len(real_vals), len(gen_vals))
        r_sorted = np.sort(real_vals)[:min_n]
        g_sorted = np.sort(gen_vals)[:min_n]
        pearson_r, pearson_p = stats.pearsonr(r_sorted, g_sorted)

        # KL divergence via histograms
        all_vals = np.concatenate([real_vals, gen_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
        real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_vals, bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        real_hist = real_hist + eps
        gen_hist = gen_hist + eps
        # Normalize to proper distributions
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()
        kl_div = float(stats.entropy(real_hist, gen_hist))

        rows.append({
            "region": region,
            "real_mean": float(np.mean(real_vals)),
            "real_std": float(np.std(real_vals)),
            "gen_mean": float(np.mean(gen_vals)),
            "gen_std": float(np.std(gen_vals)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "kl_divergence": kl_div,
        })

    return pd.DataFrame(rows)


def compute_wilcoxon_across_nfe(
    gen_dfs: dict[int, pd.DataFrame],
    nfe_pairs: list[tuple[int, int]],
    regions: list[str],
) -> pd.DataFrame:
    """Wilcoxon signed-rank for same-index volumes across NFE levels.

    Compares absolute deviation from real-population mean for matched
    generated volumes at different NFE levels.

    Args:
        gen_dfs: Dict mapping NFE to volumes DataFrame.
        nfe_pairs: Pairs of NFE levels to compare, e.g. [(1, 50), (10, 50)].
        regions: Column names to test.

    Returns:
        DataFrame with columns: nfe_a, nfe_b, region, statistic, p_value, stars.
    """
    from experiments.utils.settings import get_significance_stars

    rows: list[dict[str, Any]] = []
    for nfe_a, nfe_b in nfe_pairs:
        df_a = gen_dfs.get(nfe_a)
        df_b = gen_dfs.get(nfe_b)
        if df_a is None or df_b is None:
            continue

        # Match on subject index
        common_subjects = set(df_a["subject"]) & set(df_b["subject"])
        if len(common_subjects) < 10:
            logger.warning(
                "Too few common subjects (%d) for NFE %d vs %d",
                len(common_subjects),
                nfe_a,
                nfe_b,
            )
            continue

        df_a_matched = df_a[df_a["subject"].isin(common_subjects)].sort_values(
            "subject"
        )
        df_b_matched = df_b[df_b["subject"].isin(common_subjects)].sort_values(
            "subject"
        )

        for region in regions:
            if region not in df_a.columns or region not in df_b.columns:
                continue
            vals_a = df_a_matched[region].values
            vals_b = df_b_matched[region].values
            if len(vals_a) < 10:
                continue
            try:
                stat, p_val = stats.wilcoxon(vals_a, vals_b)
                rows.append({
                    "nfe_a": nfe_a,
                    "nfe_b": nfe_b,
                    "region": region,
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "stars": get_significance_stars(p_val),
                })
            except ValueError:
                # All differences are zero
                rows.append({
                    "nfe_a": nfe_a,
                    "nfe_b": nfe_b,
                    "region": region,
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "stars": "n.s.",
                })

    return pd.DataFrame(rows)
