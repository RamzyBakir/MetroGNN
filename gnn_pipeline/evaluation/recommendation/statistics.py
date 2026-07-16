"""Paired uncertainty estimates over actual recommendation query outcomes."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    return (
        float(values.mean()),
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def paired_bootstrap_difference(
    proposed: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if proposed.shape != baseline.shape or proposed.size == 0:
        raise ValueError("Paired samples must be non-empty and have identical shape")
    return bootstrap_mean_ci(
        proposed - baseline,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def paired_permutation_test(
    proposed: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired randomization test using sign flips."""
    differences = np.asarray(proposed, dtype=float) - np.asarray(baseline, dtype=float)
    if differences.size == 0:
        raise ValueError("Cannot test an empty paired sample")
    observed = abs(float(differences.mean()))
    if np.allclose(differences, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    # Chunking avoids allocating n_resamples x n_queries for large studies.
    remaining = n_resamples
    while remaining:
        size = min(remaining, 1000)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        remaining -= size
    return float((extreme + 1) / (n_resamples + 1))


def wilcoxon_p_value(proposed: Sequence[float], baseline: Sequence[float]) -> float:
    differences = np.asarray(proposed, dtype=float) - np.asarray(baseline, dtype=float)
    if differences.size == 0 or np.allclose(differences, 0):
        return 1.0
    return float(stats.wilcoxon(differences, zero_method="pratt").pvalue)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down family-wise error correction."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        corrected = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted.tolist()


def aggregate_seeds_per_query(
    per_query_metrics: pd.DataFrame, method: str, split: str
) -> pd.DataFrame:
    subset = per_query_metrics[
        (per_query_metrics["method"] == method)
        & (per_query_metrics["split"] == split)
    ].copy()
    if subset.empty:
        raise ValueError(f"No metrics for method={method!r}, split={split!r}")
    id_columns = {"query_id", "method", "seed", "split", "benchmark_id"}
    metric_columns = [column for column in subset.columns if column not in id_columns]
    return subset.groupby("query_id", as_index=False)[metric_columns].mean()


def compare_paired_methods(
    per_query_metrics: pd.DataFrame,
    *,
    proposed_method: str,
    baseline_method: str,
    metrics: Sequence[str],
    split: str = "test",
    bootstrap_resamples: int = 10000,
    permutation_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    correction: str = "holm",
) -> pd.DataFrame:
    """Compare methods on matched queries after averaging repeated seeds."""
    proposed = aggregate_seeds_per_query(per_query_metrics, proposed_method, split)
    baseline = aggregate_seeds_per_query(per_query_metrics, baseline_method, split)
    paired = proposed.merge(
        baseline,
        on="query_id",
        suffixes=("_proposed", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(proposed) or len(paired) != len(baseline):
        raise ValueError("Methods were not evaluated on identical query sets")

    rows: List[Dict] = []
    for offset, metric in enumerate(metrics):
        if f"{metric}_proposed" not in paired or f"{metric}_baseline" not in paired:
            raise KeyError(f"Metric not found in per-query results: {metric}")
        proposed_values = paired[f"{metric}_proposed"].to_numpy(dtype=float)
        baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
        estimate, lower, upper = paired_bootstrap_difference(
            proposed_values,
            baseline_values,
            n_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed + offset,
        )
        differences = proposed_values - baseline_values
        rows.append(
            {
                "proposed_method": proposed_method,
                "baseline_method": baseline_method,
                "split": split,
                "metric": metric,
                "num_queries": len(paired),
                "proposed_mean": float(proposed_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "mean_difference": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "permutation_p": paired_permutation_test(
                    proposed_values,
                    baseline_values,
                    n_resamples=permutation_resamples,
                    seed=seed + offset,
                ),
                "wilcoxon_p": wilcoxon_p_value(proposed_values, baseline_values),
                "wins": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "losses": int(np.sum(differences < -1e-12)),
            }
        )

    result = pd.DataFrame(rows)
    if correction == "holm" and not result.empty:
        result["permutation_p_adjusted"] = holm_adjust(
            result["permutation_p"].tolist()
        )
        result["wilcoxon_p_adjusted"] = holm_adjust(
            result["wilcoxon_p"].tolist()
        )
    else:
        result["permutation_p_adjusted"] = result["permutation_p"]
        result["wilcoxon_p_adjusted"] = result["wilcoxon_p"]
    return result


def summarize_repeated_seeds(per_query_metrics: pd.DataFrame) -> pd.DataFrame:
    """Report observed between-seed variation without inventing pseudo-runs."""
    id_columns = {"query_id", "method", "seed", "split", "benchmark_id"}
    metric_columns = [column for column in per_query_metrics if column not in id_columns]
    run_means = per_query_metrics.groupby(["method", "split", "seed"])[
        metric_columns
    ].mean()
    summary = run_means.groupby(["method", "split"]).agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
