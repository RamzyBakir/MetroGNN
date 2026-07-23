"""Regenerate the significance / confidence-interval report artifacts.

This replaces the stale root-level files from an older code version with a
*proper* statistical report built from real recommendation-benchmark data:
bootstrap confidence intervals (``plot_confidence_intervals``) and paired
t-tests / Wilcoxon signed-rank tests (``plot_significance_summary`` +
``generate_statistical_report``).

Outputs (at saved_results/):
    confidence_intervals.png
    significance_tests.png
    statistical_report.md
    significance_report.tex

Usage:
    python scripts/generate_significance_report.py
    python scripts/generate_significance_report.py --split test --out-dir saved_results
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.baselines import make_baseline
from gnn_pipeline.evaluation.recommendation.benchmark import TemporalCitationBenchmark
from gnn_pipeline.evaluation.recommendation.runner import (
    confidence_interval_table,
    load_experiment_config,
    run_recommenders,
)
from gnn_pipeline.evaluation.statistical_testing import (
    StatisticalTester,
    generate_statistical_report,
    plot_confidence_intervals,
    plot_significance_summary,
)


def resolve(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else PROJECT_ROOT / value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/citation_temporal_v1.yaml"),
    )
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "saved_results"))
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(args.config)
    benchmark_config = config["benchmark"]
    benchmark_dir = resolve(args.benchmark_dir or benchmark_config["output_dir"])
    benchmark = TemporalCitationBenchmark.load(benchmark_dir)

    method_names = list(config["baselines"]["enabled"])
    recommenders = [make_baseline(name, config["baselines"]) for name in method_names]

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [int(s) for s in config["experiment"]["random_seeds"]]
    )
    split = args.split
    metric_config = config["ranking"]["metrics"]
    statistics_config = config["statistics"]
    primary = statistics_config["primary_metrics"][0]

    print(f"Running {len(method_names)} recommenders on split='{split}' "
          f"over {len(seeds)} seeds ...")
    run = run_recommenders(
        benchmark,
        recommenders,
        splits=[split],
        seeds=seeds,
        top_k=int(config["ranking"]["max_k"]),
        metric_config=metric_config,
    )
    pq = run.per_query_metrics

    # ---- Bootstrap confidence intervals -> estimates dict ----
    ci = confidence_interval_table(
        pq,
        metrics=[primary],
        split=split,
        n_resamples=int(statistics_config["bootstrap_resamples"]),
        confidence_level=float(statistics_config["confidence_level"]),
        seed=seeds[0],
    )
    estimates = {
        row.method: (row.mean, row.ci_lower, row.ci_upper)
        for row in ci.itertuples()
    }
    plot_confidence_intervals(
        estimates, save_path=str(out_dir / "confidence_intervals.png")
    )

    # ---- Per-query averaged metric, per method ----
    metric_cols = [
        c for c in pq.columns
        if c not in ("method", "query_id", "split", "seed", "benchmark_id")
        and pd.api.types.is_numeric_dtype(pq[c])
    ]
    primary_col = primary if primary in metric_cols else metric_cols[0]
    subset = pq[pq["method"].isin(method_names) & (pq["split"] == split)]
    avg = (
        subset.groupby(["method", "query_id"])[metric_cols]
        .mean()
        .reset_index()
    )
    means = avg.groupby("method")[primary_col].mean()
    reference = means.idxmax()
    ref_vals = avg[avg["method"] == reference][primary_col].to_numpy()

    # ---- Paired significance vs the best baseline ----
    tester = StatisticalTester(random_seed=seeds[0])
    test_results = {}
    for method in method_names:
        if method == reference:
            continue
        vals = avg[avg["method"] == method][primary_col].to_numpy()
        if len(vals) < 2:
            continue
        test_results[f"{method} vs {reference}"] = tester.paired_ttest(
            proposed_values=vals, baseline_values=ref_vals
        )
        test_results[f"{method} vs {reference} (Wilcoxon)"] = tester.wilcoxon_test(
            proposed_values=vals, baseline_values=ref_vals
        )

    if test_results:
        plot_significance_summary(
            test_results, save_path=str(out_dir / "significance_tests.png")
        )
        generate_statistical_report(
            test_results,
            str(out_dir / "statistical_report.md"),
            format="markdown",
        )
        generate_statistical_report(
            test_results,
            str(out_dir / "significance_report.tex"),
            format="latex",
        )
        print(f"Significance report: {len(test_results)} comparisons "
              f"(reference = {reference})")
    else:
        print("Only one method present; skipping paired significance tests.")

    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
