"""Paper-ready reporting from persisted recommendation outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def generate_markdown_report(
    *,
    benchmark_metadata: dict,
    confidence_intervals: pd.DataFrame,
    comparisons: pd.DataFrame | None = None,
    timings: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    lines = [
        "# Temporal Scholarly Recommendation Evaluation",
        "",
        f"- Benchmark ID: `{benchmark_metadata.get('benchmark_id', '')}`",
        f"- Candidate policy: {benchmark_metadata.get('candidate_policy', '')}",
        f"- Relevance source: {benchmark_metadata.get('relevance_source', '')}",
        f"- Queries: {benchmark_metadata.get('counts', {}).get('queries', 0)}",
        f"- Qrels: {benchmark_metadata.get('counts', {}).get('qrels', 0)}",
        "",
        "Uncited candidates are unjudged rather than confirmed irrelevant.",
        "",
        "## Ranking results",
        "",
        "| Method | Split | Metric | Queries | Mean | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in confidence_intervals.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.split} | {row.metric} | {row.num_queries} | "
            f"{row.mean:.4f} | [{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
        )

    if comparisons is not None and not comparisons.empty:
        lines.extend(
            [
                "",
                "## Paired comparisons",
                "",
                "| Proposed | Baseline | Metric | Difference | 95% CI | Adjusted p | W/T/L |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparisons.itertuples(index=False):
            lines.append(
                f"| {row.proposed_method} | {row.baseline_method} | {row.metric} | "
                f"{row.mean_difference:.4f} | [{row.ci_lower:.4f}, {row.ci_upper:.4f}] | "
                f"{row.permutation_p_adjusted:.6f} | {row.wins}/{row.ties}/{row.losses} |"
            )

    if timings is not None and not timings.empty:
        aggregations = {
            "latency_ms": ["median", lambda values: values.quantile(0.95), "count"]
        }
        if "prepare_ms" in timings.columns:
            aggregations["prepare_ms"] = ["first"]
        latency = timings.groupby(["method", "split"]).agg(aggregations).reset_index()
        latency.columns = [
            "method",
            "split",
            "p50_ms",
            "p95_ms",
            "queries",
            *(["prepare_ms"] if "prepare_ms" in timings.columns else []),
        ]
        lines.extend(
            [
                "",
                "## Query latency",
                "",
                "| Method | Split | Measurements | Index preparation ms | p50 ms | p95 ms |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in latency.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.split} | {row.queries} | "
                f"{getattr(row, 'prepare_ms', 0.0):.2f} | "
                f"{row.p50_ms:.2f} | {row.p95_ms:.2f} |"
            )

    report = "\n".join(lines) + "\n"
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
    return report


def generate_latex_results_table(
    confidence_intervals: pd.DataFrame,
    *,
    split: str = "test",
    metrics: Sequence[str] = ("ndcg@10", "recall@20"),
    output_path: str | None = None,
) -> str:
    subset = confidence_intervals[
        (confidence_intervals["split"] == split)
        & (confidence_intervals["metric"].isin(metrics))
    ]
    lookup = {
        (row.method, row.metric): row for row in subset.itertuples(index=False)
    }
    methods = sorted(subset["method"].unique())
    header = " & ".join(["Method"] + list(metrics))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Temporal prior-work recommendation results. Values are means with 95\\% query-bootstrap confidence intervals.}",
        "\\label{tab:temporal-recommendation}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = [method.replace("_", "\\_")]
        for metric in metrics:
            row = lookup.get((method, metric))
            cells.append(
                "--"
                if row is None
                else f"{row.mean:.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]"
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    latex = "\n".join(lines) + "\n"
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex)
    return latex
