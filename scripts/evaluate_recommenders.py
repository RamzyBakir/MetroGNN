"""Evaluate scholarly recommenders on the temporal citation benchmark."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.baselines import make_baseline
from gnn_pipeline.evaluation.recommendation.benchmark import TemporalCitationBenchmark
from gnn_pipeline.evaluation.recommendation.reporting import (
    generate_latex_results_table,
    generate_markdown_report,
)
from gnn_pipeline.evaluation.recommendation.runner import (
    confidence_interval_table,
    load_experiment_config,
    run_recommenders,
)
from gnn_pipeline.evaluation.recommendation.statistics import compare_paired_methods


def resolve(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else PROJECT_ROOT / value)


def parse_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/citation_temporal_v1.yaml"),
    )
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated baseline names")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds")
    parser.add_argument("--proposed", default=None, help="Method for paired comparison")
    parser.add_argument("--baseline", default=None, help="Baseline for paired comparison")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    benchmark_config = config["benchmark"]
    benchmark_dir = resolve(args.benchmark_dir or benchmark_config["output_dir"])
    benchmark = TemporalCitationBenchmark.load(benchmark_dir)

    method_names = (
        parse_csv(args.methods)
        if args.methods
        else list(config["baselines"]["enabled"])
    )
    neural_names = {"minilm", "scholarly"}
    selected_neural = neural_names.intersection(method_names)
    if selected_neural and not config["baselines"].get("neural", {}).get("enabled", False):
        raise ValueError(
            f"Neural encoding is disabled in the config; explicitly enable {sorted(selected_neural)}"
        )
    recommenders = [make_baseline(name, config["baselines"]) for name in method_names]

    seeds = (
        [int(value) for value in parse_csv(args.seeds)]
        if args.seeds
        else [int(value) for value in config["experiment"]["random_seeds"]]
    )
    splits = parse_csv(args.splits)
    metric_config = config["ranking"]["metrics"]
    run = run_recommenders(
        benchmark,
        recommenders,
        splits=splits,
        seeds=seeds,
        top_k=int(config["ranking"]["max_k"]),
        metric_config=metric_config,
    )

    output_root = Path(
        resolve(
            args.output_dir
            or f"saved_results/evaluations/{config['experiment']['name']}/{run.metadata['run_id'][:12]}"
        )
    )
    run.save(str(output_root))

    statistics_config = config["statistics"]
    primary_metrics = statistics_config["primary_metrics"]
    ci_frames = []
    for split in splits:
        ci_frames.append(
            confidence_interval_table(
                run.per_query_metrics,
                metrics=primary_metrics,
                split=split,
                n_resamples=int(statistics_config["bootstrap_resamples"]),
                confidence_level=float(statistics_config["confidence_level"]),
                seed=seeds[0],
            )
        )
    import pandas as pd

    confidence_intervals = pd.concat(ci_frames, ignore_index=True)
    confidence_intervals.to_csv(output_root / "confidence_intervals.csv", index=False)

    comparisons = None
    if bool(args.proposed) != bool(args.baseline):
        raise ValueError("Provide both --proposed and --baseline for paired comparison")
    if args.proposed and args.baseline:
        comparisons = compare_paired_methods(
            run.per_query_metrics,
            proposed_method=args.proposed,
            baseline_method=args.baseline,
            metrics=primary_metrics,
            split="test",
            bootstrap_resamples=int(statistics_config["bootstrap_resamples"]),
            permutation_resamples=int(statistics_config["permutation_resamples"]),
            confidence_level=float(statistics_config["confidence_level"]),
            seed=seeds[0],
            correction=statistics_config.get("correction", "holm"),
        )
        comparisons.to_csv(output_root / "paired_comparisons.csv", index=False)

    generate_markdown_report(
        benchmark_metadata=benchmark.metadata,
        confidence_intervals=confidence_intervals,
        comparisons=comparisons,
        timings=run.timings,
        output_path=str(output_root / "report.md"),
    )
    generate_latex_results_table(
        confidence_intervals,
        split="test",
        metrics=primary_metrics,
        output_path=str(output_root / "results_table.tex"),
    )
    print(f"Evaluation artifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
