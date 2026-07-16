"""Build the versioned temporal citation recommendation benchmark."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.benchmark import (
    build_temporal_citation_benchmark,
)
from gnn_pipeline.evaluation.recommendation.runner import load_experiment_config


def resolve(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else PROJECT_ROOT / value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/citation_temporal_v1.yaml"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    benchmark_config = config["benchmark"]
    benchmark = build_temporal_citation_benchmark(
        resolve(benchmark_config["csv_path"]),
        splits=benchmark_config["splits"],
        query_text_fields=benchmark_config["query_text_fields"],
    )
    output_dir = resolve(args.output_dir or benchmark_config["output_dir"])
    manifest = benchmark.save(output_dir)

    print(f"Benchmark saved to: {output_dir}")
    print(f"Benchmark ID: {manifest['benchmark_id']}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print("Anomalies:")
    print(json.dumps(manifest["anomalies"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
