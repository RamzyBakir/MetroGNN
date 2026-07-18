"""Evaluate a cached frozen embedding retriever on validation only."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.embeddings import CachedEmbeddingRecommender
from gnn_pipeline.evaluation.recommendation.phase3 import (
    load_phase3_benchmark,
    load_phase3_config,
    load_phase3_embeddings,
    resolve_project_path,
)
from gnn_pipeline.evaluation.recommendation.runner import (
    confidence_interval_table,
    run_recommenders,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    table = load_phase3_embeddings(config, benchmark, args.embedding)
    recommender = CachedEmbeddingRecommender(table, name=args.embedding)
    run = run_recommenders(
        benchmark,
        [recommender],
        splits=["validation"],
        seeds=[int(config["experiment"]["seeds"][0])],
        top_k=20,
    )
    output = Path(
        resolve_project_path(
            args.output_dir
            or f"saved_results/phase3_validation/{args.embedding}/frozen_retrieval"
        )
    )
    run.save(str(output))
    ci = confidence_interval_table(
        run.per_query_metrics,
        metrics=["ndcg@10", "recall@20"],
        split="validation",
        n_resamples=10000,
        seed=int(config["experiment"]["seeds"][0]),
    )
    ci.to_csv(output / "confidence_intervals.csv", index=False)
    print(ci.to_string(index=False))
    print("Test evaluation is intentionally unavailable in this command.")


if __name__ == "__main__":
    main()
