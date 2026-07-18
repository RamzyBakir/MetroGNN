"""Build leakage-safe Phase 3 graph snapshots; final cutoff is opt-in."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.phase3 import (
    load_phase3_benchmark,
    load_phase3_config,
    load_phase3_embeddings,
    resolve_project_path,
)
from gnn_pipeline.evaluation.recommendation.temporal_graph import (
    RelationConfig,
    build_temporal_graph_snapshot,
    fit_training_topic_model,
    save_temporal_snapshot,
    save_temporal_topic_model,
)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument("--relations", default="full", help="Comma-separated ablation names")
    parser.add_argument("--include-final-test", action="store_true")
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    table = load_phase3_embeddings(config, benchmark, args.embedding)
    graph_config = config["temporal_graph"]
    topic_model = fit_training_topic_model(
        benchmark,
        table,
        training_cutoff=int(graph_config["topic_training_cutoff"]),
        n_clusters=int(graph_config["n_topics"]),
        random_seed=42,
    )
    output_root = resolve_project_path(graph_config["cache_root"])
    save_temporal_topic_model(topic_model, output_root)

    cutoffs = list(graph_config["training_cutoffs"]) + [
        graph_config["validation_cutoff"]
    ]
    if args.include_final_test:
        cutoffs.append(graph_config["final_test_cutoff"])

    for relation_name in parse_csv(args.relations):
        values = config["relation_ablations"].get(relation_name)
        if values is None:
            raise KeyError(f"Unknown relation ablation: {relation_name}")
        relations = RelationConfig(**values)
        relation_root = str(Path(output_root) / args.embedding / relation_name)
        for cutoff in sorted(set(int(value) for value in cutoffs)):
            snapshot = build_temporal_graph_snapshot(
                benchmark=benchmark,
                raw_csv_path=resolve_project_path(
                    config["experiment"]["raw_csv_path"]
                ),
                embedding_table=table,
                topic_model=topic_model,
                cutoff_year=cutoff,
                relation_config=relations,
            )
            save_temporal_snapshot(snapshot, relation_root)
            print(
                f"Saved {relation_name} cutoff={cutoff}: {snapshot.snapshot_id[:16]}"
            )


if __name__ == "__main__":
    main()
