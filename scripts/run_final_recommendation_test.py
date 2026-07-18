"""Run one auditable final test for a frozen Phase 3 model selection."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import sha256_file
from gnn_pipeline.evaluation.recommendation.embeddings import CachedEmbeddingRecommender
from gnn_pipeline.evaluation.recommendation.feature_models import PairwiseFeatureRecommender
from gnn_pipeline.evaluation.recommendation.features import TemporalFeatureStore
from gnn_pipeline.evaluation.recommendation.final_test import FinalTestRegistry
from gnn_pipeline.evaluation.recommendation.graph_training import (
    FrozenGraphRecommender,
    load_temporal_graph_ranker,
)
from gnn_pipeline.evaluation.recommendation.metrics import aggregate_metrics
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
from gnn_pipeline.evaluation.recommendation.sampling import load_pairwise_samples
from gnn_pipeline.evaluation.recommendation.statistics import (
    compare_paired_methods,
    holm_adjust,
    summarize_repeated_seeds,
)
from gnn_pipeline.evaluation.recommendation.temporal_graph import find_temporal_snapshot
from gnn_pipeline.evaluation.recommendation.text_training import (
    FrozenTextDualEncoderRecommender,
    load_text_dual_encoder,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", action="store_true", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation_v2.yaml"),
    )
    parser.add_argument("--frozen-selection", required=True)
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument(
        "--architecture",
        required=True,
        choices=["feature_linear", "feature_mlp", "text_dual_encoder", "citation_sage", "hetero_sage", "hgt"],
    )
    parser.add_argument("--relations", default="full")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    retrieval_tables = {
        name: load_phase3_embeddings(config, benchmark, name)
        for name in config["embeddings"]["candidates"]
    }
    table = retrieval_tables[args.embedding]
    samples = load_pairwise_samples(
        str(Path(resolve_project_path(config["sampling"]["output_dir"])) / args.embedding)
    )
    registry = FinalTestRegistry(
        resolve_project_path(config["final_test"]["registry_root"])
    )
    baseline_manifest_path = resolve_project_path(
        config["final_test"]["frozen_baseline_manifest"]
    )
    selection = json.loads(Path(args.frozen_selection).read_text())
    expected_experiment = (
        f"{config['experiment']['name']}:{args.embedding}:"
        f"{args.architecture}:{args.relations}"
    )
    if selection.get("experiment_name") != expected_experiment:
        raise ValueError(
            "Frozen selection does not match the requested embedding/architecture/relations"
        )
    feature_store = None
    final_snapshot = None
    if args.architecture.startswith("feature_"):
        feature_store = TemporalFeatureStore(
            benchmark,
            raw_csv_path=resolve_project_path(config["experiment"]["raw_csv_path"]),
            embedding_table=table,
            include_tfidf=True,
            include_bm25=True,
        )
    elif args.architecture in {"citation_sage", "hetero_sage", "hgt"}:
        graph_root = (
            Path(resolve_project_path(config["temporal_graph"]["cache_root"]))
            / args.embedding
            / args.relations
        )
        final_snapshot = find_temporal_snapshot(
            str(graph_root), int(config["temporal_graph"]["final_test_cutoff"])
        )
        if final_snapshot.benchmark_id != benchmark.benchmark_id:
            raise ValueError("Final graph snapshot benchmark mismatch")
        if final_snapshot.embedding_cache_key != table.manifest["cache_key"]:
            raise ValueError("Final graph snapshot embedding cache mismatch")

    plan = registry.create_plan(
        frozen_selection_path=args.frozen_selection,
        frozen_baseline_manifest_path=baseline_manifest_path,
        expected_seed_count=int(config["final_test"]["required_seed_count"]),
        final_context={
            "embedding": args.embedding,
            "embedding_cache_key": table.manifest["cache_key"],
            "retrieval_baseline_cache_keys": {
                name: value.manifest["cache_key"]
                for name, value in retrieval_tables.items()
            },
            "architecture": args.architecture,
            "relations": args.relations,
            "sample_id": samples.manifest["sample_id"],
            "final_snapshot_id": final_snapshot.snapshot_id
            if final_snapshot is not None
            else None,
        },
    )

    def callback(final_plan, output_dir):
        runs = []
        for artifact in selection["winner_artifacts"]:
            seed = int(artifact["seed"])
            checkpoint = artifact["artifact_path"]
            if args.architecture.startswith("feature_"):
                recommender = PairwiseFeatureRecommender.load(
                    checkpoint, feature_store=feature_store, samples=samples
                )
                recommender.name = args.architecture
            elif args.architecture == "text_dual_encoder":
                model, payload = load_text_dual_encoder(checkpoint, device=args.device)
                if payload.get("embedding_cache_key") != table.manifest["cache_key"]:
                    raise ValueError("Frozen text checkpoint embedding cache mismatch")
                if payload.get("sample_id") != samples.manifest["sample_id"]:
                    raise ValueError("Frozen text checkpoint sample mismatch")
                recommender = FrozenTextDualEncoderRecommender(
                    model, table, name=args.architecture, device=args.device
                )
            else:
                model, payload = load_temporal_graph_ranker(
                    checkpoint, snapshot=final_snapshot, device=args.device
                )
                if payload.get("sample_id") != samples.manifest["sample_id"]:
                    raise ValueError("Frozen graph checkpoint sample mismatch")
                if payload.get("training_config", {}).get("embedding_cache_key") != table.manifest["cache_key"]:
                    raise ValueError("Frozen graph checkpoint embedding cache mismatch")
                if payload.get("model_config", {}).get("architecture") != args.architecture:
                    raise ValueError("Frozen graph checkpoint architecture mismatch")
                recommender = FrozenGraphRecommender(
                    model=model,
                    snapshot=final_snapshot,
                    embedding_table=table,
                    name=args.architecture,
                    device=args.device,
                )
            runs.append(
                run_recommenders(
                    benchmark,
                    [recommender],
                    splits=["test"],
                    seeds=[seed],
                    top_k=20,
                )
            )

        for retrieval_name, retrieval_table in retrieval_tables.items():
            runs.append(
                run_recommenders(
                    benchmark,
                    [CachedEmbeddingRecommender(retrieval_table, retrieval_name)],
                    splits=["test"],
                    seeds=[plan["seeds"][0]],
                    top_k=20,
                )
            )

        predictions = pd.concat([run.predictions for run in runs], ignore_index=True)
        timings = pd.concat([run.timings for run in runs], ignore_index=True)
        per_query = pd.concat([run.per_query_metrics for run in runs], ignore_index=True)
        aggregate = aggregate_metrics(per_query)
        seed_summary = summarize_repeated_seeds(per_query)
        root = Path(output_dir)
        outputs = {}
        for name, frame in {
            "predictions": predictions,
            "timings": timings,
            "per_query_metrics": per_query,
            "aggregate_metrics": aggregate,
            "seed_summary": seed_summary,
        }.items():
            path = root / f"{name}.csv"
            frame.to_csv(path, index=False)
            outputs[name] = str(path)

        ci = confidence_interval_table(
            per_query,
            metrics=["ndcg@10", "recall@20"],
            split="test",
            n_resamples=10000,
            seed=plan["seeds"][0],
        )
        ci_path = root / "confidence_intervals.csv"
        ci.to_csv(ci_path, index=False)
        outputs["confidence_intervals"] = str(ci_path)

        baseline_manifest = json.loads(Path(baseline_manifest_path).read_text())
        baseline_metrics_path = (
            Path(baseline_manifest_path).parent
            / baseline_manifest["files"]["per_query_metrics"]
        )
        if sha256_file(str(baseline_metrics_path)) != baseline_manifest["file_sha256"]["per_query_metrics"]:
            raise ValueError("Frozen baseline per-query metrics checksum mismatch")
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        comparison_input = pd.concat(
            [
                per_query,
                baseline_metrics[
                    (baseline_metrics["method"] == "tfidf")
                    & (baseline_metrics["split"] == "test")
                ],
            ],
            ignore_index=True,
        )
        comparisons = []
        for baseline_name in ["tfidf", args.embedding]:
            comparisons.append(
                compare_paired_methods(
                    comparison_input,
                    proposed_method=args.architecture,
                    baseline_method=baseline_name,
                    metrics=["ndcg@10", "recall@20"],
                    split="test",
                    bootstrap_resamples=10000,
                    permutation_resamples=10000,
                    seed=plan["seeds"][0],
                )
            )
        comparison = pd.concat(comparisons, ignore_index=True)
        comparison["permutation_p_adjusted"] = holm_adjust(
            comparison["permutation_p"].tolist()
        )
        comparison["wilcoxon_p_adjusted"] = holm_adjust(
            comparison["wilcoxon_p"].tolist()
        )
        comparison_path = root / "paired_vs_frozen_baselines.csv"
        comparison.to_csv(comparison_path, index=False)
        outputs["paired_comparison"] = str(comparison_path)
        return outputs

    completion = registry.run_once(plan, callback)
    print(f"Final test complete: {completion['experiment_id']}")


if __name__ == "__main__":
    main()
