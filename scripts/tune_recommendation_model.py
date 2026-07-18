"""Tune one Phase 3 architecture using validation data only."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import sha256_file
from gnn_pipeline.evaluation.recommendation.feature_models import PairwiseFeatureRecommender
from gnn_pipeline.evaluation.recommendation.fusion import ValidationFusionEvaluator
from gnn_pipeline.evaluation.recommendation.features import TemporalFeatureStore
from gnn_pipeline.evaluation.recommendation.graph_models import TemporalGraphRanker
from gnn_pipeline.evaluation.recommendation.graph_training import TemporalGraphTrainer
from gnn_pipeline.evaluation.recommendation.phase3 import (
    load_phase3_benchmark,
    load_phase3_config,
    load_phase3_embeddings,
    resolve_project_path,
)
from gnn_pipeline.evaluation.recommendation.runner import run_recommenders
from gnn_pipeline.evaluation.recommendation.sampling import load_pairwise_samples
from gnn_pipeline.evaluation.recommendation.temporal_graph import find_temporal_snapshot
from gnn_pipeline.evaluation.recommendation.text_training import TextDualEncoderTrainer
from gnn_pipeline.evaluation.recommendation.tuning import TrialOutcome, ValidationOnlyTuner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation_v2.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument(
        "--architecture",
        required=True,
        choices=[
            "text_dual_encoder",
            "feature_linear",
            "feature_mlp",
            "citation_sage",
            "hetero_sage",
            "hgt",
        ],
    )
    parser.add_argument("--relations", default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    if args.architecture not in config["architectures"]:
        raise ValueError(
            f"Architecture {args.architecture!r} is not enabled by this experiment config"
        )
    benchmark = load_phase3_benchmark(config)
    table = load_phase3_embeddings(config, benchmark, args.embedding)
    samples = load_pairwise_samples(
        str(Path(resolve_project_path(config["sampling"]["output_dir"])) / args.embedding)
    )
    search_config = config["model_search"]
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    output_root = Path(
        resolve_project_path(
            args.output_dir
            or str(
                Path(config["experiment"].get("output_root", "saved_results/phase3_validation"))
                / args.embedding
                / args.architecture
                / args.relations
            )
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    model_version = int(config["experiment"].get("model_version", 1))
    fusion_weights = config["model_search"].get("fusion_weights", [1.0])
    fusion_evaluator = (
        ValidationFusionEvaluator(benchmark)
        if args.architecture in {"text_dual_encoder", "citation_sage", "hetero_sage", "hgt"}
        else None
    )

    feature_store = None
    if args.architecture.startswith("feature_"):
        feature_store = TemporalFeatureStore(
            benchmark,
            raw_csv_path=resolve_project_path(config["experiment"]["raw_csv_path"]),
            embedding_table=table,
            include_tfidf=True,
            include_bm25=True,
        )

    graph_trainer = None
    graph_metadata = None
    snapshots = {}
    validation_snapshot = None
    if args.architecture in {"citation_sage", "hetero_sage", "hgt"}:
        graph_root = (
            Path(resolve_project_path(config["temporal_graph"]["cache_root"]))
            / args.embedding
            / args.relations
        )
        for cutoff in config["temporal_graph"]["training_cutoffs"]:
            snapshots[int(cutoff)] = find_temporal_snapshot(str(graph_root), int(cutoff))
        validation_snapshot = find_temporal_snapshot(
            str(graph_root), int(config["temporal_graph"]["validation_cutoff"])
        )
        for snapshot in [*snapshots.values(), validation_snapshot]:
            if snapshot.benchmark_id != benchmark.benchmark_id:
                raise ValueError("Temporal snapshot benchmark mismatch")
            if snapshot.embedding_cache_key != table.manifest["cache_key"]:
                raise ValueError("Temporal snapshot embedding cache mismatch")
        graph_metadata = validation_snapshot.data.metadata()
        graph_trainer = TemporalGraphTrainer(
            benchmark=benchmark,
            embedding_table=table,
            snapshots=snapshots,
            pairwise_samples=samples,
            validation_snapshot=validation_snapshot,
            fusion_evaluator=fusion_evaluator,
            device=args.device,
        )

    def trial_directory(trial_config, seed):
        identity = hashlib.sha256(
            json.dumps(
                {"config": trial_config, "seed": seed},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        directory = output_root / "checkpoints" / identity
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def callback(trial_config, seed):
        directory = trial_directory(trial_config, seed)
        common = {
            "hidden_dim": int(trial_config.get("hidden_dim", 64)),
            "dropout": float(trial_config.get("dropout", 0.1)),
            "learning_rate": float(trial_config.get("learning_rate", 1e-3)),
        }
        if args.architecture.startswith("feature_"):
            model = PairwiseFeatureRecommender(
                feature_store,
                samples,
                architecture="linear" if args.architecture == "feature_linear" else "mlp",
                hidden_dim=common["hidden_dim"],
                dropout=common["dropout"],
                learning_rate=common["learning_rate"],
                linear_c=float(trial_config.get("linear_c", 1.0)),
                epochs=int(search_config["max_epochs"]),
                seed=seed,
                name=args.architecture,
            )
            run = run_recommenders(
                benchmark,
                [model],
                splits=["validation"],
                seeds=[seed],
                top_k=20,
                metric_config={"ndcg": [10], "recall": [20]},
            )
            metrics = run.aggregate_metrics.iloc[0].to_dict()
            checkpoint = directory / "model.pt"
            model.save(str(checkpoint))
        elif args.architecture == "text_dual_encoder":
            trainer = TextDualEncoderTrainer(
                benchmark=benchmark,
                embedding_table=table,
                samples=samples,
                fusion_evaluator=fusion_evaluator,
                device=args.device,
            )
            result = trainer.train(
                hidden_dim=common["hidden_dim"],
                dropout=common["dropout"],
                learning_rate=common["learning_rate"],
                max_epochs=int(search_config["max_epochs"]),
                patience=int(search_config["early_stopping_patience"]),
                seed=seed,
                model_version=model_version,
                fusion_weights=fusion_weights,
            )
            metrics = result.best_validation
            checkpoint = directory / "model.pt"
            result.save(
                str(checkpoint),
                embedding_cache_key=table.manifest["cache_key"],
                sample_id=samples.manifest["sample_id"],
            )
        else:
            torch.manual_seed(seed)
            model = TemporalGraphRanker(
                architecture=args.architecture,
                metadata=graph_metadata,
                text_dim=table.dimension,
                hidden_dim=common["hidden_dim"],
                num_layers=int(trial_config.get("num_layers", 2)),
                dropout=common["dropout"],
                model_version=model_version,
            )
            result = graph_trainer.train(
                model,
                learning_rate=common["learning_rate"],
                max_epochs=int(search_config["max_epochs"]),
                patience=int(search_config["early_stopping_patience"]),
                seed=seed,
                fusion_weights=fusion_weights,
            )
            metrics = result.best_validation
            checkpoint = directory / "model.pt"
            result.save(
                str(checkpoint),
                snapshot_ids={
                    **{year: snapshot.snapshot_id for year, snapshot in snapshots.items()},
                    validation_snapshot.cutoff_year: validation_snapshot.snapshot_id,
                },
                sample_id=samples.manifest["sample_id"],
            )
            result.history.to_csv(directory / "history.csv", index=False)
        return TrialOutcome(
            metrics={"ndcg@10": metrics["ndcg@10"], "recall@20": metrics["recall@20"]},
            artifact_path=str(checkpoint.resolve()),
            artifact_sha256=sha256_file(str(checkpoint)),
        )

    full_space = search_config["search_space"]
    if args.architecture == "feature_linear":
        search_space = {"linear_c": search_config["linear_c"]}
    elif args.architecture in {"feature_mlp", "text_dual_encoder"}:
        search_space = {
            key: full_space[key] for key in ("hidden_dim", "dropout", "learning_rate")
        }
    else:
        search_space = full_space

    tuner = ValidationOnlyTuner(
        experiment_name=f"{config['experiment']['name']}:{args.embedding}:{args.architecture}:{args.relations}",
        benchmark_id=benchmark.benchmark_id,
        primary_metric=search_config["primary_metric"],
        tie_breaker=search_config["tie_breaker"],
        max_coarse_trials=int(search_config["maximum_coarse_trials"]),
    )
    selection = tuner.tune(
        search_space=search_space,
        callback=callback,
        seeds=seeds,
        top_configurations=int(search_config["top_configurations"]),
        output_dir=str(output_root),
    )
    print(f"Frozen validation selection: {selection['selection_id']}")
    print(f"Winning config: {selection['winner_config']}")


if __name__ == "__main__":
    main()
