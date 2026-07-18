"""Run predeclared Phase 3 relation or feature ablations on validation only."""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import sha256_file
from gnn_pipeline.evaluation.recommendation.feature_models import PairwiseFeatureRecommender
from gnn_pipeline.evaluation.recommendation.features import TemporalFeatureStore
from gnn_pipeline.evaluation.recommendation.fusion import ValidationFusionEvaluator
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
from gnn_pipeline.evaluation.recommendation.tuning import TrialOutcome, run_validation_ablations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation_v2.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument("--kind", required=True, choices=["relation", "feature"])
    parser.add_argument("--architecture", default="hetero_sage")
    parser.add_argument("--frozen-selection", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    table = load_phase3_embeddings(config, benchmark, args.embedding)
    samples = load_pairwise_samples(
        str(Path(resolve_project_path(config["sampling"]["output_dir"])) / args.embedding)
    )
    selection = json.loads(Path(args.frozen_selection).read_text())
    if selection.get("selection_split") != "validation":
        raise ValueError("Ablations require a validation-only frozen selection")
    expected_experiment = (
        f"{config['experiment']['name']}:{args.embedding}:{args.architecture}:full"
    )
    if selection.get("experiment_name") != expected_experiment:
        raise ValueError("Frozen selection does not match this ablation experiment")
    winner = selection["winner_config"]
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    output_root = Path(
        resolve_project_path(
            args.output_dir
            or str(
                Path(config["experiment"].get("ablation_output_root", "saved_results/phase3_ablation"))
                / args.embedding
                / args.kind
                / args.architecture
            )
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.kind == "feature":
        ablations = config["feature_ablations"]

        def callback(ablation_config, seed):
            store = TemporalFeatureStore(
                benchmark,
                raw_csv_path=resolve_project_path(config["experiment"]["raw_csv_path"]),
                embedding_table=table,
                include_tfidf=True,
                include_bm25=True,
                enabled_features=ablation_config["enabled"],
            )
            model = PairwiseFeatureRecommender(
                store,
                samples,
                architecture="mlp",
                hidden_dim=int(winner.get("hidden_dim", 64)),
                dropout=float(winner.get("dropout", 0.1)),
                learning_rate=float(winner.get("learning_rate", 1e-3)),
                epochs=int(config["model_search"]["max_epochs"]),
                seed=seed,
                name="feature_mlp_ablation",
            )
            run = run_recommenders(
                benchmark,
                [model],
                splits=["validation"],
                seeds=[seed],
                top_k=20,
                metric_config={"ndcg": [10], "recall": [20]},
            )
            import hashlib

            ablation_id = hashlib.sha256(
                json.dumps(ablation_config, sort_keys=True).encode()
            ).hexdigest()[:12]
            directory = output_root / f"seed_{seed}" / ablation_id
            checkpoint = directory / "model.pt"
            model.save(str(checkpoint))
            row = run.aggregate_metrics.iloc[0]
            return TrialOutcome(
                metrics={"ndcg@10": row["ndcg@10"], "recall@20": row["recall@20"]},
                artifact_path=str(checkpoint.resolve()),
                artifact_sha256=sha256_file(str(checkpoint)),
            )

    else:
        ablations = config["relation_ablations"]
        fusion_evaluator = ValidationFusionEvaluator(benchmark)
        model_version = int(config["experiment"].get("model_version", 1))
        fusion_weights = config["model_search"].get("fusion_weights", [1.0])

        def callback(ablation_config, seed):
            relation_name = next(
                name for name, values in ablations.items() if values == ablation_config
            )
            graph_root = (
                Path(resolve_project_path(config["temporal_graph"]["cache_root"]))
                / args.embedding
                / relation_name
            )
            snapshots = {
                int(cutoff): find_temporal_snapshot(str(graph_root), int(cutoff))
                for cutoff in config["temporal_graph"]["training_cutoffs"]
            }
            validation = find_temporal_snapshot(
                str(graph_root), int(config["temporal_graph"]["validation_cutoff"])
            )
            trainer = TemporalGraphTrainer(
                benchmark=benchmark,
                embedding_table=table,
                snapshots=snapshots,
                pairwise_samples=samples,
                validation_snapshot=validation,
                fusion_evaluator=fusion_evaluator,
                device=args.device,
            )
            torch.manual_seed(seed)
            model = TemporalGraphRanker(
                architecture=args.architecture,
                metadata=validation.data.metadata(),
                text_dim=table.dimension,
                hidden_dim=int(winner.get("hidden_dim", 64)),
                num_layers=int(winner.get("num_layers", 2)),
                dropout=float(winner.get("dropout", 0.1)),
                model_version=model_version,
            )
            result = trainer.train(
                model,
                learning_rate=float(winner.get("learning_rate", 1e-3)),
                max_epochs=int(config["model_search"]["max_epochs"]),
                patience=int(config["model_search"]["early_stopping_patience"]),
                seed=seed,
                fusion_weights=fusion_weights,
            )
            checkpoint = output_root / relation_name / f"seed_{seed}" / "model.pt"
            result.save(
                str(checkpoint),
                snapshot_ids={
                    **{year: value.snapshot_id for year, value in snapshots.items()},
                    validation.cutoff_year: validation.snapshot_id,
                },
                sample_id=samples.manifest["sample_id"],
            )
            return TrialOutcome(
                metrics=result.best_validation,
                artifact_path=str(checkpoint.resolve()),
                artifact_sha256=sha256_file(str(checkpoint)),
            )

    frame = run_validation_ablations(
        ablations=ablations,
        callback=callback,
        seeds=seeds,
        output_path=str(output_root / "validation_ablation_runs.csv"),
    )
    print(frame.groupby("ablation")[["ndcg@10", "recall@20"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
