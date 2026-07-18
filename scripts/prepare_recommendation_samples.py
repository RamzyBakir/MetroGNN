"""Create deterministic positive/unjudged Phase 3 training pairs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.features import TemporalFeatureStore
from gnn_pipeline.evaluation.recommendation.phase3 import (
    load_phase3_benchmark,
    load_phase3_config,
    load_phase3_embeddings,
    resolve_project_path,
)
from gnn_pipeline.evaluation.recommendation.sampling import build_pairwise_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    table = load_phase3_embeddings(config, benchmark, args.embedding)
    features = TemporalFeatureStore(
        benchmark,
        raw_csv_path=resolve_project_path(config["experiment"]["raw_csv_path"]),
        embedding_table=table,
        include_tfidf=True,
        include_bm25=False,
    )
    sampling = config["sampling"]
    samples = build_pairwise_samples(
        benchmark,
        feature_store=features,
        random_negatives=int(sampling["random_negatives"]),
        hard_negatives=int(sampling["hard_negatives"]),
        seed=int(sampling["seed"]),
    )
    output = Path(resolve_project_path(sampling["output_dir"])) / args.embedding
    manifest = samples.save(str(output))
    print(f"Pairwise samples: {manifest['sample_id']}")
    print(f"Triples: {manifest['num_triples']}")


if __name__ == "__main__":
    main()
