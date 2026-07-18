"""Explicitly build one frozen text-embedding cache for Phase 3."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.recommendation.embeddings import build_embedding_cache
from gnn_pipeline.evaluation.recommendation.phase3 import (
    embedding_spec,
    load_phase3_benchmark,
    load_phase3_config,
    resolve_project_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/phase3_model_validation.yaml"),
    )
    parser.add_argument("--embedding", required=True, choices=["minilm", "scholarly"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    benchmark = load_phase3_benchmark(config)
    spec = embedding_spec(config, args.embedding)
    table = build_embedding_cache(
        benchmark,
        spec,
        cache_root=resolve_project_path(config["embeddings"]["cache_root"]),
        force=args.force,
        device=args.device,
    )
    print(f"Embedding cache: {table.manifest['cache_key']}")
    print(f"Papers: {len(table.paper_ids)}, dimensions: {table.dimension}")


if __name__ == "__main__":
    main()
