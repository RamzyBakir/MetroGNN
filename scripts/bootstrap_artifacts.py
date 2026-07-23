"""Create or validate the runtime artifacts needed for MetroGNN inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import (  # noqa: E402
    ArtifactCompatibilityError,
    validate_pipeline_manifest,
)
from gnn_pipeline.config import ensure_dirs, get_path  # noqa: E402
from gnn_pipeline.data.vector_store import PaperVectorStore  # noqa: E402


def _validate_pipeline_artifacts() -> dict:
    return validate_pipeline_manifest(
        get_path("models", "pipeline_manifest"),
        csv_path=get_path("data", "csv_path"),
        graph_path=get_path("models", "knowledge_graph"),
    )


def _validate_vector_store(pipeline_id: str) -> dict:
    with PaperVectorStore(
        persist_dir=get_path("models", "vector_store"),
        manifest_path=get_path("models", "vector_store_manifest"),
    ) as store:
        return store.validate_compatibility(
            csv_path=get_path("data", "csv_path"),
            topic_assignments_path=get_path("data", "topic_assignments_path"),
            pipeline_id=pipeline_id,
        )


def ensure_runtime_artifacts(
    *,
    force_train: bool = False,
    force_vector: bool = False,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Ensure graph/model/topic/vector artifacts exist and match config.yaml."""
    ensure_dirs()
    csv_path = Path(get_path("data", "csv_path"))
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}. Include the train folder or update config.yaml."
        )

    manifest = None
    if not force_train:
        try:
            manifest = _validate_pipeline_artifacts()
            print("Pipeline artifacts ready.")
        except ArtifactCompatibilityError as exc:
            print(f"Pipeline artifacts need regeneration: {exc}")

    if force_train or manifest is None:
        print("Building graph, topics, manifest, splits, and GNN checkpoint ...")
        from scripts.train_pipeline import build_and_train

        _, _, manifest = build_and_train(force_topics=True, seed=seed, device=device)
        manifest = _validate_pipeline_artifacts()

    vector_manifest = None
    if not force_vector:
        try:
            vector_manifest = _validate_vector_store(manifest["pipeline_id"])
            print("Vector store ready.")
        except ArtifactCompatibilityError as exc:
            print(f"Vector store needs regeneration: {exc}")

    if force_vector or vector_manifest is None:
        print("Building semantic vector store ...")
        with PaperVectorStore(
            persist_dir=get_path("models", "vector_store"),
            manifest_path=get_path("models", "vector_store_manifest"),
        ) as store:
            store.ingest_papers(
                csv_path=get_path("data", "csv_path"),
                force_reingest=True,
                topic_assignments_path=get_path("data", "topic_assignments_path"),
                pipeline_id=manifest["pipeline_id"],
            )
        _validate_vector_store(manifest["pipeline_id"])

    print("Runtime artifacts are ready.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-vector", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=os.environ.get("METROGNN_DEVICE", "auto"),
        help="Training device used if artifacts need to be regenerated.",
    )
    args = parser.parse_args()
    try:
        ensure_runtime_artifacts(
            force_train=args.force_train,
            force_vector=args.force_vector,
            seed=args.seed,
            device=args.device,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
