"""Build the versioned MetroGNN recommendation graph and train its GNN."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from gnn_pipeline.artifacts import (
    create_pipeline_manifest,
    sha256_file,
    update_manifest_model,
    write_json_atomic,
)
from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.feature_engineering import create_all_features, print_feature_statistics
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.parsing import build_author_index, save_author_mapping
from gnn_pipeline.data.topic_assignment import EMBED_MODEL_NAME, prepare_topics
from gnn_pipeline.data.validation import (
    ensure_paper_identity_columns,
    validate_knowledge_graph,
    validate_papers_dataframe,
)
from gnn_pipeline.models.training import (
    set_external_data,
    set_training_seed,
)


def _resolve_device(device: str | None = None) -> torch.device:
    requested = (device or os.environ.get("METROGNN_DEVICE", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a GPU. "
            "Use --device cpu or run with the GPU Docker Compose file."
        )
    return torch.device(requested)


def build_and_train(force_topics: bool = False, seed: int = 42, device: str | None = None):
    ensure_dirs()
    set_training_seed(seed)
    resolved_device = _resolve_device(device)

    csv_path = get_path("data", "csv_path")
    topic_info_path = get_path("data", "topic_json_path")
    topic_assignments_path = get_path("data", "topic_assignments_path")
    topic_centroids_path = get_path("data", "topic_centroids_path")
    graph_path = get_path("models", "knowledge_graph")
    model_path = get_path("models", "best_model_new")
    author_mapping_path = get_path("models", "author_mapping")
    manifest_path = get_path("models", "pipeline_manifest")

    print("1. Loading and validating papers...")
    df = ensure_paper_identity_columns(pd.read_csv(csv_path))
    data_stats = validate_papers_dataframe(df)
    dataset_sha256 = sha256_file(csv_path)
    print(f"Dataset: {data_stats}")

    print("\n2. Loading or generating versioned topics...")
    n_clusters = get_value("training", "n_clusters")
    df, topic_info, topic_assigner, generated = prepare_topics(
        df,
        n_clusters=n_clusters,
        assignments_path=topic_assignments_path,
        topic_info_path=topic_info_path,
        centroids_path=topic_centroids_path,
        dataset_sha256=dataset_sha256,
        force_regenerate=force_topics,
    )
    print(f"Topics: {len(topic_info)} ({'generated' if generated else 'loaded'})")

    print("\n3. Resolving stable author identities...")
    author_to_idx, author_records, paper_author_indices, author_stats = build_author_index(df)
    df["author_indices"] = paper_author_indices
    df["authors_parsed"] = [
        [author_records[idx]["author_key"] for idx in indices]
        for indices in paper_author_indices
    ]
    save_author_mapping(author_records, author_mapping_path)
    print(f"Authors: {author_stats}")

    print("\n4. Engineering features...")
    paper_features, author_features, topic_features, feature_info = create_all_features(
        df, topic_info, author_to_idx
    )
    print_feature_statistics(feature_info)

    print("\n5. Building and validating the heterogeneous graph...")
    graph, topic_names, _ = build_knowledge_graph_from_dblp(
        df,
        author_to_idx,
        topic_info,
        paper_features,
        author_features,
        topic_features,
        save_path=graph_path,
    )
    graph_stats = validate_knowledge_graph(
        graph,
        expected_papers=len(df),
        expected_authors=len(author_records),
        expected_topics=len(topic_info),
    )
    print(f"Graph: {graph_stats}")

    manifest = create_pipeline_manifest(
        csv_path=csv_path,
        graph_path=graph_path,
        author_mapping_path=author_mapping_path,
        topic_assignments_path=topic_assignments_path,
        topic_info_path=topic_info_path,
        topic_centroids_path=topic_centroids_path,
        embedding_model=EMBED_MODEL_NAME,
        n_clusters=len(topic_info),
        graph_stats={**graph_stats, "data_validation": data_stats, "author_identity": author_stats},
    )
    write_json_atomic(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")

    print(f"\n6. Training the versioned GNN on {resolved_device}...")
    model_kwargs = {
        "hidden_channels": get_value("training", "hidden_channels"),
        "out_channels": get_value("training", "out_channels"),
        "num_layers": get_value("training", "num_layers"),
        "num_classes": get_value("training", "num_classes"),
        "num_topics": len(topic_info),
    }
    from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
    from gnn_pipeline.models.training import train_enhanced_model

    model = AdvancedHeteroGCN(**model_kwargs).to(resolved_device)
    train_fn = train_enhanced_model
    train_kwargs = {"device": resolved_device}

    set_external_data(df, author_to_idx)
    trained_model, train_losses, val_metrics, *_ = train_fn(
        model,
        graph,
        epochs=get_value("training", "epochs"),
        checkpoint_path=model_path,
        pipeline_id=manifest["pipeline_id"],
        seed=seed,
        **train_kwargs,
    )
    update_manifest_model(manifest_path, model_path, trained_model.get_config())

    print("\nPipeline complete.")
    print(f"Model: {model_path}")
    print(f"Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
    return trained_model, graph, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-topics",
        action="store_true",
        help="Explicitly regenerate topic assignments and centroids",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=os.environ.get("METROGNN_DEVICE", "auto"),
        help="Training device. auto uses CUDA when PyTorch can see a GPU.",
    )
    args = parser.parse_args()
    try:
        build_and_train(
            force_topics=args.force_topics,
            seed=args.seed,
            device=args.device,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
