"""
Rebuild the Knowledge Graph and Retrain the GNN with the newly assigned topics.

This orchestrates the entire pipeline:
1. Load papers and topic info
2. Re-extract authors and build mappings
3. Create all node features (paper, author, topic)
4. Build PyG HeteroData graph
5. Train the AdvancedHeteroGCN model
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.feature_engineering import (
    create_all_features,
    print_feature_statistics,
)
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
from gnn_pipeline.models.training import train_enhanced_model


def parse_authors(authors_str: str) -> List[str]:
    try:
        authors = json.loads(authors_str.replace("'", '"'))
        return [str(a).strip() for a in authors] if isinstance(authors, list) else []
    except:
        try:
            authors = eval(authors_str)
            return (
                [str(a).strip() for a in authors] if isinstance(authors, list) else []
            )
        except:
            return []


def build_and_train():
    ensure_dirs()

    csv_path = get_path("data", "csv_path")
    topic_json_path = get_path("data", "topic_json_path")
    kg_save_path = get_path("models", "knowledge_graph")
    best_model_new_path = get_path("models", "best_model_new")

    if not csv_path:
        print("Error: Set data.csv_path in config.yaml before running.")
        return

    print("1. Loading Data...")
    df = pd.read_csv(csv_path)

    topic_info_dict = {}
    if topic_json_path and os.path.isfile(topic_json_path):
        with open(topic_json_path, "r") as f:
            topic_info_dict = json.load(f)
            topic_info_dict = {int(k): v for k, v in topic_info_dict.items()}
    else:
        print("No topic_json_path set (or file not found). Generating topics...")
        from gnn_pipeline.data.topic_assignment import assign_topics_to_dblp

        n_clusters = get_value("training", "n_clusters")
        df, _, topic_info_dict = assign_topics_to_dblp(df, n_clusters=n_clusters)

        cache_path = os.path.join(
            os.path.dirname(get_path("models", "best_model")),
            "topic_info_new.json",
        )
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in topic_info_dict.items()}, f, indent=2)
        print(f"Topics cached to {cache_path}")
        print("  Set data.topic_json_path in config.yaml to reuse on next run.")

    print(f"Loaded {len(df)} papers and {len(topic_info_dict)} topics.")

    df["authors_parsed"] = df["authors"].apply(parse_authors)
    all_authors = []
    for al in df["authors_parsed"]:
        all_authors.extend(al)

    unique_authors = list(sorted(set(all_authors)))
    author_to_idx = {author: idx for idx, author in enumerate(unique_authors)}
    df["author_indices"] = df["authors_parsed"].apply(
        lambda authors: [author_to_idx[a] for a in authors]
    )

    print("2. Creating Features...")
    paper_features, author_features, topic_features, feature_info = create_all_features(
        df, topic_info_dict, author_to_idx
    )
    print_feature_statistics(feature_info)

    print("\n3. Building Knowledge Graph...")
    graph, topic_names, _ = build_knowledge_graph_from_dblp(
        df,
        author_to_idx,
        topic_info_dict,
        paper_features,
        author_features,
        topic_features,
        save_path=kg_save_path,
    )

    print("\n4. Initializing GNN Model...")
    num_papers = graph["paper"].num_nodes
    num_authors = graph["author"].num_nodes
    num_topics = graph["topic"].num_nodes

    print(
        f"Graph nodes -> Papers: {num_papers}, Authors: {num_authors}, Topics: {num_topics}"
    )

    model = AdvancedHeteroGCN(
        hidden_channels=get_value("training", "hidden_channels"),
        out_channels=get_value("training", "out_channels"),
        num_layers=get_value("training", "num_layers"),
        num_classes=get_value("training", "num_classes"),
        num_topics=num_topics,
    )
    from gnn_pipeline.models.training import set_external_data

    set_external_data(df, author_to_idx)

    print("\n5. Training GNN...")
    epochs = get_value("training", "epochs")
    trained_model, train_losses, val_metrics, *_ = train_enhanced_model(
        model, graph, epochs=epochs
    )

    best_model_path = get_path("models", "best_model")
    if os.path.exists(best_model_path):
        os.rename(best_model_path, best_model_new_path)
        print(f"Moved best weights to {best_model_new_path}")
    else:
        torch.save(trained_model.state_dict(), best_model_new_path)
        print(f"Saved model weights to {best_model_new_path}")

    print("\nPipeline Complete!")


if __name__ == "__main__":
    build_and_train()
