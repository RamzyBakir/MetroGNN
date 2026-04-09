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
from typing import Dict, List, Tuple

import pandas as pd
import torch

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
    csv_path = None  # Change it to ur actucal data path
    topic_json_path = None # Change it to ur actucal data path

    if csv_path is None or topic_json_path is None:
        print("Error: Please set csv_path and topic_json_path to your actual data paths before running.")
        return

    print("1. Loading Data...")
    df = pd.read_csv(csv_path)
    with open(topic_json_path, "r") as f:
        topic_info_dict = json.load(f)
        topic_info_dict = {int(k): v for k, v in topic_info_dict.items()}

    print(f"Loaded {len(df)} papers and {len(topic_info_dict)} topics.")

    # Re-extract authors
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
        save_path="models/knowledge_graph_new.pt",
    )

    print("\n4. Initializing GNN Model...")
    # Get sizes from graph
    num_papers = graph["paper"].num_nodes
    num_authors = graph["author"].num_nodes
    num_topics = graph["topic"].num_nodes

    print(
        f"Graph nodes -> Papers: {num_papers}, Authors: {num_authors}, Topics: {num_topics}"
    )

    model = AdvancedHeteroGCN(
        hidden_channels=64,
        out_channels=32,
        num_layers=4,
        num_classes=3,
        num_topics=num_topics,
    )

    print("\n5. Training GNN...")
    trained_model, train_losses, val_metrics, *_ = train_enhanced_model(
        model, graph, epochs=300
    )

    # The training script saves to 'best_model.pth' by default. We need to rename to 'best_model_new.pth'
    if os.path.exists("best_model.pth"):
        os.rename("best_model.pth", "models/best_model_new.pth")
        print("Moved best weights to models/best_model_new.pth")
    else:
        torch.save(trained_model.state_dict(), "models/best_model_new.pth")
        print("Saved model weights to models/best_model_new.pth")

    print("\nPipeline Complete!")


if __name__ == "__main__":
    build_and_train()
