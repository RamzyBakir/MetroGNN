"""
MetroGNN Community Visualization Script
========================================
Loads a trained model + knowledge graph from paths in config.yaml, detects the
research topics matching a query, runs the community analysis, and renders the
community network and quality-metrics figures into the results directory.

The network edges are taken from the real author collaboration graph (no
synthetic edges). Key members are highlighted and labeled.

Usage:
    python scripts/visualize_community.py
    python scripts/visualize_community.py --query "graph neural networks"
    python scripts/visualize_community.py --device cpu --save-dir my_results
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_pipeline.analysis.community_analysis import advanced_community_analysis
from gnn_pipeline.analysis.topic_detection import (
    extract_topics_from_query,
    extract_topics_from_query_dynamic,
    fallback_keyword_match,
)
from gnn_pipeline.config import ensure_dirs, get_path
from gnn_pipeline.data.topic_assignment import TopicAssigner
from gnn_pipeline.utils.visualization import (
    plot_quality_metrics,
    visualize_community_network,
)
from scripts.evaluate_model import load_model_and_graph


def load_topic_artifacts():
    topic_json = get_path("data", "topic_json_path")
    if not topic_json or not os.path.isfile(topic_json):
        raise FileNotFoundError(
            f"Topic metadata not found at {topic_json}. "
            "Run scripts/train_pipeline.py first to build topic artifacts."
        )
    with open(topic_json, "r") as f:
        raw = json.load(f)
    topic_info = {int(k): v for k, v in raw.items()}
    topic_names = {int(k): v.get("name", f"Topic {k}") for k, v in raw.items()}

    assigner = None
    centroids_path = get_path("data", "topic_centroids_path")
    if centroids_path and os.path.isfile(centroids_path):
        assigner = TopicAssigner.from_centroids(centroids_path)
    return topic_info, topic_names, assigner


def detect_topic_ids(query, topic_info, topic_names, assigner):
    try:
        if assigner is not None and hasattr(assigner, "rank_query_topics"):
            detected = extract_topics_from_query_dynamic(query, assigner, topic_info)
            if detected:
                return detected
    except Exception:
        pass

    detected = fallback_keyword_match(query, topic_info)
    if detected:
        return detected

    return extract_topics_from_query(query, topic_names)


def slugify(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "community"


def main():
    parser = argparse.ArgumentParser(description="MetroGNN Community Visualization")
    parser.add_argument(
        "--query",
        type=str,
        default="AI agents",
        help="Research query to detect topics from (default: 'AI agents')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save results (default: from config.yaml output.results_dir)",
    )
    args = parser.parse_args()

    ensure_dirs()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, graph = load_model_and_graph(device)

    print("Loading topic artifacts ...")
    topic_info, topic_names, assigner = load_topic_artifacts()

    print(f"Detecting topics for query: '{args.query}' ...")
    topic_ids = detect_topic_ids(args.query, topic_info, topic_names, assigner)
    if not topic_ids:
        print(f"No topics detected for query '{args.query}'. Nothing to visualize.")
        return
    detected_names = [topic_names.get(tid, f"Topic_{tid}") for tid in topic_ids]
    print(f"Detected topic IDs {topic_ids}: {', '.join(detected_names)}")

    print("Running model forward pass ...")
    model.eval()
    model_cpu = model.to("cpu")
    graph_cpu = graph.to("cpu")
    with torch.no_grad():
        x_dict, imp_logits, role_logits, inf_scores = model_cpu(
            graph_cpu.x_dict, graph_cpu.edge_index_dict
        )

    updated_embeddings = {
        "paper": x_dict.get("paper"),
        "author": x_dict.get("author", x_dict.get("researcher")),
        "paper_quality": x_dict.get("paper_quality"),
    }

    print("Analyzing community ...")
    community_data = advanced_community_analysis(
        topic_ids,
        updated_embeddings,
        graph_cpu,
        topic_names,
        imp_logits,
        role_logits,
        inf_scores,
    )
    if "error" in community_data:
        print(f"Community analysis failed: {community_data['error']}")
        return

    n_key = len(community_data.get("key_members", []))
    n_other = len(community_data.get("other_members", []))
    print(
        f"Community '{community_data.get('community_name')}' "
        f"has {n_key} key members and {n_other} other members."
    )

    results_dir = get_path("output", "results_dir") if args.save_dir is None else args.save_dir
    os.makedirs(results_dir, exist_ok=True)
    slug = slugify(args.query)

    collaboration_edges = graph_cpu[
        "author", "collaborates_with", "author"
    ].edge_index.numpy()

    print("Generating community network visualization ...")
    network_path = os.path.join(results_dir, f"community_network_{slug}.png")
    visualize_community_network(
        community_data, filename=network_path, collaboration_edges=collaboration_edges
    )
    plt.close("all")

    print("Generating community quality metrics ...")
    quality_path = os.path.join(results_dir, f"community_quality_{slug}.png")
    plot_quality_metrics(community_data, filename=quality_path)
    plt.close("all")

    print("\n" + "=" * 70)
    print("COMMUNITY VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {results_dir}/")
    print(f"  - {os.path.basename(network_path)}")
    print(f"  - {os.path.basename(quality_path)}")


if __name__ == "__main__":
    main()
