import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx

from gnn_pipeline.config import get_path


def _default_path(*config_keys: str, fallback: str) -> str:
    try:
        p = get_path(*config_keys)
        return p if p else fallback
    except Exception:
        return fallback


def ensure_directory_exists(filename):

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}")
    return directory


def save_comprehensive_results(
    comprehensive_strategy, community_data, user_query, filename_prefix=None
):

    if filename_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"research_strategy_{timestamp}"

    ensure_directory_exists(filename_prefix + "_strategy.json")

    strategy_filename = f"{filename_prefix}_strategy.json"
    with open(strategy_filename, "w", encoding="utf-8") as f:
        json.dump(comprehensive_strategy, f, indent=2, ensure_ascii=False)
    print(f" Strategy saved to: {strategy_filename}")

    community_filename = f"{filename_prefix}_community_data.json"
    with open(community_filename, "w", encoding="utf-8") as f:
        json.dump(community_data, f, indent=2, ensure_ascii=False)
    print(f" Community data saved to: {community_filename}")

    save_key_metrics_to_csv(
        comprehensive_strategy, community_data, f"{filename_prefix}_metrics.csv"
    )

    save_implementation_plan(
        comprehensive_strategy, f"{filename_prefix}_implementation_plan.json"
    )

    return strategy_filename


def save_key_metrics_to_csv(strategy, community_data, filename):

    metrics_data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "user_query": [strategy["metadata"]["query"]],
        "user_intent": [strategy["metadata"]["user_intent"]],
        "community_name": [strategy["metadata"]["community"]],
        "community_size": [community_data["community_metrics"]["size"]],
        "confidence_score": [strategy["metadata"]["confidence_score"]],
        "research_quality": [community_data["enhanced_analysis"]["research_quality"]],
        "collaboration_potential": [
            community_data["enhanced_analysis"]["collaboration_potential"]
        ],
        "topic_alignment": [community_data["enhanced_analysis"]["topic_alignment"]],
        "key_members_count": [len(community_data["key_members"])],
        "total_members_count": [
            len(community_data["key_members"])
            + len(community_data.get("other_members", []))
        ],
    }

    df = pd.DataFrame(metrics_data)
    df.to_csv(filename, index=False)
    print(f" Metrics saved to: {filename}")


def save_implementation_plan(strategy, filename):

    implementation_data = {
        "immediate_actions": strategy["implementation_guide"]["immediate_next_steps"],
        "key_connections": strategy["implementation_guide"]["key_connections"],
        "quarterly_goals": strategy["implementation_guide"]["quarterly_goals"],
        "resources": strategy["implementation_guide"]["resource_recommendations"],
        "success_framework": strategy["success_framework"],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(implementation_data, f, indent=2, ensure_ascii=False)
    print(f" Implementation plan saved to: {filename}")


def save_training_artifacts(model, train_losses, val_metrics, filename_prefix):
    """Save training artifacts for later analysis"""
    # Save model state
    model_filename = f"{filename_prefix}_model.pth"
    torch.save(model.state_dict(), model_filename)
    print(f" Model saved to: {model_filename}")

    # Save training history
    training_history = {
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    history_filename = f"{filename_prefix}_training_history.json"
    with open(history_filename, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f" Training history saved to: {history_filename}")


def save_community_network(community_data, filename_prefix):

    network_data = {
        "key_members": community_data["key_members"],
        "community_metrics": community_data["community_metrics"],
        "enhanced_analysis": community_data["enhanced_analysis"],
        "query_topics": community_data["query_topics"],
    }

    network_filename = f"{filename_prefix}_network_data.json"
    with open(network_filename, "w", encoding="utf-8") as f:
        json.dump(network_data, f, indent=2, ensure_ascii=False)
    print(f"🕸️ Network data saved to: {network_filename}")


def save_knowledge_graph_data(data, topic_names, filename=None):

    if filename is None:
        filename = get_path("models", "knowledge_graph_data")

    ensure_directory_exists(filename)

    edge_types_str = [str(edge_type) for edge_type in data.edge_types]

    graph_data = {
        "hetero_data": data,
        "topic_names": topic_names,
        "node_types": list(data.node_types),
        "edge_types": edge_types_str,
        "node_counts": {
            node_type: data[node_type].x.shape[0] for node_type in data.node_types
        },
        "edge_counts": {
            str(edge_type): data[edge_type].edge_index.shape[1]
            for edge_type in data.edge_types
        },
        "node_features": {
            node_type: data[node_type].x.shape[1] for node_type in data.node_types
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(filename, "wb") as f:
        pickle.dump(graph_data, f)
    print(f" Knowledge graph saved to: {filename}")

    metadata_filename = filename.replace(".pkl", "_metadata.json")
    save_graph_metadata_readable(graph_data, metadata_filename)

    return graph_data


def save_graph_metadata_readable(graph_data, filename):

    ensure_directory_exists(filename)

    def convert_tuples_to_strings(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if isinstance(key, tuple):
                    new_key = str(key)
                else:
                    new_key = key

                new_dict[new_key] = convert_tuples_to_strings(value)
            return new_dict
        elif isinstance(obj, list):
            return [convert_tuples_to_strings(item) for item in obj]
        else:
            return obj

    readable_data = {
        "timestamp": graph_data["timestamp"],
        "topic_names": graph_data["topic_names"],
        "node_types": graph_data["node_types"],
        "edge_types": [str(edge_type) for edge_type in graph_data["edge_types"]],
        "node_counts": convert_tuples_to_strings(graph_data["node_counts"]),
        "edge_counts": convert_tuples_to_strings(graph_data["edge_counts"]),
        "node_features": convert_tuples_to_strings(graph_data["node_features"]),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(readable_data, f, indent=2, ensure_ascii=False)
    print(f" Graph metadata saved to: {filename}")


def save_graph_to_networkx(data, filename=None):

    if filename is None:
        filename = get_path("models", "networkx_graph")

    import networkx as nx

    ensure_directory_exists(filename)

    G = nx.Graph()

    for node_type in data.node_types:
        for i in range(data[node_type].x.shape[0]):
            node_id = f"{node_type}_{i}"
            features = data[node_type].x[i].numpy().tolist()
            G.add_node(
                node_id, node_type=node_type, features=features, original_index=i
            )

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index.numpy()

        for j in range(edge_index.shape[1]):
            src_idx = edge_index[0, j]
            dst_idx = edge_index[1, j]
            src_id = f"{src_type}_{src_idx}"
            dst_id = f"{dst_type}_{dst_idx}"

            G.add_edge(src_id, dst_id, relation=relation, edge_type=str(edge_type))

    with open(filename, "wb") as f:
        pickle.dump(G, f)
    print(f"🕸️ NetworkX graph saved to: {filename}")

    return G


def save_node_edge_lists(data, filename_prefix=None):

    if filename_prefix is None:
        filename_prefix = os.path.join(
            os.path.dirname(get_path("models", "knowledge_graph_data")),
            "knowledge_graph",
        )

    ensure_directory_exists(filename_prefix + "_nodes.csv")

    all_nodes = []
    for node_type in data.node_types:
        for i in range(data[node_type].x.shape[0]):
            node_features = data[node_type].x[i].numpy()
            node_data = {
                "node_id": f"{node_type}_{i}",
                "node_type": node_type,
                "original_index": i,
            }

            if node_type == "researcher":
                feature_names = [
                    "paper_count",
                    "collaboration_count",
                    "topic_diversity",
                    "years_active",
                    "citation_impact",
                    "reserved_5",
                    "reserved_6",
                    "reserved_7",
                ]
            elif node_type == "paper":
                feature_names = [
                    "citation_count",
                    "topic_relevance",
                    "recency",
                    "author_count",
                    "reserved_4",
                    "reserved_5",
                    "reserved_6",
                    "reserved_7",
                    "reserved_8",
                    "reserved_9",
                ]
            else:
                feature_names = [
                    f"topic_{j}_indicator" for j in range(len(node_features))
                ]

            for j, feat in enumerate(node_features):
                if j < len(feature_names):
                    col_name = feature_names[j]
                else:
                    col_name = f"feature_{j}"
                node_data[col_name] = float(feat)

            all_nodes.append(node_data)

    nodes_filename = f"{filename_prefix}_nodes.csv"
    nodes_df = pd.DataFrame(all_nodes)
    nodes_df.to_csv(nodes_filename, index=False)
    print(f" Node list saved to: {nodes_filename}")

    all_edges = []
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index.numpy()

        for j in range(edge_index.shape[1]):
            src_idx = edge_index[0, j]
            dst_idx = edge_index[1, j]

            all_edges.append(
                {
                    "source_node": f"{src_type}_{src_idx}",
                    "target_node": f"{dst_type}_{dst_idx}",
                    "relation": relation,
                    "edge_type": str(edge_type),
                    "source_type": src_type,
                    "target_type": dst_type,
                }
            )

    edges_filename = f"{filename_prefix}_edges.csv"
    edges_df = pd.DataFrame(all_edges)
    edges_df.to_csv(edges_filename, index=False)
    print(f" Edge list saved to: {edges_filename}")

    return nodes_df, edges_df


def create_graph_summary(data, topic_names, filename=None):

    if filename is None:
        filename = get_path("output", "graph_summary")

    ensure_directory_exists(filename)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("KNOWLEDGE GRAPH SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        f.write("OVERVIEW:\n")
        f.write(
            f"- Total nodes: {sum(data[node_type].x.shape[0] for node_type in data.node_types)}\n"
        )
        f.write(
            f"- Total edges: {sum(data[edge_type].edge_index.shape[1] for edge_type in data.edge_types)}\n"
        )
        f.write(f"- Node types: {len(data.node_types)}\n")
        f.write(f"- Edge types: {len(data.edge_types)}\n\n")

        f.write("NODE COUNTS:\n")
        for node_type in data.node_types:
            count = data[node_type].x.shape[0]
            features = data[node_type].x.shape[1]
            f.write(f"- {node_type}: {count} nodes, {features} features each\n")

        f.write("\nEDGE COUNTS:\n")
        for edge_type in data.edge_types:
            count = data[edge_type].edge_index.shape[1]
            f.write(f"- {edge_type}: {count} edges\n")

        f.write(f"\nTOPICS:\n")
        for topic_id, topic_name in topic_names.items():
            f.write(f"- {topic_id}: {topic_name}\n")

        if "author" in data.node_types:
            researcher_features = data["author"].x.numpy()
            f.write(f"\nAUTHOR STATISTICS:\n")
            f.write(
                f"- Average papers per author: {np.mean(researcher_features[:, 0]):.1f}\n"
            )
            f.write(
                f"- Average collaborators: {np.mean(researcher_features[:, 1]):.1f}\n"
            )
            f.write(
                f"- Average years active: {np.mean(researcher_features[:, 3]):.1f}\n"
            )

        f.write(f"\nGenerated: {datetime.now().isoformat()}\n")

    print(f" Graph summary saved to: {filename}")


def explain_node_features(data, topic_names):

    feature_explanations = {
        "author": [
            "Feature 0: Paper count - Total number of papers published",
            "Feature 1: Collaboration count - Number of collaborators",
            "Feature 2: Topic diversity - How diverse their research topics are (0-1)",
            "Feature 3: Years active - Number of years in research",
            "Feature 4: Citation impact - Average citations per paper",
            "Feature 5-7: Reserved for future use",
        ],
        "paper": [
            "Feature 0: Citation count - Number of citations received",
            "Feature 1: Topic relevance - How relevant to main topics (0-1)",
            "Feature 2: Recency - 2025 - publication year (lower = more recent)",
            "Feature 3: Author count - Number of authors",
            "Feature 4-9: Reserved for future use",
        ],
        "topic": [
            "Feature 0-7: One-hot encoding - Each feature represents one topic",
            f"Topic mapping: {topic_names}",
        ],
    }

    print("\n🔍 NODE FEATURE EXPLANATIONS:")
    print("=" * 60)

    for node_type in data.node_types:
        if node_type in feature_explanations:
            print(f"\n{node_type.upper()} NODES:")
            print(f"Number of {node_type}s: {data[node_type].x.shape[0]}")
            print(f"Number of features: {data[node_type].x.shape[1]}")
            for explanation in feature_explanations[node_type]:
                print(f"  • {explanation}")

            print(f"\n  Sample feature values (first 3 {node_type}s):")
            for i in range(min(3, data[node_type].x.shape[0])):
                features = data[node_type].x[i].numpy()
                print(f"    {node_type}_{i}: {[f'{x:.2f}' for x in features]}")

    return feature_explanations


def explain_edge_types(data):

    edge_explanations = {
        ("researcher", "authored", "paper"): "Researcher authored a paper",
        (
            "paper",
            "authored_by",
            "researcher",
        ): "Paper is authored by researcher (reverse)",
        ("paper", "has_topic", "topic"): "Paper belongs to a research topic",
        ("topic", "has_paper", "paper"): "Topic contains paper (reverse)",
        (
            "researcher",
            "collaborates_with",
            "researcher",
        ): "Researchers collaborate together",
    }

    print("\n EDGE TYPE EXPLANATIONS:")
    print("=" * 60)

    for edge_type in data.edge_types:
        explanation = edge_explanations.get(edge_type, "Unknown relationship")
        count = data[edge_type].edge_index.shape[1]
        print(f"  • {edge_type}: {explanation} ({count} edges)")

    return edge_explanations


def create_data_dictionary(data, topic_names, filename=None):
    if filename is None:
        filename = get_path("output", "data_dictionary")

    ensure_directory_exists(filename)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("KNOWLEDGE GRAPH DATA DICTIONARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("NODE TYPES AND FEATURES:\n")
        f.write("-" * 30 + "\n")

        for node_type in data.node_types:
            f.write(f"\n{node_type.upper()}:\n")
            f.write(f"  Count: {data[node_type].x.shape[0]}\n")
            f.write(f"  Features: {data[node_type].x.shape[1]}\n")

            if node_type == "researcher":
                f.write("  Feature meanings:\n")
                f.write("    0: Paper count\n")
                f.write("    1: Collaboration count\n")
                f.write("    2: Topic diversity (0-1)\n")
                f.write("    3: Years active\n")
                f.write("    4: Citation impact\n")
                f.write("    5-7: Reserved\n")
            elif node_type == "paper":
                f.write("  Feature meanings:\n")
                f.write("    0: Citation count\n")
                f.write("    1: Topic relevance (0-1)\n")
                f.write("    2: Recency (2025 - publication year)\n")
                f.write("    3: Author count\n")
                f.write("    4-9: Reserved\n")
            elif node_type == "topic":
                f.write("  Feature meanings:\n")
                f.write("    One-hot encoding (each feature = one topic)\n")
                f.write(f"    Topic names: {topic_names}\n")

        f.write("\nEDGE TYPES:\n")
        f.write("-" * 20 + "\n")

        edge_descriptions = {
            ("researcher", "authored", "paper"): "Author relationship",
            ("paper", "authored_by", "researcher"): "Reverse author relationship",
            ("paper", "has_topic", "topic"): "Paper classification",
            ("topic", "has_paper", "paper"): "Reverse classification",
            ("researcher", "collaborates_with", "researcher"): "Collaboration network",
        }

        for edge_type in data.edge_types:
            desc = edge_descriptions.get(edge_type, "Unknown")
            count = data[edge_type].edge_index.shape[1]
            f.write(f"  {edge_type}: {desc} ({count} edges)\n")

        f.write(f"\n Generated: {datetime.now().isoformat()}\n")

    print(f" Data dictionary saved to: {filename}")


def save_summary_report(all_results, filename):

    ensure_directory_exists(filename)

    summary = {
        "total_queries": len(all_results),
        "generation_date": datetime.now().isoformat(),
        "results_summary": [],
    }

    for i, result in enumerate(all_results):
        summary["results_summary"].append(
            {
                "query_index": i + 1,
                "query": result["query"],
                "community_name": result["community_data"]["community_name"],
                "community_size": result["community_data"]["community_metrics"]["size"],
                "confidence_score": result["strategy"]["metadata"]["confidence_score"],
                "saved_files": result["saved_files"],
            }
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f" Summary report saved to: {filename}")


def load_knowledge_graph_data(filename=None):

    if filename is None:
        filename = get_path("models", "knowledge_graph_data")

    with open(filename, "rb") as f:
        graph_data = pickle.load(f)
    print(f" Knowledge graph loaded from: {filename}")
    return graph_data["hetero_data"], graph_data["topic_names"]


def load_networkx_graph(filename=None):

    if filename is None:
        filename = get_path("models", "networkx_graph")

    with open(filename, "rb") as f:
        G = pickle.load(f)
    print(f"🕸️ NetworkX graph loaded from: {filename}")
    return G
