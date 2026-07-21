"""
MetroGNN Model Evaluation Script
=================================
Loads a trained model + knowledge graph from paths in config.yaml,
runs diagnostic evaluation across the three legacy auxiliary tasks
(importance classification, role prediction, influence regression), and generates
plots, reports, and a LaTeX table. Recommendation-system statistical testing lives
in ``scripts/evaluate_recommenders.py`` and uses persisted per-query outcomes.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --device cpu
    python scripts/evaluate_model.py --save-dir my_eval_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_pipeline.artifacts import validate_pipeline_manifest
from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.parsing import build_author_index
from gnn_pipeline.data.validation import ensure_paper_identity_columns
from gnn_pipeline.evaluation import GNNEvaluator
from gnn_pipeline.models.checkpoint import load_model_for_pipeline
from gnn_pipeline.models.training import set_external_data


def load_model_and_graph(device: str):
    model_path = get_path("models", "best_model_new")
    graph_path = get_path("models", "knowledge_graph")

    manifest_path = get_path("models", "pipeline_manifest")
    validate_pipeline_manifest(
        manifest_path,
        csv_path=get_path("data", "csv_path"),
        graph_path=graph_path,
    )
    print(f"Loading graph from {graph_path} ...")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    num_topics = graph["topic"].num_nodes
    print(f"Loading model from {model_path} ...")
    model, _ = load_model_for_pipeline(
        model_path, manifest_path, map_location=device
    )

    print(
        f"Graph: {graph['paper'].num_nodes} papers, "
        f"{graph['author'].num_nodes} authors, {num_topics} topics"
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, graph


def load_split_indices():
    splits_path = get_path("models", "splits_path")
    if os.path.isfile(splits_path):
        split_data = torch.load(splits_path, map_location="cpu", weights_only=False)
        print(f"Loaded split indices from {splits_path}")
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data.get("test_idx")
        if test_idx is None:
            print(
                "Legacy split file has no held-out test split; using validation "
                "indices for backward-compatible diagnostics."
            )
            test_idx = val_idx
        return train_idx, val_idx, test_idx
    else:
        raise FileNotFoundError(
            f"Split indices not found at {splits_path}. "
            "Run scripts/train_pipeline.py first to generate them."
        )


def setup_external_data():
    csv_path = get_path("data", "csv_path")
    topic_json_path = get_path("data", "topic_json_path")

    df = ensure_paper_identity_columns(pd.read_csv(csv_path))

    topic_info_dict = {}
    if topic_json_path and os.path.isfile(topic_json_path):
        with open(topic_json_path, "r") as f:
            topic_info_dict = json.load(f)
            topic_info_dict = {int(k): v for k, v in topic_info_dict.items()}
    else:
        from gnn_pipeline.data.topic_assignment import assign_topics_to_dblp

        n_clusters = get_value("training", "n_clusters")
        df, _, topic_info_dict = assign_topics_to_dblp(df, n_clusters=n_clusters)

    author_to_idx, author_records, paper_author_indices, _ = build_author_index(df)
    df["author_indices"] = paper_author_indices
    set_external_data(df, author_to_idx)
    return df, author_to_idx


def run_evaluation(model, graph, device, save_dir):
    evaluator = GNNEvaluator(device=device)
    results_dir = get_path("output", "results_dir") if save_dir is None else save_dir
    os.makedirs(results_dir, exist_ok=True)

    targets = evaluator._generate_targets_from_data(graph)

    _, val_idx, test_idx = load_split_indices()

    print(
        f"\nUsing validation split with {len(val_idx)} authors for checkpoint "
        f"selection context and held-out diagnostic test split with {len(test_idx)} authors."
    )

    print("\n" + "=" * 70)
    print("FULL GRAPH EVALUATION")
    print("=" * 70)
    full_metrics = evaluator.evaluate_model_on_data(model, graph, targets=targets)
    print(evaluator.generate_report(full_metrics))

    print("\n" + "=" * 70)
    print("HELD-OUT DIAGNOSTIC TEST SPLIT EVALUATION")
    print("=" * 70)
    test_metrics = evaluator.evaluate_model_on_data(
        model, graph, targets=targets, split_idx=test_idx
    )
    print(evaluator.generate_report(test_metrics))

    report_path = os.path.join(results_dir, "eval_report.txt")
    evaluator.generate_report(test_metrics, output_path=report_path)

    print("\nGenerating confusion matrices ...")
    evaluator.plot_confusion_matrices(
        test_metrics,
        save_path=os.path.join(results_dir, "confusion_matrices.png"),
    )
    plt.close("all")

    print("Generating regression analysis ...")
    model.eval()
    model_cpu = model.to("cpu")
    graph_cpu = graph.to("cpu")
    with torch.no_grad():
        x_dict, imp_logits, role_logits, inf_scores = model_cpu(
            graph_cpu.x_dict, graph_cpu.edge_index_dict
        )
    inf_true_all = targets["influence"][test_idx].cpu().numpy()
    inf_pred_all = inf_scores[test_idx].squeeze().cpu().numpy()
    evaluator.plot_regression_analysis(
        test_metrics.influence,
        inf_true_all,
        inf_pred_all,
        save_path=os.path.join(results_dir, "regression_analysis.png"),
    )
    plt.close("all")

    print("Generating t-SNE visualizations ...")
    author_emb = x_dict.get("author", x_dict.get("author_enhanced"))
    if author_emb is not None:
        emb_np = author_emb.cpu().numpy()
        imp_labels = targets["importance"].numpy()
        evaluator.plot_tsne_embeddings(
            emb_np,
            imp_labels,
            label_names=evaluator.IMPORTANCE_NAMES,
            title="t-SNE (colored by Importance)",
            save_path=os.path.join(results_dir, "tsne_importance.png"),
        )
        plt.close("all")

        role_labels = targets["role"].numpy()
        evaluator.plot_tsne_embeddings(
            emb_np,
            role_labels,
            label_names=evaluator.ROLE_NAMES,
            title="t-SNE (colored by Role)",
            save_path=os.path.join(results_dir, "tsne_role.png"),
        )
        plt.close("all")

    print("Analyzing attention weights ...")
    attention = evaluator.analyze_attention_weights(model, graph, num_samples=10)
    if "error" not in attention:
        print("\n  Attention Weight Analysis:")
        for nt, info in attention.get("interpretation", {}).items():
            print(f"    {nt}: {info['mean_weight']:.4f} +/- {info['std_weight']:.4f}")
            print(f"      -> {info['interpretation']}")
        print(f"  Dominant source: {attention.get('dominant_source', 'N/A')}")

    print("Evaluating embedding quality ...")
    if author_emb is not None:
        emb_quality_imp = evaluator.evaluate_embedding_quality(
            emb_np, targets["importance"].numpy()
        )
        emb_quality_role = evaluator.evaluate_embedding_quality(
            emb_np, targets["role"].numpy()
        )
        print(f"\n  Embedding quality (importance labels):")
        for k, v in emb_quality_imp.items():
            print(f"    {k}: {v:.4f}")
        print(f"  Embedding quality (role labels):")
        for k, v in emb_quality_role.items():
            print(f"    {k}: {v:.4f}")

    return full_metrics, test_metrics, evaluator, targets, test_idx


def generate_latex_table(test_metrics, save_dir):
    results_dir = get_path("output", "results_dir") if save_dir is None else save_dir
    os.makedirs(results_dir, exist_ok=True)

    evaluator = GNNEvaluator()
    results = {"MetroGNN": test_metrics}

    latex = evaluator.generate_latex_table(
        results,
        output_path=os.path.join(results_dir, "results_table.tex"),
    )
    print(f"\nLaTeX table saved to {os.path.join(results_dir, 'results_table.tex')}")


def main():
    parser = argparse.ArgumentParser(description="MetroGNN Model Evaluation")
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

    print("Setting up external data for target generation...")
    setup_external_data()

    full_metrics, test_metrics, evaluator, targets, test_idx = run_evaluation(
        model, graph, device, args.save_dir
    )

    print(
        "\nStatistical testing is intentionally omitted from this auxiliary "
        "heuristic-label evaluation. Use scripts/evaluate_recommenders.py "
        "for query-level confidence intervals and paired tests."
    )
    generate_latex_table(test_metrics, args.save_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    results_dir = (
        get_path("output", "results_dir") if args.save_dir is None else args.save_dir
    )
    print(f"All outputs saved to: {results_dir}/")
    print("  - eval_report.txt")
    print("  - confusion_matrices.png")
    print("  - regression_analysis.png")
    print("  - tsne_importance.png")
    print("  - tsne_role.png")
    print("  - results_table.tex")


if __name__ == "__main__":
    main()
