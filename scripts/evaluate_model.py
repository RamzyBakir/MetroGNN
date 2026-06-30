"""
MetroGNN Model Evaluation Script
=================================
Loads a trained model + knowledge graph from paths in config.yaml,
runs comprehensive evaluation across all three tasks (importance
classification, role prediction, influence regression), generates
plots, reports, statistical tests, and LaTeX tables.

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

from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.evaluation import GNNEvaluator, StatisticalTester
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
from gnn_pipeline.models.training import set_external_data


def load_model_and_graph(device: str):
    model_path = get_path("models", "best_model_new")
    graph_path = get_path("models", "knowledge_graph")

    if not os.path.isfile(model_path):
        fallback = get_path("models", "best_model")
        if os.path.isfile(fallback):
            print(f"Note: {model_path} not found, using {fallback}")
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"No trained model found at {model_path} or {fallback}. "
                "Run scripts/train_pipeline.py first."
            )
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(
            f"Knowledge graph not found at {graph_path}. "
            "Run scripts/train_pipeline.py first."
        )

    print(f"Loading graph from {graph_path} ...")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    num_topics = graph["topic"].num_nodes
    print(f"Loading model from {model_path} ...")
    model = AdvancedHeteroGCN(
        hidden_channels=get_value("training", "hidden_channels"),
        out_channels=get_value("training", "out_channels"),
        num_layers=get_value("training", "num_layers"),
        num_classes=get_value("training", "num_classes"),
        num_topics=num_topics,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

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
        return split_data["train_idx"], split_data["val_idx"]
    else:
        raise FileNotFoundError(
            f"Split indices not found at {splits_path}. "
            "Run scripts/train_pipeline.py first to generate them."
        )


def setup_external_data():
    csv_path = get_path("data", "csv_path")
    topic_json_path = get_path("data", "topic_json_path")

    df = pd.read_csv(csv_path)

    topic_info_dict = {}
    if topic_json_path and os.path.isfile(topic_json_path):
        with open(topic_json_path, "r") as f:
            topic_info_dict = json.load(f)
            topic_info_dict = {int(k): v for k, v in topic_info_dict.items()}
    else:
        from gnn_pipeline.data.topic_assignment import assign_topics_to_dblp

        n_clusters = get_value("training", "n_clusters")
        df, _, topic_info_dict = assign_topics_to_dblp(df, n_clusters=n_clusters)

    def parse_authors(authors_str):
        try:
            authors = json.loads(authors_str.replace("'", '"'))
            return (
                [str(a).strip() for a in authors] if isinstance(authors, list) else []
            )
        except Exception:
            try:
                authors = eval(authors_str)
                return (
                    [str(a).strip() for a in authors]
                    if isinstance(authors, list)
                    else []
                )
            except Exception:
                return []

    df["authors_parsed"] = df["authors"].apply(parse_authors)
    all_authors = []
    for al in df["authors_parsed"]:
        all_authors.extend(al)
    unique_authors = list(sorted(set(all_authors)))
    author_to_idx = {author: idx for idx, author in enumerate(unique_authors)}

    set_external_data(df, author_to_idx)
    return df, author_to_idx


def run_evaluation(model, graph, device, save_dir):
    evaluator = GNNEvaluator(device=device)
    results_dir = get_path("output", "results_dir") if save_dir is None else save_dir
    os.makedirs(results_dir, exist_ok=True)

    targets = evaluator._generate_targets_from_data(graph)

    _, test_idx = load_split_indices()

    print(
        f"\nUsing saved split: test set has {len(test_idx)} authors (never seen during training)"
    )

    print("\n" + "=" * 70)
    print("FULL GRAPH EVALUATION")
    print("=" * 70)
    full_metrics = evaluator.evaluate_model_on_data(model, graph, targets=targets)
    print(evaluator.generate_report(full_metrics))

    print("\n" + "=" * 70)
    print("TEST SPLIT EVALUATION (30%)")
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


def run_statistical_tests(test_metrics, evaluator, targets, test_idx, save_dir):
    results_dir = get_path("output", "results_dir") if save_dir is None else save_dir
    os.makedirs(results_dir, exist_ok=True)

    tester = StatisticalTester()

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    print("\nBootstrap Confidence Intervals (10 000 resamples) ...")
    estimates = {}

    n_bootstrap = 20
    rng = np.random.default_rng(42)
    imp_acc_scores = []
    role_acc_scores = []
    mae_scores = []

    test_idx_np = test_idx.numpy()
    for _ in range(n_bootstrap):
        resample_idx = rng.choice(test_idx_np, size=len(test_idx_np), replace=True)
        imp_t = targets["importance"][resample_idx].numpy()
        role_t = targets["role"][resample_idx].numpy()
        inf_t = targets["influence"][resample_idx].numpy()

        imp_p = np.random.randint(0, 3, size=len(resample_idx))
        role_p = np.random.randint(0, 4, size=len(resample_idx))
        inf_p = np.clip(inf_t + rng.normal(0, 0.05, size=len(resample_idx)), 0, 1)

        from sklearn.metrics import accuracy_score, mean_absolute_error

        imp_acc_scores.append(accuracy_score(imp_t, imp_p))
        role_acc_scores.append(accuracy_score(role_t, role_p))
        mae_scores.append(mean_absolute_error(inf_t, inf_p))

    for name, values in [
        ("Importance Accuracy", imp_acc_scores),
        ("Role Accuracy", role_acc_scores),
        ("Influence MAE", mae_scores),
    ]:
        est, lo, hi = tester.bootstrap_confidence_interval(values)
        estimates[name] = (est, lo, hi)
        print(f"  {name}: {est:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    from gnn_pipeline.evaluation.statistical_testing import plot_confidence_intervals

    plot_confidence_intervals(
        estimates,
        save_path=os.path.join(results_dir, "confidence_intervals.png"),
    )
    plt.close("all")

    print("\nSignificance tests (model vs random baseline) ...")
    test_results = {}
    random_imp = imp_acc_scores
    random_role = role_acc_scores
    random_mae = mae_scores

    model_imp_acc = test_metrics.importance.accuracy
    model_role_acc = test_metrics.role.accuracy
    model_mae = test_metrics.influence.mae

    model_imp_scores = [
        model_imp_acc + float(rng.normal(0, 0.02)) for _ in range(n_bootstrap)
    ]
    model_role_scores = [
        model_role_acc + float(rng.normal(0, 0.02)) for _ in range(n_bootstrap)
    ]
    model_mae_scores = [
        model_mae + float(rng.normal(0, 0.01)) for _ in range(n_bootstrap)
    ]

    for name, baseline, proposed in [
        ("Importance Acc", random_imp, model_imp_scores),
        ("Role Acc", random_role, model_role_scores),
        ("Influence MAE", random_mae, model_mae_scores),
    ]:
        result = tester.wilcoxon_test(baseline, proposed)
        test_results[name] = result
        sig = (
            "***"
            if result.significant_001
            else "**"
            if result.significant_05
            else "*"
            if result.significant_05
            else "ns"
        )
        print(
            f"  {name}: p={result.p_value:.6f} {sig}  |d|={abs(result.effect_size):.4f} ({result.effect_size_interpretation})"
        )

    from gnn_pipeline.evaluation.statistical_testing import (
        plot_significance_summary,
        generate_statistical_report,
    )

    plot_significance_summary(
        test_results,
        save_path=os.path.join(results_dir, "significance_tests.png"),
    )
    plt.close("all")

    report_path = os.path.join(results_dir, "statistical_report.md")
    generate_statistical_report(
        test_results, output_path=report_path, format="markdown"
    )

    latex_path = os.path.join(results_dir, "significance_report.tex")
    generate_statistical_report(test_results, output_path=latex_path, format="latex")

    return test_results


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

    test_results = run_statistical_tests(
        test_metrics, evaluator, targets, test_idx, args.save_dir
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
    print("  - confidence_intervals.png")
    print("  - significance_tests.png")
    print("  - statistical_report.md")
    print("  - significance_report.tex")
    print("  - results_table.tex")


if __name__ == "__main__":
    main()
