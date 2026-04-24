"""
================================================================================
MetroGNN GNN Model Evaluation Module
================================================================================
Comprehensive evaluation for the heterogeneous multi-task GNN including:
- Classification metrics (Accuracy, Macro-F1, Weighted-F1, per-class F1)
- Regression metrics (MAE, RMSE, Pearson/Spearman correlation, R^2)
- Multi-task composite scoring (MTS)
- Ablation study framework
- Attention weight analysis and visualization
- Statistical significance testing
- Confusion matrix generation
- Embedding quality evaluation (silhouette, trustworthiness)
================================================================================
"""

import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import HeteroData


# ==============================================================================
# Data Structures for Evaluation Results
# ==============================================================================


@dataclass
class ClassificationMetrics:
    """Container for classification task evaluation results."""

    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: Dict[int, float] = field(default_factory=dict)
    per_class_precision: Dict[int, float] = field(default_factory=dict)
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_distribution: Dict[int, int] = field(default_factory=dict)


@dataclass
class RegressionMetrics:
    """Container for regression task evaluation results."""

    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    pearson_r: float = 0.0
    pearson_p: float = 0.0
    spearman_r: float = 0.0
    spearman_p: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error


@dataclass
class MultiTaskMetrics:
    """Container for combined multi-task evaluation results."""

    importance: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    role: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    influence: RegressionMetrics = field(default_factory=RegressionMetrics)
    mts: float = 0.0  # Multi-Task Score
    combined_loss: float = 0.0
    task_losses: Dict[str, float] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Container for a single ablation experiment result."""

    variant_name: str = ""
    metrics: MultiTaskMetrics = field(default_factory=MultiTaskMetrics)
    delta_mts: float = 0.0
    parameter_count: int = 0
    inference_time_ms: float = 0.0


# ==============================================================================
# Core Evaluation Functions
# ==============================================================================


class GNNEvaluator:
    """
    Comprehensive evaluator for the MetroGNN multi-task heterogeneous GNN.

    Provides evaluation across three tasks:
    1. Researcher importance classification (3 classes: Low, Medium, High)
    2. Researcher role prediction (4 classes: Leader, Core, Contributor, Sole Expert)
    3. Influence score regression (continuous [0,1])
    """

    IMPORTANCE_NAMES = {0: "Low", 1: "Medium", 2: "High"}
    ROLE_NAMES = {0: "Leader", 1: "Core", 2: "Contributor", 3: "Sole Expert"}

    def __init__(self, device: str = "auto", random_seed: int = 42):
        """
        Initialize the evaluator.

        Args:
            device: Computation device ('cuda', 'cpu', or 'auto')
            random_seed: Random seed for reproducibility
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.results_history: List[Dict] = []

    # --------------------------------------------------------------------------
    # Classification Evaluation
    # --------------------------------------------------------------------------

    def evaluate_classification(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> ClassificationMetrics:
        """
        Comprehensive classification evaluation.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for confidence analysis)
            class_names: Mapping from label indices to class names

        Returns:
            ClassificationMetrics dataclass with all computed metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        metrics = ClassificationMetrics()

        # Basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics.weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        labels = sorted(np.unique(y_true))
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        per_class_precision, per_class_recall, _, _ = precision_recall_per_class(
            y_true, y_pred, labels
        )

        for i, label in enumerate(labels):
            name = class_names.get(label, str(label)) if class_names else str(label)
            metrics.per_class_f1[name] = per_class_f1[i]
            metrics.per_class_precision[name] = per_class_precision[i]
            metrics.per_class_recall[name] = per_class_recall[i]

        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        for u, c in zip(unique, counts):
            name = class_names.get(u, str(u)) if class_names else str(u)
            metrics.class_distribution[name] = c

        return metrics

    def evaluate_importance(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> ClassificationMetrics:
        """Evaluate importance classification task."""
        return self.evaluate_classification(
            y_true, y_pred, y_prob, class_names=self.IMPORTANCE_NAMES
        )

    def evaluate_role(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> ClassificationMetrics:
        """Evaluate role prediction task."""
        return self.evaluate_classification(
            y_true, y_pred, y_prob, class_names=self.ROLE_NAMES
        )

    # --------------------------------------------------------------------------
    # Regression Evaluation
    # --------------------------------------------------------------------------

    def evaluate_regression(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> RegressionMetrics:
        """
        Comprehensive regression evaluation for influence scores.

        Args:
            y_true: Ground truth influence scores [0, 1]
            y_pred: Predicted influence scores

        Returns:
            RegressionMetrics dataclass with all computed metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy().flatten()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy().flatten()

        metrics = RegressionMetrics()

        # Basic metrics
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics.r2 = r2_score(y_true, y_pred)

        # Correlation metrics
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
            spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
            metrics.pearson_r = pearson_r
            metrics.pearson_p = pearson_p
            metrics.spearman_r = spearman_r
            metrics.spearman_p = spearman_p

        # MAPE (with protection against zero division)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.01, 1.0))) * 100
            metrics.mape = mape if np.isfinite(mape) else 0.0

        return metrics

    # --------------------------------------------------------------------------
    # Multi-Task Combined Evaluation
    # --------------------------------------------------------------------------

    def evaluate_multi_task(
        self,
        importance_true: Union[np.ndarray, torch.Tensor],
        importance_pred: Union[np.ndarray, torch.Tensor],
        role_true: Union[np.ndarray, torch.Tensor],
        role_pred: Union[np.ndarray, torch.Tensor],
        influence_true: Union[np.ndarray, torch.Tensor],
        influence_pred: Union[np.ndarray, torch.Tensor],
        importance_logits: Optional[torch.Tensor] = None,
        role_logits: Optional[torch.Tensor] = None,
        influence_scores: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> MultiTaskMetrics:
        """
        Evaluate all three tasks simultaneously and compute the Multi-Task Score.

        Args:
            importance_true, importance_pred: Importance classification labels and predictions
            role_true, role_pred: Role classification labels and predictions
            influence_true, influence_pred: Influence regression targets and predictions
            importance_logits, role_logits, influence_scores: Raw model outputs for loss computation
            loss_weights: Dictionary of task loss weights

        Returns:
            MultiTaskMetrics dataclass with all task results and combined score
        """
        metrics = MultiTaskMetrics()

        # Evaluate individual tasks
        metrics.importance = self.evaluate_importance(importance_true, importance_pred)
        metrics.role = self.evaluate_role(role_true, role_pred)
        metrics.influence = self.evaluate_regression(influence_true, influence_pred)

        # Compute Multi-Task Score (MTS)
        # MTS = (Macro-F1_imp + Macro-F1_role + (1 - MAE_inf)) / 3
        mts = (
            metrics.importance.macro_f1
            + metrics.role.macro_f1
            + (1.0 - metrics.influence.mae)
        ) / 3.0
        metrics.mts = max(0.0, mts)  # Clip to non-negative

        # Compute individual task losses if logits provided
        if (
            importance_logits is not None
            and role_logits is not None
            and influence_scores is not None
        ):
            metrics.task_losses = self._compute_task_losses(
                importance_logits,
                role_logits,
                influence_scores,
                importance_true,
                role_true,
                influence_true,
                loss_weights,
            )
            metrics.combined_loss = sum(
                w * metrics.task_losses[k] for k, w in (loss_weights or {}).items()
            )

        return metrics

    def _compute_task_losses(
        self,
        imp_logits: torch.Tensor,
        role_logits: torch.Tensor,
        inf_scores: torch.Tensor,
        imp_true: Union[np.ndarray, torch.Tensor],
        role_true: Union[np.ndarray, torch.Tensor],
        inf_true: Union[np.ndarray, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute individual task losses for monitoring."""
        weights = loss_weights or {"importance": 0.5, "role": 0.3, "influence": 0.2}

        if isinstance(imp_true, np.ndarray):
            imp_true = torch.tensor(
                imp_true, dtype=torch.long, device=imp_logits.device
            )
        if isinstance(role_true, np.ndarray):
            role_true = torch.tensor(
                role_true, dtype=torch.long, device=role_logits.device
            )
        if isinstance(inf_true, np.ndarray):
            inf_true = torch.tensor(
                inf_true, dtype=torch.float32, device=inf_scores.device
            )

        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        losses = {}
        losses["importance"] = ce_loss(imp_logits, imp_true).item()
        losses["role"] = ce_loss(role_logits, role_true).item()
        losses["influence"] = mse_loss(inf_scores.squeeze(), inf_true).item()

        return losses

    # --------------------------------------------------------------------------
    # Model Evaluation on HeteroData
    # --------------------------------------------------------------------------

    def evaluate_model_on_data(
        self,
        model: nn.Module,
        data: HeteroData,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        split_idx: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> MultiTaskMetrics:
        """
        Run complete model evaluation on a HeteroData graph.

        Args:
            model: The GNN model to evaluate
            data: PyTorch Geometric HeteroData object
            targets: Pre-computed targets dictionary with 'importance', 'role', 'influence' keys.
                     If None, targets are generated from data features.
            split_idx: Indices to evaluate on (e.g., validation or test set).
                       If None, evaluates on all nodes.
            loss_weights: Task loss weights

        Returns:
            MultiTaskMetrics with all evaluation results
        """
        model.eval()
        model = model.to(self.device)
        data = data.to(self.device)

        # Generate targets if not provided
        if targets is None:
            targets = self._generate_targets_from_data(data)

        with torch.no_grad():
            # Forward pass
            try:
                x_dict, imp_logits, role_logits, inf_scores = model(
                    data.x_dict, data.edge_index_dict
                )
            except Exception as e:
                print(f"Warning: Model forward pass failed: {e}")
                return MultiTaskMetrics()

            # Apply split if provided
            if split_idx is not None:
                split_idx = (
                    split_idx.to(self.device)
                    if isinstance(split_idx, torch.Tensor)
                    else split_idx
                )
                imp_logits = imp_logits[split_idx]
                role_logits = role_logits[split_idx]
                inf_scores = inf_scores[split_idx]
                imp_true = targets["importance"][split_idx]
                role_true = targets["role"][split_idx]
                inf_true = targets["influence"][split_idx]
            else:
                imp_true = targets["importance"]
                role_true = targets["role"]
                inf_true = targets["influence"]

            # Get predictions
            imp_pred = imp_logits.argmax(dim=1)
            role_pred = role_logits.argmax(dim=1)
            inf_pred = inf_scores.squeeze()

            # Evaluate
            metrics = self.evaluate_multi_task(
                imp_true,
                imp_pred,
                role_true,
                role_pred,
                inf_true,
                inf_pred,
                imp_logits,
                role_logits,
                inf_scores,
                loss_weights,
            )

        return metrics

    def _generate_targets_from_data(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Generate training targets from graph structural features."""
        from gnn_pipeline.models.training import create_enhanced_training_targets

        importance, role, influence = create_enhanced_training_targets(data)
        return {
            "importance": importance,
            "role": role,
            "influence": influence,
        }

    # --------------------------------------------------------------------------
    # Ablation Study Framework
    # --------------------------------------------------------------------------

    def run_ablation_study(
        self,
        base_model_fn: Callable[[], nn.Module],
        data: HeteroData,
        targets: Dict[str, torch.Tensor],
        split_idx: torch.Tensor,
        ablation_configs: List[Dict[str, Any]],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> List[AblationResult]:
        """
        Run systematic ablation study by modifying model configurations.

        Args:
            base_model_fn: Factory function that creates the base model
            data: Evaluation data
            targets: Target labels
            split_idx: Evaluation split indices
            ablation_configs: List of configuration dicts, each with:
                - 'name': Variant name
                - 'modifications': Dict of parameter changes (e.g., {'num_layers': 2})
            loss_weights: Task loss weights

        Returns:
            List of AblationResult with metrics for each variant
        """
        # First evaluate full model as baseline
        print("=" * 70)
        print("ABLATION STUDY")
        print("=" * 70)

        base_model = base_model_fn()
        base_metrics = self.evaluate_model_on_data(
            base_model, data, targets, split_idx, loss_weights
        )
        base_mts = base_metrics.mts

        base_result = AblationResult(
            variant_name="Full MetroGNN",
            metrics=base_metrics,
            delta_mts=0.0,
            parameter_count=count_parameters(base_model),
            inference_time_ms=measure_inference_time(base_model, data, self.device),
        )
        results = [base_result]

        print(f"\nBaseline MTS: {base_mts:.4f}")
        print("-" * 70)

        # Evaluate each ablation variant
        for config in ablation_configs:
            name = config["name"]
            print(f"\nEvaluating: {name}...")

            try:
                # Create modified model
                modified_model = base_model_fn(**config.get("modifications", {}))
                modified_metrics = self.evaluate_model_on_data(
                    modified_model, data, targets, split_idx, loss_weights
                )

                result = AblationResult(
                    variant_name=name,
                    metrics=modified_metrics,
                    delta_mts=modified_metrics.mts - base_mts,
                    parameter_count=count_parameters(modified_model),
                    inference_time_ms=measure_inference_time(
                        modified_model, data, self.device
                    ),
                )
                results.append(result)

                print(f"  MTS: {modified_metrics.mts:.4f} ({result.delta_mts:+.4f})")
                print(f"  Params: {result.parameter_count:,}")
                print(f"  Inference: {result.inference_time_ms:.2f}ms")

            except Exception as e:
                print(f"  FAILED: {e}")
                results.append(AblationResult(variant_name=name + " (FAILED)"))

        return results

    # --------------------------------------------------------------------------
    # Attention Analysis
    # --------------------------------------------------------------------------

    def analyze_attention_weights(
        self,
        model: nn.Module,
        data: HeteroData,
        num_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze learned cross-type attention weights.

        Args:
            model: Trained GNN model with attention_weights parameter
            data: HeteroData graph
            num_samples: Number of forward passes for statistics

        Returns:
            Dictionary with attention statistics and analysis
        """
        model.eval()
        model = model.to(self.device)
        data = data.to(self.device)

        attention_weights_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                try:
                    _, imp_logits, role_logits, inf_scores = model(
                        data.x_dict, data.edge_index_dict
                    )
                    if hasattr(model, "attention_weights"):
                        w = F.softmax(model.attention_weights, dim=0)
                        attention_weights_list.append(w.cpu().numpy())
                except Exception as e:
                    continue

        if not attention_weights_list:
            return {"error": "No attention weights collected"}

        attention_array = np.stack(attention_weights_list)
        mean_weights = attention_array.mean(axis=0)
        std_weights = attention_array.std(axis=0)

        # Interpretation
        node_types = ["author", "paper", "topic"]
        interpretation = {}
        for i, nt in enumerate(node_types[: len(mean_weights)]):
            interpretation[nt] = {
                "mean_weight": float(mean_weights[i]),
                "std_weight": float(std_weights[i]),
                "interpretation": self._interpret_attention_weight(mean_weights[i], nt),
            }

        return {
            "mean_weights": mean_weights.tolist(),
            "std_weights": std_weights.tolist(),
            "node_types": node_types[: len(mean_weights)],
            "interpretation": interpretation,
            "dominant_source": node_types[int(np.argmax(mean_weights))],
        }

    def _interpret_attention_weight(self, weight: float, node_type: str) -> str:
        """Generate human-readable interpretation of attention weight."""
        if weight > 0.4:
            return f"{node_type} features are the dominant information source"
        elif weight > 0.3:
            return f"{node_type} features provide significant complementary signal"
        elif weight > 0.2:
            return f"{node_type} features contribute moderate contextual information"
        else:
            return f"{node_type} features have minor influence on final representations"

    # --------------------------------------------------------------------------
    # Embedding Quality Evaluation
    # --------------------------------------------------------------------------

    def evaluate_embedding_quality(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metric: str = "cosine",
    ) -> Dict[str, float]:
        """
        Evaluate quality of learned embeddings using geometric metrics.

        Args:
            embeddings: Learned node embeddings (N, D)
            labels: Node labels for supervised quality assessment
            metric: Distance metric for silhouette score

        Returns:
            Dictionary of embedding quality metrics
        """
        quality = {}

        # Silhouette score (higher = better separated clusters)
        if len(np.unique(labels)) > 1:
            quality["silhouette_score"] = silhouette_score(
                embeddings, labels, metric=metric
            )

        # Intra-class compactness and inter-class separation
        unique_labels = np.unique(labels)
        intra_distances = []
        inter_distances = []

        _MAX_SUBSAMPLE = 5000

        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            if len(class_embeddings) > _MAX_SUBSAMPLE:
                class_embeddings = class_embeddings[
                    np.random.choice(
                        len(class_embeddings), _MAX_SUBSAMPLE, replace=False
                    )
                ]
            if len(class_embeddings) > 1:
                intra_distances.append(np.mean(pdist(class_embeddings, metric=metric)))

            other_embeddings = embeddings[labels != label]
            if len(other_embeddings) > _MAX_SUBSAMPLE:
                other_embeddings = other_embeddings[
                    np.random.choice(
                        len(other_embeddings), _MAX_SUBSAMPLE, replace=False
                    )
                ]
            if len(other_embeddings) > 0 and len(class_embeddings) > 0:
                inter_distances.append(
                    float(
                        np.mean(
                            cdist(class_embeddings, other_embeddings, metric=metric)
                        )
                    )
                )

        quality["mean_intra_distance"] = (
            np.mean(intra_distances) if intra_distances else 0.0
        )
        quality["mean_inter_distance"] = (
            np.mean(inter_distances) if inter_distances else 0.0
        )
        quality["separation_ratio"] = quality["mean_inter_distance"] / max(
            quality["mean_intra_distance"], 1e-8
        )

        # Embedding norm statistics
        norms = np.linalg.norm(embeddings, axis=1)
        quality["mean_norm"] = float(np.mean(norms))
        quality["norm_std"] = float(np.std(norms))

        return quality

    # --------------------------------------------------------------------------
    # Statistical Significance Testing
    # --------------------------------------------------------------------------

    def statistical_significance_test(
        self,
        baseline_metrics: List[float],
        proposed_metrics: List[float],
        test: str = "paired_ttest",
    ) -> Dict[str, float]:
        """
        Perform statistical significance testing between two methods.

        Args:
            baseline_metrics: List of metric values for baseline
            proposed_metrics: List of metric values for proposed method
            test: Type of statistical test ('paired_ttest', 'wilcoxon')

        Returns:
            Dictionary with test statistics and p-values
        """
        baseline = np.array(baseline_metrics)
        proposed = np.array(proposed_metrics)

        results = {}

        if test == "paired_ttest":
            stat, p_value = stats.ttest_rel(proposed, baseline)
            results["test"] = "Paired t-test"
            results["statistic"] = float(stat)
            results["p_value"] = float(p_value)
            results["significant_at_05"] = p_value < 0.05
            results["significant_at_01"] = p_value < 0.01

        elif test == "wilcoxon":
            stat, p_value = stats.wilcoxon(proposed, baseline)
            results["test"] = "Wilcoxon signed-rank"
            results["statistic"] = float(stat)
            results["p_value"] = float(p_value)
            results["significant_at_05"] = p_value < 0.05
            results["significant_at_01"] = p_value < 0.01

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline) ** 2 + np.std(proposed) ** 2) / 2)
        cohens_d = (np.mean(proposed) - np.mean(baseline)) / max(pooled_std, 1e-8)
        results["cohens_d"] = float(cohens_d)
        results["effect_size"] = self._interpret_cohens_d(cohens_d)

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------

    def plot_confusion_matrices(
        self,
        metrics: MultiTaskMetrics,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """Plot confusion matrices for both classification tasks."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Importance confusion matrix
        if metrics.importance.confusion_matrix is not None:
            sns.heatmap(
                metrics.importance.confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[
                    self.IMPORTANCE_NAMES.get(i, str(i))
                    for i in range(metrics.importance.confusion_matrix.shape[0])
                ],
                yticklabels=[
                    self.IMPORTANCE_NAMES.get(i, str(i))
                    for i in range(metrics.importance.confusion_matrix.shape[0])
                ],
                ax=axes[0],
            )
            axes[0].set_title(
                f"Importance Classification\nAcc: {metrics.importance.accuracy:.3f}, "
                f"Macro-F1: {metrics.importance.macro_f1:.3f}"
            )
            axes[0].set_ylabel("True")
            axes[0].set_xlabel("Predicted")

        # Role confusion matrix
        if metrics.role.confusion_matrix is not None:
            sns.heatmap(
                metrics.role.confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=[
                    self.ROLE_NAMES.get(i, str(i))
                    for i in range(metrics.role.confusion_matrix.shape[0])
                ],
                yticklabels=[
                    self.ROLE_NAMES.get(i, str(i))
                    for i in range(metrics.role.confusion_matrix.shape[0])
                ],
                ax=axes[1],
            )
            axes[1].set_title(
                f"Role Prediction\nAcc: {metrics.role.accuracy:.3f}, "
                f"Macro-F1: {metrics.role.macro_f1:.3f}"
            )
            axes[1].set_ylabel("True")
            axes[1].set_xlabel("Predicted")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrices saved to {save_path}")

        return fig

    def plot_ablation_results(
        self,
        ablation_results: List[AblationResult],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """Plot ablation study results."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        names = [r.variant_name for r in ablation_results]
        mts_values = [r.metrics.mts for r in ablation_results]
        delta_values = [r.delta_mts for r in ablation_results]

        # MTS comparison
        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in delta_values]
        axes[0].barh(names, mts_values, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("Multi-Task Score (MTS)")
        axes[0].set_title("Ablation Study: MTS Comparison")
        axes[0].axvline(
            mts_values[0], color="blue", linestyle="--", alpha=0.5, label="Baseline"
        )
        axes[0].legend()

        # Delta from baseline
        axes[1].barh(
            names, delta_values, color=colors, edgecolor="black", linewidth=0.5
        )
        axes[1].set_xlabel("$\\Delta$ MTS from Baseline")
        axes[1].set_title("Performance Change")
        axes[1].axvline(0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Ablation plot saved to {save_path}")

        return fig

    def plot_regression_analysis(
        self,
        metrics: RegressionMetrics,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """Plot regression analysis visualizations."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy().flatten()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy().flatten()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Scatter plot: predicted vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")
        axes[0].plot([0, 1], [0, 1], "r--", label="Perfect prediction")
        axes[0].set_xlabel("Actual Influence Score")
        axes[0].set_ylabel("Predicted Influence Score")
        axes[0].set_title(
            f"Influence Regression\nMAE: {metrics.mae:.4f}, RMSE: {metrics.rmse:.4f}, "
            f"$r$: {metrics.pearson_r:.4f}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        residuals = y_true - y_pred
        axes[1].hist(
            residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue"
        )
        axes[1].axvline(0, color="red", linestyle="--", label="Zero residual")
        axes[1].set_xlabel("Residual (Actual - Predicted)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(
            f"Residual Distribution\nMean: {np.mean(residuals):.4f}, "
            f"Std: {np.std(residuals):.4f}"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Regression plots saved to {save_path}")

        return fig

    def plot_tsne_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[Dict[int, str]] = None,
        title: str = "t-SNE Visualization",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot t-SNE visualization of embeddings colored by labels."""
        tsne = TSNE(
            n_components=2,
            random_state=self.random_seed,
            perplexity=min(30, len(embeddings) - 1),
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=figsize)

        unique_labels = sorted(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names.get(label, str(label)) if label_names else str(label)
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=name,
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_title(title)
        ax.legend(loc="best", title="Class")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"t-SNE plot saved to {save_path}")

        return fig

    # --------------------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------------------

    def generate_report(
        self,
        metrics: MultiTaskMetrics,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted evaluation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("METROGNN GNN EVALUATION REPORT")
        lines.append("=" * 70)

        # Importance Classification
        lines.append("\n--- Importance Classification ---")
        lines.append(f"  Accuracy:     {metrics.importance.accuracy:.4f}")
        lines.append(f"  Macro-F1:     {metrics.importance.macro_f1:.4f}")
        lines.append(f"  Weighted-F1:  {metrics.importance.weighted_f1:.4f}")
        lines.append("  Per-Class F1:")
        for cls, f1 in metrics.importance.per_class_f1.items():
            lines.append(f"    {cls}: {f1:.4f}")

        # Role Prediction
        lines.append("\n--- Role Prediction ---")
        lines.append(f"  Accuracy:     {metrics.role.accuracy:.4f}")
        lines.append(f"  Macro-F1:     {metrics.role.macro_f1:.4f}")
        lines.append(f"  Weighted-F1:  {metrics.role.weighted_f1:.4f}")
        lines.append("  Per-Class F1:")
        for cls, f1 in metrics.role.per_class_f1.items():
            lines.append(f"    {cls}: {f1:.4f}")

        # Influence Regression
        lines.append("\n--- Influence Regression ---")
        lines.append(f"  MAE:          {metrics.influence.mae:.4f}")
        lines.append(f"  RMSE:         {metrics.influence.rmse:.4f}")
        lines.append(f"  R^2:          {metrics.influence.r2:.4f}")
        lines.append(
            f"  Pearson r:    {metrics.influence.pearson_r:.4f} (p={metrics.influence.pearson_p:.4f})"
        )
        lines.append(
            f"  Spearman rho: {metrics.influence.spearman_r:.4f} (p={metrics.influence.spearman_p:.4f})"
        )

        # Multi-Task Score
        lines.append("\n--- Multi-Task Score ---")
        lines.append(f"  MTS:          {metrics.mts:.4f}")

        # Task losses
        if metrics.task_losses:
            lines.append("\n--- Task Losses ---")
            for task, loss in metrics.task_losses.items():
                lines.append(f"  {task}: {loss:.4f}")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")

        return report

    def generate_latex_table(
        self,
        results: Dict[str, MultiTaskMetrics],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate LaTeX table comparing multiple methods."""
        latex_lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Performance comparison of MetroGNN against baseline methods.}",
            "\\label{tab:gnn-results}",
            "\\begin{tabular}{@{}l|cc|cc|cc|c@{}}",
            "\\toprule",
            "& \\multicolumn{2}{c|}{\\textbf{Importance (3-cls)}} & "
            "\\multicolumn{2}{c|}{\\textbf{Role (4-cls)}} & "
            "\\multicolumn{2}{c|}{\\textbf{Influence (reg)}} & \\",
            "\\cmidrule{2-7}",
            "\\textbf{Method} & Acc $\\uparrow$ & Macro-F1 $\\uparrow$ & "
            "Acc $\\uparrow$ & Macro-F1 $\\uparrow$ & MAE $\\downarrow$ & $r$ $\\uparrow$ & "
            "\\textbf{MTS} $\\uparrow$ \\",
            "\\midrule",
        ]

        for method_name, metrics in results.items():
            imp = metrics.importance
            role = metrics.role
            inf = metrics.influence
            latex_lines.append(
                f"{method_name} & {imp.accuracy:.4f} & {imp.macro_f1:.4f} & "
                f"{role.accuracy:.4f} & {role.macro_f1:.4f} & "
                f"{inf.mae:.4f} & {inf.pearson_r:.4f} & {metrics.mts:.4f} \\"
            )

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table*}",
                "",
            ]
        )

        latex = "\n".join(latex_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(latex)
            print(f"LaTeX table saved to {output_path}")

        return latex


# ==============================================================================
# Helper Functions
# ==============================================================================


def precision_recall_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute per-class precision and recall."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return precision.tolist(), recall.tolist(), f1.tolist(), support.tolist()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    num_runs: int = 50,
    warmup: int = 10,
) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    model = model.to(device)
    data = data.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = model(data.x_dict, data.edge_index_dict)
            except Exception:
                pass

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            try:
                _ = model(data.x_dict, data.edge_index_dict)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    return float(np.median(times))


def generate_standard_ablation_configs() -> List[Dict[str, Any]]:
    """Generate standard ablation configurations for MetroGNN."""
    return [
        {"name": "No Cross-Type Attention", "modifications": {"use_attention": False}},
        {"name": "No Residual Connections", "modifications": {"use_residual": False}},
        {"name": "2 Layers", "modifications": {"num_layers": 2}},
        {"name": "8 Layers", "modifications": {"num_layers": 8}},
        {"name": "Hidden Dim 32", "modifications": {"hidden_channels": 32}},
        {"name": "Hidden Dim 128", "modifications": {"hidden_channels": 128}},
    ]


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MetroGNN GNN Evaluation Module")
    print("=" * 70)
    print("\nThis module provides comprehensive evaluation capabilities:")
    print("  - Classification metrics (Accuracy, Macro-F1, per-class F1)")
    print("  - Regression metrics (MAE, RMSE, Pearson/Spearman correlation)")
    print("  - Multi-Task Score (MTS) computation")
    print("  - Ablation study framework")
    print("  - Attention weight analysis")
    print("  - Embedding quality evaluation")
    print("  - Statistical significance testing")
    print("  - Visualization (confusion matrices, t-SNE, regression plots)")
    print("  - LaTeX table generation")
    print("\nUsage:")
    print("  from gnn_evaluation import GNNEvaluator, MultiTaskMetrics")
    print("  evaluator = GNNEvaluator(device='cuda')")
    print("  metrics = evaluator.evaluate_model_on_data(model, data, targets)")
    print("  evaluator.generate_report(metrics)")
