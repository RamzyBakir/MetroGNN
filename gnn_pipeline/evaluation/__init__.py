"""MetroGNN evaluation utilities.

Heavy GNN/statistical modules are imported lazily so lightweight subpackages
such as ``gnn_pipeline.evaluation.agent`` do not load Torch on import.
"""

_GNN_EXPORTS = {
    "AblationResult",
    "ClassificationMetrics",
    "GNNEvaluator",
    "MultiTaskMetrics",
    "RegressionMetrics",
    "count_parameters",
    "generate_standard_ablation_configs",
    "measure_inference_time",
}

_STATISTICAL_EXPORTS = {
    "MultipleComparisonResult",
    "StatisticalTestResult",
    "StatisticalTester",
    "generate_statistical_report",
    "plot_confidence_intervals",
    "plot_significance_summary",
}

__all__ = [
    "AblationResult",
    "ClassificationMetrics",
    "GNNEvaluator",
    "MultiTaskMetrics",
    "RegressionMetrics",
    "MultipleComparisonResult",
    "StatisticalTestResult",
    "StatisticalTester",
    "count_parameters",
    "generate_standard_ablation_configs",
    "measure_inference_time",
    "generate_statistical_report",
    "plot_confidence_intervals",
    "plot_significance_summary",
]


def __getattr__(name):
    if name in _GNN_EXPORTS:
        from gnn_pipeline.evaluation import gnn_evaluation

        return getattr(gnn_evaluation, name)
    if name in _STATISTICAL_EXPORTS:
        from gnn_pipeline.evaluation import statistical_testing

        return getattr(statistical_testing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
