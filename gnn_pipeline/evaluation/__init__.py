"""
MetroGNN Evaluation Sub-package
================================
Re-exports key classes for convenience:

    from gnn_pipeline.evaluation import GNNEvaluator, MultiTaskMetrics
    from gnn_pipeline.evaluation import StatisticalTester, StatisticalTestResult
"""

from gnn_pipeline.evaluation.gnn_evaluation import (
    AblationResult,
    ClassificationMetrics,
    GNNEvaluator,
    MultiTaskMetrics,
    RegressionMetrics,
    count_parameters,
    generate_standard_ablation_configs,
    measure_inference_time,
)
from gnn_pipeline.evaluation.statistical_testing import (
    MultipleComparisonResult,
    StatisticalTestResult,
    StatisticalTester,
    generate_statistical_report,
    plot_confidence_intervals,
    plot_significance_summary,
)

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
