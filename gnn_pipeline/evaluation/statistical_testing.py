"""
================================================================================
MetroGNN Statistical Testing and Reporting Utilities
================================================================================
Provides statistical significance testing, confidence interval computation,
and comprehensive reporting utilities for both GNN and multi-agent evaluations.

Includes:
- Paired/unpaired t-tests
- Wilcoxon signed-rank tests
- McNemar's test for classification
- Bootstrap confidence intervals
- Bonferroni correction for multiple comparisons
- Effect size computation (Cohen's d, Cliff's delta)
- Comprehensive LaTeX and Markdown report generation
================================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""

    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 1.0
    significant_05: bool = False
    significant_01: bool = False
    significant_001: bool = False
    effect_size: float = 0.0
    effect_size_interpretation: str = ""
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    baseline_mean: float = 0.0
    proposed_mean: float = 0.0
    baseline_std: float = 0.0
    proposed_std: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipleComparisonResult:
    """Results from multiple comparison correction."""

    original_p_values: List[float] = field(default_factory=list)
    corrected_p_values: List[float] = field(default_factory=list)
    rejected_hypotheses: List[bool] = field(default_factory=list)
    correction_method: str = ""
    alpha: float = 0.05


# ==============================================================================
# Core Statistical Tests
# ==============================================================================


class StatisticalTester:
    """
    Comprehensive statistical testing framework for evaluating
    MetroGNN against baseline methods.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize statistical tester.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def paired_ttest(
        self,
        baseline_values: Union[List[float], np.ndarray],
        proposed_values: Union[List[float], np.ndarray],
        confidence_level: float = 0.95,
    ) -> StatisticalTestResult:
        """
        Paired t-test for comparing two methods on matched samples.

        Args:
            baseline_values: Metric values from baseline method
            proposed_values: Metric values from proposed method
            confidence_level: Confidence level for interval

        Returns:
            StatisticalTestResult with all test information
        """
        baseline = np.array(baseline_values)
        proposed = np.array(proposed_values)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(proposed, baseline)

        # Confidence interval for mean difference
        diff = proposed - baseline
        se = stats.sem(diff)
        df = len(diff) - 1
        t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
        ci_lower = np.mean(diff) - t_crit * se
        ci_upper = np.mean(diff) + t_crit * se

        # Effect size (Cohen's d for paired)
        cohens_d = np.mean(diff) / max(np.std(diff, ddof=1), 1e-8)

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            significant_05=p_value < 0.05,
            significant_01=p_value < 0.01,
            significant_001=p_value < 0.001,
            effect_size=float(cohens_d),
            effect_size_interpretation=self._interpret_cohens_d(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            baseline_mean=float(np.mean(baseline)),
            proposed_mean=float(np.mean(proposed)),
            baseline_std=float(np.std(baseline)),
            proposed_std=float(np.std(proposed)),
        )

    def wilcoxon_test(
        self,
        baseline_values: Union[List[float], np.ndarray],
        proposed_values: Union[List[float], np.ndarray],
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            baseline_values: Metric values from baseline method
            proposed_values: Metric values from proposed method

        Returns:
            StatisticalTestResult with test information
        """
        baseline = np.array(baseline_values)
        proposed = np.array(proposed_values)

        # Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(proposed, baseline)

        # Effect size: rank-biserial correlation
        diff = proposed - baseline
        pos_ranks = np.sum(diff > 0)
        neg_ranks = np.sum(diff < 0)
        rank_biserial = (pos_ranks - neg_ranks) / max(pos_ranks + neg_ranks, 1)

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(w_stat),
            p_value=float(p_value),
            significant_05=p_value < 0.05,
            significant_01=p_value < 0.01,
            significant_001=p_value < 0.001,
            effect_size=float(rank_biserial),
            effect_size_interpretation=self._interpret_rank_biserial(rank_biserial),
            baseline_mean=float(np.mean(baseline)),
            proposed_mean=float(np.mean(proposed)),
            baseline_std=float(np.std(baseline)),
            proposed_std=float(np.std(proposed)),
        )

    def independent_ttest(
        self,
        group1_values: Union[List[float], np.ndarray],
        group2_values: Union[List[float], np.ndarray],
        equal_var: bool = False,
    ) -> StatisticalTestResult:
        """
        Independent samples t-test for comparing two independent groups.

        Args:
            group1_values: Values from first group
            group2_values: Values from second group
            equal_var: Assume equal variance (standard t-test) or not (Welch's)

        Returns:
            StatisticalTestResult
        """
        g1 = np.array(group1_values)
        g2 = np.array(group2_values)

        t_stat, p_value = stats.ttest_ind(g2, g1, equal_var=equal_var)

        # Cohen's d for independent samples
        pooled_std = np.sqrt((np.std(g1, ddof=1) ** 2 + np.std(g2, ddof=1) ** 2) / 2)
        cohens_d = (np.mean(g2) - np.mean(g1)) / max(pooled_std, 1e-8)

        test_name = "Welch's t-test" if not equal_var else "Independent t-test"

        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(t_stat),
            p_value=float(p_value),
            significant_05=p_value < 0.05,
            significant_01=p_value < 0.01,
            significant_001=p_value < 0.001,
            effect_size=float(cohens_d),
            effect_size_interpretation=self._interpret_cohens_d(cohens_d),
            baseline_mean=float(np.mean(g1)),
            proposed_mean=float(np.mean(g2)),
            baseline_std=float(np.std(g1)),
            proposed_std=float(np.std(g2)),
        )

    def mcnemar_test(
        self,
        baseline_correct: Union[List[bool], np.ndarray],
        proposed_correct: Union[List[bool], np.ndarray],
    ) -> StatisticalTestResult:
        """
        McNemar's test for comparing two classifiers on matched samples.

        Args:
            baseline_correct: Boolean array of baseline correctness
            proposed_correct: Boolean array of proposed correctness

        Returns:
            StatisticalTestResult
        """
        b = np.array(baseline_correct)
        p = np.array(proposed_correct)

        # Contingency table
        b_correct_p_incorrect = np.sum(b & ~p)
        b_incorrect_p_correct = np.sum(~b & p)

        # McNemar's test with continuity correction
        if b_correct_p_incorrect + b_incorrect_p_correct > 0:
            statistic = (abs(b_incorrect_p_correct - b_correct_p_incorrect) - 1) ** 2
            statistic /= b_correct_p_incorrect + b_incorrect_p_correct
            p_value = stats.chi2.sf(statistic, df=1)
        else:
            statistic = 0.0
            p_value = 1.0

        # Effect size: proportion difference
        prop_diff = np.mean(p) - np.mean(b)

        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant_05=p_value < 0.05,
            significant_01=p_value < 0.01,
            significant_001=p_value < 0.001,
            effect_size=float(prop_diff),
            effect_size_interpretation="proportion difference",
            baseline_mean=float(np.mean(b)),
            proposed_mean=float(np.mean(p)),
            additional_info={
                "b_correct_p_incorrect": int(b_correct_p_incorrect),
                "b_incorrect_p_correct": int(b_incorrect_p_correct),
            },
        )

    def bootstrap_confidence_interval(
        self,
        values: Union[List[float], np.ndarray],
        statistic_fn: Callable = np.mean,
        confidence_level: float = 0.95,
        n_resamples: int = 10000,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a statistic.

        Args:
            values: Sample values
            statistic_fn: Function to compute statistic (default: mean)
            confidence_level: Confidence level
            n_resamples: Number of bootstrap resamples

        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        values = np.array(values)
        point_estimate = statistic_fn(values)

        # Bootstrap resampling
        rng = np.random.default_rng(self.random_seed)
        bootstrap_estimates = []
        for _ in range(n_resamples):
            resample = rng.choice(values, size=len(values), replace=True)
            bootstrap_estimates.append(statistic_fn(resample))

        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        return float(point_estimate), float(ci_lower), float(ci_upper)

    def compare_multiple_methods(
        self,
        method_results: Dict[str, List[float]],
        baseline_name: str,
        correction: str = "bonferroni",
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare multiple methods against a baseline with multiple comparison correction.

        Args:
            method_results: Dict mapping method names to lists of metric values
            baseline_name: Name of baseline method
            correction: Multiple comparison correction method

        Returns:
            Dict mapping method names to StatisticalTestResults
        """
        if baseline_name not in method_results:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")

        baseline_values = method_results[baseline_name]
        other_methods = {k: v for k, v in method_results.items() if k != baseline_name}

        # Run tests
        raw_results = {}
        p_values = []

        for method_name, values in other_methods.items():
            result = self.paired_ttest(baseline_values, values)
            raw_results[method_name] = result
            p_values.append(result.p_value)

        # Apply correction
        if correction == "bonferroni" and p_values:
            n_comparisons = len(p_values)
            corrected_p = [min(p * n_comparisons, 1.0) for p in p_values]
            for i, method_name in enumerate(other_methods.keys()):
                raw_results[method_name].p_value = corrected_p[i]
                raw_results[method_name].significant_05 = corrected_p[i] < 0.05
                raw_results[method_name].significant_01 = corrected_p[i] < 0.01
                raw_results[method_name].additional_info["correction"] = "Bonferroni"
                raw_results[method_name].additional_info["n_comparisons"] = (
                    n_comparisons
                )

        return raw_results

    # --------------------------------------------------------------------------
    # Effect Size Interpretation
    # --------------------------------------------------------------------------

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
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

    @staticmethod
    def _interpret_rank_biserial(r: float) -> str:
        """Interpret rank-biserial correlation effect size."""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"


# ==============================================================================
# Visualization
# ==============================================================================


def plot_significance_summary(
    test_results: Dict[str, StatisticalTestResult],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot summary of statistical test results.

    Args:
        test_results: Dict mapping method names to test results
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    methods = list(test_results.keys())
    p_values = [test_results[m].p_value for m in methods]
    effect_sizes = [abs(test_results[m].effect_size) for m in methods]
    colors = [
        "#2ecc71"
        if test_results[m].significant_01
        else "#f39c12"
        if test_results[m].significant_05
        else "#e74c3c"
        for m in methods
    ]

    # P-values with significance threshold
    axes[0].barh(
        methods, [-np.log10(p) for p in p_values], color=colors, edgecolor="black"
    )
    axes[0].axvline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    axes[0].axvline(-np.log10(0.01), color="darkred", linestyle="--", label="p=0.01")
    axes[0].set_xlabel(r"$-\log_{10}(p)$")
    axes[0].set_title("Statistical Significance")
    axes[0].legend(fontsize=8)

    # Effect sizes
    axes[1].barh(methods, effect_sizes, color=colors, edgecolor="black")
    axes[1].axvline(0.2, color="gray", linestyle="--", alpha=0.5, label="small")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="medium")
    axes[1].axvline(0.8, color="gray", linestyle="--", alpha=0.5, label="large")
    axes[1].set_xlabel("|Effect Size|")
    axes[1].set_title("Effect Size Magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Significance summary saved to {save_path}")

    return fig


def plot_confidence_intervals(
    estimates: Dict[str, Tuple[float, float, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot confidence intervals for multiple estimates.

    Args:
        estimates: Dict mapping names to (point_estimate, ci_lower, ci_upper)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(estimates.keys())
    y_pos = np.arange(len(names))
    point_estimates = [estimates[n][0] for n in names]
    ci_lowers = [estimates[n][1] for n in names]
    ci_uppers = [estimates[n][2] for n in names]
    errors = [
        [p - l for p, l in zip(point_estimates, ci_lowers)],
        [u - p for p, u in zip(point_estimates, ci_uppers)],
    ]

    ax.errorbar(
        point_estimates,
        y_pos,
        xerr=errors,
        fmt="o",
        capsize=5,
        capthick=2,
        color="#3498db",
        ecolor="#34495e",
        markersize=8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estimate")
    ax.set_title("Bootstrap Confidence Intervals (95%)")
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confidence intervals plot saved to {save_path}")

    return fig


# ==============================================================================
# Report Generation
# ==============================================================================


def generate_statistical_report(
    test_results: Dict[str, StatisticalTestResult],
    output_path: Optional[str] = None,
    format: str = "markdown",
) -> str:
    """
    Generate a statistical report in Markdown or LaTeX format.

    Args:
        test_results: Statistical test results
        output_path: Path to save report
        format: 'markdown' or 'latex'

    Returns:
        Report string
    """
    if format == "latex":
        return _generate_latex_report(test_results, output_path)
    else:
        return _generate_markdown_report(test_results, output_path)


def _generate_markdown_report(
    test_results: Dict[str, StatisticalTestResult],
    output_path: Optional[str] = None,
) -> str:
    """Generate Markdown report."""
    lines = []
    lines.append("# Statistical Significance Report")
    lines.append("")
    lines.append(
        "| Method | Test | Statistic | p-value | Sig. (p<0.05) | Effect Size | Interpretation |"
    )
    lines.append(
        "|--------|------|-----------|---------|---------------|-------------|----------------|"
    )

    for method_name, result in test_results.items():
        sig_marker = (
            "**Yes**"
            if result.significant_01
            else "Yes"
            if result.significant_05
            else "No"
        )
        lines.append(
            f"| {method_name} | {result.test_name} | "
            f"{result.statistic:.4f} | {result.p_value:.6f} | "
            f"{sig_marker} | {result.effect_size:.4f} | {result.effect_size_interpretation} |"
        )

    lines.append("")
    lines.append("## Summary")
    n_significant_01 = sum(1 for r in test_results.values() if r.significant_01)
    n_significant_05 = sum(1 for r in test_results.values() if r.significant_05)
    n_total = len(test_results)
    lines.append(f"- Significant at p<0.01: {n_significant_01}/{n_total}")
    lines.append(f"- Significant at p<0.05: {n_significant_05}/{n_total}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Markdown report saved to {output_path}")

    return report


def _generate_latex_report(
    test_results: Dict[str, StatisticalTestResult],
    output_path: Optional[str] = None,
) -> str:
    """Generate LaTeX report."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical Significance Tests}")
    lines.append("\\label{tab:significance}")
    lines.append("\\begin{tabular}{@{}l|l|c|c|c|c@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Method} & \\textbf{Test} & \\textbf{Statistic} & \\textbf{p-value} & \\textbf{Sig.} & \\textbf{Effect Size} \\\\"
    )
    lines.append("\\midrule")

    for method_name, result in test_results.items():
        sig = (
            "$\\star\\star$"
            if result.significant_01
            else "$\\star$"
            if result.significant_05
            else "ns"
        )
        lines.append(
            f"{method_name} & {result.test_name} & "
            f"{result.statistic:.4f} & {result.p_value:.6f} & "
            f"{sig} & {result.effect_size:.4f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"LaTeX report saved to {output_path}")

    return latex


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MetroGNN Statistical Testing Utilities")
    print("=" * 70)
    print("\nThis module provides statistical testing capabilities:")
    print("  - Paired t-tests")
    print("  - Wilcoxon signed-rank tests")
    print("  - Independent t-tests (Welch's)")
    print("  - McNemar's test for classification")
    print("  - Bootstrap confidence intervals")
    print("  - Multiple comparison correction (Bonferroni)")
    print("  - Effect size computation and interpretation")
    print("  - Visualization (significance plots, CI plots)")
    print("  - Report generation (Markdown and LaTeX)")
    print("\nUsage:")
    print("  from statistical_testing import StatisticalTester")
    print("  tester = StatisticalTester()")
    print("  result = tester.paired_ttest(baseline_scores, proposed_scores)")
    print("  print(f'p-value: {result.p_value:.6f}')")
