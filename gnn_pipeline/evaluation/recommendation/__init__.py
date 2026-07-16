"""Temporal scholarly recommendation evaluation framework."""

from .benchmark import TemporalCitationBenchmark, build_temporal_citation_benchmark
from .interfaces import RankedItem, Recommender
from .metrics import evaluate_rankings
from .statistics import compare_paired_methods

__all__ = [
    "TemporalCitationBenchmark",
    "build_temporal_citation_benchmark",
    "RankedItem",
    "Recommender",
    "evaluate_rankings",
    "compare_paired_methods",
]
