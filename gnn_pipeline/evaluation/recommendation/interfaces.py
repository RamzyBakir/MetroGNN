"""Common interfaces for recommendation baselines and learned systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd


@dataclass(frozen=True)
class RecommendationTrainingData:
    """Training-only labels plus temporally observable corpus/edge data."""

    papers: pd.DataFrame
    train_queries: pd.DataFrame
    train_qrels: pd.DataFrame
    citation_edges: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RankedItem:
    paper_id: str
    score: float


class Recommender(ABC):
    """A recommender that ranks a supplied, temporally eligible corpus."""

    name: str
    stochastic: bool = False

    def get_config(self) -> Dict[str, Any]:
        values = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_")
            and isinstance(value, (str, int, float, bool, type(None), list, tuple))
        }
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            **values,
        }

    def fit(self, training_data: RecommendationTrainingData) -> "Recommender":
        """Fit optional state using training labels and observable graph data only."""
        return self

    def prepare(
        self,
        query: pd.Series,
        candidates: pd.DataFrame,
        visible_citation_edges: pd.DataFrame | None = None,
    ) -> None:
        """Build a cutoff-specific index outside measured warm query latency."""
        return None

    @abstractmethod
    def rank(
        self,
        query: pd.Series,
        candidates: pd.DataFrame,
        top_k: int,
        seed: int,
        visible_citation_edges: pd.DataFrame | None = None,
    ) -> Sequence[RankedItem]:
        """Return up to top_k unique candidates in descending score order."""


