"""Validation-only lexical/neural score fusion for temporal recommendation."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from .baselines import TfidfRecommender
from .metrics import evaluate_query


def fuse_scores(neural_scores, lexical_scores, weight: float):
    """Convexly combine comparable cosine and TF-IDF scores."""
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Fusion weight must be in [0, 1]")
    return weight * neural_scores + (1.0 - weight) * lexical_scores


def deterministic_top_indices(
    scores: np.ndarray, candidate_ids: Sequence[str], top_k: int
) -> np.ndarray:
    """Rank each row by score descending and paper ID ascending."""
    values = np.asarray(scores)
    if values.ndim == 1:
        values = values[None, :]
    ids = np.asarray([str(value) for value in candidate_ids])
    if values.shape[1] != len(ids):
        raise ValueError("Score width does not match candidate IDs")
    return np.vstack(
        [np.lexsort((ids, -row))[: min(top_k, len(ids))] for row in values]
    )


class ValidationFusionEvaluator:
    """Cache validation TF-IDF scores and select a predeclared fusion weight."""

    def __init__(self, benchmark, *, split: str = "validation", max_k: int = 20):
        if split == "test":
            raise ValueError("Fusion weights must never be selected on the test split")
        self.benchmark = benchmark
        self.split = split
        self.max_k = int(max_k)
        self.queries = benchmark.split_queries(split).reset_index(drop=True)
        years = sorted(self.queries["query_year"].astype(int).unique().tolist())
        if len(years) != 1:
            raise ValueError("Fusion evaluator requires a single temporal query year")
        first_query = self.queries.iloc[0]
        self.candidates = benchmark.candidates_for_query(first_query)
        self.candidate_ids = self.candidates["paper_id"].astype(str).tolist()
        scorer = TfidfRecommender()
        lexical_rows = []
        for _, query in self.queries.iterrows():
            candidates = benchmark.candidates_for_query(query)
            ids = candidates["paper_id"].astype(str).tolist()
            if ids != self.candidate_ids:
                raise ValueError("Validation candidate corpus changed within the split")
            lexical_rows.append(scorer.score_candidates(query, candidates))
        self.lexical_scores = np.vstack(lexical_rows).astype(np.float32)

    def evaluate(
        self,
        neural_scores: torch.Tensor | np.ndarray,
        fusion_weights: Iterable[float],
    ) -> Dict[str, float]:
        if isinstance(neural_scores, torch.Tensor):
            neural = neural_scores.detach().cpu().numpy()
        else:
            neural = np.asarray(neural_scores)
        if neural.shape != self.lexical_scores.shape:
            raise ValueError(
                f"Neural score shape {neural.shape} does not match cached lexical "
                f"shape {self.lexical_scores.shape}"
            )

        best = None
        for weight in [float(value) for value in fusion_weights]:
            combined = fuse_scores(neural, self.lexical_scores, weight)
            top_indices = deterministic_top_indices(
                combined, self.candidate_ids, self.max_k
            )
            rows = []
            for row_index, query in self.queries.iterrows():
                ranking = [
                    self.candidate_ids[index] for index in top_indices[row_index]
                ]
                rows.append(
                    evaluate_query(
                        ranking,
                        self.benchmark.qrels_for_query(str(query["query_id"])),
                        {"ndcg": [10], "recall": [20]},
                    )
                )
            result = {
                "ndcg@10": float(np.mean([row["ndcg@10"] for row in rows])),
                "recall@20": float(np.mean([row["recall@20"] for row in rows])),
                "fusion_weight": weight,
            }
            key = (result["ndcg@10"], result["recall@20"], -weight)
            if best is None or key > best[0]:
                best = (key, result)
        if best is None:
            raise ValueError("At least one fusion weight is required")
        return best[1]
