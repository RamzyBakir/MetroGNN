"""Ranking metrics computed from persisted recommendations and real qrels."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def _relevance_sequence(
    ranking: Sequence[str], qrels: Mapping[str, float], k: int
) -> np.ndarray:
    return np.asarray([float(qrels.get(str(item), 0.0)) for item in ranking[:k]])


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    relevant = {item for item, relevance in qrels.items() if relevance > 0}
    if not relevant:
        return 0.0
    retrieved = set(str(item) for item in ranking[:k])
    return len(relevant & retrieved) / len(relevant)


def hit_rate_at_k(
    ranking: Sequence[str], qrels: Mapping[str, float], k: int
) -> float:
    return float(recall_at_k(ranking, qrels, k) > 0)


def reciprocal_rank_at_k(
    ranking: Sequence[str], qrels: Mapping[str, float], k: int
) -> float:
    for rank, item in enumerate(ranking[:k], start=1):
        if qrels.get(str(item), 0.0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranking: Sequence[str], qrels: Mapping[str, float], k: int
) -> float:
    relevant = {item for item, relevance in qrels.items() if relevance > 0}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranking[:k], start=1):
        if str(item) in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    gains = _relevance_sequence(ranking, qrels, k)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    dcg = np.sum((np.power(2.0, gains) - 1.0) / discounts)
    ideal = np.asarray(sorted((float(value) for value in qrels.values()), reverse=True)[:k])
    if ideal.size == 0:
        return 0.0
    ideal_discounts = np.log2(np.arange(2, ideal.size + 2))
    idcg = np.sum((np.power(2.0, ideal) - 1.0) / ideal_discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_query(
    ranking: Sequence[str],
    qrels: Mapping[str, float],
    metric_config: Mapping[str, Sequence[int]] | None = None,
) -> Dict[str, float]:
    config = metric_config or {
        "recall": [5, 10, 20],
        "ndcg": [10, 20],
        "hit_rate": [5, 10, 20],
        "mrr": [10],
        "map": [20],
    }
    values: Dict[str, float] = {}
    for k in config.get("recall", []):
        values[f"recall@{k}"] = recall_at_k(ranking, qrels, int(k))
    for k in config.get("ndcg", []):
        values[f"ndcg@{k}"] = ndcg_at_k(ranking, qrels, int(k))
    for k in config.get("hit_rate", []):
        values[f"hit_rate@{k}"] = hit_rate_at_k(ranking, qrels, int(k))
    for k in config.get("mrr", []):
        values[f"mrr@{k}"] = reciprocal_rank_at_k(ranking, qrels, int(k))
    for k in config.get("map", []):
        values[f"map@{k}"] = average_precision_at_k(ranking, qrels, int(k))
    return values


def validate_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "query_id",
        "candidate_id",
        "rank",
        "score",
        "method",
        "seed",
        "split",
        "benchmark_id",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    duplicate = predictions.duplicated(
        ["query_id", "candidate_id", "method", "seed", "split"]
    )
    if duplicate.any():
        raise ValueError("Predictions contain duplicate candidates for a query run")
    expected_ranks = predictions.groupby(
        ["query_id", "method", "seed", "split"]
    )["rank"].apply(lambda ranks: sorted(ranks.astype(int)) == list(range(1, len(ranks) + 1)))
    if not expected_ranks.all():
        raise ValueError("Prediction ranks must be contiguous and one-indexed")


def evaluate_rankings(
    predictions: pd.DataFrame,
    qrels: pd.DataFrame,
    metric_config: Mapping[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Return one metric row per query, method, seed, and split."""
    validate_predictions(predictions)
    qrel_groups = {
        str(query_id): {
            str(row.candidate_id): float(row.relevance)
            for row in group.itertuples(index=False)
        }
        for query_id, group in qrels.groupby("query_id")
    }

    rows = []
    group_columns = ["query_id", "method", "seed", "split", "benchmark_id"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        query_id, method, seed, split, benchmark_id = keys
        if str(query_id) not in qrel_groups:
            raise ValueError(f"Predictions have no qrels for query {query_id}")
        ordered = group.sort_values("rank")["candidate_id"].astype(str).tolist()
        metrics = evaluate_query(ordered, qrel_groups[str(query_id)], metric_config)
        rows.append(
            {
                "query_id": str(query_id),
                "method": method,
                "seed": int(seed),
                "split": split,
                "benchmark_id": benchmark_id,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(per_query_metrics: pd.DataFrame) -> pd.DataFrame:
    """Macro-average queries for each method/seed/split without synthetic data."""
    id_columns = {"query_id", "method", "seed", "split", "benchmark_id"}
    metric_columns = [column for column in per_query_metrics if column not in id_columns]
    return (
        per_query_metrics.groupby(["method", "seed", "split"], as_index=False)[
            metric_columns
        ]
        .mean()
        .sort_values(["split", "method", "seed"])
        .reset_index(drop=True)
    )


def catalog_coverage_at_k(
    predictions: pd.DataFrame, candidate_catalog_size: int, k: int
) -> float:
    if candidate_catalog_size <= 0:
        return 0.0
    selected = predictions[predictions["rank"].astype(int) <= k]
    return selected["candidate_id"].astype(str).nunique() / candidate_catalog_size
