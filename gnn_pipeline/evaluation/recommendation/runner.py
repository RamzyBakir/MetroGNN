"""Execute reproducible recommendation runs and persist every ranked item."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd
import yaml

from gnn_pipeline.artifacts import sha256_file, write_json_atomic

from .benchmark import TemporalCitationBenchmark
from .interfaces import Recommender
from .metrics import aggregate_metrics, evaluate_rankings, validate_predictions
from .statistics import bootstrap_mean_ci, summarize_repeated_seeds


RUN_SCHEMA_VERSION = 1


@dataclass
class EvaluationRun:
    predictions: pd.DataFrame
    timings: pd.DataFrame
    per_query_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    seed_summary: pd.DataFrame
    metadata: Dict

    def save(self, output_dir: str) -> Dict:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        frames = {
            "predictions": self.predictions,
            "timings": self.timings,
            "per_query_metrics": self.per_query_metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "seed_summary": self.seed_summary,
        }
        files = {}
        checksums = {}
        for name, frame in frames.items():
            path = root / f"{name}.csv"
            frame.to_csv(path, index=False)
            files[name] = path.name
            checksums[name] = sha256_file(str(path))
        metadata = {**self.metadata, "files": files, "file_sha256": checksums}
        write_json_atomic(str(root / "run_manifest.json"), metadata)
        self.metadata = metadata
        return metadata


def load_experiment_config(path: str) -> Dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    required = {"experiment", "benchmark", "ranking", "baselines", "statistics"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"Experiment config is missing sections: {sorted(missing)}")
    return config


def _run_id(benchmark_id: str, method_configs: Sequence[Dict], seeds: Sequence[int], splits) -> str:
    value = {
        "benchmark_id": benchmark_id,
        "method_configs": list(method_configs),
        "seeds": [int(seed) for seed in seeds],
        "splits": list(splits),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_recommenders(
    benchmark: TemporalCitationBenchmark,
    recommenders: Sequence[Recommender],
    *,
    splits: Sequence[str] = ("validation", "test"),
    seeds: Sequence[int] = (13, 42, 77, 101, 202),
    top_k: int = 20,
    metric_config: Mapping[str, Sequence[int]] | None = None,
) -> EvaluationRun:
    """Run full-corpus ranking; stochastic/learned methods use every seed."""
    benchmark.validate()
    if not recommenders:
        raise ValueError("At least one recommender is required")
    if not seeds:
        raise ValueError("At least one random seed is required")
    names = [recommender.name for recommender in recommenders]
    if len(set(names)) != len(names):
        raise ValueError("Recommender names must be unique")

    method_configs = [recommender.get_config() for recommender in recommenders]
    training_data = benchmark.training_data()
    prediction_rows: List[Dict] = []
    timing_rows: List[Dict] = []

    for recommender in recommenders:
        fit_start = time.perf_counter()
        recommender.fit(training_data)
        fit_ms = (time.perf_counter() - fit_start) * 1000.0
        method_seeds = list(seeds) if recommender.stochastic else [int(seeds[0])]

        for split in splits:
            queries = benchmark.split_queries(split)
            for seed in method_seeds:
                for query in queries.to_dict("records"):
                    query_series = pd.Series(query)
                    candidates = benchmark.candidates_for_query(query_series)
                    visible_edges = benchmark.visible_citation_edges(query_series)
                    candidate_ids = set(candidates["paper_id"].astype(str))
                    start = time.perf_counter()
                    ranking = list(
                        recommender.rank(
                            query_series,
                            candidates,
                            top_k,
                            int(seed),
                            visible_citation_edges=visible_edges,
                        )
                    )
                    latency_ms = (time.perf_counter() - start) * 1000.0

                    ranked_ids = [str(item.paper_id) for item in ranking]
                    if len(ranked_ids) != len(set(ranked_ids)):
                        raise ValueError(
                            f"{recommender.name} returned duplicate candidates"
                        )
                    invalid = set(ranked_ids) - candidate_ids
                    if invalid:
                        raise ValueError(
                            f"{recommender.name} returned temporally invalid candidates"
                        )
                    expected_length = min(top_k, len(candidates))
                    if len(ranking) != expected_length:
                        raise ValueError(
                            f"{recommender.name} returned {len(ranking)} items; "
                            f"expected {expected_length}"
                        )

                    timing_rows.append(
                        {
                            "query_id": str(query["query_id"]),
                            "method": recommender.name,
                            "seed": int(seed),
                            "split": split,
                            "fit_ms": fit_ms,
                            "latency_ms": latency_ms,
                            "candidate_count": len(candidates),
                            "benchmark_id": benchmark.benchmark_id,
                        }
                    )
                    for rank, item in enumerate(ranking, start=1):
                        prediction_rows.append(
                            {
                                "query_id": str(query["query_id"]),
                                "candidate_id": str(item.paper_id),
                                "rank": rank,
                                "score": float(item.score),
                                "method": recommender.name,
                                "seed": int(seed),
                                "split": split,
                                "benchmark_id": benchmark.benchmark_id,
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    timings = pd.DataFrame(timing_rows)
    validate_predictions(predictions)
    per_query = evaluate_rankings(predictions, benchmark.qrels, metric_config)
    aggregate = aggregate_metrics(per_query)
    seed_summary = summarize_repeated_seeds(per_query)
    run_id = _run_id(benchmark.benchmark_id, method_configs, seeds, splits)
    metadata = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "benchmark_id": benchmark.benchmark_id,
        "methods": names,
        "method_configs": method_configs,
        "seeds": [int(seed) for seed in seeds],
        "splits": list(splits),
        "top_k": int(top_k),
        "prediction_rows": len(predictions),
        "timing_rows": len(timings),
    }
    return EvaluationRun(
        predictions=predictions,
        timings=timings,
        per_query_metrics=per_query,
        aggregate_metrics=aggregate,
        seed_summary=seed_summary,
        metadata=metadata,
    )


def confidence_interval_table(
    per_query_metrics: pd.DataFrame,
    *,
    metrics: Sequence[str],
    split: str,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap methods over queries after averaging their observed seeds."""
    rows = []
    for method in sorted(per_query_metrics["method"].unique()):
        subset = per_query_metrics[
            (per_query_metrics["method"] == method)
            & (per_query_metrics["split"] == split)
        ]
        averaged = subset.groupby("query_id", as_index=False)[list(metrics)].mean()
        for offset, metric in enumerate(metrics):
            estimate, lower, upper = bootstrap_mean_ci(
                averaged[metric].to_numpy(dtype=float),
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + offset,
            )
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "metric": metric,
                    "num_queries": len(averaged),
                    "mean": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)
