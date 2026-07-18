"""Validation-only model selection and pre-registered ablation utilities."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from gnn_pipeline.artifacts import write_json_atomic


@dataclass(frozen=True)
class TrialOutcome:
    metrics: Dict[str, float]
    artifact_path: str
    artifact_sha256: str


def expand_search_space(search_space: Mapping[str, Sequence]) -> List[Dict]:
    keys = list(search_space)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(search_space[key] for key in keys))
    ]


class ValidationOnlyTuner:
    """Select configurations without exposing a test split to the callback."""

    def __init__(
        self,
        *,
        experiment_name: str,
        benchmark_id: str,
        primary_metric: str = "ndcg@10",
        tie_breaker: str = "recall@20",
        max_coarse_trials: int = 12,
    ):
        self.experiment_name = experiment_name
        self.benchmark_id = benchmark_id
        self.primary_metric = primary_metric
        self.tie_breaker = tie_breaker
        self.max_coarse_trials = int(max_coarse_trials)

    def _run(
        self,
        callback: Callable[[Dict, int], TrialOutcome],
        config: Dict,
        seed: int,
        stage: str,
    ) -> Dict:
        if any("test" in str(key).casefold() for key in config):
            raise ValueError("Test-related configuration is forbidden during tuning")
        outcome = callback(dict(config), int(seed))
        for metric in (self.primary_metric, self.tie_breaker):
            if metric not in outcome.metrics or not np.isfinite(outcome.metrics[metric]):
                raise ValueError(f"Trial did not return finite validation metric {metric}")
        config_id = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "stage": stage,
            "config_id": config_id,
            "config": dict(config),
            "seed": int(seed),
            "validation_split": "validation",
            self.primary_metric: float(outcome.metrics[self.primary_metric]),
            self.tie_breaker: float(outcome.metrics[self.tie_breaker]),
            "artifact_path": outcome.artifact_path,
            "artifact_sha256": outcome.artifact_sha256,
        }

    def tune(
        self,
        *,
        search_space: Mapping[str, Sequence],
        callback: Callable[[Dict, int], TrialOutcome],
        seeds: Sequence[int],
        top_configurations: int = 2,
        output_dir: str,
    ) -> Dict:
        if not seeds:
            raise ValueError("At least one seed is required")
        configurations = expand_search_space(search_space)
        if len(configurations) > self.max_coarse_trials:
            configurations = configurations[: self.max_coarse_trials]

        rows = [
            self._run(callback, config, int(seeds[0]), "coarse")
            for config in configurations
        ]
        coarse = pd.DataFrame(rows).sort_values(
            [self.primary_metric, self.tie_breaker], ascending=False
        )
        selected_ids = coarse.head(top_configurations)["config_id"].tolist()
        selected_configs = [
            row["config"] for row in rows if row["config_id"] in selected_ids
        ]
        repeated = [
            self._run(callback, config, int(seed), "repeated")
            for config in selected_configs
            for seed in seeds
        ]
        all_trials = pd.DataFrame(rows + repeated)
        repeated_frame = pd.DataFrame(repeated)
        summary = (
            repeated_frame.groupby("config_id")[[self.primary_metric, self.tie_breaker]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary.columns = [
            column if isinstance(column, str) else "_".join(part for part in column if part)
            for column in summary.columns
        ]
        winner_id = summary.sort_values(
            [f"{self.primary_metric}_mean", f"{self.tie_breaker}_mean"],
            ascending=False,
        ).iloc[0]["config_id"]
        winner_rows = repeated_frame[repeated_frame["config_id"] == winner_id]
        winner_config = winner_rows.iloc[0]["config"]

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        serializable_trials = all_trials.copy()
        serializable_trials["config"] = serializable_trials["config"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
        serializable_trials.to_csv(root / "validation_trials.csv", index=False)
        summary.to_csv(root / "validation_summary.csv", index=False)
        selection = {
            "experiment_name": self.experiment_name,
            "benchmark_id": self.benchmark_id,
            "selection_split": "validation",
            "primary_metric": self.primary_metric,
            "tie_breaker": self.tie_breaker,
            "coarse_seed": int(seeds[0]),
            "repeated_seeds": [int(seed) for seed in seeds],
            "winner_config_id": winner_id,
            "winner_config": winner_config,
            "winner_artifacts": winner_rows[
                ["seed", "artifact_path", "artifact_sha256"]
            ].to_dict("records"),
        }
        selection["selection_id"] = hashlib.sha256(
            json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        write_json_atomic(str(root / "frozen_selection.json"), selection)
        return selection


def run_validation_ablations(
    *,
    ablations: Mapping[str, Dict],
    callback: Callable[[Dict, int], TrialOutcome],
    seeds: Sequence[int],
    output_path: str,
) -> pd.DataFrame:
    """Evaluate only predeclared ablations on validation across real seeds."""
    rows = []
    for name, config in ablations.items():
        for seed in seeds:
            outcome = callback(dict(config), int(seed))
            rows.append(
                {
                    "ablation": name,
                    "seed": int(seed),
                    "split": "validation",
                    **outcome.metrics,
                    "artifact_path": outcome.artifact_path,
                    "artifact_sha256": outcome.artifact_sha256,
                }
            )
    frame = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
