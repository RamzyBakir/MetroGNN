"""Configuration helpers for opt-in Phase 3 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from .benchmark import TemporalCitationBenchmark
from .embeddings import EmbeddingSpec, load_embedding_cache


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_phase3_config(path: str) -> Dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    required = {
        "experiment",
        "embeddings",
        "temporal_graph",
        "sampling",
        "model_search",
        "architectures",
        "relation_ablations",
        "final_test",
    }
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"Phase 3 config is missing sections: {sorted(missing)}")
    if config["experiment"].get("validation_split") != "validation":
        raise ValueError("Phase 3 tuning must use the validation split")
    if config["experiment"].get("forbidden_during_tuning") != "test":
        raise ValueError("Phase 3 config must explicitly forbid test during tuning")
    model_version = int(config["experiment"].get("model_version", 1))
    if model_version not in (1, 2):
        raise ValueError("Unsupported Phase 3 model version")
    weights = [float(value) for value in config["model_search"].get("fusion_weights", [1.0])]
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("Fusion weights must be a non-empty list within [0, 1]")
    if model_version == 2 and 0.0 not in weights:
        raise ValueError("Phase 3 v2 must include zero to recover the frozen TF-IDF baseline")
    return config


def resolve_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def embedding_spec(config: Dict, name: str) -> EmbeddingSpec:
    values = config["embeddings"]["candidates"].get(name)
    if values is None:
        raise KeyError(f"Unknown configured embedding: {name}")
    return EmbeddingSpec(name=name, **values)


def load_phase3_benchmark(config: Dict) -> TemporalCitationBenchmark:
    return TemporalCitationBenchmark.load(
        resolve_project_path(config["experiment"]["benchmark_dir"])
    )


def load_phase3_embeddings(config: Dict, benchmark, name: str):
    return load_embedding_cache(
        benchmark,
        embedding_spec(config, name),
        cache_root=resolve_project_path(config["embeddings"]["cache_root"]),
    )
