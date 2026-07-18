"""Deterministic positive-unlabeled pair sampling for recommendation training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from gnn_pipeline.artifacts import sha256_file, write_json_atomic


PAIRWISE_SAMPLE_SCHEMA_VERSION = 1


@dataclass
class PairwiseSamples:
    triples: pd.DataFrame
    manifest: Dict

    def save(self, output_dir: str) -> Dict:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "pairwise_triples.csv"
        self.triples.to_csv(path, index=False)
        manifest = {
            **self.manifest,
            "file": path.name,
            "file_sha256": sha256_file(str(path)),
        }
        write_json_atomic(str(root / "sampling_manifest.json"), manifest)
        self.manifest = manifest
        return manifest


def load_pairwise_samples(output_dir: str) -> PairwiseSamples:
    root = Path(output_dir)
    manifest = json.loads((root / "sampling_manifest.json").read_text())
    if manifest.get("pairwise_sample_schema_version") != PAIRWISE_SAMPLE_SCHEMA_VERSION:
        raise ValueError("Unsupported pairwise sample schema")
    path = root / manifest["file"]
    if sha256_file(str(path)) != manifest["file_sha256"]:
        raise ValueError("Pairwise sample checksum mismatch")
    triples = pd.read_csv(path, dtype={
        "query_id": str,
        "positive_id": str,
        "negative_id": str,
        "negative_type": str,
    })
    return PairwiseSamples(triples=triples, manifest=manifest)


def build_pairwise_samples(
    benchmark,
    *,
    feature_store,
    random_negatives: int = 20,
    hard_negatives: int = 20,
    seed: int = 42,
) -> PairwiseSamples:
    """Pair each known citation with reproducible random and TF-IDF hard negatives."""
    train_queries = benchmark.split_queries("train")
    all_qrels = {
        str(query_id): set(group["candidate_id"].astype(str))
        for query_id, group in benchmark.qrels.groupby("query_id")
    }
    rows = []
    for query in train_queries.to_dict("records"):
        query_series = pd.Series(query)
        query_id = str(query["query_id"])
        positives = sorted(all_qrels[query_id])
        candidates = benchmark.candidates_for_query(query_series)
        candidate_ids = candidates["paper_id"].astype(str).tolist()
        eligible = [paper_id for paper_id in candidate_ids if paper_id not in all_qrels[query_id]]
        if not eligible:
            continue

        tfidf_scores = feature_store.tfidf.score_candidates(query_series, candidates)
        hard_order = [
            candidate_ids[index]
            for index in np.argsort(tfidf_scores)[::-1]
            if candidate_ids[index] not in all_qrels[query_id]
        ]
        material = f"{seed}:{query_id}".encode()
        query_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        rng = np.random.default_rng(query_seed)

        for positive_id in positives:
            hard = hard_order[: min(hard_negatives, len(hard_order))]
            remaining = [paper_id for paper_id in eligible if paper_id not in set(hard)]
            random_count = min(random_negatives, len(remaining))
            random_items = (
                rng.choice(remaining, size=random_count, replace=False).tolist()
                if random_count
                else []
            )
            for negative_type, negatives in (("hard", hard), ("random", random_items)):
                for negative_id in negatives:
                    rows.append(
                        {
                            "query_id": query_id,
                            "positive_id": str(positive_id),
                            "negative_id": str(negative_id),
                            "negative_type": negative_type,
                            "query_year": int(query["query_year"]),
                            "snapshot_cutoff": int(query["query_year"]) - 1,
                        }
                    )

    triples = pd.DataFrame(rows)
    if triples.empty:
        raise ValueError("No pairwise training triples could be generated")
    if (triples["positive_id"] == triples["negative_id"]).any():
        raise ValueError("A pairwise sample uses its positive as a negative")
    identity = {
        "pairwise_sample_schema_version": PAIRWISE_SAMPLE_SCHEMA_VERSION,
        "benchmark_id": benchmark.benchmark_id,
        "random_negatives": int(random_negatives),
        "hard_negatives": int(hard_negatives),
        "seed": int(seed),
        "num_triples": len(triples),
        "num_queries": int(triples["query_id"].nunique()),
    }
    identity["sample_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PairwiseSamples(triples=triples, manifest=identity)
