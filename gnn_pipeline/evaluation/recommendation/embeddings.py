"""Versioned frozen-text embedding caches and retrieval adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from gnn_pipeline.artifacts import sha256_file, write_json_atomic

from .benchmark import TemporalCitationBenchmark
from .interfaces import RankedItem, Recommender


EMBEDDING_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    model_name: str
    text_mode: str = "document"
    batch_size: int = 256
    normalize: bool = True

    def texts(self, frame: pd.DataFrame) -> List[str]:
        if self.text_mode == "title_abstract_sep":
            return (
                frame["title"].fillna("").astype(str)
                + " [SEP] "
                + frame["abstract"].fillna("").astype(str)
            ).tolist()
        if self.text_mode != "document":
            raise ValueError(f"Unknown embedding text mode: {self.text_mode}")
        return frame["document"].fillna("").astype(str).tolist()


@dataclass
class EmbeddingTable:
    paper_ids: np.ndarray
    embeddings: np.ndarray
    manifest: Dict

    def __post_init__(self):
        if len(self.paper_ids) != len(self.embeddings):
            raise ValueError("Embedding IDs and matrix have different lengths")
        if len(set(self.paper_ids.astype(str))) != len(self.paper_ids):
            raise ValueError("Embedding cache contains duplicate paper IDs")
        self._index = {
            str(paper_id): index for index, paper_id in enumerate(self.paper_ids)
        }

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def lookup(self, paper_ids: Sequence[str]) -> np.ndarray:
        missing = [str(paper_id) for paper_id in paper_ids if str(paper_id) not in self._index]
        if missing:
            raise KeyError(f"Embedding cache is missing paper IDs: {missing[:5]}")
        return self.embeddings[[self._index[str(paper_id)] for paper_id in paper_ids]]


def _cache_key(benchmark: TemporalCitationBenchmark, spec: EmbeddingSpec) -> str:
    identity = {
        "benchmark_id": benchmark.benchmark_id,
        "paper_ids": benchmark.papers["paper_id"].astype(str).tolist(),
        "spec": asdict(spec),
        "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def embedding_cache_paths(
    root: str, benchmark: TemporalCitationBenchmark, spec: EmbeddingSpec
) -> tuple[Path, Path]:
    key = _cache_key(benchmark, spec)
    directory = Path(root) / spec.name / key[:16]
    return directory / "embeddings.npz", directory / "manifest.json"


def build_embedding_cache(
    benchmark: TemporalCitationBenchmark,
    spec: EmbeddingSpec,
    *,
    cache_root: str,
    force: bool = False,
    device: str | None = None,
) -> EmbeddingTable:
    """Explicitly encode all paper texts; never called implicitly by evaluation."""
    archive_path, manifest_path = embedding_cache_paths(cache_root, benchmark, spec)
    if archive_path.exists() and manifest_path.exists() and not force:
        return load_embedding_cache(benchmark, spec, cache_root=cache_root)
    if archive_path.exists() != manifest_path.exists() and not force:
        raise ValueError("Incomplete embedding cache; use force=True to replace it")

    from sentence_transformers import SentenceTransformer

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(spec.model_name, device=device)
    texts = spec.texts(benchmark.papers)
    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=spec.batch_size,
            show_progress_bar=True,
            normalize_embeddings=spec.normalize,
        ),
        dtype=np.float32,
    )
    paper_ids = benchmark.papers["paper_id"].astype(str).to_numpy(dtype=str)
    np.savez(archive_path, paper_ids=paper_ids, embeddings=embeddings)
    manifest = {
        "embedding_cache_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "cache_key": _cache_key(benchmark, spec),
        "benchmark_id": benchmark.benchmark_id,
        "spec": asdict(spec),
        "paper_count": len(paper_ids),
        "dimension": int(embeddings.shape[1]),
        "archive": archive_path.name,
        "archive_sha256": sha256_file(str(archive_path)),
    }
    write_json_atomic(str(manifest_path), manifest)
    return EmbeddingTable(paper_ids, embeddings, manifest)


def load_embedding_cache(
    benchmark: TemporalCitationBenchmark,
    spec: EmbeddingSpec,
    *,
    cache_root: str,
) -> EmbeddingTable:
    archive_path, manifest_path = embedding_cache_paths(cache_root, benchmark, spec)
    if not archive_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Embedding cache for {spec.name} is absent. Build it explicitly first."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("embedding_cache_schema_version") != EMBEDDING_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported embedding cache schema")
    if manifest.get("cache_key") != _cache_key(benchmark, spec):
        raise ValueError("Embedding cache identity does not match benchmark/spec")
    if sha256_file(str(archive_path)) != manifest.get("archive_sha256"):
        raise ValueError("Embedding cache checksum mismatch")
    with np.load(archive_path, allow_pickle=False) as payload:
        paper_ids = payload["paper_ids"].astype(str)
        embeddings = payload["embeddings"].astype(np.float32)
    if paper_ids.tolist() != benchmark.papers["paper_id"].astype(str).tolist():
        raise ValueError("Embedding cache paper order does not match benchmark")
    return EmbeddingTable(paper_ids, embeddings, manifest)


class CachedEmbeddingRecommender(Recommender):
    """Cosine retrieval from a validated, normalized frozen embedding cache."""

    def __init__(self, table: EmbeddingTable, name: str):
        self.table = table
        self.name = name

    def get_config(self):
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "cache_key": self.table.manifest["cache_key"],
            "dimension": self.table.dimension,
        }

    def rank(
        self,
        query,
        candidates,
        top_k,
        seed,
        visible_citation_edges=None,
    ):
        query_embedding = self.table.lookup([str(query["query_id"])])[0]
        candidate_ids = candidates["paper_id"].astype(str).tolist()
        candidate_embeddings = self.table.lookup(candidate_ids)
        scores = candidate_embeddings @ query_embedding
        frame = pd.DataFrame({"paper_id": candidate_ids, "score": scores})
        frame = frame.sort_values(
            ["score", "paper_id"], ascending=[False, True], kind="mergesort"
        ).head(top_k)
        return [
            RankedItem(str(row.paper_id), float(row.score))
            for row in frame.itertuples(index=False)
        ]
