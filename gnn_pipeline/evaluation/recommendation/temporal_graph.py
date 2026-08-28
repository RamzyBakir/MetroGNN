"""Temporal heterogeneous graph snapshots for cold-query recommendation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from gnn_pipeline.artifacts import sha256_file, write_json_atomic
from gnn_pipeline.data.feature_engineering import create_author_features
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.parsing import build_author_index, parse_list_field
from gnn_pipeline.data.validation import ensure_paper_identity_columns

from .embeddings import EmbeddingTable


TEMPORAL_GRAPH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelationConfig:
    authorship: bool = True
    collaboration: bool = True
    citation: bool = True
    topic: bool = True

    @property
    def name(self) -> str:
        enabled = [
            name
            for name, value in asdict(self).items()
            if value
        ]
        return "+".join(enabled) if enabled else "no_relations"


@dataclass
class TemporalTopicModel:
    n_clusters: int
    random_seed: int
    centers: np.ndarray
    model_id: str

    @classmethod
    def fit(
        cls,
        paper_ids: List[str],
        embeddings: np.ndarray,
        *,
        n_clusters: int,
        random_seed: int = 42,
    ) -> "TemporalTopicModel":
        if len(paper_ids) < n_clusters:
            raise ValueError("Fewer training papers than requested topic clusters")
        model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        model.fit(embeddings)
        identity = {
            "paper_ids": list(paper_ids),
            "n_clusters": int(n_clusters),
            "random_seed": int(random_seed),
            "centers_sha256": hashlib.sha256(
                model.cluster_centers_.astype(np.float32).tobytes()
            ).hexdigest(),
        }
        model_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            n_clusters=n_clusters,
            random_seed=random_seed,
            centers=model.cluster_centers_.astype(np.float32),
            model_id=model_id,
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        distances = ((embeddings[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(np.int64)


@dataclass
class TemporalGraphSnapshot:
    cutoff_year: int
    data: object
    paper_ids: List[str]
    author_records: List[Dict]
    relation_config: RelationConfig
    snapshot_id: str
    benchmark_id: str
    embedding_cache_key: str
    topic_model_id: str

    @property
    def paper_to_idx(self) -> Dict[str, int]:
        return {paper_id: index for index, paper_id in enumerate(self.paper_ids)}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def fit_training_topic_model(
    benchmark,
    embedding_table: EmbeddingTable,
    *,
    training_cutoff: int,
    n_clusters: int,
    random_seed: int = 42,
) -> TemporalTopicModel:
    papers = benchmark.papers[benchmark.papers["year"] <= training_cutoff]
    paper_ids = papers["paper_id"].astype(str).tolist()
    embeddings = embedding_table.lookup(paper_ids)
    return TemporalTopicModel.fit(
        paper_ids,
        embeddings,
        n_clusters=n_clusters,
        random_seed=random_seed,
    )


def build_temporal_graph_snapshot(
    *,
    benchmark,
    raw_csv_path: str,
    embedding_table: EmbeddingTable,
    topic_model: TemporalTopicModel,
    cutoff_year: int,
    relation_config: RelationConfig,
) -> TemporalGraphSnapshot:
    """Build a graph containing no paper or relation after cutoff_year."""
    raw = ensure_paper_identity_columns(benchmark.papers.copy())
    if "authors" not in raw.columns:
        try:
            raw_source = ensure_paper_identity_columns(pd.read_csv(raw_csv_path))
            if "authors" in raw_source.columns and len(raw_source) >= len(raw):
                raw["authors"] = raw_source["authors"].iloc[: len(raw)].values
            else:
                raw["authors"] = "[]"
        except Exception:
            raw["authors"] = "[]"
    raw["paper_id"] = raw["id"].astype(str)
    raw["year_numeric"] = pd.to_numeric(raw["year"], errors="coerce")
    frame = raw[raw["year_numeric"].notna() & (raw["year_numeric"] <= cutoff_year)].copy()
    frame = frame.reset_index(drop=True)
    paper_ids = frame["paper_id"].astype(str).tolist()
    expected_ids = benchmark.papers[benchmark.papers["year"] <= cutoff_year][
        "paper_id"
    ].astype(str).tolist()
    if paper_ids != expected_ids:
        raise ValueError("Temporal graph paper order differs from benchmark candidate order")

    if "references" not in frame.columns or frame["references"].apply(lambda x: len(parse_list_field(x))).sum() == 0:
        if hasattr(benchmark, "citation_edges") and not benchmark.citation_edges.empty:
            edges = benchmark.citation_edges[
                (benchmark.citation_edges["source_year"] <= cutoff_year)
                & (benchmark.citation_edges["target_year"] <= cutoff_year)
            ]
            refs_by_source = defaultdict(list)
            for _, edge in edges.iterrows():
                refs_by_source[str(edge["source_id"])].append(str(edge["target_id"]))
            frame["references"] = frame["paper_id"].apply(
                lambda pid: str(refs_by_source.get(str(pid), []))
            )

    paper_embeddings = embedding_table.lookup(paper_ids).astype(np.float32)
    paper_embeddings = _normalize_rows(paper_embeddings)
    topic_ids = topic_model.predict(paper_embeddings)
    frame["topic_id"] = topic_ids

    author_to_idx, author_records, paper_author_indices, _ = build_author_index(frame)
    frame["author_indices"] = paper_author_indices
    frame["authors_parsed"] = [
        [author_records[index]["author_key"] for index in indices]
        for indices in paper_author_indices
    ]
    author_features = create_author_features(frame, author_to_idx)[:, :4]
    author_features = np.log1p(np.maximum(author_features, 0.0)).astype(np.float32)
    means = author_features.mean(axis=0, keepdims=True)
    stds = author_features.std(axis=0, keepdims=True)
    author_features = (author_features - means) / np.maximum(stds, 1e-6)
    topic_features = _normalize_rows(topic_model.centers.astype(np.float32))
    topic_info = {
        topic_id: {"name": f"Temporal Topic {topic_id}"}
        for topic_id in range(topic_model.n_clusters)
    }

    data, _, _ = build_knowledge_graph_from_dblp(
        frame,
        author_to_idx,
        topic_info,
        paper_embeddings,
        author_features,
        topic_features,
        save_path=None,
    )
    relation_groups = {
        "authorship": [
            ("author", "authored", "paper"),
            ("paper", "authored_by", "author"),
        ],
        "collaboration": [("author", "collaborates_with", "author")],
        "citation": [
            ("paper", "cites", "paper"),
            ("paper", "cited_by", "paper"),
        ],
        "topic": [
            ("paper", "has_topic", "topic"),
            ("topic", "has_paper", "paper"),
        ],
    }
    for group, edge_types in relation_groups.items():
        if not getattr(relation_config, group):
            for edge_type in edge_types:
                if edge_type in data.edge_types:
                    del data[edge_type]

    identity = {
        "schema": TEMPORAL_GRAPH_SCHEMA_VERSION,
        "benchmark_id": benchmark.benchmark_id,
        "embedding_cache_key": embedding_table.manifest["cache_key"],
        "topic_model_id": topic_model.model_id,
        "cutoff_year": int(cutoff_year),
        "relations": asdict(relation_config),
        "paper_ids": paper_ids,
        "author_keys": [record["author_key"] for record in author_records],
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data.temporal_cutoff_year = int(cutoff_year)
    data.temporal_snapshot_id = snapshot_id
    return TemporalGraphSnapshot(
        cutoff_year=int(cutoff_year),
        data=data,
        paper_ids=paper_ids,
        author_records=author_records,
        relation_config=relation_config,
        snapshot_id=snapshot_id,
        benchmark_id=benchmark.benchmark_id,
        embedding_cache_key=embedding_table.manifest["cache_key"],
        topic_model_id=topic_model.model_id,
    )


def save_temporal_topic_model(model: TemporalTopicModel, output_dir: str) -> Dict:
    root = Path(output_dir) / "topic_models" / model.model_id[:16]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "centers.npz"
    np.savez(
        archive,
        centers=model.centers,
        n_clusters=np.asarray([model.n_clusters]),
        random_seed=np.asarray([model.random_seed]),
    )
    manifest = {
        "model_id": model.model_id,
        "n_clusters": model.n_clusters,
        "random_seed": model.random_seed,
        "archive": archive.name,
        "archive_sha256": sha256_file(str(archive)),
    }
    write_json_atomic(str(root / "manifest.json"), manifest)
    return manifest


def load_temporal_topic_model(model_dir: str) -> TemporalTopicModel:
    root = Path(model_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    archive = root / manifest["archive"]
    if sha256_file(str(archive)) != manifest["archive_sha256"]:
        raise ValueError("Temporal topic model checksum mismatch")
    with np.load(archive, allow_pickle=False) as payload:
        centers = payload["centers"].astype(np.float32)
        n_clusters = int(payload["n_clusters"][0])
        random_seed = int(payload["random_seed"][0])
    return TemporalTopicModel(
        n_clusters=n_clusters,
        random_seed=random_seed,
        centers=centers,
        model_id=manifest["model_id"],
    )


def save_temporal_snapshot(snapshot: TemporalGraphSnapshot, output_dir: str) -> Dict:
    root = Path(output_dir) / str(snapshot.cutoff_year) / snapshot.snapshot_id[:16]
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / "graph.pt"
    torch.save(snapshot.data, graph_path)
    mapping_path = root / "mappings.json"
    mapping_path.write_text(
        json.dumps(
            {
                "paper_ids": snapshot.paper_ids,
                "author_records": snapshot.author_records,
            },
            ensure_ascii=False,
        )
    )
    manifest = {
        "temporal_graph_schema_version": TEMPORAL_GRAPH_SCHEMA_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "cutoff_year": snapshot.cutoff_year,
        "benchmark_id": snapshot.benchmark_id,
        "embedding_cache_key": snapshot.embedding_cache_key,
        "topic_model_id": snapshot.topic_model_id,
        "relations": asdict(snapshot.relation_config),
        "node_counts": {
            node_type: int(snapshot.data[node_type].num_nodes or 0)
            for node_type in snapshot.data.node_types
        },
        "edge_counts": {
            "|".join(edge_type): int(snapshot.data[edge_type].edge_index.shape[1])
            for edge_type in snapshot.data.edge_types
        },
        "graph": graph_path.name,
        "graph_sha256": sha256_file(str(graph_path)),
        "mappings": mapping_path.name,
        "mappings_sha256": sha256_file(str(mapping_path)),
    }
    write_json_atomic(str(root / "manifest.json"), manifest)
    return manifest


def find_temporal_snapshot(root: str, cutoff_year: int) -> TemporalGraphSnapshot:
    cutoff_root = Path(root) / str(cutoff_year)
    manifests = sorted(cutoff_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Expected exactly one snapshot for cutoff {cutoff_year} under {root}; "
            f"found {len(manifests)}"
        )
    return load_temporal_snapshot(str(manifests[0].parent))


def load_temporal_snapshot(snapshot_dir: str) -> TemporalGraphSnapshot:
    root = Path(snapshot_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("temporal_graph_schema_version") != TEMPORAL_GRAPH_SCHEMA_VERSION:
        raise ValueError("Unsupported temporal graph snapshot schema")
    graph_path = root / manifest["graph"]
    mapping_path = root / manifest["mappings"]
    if sha256_file(str(graph_path)) != manifest["graph_sha256"]:
        raise ValueError("Temporal graph checksum mismatch")
    if sha256_file(str(mapping_path)) != manifest["mappings_sha256"]:
        raise ValueError("Temporal graph mapping checksum mismatch")
    mappings = json.loads(mapping_path.read_text())
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if getattr(data, "temporal_snapshot_id", None) != manifest["snapshot_id"]:
        raise ValueError("Temporal graph object and manifest IDs differ")
    return TemporalGraphSnapshot(
        cutoff_year=int(manifest["cutoff_year"]),
        data=data,
        paper_ids=[str(value) for value in mappings["paper_ids"]],
        author_records=mappings["author_records"],
        relation_config=RelationConfig(**manifest["relations"]),
        snapshot_id=manifest["snapshot_id"],
        benchmark_id=manifest["benchmark_id"],
        embedding_cache_key=manifest["embedding_cache_key"],
        topic_model_id=manifest["topic_model_id"],
    )


def relation_ablation_configs() -> Dict[str, RelationConfig]:
    return {
        "authorship_only": RelationConfig(
            authorship=True, collaboration=False, citation=False, topic=False
        ),
        "authorship_collaboration": RelationConfig(
            authorship=True, collaboration=True, citation=False, topic=False
        ),
        "authorship_citation": RelationConfig(
            authorship=True, collaboration=False, citation=True, topic=False
        ),
        "full_without_topics": RelationConfig(
            authorship=True, collaboration=True, citation=True, topic=False
        ),
        "full": RelationConfig(),
    }
