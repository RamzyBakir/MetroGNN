"""Build and persist a leakage-aware temporal citation benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from gnn_pipeline.artifacts import sha256_file, write_json_atomic
from gnn_pipeline.data.parsing import parse_list_field
from gnn_pipeline.data.validation import validate_papers_dataframe


BENCHMARK_SCHEMA_VERSION = 1
PAPER_TEXT_FIELDS = ("title", "abstract", "keywords", "venue")


def _document(row: pd.Series, fields: Sequence[str]) -> str:
    parts = []
    for field in fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return " ".join(parts)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class TemporalCitationBenchmark:
    papers: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    citation_edges: pd.DataFrame
    metadata: Dict[str, Any]

    @property
    def benchmark_id(self) -> str:
        return str(self.metadata["benchmark_id"])

    def split_queries(self, split: str) -> pd.DataFrame:
        result = self.queries[self.queries["split"] == split].copy()
        if result.empty and split not in set(self.queries["split"]):
            raise KeyError(f"Unknown or empty benchmark split: {split}")
        return result.reset_index(drop=True)

    def training_data(self):
        from .interfaces import RecommendationTrainingData

        train_queries = self.split_queries("train")
        train_ids = set(train_queries["query_id"].astype(str))
        training_cutoff = int(train_queries["query_year"].max())
        return RecommendationTrainingData(
            papers=self.papers[self.papers["year"] <= training_cutoff].copy(),
            train_queries=train_queries,
            train_qrels=self.qrels[self.qrels["query_id"].isin(train_ids)].copy(),
            citation_edges=self.citation_edges[
                self.citation_edges["source_year"].astype(int) <= training_cutoff
            ].copy(),
            metadata=dict(self.metadata),
        )

    def qrels_for_query(self, query_id: str) -> Dict[str, float]:
        rows = self.qrels[self.qrels["query_id"] == str(query_id)]
        return {
            str(row.candidate_id): float(row.relevance)
            for row in rows.itertuples(index=False)
        }

    def visible_citation_edges(self, query: pd.Series) -> pd.DataFrame:
        """Return only citation events observable before the query year."""
        query_year = int(query["query_year"])
        return self.citation_edges[
            self.citation_edges["source_year"].astype(int) < query_year
        ].copy()

    def candidates_for_query(self, query: pd.Series) -> pd.DataFrame:
        """Return the full strictly-prior candidate corpus for one query."""
        query_year = int(query["query_year"])
        candidates = self.papers[
            (self.papers["year"] < query_year)
            & (self.papers["paper_id"] != str(query["query_id"]))
        ]
        return candidates.reset_index(drop=True)

    def validate(self) -> Dict[str, int]:
        paper_ids = set(self.papers["paper_id"].astype(str))
        if not self.papers["paper_id"].is_unique:
            raise ValueError("Benchmark paper IDs are not unique")
        if not self.queries["query_id"].is_unique:
            raise ValueError("Benchmark query IDs are not unique")
        if not set(self.qrels["query_id"]).issubset(set(self.queries["query_id"])):
            raise ValueError("A qrel references an unknown query")
        if not set(self.qrels["candidate_id"]).issubset(paper_ids):
            raise ValueError("A qrel candidate is absent from the paper table")

        query_years = dict(zip(self.queries["query_id"], self.queries["query_year"]))
        paper_years = dict(zip(self.papers["paper_id"], self.papers["year"]))
        for row in self.qrels.itertuples(index=False):
            if paper_years[row.candidate_id] >= query_years[row.query_id]:
                raise ValueError("A qrel violates the strictly-prior candidate policy")

        positives_per_query = self.qrels[self.qrels["relevance"].astype(float) > 0].groupby(
            "query_id"
        ).size()
        if set(positives_per_query.index) != set(self.queries["query_id"]):
            raise ValueError("Every benchmark query must have at least one positive qrel")
        return {
            "papers": len(self.papers),
            "queries": len(self.queries),
            "qrels": len(self.qrels),
            "citation_edges": len(self.citation_edges),
        }

    def save(self, output_dir: str) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "papers": output / "papers.csv",
            "queries": output / "queries.csv",
            "qrels": output / "qrels.csv",
            "citation_edges": output / "citation_edges.csv",
        }
        self.papers.to_csv(paths["papers"], index=False)
        self.queries.to_csv(paths["queries"], index=False)
        self.qrels.to_csv(paths["qrels"], index=False)
        self.citation_edges.to_csv(paths["citation_edges"], index=False)

        manifest = dict(self.metadata)
        manifest["files"] = {key: path.name for key, path in paths.items()}
        manifest["file_sha256"] = {
            key: sha256_file(str(path)) for key, path in paths.items()
        }
        write_json_atomic(str(output / "benchmark_manifest.json"), manifest)
        self.metadata = manifest
        return manifest

    @classmethod
    def load(cls, output_dir: str) -> "TemporalCitationBenchmark":
        root = Path(output_dir)
        manifest_path = root / "benchmark_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise ValueError("Unsupported recommendation benchmark schema")

        loaded = {}
        for key, filename in manifest["files"].items():
            path = root / filename
            if sha256_file(str(path)) != manifest["file_sha256"][key]:
                raise ValueError(f"Benchmark file checksum mismatch: {key}")
            loaded[key] = pd.read_csv(path, dtype=str)

        for frame_name in ("papers", "queries"):
            year_column = "year" if frame_name == "papers" else "query_year"
            loaded[frame_name][year_column] = pd.to_numeric(
                loaded[frame_name][year_column], errors="raise"
            ).astype(int)
        loaded["qrels"]["relevance"] = pd.to_numeric(
            loaded["qrels"]["relevance"], errors="raise"
        ).astype(float)
        for column in ("source_year", "target_year"):
            loaded["citation_edges"][column] = pd.to_numeric(
                loaded["citation_edges"][column], errors="raise"
            ).astype(int)

        benchmark = cls(
            papers=loaded["papers"],
            queries=loaded["queries"],
            qrels=loaded["qrels"],
            citation_edges=loaded["citation_edges"],
            metadata=manifest,
        )
        benchmark.validate()
        return benchmark


def build_temporal_citation_benchmark(
    csv_path: str,
    *,
    splits: Mapping[str, Sequence[int]],
    query_text_fields: Sequence[str] = ("title", "abstract", "keywords"),
) -> TemporalCitationBenchmark:
    """Construct query/qrel tables from strictly prior internal citations."""
    raw = pd.read_csv(csv_path)
    data_validation = validate_papers_dataframe(raw)
    raw = raw.reset_index(drop=True)
    raw["paper_id"] = raw["id"].astype(str)
    raw["year_numeric"] = pd.to_numeric(raw["year"], errors="coerce")

    papers = pd.DataFrame(
        {
            "paper_id": raw["paper_id"],
            "year": raw["year_numeric"],
            "title": raw["title"].fillna("").astype(str),
            "abstract": raw["abstract"].fillna("").astype(str),
            "keywords": raw["keywords"].fillna("").astype(str),
            "venue": raw["venue"].fillna("").astype(str),
        }
    )
    papers = papers[papers["year"].notna()].copy()
    papers["year"] = papers["year"].astype(int)
    papers["document"] = papers.apply(
        lambda row: _document(row, query_text_fields), axis=1
    )
    paper_year = dict(zip(papers["paper_id"], papers["year"]))

    split_by_year: Dict[int, str] = {}
    for split, years in splits.items():
        for year in years:
            year = int(year)
            if year in split_by_year:
                raise ValueError(f"Year {year} belongs to multiple splits")
            split_by_year[year] = split

    anomaly_counts = {
        "malformed_reference_rows": 0,
        "external_references": 0,
        "self_references": 0,
        "missing_year_edges": 0,
        "future_edges": 0,
        "same_year_edges": 0,
    }
    historical_edges = set()
    strict_qrels = set()

    for row in raw.itertuples(index=False):
        source_id = str(row.paper_id)
        source_year = row.year_numeric
        references = parse_list_field(row.references)
        if not references and str(row.references).strip() not in {"", "[]", "nan", "None"}:
            anomaly_counts["malformed_reference_rows"] += 1
        for target_id in set(str(value) for value in references):
            if target_id == source_id:
                anomaly_counts["self_references"] += 1
                continue
            if target_id not in paper_year:
                anomaly_counts["external_references"] += 1
                continue
            target_year = paper_year[target_id]
            if pd.isna(source_year):
                anomaly_counts["missing_year_edges"] += 1
                continue
            source_year = int(source_year)
            if target_year > source_year:
                anomaly_counts["future_edges"] += 1
                continue
            historical_edges.add((source_id, target_id, source_year, target_year))
            if target_year == source_year:
                anomaly_counts["same_year_edges"] += 1
                continue
            if source_year in split_by_year:
                strict_qrels.add((source_id, target_id, source_year, target_year))

    qrels = pd.DataFrame(
        sorted(strict_qrels),
        columns=["query_id", "candidate_id", "query_year", "target_year"],
    )
    if qrels.empty:
        raise ValueError("No strictly-prior internal citation qrels were found")
    qrels["relevance"] = 1.0
    qrels["split"] = qrels["query_year"].map(split_by_year)

    query_ids = set(qrels["query_id"])
    query_rows = raw[raw["paper_id"].isin(query_ids)].copy()
    queries = pd.DataFrame(
        {
            "query_id": query_rows["paper_id"],
            "query_year": query_rows["year_numeric"].astype(int),
            "split": query_rows["year_numeric"].astype(int).map(split_by_year),
            "title": query_rows["title"].fillna("").astype(str),
            "abstract": query_rows["abstract"].fillna("").astype(str),
            "keywords": query_rows["keywords"].fillna("").astype(str),
        }
    )
    queries["query_text"] = queries.apply(
        lambda row: _document(row, query_text_fields), axis=1
    )
    queries = queries.sort_values(["split", "query_year", "query_id"]).reset_index(drop=True)
    qrels = qrels[["query_id", "candidate_id", "relevance", "split", "query_year", "target_year"]]

    citation_edges = pd.DataFrame(
        sorted(historical_edges),
        columns=["source_id", "target_id", "source_year", "target_year"],
    )

    identity = {
        "dataset_sha256": sha256_file(csv_path),
        "splits": {key: [int(year) for year in years] for key, years in splits.items()},
        "candidate_policy": "strictly_earlier_year",
        "relevance_source": "internal_citation",
        "query_text_fields": list(query_text_fields),
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
    }
    metadata = {
        **identity,
        "benchmark_id": _canonical_hash(identity),
        "source_csv": os.path.abspath(csv_path),
        "data_validation": data_validation,
        "anomalies": anomaly_counts,
        "counts": {
            "papers": len(papers),
            "queries": len(queries),
            "qrels": len(qrels),
            "citation_edges": len(citation_edges),
            "queries_by_split": queries["split"].value_counts().to_dict(),
            "qrels_by_split": qrels["split"].value_counts().to_dict(),
        },
    }
    benchmark = TemporalCitationBenchmark(
        papers=papers.reset_index(drop=True),
        queries=queries,
        qrels=qrels.reset_index(drop=True),
        citation_edges=citation_edges.reset_index(drop=True),
        metadata=metadata,
    )
    benchmark.validate()
    return benchmark
