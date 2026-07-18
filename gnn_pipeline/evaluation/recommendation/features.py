"""Leakage-safe pair features for scholarly recommendation rerankers."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from gnn_pipeline.data.parsing import parse_authors

from .baselines import BM25Recommender, TfidfRecommender
from .embeddings import EmbeddingTable


FEATURE_NAMES = (
    "semantic_cosine",
    "tfidf_score",
    "bm25_score",
    "log_historical_indegree",
    "log_candidate_age",
    "log_author_productivity",
)


class TemporalFeatureStore:
    """Compute features solely from the query text and cutoff-visible history."""

    def __init__(
        self,
        benchmark,
        *,
        raw_csv_path: str,
        embedding_table: EmbeddingTable | None = None,
        include_tfidf: bool = True,
        include_bm25: bool = True,
        enabled_features: Sequence[str] | None = None,
    ):
        self.benchmark = benchmark
        self.embedding_table = embedding_table
        self.tfidf = TfidfRecommender() if include_tfidf else None
        self.bm25 = BM25Recommender() if include_bm25 else None
        self.enabled_features = set(enabled_features or FEATURE_NAMES)
        unknown = self.enabled_features - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown recommendation features: {sorted(unknown)}")
        raw = pd.read_csv(raw_csv_path, usecols=["id", "year", "authors"])
        raw["paper_id"] = raw["id"].astype(str)
        raw["year_numeric"] = pd.to_numeric(raw["year"], errors="coerce")
        self.paper_authors: Dict[str, List[str]] = {}
        for row in raw.itertuples(index=False):
            self.paper_authors[str(row.paper_id)] = [
                author["author_key"] for author in parse_authors(row.authors)
            ]
        self.paper_year = dict(zip(raw["paper_id"], raw["year_numeric"]))
        self._author_productivity: Dict[int, Dict[str, int]] = {}
        self._feature_cache: Dict[str, tuple[List[str], np.ndarray]] = {}

    def _productivity_at_cutoff(self, query_year: int) -> Dict[str, int]:
        if query_year not in self._author_productivity:
            counts: Dict[str, int] = {}
            for paper_id, authors in self.paper_authors.items():
                year = self.paper_year.get(paper_id)
                if pd.isna(year) or int(year) >= query_year:
                    continue
                for author in set(authors):
                    counts[author] = counts.get(author, 0) + 1
            self._author_productivity[query_year] = counts
        return self._author_productivity[query_year]

    def candidate_features(
        self,
        query: pd.Series,
        candidates: pd.DataFrame,
        *,
        visible_citation_edges: pd.DataFrame,
    ) -> np.ndarray:
        query_year = int(query["query_year"])
        candidate_ids = candidates["paper_id"].astype(str).tolist()
        cache_key = str(query["query_id"])
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            cached_ids, cached_matrix = cached
            if cached_ids != candidate_ids:
                raise ValueError("Candidate corpus changed for cached temporal features")
            return cached_matrix.copy()
        columns = []

        if self.embedding_table is not None:
            query_embedding = self.embedding_table.lookup([str(query["query_id"])])[0]
            candidate_embeddings = self.embedding_table.lookup(candidate_ids)
            query_norm = max(float(np.linalg.norm(query_embedding)), 1e-12)
            candidate_norms = np.maximum(
                np.linalg.norm(candidate_embeddings, axis=1), 1e-12
            )
            semantic = candidate_embeddings.dot(query_embedding) / (
                candidate_norms * query_norm
            )
        else:
            semantic = np.zeros(len(candidates), dtype=np.float32)
        columns.append(semantic)

        columns.append(
            self.tfidf.score_candidates(query, candidates)
            if self.tfidf is not None
            else np.zeros(len(candidates))
        )
        columns.append(
            self.bm25.score_candidates(query, candidates)
            if self.bm25 is not None
            else np.zeros(len(candidates))
        )

        indegree = visible_citation_edges["target_id"].astype(str).value_counts()
        columns.append(
            np.log1p(
                pd.Series(candidate_ids).map(indegree).fillna(0).to_numpy(dtype=float)
            )
        )
        columns.append(
            np.log1p(
                np.maximum(
                    query_year - candidates["year"].to_numpy(dtype=float), 0.0
                )
            )
        )

        productivity = self._productivity_at_cutoff(query_year)
        candidate_productivity = []
        for paper_id in candidate_ids:
            authors = self.paper_authors.get(paper_id, [])
            values = [productivity.get(author, 0) for author in authors]
            candidate_productivity.append(float(np.mean(values)) if values else 0.0)
        columns.append(np.log1p(np.asarray(candidate_productivity)))

        matrix = np.column_stack(columns).astype(np.float32)
        for index, feature_name in enumerate(FEATURE_NAMES):
            if feature_name not in self.enabled_features:
                matrix[:, index] = 0.0
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite temporal recommendation feature")
        self._feature_cache[cache_key] = (candidate_ids, matrix.copy())
        return matrix
