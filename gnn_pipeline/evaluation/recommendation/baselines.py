"""Leakage-aware recommendation baselines for the temporal benchmark."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .interfaces import RankedItem, Recommender


_TOKEN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]+\b")


def tokenize(text: str) -> List[str]:
    return [token.casefold() for token in _TOKEN.findall(str(text))]


def _rank_scores(
    candidate_ids: Sequence[str], scores: np.ndarray, top_k: int
) -> List[RankedItem]:
    frame = pd.DataFrame(
        {"paper_id": [str(value) for value in candidate_ids], "score": scores.astype(float)}
    )
    frame = frame.sort_values(
        ["score", "paper_id"], ascending=[False, True], kind="mergesort"
    ).head(top_k)
    return [
        RankedItem(str(row.paper_id), float(row.score))
        for row in frame.itertuples(index=False)
    ]


class RandomRecommender(Recommender):
    name = "random"
    stochastic = True

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        material = f"{seed}:{query['query_id']}".encode()
        query_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        rng = np.random.default_rng(query_seed)
        return _rank_scores(
            candidates["paper_id"].astype(str).tolist(),
            rng.random(len(candidates)),
            top_k,
        )


class RecencyRecommender(Recommender):
    name = "recency"

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        return _rank_scores(
            candidates["paper_id"].astype(str).tolist(),
            candidates["year"].to_numpy(dtype=float),
            top_k,
        )


class HistoricalPopularityRecommender(Recommender):
    """Rank by citation in-degree observable before the query year."""

    name = "popularity"

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        visible = visible_citation_edges
        if visible is None:
            raise ValueError("Historical popularity requires cutoff-visible citation edges")
        degree = visible["target_id"].astype(str).value_counts()
        candidate_ids = candidates["paper_id"].astype(str)
        scores = candidate_ids.map(degree).fillna(0).to_numpy(dtype=float)
        return _rank_scores(candidate_ids.tolist(), scores, top_k)


class TfidfRecommender(Recommender):
    name = "tfidf"

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._cache: Dict[int, Tuple[List[str], TfidfVectorizer, object]] = {}

    def _index(self, query_year: int, candidates: pd.DataFrame):
        if query_year not in self._cache:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(candidates["document"].fillna("").astype(str))
            self._cache[query_year] = (
                candidates["paper_id"].astype(str).tolist(),
                vectorizer,
                matrix,
            )
        ids, vectorizer, matrix = self._cache[query_year]
        current_ids = candidates["paper_id"].astype(str).tolist()
        if ids != current_ids:
            raise ValueError("Candidate corpus changed for an already-indexed year")
        return ids, vectorizer, matrix

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        ids, vectorizer, matrix = self._index(int(query["query_year"]), candidates)
        query_vector = vectorizer.transform([str(query["query_text"])])
        scores = (matrix @ query_vector.T).toarray().ravel()
        return _rank_scores(ids, scores, top_k)


class BM25Recommender(Recommender):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._cache: Dict[int, Dict] = {}

    def _index(self, query_year: int, candidates: pd.DataFrame) -> Dict:
        if query_year not in self._cache:
            token_counts = [Counter(tokenize(text)) for text in candidates["document"]]
            lengths = np.asarray([sum(counts.values()) for counts in token_counts], dtype=float)
            document_frequency = Counter()
            for counts in token_counts:
                document_frequency.update(counts.keys())
            count = len(token_counts)
            idf = {
                term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
                for term, frequency in document_frequency.items()
            }
            self._cache[query_year] = {
                "ids": candidates["paper_id"].astype(str).tolist(),
                "counts": token_counts,
                "lengths": lengths,
                "avg_length": float(lengths.mean()) if count else 0.0,
                "idf": idf,
            }
        index = self._cache[query_year]
        if index["ids"] != candidates["paper_id"].astype(str).tolist():
            raise ValueError("Candidate corpus changed for an already-indexed year")
        return index

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        index = self._index(int(query["query_year"]), candidates)
        scores = np.zeros(len(index["ids"]), dtype=float)
        avg_length = max(index["avg_length"], 1.0)
        for term in set(tokenize(query["query_text"])):
            idf = index["idf"].get(term)
            if idf is None:
                continue
            frequencies = np.fromiter(
                (counts.get(term, 0) for counts in index["counts"]),
                dtype=float,
                count=len(index["counts"]),
            )
            denominator = frequencies + self.k1 * (
                1.0 - self.b + self.b * index["lengths"] / avg_length
            )
            scores += idf * (frequencies * (self.k1 + 1.0)) / np.maximum(
                denominator, 1e-12
            )
        return _rank_scores(index["ids"], scores, top_k)


class SentenceEmbeddingRecommender(Recommender):
    """Optional neural baseline; model loading and encoding are strictly lazy."""

    def __init__(self, model_name: str, name: str, text_mode: str = "document"):
        self.model_name = model_name
        self.name = name
        self.text_mode = text_mode
        self._model = None
        self._cache: Dict[int, Tuple[List[str], np.ndarray]] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _candidate_texts(self, candidates: pd.DataFrame) -> List[str]:
        if self.text_mode == "title_abstract_sep":
            return (
                candidates["title"].fillna("").astype(str)
                + " [SEP] "
                + candidates["abstract"].fillna("").astype(str)
            ).tolist()
        return candidates["document"].fillna("").astype(str).tolist()

    def _query_text(self, query: pd.Series) -> str:
        if self.text_mode == "title_abstract_sep":
            return f"{query.get('title', '')} [SEP] {query.get('abstract', '')}"
        return str(query["query_text"])

    def _index(self, query_year: int, candidates: pd.DataFrame):
        if query_year not in self._cache:
            embeddings = self.model.encode(
                self._candidate_texts(candidates),
                batch_size=256,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            self._cache[query_year] = (
                candidates["paper_id"].astype(str).tolist(),
                np.asarray(embeddings, dtype=np.float32),
            )
        ids, embeddings = self._cache[query_year]
        if ids != candidates["paper_id"].astype(str).tolist():
            raise ValueError("Candidate corpus changed for an already-indexed year")
        return ids, embeddings

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        ids, embeddings = self._index(int(query["query_year"]), candidates)
        query_embedding = self.model.encode(
            [self._query_text(query)], normalize_embeddings=True
        )[0]
        return _rank_scores(ids, embeddings @ query_embedding, top_k)


class SemanticSeededPPRRecommender(Recommender):
    """Personalized PageRank seeded by leakage-free TF-IDF retrieval."""

    name = "seeded_ppr"

    def __init__(self, seed_count: int = 20, alpha: float = 0.85):
        self.seed_count = int(seed_count)
        self.alpha = float(alpha)
        self.semantic = TfidfRecommender()
        self._graphs: Dict[int, nx.DiGraph] = {}

    def _graph(
        self,
        query_year: int,
        candidates: pd.DataFrame,
        visible_citation_edges: pd.DataFrame,
    ) -> nx.DiGraph:
        if query_year not in self._graphs:
            ids = set(candidates["paper_id"].astype(str))
            visible = visible_citation_edges
            graph = nx.DiGraph()
            graph.add_edges_from(
                (str(row.source_id), str(row.target_id))
                for row in visible.itertuples(index=False)
                if str(row.source_id) in ids and str(row.target_id) in ids
            )
            self._graphs[query_year] = graph
        return self._graphs[query_year]

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        if visible_citation_edges is None:
            raise ValueError("Seeded PPR requires cutoff-visible citation edges")
        semantic_seeds = self.semantic.rank(
            query, candidates, max(self.seed_count, top_k), seed
        )
        if not semantic_seeds:
            return []
        personalization = {
            item.paper_id: max(item.score, 0.0) + 1e-9
            for item in semantic_seeds[: self.seed_count]
        }
        graph = self._graph(
            int(query["query_year"]), candidates, visible_citation_edges
        ).copy()
        graph.add_nodes_from(personalization)
        scores = nx.pagerank(
            graph,
            alpha=self.alpha,
            personalization=personalization,
            dangling=personalization,
        )
        semantic_scores = {item.paper_id: item.score for item in semantic_seeds}
        ids = candidates["paper_id"].astype(str).tolist()
        combined = np.asarray(
            [scores.get(item, 0.0) + 1e-6 * semantic_scores.get(item, 0.0) for item in ids]
        )
        return _rank_scores(ids, combined, top_k)


def make_baseline(name: str, config: Dict | None = None) -> Recommender:
    config = config or {}
    if name == "random":
        return RandomRecommender()
    if name == "popularity":
        return HistoricalPopularityRecommender()
    if name == "recency":
        return RecencyRecommender()
    if name == "tfidf":
        tfidf = config.get("tfidf", {})
        return TfidfRecommender(
            max_features=int(tfidf.get("max_features", 50000)),
            ngram_range=tuple(tfidf.get("ngram_range", [1, 2])),
        )
    if name == "bm25":
        bm25 = config.get("bm25", {})
        return BM25Recommender(k1=bm25.get("k1", 1.5), b=bm25.get("b", 0.75))
    if name == "seeded_ppr":
        ppr = config.get("seeded_ppr", {})
        return SemanticSeededPPRRecommender(
            seed_count=ppr.get("semantic_seed_count", 20),
            alpha=ppr.get("alpha", 0.85),
        )
    if name in {"minilm", "scholarly"}:
        models = config.get("neural", {}).get("models", {})
        model_name = models.get(name)
        if not model_name:
            raise ValueError(f"No embedding model configured for {name}")
        return SentenceEmbeddingRecommender(
            model_name=model_name,
            name=name,
            text_mode="title_abstract_sep" if name == "scholarly" else "document",
        )
    raise KeyError(f"Unknown recommendation baseline: {name}")
