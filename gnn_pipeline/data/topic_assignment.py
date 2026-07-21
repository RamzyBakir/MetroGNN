"""
Topic assignment using sentence embeddings (all-MiniLM-L6-v2) + K-Means.

Replaces the old TF-IDF + K-Means approach with dense 384-dim embeddings
from a pretrained sentence transformer, producing much better semantic
clusters.
"""

import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _show_progress_bars() -> bool:
    return os.environ.get("TQDM_DISABLE", "").lower() not in {"1", "true", "yes"}


class TopicAssigner:
    """Assign papers to topics using sentence-embedding K-Means clustering."""

    def __init__(self, n_clusters: int = 8, model_name: str = EMBED_MODEL_NAME):
        self.n_clusters = n_clusters
        self.model_name = model_name
        self._embed_model: Optional[SentenceTransformer] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def save_centroids(self, path: str, dataset_sha256: str = "") -> None:
        """Persist query-time topic centroids without pickling executable objects."""
        if self.cluster_centers_ is None:
            raise RuntimeError("TopicAssigner has no fitted cluster centers")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            cluster_centers=self.cluster_centers_,
            n_clusters=np.array([self.n_clusters], dtype=np.int64),
            model_name=np.array([self.model_name]),
            dataset_sha256=np.array([dataset_sha256]),
        )

    @classmethod
    def from_centroids(cls, path: str) -> "TopicAssigner":
        """Load a query-time assigner from a safe NumPy archive."""
        with np.load(path, allow_pickle=False) as payload:
            model_name = str(payload["model_name"][0])
            n_clusters = int(payload["n_clusters"][0])
            centers = payload["cluster_centers"].astype(np.float32)
        assigner = cls(n_clusters=n_clusters, model_name=model_name)
        assigner.cluster_centers_ = centers
        return assigner

    def rank_query_topics(self, query: str, top_k: int = 2) -> List[Tuple[int, float]]:
        """Rank fitted topics for a query using embedding cosine similarity."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Topic centroids are not loaded")
        text = str(query).strip()
        if not text:
            return []
        query_embedding = np.asarray(self.embed_model.encode([text]))[0]
        centers = np.asarray(self.cluster_centers_)
        query_norm = np.linalg.norm(query_embedding)
        center_norms = np.linalg.norm(centers, axis=1)
        denominators = np.maximum(query_norm * center_norms, 1e-12)
        scores = centers.dot(query_embedding) / denominators
        indices = np.argsort(scores)[::-1][: max(1, min(top_k, len(scores)))]
        return [(int(idx), float(scores[idx])) for idx in indices]

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    # ── text helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_keywords(kw_raw) -> List[str]:
        if pd.isna(kw_raw) or not str(kw_raw).strip():
            return []
        try:
            kw_list = ast.literal_eval(str(kw_raw))
            if isinstance(kw_list, list):
                return [str(k).strip() for k in kw_list if str(k).strip()]
        except Exception:
            pass
        return []

    @staticmethod
    def _build_document(row: pd.Series) -> str:
        """Combine title + venue + keywords into one string for embedding."""
        parts = []
        title = str(row.get("title", "")).strip()
        if title and title.lower() != "nan":
            parts.append(title)

        venue = str(row.get("venue", "")).strip()
        if venue and venue.lower() != "nan":
            parts.append(venue)

        kws = TopicAssigner._parse_keywords(row.get("keywords", ""))
        if kws:
            parts.append(", ".join(kws))

        return " ".join(parts) if parts else "unknown"

    # ── clustering ──────────────────────────────────────────────────────

    def _find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (4, 16),
    ) -> int:
        """Pick the k with the highest silhouette score."""
        best_k, best_score = self.n_clusters, -1.0
        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
        print(f"  Best k={best_k}  (silhouette={best_score:.3f})")
        return best_k

    def _generate_topic_name(self, titles: List[str], keywords_flat: List[str]) -> str:
        """Heuristic label from top keywords and sample titles."""
        # Most common keywords in the cluster
        kw_counts = Counter(kw.lower() for kw in keywords_flat)
        top_kws = [kw for kw, _ in kw_counts.most_common(5)]

        # Try to find a concise label from top keywords
        if top_kws:
            label = " & ".join(top_kws[:3]).title()
        else:
            # Fall back to most common words from titles
            words = []
            stopwords = {
                "a",
                "an",
                "the",
                "of",
                "for",
                "and",
                "in",
                "on",
                "to",
                "with",
                "by",
                "from",
                "is",
                "are",
                "was",
                "using",
                "based",
                "via",
                "its",
                "their",
                "as",
                "at",
                "or",
                "that",
                "this",
            }
            for t in titles:
                words.extend(
                    w.lower()
                    for w in re.findall(r"\b[a-zA-Z]{3,}\b", t)
                    if w.lower() not in stopwords
                )
            word_counts = Counter(words)
            top_words = [w for w, _ in word_counts.most_common(3)]
            label = " & ".join(top_words).title() if top_words else "Miscellaneous"
        return label

    # ── main API ────────────────────────────────────────────────────────

    def assign_topics(
        self,
        df: pd.DataFrame,
        auto_k: bool = False,
        batch_size: int = 256,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Assign topics to papers via sentence-embedding K-Means.

        Args:
            df: DataFrame with at least 'title' column
            auto_k: If True, search for the optimal number of clusters
            batch_size: Embedding batch size

        Returns:
            df_with_topics: Copy of df with 'topic_id' and 'topic_name' added
            results_dict: Clustering stats
            topic_info_dict: {topic_id: {topic_id, name, top_terms, sample_titles}}
        """
        # 1. Build document strings
        documents = [self._build_document(row) for _, row in df.iterrows()]
        print(f"Embedding {len(documents)} papers ...")

        # 2. Encode
        embeddings = self.embed_model.encode(
            documents, show_progress_bar=_show_progress_bars(), batch_size=batch_size
        )

        # 3. Optionally pick best k
        if auto_k:
            self.n_clusters = self._find_optimal_k(embeddings)

        # 4. K-Means on 384-dim embeddings
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300
        )
        labels = self.kmeans.fit_predict(embeddings)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        sil = silhouette_score(embeddings, labels) if self.n_clusters > 1 else 0.0
        print(f"Clustered into {self.n_clusters} topics  (silhouette={sil:.3f})")

        # 5. Build topic info
        df_tmp = df.copy()
        df_tmp["topic_id"] = labels

        topic_info_dict: Dict[int, Dict] = {}
        for tid in range(self.n_clusters):
            mask = labels == tid
            cluster_rows = df.iloc[np.where(mask)[0]]
            sample_titles = cluster_rows["title"].dropna().head(10).tolist()

            # Collect all keywords for this cluster
            all_kws: List[str] = []
            for _, row in cluster_rows.iterrows():
                all_kws.extend(self._parse_keywords(row.get("keywords", "")))

            name = self._generate_topic_name(sample_titles, all_kws)

            # Top terms = most frequent keywords
            kw_counts = Counter(kw.lower() for kw in all_kws)
            top_terms = [kw for kw, _ in kw_counts.most_common(10)]

            topic_info_dict[tid] = {
                "topic_id": tid,
                "name": name,
                "top_terms": top_terms,
                "sample_titles": sample_titles[:5],
                "size": int(mask.sum()),
            }

        # 6. Attach to dataframe
        df_with_topics = df.copy()
        df_with_topics["topic_id"] = labels
        topic_names = {tid: info["name"] for tid, info in topic_info_dict.items()}
        df_with_topics["topic_name"] = df_with_topics["topic_id"].map(topic_names)

        results_dict = {
            "n_clusters": self.n_clusters,
            "silhouette_score": sil,
            "topic_distribution": df_with_topics["topic_id"].value_counts().to_dict(),
            "topic_info": topic_info_dict,
        }

        return df_with_topics, results_dict, topic_info_dict


# ── convenience function ────────────────────────────────────────────────


def save_topic_artifacts(
    df: pd.DataFrame,
    topic_info: Dict[int, Dict],
    assigner: TopicAssigner,
    *,
    assignments_path: str,
    topic_info_path: str,
    centroids_path: str,
    dataset_sha256: str = "",
) -> None:
    """Persist stable per-paper assignments, topic metadata, and centroids."""
    if "topic_id" not in df.columns:
        raise ValueError("DataFrame has no topic_id column")
    if "id" not in df.columns or df["id"].isna().any() or not df["id"].is_unique:
        raise ValueError("Stable, unique paper IDs are required for topic persistence")

    assignments = pd.DataFrame(
        {
            "paper_id": df["id"].astype(str).to_numpy(),
            "row_index": np.arange(len(df), dtype=np.int64),
            "topic_id": df["topic_id"].astype(int).to_numpy(),
            "topic_name": df.get(
                "topic_name", pd.Series([""] * len(df), index=df.index)
            ).astype(str).to_numpy(),
        }
    )
    output = Path(assignments_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output, index=False)

    info_output = Path(topic_info_path)
    info_output.parent.mkdir(parents=True, exist_ok=True)
    info_output.write_text(
        json.dumps({str(k): v for k, v in topic_info.items()}, indent=2, ensure_ascii=False)
    )
    assigner.save_centroids(centroids_path, dataset_sha256=dataset_sha256)


def load_topic_artifacts(
    df: pd.DataFrame,
    *,
    assignments_path: str,
    topic_info_path: str,
    centroids_path: str,
    expected_n_clusters: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict], TopicAssigner]:
    """Load and strictly validate persisted topic artifacts."""
    for path in (assignments_path, topic_info_path, centroids_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required topic artifact not found: {path}")

    assignments = pd.read_csv(assignments_path, dtype={"paper_id": str})
    required = {"paper_id", "row_index", "topic_id"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Topic assignments are missing columns: {required - set(assignments.columns)}")
    if len(assignments) != len(df):
        raise ValueError("Topic assignments have a different paper count than the CSV")
    expected_ids = df["id"].astype(str).tolist()
    if assignments["paper_id"].tolist() != expected_ids:
        raise ValueError("Topic assignments do not match the ordered CSV paper IDs")

    raw_info = json.loads(Path(topic_info_path).read_text())
    topic_info = {int(k): v for k, v in raw_info.items()}
    assigner = TopicAssigner.from_centroids(centroids_path)
    if expected_n_clusters is not None and assigner.n_clusters != expected_n_clusters:
        raise ValueError(
            f"Cached topics use k={assigner.n_clusters}, expected k={expected_n_clusters}"
        )
    expected_topic_ids = set(range(assigner.n_clusters))
    if set(topic_info) != expected_topic_ids:
        raise ValueError("Topic metadata IDs do not match the persisted centroids")
    assigned_topic_ids = set(assignments["topic_id"].astype(int).tolist())
    if not assigned_topic_ids.issubset(expected_topic_ids):
        raise ValueError("Topic assignments contain an out-of-range topic ID")

    result = df.copy()
    result["topic_id"] = assignments["topic_id"].astype(int).to_numpy()
    names = {tid: info.get("name", f"Topic {tid}") for tid, info in topic_info.items()}
    result["topic_name"] = result["topic_id"].map(names)
    if result["topic_id"].nunique() < 2 and len(result) > 1:
        raise ValueError("Cached topic assignments populate fewer than two topics")
    return result, topic_info, assigner


def prepare_topics(
    df: pd.DataFrame,
    *,
    n_clusters: int,
    assignments_path: str,
    topic_info_path: str,
    centroids_path: str,
    dataset_sha256: str,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, Dict], TopicAssigner, bool]:
    """Load compatible topic artifacts or explicitly generate a complete set."""
    artifact_paths = (assignments_path, topic_info_path, centroids_path)
    existing_artifacts = [os.path.isfile(path) for path in artifact_paths]
    artifacts_exist = all(existing_artifacts)
    if any(existing_artifacts) and not artifacts_exist and not force_regenerate:
        missing = [
            path for path, exists in zip(artifact_paths, existing_artifacts) if not exists
        ]
        raise ValueError(
            f"Incomplete topic artifact set; missing {missing}. "
            "Regenerate topics explicitly."
        )
    if artifacts_exist and not force_regenerate:
        loaded_df, topic_info, assigner = load_topic_artifacts(
            df,
            assignments_path=assignments_path,
            topic_info_path=topic_info_path,
            centroids_path=centroids_path,
            expected_n_clusters=n_clusters,
        )
        with np.load(centroids_path, allow_pickle=False) as payload:
            cached_sha = str(payload["dataset_sha256"][0])
        if cached_sha != dataset_sha256:
            raise ValueError(
                "Topic artifacts were generated from a different dataset. "
                "Regenerate them explicitly."
            )
        return loaded_df, topic_info, assigner, False

    assigner = TopicAssigner(n_clusters=n_clusters)
    assigned_df, _, topic_info = assigner.assign_topics(df)
    save_topic_artifacts(
        assigned_df,
        topic_info,
        assigner,
        assignments_path=assignments_path,
        topic_info_path=topic_info_path,
        centroids_path=centroids_path,
        dataset_sha256=dataset_sha256,
    )
    return assigned_df, topic_info, assigner, True


def assign_topics_to_dblp(
    df: pd.DataFrame, n_clusters: int = 8, auto_k: bool = False
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Convenience wrapper: assign topics using sentence embeddings.

    Args:
        df: DataFrame with DBLP papers
        n_clusters: Default number of topics
        auto_k: If True, automatically find the best k via silhouette
    """
    assigner = TopicAssigner(n_clusters=n_clusters)
    return assigner.assign_topics(df, auto_k=auto_k)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from gnn_pipeline.artifacts import sha256_file
    from gnn_pipeline.config import get_path, get_value

    parser = argparse.ArgumentParser(description="Generate versioned paper topic artifacts")
    parser.add_argument("--csv", default=get_path("data", "csv_path"), help="Input CSV")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=get_value("training", "n_clusters"),
        help="Number of topics",
    )
    parser.add_argument("--auto-k", action="store_true", help="Auto-find best k")
    parser.add_argument("--out-csv", default=None, help="Optional derived CSV; source is never overwritten")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    assigner = TopicAssigner(n_clusters=args.n_clusters)
    df_new, results, topic_info = assigner.assign_topics(df, auto_k=args.auto_k)
    save_topic_artifacts(
        df_new,
        topic_info,
        assigner,
        assignments_path=get_path("data", "topic_assignments_path"),
        topic_info_path=get_path("data", "topic_json_path"),
        centroids_path=get_path("data", "topic_centroids_path"),
        dataset_sha256=sha256_file(args.csv),
    )
    if args.out_csv:
        df_new.to_csv(args.out_csv, index=False)
        print(f"Saved derived CSV to: {args.out_csv}")

    print("\nTopic Summary")
    for tid, info in sorted(topic_info.items()):
        print(f"  Topic {tid}: {info['name']} ({info['size']} papers)")
