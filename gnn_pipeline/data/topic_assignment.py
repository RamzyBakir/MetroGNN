"""
Topic assignment using sentence embeddings (all-MiniLM-L6-v2) + K-Means.

Replaces the old TF-IDF + K-Means approach with dense 384-dim embeddings
from a pretrained sentence transformer, producing much better semantic
clusters.
"""

import ast
import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TopicAssigner:
    """Assign papers to topics using sentence-embedding K-Means clustering."""

    def __init__(self, n_clusters: int = 8, model_name: str = EMBED_MODEL_NAME):
        self.n_clusters = n_clusters
        self.model_name = model_name
        self._embed_model: Optional[SentenceTransformer] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers_: Optional[np.ndarray] = None

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
            documents, show_progress_bar=True, batch_size=batch_size
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

    from gnn_pipeline.config import get_path

    default_csv = get_path("data", "csv_path")

    parser = argparse.ArgumentParser(description="Assign topics to DBLP papers")
    parser.add_argument("--csv", default=default_csv, help="Input CSV")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of topics")
    parser.add_argument("--auto-k", action="store_true", help="Auto-find best k")
    parser.add_argument(
        "--out-csv", default=None, help="Output CSV (default: overwrite input)"
    )
    parser.add_argument(
        "--out-topic-info",
        default=None,
        help="Output topic info JSON (default: models/topic_info_new.json)",
    )
    args = parser.parse_args()

    if args.out_topic_info is None:
        import os

        args.out_topic_info = os.path.join(
            os.path.dirname(get_path("models", "dir")), "topic_info_new.json"
        )

    try:
        df = pd.read_csv(args.csv, on_bad_lines="skip")
    except UnicodeDecodeError:
        print(
            f"File {args.csv} could not be read as UTF-8. Retrying with latin-1 encoding..."
        )
        df = pd.read_csv(args.csv, encoding="latin-1", on_bad_lines="skip")
    df_new, results, topic_info = assign_topics_to_dblp(
        df, n_clusters=args.n_clusters, auto_k=args.auto_k
    )

    out_csv = args.out_csv or args.csv
    df_new.to_csv(out_csv, index=False)
    print(f"\nSaved topics to: {out_csv}")

    with open(args.out_topic_info, "w") as f:
        json.dump({str(k): v for k, v in topic_info.items()}, f, indent=2)
    print(f"Saved topic info to: {args.out_topic_info}")

    print("\n── Topic Summary ──")
    for tid, info in sorted(topic_info.items()):
        print(f"  Topic {tid}: {info['name']}  ({info['size']} papers)")
        print(f"    Top terms: {', '.join(info['top_terms'][:5])}")
