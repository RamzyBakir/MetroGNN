"""
Vector store for semantic paper retrieval using ChromaDB + all-MiniLM-L6-v2.

Usage:
    # Ingest papers (one-time)
    store = PaperVectorStore()
    store.ingest_papers("data/dblp_with_topics_new.csv")

    # Query
    results = store.query("graph neural networks", top_k=50)
"""

import os
import ast
import gc
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from gnn_pipeline.artifacts import (
    ArtifactCompatibilityError,
    VECTOR_STORE_SCHEMA_VERSION,
    sha256_file,
    write_json_atomic,
)
from gnn_pipeline.data.validation import ensure_paper_identity_columns


COLLECTION_NAME = "dblp_papers"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "models/vector_store_semantic_v2"


def _show_progress_bars() -> bool:
    return os.environ.get("TQDM_DISABLE", "").lower() not in {"1", "true", "yes"}


class PaperVectorStore:
    """Manages ChromaDB-backed semantic search over DBLP papers."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        model_name: str = EMBED_MODEL_NAME,
        manifest_path: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.manifest_path = manifest_path or os.path.join(
            persist_dir, "vector_store_manifest.json"
        )

        # Lazy-loaded
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embed_model: Optional[SentenceTransformer] = None
        self._validated_signature = None

    def close(self) -> None:
        """Release Chroma resources so persistent files can be removed on Windows."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._collection = None
        self._client = None
        gc.collect()

    def __enter__(self) -> "PaperVectorStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties (lazy init)
    # ------------------------------------------------------------------

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_document_text(row: pd.Series) -> str:
        """Combine title + venue + keywords into a single search document."""
        parts = []
        title = str(row.get("title", "")).strip()
        if title:
            parts.append(title)

        venue = str(row.get("venue", "")).strip()
        if venue and venue.lower() != "nan":
            parts.append(venue)

        # Try to parse keywords
        kw_raw = row.get("keywords", "")
        if pd.notna(kw_raw) and str(kw_raw).strip():
            try:
                kw_list = ast.literal_eval(str(kw_raw))
                if isinstance(kw_list, list):
                    parts.append(" ".join(str(k) for k in kw_list))
            except Exception:
                parts.append(str(kw_raw))

        return " ".join(parts)

    def ingest_papers(
        self,
        csv_path: str = "data/dblp_with_topics_new.csv",
        batch_size: int = 256,
        force_reingest: bool = False,
        topic_assignments_path: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ) -> int:
        """
        Embed every paper and upsert into ChromaDB.

        Returns the number of papers ingested.
        """
        if not force_reingest and self.collection.count() > 0:
            if topic_assignments_path:
                self.validate_compatibility(
                    csv_path=csv_path,
                    topic_assignments_path=topic_assignments_path,
                    pipeline_id=pipeline_id,
                )
            print(
                f"Collection already has {self.collection.count()} papers. "
                "Use force_reingest=True to re-ingest."
            )
            return self.collection.count()

        if force_reingest:
            # Drop and recreate
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception as exc:
                if exc.__class__.__name__ != "NotFoundError":
                    raise
            self._collection = None  # force re-creation via property

        df = ensure_paper_identity_columns(pd.read_csv(csv_path))
        if topic_assignments_path:
            assignments = pd.read_csv(topic_assignments_path, dtype={"paper_id": str})
            if len(assignments) != len(df) or assignments["paper_id"].tolist() != df["id"].astype(str).tolist():
                raise ArtifactCompatibilityError(
                    "Topic assignments do not match the vector-store CSV"
                )
            df["topic_id"] = assignments["topic_id"].astype(int).to_numpy()
        elif "topic_id" not in df.columns:
            raise ArtifactCompatibilityError(
                "Versioned topic assignments are required for vector ingestion"
            )
        print(f"Ingesting {len(df)} papers from {csv_path} ...")

        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict] = []

        for idx, row in df.iterrows():
            doc_text = self._build_document_text(row)
            documents.append(doc_text)
            ids.append(str(idx))  # Paper index == graph node id

            def safe_int(val, default=0):
                if pd.isna(val):
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            metadatas.append(
                {
                    "title": str(row.get("title", "")),
                    "year": safe_int(row.get("year")),
                    "n_citation": safe_int(row.get("n_citation")),
                    "venue": str(row.get("venue", "")),
                    "topic_id": safe_int(row.get("topic_id"), default=-1),
                }
            )

        # Embed in batches
        print("Encoding documents with SentenceTransformer ...")
        all_embeddings = self.embed_model.encode(
            documents, show_progress_bar=_show_progress_bars(), batch_size=batch_size
        )

        # Upsert into ChromaDB in chunks (ChromaDB limit per call)
        chroma_batch = 5000
        total_upserted = 0
        for start in range(0, len(ids), chroma_batch):
            end = min(start + chroma_batch, len(ids))
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=[emb.tolist() for emb in all_embeddings[start:end]],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            total_upserted += end - start
            print(f"  Upserted {total_upserted}/{len(ids)} papers")

        final_count = self.collection.count()
        manifest = {
            "vector_store_schema_version": VECTOR_STORE_SCHEMA_VERSION,
            "pipeline_id": pipeline_id,
            "dataset_sha256": sha256_file(csv_path),
            "topic_assignments_sha256": sha256_file(topic_assignments_path)
            if topic_assignments_path
            else None,
            "embedding_model": self.model_name,
            "collection_name": COLLECTION_NAME,
            "paper_count": final_count,
        }
        write_json_atomic(self.manifest_path, manifest)
        print(f"Ingestion complete. Collection now has {final_count} papers.")
        return final_count

    def validate_compatibility(
        self,
        *,
        csv_path: str,
        topic_assignments_path: str,
        pipeline_id: Optional[str] = None,
    ) -> Dict:
        """Reject a stale vector index before serving recommendations."""
        if not os.path.isfile(self.manifest_path):
            raise ArtifactCompatibilityError(
                f"Vector-store manifest not found: {self.manifest_path}"
            )
        signature = (
            os.path.getmtime(self.manifest_path),
            os.path.getmtime(csv_path),
            os.path.getsize(csv_path),
            os.path.getmtime(topic_assignments_path),
            os.path.getsize(topic_assignments_path),
            pipeline_id,
            self.collection.count(),
        )
        if self._validated_signature == signature:
            import json

            return json.loads(Path(self.manifest_path).read_text())

        import json

        manifest = json.loads(Path(self.manifest_path).read_text())
        if manifest.get("vector_store_schema_version") != VECTOR_STORE_SCHEMA_VERSION:
            raise ArtifactCompatibilityError("Unsupported vector-store schema")
        checks = {
            "dataset_sha256": sha256_file(csv_path),
            "topic_assignments_sha256": sha256_file(topic_assignments_path),
            "embedding_model": self.model_name,
            "paper_count": self.collection.count(),
        }
        for key, actual in checks.items():
            if manifest.get(key) != actual:
                raise ArtifactCompatibilityError(
                    f"Vector store is incompatible: {key} does not match"
                )
        if pipeline_id and manifest.get("pipeline_id") != pipeline_id:
            raise ArtifactCompatibilityError(
                "Vector store and GNN graph belong to different pipelines"
            )
        self._validated_signature = signature
        return manifest

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 50,
        interests: Optional[List[str]] = None,
    ) -> Dict:
        """
        Embed the query, retrieve top-K papers from the vector store.

        Args:
            query_text: Free-text user query
            top_k: Number of papers to retrieve
            interests: Optional list of interest strings to append to query

        Returns:
            Dict with keys:
                paper_ids: list[int] — graph-level paper node IDs
                distances: list[float] — cosine distances (lower=more similar)
                metadatas: list[dict] — per-paper metadata
                documents: list[str] — the stored document texts
        """
        combined = query_text
        if interests:
            combined += " " + " ".join(interests)

        if self.collection.count() == 0:
            return {"paper_ids": [], "distances": [], "metadatas": [], "documents": []}

        query_embedding = self.embed_model.encode([combined]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["distances", "metadatas", "documents"],
        )

        paper_ids = [int(pid) for pid in results["ids"][0]]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        return {
            "paper_ids": paper_ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
        }


# ------------------------------------------------------------------
# Convenience CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from gnn_pipeline.artifacts import validate_pipeline_manifest
    from gnn_pipeline.config import get_path

    parser = argparse.ArgumentParser(description="Paper Vector Store CLI")
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser("ingest", help="Ingest papers into the vector store")
    ingest_parser.add_argument(
        "--csv", default=get_path("data", "csv_path"), help="Path to papers CSV"
    )
    ingest_parser.add_argument(
        "--force", action="store_true", help="Force re-ingestion"
    )

    query_parser = sub.add_parser("query", help="Query the vector store")
    query_parser.add_argument("query_text", type=str, help="Search query")
    query_parser.add_argument("--top-k", type=int, default=10, help="Top K results")

    args = parser.parse_args()

    pipeline_manifest = validate_pipeline_manifest(
        get_path("models", "pipeline_manifest"),
        csv_path=get_path("data", "csv_path"),
    )
    store = PaperVectorStore(
        persist_dir=get_path("models", "vector_store"),
        manifest_path=get_path("models", "vector_store_manifest"),
    )

    if args.command == "ingest":
        store.ingest_papers(
            csv_path=args.csv,
            force_reingest=args.force,
            topic_assignments_path=get_path("data", "topic_assignments_path"),
            pipeline_id=pipeline_manifest["pipeline_id"],
        )

    elif args.command == "query":
        store.validate_compatibility(
            csv_path=get_path("data", "csv_path"),
            topic_assignments_path=get_path("data", "topic_assignments_path"),
            pipeline_id=pipeline_manifest["pipeline_id"],
        )
        results = store.query(args.query_text, top_k=args.top_k)
        print(f"\nTop {len(results['paper_ids'])} results for: '{args.query_text}'\n")
        for i, (pid, dist, meta) in enumerate(
            zip(results["paper_ids"], results["distances"], results["metadatas"]), 1
        ):
            print(
                f"  {i:3}. [dist={dist:.4f}] (id={pid}) "
                f"{meta['title'][:80]}  ({meta['year']}, cit={meta['n_citation']})"
            )
