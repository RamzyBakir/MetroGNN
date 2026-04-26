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
import json
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "dblp_papers"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "models/vector_store_semantic"


class PaperVectorStore:
    """Manages ChromaDB-backed semantic search over DBLP papers."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        model_name: str = EMBED_MODEL_NAME,
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name

        # Lazy-loaded
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embed_model: Optional[SentenceTransformer] = None

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
    ) -> int:
        """
        Embed every paper and upsert into ChromaDB.

        Returns the number of papers ingested.
        """
        if not force_reingest and self.collection.count() > 0:
            print(
                f"Collection already has {self.collection.count()} papers. "
                "Use force_reingest=True to re-ingest."
            )
            return self.collection.count()

        if force_reingest:
            # Drop and recreate
            self.client.delete_collection(COLLECTION_NAME)
            self._collection = None  # force re-creation via property

        df = pd.read_csv(csv_path)
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
            documents, show_progress_bar=True, batch_size=batch_size
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

        print(
            f"Ingestion complete. Collection now has {self.collection.count()} papers."
        )
        return self.collection.count()

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

    parser = argparse.ArgumentParser(description="Paper Vector Store CLI")
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser("ingest", help="Ingest papers into the vector store")
    ingest_parser.add_argument(
        "--csv", default="data/dblp_with_topics_new.csv", help="Path to papers CSV"
    )
    ingest_parser.add_argument(
        "--force", action="store_true", help="Force re-ingestion"
    )

    query_parser = sub.add_parser("query", help="Query the vector store")
    query_parser.add_argument("query_text", type=str, help="Search query")
    query_parser.add_argument("--top-k", type=int, default=10, help="Top K results")

    args = parser.parse_args()

    store = PaperVectorStore()

    if args.command == "ingest":
        store.ingest_papers(csv_path=args.csv, force_reingest=args.force)

    elif args.command == "query":
        results = store.query(args.query_text, top_k=args.top_k)
        print(f"\nTop {len(results['paper_ids'])} results for: '{args.query_text}'\n")
        for i, (pid, dist, meta) in enumerate(
            zip(results["paper_ids"], results["distances"], results["metadatas"]), 1
        ):
            print(
                f"  {i:3}. [dist={dist:.4f}] (id={pid}) "
                f"{meta['title'][:80]}  ({meta['year']}, cit={meta['n_citation']})"
            )
