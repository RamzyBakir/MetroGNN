"""
Inference pipeline: Query → Embed → Vector DB Top-K → GNN Lookup → Results

The GNN runs once on the **full** graph (cached).  The vector store selects
which papers are relevant, and then we simply look up those papers' authors
and their pre-computed GNN scores.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gnn_pipeline.artifacts import validate_pipeline_manifest
from gnn_pipeline.config import get_path
from gnn_pipeline.data.parsing import load_author_mapping
from gnn_pipeline.data.vector_store import PaperVectorStore
from gnn_pipeline.models.checkpoint import load_model_for_pipeline
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN

# ── Label maps ──────────────────────────────────────────────────────────
IMPORTANCE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ROLE_LABELS = {0: "Leader", 1: "Core", 2: "Contributor", 3: "Sole Expert"}


class GNNInferencer:
    """Load the full graph + model once, run GNN on the full graph (cached),
    then slice results per query."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        papers_csv: Optional[str] = None,
        vector_store_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or get_path("models", "best_model_new")
        self.graph_path = graph_path or get_path("models", "knowledge_graph")
        self.papers_csv = papers_csv or get_path("data", "csv_path")
        self.vector_store_dir = vector_store_dir or get_path("models", "vector_store")
        self.device = self._resolve_device(device)
        self.manifest_path = get_path("models", "pipeline_manifest")
        self.author_mapping_path = get_path("models", "author_mapping")
        self.vector_manifest_path = get_path("models", "vector_store_manifest")

        # Validate the dataset/artifact family before loading trusted tensors.
        self._manifest = validate_pipeline_manifest(
            self.manifest_path,
            csv_path=self.papers_csv,
            graph_path=self.graph_path,
        )

        # Lazy-loaded
        self._model: Optional[AdvancedHeteroGCN] = None
        self._graph = None
        self._papers_df: Optional[pd.DataFrame] = None
        self._vector_store: Optional[PaperVectorStore] = None
        self._author_predictions: Optional[Dict[int, Dict]] = None
        self._author_records: Optional[Dict[int, Dict]] = None
        self._topic_names: Optional[Dict[int, str]] = None

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> torch.device:
        requested = (device or os.environ.get("METROGNN_DEVICE", "auto")).lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot see a GPU. "
                "Use --device cpu or run with the GPU Docker Compose file."
            )
        return torch.device(requested)

    # ── Lazy loaders ────────────────────────────────────────────────────

    @property
    def model(self) -> AdvancedHeteroGCN:
        if self._model is None:
            self._model, _ = load_model_for_pipeline(
                self.model_path, self.manifest_path, map_location=self.device
            )
        return self._model

    @property
    def graph(self):
        if self._graph is None:
            self._graph = torch.load(
                self.graph_path, map_location=self.device, weights_only=False
            )
            if getattr(self._graph, "pipeline_schema_version", None) != 2:
                raise RuntimeError("Knowledge graph is not a v2 graph")
        return self._graph

    @property
    def papers_df(self) -> pd.DataFrame:
        if self._papers_df is None:
            self._papers_df = pd.read_csv(self.papers_csv)
        return self._papers_df

    @property
    def vector_store(self) -> PaperVectorStore:
        if self._vector_store is None:
            self._vector_store = PaperVectorStore(
                persist_dir=self.vector_store_dir,
                manifest_path=self.vector_manifest_path,
            )
            self._vector_store.validate_compatibility(
                csv_path=self.papers_csv,
                topic_assignments_path=get_path("data", "topic_assignments_path"),
                pipeline_id=self._manifest["pipeline_id"],
            )
        return self._vector_store

    @property
    def topic_names(self) -> Dict[int, str]:
        if self._topic_names is None:
            with open(get_path("data", "topic_json_path"), "r") as handle:
                raw = json.load(handle)
            self._topic_names = {
                int(key): value.get("name", f"Topic {key}")
                for key, value in raw.items()
            }
        return self._topic_names

    # ── Full-graph GNN (run once, cached) ───────────────────────────────

    def run_full_graph_inference(self) -> Dict[int, Dict]:
        """Run GNN on the full graph and cache per-author predictions."""
        if self._author_predictions is not None:
            return self._author_predictions

        model = self.model
        graph = self.graph

        with torch.no_grad():
            outputs = model(graph.x_dict, graph.edge_index_dict)

        if len(outputs) == 4:
            x_dict, importance_logits, role_logits, influence_scores = outputs
        else:
            x_dict = outputs[0]
            importance_logits = outputs[1] if len(outputs) > 1 else None
            role_logits = outputs[2] if len(outputs) > 2 else None
            influence_scores = outputs[3] if len(outputs) > 3 else None

        author_emb = x_dict.get("author", x_dict.get("researcher"))
        num_authors = author_emb.shape[0] if author_emb is not None else 0

        imp_cls = (
            importance_logits.argmax(dim=1).detach().cpu().numpy()
            if importance_logits is not None
            else None
        )
        role_cls = (
            role_logits.argmax(dim=1).detach().cpu().numpy()
            if role_logits is not None
            else None
        )
        inf_vals = (
            influence_scores.squeeze().detach().cpu().numpy()
            if influence_scores is not None
            else None
        )

        author_features = (
            graph["author"].x if "author" in graph.node_types else graph["researcher"].x
        )

        if self._author_records is None:
            self._author_records = load_author_mapping(self.author_mapping_path)

        self._author_predictions = {}
        for i in range(num_authors):
            identity = self._author_records.get(i, {})
            self._author_predictions[i] = {
                "author_id": i,
                "author_key": identity.get("author_key", ""),
                "name": identity.get("name", ""),
                "organization": identity.get("organization", ""),
                "importance_class": int(imp_cls[i]) if imp_cls is not None else None,
                "importance_label": IMPORTANCE_LABELS.get(int(imp_cls[i]), "Unknown")
                if imp_cls is not None
                else "Unknown",
                "role_class": int(role_cls[i]) if role_cls is not None else None,
                "role_label": ROLE_LABELS.get(int(role_cls[i]), "Unknown")
                if role_cls is not None
                else "Unknown",
                "influence": float(inf_vals[i]) if inf_vals is not None else 0.0,
                "paper_count": int(author_features[i, 0].item())
                if author_features is not None
                else 0,
                "collaboration_count": int(author_features[i, 1].item())
                if author_features is not None
                else 0,
                "topic_diversity": float(author_features[i, 2].item())
                if author_features is not None and author_features.shape[1] > 2
                else 0.0,
                "years_active": float(author_features[i, 3].item())
                if author_features is not None and author_features.shape[1] > 3
                else 0.0,
            }

        return self._author_predictions

    # ── Edge lookups ────────────────────────────────────────────────────

    def get_paper_author_edges(self) -> Dict[int, List[int]]:
        """paper_id → [author_ids] from the full graph."""
        if ("author", "authored", "paper") in self.graph.edge_types:
            edge_index = self.graph["author", "authored", "paper"].edge_index
        else:
            edge_index = self.graph["researcher", "authored", "paper"].edge_index

        mapping: Dict[int, List[int]] = {}
        for i in range(edge_index.shape[1]):
            author_id = int(edge_index[0, i].item())
            paper_id = int(edge_index[1, i].item())
            mapping.setdefault(paper_id, []).append(author_id)
        return mapping

    def get_paper_topic_edges(self) -> Dict[int, int]:
        """paper_id → topic_id from the full graph."""
        if ("paper", "has_topic", "topic") not in self.graph.edge_types:
            return {}
        edge_index = self.graph["paper", "has_topic", "topic"].edge_index
        mapping: Dict[int, int] = {}
        for i in range(edge_index.shape[1]):
            mapping[int(edge_index[0, i].item())] = int(edge_index[1, i].item())
        return mapping


# ── Public API ──────────────────────────────────────────────────────────


def run_inference(
    query: str,
    experience_level: str = "intermediate",
    interests: Optional[List[str]] = None,
    goals: str = "research",
    top_papers: int = 20,
    retrieval_k: int = 100,
    sort_by: str = "influence",
    device: Optional[str] = None,
) -> Dict:
    """
    End-to-end pipeline:
        Query → Embed → Vector DB Top-K → GNN Lookup → Ranked Results
    """
    inferencer = GNNInferencer(device=device)

    # 1. Semantic retrieval
    retrieval = inferencer.vector_store.query(
        query, top_k=retrieval_k, interests=interests
    )
    retrieved_ids = retrieval["paper_ids"]
    distances = retrieval["distances"]

    if not retrieved_ids:
        return {
            "user_profile": {
                "query": query,
                "experience_level": experience_level,
                "interests": interests or [],
                "goals": goals,
            },
            "papers": [],
            "summary": {
                "total_papers": len(inferencer.papers_df),
                "relevant_papers": 0,
                "top_influential_authors": [],
            },
        }

    # 2. GNN inference on the full graph (cached)
    author_preds = inferencer.run_full_graph_inference()

    # 3. Build similarity map  (lower distance == higher similarity)
    similarity_map: Dict[int, float] = {}
    for pid, dist in zip(retrieved_ids, distances):
        similarity_map[pid] = 1.0 - dist  # cosine similarity

    # 4. Assemble paper results with author info
    paper_author_edges = inferencer.get_paper_author_edges()
    paper_topic_edges = inferencer.get_paper_topic_edges()
    papers_df = inferencer.papers_df

    papers: List[Dict] = []
    for pid in retrieved_ids:
        if pid >= len(papers_df):
            continue
        row = papers_df.iloc[pid]

        author_ids = paper_author_edges.get(pid, [])
        authors_data = [author_preds[aid] for aid in author_ids if aid in author_preds]

        agg_influence = (
            sum(a["influence"] for a in authors_data) / len(authors_data)
            if authors_data
            else 0.0
        )

        topic_id = paper_topic_edges.get(
            pid, int(row.get("topic_id", -1)) if pd.notna(row.get("topic_id")) else -1
        )

        # Combined relevance = semantic similarity * 0.6 + GNN influence * 0.4
        semantic_sim = similarity_map.get(pid, 0.0)
        combined_score = 0.6 * semantic_sim + 0.4 * agg_influence

        papers.append(
            {
                "paper_id": pid,
                "title": row.get("title", f"Paper {pid}"),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "citations": int(row["n_citation"])
                if pd.notna(row.get("n_citation"))
                else 0,
                "venue": row.get("venue", ""),
                "topic_id": int(topic_id) if topic_id >= 0 else None,
                "topic_name": inferencer.topic_names.get(
                    int(topic_id), f"Topic {int(topic_id)}"
                )
                if topic_id >= 0
                else "",
                "authors": authors_data,
                "aggregated_influence": agg_influence,
                "semantic_similarity": semantic_sim,
                "relevance_score": combined_score,
                "num_authors": len(authors_data),
            }
        )

    # Sort by combined relevance score
    if sort_by == "influence":
        papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    elif sort_by == "citations":
        papers.sort(key=lambda x: x["citations"], reverse=True)
    elif sort_by == "year":
        papers.sort(key=lambda x: x["year"] or 0, reverse=True)

    papers = papers[:top_papers]

    # Top authors across selected papers
    all_authors: Dict[int, Dict] = {}
    for paper in papers:
        for author in paper["authors"]:
            aid = author["author_id"]
            if aid not in all_authors:
                all_authors[aid] = author
    top_authors = sorted(
        all_authors.values(), key=lambda x: x["influence"], reverse=True
    )[:10]

    return {
        "user_profile": {
            "query": query,
            "experience_level": experience_level,
            "interests": interests or [],
            "goals": goals,
        },
        "papers": papers,
        "summary": {
            "total_papers": len(papers_df),
            "retrieved_papers": len(retrieved_ids),
            "relevant_papers": len(papers),
            "top_influential_authors": top_authors,
        },
    }


# ── Pretty printer ──────────────────────────────────────────────────────


def print_results(result: Dict, verbose: bool = True):
    print("=" * 70)
    print("GNN INFERENCE RESULTS  (Vector Retrieval Pipeline)")
    print("=" * 70)

    profile = result["user_profile"]
    print(f"\nUser Query: {profile['query']}")
    print(f"Experience Level: {profile['experience_level']}")
    print(f"Goals: {profile['goals']}")
    print(f"Interests: {', '.join(profile['interests']) or 'None'}")

    print(f"\n{'=' * 70}")
    print(f"PAPERS — Showing top {len(result['papers'])}")
    print("=" * 70)

    for i, paper in enumerate(result["papers"], 1):
        title = paper["title"][:70]
        print(f"\n[{i}] {title}...")
        print(
            f"    Year: {paper['year'] or 'N/A'} | Citations: {paper['citations']} "
            f"| Venue: {paper['venue']}"
        )
        print(
            f"    Semantic Sim: {paper['semantic_similarity']:.3f} "
            f"| GNN Influence: {paper['aggregated_influence']:.3f} "
            f"| Combined: {paper['relevance_score']:.3f}"
        )
        if paper["authors"]:
            descs = [
                f"Author {a['author_id']}(Inf={a['influence']:.2f})"
                for a in paper["authors"][:4]
            ]
            print(f"    Authors ({len(paper['authors'])}): {', '.join(descs)}")
        else:
            print("    Authors: —")

    print(f"\n{'=' * 70}")
    print("TOP INFLUENTIAL AUTHORS (across selected papers)")
    print("=" * 70)
    for i, author in enumerate(result["summary"]["top_influential_authors"][:10], 1):
        print(
            f"{i:2}. Author {author['author_id']:3}: {author['importance_label']:6} | "
            f"{author['role_label']:9} | Influence: {author['influence']:.3f} | "
            f"Papers: {author['paper_count']:2} | Collabs: {author['collaboration_count']}"
        )

    summary = result["summary"]
    print(
        f"\nSummary: Retrieved {summary.get('retrieved_papers', '?')} papers via vector search, "
        f"returning top {summary['relevant_papers']} out of {summary['total_papers']} total."
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GNN Inference — Vector Retrieval Pipeline"
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="Research query")
    parser.add_argument(
        "--experience",
        "-e",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
    )
    parser.add_argument(
        "--interests", "-i", type=str, default="", help="Comma-separated interests"
    )
    parser.add_argument(
        "--goals",
        "-g",
        type=str,
        default="research",
        choices=["learn", "research", "career"],
    )
    parser.add_argument(
        "--top", "-t", type=int, default=20, help="Final papers to return"
    )
    parser.add_argument(
        "--retrieval-k", "-k", type=int, default=100, help="Top-K for vector retrieval"
    )
    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default="influence",
        choices=["influence", "citations", "year"],
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Optional JSON output path"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=os.environ.get("METROGNN_DEVICE", "auto"),
        help="Inference device. auto uses CUDA when PyTorch can see a GPU.",
    )

    args = parser.parse_args()
    interests = [i.strip() for i in args.interests.split(",") if i.strip()]

    try:
        result = run_inference(
            query=args.query,
            experience_level=args.experience,
            interests=interests or None,
            goals=args.goals,
            top_papers=args.top,
            retrieval_k=args.retrieval_k,
            sort_by=args.sort,
            device=args.device,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None

    print_results(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
