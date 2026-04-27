import logging
from typing import Any, Dict, List, Optional

import torch
from mcp.server.fastmcp import Context

from gnn_pipeline.config import get_path
from gnn_pipeline.data.vector_store import PaperVectorStore
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN

from mcp_server.models.schemas import (
    AuthorProfile,
    GetAuthorProfilesInput,
    GetAuthorProfilesOutput,
    InferenceSummary,
    PaperResult,
    QueryPapersInput,
    QueryPapersOutput,
    RunInferenceInput,
    RunInferenceOutput,
)
from mcp_server.server import _get_ctx, mcp

logger = logging.getLogger("mcp_server.tools.inference")

IMPORTANCE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ROLE_LABELS = {0: "Hub", 1: "Active", 2: "Occasional", 3: "Isolated"}


def _ensure_model_and_graph(app) -> None:
    if not app.is_model_ready():
        model_path = get_path("models", "best_model_new")
        cfg = {}
        try:
            from gnn_pipeline.config import get_value

            cfg = {
                "hidden_channels": get_value("training", "hidden_channels"),
                "out_channels": get_value("training", "out_channels"),
                "num_layers": get_value("training", "num_layers"),
                "num_classes": get_value("training", "num_classes"),
                "n_clusters": get_value("training", "n_clusters"),
            }
        except Exception:
            cfg = {
                "hidden_channels": 64,
                "out_channels": 32,
                "num_layers": 4,
                "num_classes": 3,
                "n_clusters": 8,
            }

        model = AdvancedHeteroGCN(
            hidden_channels=cfg["hidden_channels"],
            out_channels=cfg["out_channels"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            num_topics=cfg.get("n_clusters", 8),
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        app.model = model
        app._model_loaded = True

    if not app.is_graph_ready():
        graph_path = get_path("models", "knowledge_graph")
        app.graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        app._graph_loaded = True


def _run_gnn_inference(app) -> Dict[int, Dict]:
    if app.author_predictions is not None:
        return app.author_predictions

    model = app.model
    graph = app.graph

    with torch.no_grad():
        outputs = model(graph.x_dict, graph.edge_index_dict)

    x_dict, imp_logits, role_logits, inf_scores = outputs

    imp_cls = imp_logits.argmax(dim=1).numpy()
    role_cls = role_logits.argmax(dim=1).numpy()
    inf_vals = inf_scores.squeeze().numpy()

    num_authors = x_dict.get("author", x_dict.get("researcher")).shape[0]
    author_features = (
        graph["author"].x if "author" in graph.node_types else graph["researcher"].x
    )

    app.author_predictions = {}
    app.importance_logits = imp_logits
    app.role_logits = role_logits
    app.influence_scores = inf_scores

    for i in range(num_authors):
        app.author_predictions[i] = {
            "author_id": i,
            "importance_class": int(imp_cls[i]),
            "importance_label": IMPORTANCE_LABELS.get(int(imp_cls[i]), "Unknown"),
            "role_class": int(role_cls[i]),
            "role_label": ROLE_LABELS.get(int(role_cls[i]), "Unknown"),
            "influence": float(inf_vals[i]),
            "paper_count": int(author_features[i, 0].item()),
            "collaboration_count": int(author_features[i, 1].item()),
            "topic_diversity": float(author_features[i, 2].item()),
            "years_active": float(author_features[i, 3].item()),
            "citation_impact": float(author_features[i, 4].item()),
        }

    return app.author_predictions


def _make_author_profile(d: Dict) -> AuthorProfile:
    return AuthorProfile(**d)


def _get_paper_author_edges(app) -> Dict[int, List[int]]:
    graph = app.graph
    if ("author", "authored", "paper") in graph.edge_types:
        ei = graph["author", "authored", "paper"].edge_index
    else:
        ei = graph["researcher", "authored", "paper"].edge_index
    mapping: Dict[int, List[int]] = {}
    for i in range(ei.shape[1]):
        mapping.setdefault(int(ei[1, i].item()), []).append(int(ei[0, i].item()))
    return mapping


def _get_paper_topic_edges(app) -> Dict[int, int]:
    graph = app.graph
    if ("paper", "has_topic", "topic") not in graph.edge_types:
        return {}
    ei = graph["paper", "has_topic", "topic"].edge_index
    return {int(ei[0, i].item()): int(ei[1, i].item()) for i in range(ei.shape[1])}


@mcp.tool()
async def query_papers(
    query: str,
    top_k: int = 50,
    interests: Optional[List[str]] = None,
    ctx: Context = None,
) -> QueryPapersOutput:
    """Search for papers using semantic vector similarity.

    Embeds the query with MiniLM and retrieves the top-K most similar
    papers from the ChromaDB vector store. Requires that build_vector_store
    has been run first.

    Args:
        query: Free-text search query (e.g. "graph neural networks").
        top_k: Number of papers to retrieve (1-500).
        interests: Optional extra keywords appended to the query for boosting.
    """
    app = _get_ctx(ctx)
    store = PaperVectorStore(persist_dir=get_path("models", "vector_store"))

    result = store.query(query_text=query, top_k=top_k, interests=interests)

    papers = []
    for pid, dist, meta, doc in zip(
        result["paper_ids"],
        result["distances"],
        result["metadatas"],
        result["documents"],
    ):
        papers.append(
            {
                "paper_id": pid,
                "distance": float(dist),
                "title": meta.get("title", ""),
                "year": meta.get("year"),
                "citations": meta.get("n_citation", 0),
                "venue": meta.get("venue", ""),
            }
        )

    return QueryPapersOutput(query=query, total_results=len(papers), papers=papers)


@mcp.tool()
async def get_author_profiles(
    author_ids: Optional[List[int]] = None,
    importance_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    min_influence: Optional[float] = None,
    limit: int = 50,
    ctx: Context = None,
) -> GetAuthorProfilesOutput:
    """Retrieve GNN-predicted author profiles from the knowledge graph.

    Returns importance class, role, influence score, paper count, collaboration
    count, and other features for each author. Can filter by importance, role,
    or minimum influence. Requires that train_pipeline has been run.

    Args:
        author_ids: Specific author IDs to look up. Omit for all authors.
        importance_filter: Filter by importance label (Low, Medium, High).
        role_filter: Filter by role label (Hub, Active, Occasional, Isolated).
        min_influence: Minimum influence score threshold.
        limit: Maximum number of authors to return (1-500).
    """
    app = _get_ctx(ctx)
    _ensure_model_and_graph(app)
    all_preds = _run_gnn_inference(app)

    total = len(all_preds)
    results = list(all_preds.values())

    if author_ids is not None:
        id_set = set(author_ids)
        results = [r for r in results if r["author_id"] in id_set]
    if importance_filter is not None:
        results = [r for r in results if r["importance_label"] == importance_filter]
    if role_filter is not None:
        results = [r for r in results if r["role_label"] == role_filter]
    if min_influence is not None:
        results = [r for r in results if r["influence"] >= min_influence]

    results.sort(key=lambda x: x["influence"], reverse=True)
    results = results[:limit]

    return GetAuthorProfilesOutput(
        total_authors=total,
        returned_count=len(results),
        authors=[_make_author_profile(r) for r in results],
    )


@mcp.tool()
async def run_inference(
    query: str,
    experience_level: str = "intermediate",
    interests: Optional[List[str]] = None,
    goals: str = "research",
    top_papers: int = 20,
    retrieval_k: int = 100,
    sort_by: str = "influence",
    ctx: Context = None,
) -> RunInferenceOutput:
    """Run the full inference pipeline: semantic retrieval + GNN scoring.

    Takes a free-text query, retrieves matching papers from the vector store,
    runs the GNN on the full knowledge graph, and combines semantic similarity
    with GNN author influence to produce ranked paper results with author
    profiles. Requires train_pipeline and build_vector_store to have been run.

    Args:
        query: Free-text research query (e.g. "multi-agent reinforcement learning").
        experience_level: User experience level (beginner, intermediate, advanced).
        interests: Optional interest keywords for boosting retrieval.
        goals: User goal (learn, research, career).
        top_papers: Number of final papers to return (1-100).
        retrieval_k: Top-K for initial vector retrieval (1-500).
        sort_by: How to sort results (influence, citations, year).
    """
    import pandas as pd

    app = _get_ctx(ctx)
    _ensure_model_and_graph(app)

    store = PaperVectorStore(persist_dir=get_path("models", "vector_store"))
    retrieval = store.query(query_text=query, top_k=retrieval_k, interests=interests)
    retrieved_ids = retrieval["paper_ids"]
    distances = retrieval["distances"]

    if not retrieved_ids:
        return RunInferenceOutput(
            user_profile={
                "query": query,
                "experience_level": experience_level,
                "interests": interests or [],
                "goals": goals,
            },
            papers=[],
            summary=InferenceSummary(
                total_papers=0,
                retrieved_papers=0,
                relevant_papers=0,
                top_influential_authors=[],
            ),
        )

    author_preds = _run_gnn_inference(app)
    paper_author_edges = _get_paper_author_edges(app)
    paper_topic_edges = _get_paper_topic_edges(app)

    if app.papers_df is None:
        app.papers_df = pd.read_csv(get_path("data", "csv_path"))
    papers_df = app.papers_df

    similarity_map = {pid: 1.0 - dist for pid, dist in zip(retrieved_ids, distances)}

    papers: List[PaperResult] = []
    for pid in retrieved_ids:
        if pid >= len(papers_df):
            continue
        row = papers_df.iloc[pid]

        author_ids_list = paper_author_edges.get(pid, [])
        authors_data = [
            author_preds[aid] for aid in author_ids_list if aid in author_preds
        ]

        agg_influence = (
            (sum(a["influence"] for a in authors_data) / len(authors_data))
            if authors_data
            else 0.0
        )

        topic_id = paper_topic_edges.get(
            pid, int(row.get("topic_id", -1)) if pd.notna(row.get("topic_id")) else -1
        )

        semantic_sim = similarity_map.get(pid, 0.0)
        combined_score = 0.6 * semantic_sim + 0.4 * agg_influence

        papers.append(
            PaperResult(
                paper_id=pid,
                title=str(row.get("title", f"Paper {pid}")),
                year=int(row["year"]) if pd.notna(row.get("year")) else None,
                citations=int(row["n_citation"])
                if pd.notna(row.get("n_citation"))
                else 0,
                venue=str(row.get("venue", "")),
                topic_id=int(topic_id) if topic_id >= 0 else None,
                topic_name=f"Topic {int(topic_id)}" if topic_id >= 0 else "",
                authors=[_make_author_profile(a) for a in authors_data],
                aggregated_influence=agg_influence,
                semantic_similarity=semantic_sim,
                relevance_score=combined_score,
                num_authors=len(authors_data),
            )
        )

    if sort_by == "influence":
        papers.sort(key=lambda x: x.relevance_score, reverse=True)
    elif sort_by == "citations":
        papers.sort(key=lambda x: x.citations, reverse=True)
    elif sort_by == "year":
        papers.sort(key=lambda x: x.year or 0, reverse=True)

    papers = papers[:top_papers]

    all_authors: Dict[int, AuthorProfile] = {}
    for paper in papers:
        for author in paper.authors:
            if author.author_id not in all_authors:
                all_authors[author.author_id] = author
    top_authors = sorted(all_authors.values(), key=lambda x: x.influence, reverse=True)[
        :10
    ]

    return RunInferenceOutput(
        user_profile={
            "query": query,
            "experience_level": experience_level,
            "interests": interests or [],
            "goals": goals,
        },
        papers=papers,
        summary=InferenceSummary(
            total_papers=len(papers_df),
            retrieved_papers=len(retrieved_ids),
            relevant_papers=len(papers),
            top_influential_authors=top_authors,
        ),
    )
