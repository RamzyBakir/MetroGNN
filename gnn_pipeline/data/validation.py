"""Fail-fast validation for the recommendation pipeline data and graph."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import torch
from torch_geometric.data import HeteroData


REQUIRED_PAPER_COLUMNS = {
    "id",
    "title",
    "authors",
    "references",
    "year",
    "n_citation",
    "venue",
}

LEGACY_SYNTHETIC_ID_PREFIX = "legacy-row:"


def ensure_paper_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with v2 identity/reference columns for legacy local CSVs.

    Older local DBLP extracts contain paper metadata but no stable paper IDs or
    reference lists. The v2 graph/vector code needs those columns for artifact
    alignment, so we synthesize deterministic row IDs and empty references in
    memory without modifying the source CSV.
    """
    result = df.copy()
    if "id" not in result.columns:
        if "paper_id" in result.columns:
            result.insert(0, "id", result["paper_id"].astype(str))
        else:
            result.insert(
                0,
                "id",
                [f"{LEGACY_SYNTHETIC_ID_PREFIX}{idx}" for idx in range(len(result))],
            )
    if "references" not in result.columns:
        result["references"] = "[]"
    return result


def validate_papers_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    missing = REQUIRED_PAPER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset contains no papers")
    if df["id"].isna().any():
        raise ValueError("Paper IDs contain missing values")
    if not df["id"].astype(str).is_unique:
        raise ValueError("Paper IDs must be unique")
    years = pd.to_numeric(df["year"], errors="coerce")
    citations = pd.to_numeric(df["n_citation"], errors="coerce")
    return {
        "num_papers": len(df),
        "missing_titles": int(df["title"].isna().sum()),
        "missing_venues": int(df["venue"].isna().sum()),
        "invalid_years": int(years.isna().sum()),
        "year_min": int(years.min()) if years.notna().any() else None,
        "year_max": int(years.max()) if years.notna().any() else None,
        "invalid_citations": int(citations.isna().sum()),
        "negative_citations": int((citations < 0).sum()),
    }


def validate_knowledge_graph(
    data: HeteroData,
    *,
    expected_papers: int,
    expected_authors: int,
    expected_topics: int,
) -> Dict[str, Any]:
    if getattr(data, "pipeline_schema_version", None) != 2:
        raise ValueError("Graph does not declare pipeline schema version 2")

    required_nodes = {"paper", "author", "topic"}
    if not required_nodes.issubset(data.node_types):
        raise ValueError(f"Graph is missing node types: {required_nodes - set(data.node_types)}")

    expected_counts = {
        "paper": expected_papers,
        "author": expected_authors,
        "topic": expected_topics,
    }
    for node_type, expected in expected_counts.items():
        actual = int(data[node_type].num_nodes or 0)
        if actual != expected:
            raise ValueError(
                f"Graph {node_type} count is {actual}, expected {expected}"
            )
        if data[node_type].x is None or not torch.isfinite(data[node_type].x).all():
            raise ValueError(f"Graph {node_type} features are missing or non-finite")

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape for {edge_type}")
        if edge_index.numel() == 0:
            continue
        src, _, dst = edge_type
        if int(edge_index.min()) < 0:
            raise ValueError(f"Negative node index in {edge_type}")
        if int(edge_index[0].max()) >= expected_counts[src]:
            raise ValueError(f"Source index out of range in {edge_type}")
        if int(edge_index[1].max()) >= expected_counts[dst]:
            raise ValueError(f"Destination index out of range in {edge_type}")

    topic_edge = ("paper", "has_topic", "topic")
    if topic_edge not in data.edge_types:
        raise ValueError("Graph has no paper-topic edges")
    topic_edges = data[topic_edge].edge_index
    if topic_edges.shape[1] != expected_papers:
        raise ValueError("Every paper must have exactly one topic assignment")
    per_paper_topics = torch.bincount(topic_edges[0], minlength=expected_papers)
    if not torch.all(per_paper_topics == 1):
        raise ValueError("A paper has a missing or duplicate topic assignment")
    populated_topics = int(topic_edges[1].unique().numel())
    if expected_topics > 1 and populated_topics < 2:
        raise ValueError("Paper-topic edges populate fewer than two topics")

    citation_edge = ("paper", "cites", "paper")
    if citation_edge not in data.edge_types:
        raise ValueError("Graph has no paper citation relation")
    reverse_citation_edge = ("paper", "cited_by", "paper")
    if reverse_citation_edge not in data.edge_types:
        raise ValueError("Graph has no reverse paper citation relation")
    cites = {tuple(edge) for edge in data[citation_edge].edge_index.t().tolist()}
    cited_by = {
        (target, source)
        for source, target in data[reverse_citation_edge].edge_index.t().tolist()
    }
    if cites != cited_by:
        raise ValueError("Citation and reverse-citation edges are inconsistent")

    return {
        "num_papers": expected_papers,
        "num_authors": expected_authors,
        "num_topics": expected_topics,
        "populated_topics": populated_topics,
        "authorship_edges": int(
            data["author", "authored", "paper"].edge_index.shape[1]
        ),
        "collaboration_edges": int(
            data["author", "collaborates_with", "author"].edge_index.shape[1]
        ),
        "citation_edges": int(data[citation_edge].edge_index.shape[1]),
    }
