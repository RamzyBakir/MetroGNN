import json
import logging
import os
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from mcp.server.mcpserver import Context

from mcp_server.models.schemas import (
    AnalyzeCommunityInput,
    AnalyzeCommunityOutput,
    CommunityMetrics,
    DetectTopicsInput,
    DetectTopicsOutput,
    GenerateStrategyInput,
    GenerateStrategyOutput,
    ResearcherData,
    TopicInfo,
)
from mcp_server.server import _get_ctx, mcp

logger = logging.getLogger("mcp_server.tools.analysis")


def _ensure_graph_and_model(app):
    from mcp_server.tools.inference import _ensure_model_and_graph

    _ensure_model_and_graph(app)


def _ensure_topic_info(app):
    if app.topic_info_dict is None or app.topic_names is None:
        try:
            from gnn_pipeline.config import get_path

            topic_json = get_path("data", "topic_json_path")
            if topic_json and os.path.isfile(topic_json):
                with open(topic_json) as f:
                    raw = json.load(f)
                app.topic_info_dict = {int(k): v for k, v in raw.items()}
                app.topic_names = {
                    int(k): v.get("name", f"Topic {k}") for k, v in raw.items()
                }
                from gnn_pipeline.data.topic_assignment import TopicAssigner

                app.topic_assigner = TopicAssigner.from_centroids(
                    get_path("data", "topic_centroids_path")
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not load topic artifacts: %s", exc)
        if app.topic_names is None:
            raise RuntimeError(
                "Versioned topic metadata/centroids are unavailable. Build the v2 pipeline first."
            )


def _detect_topic_ids(app, query: str) -> List[int]:
    from gnn_pipeline.analysis.topic_detection import (
        extract_topics_from_query,
        extract_topics_from_query_dynamic,
        fallback_keyword_match,
    )

    topic_info = app.topic_info_dict or {}

    try:
        if app.topic_assigner is not None and hasattr(
            app.topic_assigner, "rank_query_topics"
        ):
            detected = extract_topics_from_query_dynamic(
                query, app.topic_assigner, topic_info
            )
            if detected:
                return detected
    except Exception:
        pass

    detected = fallback_keyword_match(query, topic_info)
    if detected:
        return detected

    return extract_topics_from_query(query, app.topic_names or {})


@mcp.tool()
async def detect_topics(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    method: Literal["dynamic", "keyword"] = "dynamic",
    ctx: Context = None,
) -> DetectTopicsOutput:
    """Detect which research topics a query relates to.

    Maps a free-text query to topic IDs in the knowledge graph. Two methods
    are available: 'dynamic' uses TF-IDF cosine similarity against topic
    descriptions (requires a TopicAssigner), 'keyword' uses hardcoded keyword
    matching. Returns matched topic IDs with relevance scores.

    Args:
        query: Free-text query to analyze.
        method: Detection method ('dynamic' or 'keyword').
    """
    app = _get_ctx(ctx)
    _ensure_topic_info(app)

    from gnn_pipeline.analysis.topic_detection import (
        extract_topics_from_query,
        extract_topics_from_query_dynamic,
        fallback_keyword_match,
    )

    topic_names = app.topic_names or {}
    topic_info = app.topic_info_dict or {}

    if method == "keyword":
        detected = fallback_keyword_match(query, topic_info)
    else:
        detected = _detect_topic_ids(app, query)

    score_map = {}
    if method == "dynamic" and app.topic_assigner is not None:
        score_map = dict(app.topic_assigner.rank_query_topics(query, top_k=2))

    detected_topics = []
    for tid in detected:
        detected_topics.append(
            TopicInfo(
                topic_id=tid,
                name=topic_names.get(tid, f"Topic {tid}"),
                relevance_score=float(score_map.get(tid, 0.0)),
            )
        )

    return DetectTopicsOutput(
        query=query,
        detected_topics=detected_topics,
        all_topic_names=topic_names,
    )


@mcp.tool()
async def analyze_community(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    topic_ids: Optional[List[Annotated[int, Field(ge=0)]]] = None,
    ctx: Context = None,
) -> AnalyzeCommunityOutput:
    """Analyze the research community around detected topics.

    Given a query (or explicit topic IDs), discovers the sub-community of
    relevant authors, builds a NetworkX subgraph, computes centrality metrics,
    identifies key members, and returns detailed researcher profiles with
    composite scores. Requires train_pipeline to have been run.

    Args:
        query: User research query to detect topics from.
        topic_ids: Explicit topic IDs to analyze. Auto-detected from query if omitted.
    """
    app = _get_ctx(ctx)
    _ensure_graph_and_model(app)
    _ensure_topic_info(app)

    from mcp_server.tools.inference import _run_gnn_inference

    _run_gnn_inference(app)

    if topic_ids is None:
        topic_ids = _detect_topic_ids(app, query)

    invalid_topic_ids = sorted(set(topic_ids or []) - set(app.topic_info_dict or {}))
    if invalid_topic_ids:
        raise ValueError(f"Unknown topic IDs: {invalid_topic_ids}")

    if not topic_ids:
        return AnalyzeCommunityOutput(
            community_name="No matching topics",
            query_topics=[],
            key_members=[],
            other_members=[],
            community_metrics=CommunityMetrics(
                size=0,
                cohesion=0,
                avg_clustering=0,
                internal_edges=0,
                research_productivity=0,
                collaboration_potential=0,
                influence_concentration=0,
            ),
            enhanced_analysis={},
        )

    import torch
    import numpy as np
    import networkx as nx
    from gnn_pipeline.analysis.community_analysis import (
        advanced_community_analysis,
        dynamic_key_member_threshold,
        topic_specialized_researcher_ranking,
        calculate_additional_metrics,
        calculate_realistic_paper_quality,
        calculate_researcher_freshness,
    )

    graph = app.graph
    with torch.no_grad():
        outputs = app.model(graph.x_dict, graph.edge_index_dict)
        x_dict, imp_logits, role_logits, inf_scores = outputs

    updated_embeddings = {
        "paper": x_dict.get("paper"),
        "author": x_dict.get("author", x_dict.get("researcher")),
        "paper_quality": x_dict.get("paper_quality"),
    }

    if app.importance_logits is None:
        app.importance_logits = imp_logits
        app.role_logits = role_logits
        app.influence_scores = inf_scores

    paper_topic_edges = graph["paper", "has_topic", "topic"].edge_index.numpy()
    relevant_papers = set()
    for tid in topic_ids:
        pidx = paper_topic_edges[0][paper_topic_edges[1] == tid]
        relevant_papers.update(pidx.tolist())

    author_edges = graph["author", "authored", "paper"].edge_index.numpy()
    relevant_authors = set()
    for pid in relevant_papers:
        auths = author_edges[0][author_edges[1] == pid]
        relevant_authors.update(auths.tolist())

    max_community_size = 500
    if len(relevant_authors) > max_community_size:
        inf_vals = inf_scores.squeeze().numpy()
        sorted_authors = sorted(
            relevant_authors,
            key=lambda a: inf_vals[a] if a < len(inf_vals) else 0,
            reverse=True,
        )
        relevant_authors = set(sorted_authors[:max_community_size])

    collab_edges = graph["author", "collaborates_with", "author"].edge_index.numpy()
    community_members = set(relevant_authors)
    for res in list(relevant_authors):
        collabs = collab_edges[1][collab_edges[0] == res]
        for c in collabs:
            ci = int(c)
            if ci in relevant_authors:
                community_members.add(ci)

    community_members = list(community_members)
    if not community_members:
        return AnalyzeCommunityOutput(
            community_name="No researchers found",
            query_topics=topic_ids,
            key_members=[],
            other_members=[],
            community_metrics=CommunityMetrics(
                size=0,
                cohesion=0,
                avg_clustering=0,
                internal_edges=0,
                research_productivity=0,
                collaboration_potential=0,
                influence_concentration=0,
            ),
            enhanced_analysis={},
        )

    G = nx.Graph()
    for aid in community_members:
        G.add_node(aid)
    for i in range(collab_edges.shape[1]):
        s, d = int(collab_edges[0, i]), int(collab_edges[1, i])
        if s in community_members and d in community_members:
            G.add_edge(s, d)

    centrality = nx.degree_centrality(G) if len(community_members) > 1 else {}
    betweenness = (
        {}
        if len(community_members) > 200
        else (nx.betweenness_centrality(G) if len(community_members) > 1 else {})
    )
    clustering = nx.clustering(G) if len(community_members) > 1 else {}
    density = nx.density(G)

    imp_pred = imp_logits.argmax(dim=1) if imp_logits is not None else None
    role_pred = role_logits.argmax(dim=1) if role_logits is not None else None
    inf_vals = inf_scores.squeeze() if inf_scores is not None else None

    authored_edges = graph["author", "authored", "paper"].edge_index.numpy()
    key_members, other_members = [], []
    role_names = ["Community Leader", "Core Member", "Contributor", "Sole Expert"]
    imp_names = ["Low", "Medium", "High"]

    for rid in community_members:
        r_papers = authored_edges[1][authored_edges[0] == rid]
        tp = len([p for p in r_papers if p in relevant_papers])
        total_p = len(r_papers)
        ts = tp / max(total_p, 1)

        rd = {
            "id": f"R{rid}",
            "global_influence": inf_vals[rid].item()
            if inf_vals is not None and rid < len(inf_vals)
            else 0.5,
            "centrality": centrality.get(rid, 0),
            "betweenness": betweenness.get(rid, 0),
            "clustering": clustering.get(rid, 0),
            "topic_specialization": ts,
            "topic_paper_count": tp,
            "total_papers": total_p,
            "topic_specialization_score": ts,
        }

        if role_pred is not None and rid < len(role_pred):
            rd["role"] = role_names[role_pred[rid].item()]
        else:
            rd["role"] = (
                "Community Leader"
                if rd["centrality"] > 0.6
                else ("Core Member" if rd["centrality"] > 0.3 else "Contributor")
            )

        if imp_pred is not None and rid < len(imp_pred):
            idx = imp_pred[rid].item()
            rd["importance"] = imp_names[idx]
            rd["importance_score"] = (idx + 1) / 3.0
        else:
            rd["importance"] = "Medium"
            rd["importance_score"] = 0.5

        rd["paper_quality"] = 0.5
        rd["freshness_score"] = 0.5

        composite = (
            rd["importance_score"] * 0.3
            + rd["global_influence"] * 0.25
            + rd["centrality"] * 0.15
            + rd["paper_quality"] * 0.15
            + rd["topic_specialization"] * 0.15
        )
        rd["composite_score"] = min(composite, 1.0)

        dt = dynamic_key_member_threshold(
            len(community_members),
            np.mean([m["global_influence"] for m in key_members])
            if key_members
            else 0.5,
        )
        is_key = (
            "Leader" in rd["role"]
            or rd["composite_score"] > dt
            or rd["betweenness"] > 0.15
            or rd["topic_specialization"] > 0.6
        )
        (key_members if is_key else other_members).append(rd)

    if key_members:
        key_members = topic_specialized_researcher_ranking(
            key_members, topic_ids, graph
        )
    if other_members:
        other_members = topic_specialized_researcher_ranking(
            other_members, topic_ids, graph
        )

    all_m = key_members + other_members
    avg_inf = np.mean([m["global_influence"] for m in all_m]) if all_m else 0
    avg_pq = np.mean([m.get("paper_quality", 0.5) for m in all_m]) if all_m else 0
    avg_ts = np.mean([m.get("topic_specialization", 0) for m in all_m]) if all_m else 0

    cm = {
        "size": len(community_members),
        "cohesion": density,
        "avg_clustering": np.mean(list(clustering.values())) if clustering else 0,
        "avg_betweenness": np.mean(list(betweenness.values())) if betweenness else 0,
        "internal_edges": G.number_of_edges(),
        "paper_quality": avg_pq,
        "topic_specialization": avg_ts,
    }

    add_m = calculate_additional_metrics(
        {"key_members": key_members, "community_metrics": cm}
    )
    cm.update(add_m)

    enhanced = {
        "avg_influence": avg_inf,
        "max_influence": max((m["global_influence"] for m in all_m), default=0),
        "role_diversity": len(set(m["role"] for m in all_m)),
        "network_health": cm.get("avg_betweenness", 0) * density,
        "research_quality": avg_pq,
        "topic_alignment": avg_ts,
        "key_member_ratio": len(key_members) / len(community_members)
        if community_members
        else 0,
        "research_productivity": add_m["research_productivity"],
        "influence_concentration": add_m["influence_concentration"],
        "collaboration_potential": add_m["collaboration_potential"],
    }

    community_name = " & ".join(
        [app.topic_names.get(tid, f"Topic_{tid}") for tid in topic_ids]
    )

    def _to_researcher(m):
        return ResearcherData(
            id=m.get("id", ""),
            global_influence=m.get("global_influence", 0),
            centrality=m.get("centrality", 0),
            betweenness=m.get("betweenness", 0),
            clustering=m.get("clustering", 0),
            topic_specialization=m.get("topic_specialization", 0),
            role=m.get("role", "Unknown"),
            importance=m.get("importance", "Medium"),
            composite_score=m.get("composite_score", 0),
        )

    metrics = CommunityMetrics(
        size=cm["size"],
        cohesion=cm["cohesion"],
        avg_clustering=cm["avg_clustering"],
        internal_edges=cm["internal_edges"],
        research_productivity=cm["research_productivity"],
        collaboration_potential=cm["collaboration_potential"],
        influence_concentration=cm["influence_concentration"],
    )

    return AnalyzeCommunityOutput(
        community_name=community_name,
        query_topics=topic_ids,
        key_members=[_to_researcher(m) for m in key_members],
        other_members=[_to_researcher(m) for m in other_members],
        community_metrics=metrics,
        enhanced_analysis=enhanced,
    )


@mcp.tool()
async def generate_strategy(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    topic_ids: Optional[List[Annotated[int, Field(ge=0)]]] = None,
    ctx: Context = None,
) -> GenerateStrategyOutput:
    """Generate a personalized research strategy from community analysis.

    Detects topics, analyzes the relevant research community, then assembles
    a comprehensive strategy including learning path, collaboration
    recommendations, skill development, research project ideas, timeline,
    and success metrics. Requires train_pipeline to have been run.

    Args:
        query: User research query.
        topic_ids: Explicit topic IDs. Auto-detected if omitted.
    """
    app = _get_ctx(ctx)
    _ensure_graph_and_model(app)
    _ensure_topic_info(app)

    from gnn_pipeline.strategy.generator import StrategyGenerator
    from gnn_pipeline.strategy.components import (
        DynamicContentGenerator,
        PersonalizedPlanAssembler,
    )

    generator = StrategyGenerator(topic_names=app.topic_names)
    content_gen = DynamicContentGenerator()
    assembler = PersonalizedPlanAssembler()

    if topic_ids is None:
        topic_ids = _detect_topic_ids(app, query)

    community_result = await analyze_community(
        query=query, topic_ids=topic_ids, ctx=ctx
    )

    community_data = {
        "community_name": community_result.community_name,
        "query_topics": community_result.query_topics,
        "key_members": [m.model_dump() for m in community_result.key_members],
        "other_members": [m.model_dump() for m in community_result.other_members],
        "community_metrics": community_result.community_metrics.model_dump(),
        "enhanced_analysis": community_result.enhanced_analysis,
    }

    strategy_context = generator.assemble_strategy_context(community_data, query)
    llm_strategy = content_gen.generate_all_components(
        strategy_context, app.topic_info_dict
    )

    final_plan = assembler.assemble_final_plan(
        strategy_context, llm_strategy, app.topic_info_dict
    )

    return GenerateStrategyOutput(
        metadata=final_plan.get("metadata", {}),
        executive_summary=final_plan.get("executive_summary", ""),
        community_analysis=final_plan.get("community_analysis", {}),
        strategic_plan=final_plan.get("strategic_plan", {}),
        implementation_guide=final_plan.get("implementation_guide", {}),
        success_framework=final_plan.get("success_framework", {}),
    )
