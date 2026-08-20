import json
import logging
from typing import Any, Dict

from mcp.server.mcpserver import Context

from mcp_server.models.schemas import GraphInfoOutput
from mcp_server.server import _get_ctx, mcp

logger = logging.getLogger("mcp_server.tools.utility")


@mcp.tool()
async def get_graph_info(ctx: Context) -> GraphInfoOutput:
    """Load and inspect the MetroGNN heterogeneous knowledge graph.

    Returns node counts, edge counts, and feature dimensions for each node
    type (paper, author, topic). Requires that train_pipeline has been run
    at least once so the graph file exists.
    """
    app = _get_ctx(ctx)

    if not app.is_graph_ready():
        from gnn_pipeline.artifacts import validate_pipeline_manifest
        from gnn_pipeline.config import get_path
        import torch

        graph_path = get_path("models", "knowledge_graph")
        try:
            validate_pipeline_manifest(
                get_path("models", "pipeline_manifest"),
                csv_path=get_path("data", "csv_path"),
                graph_path=graph_path,
            )
            app.graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            app._graph_loaded = True
        except FileNotFoundError:
            return GraphInfoOutput(
                node_types={},
                edge_types={},
                feature_shapes={},
                graph_path=graph_path,
            )

    graph_path_str = ""
    try:
        from gnn_pipeline.config import get_path

        graph_path_str = get_path("models", "knowledge_graph")
    except Exception:
        pass

    data = app.graph
    node_types = {}
    edge_types = {}
    feature_shapes = {}

    for ntype in data.node_types:
        count = data[ntype].num_nodes
        node_types[ntype] = count if count is not None else 0
        if hasattr(data[ntype], "x") and data[ntype].x is not None:
            feature_shapes[ntype] = list(data[ntype].x.shape)
        else:
            feature_shapes[ntype] = []

    for etype in data.edge_types:
        src, rel, dst = etype
        key = f"{src} -> {dst} ({rel})"
        edge_types[key] = data[etype].edge_index.shape[1]

    if app.topic_names is None:
        try:
            import os

            topic_json = get_path("data", "topic_json_path")
            if topic_json and os.path.isfile(topic_json):
                with open(topic_json) as f:
                    raw = json.load(f)
                app.topic_names = {
                    int(k): v.get("name", f"Topic {k}") for k, v in raw.items()
                }
                app.topic_info_dict = {int(k): v for k, v in raw.items()}
        except Exception:
            app.topic_names = {}
            app.topic_info_dict = {}

    return GraphInfoOutput(
        node_types=node_types,
        edge_types=edge_types,
        feature_shapes=feature_shapes,
        graph_path=graph_path_str,
    )
