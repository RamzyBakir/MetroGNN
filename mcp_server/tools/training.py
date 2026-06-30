import ast
import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from mcp.server.fastmcp import Context

from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.feature_engineering import create_all_features
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.topic_assignment import TopicAssigner, assign_topics_to_dblp
from gnn_pipeline.data.vector_store import PaperVectorStore
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
from gnn_pipeline.models.training import set_external_data, train_enhanced_model

from mcp_server.models.schemas import (
    BuildVectorStoreInput,
    BuildVectorStoreOutput,
    TrainPipelineInput,
    TrainPipelineOutput,
)
from mcp_server.server import _get_ctx, mcp

logger = logging.getLogger("mcp_server.tools.training")


def _parse_authors(authors_str: str):
    if pd.isna(authors_str):
        return []
    try:
        result = ast.literal_eval(str(authors_str))
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    return []


@mcp.tool()
async def train_pipeline(
    epochs: int = 300,
    hidden_channels: int = 64,
    out_channels: int = 32,
    num_layers: int = 4,
    n_clusters: int = 8,
    force_retrain: bool = False,
    ctx: Context = None,
) -> TrainPipelineOutput:
    """Run the full MetroGNN training pipeline end-to-end.

    Loads the DBLP CSV, assigns topics via sentence-embedding clustering,
    engineers features, builds the heterogeneous knowledge graph, and trains
    the GNN model. Saves model weights, the knowledge graph, and topic info
    to disk. This is a long-running operation (5-30 minutes).

    Args:
        epochs: Number of training epochs (1-2000, default from config).
        hidden_channels: GNN hidden layer size.
        out_channels: GNN output embedding size.
        num_layers: Number of GNN convolution layers.
        n_clusters: Number of topic clusters for K-Means.
        force_retrain: If True, retrain even if a model file already exists.
    """
    app = _get_ctx(ctx)
    ensure_dirs()

    model_path = get_path("models", "best_model_new")
    graph_path = get_path("models", "knowledge_graph")

    if not force_retrain and os.path.isfile(model_path) and os.path.isfile(graph_path):
        return TrainPipelineOutput(
            status="completed",
            epochs_trained=0,
            final_train_loss=0.0,
            model_path=model_path,
            graph_path=graph_path,
            graph_stats={
                "message": "Model and graph already exist. Use force_retrain=True to retrain."
            },
        )

    csv_path = get_path("data", "csv_path")
    await ctx.report_progress(progress=0.0, total=1.0, message="Loading CSV data...")

    df = pd.read_csv(csv_path)

    all_authors = set()
    for idx, row in df.iterrows():
        for author in _parse_authors(row.get("authors", "[]")):
            all_authors.add(str(author).strip())
    author_to_idx = {name: idx for idx, name in enumerate(sorted(all_authors))}
    df["author_indices"] = df["authors"].apply(
        lambda x: [author_to_idx[a] for a in _parse_authors(x) if a in author_to_idx]
    )

    await ctx.report_progress(progress=0.05, total=1.0, message="Assigning topics...")

    topic_json_path = (
        get_value("data", "topic_json_path")
        if get_value("data", "topic_json_path")
        else None
    )
    if topic_json_path and os.path.isfile(topic_json_path):
        with open(topic_json_path) as f:
            raw = json.load(f)
        topic_info_dict = {int(k): v for k, v in raw.items()}
        topic_names = {int(k): v.get("name", f"Topic {k}") for k, v in raw.items()}
        if "topic_id" not in df.columns:
            df["topic_id"] = 0
    else:
        df, results_dict, topic_info_dict = assign_topics_to_dblp(
            df, n_clusters=n_clusters
        )
        topic_names = {
            tid: info.get("name", f"Topic {tid}")
            for tid, info in topic_info_dict.items()
        }
        if topic_json_path:
            os.makedirs(os.path.dirname(topic_json_path), exist_ok=True)
            with open(topic_json_path, "w") as f:
                json.dump(topic_info_dict, f, indent=2)

    app.topic_info_dict = topic_info_dict
    app.topic_names = topic_names

    await ctx.report_progress(
        progress=0.15, total=1.0, message="Engineering features..."
    )

    paper_f, author_f, topic_f, feature_info = create_all_features(
        df, topic_info_dict, author_to_idx
    )

    await ctx.report_progress(
        progress=0.25, total=1.0, message="Building knowledge graph..."
    )

    data, topic_names, topic_info_dict = build_knowledge_graph_from_dblp(
        df,
        author_to_idx,
        topic_info_dict,
        paper_f,
        author_f,
        topic_f,
        save_path=graph_path,
    )

    graph_stats = {
        "num_papers": int(data["paper"].num_nodes or 0),
        "num_authors": int(data["author"].num_nodes or 0),
        "num_topics": int(data["topic"].num_nodes or 0),
        "edge_types": [str(et) for et in data.edge_types],
    }

    await ctx.report_progress(progress=0.3, total=1.0, message="Initializing model...")

    num_classes = (
        get_value("training", "num_classes")
        if get_value("training", "num_classes")
        else 3
    )
    model = AdvancedHeteroGCN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_classes=num_classes,
        num_topics=n_clusters,
    )

    await ctx.report_progress(
        progress=0.35, total=1.0, message=f"Training for {epochs} epochs..."
    )

    set_external_data(df, author_to_idx)

    model, train_losses, val_metrics, imp_logits, role_logits, inf_scores = (
        train_enhanced_model(
            model,
            data,
            epochs=epochs,
            checkpoint_path=model_path,
        )
    )

    app.model = model
    app._model_loaded = True
    app.graph = data
    app._graph_loaded = True
    app.author_predictions = None
    app.importance_logits = imp_logits
    app.role_logits = role_logits
    app.influence_scores = inf_scores
    app.author_to_idx = author_to_idx

    await ctx.report_progress(progress=1.0, total=1.0, message="Training complete!")

    final_loss = train_losses[-1] if train_losses else 0.0
    actual_epochs = len(train_losses)

    return TrainPipelineOutput(
        status="completed",
        epochs_trained=actual_epochs,
        final_train_loss=final_loss,
        model_path=model_path,
        graph_path=graph_path,
        graph_stats=graph_stats,
    )


@mcp.tool()
async def build_vector_store(
    force_reingest: bool = False,
    ctx: Context = None,
) -> BuildVectorStoreOutput:
    """Ingest DBLP papers into the ChromaDB vector store for semantic search.

    Embeds all paper titles + venues + keywords using MiniLM and stores them
    in a persistent ChromaDB collection. This is a one-time operation
    (1-10 minutes) that must complete before query_papers or run_inference
    can be used.

    Args:
        force_reingest: If True, delete existing store and re-ingest from scratch.
    """
    csv_path = get_path("data", "csv_path")
    persist_dir = get_path("models", "vector_store")

    store = PaperVectorStore(persist_dir=persist_dir)

    if not force_reingest and store.collection.count() > 0:
        return BuildVectorStoreOutput(
            status="skipped",
            papers_indexed=store.collection.count(),
            persist_dir=persist_dir,
        )

    await ctx.report_progress(
        progress=0.0, total=1.0, message="Ingesting papers into vector store..."
    )

    count = store.ingest_papers(csv_path=csv_path, force_reingest=force_reingest)

    await ctx.report_progress(
        progress=1.0, total=1.0, message=f"Done. {count} papers indexed."
    )

    return BuildVectorStoreOutput(
        status="completed",
        papers_indexed=count,
        persist_dir=persist_dir,
    )
