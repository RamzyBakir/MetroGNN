import logging
import os
from typing import Annotated

import pandas as pd
from mcp.server.fastmcp import Context
from pydantic import Field

from gnn_pipeline.artifacts import (
    create_pipeline_manifest,
    sha256_file,
    update_manifest_model,
    validate_pipeline_manifest,
    write_json_atomic,
)
from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.feature_engineering import create_all_features
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.parsing import build_author_index, save_author_mapping
from gnn_pipeline.data.topic_assignment import EMBED_MODEL_NAME, prepare_topics
from gnn_pipeline.data.validation import validate_knowledge_graph, validate_papers_dataframe
from gnn_pipeline.data.vector_store import PaperVectorStore
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
from gnn_pipeline.models.training import set_external_data, train_enhanced_model

from mcp_server.models.schemas import BuildVectorStoreOutput, TrainPipelineOutput
from mcp_server.server import _get_ctx, mcp

logger = logging.getLogger("mcp_server.tools.training")


@mcp.tool()
async def train_pipeline(
    epochs: Annotated[int, Field(ge=1, le=2000)] = 300,
    hidden_channels: Annotated[int, Field(ge=8, le=1024)] = 64,
    out_channels: Annotated[int, Field(ge=8, le=1024)] = 32,
    num_layers: Annotated[int, Field(ge=1, le=8)] = 4,
    n_clusters: Annotated[int, Field(ge=2, le=20)] = 8,
    force_retrain: bool = False,
    force_topics: bool = False,
    ctx: Context = None,
) -> TrainPipelineOutput:
    """Build the validated v2 scholarly graph and train its GNN.

    This is an expensive, state-mutating administrative operation. Topic
    regeneration occurs only when artifacts are absent or force_topics is true.
    """
    app = _get_ctx(ctx)
    ensure_dirs()

    csv_path = get_path("data", "csv_path")
    graph_path = get_path("models", "knowledge_graph")
    model_path = get_path("models", "best_model_new")
    manifest_path = get_path("models", "pipeline_manifest")
    author_mapping_path = get_path("models", "author_mapping")
    assignments_path = get_path("data", "topic_assignments_path")
    topic_info_path = get_path("data", "topic_json_path")
    centroids_path = get_path("data", "topic_centroids_path")

    if not force_retrain and not force_topics and os.path.isfile(model_path) and os.path.isfile(graph_path):
        manifest = validate_pipeline_manifest(
            manifest_path, csv_path=csv_path, graph_path=graph_path
        )
        return TrainPipelineOutput(
            status="completed",
            epochs_trained=0,
            final_train_loss=0.0,
            model_path=model_path,
            graph_path=graph_path,
            graph_stats={
                **manifest.get("graph_stats", {}),
                "message": "Compatible v2 artifacts already exist.",
            },
        )

    await ctx.report_progress(progress=0.0, total=1.0, message="Validating CSV...")
    df = pd.read_csv(csv_path)
    data_stats = validate_papers_dataframe(df)
    dataset_sha = sha256_file(csv_path)

    await ctx.report_progress(progress=0.05, total=1.0, message="Preparing topics...")
    df, topic_info, topic_assigner, _ = prepare_topics(
        df,
        n_clusters=n_clusters,
        assignments_path=assignments_path,
        topic_info_path=topic_info_path,
        centroids_path=centroids_path,
        dataset_sha256=dataset_sha,
        force_regenerate=force_topics,
    )

    await ctx.report_progress(progress=0.15, total=1.0, message="Resolving authors...")
    author_to_idx, author_records, paper_author_indices, author_stats = build_author_index(df)
    df["author_indices"] = paper_author_indices
    df["authors_parsed"] = [
        [author_records[idx]["author_key"] for idx in indices]
        for indices in paper_author_indices
    ]
    save_author_mapping(author_records, author_mapping_path)

    await ctx.report_progress(progress=0.22, total=1.0, message="Engineering features...")
    paper_f, author_f, topic_f, _ = create_all_features(
        df, topic_info, author_to_idx
    )

    await ctx.report_progress(progress=0.28, total=1.0, message="Building graph...")
    data, topic_names, topic_info = build_knowledge_graph_from_dblp(
        df,
        author_to_idx,
        topic_info,
        paper_f,
        author_f,
        topic_f,
        save_path=graph_path,
    )
    graph_stats = validate_knowledge_graph(
        data,
        expected_papers=len(df),
        expected_authors=len(author_records),
        expected_topics=len(topic_info),
    )

    manifest = create_pipeline_manifest(
        csv_path=csv_path,
        graph_path=graph_path,
        author_mapping_path=author_mapping_path,
        topic_assignments_path=assignments_path,
        topic_info_path=topic_info_path,
        topic_centroids_path=centroids_path,
        embedding_model=EMBED_MODEL_NAME,
        n_clusters=len(topic_info),
        graph_stats={**graph_stats, "data_validation": data_stats, "author_identity": author_stats},
    )
    write_json_atomic(manifest_path, manifest)

    model = AdvancedHeteroGCN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_classes=get_value("training", "num_classes"),
        num_topics=len(topic_info),
    )
    set_external_data(df, author_to_idx)

    await ctx.report_progress(progress=0.35, total=1.0, message="Training GNN...")
    model, train_losses, val_metrics, imp_logits, role_logits, inf_scores = train_enhanced_model(
        model,
        data,
        epochs=epochs,
        checkpoint_path=model_path,
        pipeline_id=manifest["pipeline_id"],
    )
    app.pipeline_manifest = update_manifest_model(
        manifest_path, model_path, model.get_config()
    )

    app.model = model
    app.graph = data
    app.topic_assigner = topic_assigner
    app.topic_info_dict = topic_info
    app.topic_names = topic_names
    app.author_to_idx = author_to_idx
    app.author_predictions = None
    app.paper_author_edges = None
    app.paper_topic_edges = None
    app.importance_logits = imp_logits
    app.role_logits = role_logits
    app.influence_scores = inf_scores
    app._model_loaded = True
    app._graph_loaded = True

    await ctx.report_progress(progress=1.0, total=1.0, message="Training complete")
    return TrainPipelineOutput(
        status="completed",
        epochs_trained=len(train_losses),
        final_train_loss=train_losses[-1] if train_losses else 0.0,
        model_path=model_path,
        graph_path=graph_path,
        graph_stats=graph_stats,
    )


@mcp.tool()
async def build_vector_store(
    force_reingest: bool = False,
    ctx: Context = None,
) -> BuildVectorStoreOutput:
    """Build a versioned semantic index after the v2 graph pipeline exists."""
    csv_path = get_path("data", "csv_path")
    assignments_path = get_path("data", "topic_assignments_path")
    manifest = validate_pipeline_manifest(
        get_path("models", "pipeline_manifest"), csv_path=csv_path
    )
    persist_dir = get_path("models", "vector_store")
    vector_manifest_path = get_path("models", "vector_store_manifest")
    store = PaperVectorStore(
        persist_dir=persist_dir, manifest_path=vector_manifest_path
    )

    if not force_reingest and store.collection.count() > 0:
        store.validate_compatibility(
            csv_path=csv_path,
            topic_assignments_path=assignments_path,
            pipeline_id=manifest["pipeline_id"],
        )
        return BuildVectorStoreOutput(
            status="skipped",
            papers_indexed=store.collection.count(),
            persist_dir=persist_dir,
        )

    await ctx.report_progress(progress=0.0, total=1.0, message="Embedding papers...")
    count = store.ingest_papers(
        csv_path=csv_path,
        force_reingest=force_reingest,
        topic_assignments_path=assignments_path,
        pipeline_id=manifest["pipeline_id"],
    )
    await ctx.report_progress(progress=1.0, total=1.0, message="Index complete")
    return BuildVectorStoreOutput(
        status="completed", papers_indexed=count, persist_dir=persist_dir
    )
