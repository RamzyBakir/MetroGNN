import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from gnn_pipeline.config import ensure_dirs, get_config, get_path, get_value

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_server")


@dataclass
class AppContext:
    model: Any = None
    graph: Any = None
    topic_assigner: Any = None
    topic_info_dict: Optional[Dict] = None
    topic_names: Optional[Dict] = None
    papers_df: Any = None
    author_to_idx: Optional[Dict] = None
    author_predictions: Optional[Dict[int, Dict]] = None
    importance_logits: Any = None
    role_logits: Any = None
    influence_scores: Any = None
    embeddings: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False
    _graph_loaded: bool = False

    def is_model_ready(self) -> bool:
        return self._model_loaded and self.model is not None

    def is_graph_ready(self) -> bool:
        return self._graph_loaded and self.graph is not None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    ensure_dirs()
    ctx = AppContext()
    logger.info("MetroGNN MCP server started")
    try:
        yield ctx
    finally:
        logger.info("MetroGNN MCP server shutting down")


mcp = FastMCP(
    name="MetroGNN",
    lifespan=app_lifespan,
    instructions=(
        "MetroGNN MCP server — tools for building, querying, and analyzing "
        "a heterogeneous GNN over the DBLP academic citation network. "
        "Typical workflow: build_vector_store -> train_pipeline -> run_inference. "
        "Use detect_topics and analyze_community for deeper exploration."
    ),
)


def _get_ctx(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


from mcp_server.tools import inference, training, analysis, utility  # noqa: E402, F401
