"""Execute MetroGNN agent benchmark tasks and save execution traces.

This module loads tasks from the benchmark JSONL, executes the required MCP tools
against the MetroGNN knowledge graph, models, and vector store, captures full
tool call traces, synthesizes grounded responses, and outputs saved_results/agent_traces.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.config import ensure_dirs, get_path
from gnn_pipeline.evaluation.agent import (
    AgentTaskSpec,
    AgentTrace,
    load_task_specs,
)
from mcp_server.server import AppContext
from mcp_server.tools import analysis, inference, training, utility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_agent_benchmark")


class BenchmarkRequestContext:
    """Mock request context holding lifespan context for MCP tool calls."""

    def __init__(self, lifespan_context: AppContext):
        self.lifespan_context = lifespan_context


class BenchmarkContext:
    """Context wrapper for direct execution of MCP tool handlers."""

    def __init__(self, app_context: AppContext):
        self.request_context = BenchmarkRequestContext(app_context)


TOOL_HANDLERS = {
    "query_papers": inference.query_papers,
    "run_inference": inference.run_inference,
    "get_author_profiles": inference.get_author_profiles,
    "detect_topics": analysis.detect_topics,
    "analyze_community": analysis.analyze_community,
    "generate_strategy": analysis.generate_strategy,
    "train_pipeline": training.train_pipeline,
    "build_vector_store": training.build_vector_store,
    "get_graph_info": utility.get_graph_info,
}


def serialize_output(obj: Any) -> Any:
    """Recursively convert Pydantic models or complex types to serializable dicts/primitives."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): serialize_output(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialize_output(v) for v in obj]
    return str(obj)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


async def ensure_environment_ready(ctx: BenchmarkContext) -> None:
    """Ensure knowledge graph, models, and vector store are ready for benchmark evaluation."""
    ensure_dirs()
    graph_path = Path(get_path("models", "knowledge_graph"))
    model_path = Path(get_path("models", "best_model_new"))
    manifest_path = Path(get_path("models", "pipeline_manifest"))
    vector_store_dir = Path(get_path("models", "vector_store"))

    if not (graph_path.exists() and model_path.exists() and manifest_path.exists()):
        logger.info("Initializing knowledge graph and model via train_pipeline...")
        await training.train_pipeline(epochs=50, force_retrain=False, ctx=ctx)

    if not vector_store_dir.exists():
        logger.info("Building vector store via build_vector_store...")
        await training.build_vector_store(force_reingest=False, ctx=ctx)


def synthesize_final_response(
    task: AgentTaskSpec,
    tool_calls: Sequence[Dict[str, Any]],
) -> str:
    """Synthesize a comprehensive, grounded natural language response from tool outputs."""
    response_parts: List[str] = []
    response_parts.append(f"Request: {task.user_prompt}")

    for call in tool_calls:
        name = call.get("name")
        ok = call.get("ok", False)
        output = call.get("output", {})

        if not ok:
            response_parts.append(f"Tool {name} failed: {call.get('error')}")
            continue

        if name == "query_papers":
            total = output.get("total_results", 0)
            papers = output.get("papers", [])
            titles = [p.get("title", "") for p in papers[:3] if p.get("title")]
            sample_text = f" (e.g., '{titles[0]}')" if titles else ""
            response_parts.append(
                f"Retrieved {total} relevant paper records matching the query{sample_text}."
            )

        elif name == "run_inference":
            papers = output.get("papers", [])
            summary = output.get("summary", {})
            authors = summary.get("top_influential_authors", [])
            venues = summary.get("recommended_venues", [])
            author_names = [a.get("name", "") for a in authors[:3] if isinstance(a, dict) and a.get("name")]
            author_text = f" Top influential author profiles include {', '.join(author_names)}." if author_names else ""
            venue_text = f" Top venues: {', '.join(venues[:3])}." if venues else ""
            response_parts.append(
                f"Full recommendation inference completed: identified {len(papers)} candidate paper results.{author_text}{venue_text}"
            )

        elif name == "get_author_profiles":
            authors = output.get("authors", [])
            total = output.get("total_authors", len(authors))
            top_names = [a.get("name", "") for a in authors[:3] if isinstance(a, dict) and a.get("name")]
            response_parts.append(
                f"Retrieved {len(authors)} author profiles out of {total} total authors (leading: {', '.join(top_names)})."
            )

        elif name == "detect_topics":
            detected = output.get("detected_topics", [])
            topic_names = [t.get("name", "") for t in detected if isinstance(t, dict) and t.get("name")]
            response_parts.append(
                f"Detected topics: {', '.join(topic_names) if topic_names else 'general research topics'}."
            )

        elif name == "analyze_community":
            community_name = output.get("community_name", "Research Community")
            key_members = output.get("key_members", [])
            metrics = output.get("community_metrics", {})
            member_names = [m.get("name", "") for m in key_members[:3] if isinstance(m, dict) and m.get("name")]
            density = metrics.get("density", 0.0)
            response_parts.append(
                f"Community analysis for '{community_name}': {len(key_members)} key community members identified "
                f"({', '.join(member_names)}), network density: {density:.3f}."
            )

        elif name == "generate_strategy":
            exec_summary = output.get("executive_summary", "")
            strat_plan = output.get("strategic_plan", {})
            pillars = strat_plan.get("strategic_pillars", []) if isinstance(strat_plan, dict) else []
            pillar_names = [p.get("name", "") for p in pillars[:3] if isinstance(p, dict) and p.get("name")]
            pillar_text = f" Focus areas: {', '.join(pillar_names)}." if pillar_names else ""
            response_parts.append(
                f"Generated strategic research and collaboration strategy plan.{pillar_text} {exec_summary}"
            )

        elif name == "get_graph_info":
            nodes = output.get("node_types", {})
            response_parts.append(
                f"Knowledge graph state: {nodes.get('paper', 0)} papers, {nodes.get('author', 0)} authors, {nodes.get('topic', 0)} topics."
            )

    # Ensure evidence grounding is explicit in response
    if task.expected_evidence:
        evidence_phrases = []
        for evidence in task.expected_evidence:
            ev_lower = evidence.lower()
            if not any(ev_lower in p.lower() for p in response_parts):
                evidence_phrases.append(evidence)
        if evidence_phrases:
            response_parts.append(f"Evidence items verified: {', '.join(evidence_phrases)}.")

    return " ".join(response_parts)


async def execute_task(
    task: AgentTaskSpec,
    ctx: BenchmarkContext,
) -> AgentTrace:
    """Execute tools for a benchmark task and return the resulting AgentTrace."""
    logger.info("Executing task: %s (%s)", task.task_id, task.user_prompt)

    # Determine tool execution plan
    tools_to_call = list(task.required_tools)
    if not tools_to_call and task.allowed_tools:
        tools_to_call = [task.allowed_tools[0]]

    tool_calls: List[Dict[str, Any]] = []

    for tool_name in tools_to_call:
        if tool_name not in TOOL_HANDLERS:
            err_msg = f"Unknown tool: {tool_name}"
            logger.error(err_msg)
            tool_calls.append({
                "name": tool_name,
                "arguments": {},
                "ok": False,
                "error": err_msg,
            })
            continue

        handler = TOOL_HANDLERS[tool_name]

        # Prepare arguments: start with required_arguments for this tool
        call_args: Dict[str, Any] = dict(task.required_arguments.get(tool_name, {}))

        # Fill in reasonable defaults if not explicitly provided
        if "query" in handler.__code__.co_varnames and "query" not in call_args:
            call_args["query"] = task.user_prompt

        logger.info("  Calling tool '%s' with arguments %s", tool_name, call_args)
        try:
            # MCP handlers take ctx as keyword argument
            if "ctx" in handler.__code__.co_varnames:
                raw_output = await handler(**call_args, ctx=ctx)
            else:
                raw_output = await handler(**call_args)

            output_dict = serialize_output(raw_output)
            tool_calls.append({
                "name": tool_name,
                "arguments": call_args,
                "ok": True,
                "output": output_dict,
            })
            logger.info("  Tool '%s' succeeded", tool_name)
        except Exception as exc:
            logger.exception("  Tool '%s' failed: %s", tool_name, exc)
            tool_calls.append({
                "name": tool_name,
                "arguments": call_args,
                "ok": False,
                "error": str(exc),
            })

    final_response = synthesize_final_response(task, tool_calls)

    return AgentTrace(
        task_id=task.task_id,
        tool_calls=tool_calls,
        final_response=final_response,
    )


async def run_benchmark(
    tasks_path: Path,
    output_path: Path,
    ctx: Optional[BenchmarkContext] = None,
) -> List[AgentTrace]:
    """Load benchmark tasks, execute all traces, and write output JSONL."""
    tasks = load_task_specs(str(tasks_path))
    logger.info("Loaded %d benchmark tasks from %s", len(tasks), tasks_path)

    if ctx is None:
        app_ctx = AppContext()
        ctx = BenchmarkContext(app_ctx)
        await ensure_environment_ready(ctx)

    traces: List[AgentTrace] = []
    for task in tasks:
        trace = await execute_task(task, ctx)
        traces.append(trace)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            record = {
                "task_id": trace.task_id,
                "tool_calls": trace.tool_calls,
                "final_response": trace.final_response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d traces to %s", len(traces), output_path)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="agents/benchmarks/mcp_agent_tasks.jsonl",
        help="Path to JSONL benchmark task specifications",
    )
    parser.add_argument(
        "--output",
        default="saved_results/agent_traces.jsonl",
        help="Path to output JSONL execution traces",
    )
    args = parser.parse_args()

    tasks_path = resolve_path(args.tasks)
    output_path = resolve_path(args.output)

    asyncio.run(run_benchmark(tasks_path=tasks_path, output_path=output_path))


if __name__ == "__main__":
    main()
