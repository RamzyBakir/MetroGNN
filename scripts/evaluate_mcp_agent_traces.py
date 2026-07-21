"""Evaluate saved MCP/agent traces against fixed MetroGNN tasks."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.evaluation.agent import (  # noqa: E402
    evaluate_traces,
    load_task_specs,
    load_traces,
    summarize_evaluations,
    validate_trace_coverage,
)
from gnn_pipeline.evaluation.agent.trace_benchmark import evaluations_to_rows  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def render_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# MCP Agent Trace Evaluation",
        "",
        f"- Tasks evaluated: {summary['tasks_evaluated']}",
        f"- Overall score: {summary.get('overall_score', 0.0):.4f}",
        f"- Valid tool sequence rate: {summary.get('valid_tool_sequence_rate', 0.0):.4f}",
        "",
        "| Task | Overall | Tool selection | Parameters | Grounding | Valid sequence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['overall_score']:.4f} | "
            f"{row['tool_selection_score']:.4f} | {row['parameter_accuracy']:.4f} | "
            f"{row['grounding_score']:.4f} | {row['valid_tool_sequence']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="agents/benchmarks/mcp_agent_tasks.jsonl",
        help="JSONL task specification file",
    )
    parser.add_argument("--traces", required=True, help="JSONL trace file to evaluate")
    parser.add_argument(
        "--output-dir",
        default="saved_results/mcp_agent_trace_eval",
        help="Directory for evaluation artifacts",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score only the traces present instead of requiring every benchmark task",
    )
    args = parser.parse_args()

    tasks = load_task_specs(str(resolve(args.tasks)))
    traces = load_traces(str(resolve(args.traces)))
    if not args.allow_partial:
        validate_trace_coverage(tasks, traces)
    evaluations = evaluate_traces(tasks, traces)
    rows = evaluations_to_rows(evaluations)
    summary = summarize_evaluations(evaluations)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "task_scores.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "report.md").write_text(render_markdown(summary, rows))

    print(f"MCP/agent trace evaluation saved to: {output_dir}")


if __name__ == "__main__":
    main()
