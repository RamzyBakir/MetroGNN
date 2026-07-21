"""Profile MetroGNN artifact loading and serving-time inference."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import validate_pipeline_manifest  # noqa: E402
from gnn_pipeline.config import ensure_dirs, get_path  # noqa: E402
from gnn_pipeline.inference import run_inference  # noqa: E402
from gnn_pipeline.models.checkpoint import load_model_for_pipeline  # noqa: E402


def _ms(values):
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "runs": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[p95_index],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="Timed forward passes")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup forward passes")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--query", default=None, help="Optional end-to-end query")
    parser.add_argument("--top-papers", type=int, default=5)
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument(
        "--output",
        default="saved_results/serving_profile.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is not available")

    ensure_dirs()
    manifest_path = get_path("models", "pipeline_manifest")
    graph_path = get_path("models", "knowledge_graph")
    model_path = get_path("models", "best_model_new")

    manifest, manifest_ms = _time_call(
        lambda: validate_pipeline_manifest(
            manifest_path,
            csv_path=get_path("data", "csv_path"),
            graph_path=graph_path,
        )
    )
    graph, graph_load_ms = _time_call(
        lambda: torch.load(graph_path, map_location=args.device, weights_only=False)
    )
    model, model_load_ms = _time_call(
        lambda: load_model_for_pipeline(
            model_path, manifest_path, map_location=args.device
        )[0]
    )

    model.eval()
    for _ in range(max(args.warmup, 0)):
        with torch.no_grad():
            model(graph.x_dict, graph.edge_index_dict)

    forward_times = []
    for _ in range(max(args.runs, 1)):
        start = time.perf_counter()
        with torch.no_grad():
            model(graph.x_dict, graph.edge_index_dict)
        if args.device == "cuda":
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000.0)

    result = {
        "pipeline_id": manifest["pipeline_id"],
        "device": args.device,
        "node_types": {node_type: int(graph[node_type].num_nodes or 0) for node_type in graph.node_types},
        "edge_types": {
            f"{src}->{dst}:{rel}": int(graph[(src, rel, dst)].edge_index.shape[1])
            for src, rel, dst in graph.edge_types
        },
        "artifact_validation_ms": manifest_ms,
        "graph_load_ms": graph_load_ms,
        "model_load_ms": model_load_ms,
        "gnn_forward": _ms(forward_times),
    }

    if args.query:
        _, query_ms = _time_call(
            lambda: run_inference(
                query=args.query,
                top_papers=args.top_papers,
                retrieval_k=args.retrieval_k,
            )
        )
        result["end_to_end_query"] = {
            "query": args.query,
            "top_papers": args.top_papers,
            "retrieval_k": args.retrieval_k,
            "latency_ms": query_ms,
        }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Serving profile saved to: {output_path}")


if __name__ == "__main__":
    main()
