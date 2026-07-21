"""Profile MetroGNN target construction and training-step latency."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.artifacts import validate_pipeline_manifest  # noqa: E402
from gnn_pipeline.config import ensure_dirs, get_path, get_value  # noqa: E402
from gnn_pipeline.data.parsing import load_author_mapping  # noqa: E402
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN  # noqa: E402
from gnn_pipeline.models.training import (  # noqa: E402
    create_author_split_indices,
    create_enhanced_training_targets,
    set_training_seed,
    set_external_data,
)


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _model_config(manifest: dict, graph) -> dict:
    if manifest.get("model_config"):
        return dict(manifest["model_config"])
    return {
        "hidden_channels": get_value("training", "hidden_channels"),
        "out_channels": get_value("training", "out_channels"),
        "num_layers": get_value("training", "num_layers"),
        "num_classes": get_value("training", "num_classes"),
        "num_topics": int(graph["topic"].num_nodes or 0),
    }


def _training_step(model, graph, targets, train_idx, optimizer):
    importance_targets, role_targets, influence_targets = targets
    optimizer.zero_grad(set_to_none=True)
    _, importance_logits, role_logits, influence_scores = model(
        graph.x_dict, graph.edge_index_dict
    )
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()
    loss_importance = criterion_class(
        importance_logits[train_idx], importance_targets[train_idx]
    )
    loss_role = criterion_class(role_logits[train_idx], role_targets[train_idx])
    loss_influence = criterion_reg(
        influence_scores[train_idx].squeeze(), influence_targets[train_idx]
    )
    total_loss = loss_importance * 0.5 + loss_role * 0.3 + loss_influence * 0.2
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "total_loss": float(total_loss.detach().cpu()),
        "importance_loss": float(loss_importance.detach().cpu()),
        "role_loss": float(loss_role.detach().cpu()),
        "influence_loss": float(loss_influence.detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="Timed optimizer steps")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup steps")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="saved_results/training_profile.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is not available")

    ensure_dirs()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    manifest_path = get_path("models", "pipeline_manifest")
    graph_path = get_path("models", "knowledge_graph")
    csv_path = get_path("data", "csv_path")

    manifest, manifest_ms = _time_call(
        lambda: validate_pipeline_manifest(
            manifest_path,
            csv_path=csv_path,
            graph_path=graph_path,
        )
    )
    graph, graph_load_ms = _time_call(
        lambda: torch.load(graph_path, map_location="cpu", weights_only=False)
    )

    papers_df, csv_load_ms = _time_call(lambda: pd.read_csv(csv_path))
    author_records = load_author_mapping(get_path("models", "author_mapping"))
    author_to_idx = {
        record["author_key"]: int(author_idx)
        for author_idx, record in author_records.items()
    }
    set_external_data(papers_df, author_to_idx)
    targets, target_ms = _time_call(lambda: create_enhanced_training_targets(graph))

    model = AdvancedHeteroGCN(**_model_config(manifest, graph)).to(device)
    graph = graph.to(device)
    targets = tuple(target.to(device) for target in targets)
    train_idx, val_idx, test_idx = create_author_split_indices(
        len(targets[0]), seed=args.seed
    )
    train_idx = train_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()

    last_losses = {}
    for _ in range(max(args.warmup, 0)):
        last_losses = _training_step(model, graph, targets, train_idx, optimizer)
    _synchronize(device)

    step_times = []
    for _ in range(max(args.steps, 1)):
        _synchronize(device)
        start = time.perf_counter()
        last_losses = _training_step(model, graph, targets, train_idx, optimizer)
        _synchronize(device)
        step_times.append((time.perf_counter() - start) * 1000.0)

    result = {
        "pipeline_id": manifest["pipeline_id"],
        "device": args.device,
        "seed": args.seed,
        "node_types": {
            node_type: int(graph[node_type].num_nodes or 0)
            for node_type in graph.node_types
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "artifact_validation_ms": manifest_ms,
        "graph_load_ms": graph_load_ms,
        "csv_load_ms": csv_load_ms,
        "target_construction_ms": target_ms,
        "training_step": _ms(step_times),
        "last_losses": last_losses,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Training profile saved to: {output_path}")


if __name__ == "__main__":
    main()
