"""Safe, versioned MetroGNN checkpoint helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import torch

from gnn_pipeline.artifacts import (
    ArtifactCompatibilityError,
    CHECKPOINT_SCHEMA_VERSION,
    read_manifest,
)
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN


_FIRST_LAYER_INPUTS = {
    "convs.0.convs.<author___authored___paper>.lin_l.weight": "author",
    "convs.0.convs.<author___authored___paper>.lin_r.weight": "paper",
    "convs.0.convs.<paper___authored_by___author>.lin_l.weight": "paper",
    "convs.0.convs.<paper___authored_by___author>.lin_r.weight": "author",
    "convs.0.convs.<paper___has_topic___topic>.lin_l.weight": "paper",
    "convs.0.convs.<paper___has_topic___topic>.lin_r.weight": "topic",
    "convs.0.convs.<topic___has_paper___paper>.lin_l.weight": "topic",
    "convs.0.convs.<topic___has_paper___paper>.lin_r.weight": "paper",
    "convs.0.convs.<author___collaborates_with___author>.lin_l.weight": "author",
    "convs.0.convs.<author___collaborates_with___author>.lin_r.weight": "author",
    "convs.0.convs.<paper___cites___paper>.lin_l.weight": "paper",
    "convs.0.convs.<paper___cites___paper>.lin_r.weight": "paper",
    "convs.0.convs.<paper___cited_by___paper>.lin_l.weight": "paper",
    "convs.0.convs.<paper___cited_by___paper>.lin_r.weight": "paper",
}


def save_model_checkpoint(
    model: AdvancedHeteroGCN,
    path: str,
    *,
    pipeline_id: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "pipeline_id": pipeline_id,
            "model_config": model.get_config(),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint_payload(
    path: str, *, map_location: str | torch.device = "cpu"
) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ArtifactCompatibilityError(f"Model checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ArtifactCompatibilityError(
            "Legacy unversioned model checkpoint detected. Rebuild and retrain the v2 pipeline."
        )
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"Unsupported checkpoint schema: {payload.get('checkpoint_schema_version')!r}"
        )
    return payload


def load_model_for_pipeline(
    checkpoint_path: str,
    manifest_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[AdvancedHeteroGCN, Dict[str, Any]]:
    manifest = read_manifest(manifest_path)
    payload = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    if payload.get("pipeline_id") != manifest.get("pipeline_id"):
        raise ArtifactCompatibilityError(
            "The model checkpoint and knowledge graph belong to different pipelines."
        )
    model = AdvancedHeteroGCN(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def checkpoint_matches_graph(state_dict: dict, graph) -> Tuple[bool, str]:
    """Check whether a lazy PyG checkpoint was initialized for this graph."""
    feature_dims = {
        node_type: graph[node_type].x.shape[1]
        for node_type in graph.node_types
        if hasattr(graph[node_type], "x") and graph[node_type].x is not None
    }

    for key, node_type in _FIRST_LAYER_INPUTS.items():
        tensor = state_dict.get(key)
        if tensor is None or node_type not in feature_dims:
            continue

        actual_dim = tensor.shape[1]
        expected_dim = feature_dims[node_type]
        if actual_dim != expected_dim:
            return (
                False,
                f"{os.path.basename(key)} expects {node_type} dim {actual_dim}, "
                f"but graph has {expected_dim}",
            )

    return True, ""


def _state_dict_from_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, dict):
            raise ArtifactCompatibilityError("Checkpoint state_dict is malformed")
        return state_dict
    if isinstance(payload, dict):
        return payload
    raise ArtifactCompatibilityError("Checkpoint payload is not a state dictionary")


def load_compatible_state_dict(
    candidate_paths: Iterable[str],
    graph,
    map_location: str | torch.device = "cpu",
) -> Tuple[str, Dict[str, Any]]:
    """Load the first candidate whose first-layer feature widths match graph."""
    seen = set()
    skipped = []

    for path in candidate_paths:
        if not path or path in seen:
            continue
        seen.add(path)

        if not os.path.isfile(path):
            skipped.append(f"{path} (missing)")
            continue

        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            state_dict = _state_dict_from_payload(payload)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            skipped.append(f"{path} ({exc})")
            continue

        matches, reason = checkpoint_matches_graph(state_dict, graph)
        if matches:
            return path, state_dict

        skipped.append(f"{path} ({reason})")

    details = "; ".join(skipped) if skipped else "no candidate checkpoints provided"
    raise RuntimeError(f"No checkpoint compatible with the saved graph: {details}")
