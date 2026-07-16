"""Safe, versioned MetroGNN checkpoint helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch

from gnn_pipeline.artifacts import (
    ArtifactCompatibilityError,
    CHECKPOINT_SCHEMA_VERSION,
    read_manifest,
)
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN


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
