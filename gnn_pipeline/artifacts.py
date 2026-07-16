"""Versioned artifact manifests and compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PIPELINE_SCHEMA_VERSION = 2
GRAPH_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 2
AUTHOR_RESOLUTION_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2
VECTOR_STORE_SCHEMA_VERSION = 2


class ArtifactCompatibilityError(RuntimeError):
    """Raised when persisted artifacts do not belong to the active pipeline."""


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_hash(value: Dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, output)


def read_manifest(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ArtifactCompatibilityError(
            f"Artifact manifest not found: {path}. Build the v2 pipeline first."
        )
    with open(path, "r") as handle:
        manifest = json.load(handle)
    if manifest.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"Unsupported pipeline schema in {path}: "
            f"{manifest.get('pipeline_schema_version')!r}; expected {PIPELINE_SCHEMA_VERSION}."
        )
    return manifest


def create_pipeline_manifest(
    *,
    csv_path: str,
    graph_path: str,
    author_mapping_path: str,
    topic_assignments_path: str,
    topic_info_path: str,
    topic_centroids_path: str,
    embedding_model: str,
    n_clusters: int,
    graph_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Create and return a manifest after graph construction."""
    identity = {
        "dataset_sha256": sha256_file(csv_path),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "author_resolution_version": AUTHOR_RESOLUTION_VERSION,
        "topic_assignments_sha256": sha256_file(topic_assignments_path),
        "topic_info_sha256": sha256_file(topic_info_path),
        "topic_centroids_sha256": sha256_file(topic_centroids_path),
        "embedding_model": embedding_model,
        "n_clusters": int(n_clusters),
    }
    artifact_root = os.path.dirname(os.path.abspath(graph_path))
    manifest = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_id": _canonical_hash(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        **identity,
        "artifacts": {
            "dataset": os.path.relpath(csv_path, artifact_root),
            "graph": os.path.relpath(graph_path, artifact_root),
            "author_mapping": os.path.relpath(author_mapping_path, artifact_root),
            "topic_assignments": os.path.relpath(topic_assignments_path, artifact_root),
            "topic_info": os.path.relpath(topic_info_path, artifact_root),
            "topic_centroids": os.path.relpath(topic_centroids_path, artifact_root),
        },
        "artifact_sha256": {
            "graph": sha256_file(graph_path),
            "author_mapping": sha256_file(author_mapping_path),
            "topic_assignments": identity["topic_assignments_sha256"],
            "topic_info": identity["topic_info_sha256"],
            "topic_centroids": identity["topic_centroids_sha256"],
        },
        "graph_stats": graph_stats,
    }
    return manifest


def validate_pipeline_manifest(
    manifest_path: str,
    *,
    csv_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    verify_large_files: bool = True,
) -> Dict[str, Any]:
    manifest = read_manifest(manifest_path)

    if csv_path and sha256_file(csv_path) != manifest["dataset_sha256"]:
        raise ArtifactCompatibilityError(
            "The configured CSV does not match the graph/model pipeline manifest."
        )

    if graph_path:
        if not os.path.isfile(graph_path):
            raise ArtifactCompatibilityError(f"Knowledge graph not found: {graph_path}")
        expected = manifest.get("artifact_sha256", {}).get("graph")
        if verify_large_files and expected and sha256_file(graph_path) != expected:
            raise ArtifactCompatibilityError(
                "The knowledge graph checksum does not match the pipeline manifest."
            )

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    required_artifacts = [
        "author_mapping",
        "topic_assignments",
        "topic_info",
        "topic_centroids",
    ]
    if manifest.get("artifacts", {}).get("model"):
        required_artifacts.append("model")
    for key in required_artifacts:
        recorded_path = manifest.get("artifacts", {}).get(key)
        path = (
            recorded_path
            if recorded_path and os.path.isabs(recorded_path)
            else os.path.join(manifest_dir, recorded_path or "")
        )
        if not recorded_path or not os.path.isfile(path):
            raise ArtifactCompatibilityError(f"Required v2 artifact is missing: {key}")
        expected = manifest.get("artifact_sha256", {}).get(key)
        if expected and sha256_file(path) != expected:
            raise ArtifactCompatibilityError(
                f"The {key} checksum does not match the pipeline manifest"
            )

    return manifest


def update_manifest_model(
    manifest_path: str, checkpoint_path: str, model_config: Dict[str, Any]
) -> Dict[str, Any]:
    manifest = read_manifest(manifest_path)
    manifest["artifacts"]["model"] = os.path.relpath(
        checkpoint_path, os.path.dirname(os.path.abspath(manifest_path))
    )
    manifest["artifact_sha256"]["model"] = sha256_file(checkpoint_path)
    manifest["model_config"] = model_config
    write_json_atomic(manifest_path, manifest)
    return manifest
