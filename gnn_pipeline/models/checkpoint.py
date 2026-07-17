import os
from typing import Iterable, Tuple

import torch


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
}


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


def load_compatible_state_dict(
    candidate_paths: Iterable[str], graph, map_location: str = "cpu"
) -> Tuple[str, dict]:
    """Load the first checkpoint whose first-layer feature widths match graph."""
    seen = set()
    skipped = []

    for path in candidate_paths:
        if not path or path in seen:
            continue
        seen.add(path)

        if not os.path.isfile(path):
            skipped.append(f"{path} (missing)")
            continue

        state_dict = torch.load(path, map_location=map_location, weights_only=False)
        matches, reason = checkpoint_matches_graph(state_dict, graph)
        if matches:
            return path, state_dict

        skipped.append(f"{path} ({reason})")

    details = "; ".join(skipped) if skipped else "no candidate checkpoints provided"
    raise RuntimeError(f"No checkpoint compatible with the saved graph: {details}")
