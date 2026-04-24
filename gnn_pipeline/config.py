"""
Centralised configuration loader for MetroGNN.

Reads ``config.yaml`` from the project root once, resolves every relative
path to an absolute one (relative to the project root), and exposes simple
accessor functions.

Usage
-----
    from gnn_pipeline.config import get_config, get_path

    cfg = get_config()                       # full dict
    csv = get_path("data.csv_path")           # resolved absolute path
    epochs = cfg["training"]["epochs"]        # nested scalar value
"""

import os
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml


def _find_project_root() -> str:
    """Walk up from this file until we find ``config.yaml``."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(current, "config.yaml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                "config.yaml not found in any parent directory. "
                "Make sure it exists at the project root."
            )
        current = parent


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """Load and cache the full configuration dictionary.

    Relative paths are **not** resolved here — use :func:`get_path` for that.
    """
    root = _find_project_root()
    config_path = os.path.join(root, "config.yaml")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def get_path(*keys: str) -> str:
    """Return an **absolute** path from the config.

    Parameters
    ----------
    *keys
        Dot-traversal keys, e.g. ``"data", "csv_path"`` or pass a single
        dotted string ``"data.csv_path"``.

    Returns
    -------
    str
        Absolute path resolved relative to the project root.

    Examples
    --------
    >>> get_path("data", "csv_path")
    '/home/user/MetroGNN/train/DBLP-Citation-network-V18-1-20-25_clean.csv'
    >>> get_path("data.csv_path")
    # same result
    """
    if len(keys) == 1 and "." in keys[0]:
        keys = tuple(keys[0].split("."))

    cfg = get_config()
    value: Any = cfg
    for key in keys:
        value = value[key]

    if not value:
        return value

    root = _find_project_root()
    if not os.path.isabs(value):
        value = os.path.join(root, value)
    return os.path.normpath(value)


def get_value(*keys: str) -> Any:
    """Return a raw (non-path) scalar value from the config.

    Traversal semantics are identical to :func:`get_path` but no path
    resolution is performed.
    """
    if len(keys) == 1 and "." in keys[0]:
        keys = tuple(keys[0].split("."))

    cfg = get_config()
    value: Any = cfg
    for key in keys:
        value = value[key]
    return value


def ensure_dirs() -> None:
    """Create all configured directories that do not yet exist."""
    root = _find_project_root()
    cfg = get_config()
    dirs_to_create = set()

    for section in ("models", "output"):
        section_cfg = cfg.get(section, {})
        if "dir" in section_cfg:
            dirs_to_create.add(section_cfg["dir"])
        for _key, val in section_cfg.items():
            if isinstance(val, str) and "/" in val:
                parent = os.path.dirname(val)
                if parent:
                    dirs_to_create.add(parent)

    for d in dirs_to_create:
        abs_dir = os.path.join(root, d) if not os.path.isabs(d) else d
        os.makedirs(abs_dir, exist_ok=True)
