"""Safe parsing and stable identity helpers for DBLP records."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Return a conservative normalized representation for identity matching."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return _WHITESPACE.sub(" ", str(value).strip()).casefold()


def parse_list_field(value: Any) -> List[Any]:
    """Parse a serialized list without executing arbitrary input."""
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _fallback_author_key(name: str, organization: str) -> str:
    material = f"{normalize_text(name)}\x1f{normalize_text(organization)}"
    return f"fallback:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def canonicalize_author(value: Any) -> Dict[str, str] | None:
    """Convert an author dictionary/string into a stable canonical record."""
    if isinstance(value, dict):
        dblp_id = str(value.get("id", "") or "").strip()
        name = str(value.get("name", "") or "").strip()
        organization = str(value.get("org", "") or "").strip()
    else:
        dblp_id = ""
        name = str(value or "").strip()
        organization = ""

    if not dblp_id and not name:
        return None

    key = f"dblp:{dblp_id}" if dblp_id else _fallback_author_key(name, organization)
    return {
        "author_key": key,
        "dblp_id": dblp_id,
        "name": name or dblp_id,
        "organization": organization,
        "identity_source": "dblp_id" if dblp_id else "name_org_fallback",
    }


def parse_authors(value: Any) -> List[Dict[str, str]]:
    """Parse and deduplicate authors from one paper, preserving input order."""
    authors: List[Dict[str, str]] = []
    seen = set()
    for raw_author in parse_list_field(value):
        author = canonicalize_author(raw_author)
        if author is None or author["author_key"] in seen:
            continue
        seen.add(author["author_key"])
        authors.append(author)
    return authors


def build_author_index(
    df: pd.DataFrame,
) -> Tuple[Dict[str, int], List[Dict[str, Any]], List[List[int]], Dict[str, int]]:
    """Build a deterministic author index and per-paper author indices."""
    variants: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {"name": Counter(), "organization": Counter(), "dblp_id": Counter(), "identity_source": Counter()}
    )
    parsed_rows: List[List[Dict[str, str]]] = []
    malformed_rows = 0

    for value in df.get("authors", pd.Series([], dtype=object)):
        raw = parse_list_field(value)
        if not raw and str(value).strip() not in {"", "[]", "nan", "None"}:
            malformed_rows += 1
        authors = parse_authors(value)
        parsed_rows.append(authors)
        for author in authors:
            entry = variants[author["author_key"]]
            for field in ("name", "organization", "dblp_id", "identity_source"):
                if author[field]:
                    entry[field][author[field]] += 1

    keys = sorted(variants)
    author_to_idx = {key: idx for idx, key in enumerate(keys)}

    def most_common(counter: Counter) -> str:
        if not counter:
            return ""
        return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]

    records: List[Dict[str, Any]] = []
    for key in keys:
        entry = variants[key]
        records.append(
            {
                "author_idx": author_to_idx[key],
                "author_key": key,
                "dblp_id": most_common(entry["dblp_id"]),
                "name": most_common(entry["name"]),
                "organization": most_common(entry["organization"]),
                "identity_source": most_common(entry["identity_source"]),
            }
        )

    paper_author_indices = [
        [author_to_idx[author["author_key"]] for author in authors]
        for authors in parsed_rows
    ]
    stats = {
        "num_authors": len(records),
        "dblp_id_authors": sum(r["identity_source"] == "dblp_id" for r in records),
        "fallback_authors": sum(
            r["identity_source"] == "name_org_fallback" for r in records
        ),
        "malformed_author_rows": malformed_rows,
        "papers_without_authors": sum(not indices for indices in paper_author_indices),
    }
    return author_to_idx, records, paper_author_indices, stats


def save_author_mapping(records: Iterable[Dict[str, Any]], path: str) -> None:
    """Persist the graph author index as an inspectable JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(records), indent=2, ensure_ascii=False))


def load_author_mapping(path: str) -> Dict[int, Dict[str, Any]]:
    records = json.loads(Path(path).read_text())
    return {int(record["author_idx"]): record for record in records}
