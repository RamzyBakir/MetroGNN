"""Dataset-time helpers for reproducible temporal features."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def infer_observation_year(df: pd.DataFrame, year_column: str = "year") -> int:
    """Return the latest valid publication year in a dataset.

    Falling back to the current UTC year keeps small synthetic frames usable, while
    real DBLP-derived artifacts remain tied to their own observed time horizon.
    """
    if year_column in df.columns:
        years = pd.to_numeric(df[year_column], errors="coerce")
        if years.notna().any():
            return int(years.max())
    return datetime.now(timezone.utc).year
