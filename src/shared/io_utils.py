"""IO helpers: numeric parsing, specimen ids."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def parse_number(value: Any) -> float:
    """Parse PEER-style numbers that may include thousands commas."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "NA", "N/A", "nan", "None"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def slugify_specimen(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:120]


def ensure_source(series: pd.Series | None, n: int) -> pd.Series:
    if series is None:
        return pd.Series(["peer"] * n)
    return series.fillna("peer").astype(str)
