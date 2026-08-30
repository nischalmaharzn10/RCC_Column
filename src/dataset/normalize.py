"""Normalize PEER rectangular properties into project schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.io_utils import parse_number, slugify_specimen
from src.shared.schema import PROPERTIES_REQUIRED


# PEER bulk CSV headers → project columns
_PEER_MAP = {
    "Specimen Name": "specimen",
    "f'c (MPa)": "fc_MPa",
    "Axial Load (kN)": "P_axial_kN",
    "B (mm)": "b_mm",
    "H (mm)": "h_mm",
    "L (mm)": "L_mm",
    "Lsplice (mm)": "Lsplice_mm",
    "Config.": "test_config",
    "Diameter Corner (mm)": "db_long_mm",
    "Total # Bars": "n_long_bars",
    "Clear Cover Perpendicular to Load (mm)": "cover_mm",
    "Reinf Ratio": "rho_long",
    "fyl corner (MPa)": "fyl_MPa",
    "Steel Grade": "steel_grade",
    "Region of close spacing bar dia (mm)": "db_trans_mm",
    "Spacing (mm)": "s_hoop_mm",  # first Spacing col = close spacing (col order matters)
    "Vol Trans Reinf Ratio": "rho_trans",
    "fyt (MPa)": "fyt_MPa",
    "Failure": "failure_mode",
}


def _read_peer_properties(path: Path) -> pd.DataFrame:
    # Prefer tab if present; CSV export may use commas
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "\t" in raw.splitlines()[0]:
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    return df


def normalize_peer_properties(path: Path) -> pd.DataFrame:
    """Load PEER rectangular properties and map to PROPERTIES_REQUIRED schema."""
    raw = _read_peer_properties(path)

    # Spacing appears twice (close / wide). Keep close-spacing by position.
    cols = list(raw.columns)
    spacing_idxs = [i for i, c in enumerate(cols) if c.strip() == "Spacing (mm)"]
    rename = {}
    for peer_col, ours in _PEER_MAP.items():
        if peer_col == "Spacing (mm)":
            continue
        if peer_col in raw.columns:
            rename[peer_col] = ours
    df = raw.rename(columns=rename).copy()

    if spacing_idxs:
        close_name = cols[spacing_idxs[0]]
        df["s_hoop_mm"] = raw.iloc[:, spacing_idxs[0]].map(parse_number)
        # if rename already mapped wrong, overwrite
        _ = close_name

    for col in [
        "fc_MPa",
        "P_axial_kN",
        "b_mm",
        "h_mm",
        "L_mm",
        "Lsplice_mm",
        "db_long_mm",
        "n_long_bars",
        "cover_mm",
        "rho_long",
        "fyl_MPa",
        "steel_grade",
        "db_trans_mm",
        "rho_trans",
        "fyt_MPa",
        "failure_mode",
    ]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    if "s_hoop_mm" not in df.columns and "Spacing (mm)" in raw.columns:
        df["s_hoop_mm"] = raw["Spacing (mm)"].map(parse_number)

    df["specimen"] = df["specimen"].astype(str).str.strip()
    df["test_config"] = df.get("test_config", "DE").astype(str).str.strip()
    df["source"] = "peer"
    df["specimen_id"] = df["specimen"].map(slugify_specimen)

    # Axial load ratio: P / (fc * b * h) with P in N, fc in MPa=N/mm^2, b,h in mm → dimensionless
    area = df["b_mm"] * df["h_mm"]
    capacity = df["fc_MPa"] * area  # N (since MPa * mm^2 = N)
    p_n = df["P_axial_kN"] * 1000.0
    df["axial_load_ratio"] = np.where(capacity > 0, p_n / capacity, np.nan)

    # Keep project columns
    keep = list(dict.fromkeys(PROPERTIES_REQUIRED + ["specimen_id", "source"]))
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    out = df[keep].copy()
    out = out.dropna(subset=["specimen"]).reset_index(drop=True)
    return out


def load_normalized_properties(path: Path) -> pd.DataFrame:
    """Load either PEER bulk or already-normalized properties CSV."""
    sample = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "Specimen Name" in sample or "f'c (MPa)" in sample:
        return normalize_peer_properties(path)
    df = pd.read_csv(path)
    if "specimen_id" not in df.columns:
        df["specimen_id"] = df["specimen"].map(slugify_specimen)
    if "source" not in df.columns:
        df["source"] = "peer"
    for col in PROPERTIES_REQUIRED:
        if col not in df.columns and col != "specimen":
            df[col] = np.nan
        elif col in df.columns and col not in {"specimen", "test_config"}:
            if col not in {"test_config", "failure_mode"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
