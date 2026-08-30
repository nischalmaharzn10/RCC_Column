"""Engineered features for training (no leakage from targets)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Extra numeric features derived from design params (not targets)
ENGINEERED_FEATURE_COLUMNS: list[str] = [
    "aspect_Lh",
    "section_area_mm2",
    "confinement_index",
    "long_steel_index",
    "axial_stress_MPa",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add structural indices used by training / inference."""
    out = df.copy()
    b = pd.to_numeric(out.get("b_mm"), errors="coerce")
    h = pd.to_numeric(out.get("h_mm"), errors="coerce")
    L = pd.to_numeric(out.get("L_mm"), errors="coerce")
    fc = pd.to_numeric(out.get("fc_MPa"), errors="coerce")
    P = pd.to_numeric(out.get("P_axial_kN"), errors="coerce")
    rho_t = pd.to_numeric(out.get("rho_trans"), errors="coerce")
    fyt = pd.to_numeric(out.get("fyt_MPa"), errors="coerce")
    rho_l = pd.to_numeric(out.get("rho_long"), errors="coerce")
    fyl = pd.to_numeric(out.get("fyl_MPa"), errors="coerce")

    out["aspect_Lh"] = np.where((h > 0) & h.notna(), L / h, np.nan)
    out["section_area_mm2"] = b * h
    out["confinement_index"] = np.where(
        (fc > 0) & fc.notna(), (rho_t * fyt) / fc, np.nan
    )
    out["long_steel_index"] = np.where(
        (fc > 0) & fc.notna(), (rho_l * fyl) / fc, np.nan
    )
    area = b * h
    out["axial_stress_MPa"] = np.where(
        (area > 0) & area.notna(), (P * 1000.0) / area, np.nan
    )
    return out


def add_curve_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize displacement and load by specimen peak for shape learning.
    Expects columns: specimen_id, displacement_mm, lateral_load_kN, peak_load_kN (optional).
    """
    out = df.copy()
    if "peak_load_kN" not in out.columns or out["peak_load_kN"].isna().all():
        peaks = (
            out.groupby("specimen_id")["lateral_load_kN"]
            .transform("max")
            .replace(0, np.nan)
        )
        out["peak_load_kN"] = peaks
    else:
        out["peak_load_kN"] = pd.to_numeric(out["peak_load_kN"], errors="coerce")

    peak = out["peak_load_kN"].replace(0, np.nan)
    out["load_ratio"] = out["lateral_load_kN"] / peak

    L = pd.to_numeric(out.get("L_mm"), errors="coerce")
    disp = pd.to_numeric(out["displacement_mm"], errors="coerce")
    out["disp_over_L"] = np.where((L > 0) & L.notna(), disp / L, np.nan)

    d_max = out.groupby("specimen_id")["displacement_mm"].transform("max").replace(0, np.nan)
    out["disp_ratio"] = disp / d_max
    return out


def enforce_postpeak_softening(disp: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Force predicted backbone to decay after its peak (non-increasing + gentle drop)."""
    if len(load) < 3:
        return load
    order = np.argsort(disp)
    out = np.asarray(load, dtype=float).copy()
    sorted_load = out[order]
    peak_i = int(np.argmax(sorted_load))
    peak_val = float(sorted_load[peak_i])
    n_tail = len(sorted_load) - peak_i - 1
    for j in range(peak_i + 1, len(sorted_load)):
        t = (j - peak_i) / max(n_tail, 1)
        ceiling = peak_val * (1.0 - 0.30 * t)
        sorted_load[j] = min(sorted_load[j], sorted_load[j - 1], ceiling)
    out[order] = np.maximum(sorted_load, 0.0)
    return out
