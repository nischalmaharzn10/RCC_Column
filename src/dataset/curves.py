"""Parse PEER force–displacement histories and extract backbone / peak."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_force_displacement(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """
    PEER/UW format:
      row0: specimen name
      row1: n points
      row2+: disp_mm, lateral_kN [, axial_kN]
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Curve file too short: {path}")
    name = lines[0].split("\t")[0].strip()
    rows = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        # tab-delimited preferred
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
        if len(parts) < 2:
            continue
        try:
            d = float(parts[0].replace(",", ""))
            f = float(parts[1].replace(",", ""))
        except ValueError:
            continue
        rows.append((d, f))
    if not rows:
        raise ValueError(f"No data points in {path}")
    arr = np.asarray(rows, dtype=float)
    return name, arr[:, 0], arr[:, 1]


def first_cycle_envelope(disp: np.ndarray, force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    First-cycle envelope in the positive direction, plus mirrored negative,
    returned as a single positive-magnitude backbone (disp >= 0, force >= 0).
    """
    pos_d, pos_f = _directional_envelope(disp, force, positive=True)
    neg_d, neg_f = _directional_envelope(disp, force, positive=False)

    d_parts = []
    f_parts = []
    if len(pos_d):
        d_parts.append(pos_d)
        f_parts.append(pos_f)
    if len(neg_d):
        d_parts.append(np.abs(neg_d))
        f_parts.append(np.abs(neg_f))
    if not d_parts:
        # fallback: absolute running envelope
        order = np.argsort(np.abs(disp))
        d = np.abs(disp)[order]
        f = np.maximum.accumulate(np.abs(force)[order])
        return d, f

    d = np.concatenate(d_parts)
    f = np.concatenate(f_parts)
    order = np.argsort(d)
    d, f = d[order], f[order]
    # Deduplicate displacements; keep last force at each disp so post-peak softening remains.
    d_u, idx = np.unique(d, return_index=True)
    # unique returns first index — recompute last-occurrence force
    last: dict[float, float] = {}
    for di, fi in zip(d, f, strict=False):
        last[float(di)] = float(fi)
    f_u = np.asarray([last[float(x)] for x in d_u], dtype=float)
    return d_u, f_u


def _directional_envelope(
    disp: np.ndarray, force: np.ndarray, *, positive: bool
) -> tuple[np.ndarray, np.ndarray]:
    if positive:
        mask = disp >= 0
        d = disp[mask]
        f = force[mask]
        sign = 1.0
    else:
        mask = disp <= 0
        d = disp[mask]
        f = force[mask]
        sign = -1.0
    if len(d) < 2:
        return np.array([]), np.array([])

    # Track new extremes of displacement; record force at those points
    env_d: list[float] = []
    env_f: list[float] = []
    extreme = 0.0
    for di, fi in zip(d, f, strict=False):
        mag = di * sign
        if mag >= extreme:
            extreme = mag
            env_d.append(di)
            env_f.append(fi)
    return np.asarray(env_d, dtype=float), np.asarray(env_f, dtype=float)


def resample_backbone(
    disp: np.ndarray, force: np.ndarray, n_points: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    if len(disp) < 2:
        return disp, force
    d_max = float(np.max(disp))
    if d_max <= 0:
        return disp[:1], force[:1]
    grid = np.linspace(0.0, d_max, n_points)
    # interpolate force along sorted unique disp
    order = np.argsort(disp)
    d_s, f_s = disp[order], force[order]
    # unique displ
    d_u, idx = np.unique(d_s, return_index=True)
    f_u = f_s[idx]
    f_grid = np.interp(grid, d_u, f_u)
    return grid, f_grid


def peak_from_history(disp: np.ndarray, force: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(np.abs(force)))
    return float(np.abs(force[i])), float(np.abs(disp[i]))


def build_backbone_table(
    specimen: str,
    specimen_id: str,
    disp: np.ndarray,
    force: np.ndarray,
    source: str = "peer",
    n_points: int = 25,
) -> pd.DataFrame:
    d_env, f_env = first_cycle_envelope(disp, force)
    d_rs, f_rs = resample_backbone(d_env, f_env, n_points=n_points)
    return pd.DataFrame(
        {
            "specimen": specimen,
            "specimen_id": specimen_id,
            "displacement_mm": d_rs,
            "lateral_load_kN": f_rs,
            "source": source,
        }
    )
