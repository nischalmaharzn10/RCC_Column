"""Build ML-ready tables from PEER samples / properties + curves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.curves import (
    build_backbone_table,
    peak_from_history,
    read_force_displacement,
)
from src.dataset.normalize import load_normalized_properties
from src.shared.io_utils import slugify_specimen
from src.shared.paths import data_processed, samples_dir
from src.shared.schema import FEATURE_COLUMNS, PROPERTIES_REQUIRED


def default_properties_path() -> Path:
    p = samples_dir() / "properties" / "rectangular_properties.csv"
    if p.exists():
        return p
    return samples_dir() / "rectangular_properties.txt"


def default_curves_dir() -> Path:
    return samples_dir() / "curves" / "rectangular"


def default_curve_index() -> Path:
    return samples_dir() / "properties" / "rectangular_curve_index.csv"


def build_dataset(
    properties_path: Path | None = None,
    curves_dir: Path | None = None,
    curve_index_path: Path | None = None,
    out_dir: Path | None = None,
    backbone_points: int = 40,
) -> dict[str, Path]:
    properties_path = properties_path or default_properties_path()
    curves_dir = curves_dir or default_curves_dir()
    curve_index_path = curve_index_path or default_curve_index()
    out_dir = out_dir or data_processed()
    out_dir.mkdir(parents=True, exist_ok=True)

    props = load_normalized_properties(properties_path)
    index = _load_curve_index(curve_index_path)

    # Join props to curves by specimen name (normalized)
    props["_key"] = props["specimen"].map(_norm_name)
    index["_key"] = index["specimen"].map(_norm_name)
    merged = props.merge(index[["_key", "curve_file", "downloaded"]], on="_key", how="left")

    peak_rows: list[dict] = []
    backbone_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for _, row in merged.iterrows():
        specimen = str(row["specimen"])
        specimen_id = str(row.get("specimen_id") or slugify_specimen(specimen))
        curve_file = row.get("curve_file")
        status = "no_curve"
        peak_load = float("nan")
        disp_peak = float("nan")
        n_pts = 0

        if isinstance(curve_file, str) and curve_file and str(row.get("downloaded", "yes")) != "no":
            path = curves_dir / curve_file
            if path.exists():
                try:
                    _name, disp, force = read_force_displacement(path)
                    peak_load, disp_peak = peak_from_history(disp, force)
                    bb = build_backbone_table(
                        specimen=specimen,
                        specimen_id=specimen_id,
                        disp=disp,
                        force=force,
                        source=str(row.get("source", "peer")),
                        n_points=backbone_points,
                    )
                    # attach design features
                    for col in FEATURE_COLUMNS + ["test_config", "failure_mode", "source"]:
                        if col in row.index:
                            bb[col] = row[col]
                    backbone_frames.append(bb)
                    n_pts = len(disp)
                    status = "ok"
                except Exception as exc:  # noqa: BLE001 — record and continue
                    status = f"error:{exc}"
            else:
                status = "missing_file"
        elif str(row.get("downloaded", "")) == "no":
            status = "not_downloaded"

        feat = {c: row.get(c) for c in PROPERTIES_REQUIRED if c in row.index}
        feat.update(
            {
                "specimen": specimen,
                "specimen_id": specimen_id,
                "source": row.get("source", "peer"),
                "curve_file": curve_file,
                "peak_load_kN": peak_load,
                "disp_at_peak_mm": disp_peak,
                "n_history_points": n_pts,
                "curve_status": status,
            }
        )
        if status == "ok":
            peak_rows.append(feat)
        summary_rows.append(feat)

    dataset = pd.DataFrame(peak_rows)
    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(backbone_frames, ignore_index=True) if backbone_frames else pd.DataFrame()

    dataset_path = out_dir / "dataset_ml.csv"
    curves_path = out_dir / "curves_backbone.csv"
    summary_path = out_dir / "specimen_summary.csv"
    props_path = out_dir / "properties_normalized.csv"

    dataset.to_csv(dataset_path, index=False)
    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    props.drop(columns=["_key"], errors="ignore").to_csv(props_path, index=False)

    return {
        "dataset_ml": dataset_path,
        "curves_backbone": curves_path,
        "specimen_summary": summary_path,
        "properties_normalized": props_path,
        "n_peak": len(dataset),
        "n_curve_points": len(curves),
        "n_summary": len(summary),
    }


def _load_curve_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["specimen", "curve_file", "downloaded"])
    df = pd.read_csv(path)
    if "downloaded" not in df.columns:
        df["downloaded"] = "yes"
    return df


def _norm_name(name: object) -> str:
    """Normalize specimen labels so props ↔ curve index join survives typos/spacing."""
    s = str(name).lower().strip()
    # Known PEER property typos / punctuation variants
    s = s.replace("mugumura", "muguruma")
    s = s.replace("&", " and ")
    s = s.replace(" and ", " ")
    for ch in [",", ".", " ", "-", "_", "'", '"', "/", "(", ")"]:
        s = s.replace(ch, "")
    return s
