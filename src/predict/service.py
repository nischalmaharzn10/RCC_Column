"""Inference helpers for peak and curve models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.shared.features import add_curve_shape_features, add_engineered_features, enforce_postpeak_softening
from src.shared.paths import models_dir
from src.shared.schema import CURVE_DISPLACEMENT, PEAK_TARGET, TRAINING_FEATURE_COLUMNS


def load_bundle(path: Path | None = None, mode: str = "peak") -> dict[str, Any]:
    if path is None:
        name = "best_model_peak.joblib" if mode == "peak" else "best_model_curve.joblib"
        path = models_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Train first with npm run train.")
    return joblib.load(path)


def _frame_from_inputs(
    inputs: dict[str, Any],
    displacements: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    if displacements is None:
        row = {c: inputs.get(c, np.nan) for c in TRAINING_FEATURE_COLUMNS}
        # TRAINING includes engineered — fill raw first then engineer
        for c in [
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
            "s_hoop_mm",
            "rho_trans",
            "fyt_MPa",
        ]:
            row[c] = inputs.get(c, np.nan)
        row["test_config"] = inputs.get("test_config", "DE")
        row["failure_mode"] = inputs.get("failure_mode", 1)
        df = pd.DataFrame([row])
        return add_engineered_features(df)

    rows = []
    for d in displacements:
        row = {
            c: inputs.get(c, np.nan)
            for c in [
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
                "s_hoop_mm",
                "rho_trans",
                "fyt_MPa",
            ]
        }
        row["test_config"] = inputs.get("test_config", "DE")
        row["failure_mode"] = inputs.get("failure_mode", 1)
        row[CURVE_DISPLACEMENT] = float(d)
        row["lateral_load_kN"] = np.nan
        row["specimen_id"] = "_infer"
        rows.append(row)
    df = pd.DataFrame(rows)
    df = add_engineered_features(df)
    peak_hint = float(inputs.get(PEAK_TARGET, np.nan)) if inputs.get(PEAK_TARGET) else np.nan
    if np.isfinite(peak_hint) and peak_hint > 0:
        df["peak_load_kN"] = peak_hint
    else:
        df["peak_load_kN"] = 1.0  # placeholder for ratio features; scaled later
    df = add_curve_shape_features(df)
    return df


def predict_peak(inputs: dict[str, Any], model_path: Path | None = None) -> float:
    bundle = load_bundle(model_path, mode="peak")
    pipe = bundle["pipeline"]
    X = _frame_from_inputs(inputs)
    for c in bundle.get("feature_cols", []):
        if c not in X.columns:
            X[c] = np.nan
    X = X[bundle["feature_cols"]]
    pred = float(pipe.predict(X)[0])
    if bundle.get("target_transform") == "log1p":
        pred = float(np.expm1(pred))
    return pred


def predict_curve(
    inputs: dict[str, Any],
    displacements: list[float] | np.ndarray | None = None,
    model_path: Path | None = None,
    peak_model_path: Path | None = None,
) -> pd.DataFrame:
    bundle = load_bundle(model_path, mode="curve")
    pipe = bundle["pipeline"]
    if displacements is None:
        d_max = float(inputs.get("L_mm", 1000.0)) * 0.04
        displacements = np.linspace(0.0, max(d_max, 10.0), 40)

    peak = predict_peak(inputs, model_path=peak_model_path)
    inputs_with_peak = dict(inputs)
    inputs_with_peak[PEAK_TARGET] = peak

    X = _frame_from_inputs(inputs_with_peak, displacements=displacements)
    for c in bundle.get("feature_cols", []):
        if c not in X.columns:
            X[c] = np.nan
    X = X[bundle["feature_cols"]]

    if bundle.get("curve_mode") == "load_ratio" or bundle.get("target") == "load_ratio":
        ratios = np.clip(pipe.predict(X), 0.0, 1.35)
        preds = ratios * peak
    else:
        preds = pipe.predict(X)

    preds = enforce_postpeak_softening(np.asarray(displacements, dtype=float), np.asarray(preds))
    return pd.DataFrame(
        {
            CURVE_DISPLACEMENT: list(displacements),
            "predicted_load_kN": preds,
            PEAK_TARGET: float(peak),
        }
    )
