"""
Training fit scenarios: overfit, underfit, CV instability, outliers, curve softening.

Thresholds are the single source of truth — mirror in `.cursor/rules/ml-evaluation.mdc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.shared.schema import CURVE_DISPLACEMENT, CURVE_LOAD

# --- Thresholds (SI: kN) ---
OVERFIT_R2_GAP_WARN = 0.05
OVERFIT_R2_GAP_SEVERE = 0.10
UNDERFIT_TRAIN_R2 = 0.75
UNDERFIT_TEST_R2_PEAK = 0.80
UNDERFIT_TEST_R2_CURVE = 0.82
GOOD_TEST_R2_PEAK = 0.88
GOOD_TEST_R2_CURVE = 0.88
CV_RMSE_CV_WARN = 0.40  # std / mean (PEER rectangular n~250 is fold-noisy)
HOLDOUT_VS_CV_RMSE_RATIO_WARN = 1.45  # test_rmse >> cv_rmse_mean
HOLDOUT_VS_CV_RMSE_RATIO_OPTIMISTIC = 0.55  # only flag very lucky splits
PEAK_REL_ERR_OUTLIER = 0.15
PEAK_OUTLIER_RATE_WARN = 0.20  # fraction of holdout before flagging scenario
CURVE_SPECIMEN_R2_POOR = 0.60
CURVE_POOR_RATE_WARN = 0.15  # fraction of test specimens before flagging
SOFTENING_LOAD_DROP = 0.12  # actual drop after peak vs peak load
SOFTENING_PRED_FLAT = 0.05  # pred stays within this fraction of its max after peak
SOFTENING_RATE_WARN = 0.10


class FitScenario(str, Enum):
    GOOD_FIT = "good_fit"
    OVERFIT = "overfit"
    OVERFIT_SEVERE = "overfit_severe"
    UNDERFIT = "underfit"
    HIGH_CV_VARIANCE = "high_cv_variance"
    OPTIMISTIC_HOLDOUT = "optimistic_holdout"
    PESSIMISTIC_HOLDOUT = "pessimistic_holdout"
    PEAK_OUTLIERS = "peak_outliers"
    CURVE_SOFTENING_MISS = "curve_softening_miss"
    CURVE_POOR_SPECIMENS = "curve_poor_specimens"


@dataclass
class ScenarioAssessment:
    primary: FitScenario
    flags: list[FitScenario] = field(default_factory=list)
    severity: str = "ok"  # ok | warn | critical
    messages: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def assess_model_metrics(row: dict[str, Any], *, mode: str) -> ScenarioAssessment:
    """Classify one model row from metrics_peak/curve CSV."""
    train_r2 = float(row.get("train_r2", row.get("r2", 0)))
    test_r2 = float(row.get("test_r2", row.get("r2", 0)))
    gap = float(row.get("overfit_r2_gap", train_r2 - test_r2))
    cv_mean = float(row.get("cv_rmse_mean", np.nan))
    cv_std = float(row.get("cv_rmse_std", np.nan))
    test_rmse = float(row.get("test_rmse", row.get("rmse", np.nan)))

    underfit_test = (
        UNDERFIT_TEST_R2_CURVE if mode == "curve" else UNDERFIT_TEST_R2_PEAK
    )
    good_test = GOOD_TEST_R2_CURVE if mode == "curve" else GOOD_TEST_R2_PEAK

    flags: list[FitScenario] = []
    messages: list[str] = []
    recs: list[str] = []

    if gap >= OVERFIT_R2_GAP_SEVERE:
        flags.append(FitScenario.OVERFIT_SEVERE)
        messages.append(f"Severe overfit: train-test R² gap = {gap:.3f}")
        recs.append("Use stronger regularization or simpler model (RF over XGB).")
    elif gap >= OVERFIT_R2_GAP_WARN:
        flags.append(FitScenario.OVERFIT)
        messages.append(f"Overfit: train-test R² gap = {gap:.3f}")

    if train_r2 < UNDERFIT_TRAIN_R2 and test_r2 < underfit_test:
        flags.append(FitScenario.UNDERFIT)
        messages.append(f"Underfit: train R²={train_r2:.3f}, test R²={test_r2:.3f}")
        recs.append("Increase model capacity cautiously or add features; check data quality.")

    if not np.isnan(cv_mean) and cv_mean > 0 and not np.isnan(cv_std):
        cv_coef = cv_std / cv_mean
        if cv_coef >= CV_RMSE_CV_WARN:
            flags.append(FitScenario.HIGH_CV_VARIANCE)
            messages.append(f"Unstable CV: RMSE std/mean = {cv_coef:.2f}")
            recs.append("Use group k-fold mean for reporting; avoid trusting a single holdout split.")

    if not np.isnan(cv_mean) and cv_mean > 0 and not np.isnan(test_rmse):
        ratio = test_rmse / cv_mean
        if ratio >= HOLDOUT_VS_CV_RMSE_RATIO_WARN:
            flags.append(FitScenario.PESSIMISTIC_HOLDOUT)
            messages.append(f"Holdout harder than CV: test RMSE / CV RMSE = {ratio:.2f}")
        elif ratio <= HOLDOUT_VS_CV_RMSE_RATIO_OPTIMISTIC:
            flags.append(FitScenario.OPTIMISTIC_HOLDOUT)
            messages.append(f"Holdout easier than CV: test RMSE / CV RMSE = {ratio:.2f}")
            recs.append("Report CV metrics alongside holdout; do not overclaim from one split.")

    if FitScenario.OVERFIT_SEVERE in flags or FitScenario.UNDERFIT in flags:
        primary = (
            FitScenario.OVERFIT_SEVERE
            if FitScenario.OVERFIT_SEVERE in flags
            else FitScenario.UNDERFIT
        )
        severity = "critical"
    elif flags:
        primary = flags[0]
        severity = "warn"
    elif test_r2 >= good_test and gap < OVERFIT_R2_GAP_WARN:
        primary = FitScenario.GOOD_FIT
        severity = "ok"
        messages.append(f"Balanced fit: test R²={test_r2:.3f}, gap={gap:.3f}")
    else:
        primary = FitScenario.GOOD_FIT
        severity = "ok"

    return ScenarioAssessment(
        primary=primary,
        flags=flags or [primary],
        severity=severity,
        messages=messages,
        recommendations=recs,
    )


def specimen_peak_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    specimen_ids: np.ndarray,
    specimens: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred, strict=False)):
        sid = str(specimen_ids[i])
        rel = abs(yp - yt) / yt if yt > 0 else np.nan
        rows.append(
            {
                "specimen_id": sid,
                "specimen": str(specimens[i]) if specimens is not None else sid,
                "actual_peak_kN": float(yt),
                "predicted_peak_kN": float(yp),
                "error_kN": float(yp - yt),
                "abs_pct_error": float(rel * 100) if not np.isnan(rel) else np.nan,
                "is_outlier": bool(rel > PEAK_REL_ERR_OUTLIER) if not np.isnan(rel) else False,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_pct_error", ascending=False, na_position="last")


def specimen_curve_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in pred_df.groupby("specimen_id"):
        sub = sub.sort_values(CURVE_DISPLACEMENT)
        actual = sub[CURVE_LOAD].values
        pred = sub["predicted_load_kN"].values
        disp = sub[CURVE_DISPLACEMENT].values
        r2 = float(r2_score(actual, pred)) if len(actual) > 1 else np.nan
        mae = float(mean_absolute_error(actual, pred))
        soften = _detect_softening_miss(disp, actual, pred)
        rows.append(
            {
                "specimen_id": sid,
                "specimen": sub["specimen"].iloc[0] if "specimen" in sub.columns else sid,
                "r2": r2,
                "mae_kN": mae,
                "n_points": len(sub),
                "poor_fit": bool(r2 < CURVE_SPECIMEN_R2_POOR) if not np.isnan(r2) else True,
                "softening_miss": soften,
            }
        )
    return pd.DataFrame(rows).sort_values("r2")


def _detect_softening_miss(disp: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> bool:
    if len(actual) < 4:
        return False
    peak_i = int(np.argmax(actual))
    if peak_i >= len(actual) - 2:
        return False
    peak_load = actual[peak_i]
    if peak_load <= 0:
        return False
    tail_actual = actual[peak_i + 1 :]
    tail_pred = pred[peak_i + 1 :]
    if len(tail_actual) == 0:
        return False
    actual_drop = (peak_load - np.min(tail_actual)) / peak_load
    if actual_drop < SOFTENING_LOAD_DROP:
        return False
    pred_max = np.max(pred)
    pred_tail_flat = np.mean(np.abs(tail_pred - pred_max) / max(pred_max, 1e-6)) < SOFTENING_PRED_FLAT
    return bool(pred_tail_flat)


def build_training_diagnostics(
    *,
    mode: str,
    best_row: dict[str, Any],
    all_rows: list[dict[str, Any]],
    specimen_peak_df: pd.DataFrame | None = None,
    specimen_curve_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    best_assess = assess_model_metrics(best_row, mode=mode)
    model_assessments = {}
    for r in all_rows:
        a = assess_model_metrics(r, mode=mode)
        model_assessments[r["model"]] = {
            "primary": a.primary.value,
            "severity": a.severity,
            "flags": [f.value for f in a.flags],
            "messages": a.messages,
        }

    flags = list(best_assess.flags)
    messages = list(best_assess.messages)
    recs = list(best_assess.recommendations)

    if specimen_peak_df is not None and len(specimen_peak_df):
        n_out = int(specimen_peak_df["is_outlier"].sum())
        rate = n_out / len(specimen_peak_df)
        if rate >= PEAK_OUTLIER_RATE_WARN:
            flags.append(FitScenario.PEAK_OUTLIERS)
            messages.append(
                f"{n_out}/{len(specimen_peak_df)} holdout specimen(s) "
                f"({rate:.0%}) with >{PEAK_REL_ERR_OUTLIER:.0%} peak error"
            )
            recs.append(
                "Review worst specimens in specimen_errors_peak.csv; consider excluding bad tests."
            )

    if specimen_curve_df is not None and len(specimen_curve_df):
        n_poor = int(specimen_curve_df["poor_fit"].sum())
        n_soft = int(specimen_curve_df["softening_miss"].sum())
        poor_rate = n_poor / len(specimen_curve_df)
        soft_rate = n_soft / len(specimen_curve_df)
        if poor_rate >= CURVE_POOR_RATE_WARN:
            flags.append(FitScenario.CURVE_POOR_SPECIMENS)
            messages.append(
                f"{n_poor}/{len(specimen_curve_df)} test specimen(s) "
                f"({poor_rate:.0%}) with R² < {CURVE_SPECIMEN_R2_POOR}"
            )
        if soft_rate >= SOFTENING_RATE_WARN:
            flags.append(FitScenario.CURVE_SOFTENING_MISS)
            messages.append(
                f"{n_soft}/{len(specimen_curve_df)} specimen(s) "
                f"({soft_rate:.0%}) with missed post-peak softening"
            )
            recs.append(
                "Pointwise curve model cannot enforce decay; document limit or use sequence/monotonic model."
            )

    # If core fit is good, demote lone optimistic/high-CV noise on small PEER splits
    core_ok = (
        float(best_row.get("test_r2", 0)) >= (GOOD_TEST_R2_CURVE if mode == "curve" else GOOD_TEST_R2_PEAK)
        and abs(float(best_row.get("overfit_r2_gap", 99))) < OVERFIT_R2_GAP_WARN
    )
    if core_ok:
        flags = [
            f
            for f in flags
            if f
            not in {
                FitScenario.OPTIMISTIC_HOLDOUT,
                FitScenario.HIGH_CV_VARIANCE,
            }
        ]
        if not flags:
            flags = [FitScenario.GOOD_FIT]
            best_assess.primary = FitScenario.GOOD_FIT
            best_assess.severity = "ok"
            if not any("Balanced fit" in m for m in messages):
                messages = [
                    f"Balanced fit: test R²={float(best_row.get('test_r2', 0)):.3f}, "
                    f"gap={float(best_row.get('overfit_r2_gap', 0)):.3f}"
                ] + messages

    severity = best_assess.severity
    if FitScenario.PEAK_OUTLIERS in flags or FitScenario.CURVE_SOFTENING_MISS in flags:
        severity = "warn" if severity == "ok" else severity
    if FitScenario.CURVE_POOR_SPECIMENS in flags and severity == "ok":
        severity = "warn"
    if flags == [FitScenario.GOOD_FIT] or flags == ["good_fit"]:
        severity = "ok"
        best_assess.primary = FitScenario.GOOD_FIT

    return {
        "mode": mode,
        "primary_scenario": (
            best_assess.primary.value
            if isinstance(best_assess.primary, FitScenario)
            else best_assess.primary
        ),
        "severity": severity,
        "flags": [f.value if isinstance(f, FitScenario) else f for f in flags],
        "messages": messages,
        "recommendations": list(dict.fromkeys(recs)),
        "thresholds": {
            "overfit_r2_gap_warn": OVERFIT_R2_GAP_WARN,
            "overfit_r2_gap_severe": OVERFIT_R2_GAP_SEVERE,
            "underfit_test_r2_peak": UNDERFIT_TEST_R2_PEAK,
            "underfit_test_r2_curve": UNDERFIT_TEST_R2_CURVE,
            "peak_outlier_rate_warn": PEAK_OUTLIER_RATE_WARN,
            "curve_poor_rate_warn": CURVE_POOR_RATE_WARN,
        },
        "best_model": best_row.get("model"),
        "best_metrics": best_row,
        "model_scenarios": model_assessments,
    }


def save_diagnostics(diagnostics: dict[str, Any], outputs: Path, mode: str) -> Path:
    path = outputs / f"training_diagnostics_{mode}.json"
    # serialize enums in nested model_scenarios
    path.write_text(json_dumps(diagnostics), encoding="utf-8")
    return path


def json_dumps(obj: Any) -> str:
    import json

    def default(o: Any) -> Any:
        if isinstance(o, FitScenario):
            return o.value
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(type(o))

    return json.dumps(obj, indent=2, default=default)
