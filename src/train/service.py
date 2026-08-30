"""Training service: peak and curve modes with group CV and regularization."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

from src.shared.features import (
    add_curve_shape_features,
    add_engineered_features,
    enforce_postpeak_softening,
)
from src.shared.paths import data_processed, models_dir, outputs_dir
from src.shared.schema import (
    CURVE_DISPLACEMENT,
    CURVE_LOAD,
    CURVE_LOAD_RATIO,
    CURVE_SHAPE_FEATURE_COLUMNS,
    PEAK_TARGET,
    TRAINING_FEATURE_COLUMNS,
)
from src.train.diagnostics import (
    FitScenario,
    assess_model_metrics,
    build_training_diagnostics,
    save_diagnostics,
    specimen_curve_metrics,
    specimen_peak_errors,
)

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
MAX_OVERFIT_R2_GAP = 0.05
PEAK_USE_LOG1P = False


def _numeric_features(mode: str) -> list[str]:
    cols = list(TRAINING_FEATURE_COLUMNS)
    if mode == "curve":
        cols = cols + list(CURVE_SHAPE_FEATURE_COLUMNS) + [CURVE_DISPLACEMENT]
    return cols


def _cat_features() -> list[str]:
    return ["test_config", "failure_mode"]


def build_preprocessor(mode: str) -> ColumnTransformer:
    num_cols = _numeric_features(mode)
    cat_cols = _cat_features()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def model_zoo(mode: str = "peak") -> dict[str, object]:
    """Regularized baselines — peak uses stronger regularization to keep R² gap < 0.05."""
    if mode == "peak":
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features=0.6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "xgboost": XGBRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=0.8,
                reg_lambda=2.5,
                min_child_weight=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror",
            ),
            "hist_gbr": HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=1.0,
                min_samples_leaf=12,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
            ),
            "gbr": GradientBoostingRegressor(
                n_estimators=180,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(48, 24),
                alpha=0.15,
                learning_rate_init=5e-4,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=25,
                random_state=RANDOM_STATE,
            ),
        }

    # Curve shape (load_ratio) — capacity for softening patterns
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=6,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.2,
            min_child_weight=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
        "hist_gbr": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=280,
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=RANDOM_STATE,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=0.08,
            learning_rate_init=4e-4,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _group_cv_rmse(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = CV_FOLDS,
    progress: tqdm | None = None,
    peaks: np.ndarray | None = None,
    clean_slice: pd.DataFrame | None = None,
    mode: str = "peak",
    y_raw: np.ndarray | None = None,
    target_transform: str | None = None,
) -> dict[str, float]:
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return {"cv_rmse_mean": float("nan"), "cv_rmse_std": float("nan"), "cv_folds": 0}

    gkf = GroupKFold(n_splits=n_splits)
    rmses: list[float] = []
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if progress is not None:
            progress.set_postfix_str(f"CV fold {fold_idx + 1}/{n_splits}")
        pipe_fold = Pipeline(pipe.steps)
        pipe_fold.fit(X.iloc[tr], y[tr])
        if mode == "curve" and peaks is not None and clean_slice is not None:
            pred = _predict_curve_loads(
                pipe_fold,
                X.iloc[va],
                peaks[va],
                clean_slice.iloc[va].reset_index(drop=True),
            )
            y_abs = clean_slice.iloc[va][CURVE_LOAD].astype(float).values
            rmses.append(float(np.sqrt(mean_squared_error(y_abs, pred))))
        else:
            pred = pipe_fold.predict(X.iloc[va])
            if target_transform == "log1p" and y_raw is not None:
                pred = np.expm1(pred)
                truth = y_raw[va]
            else:
                truth = y[va]
            rmses.append(float(np.sqrt(mean_squared_error(truth, pred))))
        if progress is not None:
            progress.update(1)
    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_folds": n_splits,
    }


def _predict_curve_loads(
    pipe: Pipeline,
    X: pd.DataFrame,
    peaks: np.ndarray,
    clean_slice: pd.DataFrame,
) -> np.ndarray:
    """Predict load ratios → absolute kN, then enforce post-peak softening per specimen."""
    ratios = np.clip(pipe.predict(X), 0.0, 1.35)
    loads = ratios * np.asarray(peaks, dtype=float)
    specimen_ids = clean_slice["specimen_id"].astype(str).values
    disp = clean_slice[CURVE_DISPLACEMENT].astype(float).values
    out = loads.copy()
    for sid in np.unique(specimen_ids):
        mask = specimen_ids == sid
        out[mask] = enforce_postpeak_softening(disp[mask], out[mask])
    return out


def _prepare_peak_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = add_engineered_features(df)
    feature_cols = _numeric_features("peak") + _cat_features()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    clean = df.dropna(subset=[target, "specimen_id"]).copy()
    # Keep all physical positive peaks (do not drop high-capacity columns)
    clean = clean[clean[target] > 0].copy()
    X = clean[feature_cols]
    y = clean[target].astype(float).values
    groups = clean["specimen_id"].astype(str).values
    return clean, X, y, groups


def _prepare_curve_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Curve trains on load_ratio; peaks used to reconstruct absolute load."""
    df = add_engineered_features(df)
    df = add_curve_shape_features(df)
    feature_cols = _numeric_features("curve") + _cat_features()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    clean = df.dropna(subset=[CURVE_LOAD, CURVE_DISPLACEMENT, "specimen_id", CURVE_LOAD_RATIO]).copy()
    clean = clean[np.isfinite(clean[CURVE_LOAD_RATIO])]
    X = clean[feature_cols]
    y_ratio = clean[CURVE_LOAD_RATIO].astype(float).values
    peaks = clean["peak_load_kN"].astype(float).values
    groups = clean["specimen_id"].astype(str).values
    return clean, X, y_ratio, peaks, groups


def _select_best_model(rows: list[dict], *, mode: str) -> dict:
    valid = [r for r in rows if not np.isnan(r.get("cv_rmse_mean", float("nan")))]
    if not valid:
        return max(rows, key=lambda r: r["test_r2"])

    def sort_key(r: dict) -> tuple[float, float, float]:
        assess = assess_model_metrics(r, mode=mode)
        flags = set(assess.flags)
        hard = 0.0
        if FitScenario.UNDERFIT in flags or FitScenario.OVERFIT_SEVERE in flags:
            hard += 1_000.0
        if FitScenario.OVERFIT in flags:
            hard += 50.0
        gap = abs(float(r.get("overfit_r2_gap", 0.0)))
        gap_penalty = max(0.0, gap - MAX_OVERFIT_R2_GAP) * 80.0
        # Prefer lowest group-CV RMSE; light tie-breakers only
        return (hard + r["cv_rmse_mean"] + gap_penalty, gap, -r["test_r2"])

    return min(valid, key=sort_key)


def _train_mode(
    df: pd.DataFrame,
    *,
    mode: str,
    target: str,
    models_out: Path,
    outputs: Path,
    show_progress: bool = True,
) -> dict:
    peaks_train = peaks_test = None
    target_transform: str | None = None
    y_raw_all: np.ndarray | None = None

    if mode == "peak":
        clean, X, y_raw_all, groups = _prepare_peak_frame(df, target)
        if PEAK_USE_LOG1P:
            y = np.log1p(y_raw_all)
            target_transform = "log1p"
        else:
            y = y_raw_all
    else:
        clean, X, y, peaks_all, groups = _prepare_curve_frame(df)

    feature_cols = list(X.columns)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    if mode == "curve":
        peaks_train = peaks_all[train_idx]
        peaks_test = peaks_all[test_idx]
        y_train_load = clean.iloc[train_idx][CURVE_LOAD].astype(float).values
        y_test_load = clean.iloc[test_idx][CURVE_LOAD].astype(float).values
    else:
        y_train_load = y_raw_all[train_idx] if y_raw_all is not None else y_train
        y_test_load = y_raw_all[test_idx] if y_raw_all is not None else y_test

    n_cv = min(CV_FOLDS, len(np.unique(groups_train)))
    n_cv = max(n_cv, 0)
    models = list(model_zoo(mode).items())
    total_steps = len(models) * (n_cv + 2) + 3

    rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    progress_ctx = tqdm(
        total=total_steps,
        desc=f"{mode} training",
        unit="step",
        disable=not show_progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    progress = progress_ctx.__enter__()

    try:
        for name, est in models:
            progress.set_description(f"{mode} | {name}")
            pipe = Pipeline([("pre", build_preprocessor(mode)), ("model", est)])

            progress.set_postfix_str("cross-validation")
            train_slice = clean.iloc[train_idx].reset_index(drop=True)
            y_raw_train = y_raw_all[train_idx] if y_raw_all is not None else None
            cv = _group_cv_rmse(
                pipe,
                X_train,
                y_train,
                groups_train,
                progress=progress if show_progress else None,
                peaks=peaks_train if mode == "curve" else None,
                clean_slice=train_slice if mode == "curve" else None,
                mode=mode,
                y_raw=y_raw_train,
                target_transform=target_transform,
            )

            progress.set_postfix_str("fitting train split")
            pipe.fit(X_train, y_train)
            progress.update(1)
            fitted[name] = pipe

            progress.set_postfix_str("evaluating holdout")
            if mode == "curve":
                pred_train = _predict_curve_loads(
                    pipe, X_train, peaks_train, clean.iloc[train_idx].reset_index(drop=True)
                )
                pred_test = _predict_curve_loads(
                    pipe, X_test, peaks_test, clean.iloc[test_idx].reset_index(drop=True)
                )
                train_m = _metrics(y_train_load, pred_train)
                test_m = _metrics(y_test_load, pred_test)
            else:
                pred_tr = pipe.predict(X_train)
                pred_te = pipe.predict(X_test)
                if target_transform == "log1p":
                    pred_tr = np.expm1(pred_tr)
                    pred_te = np.expm1(pred_te)
                train_m = _metrics(y_train_load, pred_tr)
                test_m = _metrics(y_test_load, pred_te)
            progress.update(1)

            row = {
                "model": name,
                "mode": mode,
                "target": target if mode == "peak" else CURVE_LOAD_RATIO,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "cv_folds": cv["cv_folds"],
                "cv_rmse_mean": cv["cv_rmse_mean"],
                "cv_rmse_std": cv["cv_rmse_std"],
                "train_r2": train_m["r2"],
                "train_rmse": train_m["rmse"],
                "train_mae": train_m["mae"],
                "test_r2": test_m["r2"],
                "test_rmse": test_m["rmse"],
                "test_mae": test_m["mae"],
                "overfit_r2_gap": train_m["r2"] - test_m["r2"],
                "r2": test_m["r2"],
                "rmse": test_m["rmse"],
                "mae": test_m["mae"],
            }
            rows.append(row)

        best_row = _select_best_model(rows, mode=mode)
        best_name = best_row["model"]
        best_pipe = fitted[best_name]

        progress.set_description(f"{mode} | {best_name}")
        progress.set_postfix_str("refit on full dataset")
        best_pipe_full = Pipeline(best_pipe.steps)
        best_pipe_full.fit(X, y)
        progress.update(1)

        metrics_df = pd.DataFrame(rows).sort_values(["cv_rmse_mean", "overfit_r2_gap"])
        metrics_name = "metrics_peak.csv" if mode == "peak" else "metrics_curve.csv"
        metrics_path = outputs / metrics_name
        metrics_df.to_csv(metrics_path, index=False)

        progress.set_postfix_str("saving model")
        model_name = "best_model_peak.joblib" if mode == "peak" else "best_model_curve.joblib"
        model_path = models_out / model_name
        joblib.dump(
            {
                "pipeline": best_pipe_full,
                "model_name": best_name,
                "target": target if mode == "peak" else CURVE_LOAD_RATIO,
                "mode": mode,
                "curve_mode": "load_ratio" if mode == "curve" else None,
                "target_transform": target_transform,
                "feature_cols": feature_cols,
                "training_features": "TRAINING_FEATURE_COLUMNS + engineered (+ curve shape)",
                "selection": best_row,
            },
            model_path,
        )
        progress.update(1)

        meta: dict = {
            "best_model": best_name,
            "selection_criterion": "lowest group-CV RMSE; prefer no overfit/high-CV flags",
            "metrics": rows,
            "model_path": str(model_path),
        }

        progress.set_postfix_str("writing plots & metrics")
        specimen_peak_df = None
        specimen_curve_df = None

        if mode == "peak":
            meta.update(
                _write_peak_artifacts(
                    best_pipe,
                    best_name,
                    clean,
                    test_idx,
                    X_test,
                    y_test_load,
                    target,
                    outputs,
                    target_transform=target_transform,
                )
            )
            _maybe_feature_importance(
                best_pipe, best_name, feature_cols, outputs / "feature_importance_peak.csv"
            )
            specimen_peak_df = pd.read_csv(outputs / "specimen_errors_peak.csv")
        else:
            meta.update(
                _write_curve_artifacts(
                    best_pipe,
                    best_name,
                    clean,
                    test_idx,
                    X_test,
                    peaks_test,
                    outputs,
                )
            )
            specimen_curve_df = pd.read_csv(outputs / "specimen_errors_curve.csv")

        progress.set_postfix_str("diagnostics")
        diagnostics = build_training_diagnostics(
            mode=mode,
            best_row=best_row,
            all_rows=rows,
            specimen_peak_df=specimen_peak_df,
            specimen_curve_df=specimen_curve_df,
        )
        diag_path = save_diagnostics(diagnostics, outputs, mode)
        meta["diagnostics"] = diagnostics
        meta["diagnostics_path"] = str(diag_path)

        summary_path = (
            outputs / "train_peak_summary.json"
            if mode == "peak"
            else outputs / "train_curve_summary.json"
        )
        summary_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        progress.update(1)
    finally:
        progress_ctx.__exit__(None, None, None)

    if show_progress:
        diag = meta.get("diagnostics", {})
        tqdm.write(
            f"Done {mode}: best={best_name} test R²={best_row['test_r2']:.4f} "
            f"scenario={diag.get('primary_scenario', '?')} severity={diag.get('severity', '?')}"
        )

    return meta


def _write_peak_artifacts(
    pipe: Pipeline,
    name: str,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target: str,
    outputs: Path,
    *,
    target_transform: str | None = None,
) -> dict:
    test_pred = pipe.predict(X_test)
    if target_transform == "log1p":
        test_pred = np.expm1(test_pred)
    test_specimen_ids = df.iloc[test_idx]["specimen_id"].values
    test_specimens = (
        df.iloc[test_idx]["specimen"].values if "specimen" in df.columns else test_specimen_ids
    )

    err_df = specimen_peak_errors(y_test, test_pred, test_specimen_ids, test_specimens)
    err_df.to_csv(outputs / "specimen_errors_peak.csv", index=False)

    parity = err_df[["specimen_id", "specimen", "actual_peak_kN", "predicted_peak_kN"]].rename(
        columns={"actual_peak_kN": "actual", "predicted_peak_kN": "predicted"}
    )
    parity.to_csv(outputs / "parity_peak.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, test_pred, alpha=0.7, edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    test_r2 = r2_score(y_test, test_pred)
    ax.set_title(f"Peak holdout — {name} (test R²={test_r2:.3f})")
    fig.tight_layout()
    parity_path = outputs / "parity_peak.png"
    fig.savefig(parity_path, dpi=140)
    plt.close(fig)

    return {"parity_png": str(parity_path), "test_r2": float(test_r2)}


def _write_curve_artifacts(
    pipe: Pipeline,
    name: str,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    X_test: pd.DataFrame,
    peaks_test: np.ndarray,
    outputs: Path,
) -> dict:
    test_slice = df.iloc[test_idx].reset_index(drop=True)
    test_pred = _predict_curve_loads(pipe, X_test, peaks_test, test_slice)

    pred_df = test_slice[["specimen", "specimen_id", CURVE_DISPLACEMENT, CURVE_LOAD]].copy()
    pred_df["predicted_load_kN"] = test_pred
    pred_path = outputs / "curves_pred_vs_actual.csv"
    pred_df.to_csv(pred_path, index=False)

    curve_err = specimen_curve_metrics(pred_df)
    curve_err.to_csv(outputs / "specimen_errors_curve.csv", index=False)

    y_test = test_slice[CURVE_LOAD].astype(float).values
    test_r2 = r2_score(y_test, test_pred)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False)
    axes = axes.ravel()
    ranked = curve_err.sort_values("r2")
    worst = ranked.head(2)["specimen_id"].tolist()
    best = ranked.tail(2)["specimen_id"].tolist()
    chosen = list(dict.fromkeys(worst + best))[:4]
    for ax, sid in zip(axes, chosen, strict=False):
        sub = pred_df[pred_df["specimen_id"] == sid].sort_values(CURVE_DISPLACEMENT)
        r2_s = ranked.loc[ranked["specimen_id"] == sid, "r2"].iloc[0]
        ax.plot(sub[CURVE_DISPLACEMENT], sub[CURVE_LOAD], "o", ms=3, label="actual")
        ax.plot(sub[CURVE_DISPLACEMENT], sub["predicted_load_kN"], "-", lw=1.5, label="pred")
        ax.set_title(f"{sid[:32]} (R²={r2_s:.2f})")
        ax.set_xlabel("disp (mm)")
        ax.set_ylabel("load (kN)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Curve holdout — {name} (test R²={test_r2:.3f})")
    fig.tight_layout()
    curve_png = outputs / "curves_pred_vs_actual.png"
    fig.savefig(curve_png, dpi=140)
    plt.close(fig)
    return {"curve_png": str(curve_png), "test_r2": float(test_r2)}


def train_peak(
    data_path: Path | None = None,
    target: str = PEAK_TARGET,
    models_out: Path | None = None,
    outputs: Path | None = None,
    show_progress: bool = True,
) -> dict:
    data_path = data_path or data_processed() / "dataset_ml.csv"
    models_out = models_out or models_dir()
    outputs = outputs or outputs_dir()
    models_out.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    return _train_mode(
        df,
        mode="peak",
        target=target,
        models_out=models_out,
        outputs=outputs,
        show_progress=show_progress,
    )


def train_curve(
    curves_path: Path | None = None,
    models_out: Path | None = None,
    outputs: Path | None = None,
    show_progress: bool = True,
) -> dict:
    curves_path = curves_path or data_processed() / "curves_backbone.csv"
    models_out = models_out or models_dir()
    outputs = outputs or outputs_dir()
    models_out.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(curves_path)
    return _train_mode(
        df,
        mode="curve",
        target=CURVE_LOAD,
        models_out=models_out,
        outputs=outputs,
        show_progress=show_progress,
    )


def _maybe_feature_importance(
    pipe: Pipeline, name: str | None, feature_cols: list[str], out_csv: Path
) -> None:
    if name not in {"random_forest", "xgboost", "gbr"} or pipe is None:
        return
    try:
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["pre"]
        names = list(pre.get_feature_names_out())
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return
        pd.DataFrame({"feature": names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(out_csv, index=False)
    except Exception:  # noqa: BLE001
        return
