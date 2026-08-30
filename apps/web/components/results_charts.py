"""Streamlit — Model results tab."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _render_diagnostics(outputs: Path, mode: str) -> None:
    path = outputs / f"training_diagnostics_{mode}.json"
    if not path.exists():
        return
    diag = json.loads(path.read_text(encoding="utf-8"))
    severity = diag.get("severity", "ok")
    primary = diag.get("primary_scenario", "unknown")
    flags = diag.get("flags", [])

    if severity == "critical":
        st.error(f"**{mode.title()} fit — {primary}** (critical)")
    elif severity == "warn":
        st.warning(f"**{mode.title()} fit — {primary}** (review recommended)")
    else:
        st.success(f"**{mode.title()} fit — {primary}**")

    for msg in diag.get("messages", []):
        st.caption(f"• {msg}")

    with st.expander(f"{mode.title()} scenario flags & recommendations"):
        st.write("**Flags:**", ", ".join(flags) if flags else "none")
        for rec in diag.get("recommendations", []):
            st.markdown(f"- {rec}")

    err_name = f"specimen_errors_{mode}.csv"
    err_path = outputs / err_name
    if err_path.exists():
        err = pd.read_csv(err_path)
        if mode == "peak" and "is_outlier" in err.columns:
            outliers = err[err["is_outlier"]]
            if len(outliers):
                st.subheader(f"Peak outliers (>{15}% error)")
                st.dataframe(outliers.head(10), use_container_width=True)
        if mode == "curve":
            poor = err[err["poor_fit"]] if "poor_fit" in err.columns else pd.DataFrame()
            soft = err[err["softening_miss"]] if "softening_miss" in err.columns else pd.DataFrame()
            if len(poor):
                st.subheader("Poor curve specimens (low R²)")
                st.dataframe(poor.head(10), use_container_width=True)
            if len(soft):
                st.subheader("Missed post-peak softening")
                st.dataframe(soft.head(10), use_container_width=True)


def render_results_tab(
    outputs: Path,
    dataset_path: Path | None = None,
    curves_path: Path | None = None,
) -> None:
    if dataset_path and dataset_path.exists():
        import pandas as pd

        n_spec = len(pd.read_csv(dataset_path))
        n_curve = len(pd.read_csv(curves_path)) if curves_path and curves_path.exists() else 0
        st.caption(
            f"Dataset: **{n_spec}** PEER rectangular specimens · "
            f"**{n_curve:,}** backbone points (25 per specimen)"
        )

    metrics_peak = outputs / "metrics_peak.csv"
    parity = outputs / "parity_peak.csv"
    importance = outputs / "feature_importance_peak.csv"
    metrics_curve = outputs / "metrics_curve.csv"

    if not metrics_peak.exists():
        st.warning("No peak metrics yet. Run `npm run train:peak`.")
        return

    st.subheader("Fit scenarios (overfit / underfit / …)")
    _render_diagnostics(outputs, "peak")

    st.subheader("Peak model leaderboard")
    st.caption(
        "Selection: lowest group-CV RMSE; penalize ΔR² > 0.05; "
        "skip severe overfit/underfit when possible. Training excludes `axial_load_ratio`."
    )
    mdf = pd.read_csv(metrics_peak)
    display_cols = [
        c
        for c in [
            "model",
            "cv_rmse_mean",
            "cv_rmse_std",
            "train_r2",
            "test_r2",
            "overfit_r2_gap",
            "train_rmse",
            "test_rmse",
            "test_mae",
        ]
        if c in mdf.columns
    ]
    st.dataframe(mdf[display_cols].sort_values("cv_rmse_mean"), use_container_width=True)

    if parity.exists():
        pdf = pd.read_csv(parity)
        fig = px.scatter(
            pdf,
            x="actual",
            y="predicted",
            hover_data=["specimen"] if "specimen" in pdf.columns else None,
            title="Parity — peak load (holdout test specimens only)",
            labels={"actual": "Actual peak_load_kN", "predicted": "Predicted"},
        )
        lim0 = min(pdf["actual"].min(), pdf["predicted"].min())
        lim1 = max(pdf["actual"].max(), pdf["predicted"].max())
        fig.add_shape(
            type="line",
            x0=lim0,
            y0=lim0,
            x1=lim1,
            y1=lim1,
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if importance.exists():
        idf = pd.read_csv(importance).head(20)
        fig = px.bar(idf, x="importance", y="feature", orientation="h", title="Feature importance")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    if metrics_curve.exists():
        st.subheader("Curve mode")
        _render_diagnostics(outputs, "curve")
        st.subheader("Curve model leaderboard")
        cdf = pd.read_csv(metrics_curve)
        curve_cols = [c for c in display_cols if c in cdf.columns]
        st.dataframe(cdf[curve_cols].sort_values("cv_rmse_mean"), use_container_width=True)

    png = outputs / "parity_peak.png"
    if png.exists():
        st.image(str(png), caption="Holdout parity (static PNG)")

    curve_png = outputs / "curves_pred_vs_actual.png"
    if curve_png.exists():
        st.image(str(curve_png), caption="Curve holdout — best & worst specimens by R²")
