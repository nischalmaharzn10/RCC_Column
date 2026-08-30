"""Streamlit — Curves tab (holdout explorer: table + backbone chart)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _build_holdout_table(outputs: Path) -> pd.DataFrame | None:
    peak_path = outputs / "specimen_errors_peak.csv"
    curve_path = outputs / "specimen_errors_curve.csv"
    if not peak_path.exists() or not curve_path.exists():
        return None

    peak = pd.read_csv(peak_path)
    curve = pd.read_csv(curve_path)
    merged = peak.merge(
        curve[["specimen_id", "r2", "mae_kN", "n_points"]],
        on="specimen_id",
        how="inner",
    )
    merged["peak_err_pct"] = np.where(
        merged["actual_peak_kN"] != 0,
        merged["error_kN"] / merged["actual_peak_kN"] * 100.0,
        np.nan,
    )
    merged = merged.sort_values("specimen", kind="stable").reset_index(drop=True)
    merged.insert(0, "row", np.arange(len(merged)))
    merged.insert(1, "#", np.arange(1, len(merged) + 1))
    return merged


def _render_summary_metrics(table: pd.DataFrame) -> None:
    n = len(table)
    median_r2 = table["r2"].median()
    mean_peak_err = table["abs_pct_error"].mean()
    good_r2 = int((table["r2"] > 0.75).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Held-out specimens", f"{n}")
    c2.metric("Median curve R²", f"{median_r2:.3f}")
    c3.metric("Mean |peak error|", f"{mean_peak_err:.1f}%")
    c4.metric("Specimens R² > 0.75", f"{good_r2} / {n}")


def _render_detail_chart(outputs: Path, row: pd.Series) -> None:
    pred_path = outputs / "curves_pred_vs_actual.csv"
    if not pred_path.exists():
        st.info("Run `npm run train:curve` to generate curve predictions.")
        return

    pdf = pd.read_csv(pred_path)
    sub = pdf[pdf["specimen_id"] == row["specimen_id"]].sort_values("displacement_mm")
    if sub.empty:
        st.warning("No curve points for this specimen.")
        return

    residuals = sub["lateral_load_kN"] - sub["predicted_load_kN"]
    rmse = float(np.sqrt((residuals**2).mean()))

    st.markdown(
        f"**{row['specimen']}**  \n"
        f"Actual peak **{row['actual_peak_kN']:.1f} kN** — "
        f"predicted **{row['predicted_peak_kN']:.1f} kN** "
        f"({row['peak_err_pct']:+.1f}%) — "
        f"curve R²=**{row['r2']:.3f}** — "
        f"RMSE **{rmse:.1f} kN** — "
        f"**{int(row['n_points'])}** points"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["displacement_mm"],
            y=sub["lateral_load_kN"],
            mode="markers",
            name="actual (measured)",
            marker=dict(color="#5B9BD5", size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["displacement_mm"],
            y=sub["predicted_load_kN"],
            mode="lines",
            name="predicted (model)",
            line=dict(color="#E74C3C", width=2.5),
        )
    )
    fig.update_layout(
        xaxis_title="Displacement (mm)",
        yaxis_title="Lateral load (kN)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=30, b=40),
        height=520,
    )
    st.plotly_chart(fig, width="stretch")


def render_curves_tab(outputs: Path, curves_path: Path) -> None:
    table = _build_holdout_table(outputs)
    if table is None:
        st.warning("No holdout results yet. Run `npm run train` first.")
        if curves_path.exists():
            cdf = pd.read_csv(curves_path)
            specs = sorted(cdf["specimen"].dropna().unique().tolist())
            choice = st.selectbox("Browse actual backbone", specs[:200] if specs else [])
            if choice:
                sub = cdf[cdf["specimen"] == choice].sort_values("displacement_mm")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sub["displacement_mm"],
                        y=sub["lateral_load_kN"],
                        mode="lines+markers",
                        name="actual backbone",
                    )
                )
                fig.update_layout(
                    title=choice,
                    xaxis_title="Displacement (mm)",
                    yaxis_title="Lateral load (kN)",
                )
                st.plotly_chart(fig, width="stretch")
        return

    _render_summary_metrics(table)

    display = table.rename(
        columns={
            "specimen": "Specimen",
            "actual_peak_kN": "Actual peak (kN)",
            "predicted_peak_kN": "Pred peak (kN)",
            "peak_err_pct": "Peak err %",
            "r2": "Curve R²",
        }
    )

    left, right = st.columns([5, 7], gap="medium")

    with left:
        selection = st.dataframe(
            display[
                ["#", "Specimen", "Actual peak (kN)", "Pred peak (kN)", "Peak err %", "Curve R²"]
            ],
            hide_index=True,
            height=560,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="holdout_specimen_table",
            column_config={
                "Actual peak (kN)": st.column_config.NumberColumn(format="%.1f"),
                "Pred peak (kN)": st.column_config.NumberColumn(format="%.1f"),
                "Peak err %": st.column_config.NumberColumn(format="%+.1f"),
                "Curve R²": st.column_config.ProgressColumn(
                    min_value=-1.0,
                    max_value=1.0,
                    format="%.3f",
                ),
            },
        )
        selected_rows = selection.selection.rows if selection.selection else []
        row_idx = selected_rows[0] if selected_rows else 0

    with right:
        _render_detail_chart(outputs, table.iloc[row_idx])
