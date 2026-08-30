"""Streamlit — Predict tab."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.predict.service import predict_curve, predict_peak
from src.shared.paths import models_dir


def render_predict_tab() -> None:
    peak_path = models_dir() / "best_model_peak.joblib"
    curve_path = models_dir() / "best_model_curve.joblib"
    if not peak_path.exists() and not curve_path.exists():
        st.warning("No saved models. Run `npm run train` first.")
        return

    st.subheader("Design parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        fc = st.number_input("fc_MPa", value=25.0)
        b = st.number_input("b_mm", value=400.0)
        h = st.number_input("h_mm", value=400.0)
        L = st.number_input("L_mm", value=1600.0)
        cover = st.number_input("cover_mm", value=30.0)
    with c2:
        P = st.number_input("P_axial_kN", value=800.0)
        rho_long = st.number_input("rho_long", value=0.015, format="%.4f")
        rho_trans = st.number_input("rho_trans", value=0.01, format="%.4f")
        fyl = st.number_input("fyl_MPa", value=400.0)
        fyt = st.number_input("fyt_MPa", value=300.0)
    with c3:
        db_long = st.number_input("db_long_mm", value=16.0)
        n_long = st.number_input("n_long_bars", value=12.0)
        db_trans = st.number_input("db_trans_mm", value=10.0)
        s_hoop = st.number_input("s_hoop_mm", value=100.0)
        steel_grade = st.number_input("steel_grade", value=380.0)

    area = b * h
    axial_ratio = (P * 1000.0) / (fc * area) if fc * area > 0 else 0.0
    st.caption(f"Derived axial_load_ratio ≈ {axial_ratio:.3f}")

    inputs = {
        "fc_MPa": fc,
        "P_axial_kN": P,
        "b_mm": b,
        "h_mm": h,
        "L_mm": L,
        "Lsplice_mm": 0.0,
        "db_long_mm": db_long,
        "n_long_bars": n_long,
        "cover_mm": cover,
        "rho_long": rho_long,
        "fyl_MPa": fyl,
        "steel_grade": steel_grade,
        "db_trans_mm": db_trans,
        "s_hoop_mm": s_hoop,
        "rho_trans": rho_trans,
        "fyt_MPa": fyt,
        "axial_load_ratio": axial_ratio,
        "test_config": "DE",
        "failure_mode": 1,
    }

    if st.button("Predict", type="primary"):
        if peak_path.exists():
            try:
                peak = predict_peak(inputs)
                st.success(f"Predicted peak load: **{peak:.1f} kN**")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Peak prediction failed: {exc}")

        if curve_path.exists():
            try:
                curve = predict_curve(inputs)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=curve["displacement_mm"],
                        y=curve["predicted_load_kN"],
                        mode="lines+markers",
                        name="predicted backbone",
                    )
                )
                fig.update_layout(
                    title="Predicted load–displacement backbone",
                    xaxis_title="Displacement (mm)",
                    yaxis_title="Lateral load (kN)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Curve prediction failed: {exc}")
