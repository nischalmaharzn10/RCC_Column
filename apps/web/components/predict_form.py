"""Streamlit — Predict tab."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.predict.service import predict_curve, predict_peak
from src.shared.paths import models_dir

# Five fields per row so the form stays short and the chart stays on-screen.
_FIELD_ROWS: list[list[tuple[str, str, float, str | None]]] = [
    [
        ("fc_MPa", "fc (MPa)", 25.0, None),
        ("P_axial_kN", "P axial (kN)", 800.0, None),
        ("b_mm", "b (mm)", 400.0, None),
        ("h_mm", "h (mm)", 400.0, None),
        ("L_mm", "L (mm)", 1600.0, None),
    ],
    [
        ("cover_mm", "cover (mm)", 30.0, None),
        ("rho_long", "ρ long", 0.015, "%.4f"),
        ("rho_trans", "ρ trans", 0.01, "%.4f"),
        ("fyl_MPa", "fyl (MPa)", 400.0, None),
        ("fyt_MPa", "fyt (MPa)", 300.0, None),
    ],
    [
        ("db_long_mm", "db long (mm)", 16.0, None),
        ("n_long_bars", "n long bars", 12.0, None),
        ("db_trans_mm", "db trans (mm)", 10.0, None),
        ("s_hoop_mm", "s hoop (mm)", 100.0, None),
        ("steel_grade", "steel grade", 380.0, None),
    ],
]


def _chart_figure(displacement, load) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=displacement,
            y=load,
            mode="lines+markers",
            name="Predicted backbone",
            line={"color": "#2563eb", "width": 2.5},
            marker={"size": 5, "color": "#1d4ed8"},
            hovertemplate="Disp %{x:.1f} mm<br>Load %{y:.1f} kN<extra></extra>",
        )
    )
    fig.update_layout(
        margin={"l": 48, "r": 16, "t": 28, "b": 44},
        height=340,
        xaxis_title="Displacement (mm)",
        yaxis_title="Lateral load (kN)",
        xaxis={"gridcolor": "#e2e8f0", "zeroline": False},
        yaxis={"gridcolor": "#e2e8f0", "zeroline": False},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"color": "#334155", "size": 12},
        showlegend=False,
    )
    return fig


def render_predict_tab() -> None:
    peak_path = models_dir() / "best_model_peak.joblib"
    curve_path = models_dir() / "best_model_curve.joblib"
    if not peak_path.exists() and not curve_path.exists():
        st.warning("No saved models. Run `npm run train` first.")
        return

    st.markdown(
        """
        <div class="rcc-hero">
          <h1>RCC Column Predict</h1>
          <p>Enter design parameters to estimate peak capacity and the load–displacement backbone.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rcc-section-label">Design parameters</div>', unsafe_allow_html=True)

    values: dict[str, float] = {}
    for row in _FIELD_ROWS:
        cols = st.columns(5, gap="small")
        for col, (key, label, default, fmt) in zip(cols, row, strict=True):
            with col:
                kwargs: dict = {"value": default, "key": f"in_{key}"}
                if fmt is not None:
                    kwargs["format"] = fmt
                values[key] = float(st.number_input(label, **kwargs))

    area = values["b_mm"] * values["h_mm"]
    axial_ratio = (
        (values["P_axial_kN"] * 1000.0) / (values["fc_MPa"] * area)
        if values["fc_MPa"] * area > 0
        else 0.0
    )

    action_l, action_r = st.columns([4, 1], gap="small")
    with action_l:
        st.caption(f"Derived axial load ratio ≈ **{axial_ratio:.3f}**")
    with action_r:
        run = st.button("Predict", type="primary", use_container_width=True)

    inputs = {
        **values,
        "axial_load_ratio": axial_ratio,
        "Lsplice_mm": 0.0,
        "test_config": "DE",
        "failure_mode": 1,
    }

    if run or "last_prediction" not in st.session_state:
        peak_val: float | None = None
        curve_disp = None
        curve_load = None
        peak_error: str | None = None
        curve_error: str | None = None

        if peak_path.exists():
            try:
                peak_val = float(predict_peak(inputs))
            except Exception as exc:  # noqa: BLE001
                peak_error = str(exc)

        if curve_path.exists():
            try:
                curve = predict_curve(inputs)
                curve_disp = curve["displacement_mm"]
                curve_load = curve["predicted_load_kN"]
            except Exception as exc:  # noqa: BLE001
                curve_error = str(exc)

        st.session_state["last_prediction"] = {
            "peak": peak_val,
            "disp": curve_disp,
            "load": curve_load,
            "peak_error": peak_error,
            "curve_error": curve_error,
        }

    result = st.session_state["last_prediction"]

    if result["peak_error"]:
        st.error(f"Peak prediction failed: {result['peak_error']}")
    elif result["peak"] is not None:
        st.markdown(
            f"""
            <div class="rcc-peak">
              <span class="label">Predicted peak load</span>
              <span class="value">{result["peak"]:.1f} kN</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result["curve_error"]:
        st.error(f"Curve prediction failed: {result['curve_error']}")
    elif result["disp"] is not None and result["load"] is not None:
        st.markdown(
            '<div class="rcc-section-label">Predicted load–displacement backbone</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _chart_figure(result["disp"], result["load"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
