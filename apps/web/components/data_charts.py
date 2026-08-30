"""Streamlit chart helpers — Data tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.shared.schema import FEATURE_COLUMNS


def render_data_tab(dataset_path: Path, curves_path: Path) -> None:
    if not dataset_path.exists():
        st.warning(
            f"Missing `{dataset_path}`. Run `npm run build:data` first."
        )
        return

    df = pd.read_csv(dataset_path)
    st.subheader("Specimen dataset")
    st.caption(f"{len(df)} specimens with peak targets")
    st.dataframe(df.head(50), use_container_width=True)

    cols = [c for c in ["fc_MPa", "rho_long", "axial_load_ratio", "peak_load_kN", "b_mm", "h_mm"] if c in df.columns]
    pick = st.multiselect("Histograms", cols, default=cols[:4])
    for c in pick:
        fig = px.histogram(df, x=c, nbins=30, title=c)
        st.plotly_chart(fig, use_container_width=True)

    num = df[[c for c in FEATURE_COLUMNS + ["peak_load_kN"] if c in df.columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True)

    if curves_path.exists():
        cdf = pd.read_csv(curves_path)
        st.caption(f"Backbone points: {len(cdf):,}")
