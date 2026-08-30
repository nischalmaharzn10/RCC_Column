"""
RCC Column ML — Streamlit entry (thin shell).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

# Ensure repo root on path when launched via streamlit
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.web.components.predict_form import render_predict_tab

# Kept for re-enabling Models / Curves later:
# from apps.web.components.curve_charts import render_curves_tab
# from apps.web.components.results_charts import render_results_tab
# from src.shared.paths import data_processed, outputs_dir

# Reload layout on each run so dev edits apply without a full server restart.
import apps.web.components.layout as layout_module

importlib.reload(layout_module)
from apps.web.components.layout import inject_app_css

st.set_page_config(
    page_title="RCC Column ML",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

inject_app_css()

# --- Active UI: Predict only ---
render_predict_tab()

# --- Models / Curves / navbar (commented — uncomment to restore multi-page nav) ---
# processed = data_processed()
# outputs = outputs_dir()
# dataset_path = processed / "dataset_ml.csv"
# curves_path = processed / "curves_backbone.csv"
#
# NAV_PAGES: list[st.Page] = []
#
#
# def render_navbar(pages: list[st.Page]) -> None:
#     cols = st.columns([1, 1, 1, 24], gap="small")
#     for col, page in zip(cols[: len(pages)], pages, strict=True):
#         with col:
#             st.page_link(page, label=page.title)
#
#
# def page_predict() -> None:
#     render_navbar(NAV_PAGES)
#     render_predict_tab()
#
#
# def page_models() -> None:
#     render_navbar(NAV_PAGES)
#     render_results_tab(outputs, dataset_path, curves_path)
#
#
# def page_curves() -> None:
#     render_navbar(NAV_PAGES)
#     render_curves_tab(outputs, curves_path)
#
#
# predict_page = st.Page(page_predict, title="Predict", url_path="predict", default=True)
# models_page = st.Page(page_models, title="Models", url_path="models")
# curves_page = st.Page(page_curves, title="Curves", url_path="curves")
# NAV_PAGES[:] = [predict_page, models_page, curves_page]
#
# nav = st.navigation(NAV_PAGES, position="hidden")
# nav.run()
