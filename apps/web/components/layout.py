"""Streamlit layout — app chrome CSS."""

from __future__ import annotations

import streamlit as st

_APP_CSS = """
<style>
    header[data-testid="stHeader"] {
        background: transparent;
        border-bottom: none;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    div[data-testid="stException"] a[href*="google.com/search"],
    div[data-testid="stException"] a[href*="chatgpt.com"] {
        display: none !important;
    }
</style>
"""


def inject_app_css() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)
