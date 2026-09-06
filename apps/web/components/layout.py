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
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 100% !important;
    }

    /* Tighter number inputs so 3 rows leave room for the chart */
    div[data-testid="stNumberInput"] {
        margin-bottom: 0.15rem;
    }

    div[data-testid="stNumberInput"] label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }

    div[data-testid="stNumberInput"] input {
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }

    .rcc-hero {
        margin-bottom: 0.75rem;
    }

    .rcc-hero h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
    }

    .rcc-hero p {
        margin: 0.2rem 0 0;
        color: #64748b;
        font-size: 0.95rem;
    }

    .rcc-section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0.25rem 0 0.5rem;
    }

    .rcc-peak {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        padding: 0.65rem 0.9rem;
        margin: 0.5rem 0 0.35rem;
        border-radius: 0.5rem;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .rcc-peak .label {
        font-size: 0.85rem;
        font-weight: 600;
    }

    .rcc-peak .value {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Soft sage primary — quieter than Streamlit's default red */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        background-color: #7A9E7E !important;
        background-image: none !important;
        border: 1px solid #6B8F6F !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 0.55rem !important;
        box-shadow: none !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
        background-color: #6B8F6F !important;
        border-color: #5F8163 !important;
        color: #ffffff !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:focus,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:focus {
        box-shadow: 0 0 0 2px rgba(122, 158, 126, 0.35) !important;
    }

    div[data-testid="stException"] a[href*="google.com/search"],
    div[data-testid="stException"] a[href*="chatgpt.com"] {
        display: none !important;
    }
</style>
"""


def inject_app_css() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)
