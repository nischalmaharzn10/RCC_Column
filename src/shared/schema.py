"""Shared schema constants and helpers for RCC Column ML."""

from __future__ import annotations

from src.shared.features import ENGINEERED_FEATURE_COLUMNS

# Used for inference / forms (includes derived axial_load_ratio)
FEATURE_COLUMNS: list[str] = [
    "fc_MPa",
    "P_axial_kN",
    "b_mm",
    "h_mm",
    "L_mm",
    "Lsplice_mm",
    "db_long_mm",
    "n_long_bars",
    "cover_mm",
    "rho_long",
    "fyl_MPa",
    "steel_grade",
    "db_trans_mm",
    "s_hoop_mm",
    "rho_trans",
    "fyt_MPa",
    "axial_load_ratio",
]

# Training excludes axial_load_ratio; adds engineered structural indices
TRAINING_FEATURE_COLUMNS: list[str] = [
    c for c in FEATURE_COLUMNS if c != "axial_load_ratio"
] + list(ENGINEERED_FEATURE_COLUMNS)

# Curve shape model also uses displacement ratios
CURVE_SHAPE_FEATURE_COLUMNS: list[str] = [
    "disp_over_L",
    "disp_ratio",
]

CATEGORICAL_COLUMNS: list[str] = [
    "test_config",
    "failure_mode",
    "source",
]

ID_COLUMNS: list[str] = [
    "specimen",
    "specimen_id",
]

PEAK_TARGET = "peak_load_kN"
DISP_AT_PEAK_TARGET = "disp_at_peak_mm"
CURVE_DISPLACEMENT = "displacement_mm"
CURVE_LOAD = "lateral_load_kN"
CURVE_LOAD_RATIO = "load_ratio"

PROPERTIES_REQUIRED: list[str] = [
    "specimen",
    "fc_MPa",
    "P_axial_kN",
    "b_mm",
    "h_mm",
    "L_mm",
    "Lsplice_mm",
    "test_config",
    "db_long_mm",
    "n_long_bars",
    "cover_mm",
    "rho_long",
    "fyl_MPa",
    "steel_grade",
    "db_trans_mm",
    "s_hoop_mm",
    "rho_trans",
    "fyt_MPa",
    "failure_mode",
    "axial_load_ratio",
]
