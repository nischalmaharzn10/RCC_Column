# Platform Guide

Canonical living document for RCC Column ML domain behavior.  
**Agent mirror:** `.cursor/rules/platform-guide.mdc` — keep condensed facts in sync.

When you change behavior, update this file’s **Changelog** in the same change.

## Goal

Train an ML surrogate that predicts an RCC column’s **load–deformation** response (peak capacity and/or full backbone curve) from design parameters. Validate against PEER experiments; optionally include ANSYS runs that share the same schema.

## Architecture (apps + features)

```
properties CSV + PEER/ANSYS curves
        ↓
  src/dataset  →  data/processed/*.csv
        ↓
  src/train    →  models/*.joblib + outputs/*
        ↓
  src/predict  →  apps/web (Streamlit graphs + live predict)
```

## Data sources

| Source | Role |
|--------|------|
| PEER UW rectangular DB | Experimental specimens — stored in `samples/` |
| ANSYS CSV exports | Numerical; same columns + `source=ansys` |
| `samples/properties/rectangular_properties.csv` | Feature table (253 specimens) |
| `samples/curves/rectangular/` | Force–displacement histories (286 files) |
| `data/raw/properties_rect.csv` | Small hand sample / normalized working copy |

Properties alone are not enough: peak and curve **targets** come from force–displacement (backbone) data.

## Modelling modes

| Mode | Row meaning | Prediction |
|------|-------------|------------|
| **Peak** | One specimen | `peak_load_kN` (and optionally `disp_at_peak_mm`) |
| **Curve** | One backbone point | `lateral_load_kN` given design params + `displacement_mm` |

Always split by `specimen_id` (GroupShuffleSplit / group CV).

## Feature columns

See [data-schema.md](../guides/data-schema.md). Summary:

`specimen`, `fc_MPa`, `P_axial_kN`, `b_mm`, `h_mm`, `L_mm`, `Lsplice_mm`, `test_config`, `db_long_mm`, `n_long_bars`, `cover_mm`, `rho_long`, `fyl_MPa`, `steel_grade`, `db_trans_mm`, `s_hoop_mm`, `rho_trans`, `fyt_MPa`, `failure_mode`, `axial_load_ratio`, optional `source`

Units: **SI only** (mm, kN, MPa).

## Baseline models

Random Forest, XGBoost, MLP — compare R² / RMSE / MAE; persist best pipeline as joblib.

## Streamlit UI (`apps/web`)

| Surface | Content |
|---------|---------|
| Predict (home) | Design inputs → peak + predicted curve |

No navbar / Models / Curves routes in the app shell.

UI must call `src.*.service` — no fitting inside Streamlit.

## Out of scope (v1)

React frontend, LSTM sequence models, Optuna deep search, ACI/Eurocode equation benchmark.

## Hosting

Deploy Predict UI on **Streamlit Community Cloud** from a **private** GitHub repo (free: one private app). Entry: `apps/web/streamlit_app.py`. Production joblibs `models/best_model_{peak,curve}.joblib` are tracked for cloud inference. Push to the connected branch auto-redeploys. Details: `apps/docs/guides/hosting.md`.

## Changelog

| Date | Change |
|------|--------|
| 2026-08-30 | Hosting guide: private Streamlit Cloud, tracked deploy joblibs, `runtime.txt` |
| 2026-08-30 | Streamlit: Predict-only UI; Models/Curves nav kept commented in `streamlit_app.py` |
| 2026-08-30 | Data+train upgrade: 251 specimens (name-match recover), 40-pt backbones, engineered features, two-stage curve (load_ratio)+softening; peak GBR / curve XGB both `good_fit` |
| 2026-08-30 | Curves tab: holdout explorer (KPI row, specimen table, backbone chart) per reference UI |
| 2026-08-30 | Streamlit: top nav routes `/predict`, `/models`, `/curves`; compact header; Predict first |
| 2026-08-30 | Anti-overfit training: group 5-fold CV, regularized models, drop `axial_load_ratio`, train/test metrics in UI |
| 2026-08-29 | Pipeline live: build from `samples/` → train peak/curve → Streamlit tabs |
| 2026-08-29 | Renamed env doc to `setup.md`; added `package.json` with `npm run dev` / `train` / `build:data` |
| 2026-08-29 | Initial organization: `apps/`, `src/{dataset,train,predict,shared}/`, docs, Cursor rules, sample properties CSV |
