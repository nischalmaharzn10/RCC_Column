# ML Pipeline

**Cursor rules:** `.cursor/rules/ml-pipeline.mdc`, `.cursor/rules/ml-evaluation.mdc`

## Overview

1. **Build** — `src.dataset` writes processed CSVs  
2. **Train** — regularized models + group CV + **fit scenario diagnostics**  
3. **Predict** — `src.predict` loads joblib pipelines for UI / CLI  

## Train

```bash
npm run train
# or peak / curve separately
```

CLI shows a **tqdm** progress bar (CV folds → fit → diagnostics).

## Peak mode

- Split: `GroupShuffleSplit` on `specimen_id`
- **5-fold group CV** on train specimens; pick model by **lowest CV RMSE** (penalize overfit gap)
- Features: `TRAINING_FEATURE_COLUMNS` (excludes derived `axial_load_ratio`)
- Outputs:
  - `metrics_peak.csv` — train/test/CV for RF, XGB, MLP
  - `training_diagnostics_peak.json` — scenario + recommendations
  - `specimen_errors_peak.csv` — per-specimen error, outlier flag
  - `parity_peak.csv` / `.png`

## Curve mode

- Pointwise `(params + displacement) → load`
- Same CV + selection logic
- `specimen_errors_curve.csv` — R², `poor_fit`, `softening_miss`
- Curve PNG: **worst 2 + best 2** test specimens

## Fit scenarios (automatic)

| Scenario | Meaning |
|----------|---------|
| `good_fit` | Acceptable test R², small train-test gap |
| `overfit` / `overfit_severe` | Train R² much higher than test |
| `underfit` | Both train and test R² too low |
| `high_cv_variance` | CV RMSE unstable across folds |
| `optimistic_holdout` | Test split easier than CV suggests |
| `pessimistic_holdout` | Test split harder than CV |
| `peak_outliers` | Specimens with >15% peak error |
| `curve_poor_specimens` | Specimen R² < 0.60 |
| `curve_softening_miss` | Post-peak drop not captured |

See Streamlit **Model results** tab or `outputs/training_diagnostics_*.json`.

## Leakage rule

Never randomly split individual backbone points from the same specimen across train and test.

## Artifacts

| Path | Contents |
|------|----------|
| `models/*.joblib` | Full preprocess + model (refit on all data) |
| `outputs/metrics_*.csv` | Leaderboard |
| `outputs/training_diagnostics_*.json` | Scenario assessment |
| `outputs/specimen_errors_*.csv` | Worst-case analysis |
| `outputs/*parity*` | Actual vs predicted |

## Extending

New model, threshold, or scenario → update `diagnostics.py`, rules, this guide, platform-guide Changelog.
