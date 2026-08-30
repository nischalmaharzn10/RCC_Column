# Changelog

High-level project changelog. Behavior details and UI/domain notes live in [apps/docs/core/platform-guide.md](apps/docs/core/platform-guide.md).

## Unreleased

### Added

- Repository organization aligned with Kaneo/jira pattern (`apps/`, `src/{feature}/`, `tests/`, `plans/`)
- Cursor rules under `.cursor/rules/`
- Living docs under `apps/docs/`
- Sample `data/raw/properties_rect.csv`
- `requirements.txt`, `pyproject.toml`, governance markdown (README, CONTRIBUTING, CLAUDE, setup.md)
- `package.json` scripts: `npm run dev`, `build:data`, `train`, etc.
- PEER Structural Performance Database under `samples/`: rectangular properties (253), 286 force–displacement curves, manuals, curve index

### Pipeline (planned next)

- ~~`src.dataset` build + PEER/backbone assembly~~
- ~~`src.train` peak & curve baselines~~
- ~~`src.predict` + Streamlit tabs~~

### Trained (local)

- Peak best: XGBoost (~R² 0.98 on held-out specimens)
- Curve best: Random Forest (~R² 0.92)
- Artifacts: `models/best_model_*.joblib`, `outputs/metrics_*.csv`
