# CLAUDE.md

Guidance for AI agents working in this repository.

## Project

RCC Column ML — predict RCC column load–deformation (peak + backbone) from design parameters; Streamlit UI for graphs and live prediction.

**Philosophy:** Solve the real modelling/UI need. Do not over-engineer. Keep `apps/web` thin and logic in `src/{feature}/`.

## Commands

Prefer npm scripts (see `setup.md` / `package.json`):

```bash
pip install -r requirements.txt
npm install

npm run build:data    # build processed CSVs
npm run train         # peak + curve
npm run dev           # Streamlit UI → http://localhost:8501

npm run lint
npm run test
```

Raw equivalents:

```bash
python -m src.dataset.build_dataset --properties data/raw/properties_rect.csv --out data/processed
python -m src.train.train --mode peak --data data/processed/dataset_ml.csv
python -m src.train.train --mode curve --curves data/processed/curves_backbone.csv
streamlit run apps/web/streamlit_app.py
```

## Architecture

```
apps/web/     → Streamlit presentation (tabs, Plotly)
apps/docs/    → Canonical docs + platform-guide Changelog
src/dataset/  → properties + curves → processed CSVs
src/train/    → RF / XGBoost / MLP, metrics, joblib
src/predict/  → load models, predict peak/curve
src/shared/   → schema constants, shared IO
```

## Hard constraints

1. Feature folders — see `.cursor/rules/feature-folder-structure.mdc`
2. Always split by `specimen_id` — no leakage
3. SI units (mm, kN, MPa)
4. Every behavior change updates `apps/docs/core/platform-guide.md` Changelog **and** the matching `.cursor/rules` file
5. Verify from the repo; do not assume PEER/ANSYS files exist until checked

## Rules to read first

| Rule | When |
|------|------|
| `project-overview.mdc` | Always |
| `platform-guide.mdc` | Domain / modes / UI tabs |
| `feature-folder-structure.mdc` | Where to put files |
| `data-schema.mdc` | CSV columns |
| `ml-pipeline.mdc` | Training / artifacts |
| `streamlit-ui.mdc` | UI work |
| `documentation.mdc` | Docs / changelog |
| `enterprise-architecture.mdc` | Analysis before large changes |

## Do not

- Put business logic in `apps/web/streamlit_app.py`
- Add React/LSTM/Optuna unless explicitly requested
- Edit applied plans under `plans/` as if they were code contracts without updating status
- Commit `.env`, secrets, or huge regenerable PEER dumps
