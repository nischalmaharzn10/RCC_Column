# RCC Column ML

Machine-learning surrogate for **RCC column load–deformation** response, with interactive graphs in Streamlit.

Organization follows the same governance pattern as our Kaneo/jira repos: `apps/` + feature modules under `src/` + living docs + `.cursor/rules`.

## Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Models | scikit-learn (RF, MLP), XGBoost |
| Persist | joblib |
| Explain | SHAP |
| Charts | Plotly (UI), matplotlib (static) |
| UI | Streamlit (`apps/web/`) |
| Lint | Ruff |

## Repository layout

```
RCC_Column/
├── apps/
│   ├── web/                 # Streamlit UI
│   └── docs/                # Canonical documentation
├── src/
│   ├── dataset/             # Build ML tables
│   ├── train/               # Peak & curve training
│   ├── predict/             # Inference
│   └── shared/              # Schema & shared helpers
├── samples/                 # PEER downloads (properties + curves)
├── data/raw|processed/
├── models/                  # joblib artifacts
├── outputs/                 # metrics & plots
├── tests/                   # mirrors src features
├── plans/                   # numbered design plans
├── scripts/
├── .cursor/rules/           # Cursor agent rules
├── requirements.txt
├── pyproject.toml
├── CONTRIBUTING.md
├── CLAUDE.md
├── setup.md
├── package.json
└── CHANGELOG.md
```

## How to start

```bash
cd RCC_Column
python -m venv .venv
.venv\Scripts\activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install                     # optional — enables npm run scripts
```

| Command | What it does |
|---------|----------------|
| **`npm run dev`** | Start Streamlit UI → http://localhost:8501 |
| `npm run build:data` | Build processed datasets |
| `npm run train` | Train peak + curve models |
| `npm start` | Same as `npm run dev` |

Without npm:

```bash
streamlit run apps/web/streamlit_app.py
```

Full details: [setup.md](setup.md).

## Documentation

| Doc | Purpose |
|-----|---------|
| [apps/docs/index.md](apps/docs/index.md) | Docs home |
| [Platform guide](apps/docs/core/platform-guide.md) | Domain, modes, UI, **Changelog** |
| [Getting started](apps/docs/guides/getting-started.md) | Setup walkthrough |
| [Data schema](apps/docs/guides/data-schema.md) | CSV contracts |
| [ML pipeline](apps/docs/guides/ml-pipeline.md) | Train / evaluate |
| [Hosting](apps/docs/guides/hosting.md) | Private Streamlit Cloud (free), security, auto-update |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [setup.md](setup.md) | Install, env vars, **how to start** |

## Cursor rules

All agent rules live in [`.cursor/rules/`](.cursor/rules/). Always-applied: project overview, conventions, feature folders, enterprise architecture, platform guide. Scoped rules cover data schema, ML pipeline, Streamlit UI, and documentation.

**When you add a feature:** update code + `platform-guide.md` Changelog + the matching `.mdc` rule in the same change.

## Training data

PEER rectangular column set is in [`samples/`](samples/README.md) (properties + force–displacement curves).

## Status

- **Done:** PEER `samples/`, dataset build, peak + curve training, Streamlit UI (Data / Results / Curves / Predict)
- **Start:** `npm run build:data` → `npm run train` → `npm run dev`

## License

Internal / project use unless otherwise stated.
