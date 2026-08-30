# Setup

How to install dependencies and **start** the RCC Column ML app (same role as `setup.md` / env docs in our other repos).

## Required

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | 3.11 / 3.12 fine |
| Node.js + npm | 18+ (optional) | Only if you want `npm run …` shortcuts |
| Git | any recent | |

## Optional

| Tool | Purpose |
|------|---------|
| Ruff | Lint / format |
| ANSYS | Export CSVs matching the PEER property/curve schema |

---

## 1. Install (one-time)

```bash
cd RCC_Column
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Optional: npm scripts (npm run dev, etc.)
npm install
```

Verify:

```bash
python -c "import pandas, sklearn, xgboost, streamlit, plotly; print('ok')"
```

---

## 2. How to start (dev)

Activate the venv first, then use **either** npm scripts or the raw commands.

### npm scripts (recommended shortcuts)

| Command | What it does |
|---------|----------------|
| `npm run dev` | Start Streamlit UI (hot-reload) — **main day-to-day start** |
| `npm run build:data` | Build processed CSVs from properties (+ curves) |
| `npm run train:peak` | Train peak-load models |
| `npm run train:curve` | Train backbone-curve models |
| `npm run train` | Train peak then curve |
| `npm run lint` | Ruff check |
| `npm run test` | Pytest |
| `npm start` | Alias of `npm run dev` |

```bash
# Everyday: open the UI
npm run dev
```

Then open the URL Streamlit prints (usually **http://localhost:8501**).

Full local pipeline once code is in place:

```bash
npm run build:data
npm run train
npm run dev
```

### Raw Python (no npm)

```bash
# UI
streamlit run apps/web/streamlit_app.py

# Data + train
python -m src.dataset.build_dataset --properties data/raw/properties_rect.csv --out data/processed
python -m src.train.train --mode peak --data data/processed/dataset_ml.csv
python -m src.train.train --mode curve --curves data/processed/curves_backbone.csv
```

---

## Environment variables

Copy `.env.example` → `.env` if you need path overrides. Defaults are fine for local work.

| Variable | Default | Meaning |
|----------|---------|---------|
| `RCC_DATA_RAW` | `data/raw` | Raw inputs |
| `RCC_DATA_PROCESSED` | `data/processed` | Built CSVs |
| `RCC_MODELS_DIR` | `models` | joblib models |
| `RCC_OUTPUTS_DIR` | `outputs` | Metrics / plots |

Never commit `.env`.

## Data

- Sample features: `data/raw/properties_rect.csv`
- ANSYS exports: same columns + `source=ansys`
- PEER force–displacement downloads: `data/raw/peer_curves/` (gitignored)

## Hosting (free)

Live URL via **Streamlit Community Cloud** (not GitHub Pages). Keep the GitHub repo **private**, commit the two `best_model_*.joblib` files, deploy `apps/web/streamlit_app.py`. Pushes to the connected branch auto-redeploy the app. Docker is optional and not required.

Full steps, security notes, and Docker vs paid hosting: [apps/docs/guides/hosting.md](apps/docs/guides/hosting.md).

## More

- [Getting started](apps/docs/guides/getting-started.md)
- [Hosting](apps/docs/guides/hosting.md)
- [Platform guide](apps/docs/core/platform-guide.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
