# Getting Started

## 1. Clone and enter the repo

```bash
cd RCC_Column
```

## 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
npm install
```

Full install + start notes: [setup.md](../../../setup.md).

## 3. Sample data

A starter properties file is at `data/raw/properties_rect.csv` (schema matches PEER-style rectangular column properties).

## 4. Build dataset (when implemented)

```bash
npm run build:data
# or: python -m src.dataset.build_dataset --properties data/raw/properties_rect.csv --out data/processed
```

Produces `dataset_ml.csv` and `curves_backbone.csv`.

## 5. Train

```bash
npm run train
# or: npm run train:peak && npm run train:curve
```

## 6. Launch UI (start here day-to-day)

```bash
npm run dev
```

Open **http://localhost:8501** (or the URL Streamlit prints).

Equivalent: `streamlit run apps/web/streamlit_app.py`

## Next reading

- [Platform guide](../core/platform-guide.md)
- [Data schema](data-schema.md)
- [ML pipeline](ml-pipeline.md)
