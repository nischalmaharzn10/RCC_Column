# RCC Column ML

Predict the **load–deformation** response of reinforced concrete columns from design parameters — peak lateral capacity and the backbone curve — in an interactive web app.

**Live demo:** [rcc-column.streamlit.app](https://rcc-column.streamlit.app)

## What it does

You enter column design inputs (geometry, concrete strength, axial load, longitudinal and transverse reinforcement). The app returns:

- **Peak load** (kN)
- **Load–displacement backbone** chart for the predicted response

Models are trained on PEER rectangular column experiments (and can use ANSYS runs in the same schema). Units are SI only (mm, kN, MPa).

## Try it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
npm run build:data
npm run train
npm run dev                     # → http://localhost:8501
```

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start the Streamlit app |
| `npm run build:data` | Build processed training tables |
| `npm run train` | Train peak and curve models |

More detail: [setup.md](setup.md).

## Learn more

| Doc | Purpose |
|-----|---------|
| [Getting started](apps/docs/guides/getting-started.md) | Setup walkthrough |
| [Platform guide](apps/docs/core/platform-guide.md) | Domain, modes, changelog |
| [Data schema](apps/docs/guides/data-schema.md) | Input / CSV contracts |
| [ML pipeline](apps/docs/guides/ml-pipeline.md) | Training and evaluation |
| [Hosting](apps/docs/guides/hosting.md) | Deploy to Streamlit Cloud |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## License

Internal / project use unless otherwise stated.
