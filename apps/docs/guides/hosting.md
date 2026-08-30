# Hosting (Streamlit Community Cloud)

How to put the Predict UI on the internet **without Docker**, keep code/models private, and get auto-updates on push.

## Short answers

| Question | Answer |
|----------|--------|
| Is hosting free? | Yes — [Streamlit Community Cloud](https://share.streamlit.io) free tier |
| Does GitHub host the live app? | No — GitHub stores code; Streamlit Cloud runs the app |
| Will the app update when I push code? | Yes — push to the connected branch → auto redeploy |
| Do I need Docker? | No for this path. Docker itself is free software; **paid** means cloud VMs that run containers (AWS/GCP/etc.), which you do not need here |
| Can the public see my code/models? | Only if the GitHub repo is **public**. Keep the repo **private** |

## Security model (what you want)

1. **GitHub repository → Private**  
   Only people you invite as collaborators can clone code or download `models/*.joblib`.
2. **Streamlit app → Private** (free: **one** private app)  
   Only emails you add as viewers (plus your workspace) can open the URL. Random internet users cannot use the app.
3. **Do not commit secrets**  
   `.env` and `.streamlit/secrets.toml` stay gitignored. Use Streamlit Cloud “Secrets” UI if you need env vars later.
4. **Never flip the repo to Public** while production joblibs are in git — anyone could then download the models.

Important nuance: Streamlit Cloud (and your GitHub collaborators) **can** read the private repo. “Private” means hidden from the public, not from the hosting service or teammates.

## What must be in the repo for Predict to work

| Path | Why |
|------|-----|
| `apps/web/streamlit_app.py` | App entry |
| `src/` | Predict + shared code |
| `requirements.txt` | Cloud installs deps |
| `runtime.txt` | Pins Python 3.11 |
| `models/best_model_peak.joblib` | Peak inference |
| `models/best_model_curve.joblib` | Curve inference |

Other `models/*.joblib` stay gitignored. Processed CSVs are not required for Predict-only.

Train locally before push:

```bash
npm run build:data
npm run train
```

Confirm both joblibs exist under `models/`, then commit them with the code.

## Deploy steps

1. Push this project to GitHub with the repo set to **Private**.
2. Open [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (allow access to private repos).
3. **New app** → pick `nischalmaharzn10/RCC_Column` (or your fork).
4. Main file path: `apps/web/streamlit_app.py`
5. Branch: `main` (or whatever you connected).
6. Deploy. In app settings, keep the app **private** and add viewer emails if others need access.

## Auto-upgrade

Community Cloud watches the linked branch. After you `git push`:

- Most code changes redeploy automatically within a short time.
- Dependency changes (`requirements.txt`) also trigger a rebuild.
- If the app looks stuck, use **Reboot** / **Rerun** in the Streamlit Cloud dashboard.

You do **not** need GitHub Actions for this basic flow.

## Docker / paid hosting (optional later)

| Option | Cost | When to use |
|--------|------|-------------|
| Streamlit Community Cloud | Free (1 private app) | Default for this project |
| Docker on your PC | Free | Local packaging only |
| Docker on AWS/Azure/Fly/Render | Often free tier then paid | You outgrow Streamlit limits or need custom infra |
| Snowflake / paid Streamlit | Paid | Enterprise controls |

A Dockerfile is **not** required for Streamlit Community Cloud.

## Checklist before first deploy

- [ ] Repo is **Private** on GitHub  
- [ ] `models/best_model_peak.joblib` and `models/best_model_curve.joblib` are committed  
- [ ] `requirements.txt` and `runtime.txt` are present  
- [ ] Local `npm run dev` Predict works  
- [ ] Streamlit app left **private**; viewers invited by email only  
