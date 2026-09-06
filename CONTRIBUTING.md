# Contributing to RCC Column ML

Thanks for contributing. This guide covers setup, branching, structure, and documentation expectations.

## Table of contents

- [What you need](#what-you-need)
- [Setup](#setup)
- [Branching and commits](#branching-and-commits)
- [Project structure](#project-structure)
- [Adding a feature](#adding-a-feature)
- [Documentation](#documentation)
- [Code style](#code-style)
- [Checks before a PR](#checks-before-a-pr)

## What you need

- Python 3.10+
- Git
- Node.js (optional; enables `npm run` scripts)

## Setup

See [setup.md](setup.md) and [apps/docs/guides/getting-started.md](apps/docs/guides/getting-started.md).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
npm install
npm run build:data
npm run train
npm run dev                  # http://localhost:8501
```

## Branching and commits

**Branches:** `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `hotfix/`

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/)

```
feat: add backbone envelope extraction
fix: group split by specimen_id in curve mode
docs: changelog platform-guide for Predict tab
```

## Project structure

| Path | Owns |
|------|------|
| `src/dataset/` | Data ingest and audit |
| `src/train/` | Training and metrics |
| `src/predict/` | Inference API |
| `src/shared/` | Shared schema/utils (when used by 2+ features) |
| `apps/web/` | Streamlit UI (thin presentation only) |
| `apps/docs/` | Product and pipeline documentation |
| `tests/{feature}/` | Tests mirroring `src/{feature}/` |
| `samples/` | PEER source properties and curves |

Keep business logic in `src/`. Streamlit pages and components should call `src.*.service` APIs only.

## Adding a feature

1. Put logic in `src/{feature}/service.py` (create the package if needed).
2. Add thin UI under `apps/web/components/` and wire it from `streamlit_app.py` when it belongs on Predict.
3. Add tests under `tests/{feature}/`.
4. Update **`apps/docs/core/platform-guide.md`** Changelog for any user-facing behavior change.
5. If env vars or start scripts change, update `setup.md`, `.env.example`, and `package.json` as needed.

## Documentation

- Domain and behavior: `apps/docs/core/platform-guide.md`
- Schema: `apps/docs/guides/data-schema.md`
- Training: `apps/docs/guides/ml-pipeline.md`
- Local setup: `setup.md`

Keep root docs limited to README, CONTRIBUTING, setup.md, and CHANGELOG unless there is a clear reason to add another.

## Code style

- Ruff format and lint (`pyproject.toml`) — double quotes, line length 100
- Prefer explicit types on public function signatures
- SI units only (mm, kN, MPa)
- Always split train/test by `specimen` / `specimen_id` (no leakage)
- No training or fitting logic inside Streamlit modules

## Checks before a PR

```bash
npm run lint
npm run test
```

Confirm the Predict UI still runs (`npm run dev`) if you touched `apps/web/` or `src/predict/`.
