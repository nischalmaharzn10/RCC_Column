# Contributing to RCC Column ML

Thanks for contributing. Keep the repo organized the same way from day one — feature folders, docs, and Cursor rules stay in sync.

## Table of contents

- [What you need](#what-you-need)
- [Setup](#setup)
- [Branching and commits](#branching-and-commits)
- [Project structure](#project-structure)
- [Adding a feature](#adding-a-feature)
- [Documentation and rules](#documentation-and-rules)
- [Code style](#code-style)

## What you need

- Python 3.10+
- Git
- (Optional) ANSYS exports using the same CSV schema as PEER properties

## Setup

See [setup.md](setup.md) and [apps/docs/guides/getting-started.md](apps/docs/guides/getting-started.md).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
npm install   # optional; enables npm run dev / train / …
```

**Start the UI:** `npm run dev` (or `streamlit run apps/web/streamlit_app.py`).

## Branching and commits

**Branches:** `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `hotfix/`

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/)

```
feat: add backbone envelope extraction
fix: group split by specimen_id in curve mode
docs: changelog platform-guide for Predict tab
```

## Project structure

| Path | Own |
|------|-----|
| `src/dataset/` | Data ingest & audit |
| `src/train/` | Training & metrics |
| `src/predict/` | Inference API |
| `src/shared/` | Shared schema/utils (only when used by 2+ features) |
| `apps/web/` | Streamlit UI (thin) |
| `apps/docs/` | Living documentation |
| `tests/{feature}/` | Tests mirroring `src` |
| `.cursor/rules/` | Cursor agent rules |

Details: `.cursor/rules/feature-folder-structure.mdc`

## Adding a feature

1. Put logic in `src/{feature}/service.py` (create the package if new).
2. Add thin UI in `apps/web/components/` and wire tabs in `streamlit_app.py`.
3. Add tests under `tests/{feature}/`.
4. Update **`apps/docs/core/platform-guide.md`** Changelog (required).
5. Update matching **`.cursor/rules/*.mdc`** (schema, ML, UI, or feature-folder-structure).
6. If env vars or start scripts change → `setup.md` + `.env.example` + `package.json` as needed.

## Documentation and rules

- Canonical domain doc: `apps/docs/core/platform-guide.md`
- Agent mirror: `.cursor/rules/platform-guide.mdc`
- How to add rules: `.cursor/rules/cursor-rules.mdc`

Do not invent a parallel docs tree at repo root beyond README / CONTRIBUTING / CLAUDE / setup.md / CHANGELOG.

## Code style

- Ruff format + lint (`pyproject.toml`)
- Double quotes, line length 100
- No training logic inside Streamlit modules
- SI units only; specimen-grouped splits only

## Need help?

Check `CLAUDE.md` and `.cursor/rules/project-overview.mdc` for architecture orientation.
