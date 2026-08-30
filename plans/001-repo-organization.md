# 001 — Repo organization, Cursor rules, and docs

## Status

done (2026-08-29)

## Goal

Match Kaneo/jira governance: `apps/` + feature `src/` modules + living docs + `.cursor/rules`, before implementing the ML pipeline.

## Delivered

- Folder layout (`apps/web`, `apps/docs`, `src/{dataset,train,predict,shared}`, `tests`, `plans`, `scripts`)
- Nine Cursor rules (always-apply + scoped)
- README, CONTRIBUTING, CLAUDE, setup.md, package.json, CHANGELOG
- Docs site content under `apps/docs/`
- Sample `data/raw/properties_rect.csv`

## Next plan

Implement dataset → train → predict → Streamlit end-to-end.
