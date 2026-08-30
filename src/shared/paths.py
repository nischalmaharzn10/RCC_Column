"""Repo path helpers."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, ROOT / default))


def data_raw() -> Path:
    return env_path("RCC_DATA_RAW", "data/raw")


def data_processed() -> Path:
    return env_path("RCC_DATA_PROCESSED", "data/processed")


def models_dir() -> Path:
    return env_path("RCC_MODELS_DIR", "models")


def outputs_dir() -> Path:
    return env_path("RCC_OUTPUTS_DIR", "outputs")


def samples_dir() -> Path:
    return ROOT / "samples"
