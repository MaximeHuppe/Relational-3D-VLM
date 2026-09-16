"""Configuration loading.

The YAML files in ``configs/`` are the single source of truth for the shape
vocabulary, generator contract, split assignment, model shapes and training
hyperparameters. Nothing in ``src/`` may hard-code a value that lives there.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

CONFIG_NAMES = ("shapes", "generator", "split", "model", "train")


def config_path(name: str) -> Path:
    """Absolute path of a named config file (``"shapes"`` -> configs/shapes.yaml)."""
    if name not in CONFIG_NAMES:
        raise KeyError(f"unknown config {name!r}; expected one of {CONFIG_NAMES}")
    return CONFIG_DIR / f"{name}.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping, got {type(data)}")
    return data


@lru_cache(maxsize=None)
def _load_cached(name: str) -> dict[str, Any]:
    return load_yaml(config_path(name))


def load_config(name: str) -> dict[str, Any]:
    """Load and cache a named config. The returned mapping must not be mutated."""
    return _load_cached(name)


def load_all_configs() -> dict[str, dict[str, Any]]:
    """Load every config, for run-metadata snapshots."""
    return {name: load_config(name) for name in CONFIG_NAMES}
