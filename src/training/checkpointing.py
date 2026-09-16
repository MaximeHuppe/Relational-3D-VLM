"""Checkpoint save/load with reproducibility metadata.

Every checkpoint records: the config snapshot, random seeds, generator version,
direction-rule version, vocabulary version, schema version, git revision,
hardware and the metric it was selected on. A checkpoint that cannot say how it
was produced is not useful six months later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from src.config import load_all_configs
from src.provenance import environment_metadata, version_metadata


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def checkpoint_metadata(
    *,
    stage: str,
    epoch: int,
    metrics: Mapping[str, Any],
    settings: Any,
    seed: int,
    selection_metric: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance block stored beside the weights."""
    return {
        "stage": stage,
        "epoch": int(epoch),
        "metrics": _jsonable(metrics),
        "selection_metric": selection_metric,
        "seed": int(seed),
        "settings": _jsonable(settings),
        "versions": version_metadata(),
        "environment": environment_metadata(),
        "configs": load_all_configs(),
        **({"extra": _jsonable(extra)} if extra else {}),
    }


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    model_config: Any | None = None,
) -> Path:
    """Write weights plus metadata, and a readable ``.json`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    if model_config is not None:
        payload["model_config"] = _jsonable(model_config)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(_jsonable(payload["metadata"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring ``model`` in place."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload
