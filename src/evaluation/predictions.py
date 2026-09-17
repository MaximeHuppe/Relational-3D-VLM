"""Test-set inference: run a chosen Stage A / Stage B pair and keep the masks.

``scripts/train_relational_model.py --eval-only`` scores a checkpoint and throws
the masks away. This module is the other half: it runs the same forward pass and
writes every prediction to disk next to the target it was scored against, so a
result can be looked at rather than only read off a table.

One prediction folder is one ``(Stage A checkpoint, Stage B checkpoint, split)``
triple, and it is self-describing:

.. code-block:: text

    <output_dir>/
      run_metadata.json          which checkpoints, under which names, on what
      metrics.json               aggregate + stratified Dice / IoU / Hausdorff
      metrics.txt                the same report as the console table
      predictions.jsonl          one row per example: prompt, scores, file paths
      predictions/<example_id>/
        prediction_mask.nii.gz   Stage B's binary mask at the report threshold
        target_mask.nii.gz       the ground-truth target it is scored against
        prediction_probability.nii.gz   (--save-probabilities)
        anchor_<slot>_<shape>.nii.gz    (--save-anchors) the channels Stage B saw

The two model *names* are part of the contract, not a nicety: ``runs/`` holds a
dozen Stage A and Stage B runs whose checkpoints are all called ``best.pt``, so a
folder of masks that only records ``best.pt`` is unattributable a week later.
The name defaults to the run directory the checkpoint sits in - the same string
the trainer used as its W&B run name - and both the name and the full path are
written into ``run_metadata.json``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.config import PROJECT_ROOT, load_all_configs
from src.data.dataset import stage_b_model_inputs
from src.data.nifti_io import save_nifti
from src.data.schema import anchor_mask_filename
from src.evaluation.metrics import StratifiedMetrics, format_stratified_table
from src.provenance import environment_metadata, version_metadata

#: Default parent of a prediction folder, relative to the project root.
DEFAULT_OUTPUT_ROOT = "predictions"

PREDICTION_MASK_FILENAME = "prediction_mask.nii.gz"
PREDICTION_PROBABILITY_FILENAME = "prediction_probability.nii.gz"
TARGET_MASK_FILENAME = "target_mask.nii.gz"
VOLUME_SUBDIR = "predictions"

RUN_METADATA_FILENAME = "run_metadata.json"
METRICS_FILENAME = "metrics.json"
METRICS_TABLE_FILENAME = "metrics.txt"
ROWS_FILENAME = "predictions.jsonl"


class PredictionError(RuntimeError):
    """Raised when a prediction run cannot be set up or written."""


# ---------------------------------------------------------------------------
# Which weights ran
# ---------------------------------------------------------------------------
def _relative_to_project(path: Path) -> str:
    """``runs/stage_a/best.pt`` when inside the repo, else the absolute path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def file_sha256(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a checkpoint, so "which weights" is answerable."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_model_name(checkpoint: Path | str) -> str:
    """The run directory a checkpoint sits in - ``runs/stage_a/best.pt`` -> ``stage_a``.

    That directory name is what ``train_*.py`` passes to the logger as the run
    name, so it is the string a W&B run, a history file and a checkpoint already
    share.
    """
    path = Path(checkpoint).resolve()
    parent = path.parent.name
    return parent or path.stem


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    """The provenance block of a checkpoint, from its sidecar or the payload.

    ``save_checkpoint`` writes a readable ``best.json`` beside ``best.pt``;
    reading that avoids deserialising a few hundred megabytes of weights twice.
    """
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, Mapping):
            return dict(data)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    return dict(metadata) if isinstance(metadata, Mapping) else {}


@dataclass(frozen=True)
class ModelIdentity:
    """Everything a prediction folder records about one of the two models."""

    role: str                       # "stage_a" or "stage_b"
    name: str                       # run name, e.g. shapeSeg_dataset_custom
    checkpoint: Path                # the path as it was given on the command line
    training: Mapping[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        resolved = Path(self.checkpoint).resolve()
        stat = resolved.stat() if resolved.is_file() else None
        return {
            "role": self.role,
            "name": self.name,
            "checkpoint": str(self.checkpoint),
            "checkpoint_absolute": str(resolved),
            "checkpoint_relative": _relative_to_project(resolved),
            "run_dir": _relative_to_project(resolved.parent),
            "checkpoint_file": resolved.name,
            "sha256": self.sha256,
            "size_bytes": stat.st_size if stat else None,
            "modified": (
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                if stat
                else None
            ),
            "training": dict(self.training),
        }


def describe_checkpoint(
    checkpoint: Path | str,
    *,
    role: str,
    name: str | None = None,
    checksum: bool = True,
) -> ModelIdentity:
    """Read a checkpoint's provenance without rebuilding the model.

    Args:
        checkpoint: the ``.pt`` file that will be loaded.
        role: ``stage_a`` or ``stage_b``.
        name: override the run name; defaults to the checkpoint's directory.
        checksum: hash the weights file (about a second per checkpoint).
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise PredictionError(f"no {role} checkpoint at {path}")
    metadata = _checkpoint_metadata(path)
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), Mapping) else {}
    extra = metadata.get("extra") if isinstance(metadata.get("extra"), Mapping) else {}
    environment = (
        metadata.get("environment") if isinstance(metadata.get("environment"), Mapping) else {}
    )
    training = {
        "stage": metadata.get("stage"),
        "epoch": metadata.get("epoch"),
        "selection_metric": metadata.get("selection_metric"),
        "selection_score": extra.get("best_dice"),
        "best_epoch": extra.get("best_epoch"),
        "seed": metadata.get("seed"),
        "settings": dict(settings),
        "versions": dict(metadata.get("versions") or {}),
        "git_revision": environment.get("git_revision"),
    }
    return ModelIdentity(
        role=role,
        name=name or default_model_name(path),
        checkpoint=path,
        training=training,
        sha256=file_sha256(path) if checksum else None,
    )


def default_run_name(
    stage_b: ModelIdentity, stage_a: ModelIdentity | None, split: str, anchor_source: str
) -> str:
    """``<stage_b name>__anchors-<source|stage_a name>__<split>``."""
    anchors = anchor_source if stage_a is None else f"{anchor_source}-{stage_a.name}"
    return f"{stage_b.name}__anchors-{anchors}__{split}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _to_numpy_mask(volume: Tensor) -> np.ndarray:
    return volume.detach().cpu().numpy().astype(np.uint8, copy=False)


def _write_example_volumes(
    directory: Path,
    *,
    prediction: Tensor,
    target: Tensor | None,
    spacing: Sequence[float],
    probability: Tensor | None = None,
    anchors: Tensor | None = None,
    anchor_shape_names: Sequence[str] | None = None,
) -> dict[str, str]:
    """Write one example's volumes; returns ``{kind: filename}``."""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    save_nifti(
        directory / PREDICTION_MASK_FILENAME, _to_numpy_mask(prediction), spacing, dtype=np.uint8
    )
    written["prediction_mask"] = PREDICTION_MASK_FILENAME
    if target is not None:
        save_nifti(
            directory / TARGET_MASK_FILENAME, _to_numpy_mask(target), spacing, dtype=np.uint8
        )
        written["target_mask"] = TARGET_MASK_FILENAME
    if probability is not None:
        save_nifti(
            directory / PREDICTION_PROBABILITY_FILENAME,
            probability.detach().cpu().numpy().astype(np.float32, copy=False),
            spacing,
            dtype=np.float32,
        )
        written["prediction_probability"] = PREDICTION_PROBABILITY_FILENAME
    if anchors is not None:
        names = list(anchor_shape_names or [f"slot{index}" for index in range(anchors.shape[0])])
        for slot in range(anchors.shape[0]):
            filename = anchor_mask_filename(slot, names[slot])
            save_nifti(directory / filename, _to_numpy_mask(anchors[slot]), spacing, dtype=np.uint8)
            written[f"anchor_{slot}"] = filename
    return written


@torch.no_grad()
def run_prediction(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path | str,
    anchor_provider: Any,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    hausdorff_percentile: float | None = None,
    save_volumes: bool = True,
    save_targets: bool = True,
    save_probabilities: bool = False,
    save_anchors: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Segment every example of ``loader``, score it and write the masks.

    The forward pass is exactly the one the trainer evaluates with - anchors from
    the provider, then :func:`~src.data.dataset.stage_b_model_inputs`, so the
    target mask cannot reach the network - and the numbers therefore match
    ``--eval-only`` on the same checkpoint and split.

    Returns the stratified report, with a ``rows`` entry holding one record per
    example (prompt, per-example Dice / IoU / Hausdorff and the files written).
    """
    output_dir = Path(output_dir)
    volume_dir = output_dir / VOLUME_SUBDIR
    if save_volumes:
        volume_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device).eval()
    reset = getattr(anchor_provider, "reset", None)
    if callable(reset):
        reset()

    metrics = StratifiedMetrics()
    rows: list[dict[str, Any]] = []

    for batch in loader:
        moved = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        anchor_masks = anchor_provider(moved).to(device)
        logits = model(**stage_b_model_inputs(moved, anchor_masks)).logits.float()

        probabilities = torch.sigmoid(logits).cpu()
        target = batch["target_mask"].float().cpu()
        prediction = (probabilities >= threshold).to(torch.uint8)

        metrics.update(
            probabilities,
            target,
            target_shapes=batch["target_shape_name"],
            anchor_shapes=batch["anchor_shape_names"],
            directions=batch["directions"],
            threshold=threshold,
            from_logits=False,
            with_hausdorff=True,
            percentile=hausdorff_percentile,
            spacing=spacing,
        )
        # The per-example rows are the scores the report just accumulated, read
        # back off the accumulator rather than recomputed: one surface
        # extraction and one pairwise distance per example is enough, and the
        # rows and the report can then never disagree.
        size = probabilities.shape[0]
        dice = metrics.overall.dice[-size:]
        iou = metrics.overall.iou[-size:]
        distances = metrics.overall.hausdorff[-size:]
        anchors_cpu = anchor_masks.detach().cpu()

        for index, example_id in enumerate(batch["example_id"]):
            row: dict[str, Any] = {
                "example_id": example_id,
                "scene_id": batch["scene_id"][index],
                "prompt": batch["prompt"][index],
                "target_shape_name": batch["target_shape_name"][index],
                "anchor_shape_names": list(batch["anchor_shape_names"][index]),
                "directions": list(batch["directions"][index]),
                "dice": float(dice[index]),
                "iou": float(iou[index]),
                "hausdorff": float(distances[index]),
                "predicted_voxels": int(prediction[index, 0].sum()),
                "target_voxels": int(target[index, 0].sum()),
            }
            if save_volumes:
                written = _write_example_volumes(
                    volume_dir / example_id,
                    prediction=prediction[index, 0],
                    target=target[index, 0] if save_targets else None,
                    spacing=spacing,
                    probability=probabilities[index, 0] if save_probabilities else None,
                    anchors=anchors_cpu[index] if save_anchors else None,
                    anchor_shape_names=batch["anchor_shape_names"][index],
                )
                row["files"] = {
                    kind: f"{VOLUME_SUBDIR}/{example_id}/{filename}"
                    for kind, filename in written.items()
                }
            rows.append(row)
            if verbose:
                print(
                    f"  {example_id:<28} dice {row['dice']:.4f}  iou {row['iou']:.4f}  "
                    f"hd {row['hausdorff']:.2f}"
                )

    if not rows:
        raise PredictionError("the loader yielded no examples")

    report = metrics.summary()
    report["anchor_source"] = getattr(anchor_provider, "source", "oracle")
    quality = getattr(anchor_provider, "anchor_quality", None)
    if callable(quality):
        measured = quality()
        if measured:
            report["anchor_quality"] = measured
    report["table"] = format_stratified_table(report)
    report["rows"] = rows
    return report


# ---------------------------------------------------------------------------
# What lands in the folder
# ---------------------------------------------------------------------------
def write_rows(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """One JSON object per example, in loader order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def write_metrics(path: Path | str, report: Mapping[str, Any]) -> Path:
    """The stratified report, without the console table or the per-example rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value for key, value in report.items() if key not in {"table", "rows"}
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_run_metadata(
    output_dir: Path | str,
    *,
    stage_b: ModelIdentity,
    stage_a: ModelIdentity | None,
    split: str,
    data_root: Path | str,
    anchor_source: str,
    report: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    include_configs: bool = True,
) -> Path:
    """Write ``run_metadata.json``: which weights, under which names, on what.

    Mirrors the provenance block every other artefact in this project carries
    (generated corpora, checkpoints, evaluation reports), with the two model
    identities as the part that makes a folder of masks attributable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = report.get("rows") or []
    models: dict[str, Any] = {"stage_b": stage_b.to_dict()}
    models["stage_a"] = stage_a.to_dict() if stage_a is not None else None
    payload: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": split,
        "data_root": str(Path(data_root).resolve()),
        "anchor_source": anchor_source,
        "examples": len(rows),
        "models": models,
        "model_names": {
            "stage_a": stage_a.name if stage_a is not None else None,
            "stage_b": stage_b.name,
        },
        "checkpoints": {
            "stage_a": str(stage_a.checkpoint) if stage_a is not None else None,
            "stage_b": str(stage_b.checkpoint),
        },
        "settings": dict(settings or {}),
        "metrics": {
            "overall": report.get("overall", {}),
            **(
                {"anchor_quality": report["anchor_quality"]}
                if "anchor_quality" in report
                else {}
            ),
        },
        "outputs": {
            "metrics": METRICS_FILENAME,
            "metrics_table": METRICS_TABLE_FILENAME,
            "rows": ROWS_FILENAME,
            "volumes": VOLUME_SUBDIR,
        },
        "command": list(command) if command is not None else None,
        "versions": version_metadata(),
        "environment": {**environment_metadata(), "hostname": platform.node()},
    }
    if include_configs:
        payload["configs"] = load_all_configs()
    path = output_dir / RUN_METADATA_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_prediction_folder(
    output_dir: Path | str,
    *,
    report: Mapping[str, Any],
    stage_b: ModelIdentity,
    stage_a: ModelIdentity | None,
    split: str,
    data_root: Path | str,
    anchor_source: str,
    settings: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write the four bookkeeping files of a prediction folder."""
    output_dir = Path(output_dir)
    written = {
        "metrics": write_metrics(output_dir / METRICS_FILENAME, report),
        "rows": write_rows(output_dir / ROWS_FILENAME, report.get("rows") or []),
    }
    table = str(report.get("table") or format_stratified_table(report))
    (output_dir / METRICS_TABLE_FILENAME).write_text(table + "\n", encoding="utf-8")
    written["metrics_table"] = output_dir / METRICS_TABLE_FILENAME
    written["run_metadata"] = write_run_metadata(
        output_dir,
        stage_b=stage_b,
        stage_a=stage_a,
        split=split,
        data_root=data_root,
        anchor_source=anchor_source,
        report=report,
        settings=settings,
        command=command,
    )
    return written
