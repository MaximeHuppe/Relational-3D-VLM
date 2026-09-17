"""Training logger: JSONL + stdout table, with optional W&B mirroring.

Mirrors the VoxWhisper ``TrainingLogger``: one record per epoch (or overfit
pass), written to ``metrics.jsonl``, printed as a compact row, and forwarded to
Weights & Biases when ``config["logging"]["backend"]`` is ``wandb``. Nested
metric dicts are flattened with ``/`` so train/val losses and Dice/IoU evolve
as separate W&B series.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.config import load_config

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.jsonl"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Recursively flatten nested metric dicts using ``/`` as separator."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_metrics(value, prefix=f"{name}/"))
        elif _is_number(value):
            flat[name] = float(value)
    return flat


def _jsonify_metric(value: Any) -> Any:
    """Round floats; recursively round dicts of floats for JSONL."""
    if isinstance(value, Mapping):
        return {str(key): _jsonify_metric(item) for key, item in value.items()}
    if not _is_number(value):
        return value
    return round(float(value), 6)


def _format_class_scores(class_scores: Mapping[str, float]) -> str:
    """Compact ``name=0.12`` list for stdout."""
    if not class_scores:
        return "[]"
    inner = " ".join(f"{name}={score:.3f}" for name, score in class_scores.items())
    return f"[{inner}]"


def _format_anchor_quality(train: Any, val: Any) -> str:
    """``anchors tr 0.367 (21% empty)  va 0.893 (2% empty)`` for stdout.

    Both splits on one row on purpose. With predicted anchors the training
    split is augmented and the validation split is not, so the two numbers can
    disagree badly; printing only the validation one is what let a Stage A that
    had never seen a rotated volume look healthy while supplying the training
    loop with mostly wrong anchors.
    """

    def one(tag: str, quality: Any) -> str:
        if not isinstance(quality, Mapping):
            return ""
        dice = quality.get("anchor_dice")
        if not _is_number(dice):
            return ""
        empty = quality.get("empty_anchor_fraction")
        suffix = f" ({float(empty):.0%} empty)" if _is_number(empty) else ""
        return f"{tag} {float(dice):.3f}{suffix}"

    shown = [text for text in (one("tr", train), one("va", val)) if text]
    return "anchors " + "  ".join(shown) if shown else ""


def logging_config(
    *,
    extra_tags: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """``configs/train.yaml`` logging block, with optional extra W&B tags.

    ``project`` overrides ``logging.wandb.project`` so a campaign can sit in
    its own W&B project without editing the YAML.
    """
    train = dict(config if config is not None else load_config("train"))
    log_cfg = dict(train.get("logging") or {})
    wandb_cfg = dict(log_cfg.get("wandb") or {})
    tags = [str(tag) for tag in (wandb_cfg.get("tags") or [])]
    for tag in extra_tags:
        if tag and tag not in tags:
            tags.append(str(tag))
    wandb_cfg["tags"] = tags
    if project:
        wandb_cfg["project"] = project
    log_cfg["wandb"] = wandb_cfg
    return log_cfg


def metrics_from_stage_a(
    *,
    train_loss: float,
    train_components: Mapping[str, float],
    train_dice: float | None = None,
    val_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train/val scalars Stage A logs each epoch."""
    metrics: dict[str, Any] = {
        "train_loss": float(train_loss),
        "train_components": dict(train_components),
    }
    if train_dice is not None:
        metrics["train_dice"] = float(train_dice)
    if not val_metrics:
        return metrics
    mean = val_metrics.get("mean") or {}
    metrics["val_loss"] = float(val_metrics.get("loss", 0.0))
    metrics["val_dice"] = float(mean.get("dice", 0.0))
    metrics["val_iou"] = float(mean.get("iou", 0.0))
    per_class = val_metrics.get("per_class") or {}
    if per_class:
        metrics["val_dice_classes"] = {
            name: float(entry["dice"]) for name, entry in per_class.items()
        }
        metrics["val_iou_classes"] = {
            name: float(entry["iou"]) for name, entry in per_class.items()
        }
    return metrics


def metrics_from_stage_b(
    *,
    train_loss: float,
    train_components: Mapping[str, float],
    train_dice: float | None = None,
    train_anchor_quality: Mapping[str, float] | None = None,
    val_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train/val scalars Stage B logs each epoch (or overfit pass).

    ``train_anchor_quality`` and the ``anchor_quality`` carried inside
    ``val_metrics`` are both reported, and they are not redundant: only the
    training split is augmented, so predicted anchors can be healthy on val and
    broken on train at the same time.
    """
    metrics: dict[str, Any] = {
        "train_loss": float(train_loss),
        "train_components": dict(train_components),
    }
    if train_dice is not None:
        metrics["train_dice"] = float(train_dice)
    if train_anchor_quality:
        metrics["train_anchor_quality"] = {
            key: float(value)
            for key, value in train_anchor_quality.items()
            if _is_number(value)
        }
    if not val_metrics:
        return metrics
    overall = val_metrics.get("overall") or {}
    metrics["val_loss"] = float(val_metrics.get("loss", 0.0))
    metrics["val_dice"] = float(overall.get("dice", 0.0))
    metrics["val_iou"] = float(overall.get("iou", 0.0))
    hausdorff = overall.get("hausdorff")
    if _is_number(hausdorff):
        metrics["val_hausdorff"] = float(hausdorff)
    strata = val_metrics.get("strata") or {}
    target_shape = strata.get("target_shape") or {}
    if target_shape:
        metrics["val_dice_classes"] = {
            name: float(entry["dice"]) for name, entry in target_shape.items()
        }
        metrics["val_iou_classes"] = {
            name: float(entry["iou"]) for name, entry in target_shape.items()
        }
    if strata:
        metrics["val_strata"] = {
            stratum: {
                key: {
                    metric: float(value)
                    for metric, value in entry.items()
                    if metric != "count" and _is_number(value)
                }
                for key, entry in buckets.items()
            }
            for stratum, buckets in strata.items()
        }
    quality = val_metrics.get("anchor_quality")
    if isinstance(quality, Mapping):
        metrics["anchor_quality"] = {
            key: float(value) for key, value in quality.items() if _is_number(value)
        }
    return metrics


class TrainingLogger:
    """Write per-epoch metrics to JSONL and print a compact stdout table row.

    Optionally mirrors metrics to Weights & Biases when ``log_cfg``
    (from ``config["logging"]``) has ``backend: wandb``.

    Parameters
    ----------
    run_dir      : directory where ``metrics.jsonl`` is written.
    total_epochs : total number of training epochs (used for the epoch column width).
    resume       : if ``True``, append to an existing metrics file.
    log_cfg      : ``config["logging"]`` dict.
    run_name     : human-readable run identifier forwarded to W&B.
    full_config  : entire config dict logged as W&B hyperparameters.
    verbose      : if ``False``, skip stdout (JSONL and W&B still run).
    """

    def __init__(
        self,
        run_dir: Path,
        total_epochs: int,
        resume: bool = False,
        *,
        log_cfg: Optional[Mapping[str, Any]] = None,
        run_name: Optional[str] = None,
        full_config: Optional[Mapping[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.total_epochs = total_epochs
        self.verbose = verbose
        self._start_time = time.monotonic()
        self._file = self._open(resume)

        log_cfg = dict(log_cfg or {})
        backend = str(log_cfg.get("backend", "none")).lower()

        self._wb_run = None
        if backend == "wandb":
            self._init_wandb(dict(log_cfg.get("wandb") or {}), run_name, full_config, resume)
        elif backend not in ("none", ""):
            logger.warning("Unknown logging.backend %r — using 'none'", backend)

    def _init_wandb(
        self,
        wb_cfg: dict[str, Any],
        run_name: Optional[str],
        full_config: Optional[Mapping[str, Any]],
        resume: bool,
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning(
                "W&B backend requested but 'wandb' is not installed. "
                "Run: pip install wandb"
            )
            return

        project = wb_cfg.get("project") or run_name or "training"
        tags = list(wb_cfg.get("tags") or [])
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name,
            "tags": tags,
            "config": dict(full_config) if full_config is not None else None,
            "resume": "allow" if resume else None,
            "dir": str(self.run_dir),
        }
        if wb_cfg.get("entity"):
            init_kwargs["entity"] = wb_cfg["entity"]
        if wb_cfg.get("mode"):
            init_kwargs["mode"] = wb_cfg["mode"]

        try:
            self._wb_run = wandb.init(**init_kwargs)
            url = getattr(self._wb_run, "url", None) if self._wb_run else None
            logger.info("W&B run initialised: %s", url or "?")
        except Exception as exc:
            logger.warning("W&B init failed (%s) — continuing without W&B", exc)
            self._wb_run = None

    def _open(self, resume: bool):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / METRICS_FILENAME
        if resume and path.exists():
            return open(path, "a", encoding="utf-8")  # noqa: SIM115
        return open(path, "w", encoding="utf-8")  # noqa: SIM115

    def log_epoch(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        lr: float,
        rank: Optional[int] = None,
    ) -> None:
        """Write one JSONL entry, print a stdout table row, and mirror to W&B."""
        elapsed = time.monotonic() - self._start_time

        record: dict[str, Any] = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "lr": lr,
        }
        record.update({key: _jsonify_metric(value) for key, value in metrics.items()})
        if rank is not None:
            record["rank"] = rank

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        self._print_row(epoch, metrics, lr, rank)
        self._log_wandb(epoch, metrics, lr)

    def _log_wandb(self, epoch: int, metrics: Mapping[str, Any], lr: float) -> None:
        if self._wb_run is None:
            return
        flat = _flatten_metrics(metrics)
        flat["lr"] = float(lr)
        try:
            self._wb_run.log(flat, step=epoch)
        except Exception as exc:
            logger.debug("W&B log failed: %s", exc)

    def close(self) -> None:
        """Flush and close the log file, and finish W&B if active."""
        self._file.flush()
        self._file.close()
        if self._wb_run is not None:
            try:
                self._wb_run.finish()
            except Exception:
                pass
            self._wb_run = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def best(self) -> dict[str, dict[str, float]]:
        """Scan the log file and return the best epoch for each scalar metric."""
        self._file.flush()
        path = Path(self._file.name)
        records = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            return {}

        metric_keys = [
            key
            for key in records[0]
            if key not in {"epoch", "elapsed_s", "lr", "rank"}
            and not isinstance(records[0][key], dict)
        ]
        result: dict[str, dict[str, float]] = {}
        for key in metric_keys:
            values = [(record["epoch"], record[key]) for record in records if key in record]
            if not values:
                continue
            minimise = key.endswith("_loss") or key.endswith("/loss")
            best_epoch, best_value = (
                min(values, key=lambda item: item[1]) if minimise else max(values, key=lambda item: item[1])
            )
            result[key] = {"epoch": best_epoch, "value": best_value}
        return result

    def print_summary(self) -> None:
        """Print a summary of the best value per metric seen so far."""
        if not self.verbose:
            return
        bests = self.best()
        if not bests:
            return
        width = max(len(key) for key in bests) + 2
        print("\n--- Training summary (best per metric) ---")
        for metric, info in bests.items():
            print(f"  {metric:<{width}} {info['value']:.4f}  (epoch {info['epoch']})")
        print()

    def _print_row(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        lr: float,
        rank: Optional[int],
    ) -> None:
        if not self.verbose:
            return
        epoch_width = len(str(self.total_epochs))
        parts = [f"Ep {epoch:{epoch_width}d}/{self.total_epochs}"]

        order = [
            "train_loss",
            "train_dice",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_hausdorff",
        ]
        seen: set[str] = set()
        for key in order:
            if key in metrics and _is_number(metrics[key]):
                parts.append(f"{key} {float(metrics[key]):.4f}")
                seen.add(key)
            class_key = f"{key}_classes"
            if class_key in metrics and isinstance(metrics[class_key], Mapping):
                parts.append(_format_class_scores(metrics[class_key]))
                seen.add(class_key)
        for key, value in metrics.items():
            if key in seen or isinstance(value, Mapping):
                continue
            if _is_number(value):
                parts.append(f"{key} {float(value):.4f}")

        anchors = _format_anchor_quality(
            metrics.get("train_anchor_quality"), metrics.get("anchor_quality")
        )
        if anchors:
            parts.append(anchors)

        parts.append(f"lr {lr:.2e}")
        if rank is not None:
            parts.append(f"[top-{rank}]")

        print("  ".join(parts))
