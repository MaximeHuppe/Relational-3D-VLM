"""JSONL + W&B training logger, matching the VoxWhisper contract."""

from __future__ import annotations

import json

from src.training.logger import (
    METRICS_FILENAME,
    TrainingLogger,
    _flatten_metrics,
    _format_anchor_quality,
    _jsonify_metric,
    logging_config,
    metrics_from_stage_a,
    metrics_from_stage_b,
)


def test_jsonify_nested_class_dice():
    payload = _jsonify_metric({"cube": 0.1234567, "sphere": 0.0})
    assert payload == {"cube": 0.123457, "sphere": 0.0}


def test_flatten_metrics_uses_slash_separators():
    flat = _flatten_metrics(
        {
            "train_loss": 1.25,
            "train_components": {"dice": 0.4, "bce": 0.8},
            "val_dice_classes": {"cube": 0.9, "sphere": 0.1},
        }
    )
    assert flat["train_loss"] == 1.25
    assert flat["train_components/dice"] == 0.4
    assert flat["val_dice_classes/cube"] == 0.9


def test_logger_writes_train_and_val_metrics(tmp_path, capsys):
    logger = TrainingLogger(tmp_path, total_epochs=10)
    logger.log_epoch(
        1,
        {
            "train_loss": 1.0,
            "train_dice": 0.2,
            "val_loss": 0.9,
            "val_dice": 0.14,
            "val_iou": 0.11,
            "val_dice_classes": {"cube": 0.84, "sphere": 0.0},
        },
        lr=1e-4,
    )
    logger.close()

    record = json.loads((tmp_path / METRICS_FILENAME).read_text().strip())
    assert record["epoch"] == 1
    assert record["train_loss"] == 1.0
    assert record["val_dice"] == 0.14
    assert record["val_dice_classes"]["cube"] == 0.84

    stdout = capsys.readouterr().out
    assert "train_loss 1.0000" in stdout
    assert "val_dice 0.1400" in stdout
    assert "cube=0.840" in stdout


def test_logger_is_quiet_when_verbose_is_false(tmp_path, capsys):
    logger = TrainingLogger(tmp_path, total_epochs=2, verbose=False)
    logger.log_epoch(1, {"train_loss": 1.0, "val_loss": 0.9}, lr=1e-3)
    logger.close()
    assert (tmp_path / METRICS_FILENAME).is_file()
    assert capsys.readouterr().out == ""


def test_logger_resume_appends(tmp_path):
    logger = TrainingLogger(tmp_path, total_epochs=5)
    logger.log_epoch(1, {"train_loss": 1.0}, lr=1e-4)
    logger.close()

    resumed = TrainingLogger(tmp_path, total_epochs=5, resume=True)
    resumed.log_epoch(2, {"train_loss": 0.5}, lr=1e-4)
    resumed.close()

    lines = (tmp_path / METRICS_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 1
    assert json.loads(lines[1])["epoch"] == 2


def test_logger_mirrors_flattened_metrics_to_wandb(tmp_path):
    logged: list[tuple[dict, int]] = []

    class FakeRun:
        def log(self, data, step=None):
            logged.append((dict(data), step))

        def finish(self):
            pass

    logger = TrainingLogger(tmp_path, total_epochs=3, verbose=False)
    logger._wb_run = FakeRun()
    logger.log_epoch(
        2,
        {
            "train_loss": 0.8,
            "train_dice": 0.4,
            "val_loss": 0.7,
            "val_dice": 0.5,
            "train_components": {"dice": 0.3, "bce": 0.5},
            "val_dice_classes": {"cuboid": 0.6},
        },
        lr=1e-3,
    )
    logger.close()

    data, step = logged[0]
    assert step == 2
    assert data["train_loss"] == 0.8
    assert data["train_dice"] == 0.4
    assert data["val_loss"] == 0.7
    assert data["val_dice"] == 0.5
    assert data["train_components/dice"] == 0.3
    assert data["val_dice_classes/cuboid"] == 0.6
    assert data["lr"] == 1e-3


def test_metrics_from_stage_a_include_train_and_val():
    metrics = metrics_from_stage_a(
        train_loss=1.2,
        train_components={"dice@16": 0.4, "bce@16": 0.8, "total": 1.2},
        train_dice=0.3,
        val_metrics={
            "loss": 0.9,
            "mean": {"dice": 0.55, "iou": 0.4},
            "per_class": {"cube": {"dice": 0.7, "iou": 0.5, "count": 2}},
        },
    )
    assert metrics["train_loss"] == 1.2
    assert metrics["train_dice"] == 0.3
    assert metrics["val_loss"] == 0.9
    assert metrics["val_dice"] == 0.55
    assert metrics["val_iou"] == 0.4
    assert metrics["val_dice_classes"]["cube"] == 0.7
    assert "table" not in metrics


def test_metrics_from_stage_b_include_strata():
    metrics = metrics_from_stage_b(
        train_loss=0.5,
        train_components={"dice": 0.2, "bce": 0.3},
        train_dice=0.8,
        val_metrics={
            "loss": 0.4,
            "overall": {"dice": 0.75, "iou": 0.6, "hausdorff": 1.5, "count": 4},
            "strata": {
                "target_shape": {"cuboid": {"dice": 0.7, "iou": 0.55, "count": 2}},
                "direction": {"lateral": {"dice": 0.8, "iou": 0.65, "count": 4}},
            },
        },
    )
    assert metrics["train_dice"] == 0.8
    assert metrics["val_dice"] == 0.75
    assert metrics["val_hausdorff"] == 1.5
    assert metrics["val_dice_classes"]["cuboid"] == 0.7
    assert metrics["val_strata"]["direction"]["lateral"]["dice"] == 0.8
    flat = _flatten_metrics(metrics)
    assert flat["val_strata/target_shape/cuboid/dice"] == 0.7


def test_logging_config_appends_tags_from_train_yaml():
    cfg = logging_config(extra_tags=["stage_a", "smoke"])
    assert cfg["backend"] == "wandb"
    assert cfg["wandb"]["project"] == "relational-3d-vlm"
    assert "stage_a" in cfg["wandb"]["tags"]
    assert "smoke" in cfg["wandb"]["tags"]
    assert "relational-3d-vlm" in cfg["wandb"]["tags"]


def test_logging_config_can_override_the_wandb_project():
    cfg = logging_config(project="relational-3d-vlm-phase-a")
    assert cfg["wandb"]["project"] == "relational-3d-vlm-phase-a"
    default = logging_config()
    assert default["wandb"]["project"] == "relational-3d-vlm"


def test_logging_config_cli_tags_replace_yaml_leftovers():
    cfg = logging_config(tags=["EX-1", "phase-a", "dataset-realistic"])
    assert cfg["wandb"]["tags"] == ["EX-1", "phase-a", "dataset-realistic"]
    assert "dataset-custom" not in cfg["wandb"]["tags"]


def test_the_epoch_row_shows_anchor_quality_for_both_splits():
    """The gap between the two is the signal, so neither may be dropped."""
    row = _format_anchor_quality(
        {"anchor_dice": 0.3671, "empty_anchor_fraction": 0.2149},
        {"anchor_dice": 0.8929, "empty_anchor_fraction": 0.0167},
    )
    assert row == "anchors tr 0.367 (21% empty)  va 0.893 (2% empty)"
    # Oracle anchors report nothing at all rather than a misleading 1.000.
    assert _format_anchor_quality(None, None) == ""
    assert _format_anchor_quality(None, {"anchor_dice": 0.5}) == "anchors va 0.500"


def test_stage_b_metrics_carry_the_training_split_anchor_quality():
    metrics = metrics_from_stage_b(
        train_loss=1.0,
        train_components={},
        train_dice=0.1,
        train_anchor_quality={"anchor_dice": 0.367, "channels": 8400.0},
        val_metrics={"overall": {"dice": 0.05, "iou": 0.03}, "anchor_quality": {"anchor_dice": 0.893}},
    )
    assert metrics["train_anchor_quality"] == {"anchor_dice": 0.367, "channels": 8400.0}
    assert metrics["anchor_quality"] == {"anchor_dice": 0.893}
    # Flattened for W&B, the two stay distinguishable.
    flat = _flatten_metrics(metrics)
    assert flat["train_anchor_quality/anchor_dice"] == 0.367
    assert flat["anchor_quality/anchor_dice"] == 0.893
