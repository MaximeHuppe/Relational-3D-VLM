"""The Phase 4 prediction folder: masks on disk, and which weights wrote them.

The point of these tests is attribution and round-tripping, not accuracy: an
untrained checkpoint predicts nonsense, but the folder still has to say which
Stage A and Stage B checkpoints ran, under which names, and the mask it writes
has to be the mask that was scored - reloadable from disk, next to its target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate import main as evaluate_main
from src.data.dataset import DatasetError, ExampleDataset
from src.data.nifti_io import load_nifti
from src.evaluation.predictions import (
    PredictionError,
    default_model_name,
    default_run_name,
    describe_checkpoint,
)
from src.models.relational_vlm import build_relational_vlm
from src.models.shape_segmenter import build_shape_segmenter
from src.training.checkpointing import checkpoint_metadata, save_checkpoint

SMOKE_ROOT = "data/smoke"
SPLIT = "test"


@pytest.fixture(scope="module")
def smoke_dataset() -> ExampleDataset:
    try:
        return ExampleDataset(SMOKE_ROOT, SPLIT, limit=2)
    except DatasetError as error:  # pragma: no cover - corpus not generated
        pytest.skip(f"smoke corpus unavailable: {error}")


def _write_checkpoint(directory: Path, model: torch.nn.Module, stage: str) -> Path:
    """A checkpoint that looks like the trainer's, in a named run directory."""
    directory.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(
        directory / "best.pt",
        model=model,
        metadata=checkpoint_metadata(
            stage=stage,
            epoch=3,
            metrics={"overall": {"dice": 0.5}},
            settings={"model_profile": "smoke"},
            seed=7,
            selection_metric="val_dice",
            extra={"best_dice": 0.5, "best_epoch": 3},
        ),
        model_config=getattr(model, "config", None),
    )


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory, smoke_dataset) -> tuple[Path, Path]:
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("runs")
    resolution = max(smoke_dataset.volume_shape)
    stage_a = _write_checkpoint(
        root / "shapeSeg_unit_test",
        build_shape_segmenter("smoke", input_resolution=resolution),
        "stage_a",
    )
    stage_b = _write_checkpoint(
        root / "relational_unit_test",
        build_relational_vlm(
            "smoke", input_resolution=resolution, spacing=smoke_dataset.spacing
        ),
        "stage_b_oracle",
    )
    return stage_a, stage_b


def _evaluate(output: Path, checkpoints: tuple[Path, Path], *extra: str) -> int:
    stage_a, stage_b = checkpoints
    return evaluate_main(
        [
            "--stage-b-checkpoint", str(stage_b),
            "--stage-a-checkpoint", str(stage_a),
            "--smoke",
            "--split", SPLIT,
            "--limit-examples", "2",
            "--batch-size", "1",
            "--device", "cpu",
            "--no-checksum",
            "--output", str(output),
            "--quiet",
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def prediction_folder(tmp_path_factory, checkpoints) -> Path:
    output = tmp_path_factory.mktemp("predictions") / "run"
    assert _evaluate(output, checkpoints) == 0
    return output


# ---------------------------------------------------------------------------
# Which weights ran
# ---------------------------------------------------------------------------
def test_the_model_name_defaults_to_the_run_directory(checkpoints):
    stage_a, stage_b = checkpoints
    assert default_model_name(stage_a) == "shapeSeg_unit_test"
    identity = describe_checkpoint(stage_b, role="stage_b", checksum=False)
    assert identity.name == "relational_unit_test"
    assert identity.training["stage"] == "stage_b_oracle"
    assert identity.training["selection_score"] == 0.5
    named = describe_checkpoint(stage_b, role="stage_b", name="my-run", checksum=False)
    assert named.name == "my-run"
    assert named.to_dict()["checkpoint_file"] == "best.pt"


def test_a_missing_checkpoint_is_reported_before_anything_runs(tmp_path):
    with pytest.raises(PredictionError):
        describe_checkpoint(tmp_path / "absent.pt", role="stage_a")


def test_the_folder_name_carries_both_runs_and_the_split(checkpoints):
    stage_a, stage_b = checkpoints
    name = default_run_name(
        describe_checkpoint(stage_b, role="stage_b", checksum=False),
        describe_checkpoint(stage_a, role="stage_a", checksum=False),
        "test",
        "predicted",
    )
    assert name == "relational_unit_test__anchors-predicted-shapeSeg_unit_test__test"


def test_run_metadata_records_both_checkpoints_and_both_names(prediction_folder, checkpoints):
    stage_a, stage_b = checkpoints
    metadata = json.loads((prediction_folder / "run_metadata.json").read_text())
    assert metadata["model_names"] == {
        "stage_a": "shapeSeg_unit_test",
        "stage_b": "relational_unit_test",
    }
    assert metadata["checkpoints"] == {"stage_a": str(stage_a), "stage_b": str(stage_b)}
    assert metadata["models"]["stage_a"]["checkpoint_absolute"] == str(stage_a.resolve())
    assert metadata["models"]["stage_b"]["run_dir"].endswith("relational_unit_test")
    assert metadata["split"] == SPLIT
    assert metadata["anchor_source"] == "predicted"
    assert metadata["examples"] == 2
    assert metadata["environment"]["git_revision"]
    assert metadata["configs"]["model"]["stage_b"]


# ---------------------------------------------------------------------------
# What is on disk
# ---------------------------------------------------------------------------
def test_every_example_keeps_its_prediction_next_to_its_target(
    prediction_folder, smoke_dataset
):
    rows = [
        json.loads(line)
        for line in (prediction_folder / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    targets = {record.example_id: index for index, record in enumerate(smoke_dataset.records)}

    for row in rows:
        assert row["prompt"] and row["target_shape_name"]
        assert set(row["files"]) == {"prediction_mask", "target_mask"}
        prediction = load_nifti(prediction_folder / row["files"]["prediction_mask"], dtype=np.uint8)
        target = load_nifti(prediction_folder / row["files"]["target_mask"], dtype=np.uint8)
        assert prediction.shape == smoke_dataset.volume_shape == target.shape

        # The target on disk is the corpus target, not a re-derived approximation.
        expected = smoke_dataset[targets[row["example_id"]]]["target_mask"][0].numpy()
        assert np.array_equal(target, expected.astype(np.uint8))
        assert int(target.sum()) == row["target_voxels"]
        assert int(prediction.sum()) == row["predicted_voxels"]

        # The mask scored in the report is the mask that was written.
        denominator = prediction.sum() + target.sum()
        dice = 2.0 * np.logical_and(prediction, target).sum() / denominator if denominator else 1.0
        assert dice == pytest.approx(row["dice"], abs=1e-5)


def test_the_report_is_stratified_and_names_the_anchor_source(prediction_folder):
    report = json.loads((prediction_folder / "metrics.json").read_text())
    assert report["anchor_source"] == "predicted"
    assert report["overall"]["count"] == 2.0
    assert "hausdorff" in report["overall"]
    assert set(report["strata"]) >= {"target_shape", "anchor_shape", "direction", "clause_slot"}
    # Predicted anchors report their own quality, or a drop is uninterpretable.
    assert "anchor_dice" in report["anchor_quality"]
    assert "overall" in (prediction_folder / "metrics.txt").read_text()
    assert "rows" not in report


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_an_existing_folder_is_not_clobbered_without_overwrite(prediction_folder, checkpoints):
    with pytest.raises(SystemExit):
        _evaluate(prediction_folder, checkpoints)
    assert _evaluate(prediction_folder, checkpoints, "--overwrite") == 0


def test_predicted_anchors_without_a_stage_a_checkpoint_stop_the_run(tmp_path, checkpoints):
    _, stage_b = checkpoints
    with pytest.raises(SystemExit):
        evaluate_main(
            [
                "--stage-b-checkpoint", str(stage_b),
                "--smoke",
                "--output", str(tmp_path / "never"),
            ]
        )


def test_oracle_anchors_need_no_stage_a_and_can_skip_the_volumes(tmp_path, checkpoints):
    _, stage_b = checkpoints
    output = tmp_path / "oracle"
    assert (
        evaluate_main(
            [
                "--stage-b-checkpoint", str(stage_b),
                "--anchor-source", "oracle",
                "--smoke",
                "--split", SPLIT,
                "--limit-examples", "1",
                "--batch-size", "1",
                "--device", "cpu",
                "--no-checksum",
                "--no-volumes",
                "--output", str(output),
                "--quiet",
            ]
        )
        == 0
    )
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["models"]["stage_a"] is None
    assert metadata["model_names"]["stage_a"] is None
    assert metadata["anchor_source"] == "oracle"
    assert not (output / "predictions").exists()
    row = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
    assert "files" not in row and "dice" in row
