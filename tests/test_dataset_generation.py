"""End-to-end tests for ``scripts/generate_dataset.py``.

Generates a one-scene-per-split dataset into a temporary directory and checks
the on-disk artefacts: the NIfTI scene directories, the manifests, filtering by
target class and the reproducibility metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_dataset import build_dataset, main, prepare_output  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.primitives import SHAPE_NAMES  # noqa: E402
from src.data.direction_rules import DIRECTION_RULE_VERSION  # noqa: E402
from src.data.nifti_io import load_nifti  # noqa: E402
from src.data.scene_io import (  # noqa: E402
    ANCHOR_UNION_FILE,
    INSTANCE_LABELS_FILE,
    SCENE_VOLUME_FILE,
    TARGET_MASK_FILE,
    anchor_mask_filename,
    example_directory,
    load_scene,
    scene_directory,
    scene_path,
)
from src.data.schema import SCHEMA_VERSION, read_manifest  # noqa: E402
from src.data.validation import validate_example, verify_relations_against_masks  # noqa: E402
from src.data.schema import Example, ExampleArrays  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One smoke-seeded scene per split, written to a temporary directory."""
    root = tmp_path_factory.mktemp("dataset")
    run = build_dataset(
        splits=("train", "val", "test"),
        output_root=root,
        smoke=True,
        limit=1,
        overwrite=True,
        keep_all_candidates=True,
        verbose=False,
    )
    return run, root


def scene_dirs(root):
    return sorted(path for path in (root / "scenes").iterdir() if path.is_dir())


def test_scene_arrays_and_manifests_are_written(dataset):
    run, root = dataset
    scenes = scene_dirs(root)
    assert len(scenes) == 3  # one scene per split
    for directory in scenes:
        for name in (SCENE_VOLUME_FILE, INSTANCE_LABELS_FILE):
            assert (directory / name).is_file(), f"{directory.name} is missing {name}"
        assert {path.name for path in directory.iterdir()} == {
            SCENE_VOLUME_FILE,
            INSTANCE_LABELS_FILE,
        }
    for split in ("train", "val", "test"):
        assert (root / "manifests" / f"{split}.jsonl").is_file()
        assert (root / "manifests" / f"{split}_candidates.jsonl").is_file()
    assert (root / "run_metadata.json").is_file()
    assert sum(report.scenes for report in run.splits.values()) == 3


def test_every_scene_holds_exactly_ten_instances(dataset):
    _, root = dataset
    for path in scene_dirs(root):
        volumes = load_scene(path)
        assert np.array_equal(np.unique(volumes.instance_labels), np.arange(0, 11))
        assert np.array_equal(volumes.occupancy.astype(bool), volumes.instance_labels != 0)


def test_the_image_is_an_mri_like_volume_not_the_binary_occupancy(dataset):
    """The stored image must be a real intensity volume with a tissue background.

    If this ever collapses back to 0/1 the appearance model has silently turned
    itself off and every downstream claim about realism is void.
    """
    _, root = dataset
    for path in scene_dirs(root):
        volumes = load_scene(path)
        image = volumes.image
        assert image is not None and image.dtype == np.float32
        assert len(np.unique(image)) > 1000, "the image is quantised like a mask"
        background = image[volumes.instance_labels == 0]
        assert background.std() > 0.0
        # The structures sit inside tissue, so most non-structure voxels are
        # well above the air floor rather than at zero.
        assert float((background > 0.2 * float(image.max())).mean()) > 0.4


def test_every_instance_is_present_in_the_label_volume(dataset):
    _, root = dataset
    for path in scene_dirs(root):
        labels = load_scene(path).instance_labels
        for instance_id in range(1, len(SHAPE_NAMES) + 1):
            assert (labels == instance_id).any(), f"{path.name} is missing instance {instance_id}"


def test_run_metadata_records_appearance_without_class_conditioned_intensities(dataset):
    run, root = dataset
    record = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    appearance = record["appearance"]
    assert appearance["enabled"] is True
    assert appearance["class_conditioned_intensities"] is False
    assert appearance["measured"]["contrast_to_noise"]["min"] > 1.0
    assert run.splits["train"].scenes == 1


def test_per_example_target_and_anchor_volumes_are_written(dataset):
    _, root = dataset
    for metadata in read_manifest(root / "manifests" / "train_candidates.jsonl"):
        directory = example_directory(root, metadata.example_id)
        labels = load_scene(scene_path(root, metadata.scene_id)).instance_labels
        target, _ = load_nifti(directory / TARGET_MASK_FILE, dtype=np.uint8)
        union, _ = load_nifti(directory / ANCHOR_UNION_FILE, dtype=np.uint8)
        assert np.array_equal(target.astype(bool), labels == metadata.target_instance_id)
        stacked = []
        for slot, (instance_id, shape_name) in enumerate(
            zip(metadata.anchor_instance_ids, metadata.anchor_shape_names)
        ):
            mask, _ = load_nifti(
                directory / anchor_mask_filename(slot, shape_name), dtype=np.uint8
            )
            assert np.array_equal(mask.astype(bool), labels == instance_id)
            stacked.append(mask)
        assert np.array_equal(union.astype(bool), np.any(np.stack(stacked).astype(bool), axis=0))


def test_candidates_hold_all_ten_targets_and_manifests_are_filtered(dataset):
    _, root = dataset
    targets = load_config("split")["target_classes"]
    for split in ("train", "val", "test"):
        candidates = list(read_manifest(root / "manifests" / f"{split}_candidates.jsonl"))
        retained = list(read_manifest(root / "manifests" / f"{split}.jsonl"))
        assert len(candidates) == 10
        assert {m.target_shape_name for m in candidates} == set(SHAPE_NAMES)
        assert len(retained) == len(targets[split])
        assert {m.target_shape_name for m in retained} == set(targets[split])
        assert all(m.split == split for m in retained)


def test_every_written_example_revalidates_from_disk(dataset):
    _, root = dataset
    margin = int(load_config("generator")["geometry"]["in_bounds_margin_voxels"])
    checked = 0
    for split in ("train", "val", "test"):
        for metadata in read_manifest(root / "manifests" / f"{split}_candidates.jsonl"):
            volumes = load_scene(scene_path(root, metadata.scene_id))
            example = Example(
                metadata=metadata,
                arrays=ExampleArrays.from_scene(
                    metadata, volumes.occupancy, volumes.instance_labels
                ),
            )
            validate_example(example, margin_voxels=margin)
            verify_relations_against_masks(example)
            checked += 1
    assert checked == 30


def test_run_metadata_records_versions_and_environment(dataset):
    _, root = dataset
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["versions"]["direction_rule_version"] == DIRECTION_RULE_VERSION
    assert metadata["versions"]["schema_version"] == SCHEMA_VERSION
    assert metadata["environment"]["platform"]
    assert "git_revision" in metadata["environment"]
    assert set(metadata["configs"]) == {
        "shapes",
        "generator",
        "appearance",
        "split",
        "model",
        "train",
    }
    assert metadata["totals"]["candidate_examples"] == 30
    assert metadata["storage"]["scene_array_format"] == "nifti"
    assert metadata["appearance"]["enabled"] is True
    assert metadata["appearance"]["measured"]["mean_contrast"]["mean"] > 0


def test_smoke_seed_ranges_never_collide_with_the_real_corpus():
    generator = load_config("generator")
    split = load_config("split")
    smoke_seeds = {
        seed
        for start, end in (
            tuple(v) for v in generator["smoke"]["scene_seed_ranges"].values()
        )
        for seed in range(start, end)
    }
    corpus_seeds = {
        seed
        for start, end in (tuple(v) for v in split["scene_seed_ranges"].values())
        for seed in range(start, end)
    }
    assert not (smoke_seeds & corpus_seeds)


def test_existing_datasets_are_not_clobbered(tmp_path):
    prepare_output(tmp_path, overwrite=False)
    (tmp_path / "scenes" / "scene_000000000.npz").write_bytes(b"not empty")
    with pytest.raises(SystemExit):
        prepare_output(tmp_path, overwrite=False)
    prepare_output(tmp_path, overwrite=True)
    assert not list((tmp_path / "scenes").iterdir())


def test_cli_entry_point_runs(tmp_path):
    assert main(
        [
            "--smoke",
            "--splits", "test",
            "--limit", "1",
            "--output-root", str(tmp_path),
            "--overwrite",
            "--quiet",
        ]
    ) == 0
    assert (tmp_path / "run_metadata.json").is_file()
