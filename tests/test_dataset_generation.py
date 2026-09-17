"""End-to-end tests for ``scripts/generate_dataset.py``.

Generates a one-scene-per-split dataset into a temporary directory and checks
the on-disk artefacts: scene arrays, manifests, filtering by target class and
the reproducibility metadata.
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
from src.data.schema import (  # noqa: E402
    ANCHOR_UNION_FILENAME,
    INSTANCE_LABELS_FILENAME,
    SCENE_VOLUME_FILENAME,
    SCHEMA_VERSION,
    TARGET_MASK_FILENAME,
    Example,
    ExampleArrays,
    anchor_mask_filename,
    load_scene_arrays,
    read_manifest,
    scene_array_dir,
)
from src.data.validation import validate_example, verify_relations_against_masks  # noqa: E402


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


def test_scene_arrays_and_manifests_are_written(dataset):
    run, root = dataset
    scenes = sorted(path for path in (root / "scenes").iterdir() if path.is_dir())
    assert len(scenes) == 3  # one scene per split
    for scene_dir in scenes:
        assert (scene_dir / SCENE_VOLUME_FILENAME).is_file()
        assert (scene_dir / INSTANCE_LABELS_FILENAME).is_file()
    for split in ("train", "val", "test"):
        assert (root / "manifests" / f"{split}.jsonl").is_file()
        assert (root / "manifests" / f"{split}_candidates.jsonl").is_file()
    assert (root / "run_metadata.json").is_file()
    assert sum(report.scenes for report in run.splits.values()) == 3


def test_every_scene_holds_exactly_ten_instances(dataset):
    _, root = dataset
    for scene_dir in sorted(path for path in (root / "scenes").iterdir() if path.is_dir()):
        scene_volume, instance_labels = load_scene_arrays(scene_dir)
        assert np.array_equal(np.unique(instance_labels), np.arange(0, 11))
        occupancy = instance_labels != 0
        assert scene_volume.dtype == np.float32
        assert float(scene_volume[occupancy].mean()) > float(scene_volume[~occupancy].mean())
        assert not np.allclose(scene_volume[~occupancy], 0.0, atol=1e-6)
        assert not np.allclose(scene_volume[occupancy], 1.0, atol=1e-6)


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
            scene_volume, instance_labels = load_scene_arrays(
                scene_array_dir(root, metadata.scene_id)
            )
            example = Example(
                metadata=metadata,
                arrays=ExampleArrays.from_scene(metadata, scene_volume, instance_labels),
            )
            validate_example(example, margin_voxels=margin)
            verify_relations_against_masks(example)
            example_dir = root / "examples" / metadata.example_id
            dumped_target = load_nifti(example_dir / TARGET_MASK_FILENAME, dtype=np.uint8)
            np.testing.assert_array_equal(dumped_target, example.arrays.target_mask)
            for slot, shape_name in enumerate(metadata.anchor_shape_names):
                dumped = load_nifti(
                    example_dir / anchor_mask_filename(slot, shape_name), dtype=np.uint8
                )
                np.testing.assert_array_equal(dumped, example.arrays.anchor_masks[slot])
            dumped_union = load_nifti(example_dir / ANCHOR_UNION_FILENAME, dtype=np.uint8)
            np.testing.assert_array_equal(dumped_union, example.arrays.anchor_union_mask)
            checked += 1
    assert checked == 30


def test_run_metadata_records_versions_and_environment(dataset):
    _, root = dataset
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["versions"]["direction_rule_version"] == DIRECTION_RULE_VERSION
    assert metadata["versions"]["schema_version"] == SCHEMA_VERSION
    assert metadata["environment"]["platform"]
    assert "git_revision" in metadata["environment"]
    assert set(metadata["configs"]) == {"shapes", "generator", "split", "model", "train"}
    assert metadata["totals"]["candidate_examples"] == 30


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
    (tmp_path / "scenes" / "scene_000000000").mkdir()
    (tmp_path / "scenes" / "scene_000000000" / SCENE_VOLUME_FILENAME).write_bytes(b"not empty")
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
