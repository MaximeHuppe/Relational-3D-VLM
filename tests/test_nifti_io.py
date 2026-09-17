"""NIfTI I/O and the on-disk scene layout.

The orientation tests are the important ones. Arrays are indexed ``(z, y, x)``
here and ``(i, j, k)`` in a NIfTI file, and the whole dataset is about
left/right, anterior/posterior and superior/inferior - so if the transpose or
the affine were wrong, every direction in every prompt would be wrong in a
viewer while every in-memory test still passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.nifti_io import (
    NiftiError,
    from_nifti_order,
    load_nifti,
    save_nifti,
    scene_affine,
    spacing_from_affine,
    to_nifti_order,
)
from src.data.primitives import SHAPE_VOCABULARY
from src.data.scene_generator import generate_scene, scene_record
from src.data.scene_io import (
    ANCHOR_UNION_FILE,
    IMAGE_FILE,
    INSTANCE_LABELS_FILE,
    LABELS_FILE,
    MASK_DIR,
    OCCUPANCY_FILE,
    SCENE_RECORD,
    SCENE_VOLUME_FILE,
    TARGET_MASK_FILE,
    SceneIOError,
    anchor_mask_filename,
    ensure_scene_name_aliases,
    example_directory,
    load_scene,
    load_scene_record,
    load_structure_mask,
    mask_filename,
    save_example_masks,
    save_scene,
    scene_directory,
    scene_path,
)
from src.data.schema import load_scene_arrays, save_scene_arrays
from src.data.voxelization import world_coordinate_grids

nib = pytest.importorskip("nibabel")


@pytest.fixture(scope="module")
def scene():
    return generate_scene(1000000)


@pytest.fixture(scope="module")
def written(tmp_path_factory, scene):
    root = tmp_path_factory.mktemp("corpus")
    save_scene(
        root,
        scene.scene_id,
        instance_labels=scene.instance_labels,
        occupancy=scene.scene_volume,
        spacing=scene.spacing,
        image=scene.appearance.image,
        record=scene_record(scene, split="train"),
    )
    return root, scene


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------
def test_the_array_order_round_trips():
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert to_nifti_order(array).shape == (4, 3, 2)
    assert np.array_equal(from_nifti_order(to_nifti_order(array)), array)


def test_a_channel_axis_becomes_the_fourth_nifti_dimension():
    stack = np.zeros((3, 5, 6, 7))
    assert to_nifti_order(stack).shape == (7, 6, 5, 3)
    assert np.array_equal(from_nifti_order(to_nifti_order(stack)), stack)


def test_a_voxels_nifti_world_coordinate_is_its_world_coordinate_here(tmp_path):
    """A marker at ``(z, y, x)`` must land at ``(x, y, z)`` millimetres in RAS."""
    spacing = (1.0, 2.0, 3.0)
    volume = np.zeros((8, 9, 10), dtype=np.uint8)
    volume[3, 5, 7] = 1  # z=3, y=5, x=7
    path = save_nifti(tmp_path / "marker.nii.gz", volume, spacing)

    image = nib.load(str(path))
    indices = np.argwhere(np.asanyarray(image.dataobj) > 0)
    assert indices.tolist() == [[7, 5, 3]]  # (i, j, k) == (x, y, z)
    world = nib.affines.apply_affine(image.affine, indices[0])
    z, y, x = world_coordinate_grids(volume.shape, spacing)
    assert world.tolist() == [float(x[0, 0, 7]), float(y[0, 5, 0]), float(z[3, 0, 0])]


def test_the_affine_encodes_the_spacing_and_no_translation():
    affine = scene_affine((1.0, 2.0, 3.0))
    assert spacing_from_affine(affine) == (1.0, 2.0, 3.0)
    assert affine[:3, 3].tolist() == [0.0, 0.0, 0.0]


def test_a_saved_volume_declares_an_anatomical_orientation(tmp_path):
    path = save_nifti(tmp_path / "v.nii.gz", np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0))
    header = nib.load(str(path))
    assert int(header.header["sform_code"]) != 0
    assert int(header.header["qform_code"]) != 0
    assert nib.aff2axcodes(header.affine) == ("R", "A", "S")


def test_a_nonpositive_spacing_is_rejected():
    with pytest.raises(NiftiError):
        scene_affine((1.0, 0.0, 1.0))


def test_a_missing_volume_fails_with_its_path(tmp_path):
    with pytest.raises(NiftiError, match="absent.nii.gz"):
        load_nifti(tmp_path / "absent.nii.gz")


# ---------------------------------------------------------------------------
# The scene directory
# ---------------------------------------------------------------------------
def test_a_scene_directory_holds_scene_volume_and_instance_labels(written):
    root, scene = written
    directory = scene_directory(root, scene.scene_id)
    for name in (SCENE_VOLUME_FILE, INSTANCE_LABELS_FILE):
        assert (directory / name).is_file()
    for name in (IMAGE_FILE, LABELS_FILE, OCCUPANCY_FILE, SCENE_RECORD):
        assert not (directory / name).exists()
    assert not (directory / MASK_DIR).exists()
    assert scene_path(root, scene.scene_id) == directory


def test_a_scene_with_only_the_shared_volume_names_still_loads(written, tmp_path, scene):
    source = scene_directory(written[0], scene.scene_id)
    directory = tmp_path / "scenes" / scene.scene_id
    directory.mkdir(parents=True)
    for name in (SCENE_VOLUME_FILE, INSTANCE_LABELS_FILE):
        (directory / name).write_bytes((source / name).read_bytes())
    volumes = load_scene(directory)
    assert np.array_equal(volumes.instance_labels, scene.instance_labels)
    assert volumes.has_image
    assert np.array_equal(volumes.occupancy.astype(bool), scene.instance_labels != 0)
    assert scene_path(tmp_path, scene.scene_id) == directory


def test_an_older_corpus_gains_the_long_names_by_aliasing(tmp_path, scene):
    root = tmp_path / "old"
    save_scene(
        root,
        scene.scene_id,
        instance_labels=scene.instance_labels,
        occupancy=scene.scene_volume,
        spacing=scene.spacing,
        image=scene.appearance.image,
        inspection_extras=True,
        record=scene_record(scene, split="train"),
    )
    directory = scene_directory(root, scene.scene_id)
    (directory / SCENE_VOLUME_FILE).unlink()
    (directory / INSTANCE_LABELS_FILE).unlink()
    assert (directory / IMAGE_FILE).is_file()
    ensure_scene_name_aliases(directory)
    assert (directory / SCENE_VOLUME_FILE).samefile(directory / IMAGE_FILE)
    assert (directory / INSTANCE_LABELS_FILE).samefile(directory / LABELS_FILE)
    volumes = load_scene(directory)
    assert np.array_equal(volumes.instance_labels, scene.instance_labels)


def test_a_scene_reads_back_with_its_labels_and_occupancy_intact(written):
    root, scene = written
    volumes = load_scene(scene_directory(root, scene.scene_id))
    assert np.array_equal(volumes.instance_labels, scene.instance_labels)
    assert np.array_equal(volumes.occupancy, scene.scene_volume)
    assert volumes.spacing == scene.spacing
    assert volumes.has_image


def test_the_image_survives_uint16_storage_well_below_the_noise_level(written):
    """uint16 with a header scale factor is how scanner data is stored."""
    root, scene = written
    volumes = load_scene(scene_directory(root, scene.scene_id))
    error = float(np.abs(volumes.image - scene.appearance.image).max())
    assert error < 1e-4
    assert error < 0.01 * scene.appearance.parameters["noise_sigma"]


def test_every_stored_structure_mask_matches_the_label_volume(tmp_path, scene):
    save_scene(
        tmp_path,
        scene.scene_id,
        instance_labels=scene.instance_labels,
        occupancy=scene.scene_volume,
        spacing=scene.spacing,
        image=scene.appearance.image,
        inspection_extras=True,
        record=scene_record(scene, split="train"),
    )
    directory = scene_directory(tmp_path, scene.scene_id)
    for spec in SHAPE_VOCABULARY:
        mask = load_structure_mask(directory, spec.id)
        assert np.array_equal(mask.astype(bool), scene.instance_labels == spec.id)


def test_the_scene_record_is_self_describing(tmp_path, scene):
    save_scene(
        tmp_path,
        scene.scene_id,
        instance_labels=scene.instance_labels,
        occupancy=scene.scene_volume,
        spacing=scene.spacing,
        image=scene.appearance.image,
        inspection_extras=True,
        record=scene_record(scene, split="train"),
    )
    record = load_scene_record(scene_directory(tmp_path, scene.scene_id))
    assert record["scene_id"] == scene.scene_id
    assert record["array_order"] == "(z, y, x)"
    assert record["world_frame"].startswith("RAS")
    assert record["affine"] == [[float(v) for v in row] for row in scene_affine(scene.spacing)]
    assert record["split"] == "train"
    assert set(record["structures"]) == set(SHAPE_VOCABULARY.names)
    assert record["appearance"]["structure_intensities"].keys() == set(SHAPE_VOCABULARY.names)


def test_example_masks_are_written_in_clause_order(written):
    root, scene = written
    example = scene.examples[0].metadata
    save_example_masks(
        root,
        example.example_id,
        instance_labels=scene.instance_labels,
        target_instance_id=example.target_instance_id,
        anchor_instance_ids=example.anchor_instance_ids,
        anchor_shape_names=example.anchor_shape_names,
        spacing=scene.spacing,
    )
    directory = example_directory(root, example.example_id)
    target, _ = load_nifti(directory / TARGET_MASK_FILE, dtype=np.uint8)
    union, _ = load_nifti(directory / ANCHOR_UNION_FILE, dtype=np.uint8)
    assert np.array_equal(target.astype(bool), scene.instance_labels == example.target_instance_id)
    stacked = []
    for slot, (instance_id, shape_name) in enumerate(
        zip(example.anchor_instance_ids, example.anchor_shape_names)
    ):
        mask, _ = load_nifti(directory / anchor_mask_filename(slot, shape_name), dtype=np.uint8)
        assert np.array_equal(mask.astype(bool), scene.instance_labels == instance_id)
        stacked.append(mask)
    assert np.array_equal(union.astype(bool), np.any(np.stack(stacked).astype(bool), axis=0))


def test_an_absent_scene_says_which_scene_is_missing(tmp_path):
    with pytest.raises(SceneIOError, match="scene_missing"):
        scene_path(tmp_path, "scene_missing")


def test_reading_something_that_is_neither_format_fails(tmp_path):
    stray = tmp_path / "stray.txt"
    stray.write_text("not a scene", encoding="utf-8")
    with pytest.raises(SceneIOError):
        load_scene(stray)


# ---------------------------------------------------------------------------
# The legacy single-file format
# ---------------------------------------------------------------------------
def test_the_npz_format_still_round_trips_including_the_image(tmp_path, scene):
    path = tmp_path / "scenes" / f"{scene.scene_id}.npz"
    save_scene_arrays(path, scene.scene_volume, scene.instance_labels, image=scene.appearance.image)
    volumes = load_scene(path)
    assert np.array_equal(volumes.instance_labels, scene.instance_labels)
    assert np.array_equal(volumes.image, scene.appearance.image)
    assert scene_path(tmp_path, scene.scene_id) == path


def test_an_npz_without_an_image_loads_as_an_appearance_free_scene(tmp_path, scene):
    path = tmp_path / "scenes" / f"{scene.scene_id}.npz"
    save_scene_arrays(path, scene.scene_volume, scene.instance_labels)
    volumes = load_scene(path)
    assert not volumes.has_image
    assert np.array_equal(volumes.model_image("auto"), scene.scene_volume)
    with pytest.raises(SceneIOError):
        volumes.model_image("intensity")


def test_load_scene_arrays_reads_either_format(written, tmp_path, scene):
    root, _ = written
    occupancy, labels = load_scene_arrays(scene_directory(root, scene.scene_id))
    assert np.array_equal(labels, scene.instance_labels)
    assert np.array_equal(occupancy, scene.scene_volume)

    path = tmp_path / "scenes" / f"{scene.scene_id}.npz"
    save_scene_arrays(path, scene.scene_volume, scene.instance_labels)
    legacy_occupancy, legacy_labels = load_scene_arrays(path)
    assert np.array_equal(legacy_labels, labels)
    assert np.array_equal(legacy_occupancy, occupancy)
