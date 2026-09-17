"""RAS NIfTI round-trip: axis order, spacing affine, integer labels exact."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.nifti_io import (
    NiftiIOError,
    load_nifti,
    ras_affine,
    save_nifti,
    to_xyz,
    voxel_world_xyz,
)


def test_round_trip_preserves_a_zyx_float_volume(tmp_path):
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "scene_volume.nii.gz"
    save_nifti(path, volume, spacing=(1.0, 2.0, 3.0), dtype=np.float32)
    loaded = load_nifti(path, dtype=np.float32)
    assert loaded.shape == volume.shape
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, volume, rtol=0, atol=1e-6)


def test_integer_labels_round_trip_exactly(tmp_path):
    labels = np.zeros((5, 6, 7), dtype=np.uint8)
    labels[1, 2, 3] = 4
    labels[4, 0, 6] = 10
    path = tmp_path / "instance_labels.nii.gz"
    save_nifti(path, labels, spacing=(1.0, 1.0, 1.0), dtype=np.uint8)
    loaded = load_nifti(path, dtype=np.uint8)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, labels)


def test_on_disk_array_is_xyz_and_affine_matches_project_world(tmp_path):
    import nibabel as nib

    volume = np.zeros((4, 5, 6), dtype=np.float32)
    volume[1, 2, 3] = 1.0  # z=1, y=2, x=3
    path = tmp_path / "marker.nii.gz"
    spacing = (0.5, 1.0, 2.0)
    save_nifti(path, volume, spacing=spacing, dtype=np.float32)
    image = nib.load(str(path))
    xyz = np.asanyarray(image.dataobj)
    assert xyz.shape == (6, 5, 4)  # (x, y, z)
    assert xyz[3, 2, 1] == pytest.approx(1.0)
    np.testing.assert_allclose(image.affine, ras_affine(spacing))
    assert int(image.header["qform_code"]) == 1
    assert int(image.header["sform_code"]) == 1
    assert voxel_world_xyz((1, 2, 3), spacing) == (3 * 0.5, 2 * 1.0, 1 * 2.0)
    # Affine applied to the NIfTI (x, y, z) index of that voxel.
    world = image.affine @ np.array([3.0, 2.0, 1.0, 1.0])
    np.testing.assert_allclose(world[:3], (1.5, 2.0, 2.0))


def test_to_xyz_is_the_inverse_of_loading_order():
    volume = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    assert to_xyz(volume).shape == (5, 4, 3)


def test_bad_paths_and_spacing_are_rejected(tmp_path):
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(NiftiIOError):
        save_nifti(tmp_path / "nope.npz", volume, (1.0, 1.0, 1.0))
    with pytest.raises(NiftiIOError):
        ras_affine((1.0, 0.0, 1.0))
    with pytest.raises(NiftiIOError):
        load_nifti(tmp_path / "missing.nii.gz")
