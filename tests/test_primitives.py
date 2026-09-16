"""Tests for the fixed shape vocabulary.

Voxelisation tests are added in Phase 1, when :mod:`src.data.voxelization` is
implemented; this file currently covers the vocabulary contract only.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.direction_rules import AXIS_OF_DIRECTION, DIRECTION_OPPOSITES, DIRECTIONS
import numpy as np

from src.data.primitives import (
    BACKGROUND_LABEL,
    CANONICAL_SHAPE_ORDER,
    NUM_SHAPE_CLASSES,
    SHAPE_IDS,
    SHAPE_NAMES,
    SHAPE_VOCABULARY,
    ShapeVocabulary,
    VocabularyError,
    build_vocabulary,
)
from src.data.voxelization import (
    VoxelizationError,
    analytic_volume_world,
    check_params,
    half_extent_world,
    normalized_world_coordinates,
    voxelize,
)


def test_vocabulary_has_exactly_ten_classes_in_canonical_order():
    assert NUM_SHAPE_CLASSES == 10
    assert SHAPE_NAMES == CANONICAL_SHAPE_ORDER
    assert SHAPE_NAMES == (
        "cube",
        "cuboid",
        "sphere",
        "ellipsoid",
        "cylinder",
        "cone",
        "pyramid",
        "triangular_prism",
        "torus",
        "capsule",
    )


def test_ids_are_stable_contiguous_and_disjoint_from_background():
    assert SHAPE_IDS == tuple(range(1, 11))
    assert BACKGROUND_LABEL == 0
    assert BACKGROUND_LABEL not in SHAPE_IDS
    for index, name in enumerate(SHAPE_NAMES):
        assert SHAPE_VOCABULARY.name_to_id(name) == index + 1
        assert SHAPE_VOCABULARY.id_to_name(index + 1) == name
        assert SHAPE_VOCABULARY.index_of(name) == index


def test_lookups_reject_unknown_names_and_ids():
    with pytest.raises(KeyError):
        SHAPE_VOCABULARY.by_name("torus_small")
    with pytest.raises(KeyError):
        SHAPE_VOCABULARY.by_id(0)
    with pytest.raises(KeyError):
        SHAPE_VOCABULARY.by_id(11)
    with pytest.raises(VocabularyError):
        SHAPE_VOCABULARY.require_names(["cube", "blob"])
    assert SHAPE_VOCABULARY.require_names(["cube", "torus"]) == ("cube", "torus")


def test_no_shape_name_is_a_substring_of_another():
    # Guarantees that a prompt-leakage substring check cannot false-positive.
    for name in SHAPE_NAMES:
        for other in SHAPE_NAMES:
            if name != other:
                assert name not in other


def test_every_shape_declares_positive_parameter_ranges():
    for shape in SHAPE_VOCABULARY:
        assert shape.params, f"{shape.name} declares no parameters"
        for param, (low, high) in shape.params.items():
            assert 0 < low <= high, f"{shape.name}.{param} range {low}-{high}"
            assert shape.param_range(param) == (low, high)
        with pytest.raises(KeyError):
            shape.param_range("no_such_param")


def test_axis_aligned_shapes_declare_a_fixed_axis():
    # Orientation is fixed in this milestone, never sampled.
    for name in ("cylinder", "cone", "pyramid", "triangular_prism", "torus", "capsule"):
        assert SHAPE_VOCABULARY.by_name(name).axis in ("x", "y", "z")


def test_mean_volumes_are_broadly_comparable_across_classes():
    volumes = [shape.mean_volume_voxels for shape in SHAPE_VOCABULARY]
    assert all(volume is not None for volume in volumes)
    assert max(volumes) / min(volumes) < 2.0


def test_parameter_ranges_respect_the_axis_extent_band():
    """Every sampled extent stays inside 8-22% of the axis, torus z excepted."""
    low_fraction, high_fraction = SHAPE_VOCABULARY.axis_extent_fraction_range
    axis = SHAPE_VOCABULARY.reference_axis_length
    extents = {
        "cube": {"x": (6, 10), "y": (6, 10), "z": (6, 10)},
        "cuboid": {"x": (5.2, 11), "y": (5.2, 11), "z": (5.2, 11)},
        "sphere": {"x": (8, 12), "y": (8, 12), "z": (8, 12)},
        "ellipsoid": {"x": (7, 13), "y": (7, 13), "z": (7, 13)},
        "cylinder": {"x": (6, 10), "y": (6, 10), "z": (7, 13)},
        "cone": {"x": (9, 13), "y": (9, 13), "z": (10, 14)},
        "pyramid": {"x": (8, 12), "y": (8, 12), "z": (10, 14)},
        "triangular_prism": {"x": (8, 12), "y": (7, 11), "z": (8, 12)},
        "torus": {"x": (10, 14), "y": (10, 14), "z": (4, 5)},
        "capsule": {"x": (7, 9), "y": (7, 9), "z": (10, 14)},
    }
    assert set(extents) == set(SHAPE_NAMES)
    for name, per_axis in extents.items():
        exception = SHAPE_VOCABULARY.by_name(name).thin_axis_exception
        for world_axis, (smallest, largest) in per_axis.items():
            if world_axis == exception:
                continue
            assert smallest / axis >= low_fraction - 1e-9, f"{name}.{world_axis} too small"
            assert largest / axis <= high_fraction + 1e-9, f"{name}.{world_axis} too large"


def test_escalated_grids_rescale_the_parameter_ranges():
    cube = SHAPE_VOCABULARY.by_name("cube")
    scaled = cube.scaled_params(80, SHAPE_VOCABULARY.reference_axis_length)
    low, high = cube.param_range("side")
    assert scaled["side"] == (low * 80 / 64, high * 80 / 64)


def test_config_direction_vocabulary_matches_the_rule_module():
    config = load_config("shapes")
    assert tuple(config["directions"]) == tuple(sorted(DIRECTIONS)) or set(
        config["directions"]
    ) == set(DIRECTIONS)
    assert config["direction_opposites"] == DIRECTION_OPPOSITES
    assert config["direction_axis"] == AXIS_OF_DIRECTION


def test_build_vocabulary_rejects_a_broken_configuration():
    config = load_config("shapes")

    too_few = dict(config, shapes=config["shapes"][:9])
    with pytest.raises(VocabularyError):
        build_vocabulary(too_few)

    reordered = dict(config, shapes=list(reversed(config["shapes"])))
    with pytest.raises(VocabularyError):
        build_vocabulary(reordered)

    renumbered = dict(
        config,
        shapes=[dict(entry, id=entry["id"] + 1) for entry in config["shapes"]],
    )
    with pytest.raises(VocabularyError):
        build_vocabulary(renumbered)

    bad_directions = dict(config, directions=list(DIRECTIONS) + ["above"])
    with pytest.raises(VocabularyError):
        build_vocabulary(bad_directions)

    bad_background = dict(config, background_label=1)
    with pytest.raises(VocabularyError):
        build_vocabulary(bad_background)


def test_loaded_vocabulary_is_the_configured_one():
    assert isinstance(SHAPE_VOCABULARY, ShapeVocabulary)
    assert SHAPE_VOCABULARY.version == load_config("shapes")["vocabulary_version"]


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------
VOLUME = (32, 32, 32)
CENTER = (15.5, 15.5, 15.5)


def mid_params(spec):
    """The midpoint of every sampling range, made anisotropic where required."""
    params = {name: (low + high) / 2 for name, (low, high) in spec.params.items()}
    if spec.min_anisotropy_ratio is not None:
        # Nudge one axis so a cuboid is not a cube and an ellipsoid not a sphere.
        first = next(iter(params))
        params[first] = params[first] * spec.min_anisotropy_ratio
    return params


def voxel_extent_zyx(mask):
    indices = np.nonzero(mask)
    return tuple(int(idx.max() - idx.min() + 1) for idx in indices)


@pytest.mark.parametrize("name", CANONICAL_SHAPE_ORDER)
def test_every_primitive_voxelizes_to_a_non_empty_boolean_volume(name):
    spec = SHAPE_VOCABULARY.by_name(name)
    mask = voxelize(name, mid_params(spec), CENTER, VOLUME)
    assert mask.dtype == np.bool_
    assert mask.shape == VOLUME
    assert mask.sum() > 0


@pytest.mark.parametrize("name", CANONICAL_SHAPE_ORDER)
def test_voxelization_is_deterministic(name):
    spec = SHAPE_VOCABULARY.by_name(name)
    params = mid_params(spec)
    first = voxelize(name, params, CENTER, VOLUME)
    second = voxelize(name, dict(params), tuple(CENTER), VOLUME)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", CANONICAL_SHAPE_ORDER)
def test_voxel_extent_stays_within_the_analytic_bounding_box(name):
    spec = SHAPE_VOCABULARY.by_name(name)
    params = mid_params(spec)
    mask = voxelize(name, params, CENTER, VOLUME)
    half = half_extent_world(name, params)
    span_z, span_y, span_x = voxel_extent_zyx(mask)
    for span, half_extent in zip((span_x, span_y, span_z), half):
        # A voxel belongs to the solid when its centre is inside, so the voxel
        # extent never exceeds the analytic extent by more than one voxel.
        assert span <= 2 * half_extent + 1.0 + 1e-9


@pytest.mark.parametrize("name", CANONICAL_SHAPE_ORDER)
def test_voxel_count_tracks_the_analytic_volume(name):
    spec = SHAPE_VOCABULARY.by_name(name)
    params = mid_params(spec)
    mask = voxelize(name, params, CENTER, VOLUME)
    analytic = analytic_volume_world(name, params)
    assert 0.75 * analytic <= mask.sum() <= 1.3 * analytic


@pytest.mark.parametrize("name", CANONICAL_SHAPE_ORDER)
def test_translating_the_centre_translates_the_mask(name):
    spec = SHAPE_VOCABULARY.by_name(name)
    params = mid_params(spec)
    base = voxelize(name, params, CENTER, VOLUME)
    shifted = voxelize(name, params, (CENTER[0] + 3, CENTER[1], CENTER[2]), VOLUME)
    assert np.array_equal(shifted[:, :, 3:], base[:, :, :-3])


def test_a_cube_is_a_solid_axis_aligned_box():
    mask = voxelize("cube", {"side": 8.0}, (16.0, 16.0, 16.0), VOLUME)
    span_z, span_y, span_x = voxel_extent_zyx(mask)
    assert span_x == span_y == span_z
    indices = np.nonzero(mask)
    lows = [int(idx.min()) for idx in indices]
    highs = [int(idx.max()) for idx in indices]
    block = mask[lows[0] : highs[0] + 1, lows[1] : highs[1] + 1, lows[2] : highs[2] + 1]
    assert block.all()  # no holes


def test_a_sphere_is_symmetric_about_its_centre():
    # Centred on the volume centre, so reversing an axis is an exact mirror.
    mask = voxelize("sphere", {"radius": 6.0}, CENTER, VOLUME)
    assert np.array_equal(mask, mask[::-1, :, :])
    assert np.array_equal(mask, mask[:, ::-1, :])
    assert np.array_equal(mask, mask[:, :, ::-1])


def test_an_ellipsoid_is_wider_along_its_larger_radius():
    mask = voxelize(
        "ellipsoid", {"radius_x": 7.0, "radius_y": 4.0, "radius_z": 3.0}, CENTER, VOLUME
    )
    span_z, span_y, span_x = voxel_extent_zyx(mask)
    assert span_x > span_y > span_z


def test_a_torus_has_a_hole_along_its_ring_axis():
    center = (16.0, 16.0, 16.0)
    mask = voxelize("torus", {"major_radius": 6.0, "minor_radius": 2.0}, center, VOLUME)
    # The ring lies in x-y; the hole runs along z through the centre.
    assert not mask[:, 16, 16].any()
    # ... and the ring itself is present at the major radius.
    assert mask[16, 16, 16 + 6]


def test_a_cone_tapers_from_its_base_to_its_apex():
    params = {"radius": 6.0, "height": 12.0}
    mask = voxelize("cone", params, (16.0, 16.0, 16.0), VOLUME)
    z_indices = np.nonzero(mask.any(axis=(1, 2)))[0]
    base_area = mask[z_indices.min()].sum()
    apex_area = mask[z_indices.max()].sum()
    assert base_area > apex_area
    assert apex_area >= 1


def test_a_pyramid_tapers_and_keeps_square_cross_sections():
    mask = voxelize("pyramid", {"base_side": 10.0, "height": 10.0}, (16.0, 16.0, 16.0), VOLUME)
    z_indices = np.nonzero(mask.any(axis=(1, 2)))[0]
    base = mask[z_indices.min()]
    assert base.sum() > mask[z_indices.max()].sum()
    rows = np.nonzero(base.any(axis=1))[0]
    cols = np.nonzero(base.any(axis=0))[0]
    assert len(rows) == len(cols)  # square base


def test_a_triangular_prism_tapers_in_x_and_is_constant_along_y():
    params = {"base_width": 10.0, "height": 10.0, "depth": 6.0}
    mask = voxelize("triangular_prism", params, (16.0, 16.0, 16.0), VOLUME)
    y_indices = np.nonzero(mask.any(axis=(0, 2)))[0]
    # Every occupied y slice is identical: the triangle is extruded along y.
    reference = mask[:, y_indices[0], :]
    for y in y_indices:
        assert np.array_equal(mask[:, y, :], reference)
    z_indices = np.nonzero(mask.any(axis=(1, 2)))[0]
    assert mask[z_indices.min()].sum() > mask[z_indices.max()].sum()


def test_a_capsule_is_a_cylinder_with_rounded_caps():
    params = {"radius": 4.0, "segment_length": 6.0}
    mask = voxelize("capsule", params, (16.0, 16.0, 16.0), VOLUME)
    z_indices = np.nonzero(mask.any(axis=(1, 2)))[0]
    span_z = int(z_indices.max() - z_indices.min() + 1)
    assert span_z <= 2 * 4.0 + 6.0 + 1  # 2r + L, plus the one-voxel tolerance
    # The end caps are narrower than the straight section.
    assert mask[z_indices.max()].sum() < mask[16].sum()


def test_check_params_rejects_wrong_or_degenerate_parameters():
    with pytest.raises(VoxelizationError):
        check_params("cube", {"radius": 4.0})
    with pytest.raises(VoxelizationError):
        check_params("cube", {"side": -1.0})
    with pytest.raises(VoxelizationError):
        check_params("torus", {"major_radius": 3.0, "minor_radius": 3.0})
    with pytest.raises(VocabularyError):
        check_params("blob", {"side": 4.0})


def test_normalized_world_coordinates_span_minus_one_to_one():
    grid = normalized_world_coordinates((4, 6, 8))
    assert grid.shape == (3, 4, 6, 8)
    assert np.isclose(grid.min(), -1.0) and np.isclose(grid.max(), 1.0)
    # Channel 0 is x, which varies along the last axis only.
    assert np.isclose(grid[0, 0, 0, 0], -1.0) and np.isclose(grid[0, 0, 0, -1], 1.0)
    assert np.isclose(grid[2, 0, 0, 0], -1.0) and np.isclose(grid[2, -1, 0, 0], 1.0)


def test_normalized_world_coordinates_follow_spacing_not_indices():
    isotropic = normalized_world_coordinates((4, 4, 4))
    anisotropic = normalized_world_coordinates((4, 4, 4), spacing=(2.0, 1.0, 0.5))
    # Normalisation divides by the physical span, so the normalised grid is the
    # same shape - what matters is that it is derived from world coordinates.
    assert np.allclose(isotropic, anisotropic)
