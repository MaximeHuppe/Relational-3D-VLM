"""Tests for the target-relative direction rules.

Covers: centroid and extent computation under ``(z, y, x)`` indexing and
anisotropic spacing, main-axis selection, the ``z > y > x`` tie priority, the
six direction tokens, the medial/lateral centre-plane comparison on both sides
of the volume, and the two rejection cases where a direction must never be
invented.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.direction_rules import (
    ANTERIOR,
    AXIS_OF_DIRECTION,
    DEFAULT_ATOL,
    DIRECTION_OPPOSITES,
    DIRECTION_RULE_VERSION,
    DIRECTIONS,
    INFERIOR,
    LATERAL,
    MEDIAL,
    POSTERIOR,
    SUPERIOR,
    AmbiguityReason,
    AmbiguousDirectionError,
    DirectionError,
    DirectionResult,
    bbox_extent_world,
    centroid_voxel_zyx,
    centroid_world,
    classify_direction,
    direction_between_masks,
    is_valid_direction,
    opposite_direction,
    select_axis,
    validate_distinct_directions,
    volume_center_world,
)

VOLUME_SHAPE = (64, 64, 64)  # (D, H, W)
CENTER = 31.5  # (64 - 1) / 2 on every axis at unit spacing


def classify(target, anchor, *, volume_shape=VOLUME_SHAPE, spacing=(1.0, 1.0, 1.0)):
    """Classify from two world-space points ordered (x, y, z)."""
    return classify_direction(
        target, anchor, volume_shape=volume_shape, spacing=spacing
    )


def cube_mask(shape, center_zyx, side=3, value=True):
    """Place an axis-aligned cube of ``side`` voxels centred on ``center_zyx``."""
    mask = np.zeros(shape, dtype=bool)
    half = side // 2
    z, y, x = center_zyx
    mask[z - half : z + half + 1, y - half : y + half + 1, x - half : x + half + 1] = value
    return mask


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def test_direction_vocabulary_is_exactly_the_six_accepted_tokens():
    assert set(DIRECTIONS) == {
        "anterior",
        "posterior",
        "superior",
        "inferior",
        "medial",
        "lateral",
    }
    assert len(DIRECTIONS) == 6


def test_opposites_are_an_involution_and_stay_on_the_same_axis():
    for direction in DIRECTIONS:
        opposite = opposite_direction(direction)
        assert opposite != direction
        assert opposite_direction(opposite) == direction
        assert AXIS_OF_DIRECTION[opposite] == AXIS_OF_DIRECTION[direction]
    assert DIRECTION_OPPOSITES[MEDIAL] == LATERAL


@pytest.mark.parametrize("token", ["left", "right", "above", "below", "first", "cube", ""])
def test_synonyms_and_ordinals_are_not_valid_directions(token):
    assert not is_valid_direction(token)
    with pytest.raises(DirectionError):
        opposite_direction(token)


def test_rule_version_is_recorded_on_every_result():
    result = classify((0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    assert result.rule_version == DIRECTION_RULE_VERSION
    assert DIRECTION_RULE_VERSION.count(".") == 2


def test_direction_result_rejects_a_direction_axis_mismatch():
    with pytest.raises(DirectionError):
        DirectionResult(direction=SUPERIOR, axis="x", delta=(1.0, 0.0, 0.0), tie_broken=False)
    with pytest.raises(DirectionError):
        DirectionResult(direction="up", axis="z", delta=(0.0, 0.0, 1.0), tie_broken=False)


def test_validate_distinct_directions():
    assert validate_distinct_directions([SUPERIOR, ANTERIOR, LATERAL]) == (
        SUPERIOR,
        ANTERIOR,
        LATERAL,
    )
    with pytest.raises(DirectionError):
        validate_distinct_directions([SUPERIOR, SUPERIOR, LATERAL])
    with pytest.raises(DirectionError):
        validate_distinct_directions([SUPERIOR, "above", LATERAL])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def test_volume_center_matches_the_specified_formula():
    center = volume_center_world((64, 64, 64))
    assert np.allclose(center, (CENTER, CENTER, CENTER))
    # Non-cubic volume: (D, H, W) is (z, y, x) but the point is (x, y, z).
    center = volume_center_world((10, 20, 30))
    assert np.allclose(center, ((30 - 1) / 2, (20 - 1) / 2, (10 - 1) / 2))


def test_volume_center_scales_with_spacing():
    center = volume_center_world((10, 20, 30), spacing=(2.0, 0.5, 4.0))
    assert np.allclose(center, (14.5 * 2.0, 9.5 * 0.5, 4.5 * 4.0))


def test_centroid_uses_zyx_indexing_and_returns_xyz_world_coordinates():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2, 3, 5] = True  # (z, y, x)
    assert np.allclose(centroid_voxel_zyx(mask), (2.0, 3.0, 5.0))
    assert np.allclose(centroid_world(mask), (5.0, 3.0, 2.0))


def test_centroid_applies_spacing_per_world_axis():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2, 3, 5] = True
    assert np.allclose(centroid_world(mask, spacing=(0.5, 1.0, 2.0)), (2.5, 3.0, 4.0))


def test_centroid_of_a_symmetric_object_is_its_geometric_centre():
    mask = cube_mask((16, 16, 16), (8, 6, 4), side=5)
    assert np.allclose(centroid_world(mask), (4.0, 6.0, 8.0))


def test_empty_masks_are_rejected_rather_than_given_a_centroid():
    empty = np.zeros((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError):
        centroid_world(empty)
    with pytest.raises(ValueError):
        bbox_extent_world(empty)


def test_bbox_extent_is_inclusive_and_spacing_aware():
    mask = cube_mask((16, 16, 16), (8, 8, 8), side=3)
    assert np.allclose(bbox_extent_world(mask), (3.0, 3.0, 3.0))
    assert np.allclose(bbox_extent_world(mask, spacing=(2.0, 1.0, 0.5)), (6.0, 3.0, 1.5))
    single = np.zeros((4, 4, 4), dtype=bool)
    single[1, 1, 1] = True
    assert np.allclose(bbox_extent_world(single), (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Axis selection and tie priority
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "delta,expected_axis",
    [
        ((10.0, 1.0, 2.0), "x"),
        ((1.0, 10.0, 2.0), "y"),
        ((1.0, 2.0, 10.0), "z"),
        ((-10.0, 1.0, 2.0), "x"),
        ((0.0, 0.0, -0.5), "z"),
    ],
)
def test_main_axis_is_the_largest_absolute_component(delta, expected_axis):
    axis, tie_broken = select_axis(delta)
    assert axis == expected_axis
    assert tie_broken is False


@pytest.mark.parametrize(
    "delta,expected_axis",
    [
        ((5.0, 5.0, 5.0), "z"),   # three-way tie -> z
        ((5.0, 5.0, 1.0), "y"),   # x/y tie -> y
        ((5.0, 1.0, 5.0), "z"),   # x/z tie -> z
        ((1.0, 5.0, 5.0), "z"),   # y/z tie -> z
        ((-5.0, 5.0, -5.0), "z"), # signs do not affect the priority
    ],
)
def test_ties_follow_the_deterministic_z_then_y_then_x_priority(delta, expected_axis):
    axis, tie_broken = select_axis(delta)
    assert axis == expected_axis
    assert tie_broken is True


def test_zero_delta_is_rejected_not_resolved():
    with pytest.raises(AmbiguousDirectionError) as excinfo:
        select_axis((0.0, 0.0, 0.0))
    assert excinfo.value.reason is AmbiguityReason.ZERO_DELTA

    with pytest.raises(AmbiguousDirectionError) as excinfo:
        classify((10.0, 20.0, 30.0), (10.0, 20.0, 30.0))
    assert excinfo.value.reason is AmbiguityReason.ZERO_DELTA


def test_a_delta_below_tolerance_counts_as_zero():
    with pytest.raises(AmbiguousDirectionError):
        select_axis((0.0, 0.0, DEFAULT_ATOL / 2))


# ---------------------------------------------------------------------------
# z and y axes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "delta,expected",
    [
        ((0.0, 0.0, 7.0), SUPERIOR),
        ((0.0, 0.0, -7.0), INFERIOR),
        ((0.0, 7.0, 0.0), ANTERIOR),
        ((0.0, -7.0, 0.0), POSTERIOR),
        ((2.0, 3.0, 9.0), SUPERIOR),
        ((2.0, -9.0, 3.0), POSTERIOR),
    ],
)
def test_superior_inferior_anterior_posterior(delta, expected):
    anchor = (20.0, 20.0, 20.0)
    target = tuple(a + d for a, d in zip(anchor, delta))
    result = classify(target, anchor)
    assert result.direction == expected
    assert result.axis == AXIS_OF_DIRECTION[expected]
    assert np.allclose(result.delta, delta)


def test_the_prompt_describes_the_target_relative_to_the_anchor():
    # Target above the anchor -> "superior to the <anchor>".
    below, above = (30.0, 30.0, 10.0), (30.0, 30.0, 50.0)
    assert classify(above, below).direction == SUPERIOR
    assert classify(below, above).direction == INFERIOR


@pytest.mark.parametrize("direction_pair", [("z", SUPERIOR, INFERIOR), ("y", ANTERIOR, POSTERIOR)])
def test_swapping_target_and_anchor_flips_the_token(direction_pair):
    axis, positive, negative = direction_pair
    offset = {"z": (0.0, 0.0, 6.0), "y": (0.0, 6.0, 0.0)}[axis]
    anchor = (25.0, 25.0, 25.0)
    target = tuple(a + d for a, d in zip(anchor, offset))
    assert classify(target, anchor).direction == positive
    assert classify(anchor, target).direction == negative
    assert opposite_direction(positive) == negative


# ---------------------------------------------------------------------------
# x axis: medial / lateral
# ---------------------------------------------------------------------------
def test_lateral_when_the_target_is_farther_from_the_centre_plane():
    # Right half: centre plane at x = 31.5.
    result = classify((50.0, 30.0, 30.0), (40.0, 30.0, 30.0))
    assert result.direction == LATERAL
    assert result.axis == "x"


def test_medial_when_the_target_is_closer_to_the_centre_plane():
    result = classify((40.0, 30.0, 30.0), (50.0, 30.0, 30.0))
    assert result.direction == MEDIAL
    assert result.axis == "x"


def test_medial_lateral_is_a_distance_comparison_not_a_sign_test():
    # Left half: the target has a SMALLER x but is FARTHER from the centre.
    result = classify((10.0, 30.0, 30.0), (20.0, 30.0, 30.0))
    assert result.delta[0] < 0
    assert result.direction == LATERAL
    # Mirror case on the left half.
    result = classify((20.0, 30.0, 30.0), (10.0, 30.0, 30.0))
    assert result.delta[0] > 0
    assert result.direction == MEDIAL


def test_medial_lateral_across_the_centre_plane():
    # Target on the right at 13.5 from centre, anchor on the left at 11.5.
    result = classify((45.0, 30.0, 30.0), (20.0, 30.0, 30.0))
    assert result.direction == LATERAL
    result = classify((20.0, 30.0, 30.0), (45.0, 30.0, 30.0))
    assert result.direction == MEDIAL


def test_swapping_target_and_anchor_flips_medial_and_lateral():
    target, anchor = (50.0, 30.0, 30.0), (40.0, 30.0, 30.0)
    forward = classify(target, anchor).direction
    backward = classify(anchor, target).direction
    assert {forward, backward} == {MEDIAL, LATERAL}
    assert opposite_direction(forward) == backward


def test_equidistant_from_the_centre_plane_is_rejected_not_invented():
    # |20 - 31.5| == |43 - 31.5| == 11.5, on opposite sides.
    with pytest.raises(AmbiguousDirectionError) as excinfo:
        classify((20.0, 30.0, 30.0), (43.0, 30.0, 30.0))
    assert excinfo.value.reason is AmbiguityReason.MEDIAL_LATERAL_TIE
    assert "regenerate" in str(excinfo.value)


def test_the_medial_lateral_tie_only_applies_when_x_actually_wins():
    # Same equidistant x pair, but a larger z component takes the axis.
    result = classify((20.0, 30.0, 60.0), (43.0, 30.0, 0.0))
    assert result.direction == SUPERIOR


def test_the_centre_plane_can_be_supplied_directly():
    by_shape = classify((50.0, 30.0, 30.0), (40.0, 30.0, 30.0))
    by_centre = classify_direction(
        (50.0, 30.0, 30.0), (40.0, 30.0, 30.0), center_world=(CENTER, CENTER, CENTER)
    )
    assert by_shape.direction == by_centre.direction


def test_exactly_one_centre_specification_is_required():
    with pytest.raises(ValueError):
        classify_direction((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        classify_direction(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), volume_shape=VOLUME_SHAPE, center_world=(0.0, 0.0, 0.0)
        )


def test_a_smaller_volume_moves_the_centre_plane_and_can_flip_the_token():
    target, anchor = (28.0, 10.0, 10.0), (20.0, 10.0, 10.0)
    # Centre 31.5: the target is closer to the centre -> medial.
    assert classify(target, anchor).direction == MEDIAL
    # Centre 15.5 (a 32^3 volume): the target is now farther out -> lateral.
    assert classify(target, anchor, volume_shape=(32, 32, 32)).direction == LATERAL


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------
def test_axis_selection_is_normalised_by_physical_spacing():
    shape = (32, 32, 32)
    anchor = cube_mask(shape, (10, 10, 16), side=3)
    target = cube_mask(shape, (12, 15, 16), side=3)  # +2 voxels in z, +5 in y

    isotropic = direction_between_masks(target, anchor, spacing=(1.0, 1.0, 1.0))
    assert isotropic.axis == "y" and isotropic.direction == ANTERIOR

    # z voxels are four times larger: 2 * 4 = 8 world units beats 5.
    anisotropic = direction_between_masks(target, anchor, spacing=(1.0, 1.0, 4.0))
    assert anisotropic.axis == "z" and anisotropic.direction == SUPERIOR


def test_direction_between_masks_matches_classify_direction():
    shape = (32, 32, 32)
    anchor = cube_mask(shape, (8, 8, 8), side=3)
    target = cube_mask(shape, (20, 9, 10), side=3)
    from_masks = direction_between_masks(target, anchor)
    from_points = classify_direction(
        centroid_world(target), centroid_world(anchor), volume_shape=shape
    )
    assert from_masks.direction == from_points.direction
    assert from_masks.axis == from_points.axis
    assert np.allclose(from_masks.delta, from_points.delta)


def test_direction_between_masks_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        direction_between_masks(np.ones((4, 4, 4), bool), np.ones((4, 4, 5), bool))


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------
def _reference_direction(target, anchor, center_x):
    """Independent re-implementation of the rule, used as a cross-check."""
    delta = [t - a for t, a in zip(target, anchor)]
    magnitudes = [abs(value) for value in delta]
    largest = max(magnitudes)
    for axis, index in (("z", 2), ("y", 1), ("x", 0)):
        if magnitudes[index] == largest:
            break
    if axis == "z":
        return SUPERIOR if delta[2] > 0 else INFERIOR
    if axis == "y":
        return ANTERIOR if delta[1] > 0 else POSTERIOR
    return LATERAL if abs(target[0] - center_x) > abs(anchor[0] - center_x) else MEDIAL


def test_random_pairs_always_yield_a_valid_token_matching_the_reference_rule():
    rng = np.random.default_rng(20260915)
    checked = 0
    for _ in range(2000):
        target = tuple(float(v) for v in rng.integers(0, 64, size=3))
        anchor = tuple(float(v) for v in rng.integers(0, 64, size=3))
        try:
            result = classify(target, anchor)
        except AmbiguousDirectionError as error:
            assert error.reason in tuple(AmbiguityReason)
            continue
        assert result.direction in DIRECTIONS
        assert AXIS_OF_DIRECTION[result.direction] == result.axis
        assert result.direction == _reference_direction(target, anchor, CENTER)
        checked += 1
    assert checked > 1800  # ambiguities must stay rare, not dominate


def test_relations_are_antisymmetric_for_random_pairs():
    rng = np.random.default_rng(7)
    for _ in range(500):
        target = tuple(float(v) for v in rng.integers(0, 64, size=3))
        anchor = tuple(float(v) for v in rng.integers(0, 64, size=3))
        try:
            forward = classify(target, anchor).direction
            backward = classify(anchor, target).direction
        except AmbiguousDirectionError:
            continue
        assert backward == opposite_direction(forward)
