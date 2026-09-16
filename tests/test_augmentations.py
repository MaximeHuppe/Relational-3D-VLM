"""Augmentation relation-rewrite tests for the 90-degree rotation augmentation.

No augmentation may be turned on in ``configs/train.yaml`` before its rewrite is
tested here: a flip, translation, crop or axis permutation must consistently
rewrite the scene and instance masks, the three anchor channels, world
coordinates, centroids and extents, the direction labels and the prompt clauses.
``rotation_90`` is enabled, so this module carries that burden of proof.

The load-bearing test is
:func:`test_rewritten_directions_match_directions_recomputed_from_rotated_voxels`:
for every example it rotates the *voxels*, re-derives each direction from the
rotated masks with the project's canonical rule, and requires the rewrite to
have predicted exactly that. The rewrite works from centroids and never touches
a voxel, so the two derivations are genuinely independent - an axis-order or
sign error in :meth:`AxisRotation.apply` would show up as a disagreement rather
than cancel out.

Everything else here guards one specific way the rewrite could be wrong:

* the sampled transforms really are the 24 proper rotations, never a mirror;
* the voxel rotation and the world-coordinate map agree on centroids, extents
  and array shape, on a non-cubic grid with anisotropic spacing;
* a 180-degree turn swaps the two *signed* pairs but deliberately leaves
  medial/lateral alone, because that rule reads centre distance, not sign;
* a 90-degree turn moves a pair between the two families, which is why no
  token-to-token relabelling can be correct;
* an illegal rewrite is rejected, never repaired by inventing a direction;
* the pose is a pure function of ``(seed, epoch, index)``;
* what the datasets emit - ``direction_ids``, ``prompt``, ``directions`` and
  the mask tensors - all describe one and the same pose.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import DatasetError, ExampleDataset, SceneDataset
from src.data.direction_rules import (
    ANTERIOR,
    AXIS_OF_DIRECTION,
    DIRECTION_OPPOSITES,
    INFERIOR,
    LATERAL,
    MEDIAL,
    POSTERIOR,
    SUPERIOR,
    AmbiguousDirectionError,
    bbox_extent_world,
    centroid_world,
    classify_direction,
    direction_between_masks,
)
from src.data.prompt_generator import parse_prompt, render_prompt
from src.training.augmentations import (
    AUGMENTATION_VERSION,
    ROTATION_ANGLES,
    AugmentationError,
    AxisRotation,
    RelationRewriteError,
    RotationAugmentation,
    all_rotations,
    build_rotation_augmentation,
    rewrite_relations,
)

SMOKE_ROOT = "data/smoke"

#: Deliberately not a cube, and not isotropic: a transform that confuses the
#: (z, y, x) array order with the (x, y, z) world order still looks right on a
#: cubic isotropic grid, and wrong here.
ODD_SHAPE = (6, 5, 4)  # (D, H, W), i.e. world sizes (N_x, N_y, N_z) = (4, 5, 6)
ODD_SPACING = (1.0, 2.0, 3.0)  # world units per voxel, (x, y, z)

CUBE_SHAPE = (32, 32, 32)
CUBE_CENTER = 15.5  # (32 - 1) / 2 at unit spacing


@pytest.fixture(scope="module")
def examples() -> ExampleDataset:
    try:
        return ExampleDataset(SMOKE_ROOT, "train", limit=12)
    except DatasetError as error:  # pragma: no cover - corpus not generated
        pytest.skip(f"smoke corpus unavailable: {error}")


def asymmetric_mask(shape=ODD_SHAPE) -> np.ndarray:
    """A blob with no symmetry, so every one of the 24 poses is distinguishable."""
    mask = np.zeros(shape, dtype=bool)
    mask[1:4, 0:2, 2:4] = True
    mask[0, shape[1] - 1, 0] = True
    return mask


def clauses(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"direction": direction, "anchor": anchor} for direction, anchor in pairs]


def directions_of(relations) -> list[str]:
    return [clause["direction"] for clause in relations]


def classify_cubic(target, anchor) -> str:
    """Direction of ``target`` relative to ``anchor`` on the 32^3 unit grid."""
    return classify_direction(
        target, anchor, volume_shape=CUBE_SHAPE, spacing=(1.0, 1.0, 1.0)
    ).direction


# ---------------------------------------------------------------------------
# Config guard
# ---------------------------------------------------------------------------
def test_no_augmentation_is_enabled_before_its_rewrite_is_tested():
    """Config must not enable an untested rewrite."""
    augmentations = load_config("train")["augmentations"]
    for name, settings in augmentations.items():
        if not isinstance(settings, dict):
            continue
        if settings.get("enabled"):
            assert settings.get("rewrite_tested") is True, (
                f"augmentation {name!r} is enabled but its relation rewrite is untested"
            )
    assert augmentations["arbitrary_rotation"]["allowed_in_milestone"] is False


def test_rotation_90_is_the_only_enabled_augmentation():
    """The other entries are still placeholders; none of them may be live."""
    augmentations = load_config("train")["augmentations"]
    enabled = {
        name
        for name, settings in augmentations.items()
        if isinstance(settings, dict) and settings.get("enabled")
    }
    assert enabled == {"rotation_90"}


# ---------------------------------------------------------------------------
# The rotation group
# ---------------------------------------------------------------------------
def test_all_rotations_is_the_24_element_octahedral_group():
    rotations = all_rotations()
    assert len(rotations) == 24
    assert rotations[0].is_identity, "the identity must come first"
    assert len({(r.perm, r.signs) for r in rotations}) == 24


@pytest.mark.parametrize("rotation", all_rotations(), ids=lambda r: r.code)
def test_every_sampled_transform_is_a_proper_rotation(rotation):
    """Determinant +1: a mirrored volume would silently swap left and right."""
    assert round(float(np.linalg.det(rotation.matrix()))) == 1


def test_the_group_is_closed_under_composition_and_inversion():
    rotations = all_rotations()
    keys = {(r.perm, r.signs) for r in rotations}
    for left, right in itertools.product(rotations, repeat=2):
        composed = left.compose(right)
        assert (composed.perm, composed.signs) in keys
    for rotation in rotations:
        assert rotation.compose(rotation.inverse()).is_identity
        assert rotation.inverse().compose(rotation).is_identity


def test_compose_matches_matrix_multiplication():
    rotations = all_rotations()
    for left, right in itertools.product(rotations[:8], rotations[:8]):
        expected = left.matrix() @ right.matrix()
        assert np.array_equal(left.compose(right).matrix(), expected)


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_four_quarter_turns_return_to_the_identity(axis):
    quarter = AxisRotation.about_axis(axis, 90)
    assert not quarter.is_identity
    assert quarter.compose(quarter).is_identity is False
    assert quarter.compose(quarter).compose(quarter).compose(quarter).is_identity
    assert AxisRotation.about_axis(axis, 360).is_identity
    assert AxisRotation.about_axis(axis, 0).is_identity


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_minus_90_is_the_inverse_of_90_and_the_same_as_270(axis):
    forward = AxisRotation.about_axis(axis, 90)
    backward = AxisRotation.about_axis(axis, -90)
    assert backward == AxisRotation.about_axis(axis, 270)
    assert forward.compose(backward).is_identity


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_180_is_a_quarter_turn_applied_twice_and_fixes_its_own_axis(axis):
    half = AxisRotation.about_axis(axis, 180)
    quarter = AxisRotation.about_axis(axis, 90)
    assert half == quarter.compose(quarter)
    assert half == half.inverse()
    # The axis turned about is the one left alone; the other two are negated.
    signs = dict(zip("xyz", half.signs))
    assert signs[axis] == 1
    assert [sign for name, sign in signs.items() if name != axis] == [-1, -1]


def test_from_angles_composes_z_after_y_after_x():
    """The angle triple to pose mapping is a contract, not an accident."""
    composed = AxisRotation.from_angles(90, 90, 90)
    expected = AxisRotation.about_axis("z", 90).compose(
        AxisRotation.about_axis("y", 90).compose(AxisRotation.about_axis("x", 90))
    )
    assert composed == expected
    assert AxisRotation.from_angles(0, 0, 0).is_identity


def test_a_reflection_is_rejected():
    with pytest.raises(AugmentationError, match="reflection"):
        AxisRotation(perm=(0, 1, 2), signs=(-1, 1, 1))
    with pytest.raises(AugmentationError, match="permutation"):
        AxisRotation(perm=(0, 0, 1), signs=(1, 1, 1))
    with pytest.raises(AugmentationError, match="multiple of 90"):
        AxisRotation.about_axis("x", 45)


def test_code_names_the_old_axis_each_new_axis_comes_from():
    assert AxisRotation.identity().code == "+x+y+z"
    # A quarter turn about x sends y onto z and z onto -y, so the new y axis is
    # the old -z and the new z axis is the old +y.
    assert AxisRotation.about_axis("x", 90).code == "+x-z+y"


# ---------------------------------------------------------------------------
# The voxel rotation and the world map must agree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rotation", all_rotations(), ids=lambda r: r.code)
def test_rotated_centroid_matches_the_transformed_centroid(rotation):
    """The load-bearing link between ``apply`` and ``transform_point_world``.

    The rewrite classifies transformed centroids; the model sees rotated voxels.
    If these two disagreed, every direction label would be quietly wrong.
    """
    mask = asymmetric_mask()
    rotated = rotation.apply(mask)
    rotated_spacing = rotation.transform_spacing(ODD_SPACING)

    expected = rotation.transform_point_world(
        centroid_world(mask, ODD_SPACING), volume_shape=ODD_SHAPE, spacing=ODD_SPACING
    )
    assert np.allclose(centroid_world(rotated, rotated_spacing), expected, atol=1e-9)


@pytest.mark.parametrize("rotation", all_rotations(), ids=lambda r: r.code)
def test_rotation_preserves_shape_extent_and_voxel_count(rotation):
    mask = asymmetric_mask()
    rotated = rotation.apply(mask)
    rotated_spacing = rotation.transform_spacing(ODD_SPACING)

    assert rotated.shape == rotation.transform_shape(ODD_SHAPE)
    assert int(rotated.sum()) == int(mask.sum()), "a rotation may not lose voxels"
    assert np.allclose(
        bbox_extent_world(rotated, rotated_spacing),
        rotation.transform_extent_world(bbox_extent_world(mask, ODD_SPACING)),
        atol=1e-9,
    )


@pytest.mark.parametrize("rotation", all_rotations(), ids=lambda r: r.code)
def test_applying_a_rotation_then_its_inverse_is_the_identity(rotation):
    mask = asymmetric_mask()
    assert np.array_equal(rotation.inverse().apply(rotation.apply(mask)), mask)


@pytest.mark.parametrize("rotation", all_rotations(), ids=lambda r: r.code)
def test_tensor_and_numpy_paths_agree(rotation):
    mask = asymmetric_mask().astype(np.float32)
    from_numpy = rotation.apply(mask)
    from_tensor = rotation.apply_tensor(torch.from_numpy(mask)).numpy()
    assert np.array_equal(from_numpy, from_tensor)
    assert from_tensor.flags["C_CONTIGUOUS"], "torch.from_numpy rejects negative strides"


def test_leading_axes_are_untouched():
    """Channel-stacked masks rotate as one, keeping channel order."""
    rotation = AxisRotation.about_axis("y", 90)
    stack = np.stack([asymmetric_mask(), ~asymmetric_mask()]).astype(np.float32)
    rotated = rotation.apply(stack)
    assert rotated.shape == (2,) + rotation.transform_shape(ODD_SHAPE)
    for channel in range(2):
        assert np.array_equal(rotated[channel], rotation.apply(stack[channel]))


def test_the_identity_changes_nothing():
    identity = AxisRotation.identity()
    mask = asymmetric_mask()
    assert np.array_equal(identity.apply(mask), mask)
    assert identity.transform_shape(ODD_SHAPE) == ODD_SHAPE
    assert identity.transform_spacing(ODD_SPACING) == ODD_SPACING
    point = (1.0, 2.0, 3.0)
    assert identity.transform_point_world(
        point, volume_shape=ODD_SHAPE, spacing=ODD_SPACING
    ) == point


def test_preserves_grid_is_false_when_a_rotation_reshapes_the_volume():
    """Batching needs one shape per split, so those poses must never be drawn."""
    swap_xy = AxisRotation.about_axis("z", 90)
    assert not swap_xy.preserves_grid(ODD_SHAPE, ODD_SPACING)
    assert AxisRotation.identity().preserves_grid(ODD_SHAPE, ODD_SPACING)
    assert swap_xy.preserves_grid(CUBE_SHAPE, (1.0, 1.0, 1.0))
    # Cubic but anisotropic: swapping x and y also swaps their spacing.
    assert not swap_xy.preserves_grid(CUBE_SHAPE, (1.0, 2.0, 1.0))


# ---------------------------------------------------------------------------
# What a rotation does to the six direction tokens
# ---------------------------------------------------------------------------
#: A legal triple whose three clauses sit on the three different world axes, so
#: one rewrite exercises all three rules at once. The x clause is `lateral`: the
#: target is farther from the centre plane than its anchor.
LATERAL_TARGET = (20.5, CUBE_CENTER, 19.5)
LATERAL_ANCHORS = (
    ("cube", (20.5, CUBE_CENTER, CUBE_CENTER)),      # separated on z -> superior
    ("sphere", (20.5, 9.5, 19.5)),                   # separated on y -> anterior
    ("torus", (13.5, CUBE_CENTER, 19.5)),            # separated on x -> lateral
)

#: The same, but the target is nearer the centre plane than its x anchor.
MEDIAL_TARGET = (17.5, CUBE_CENTER, 19.5)
MEDIAL_ANCHORS = (
    ("cube", (17.5, CUBE_CENTER, CUBE_CENTER)),
    ("sphere", (17.5, 9.5, 19.5)),
    ("torus", (7.5, CUBE_CENTER, 19.5)),
)


def triple(target, anchors):
    """The stored clause triple for a synthetic target/anchor layout."""
    return clauses(
        *((classify_cubic(target, centroid), name) for name, centroid in anchors)
    )


def rewrite(rotation, target, anchors):
    return rewrite_relations(
        rotation,
        relations=triple(target, anchors),
        target_centroid_world=target,
        anchor_centroids_world=[centroid for _, centroid in anchors],
        volume_shape=CUBE_SHAPE,
    )


def test_the_synthetic_triples_cover_all_three_rules():
    assert directions_of(triple(LATERAL_TARGET, LATERAL_ANCHORS)) == [
        SUPERIOR,
        ANTERIOR,
        LATERAL,
    ]
    assert directions_of(triple(MEDIAL_TARGET, MEDIAL_ANCHORS)) == [
        SUPERIOR,
        ANTERIOR,
        MEDIAL,
    ]


@pytest.mark.parametrize(
    ("turn_axis", "expected"),
    (
        # A half turn negates the two axes it does not turn about. Each signed
        # token flips exactly when its own axis is one of them.
        ("x", [INFERIOR, POSTERIOR, LATERAL]),
        ("y", [INFERIOR, ANTERIOR, LATERAL]),
        ("z", [SUPERIOR, POSTERIOR, LATERAL]),
    ),
)
def test_180_degrees_swaps_the_signed_pairs_but_not_medial_lateral(turn_axis, expected):
    """The precise sense in which a half turn "swaps opposite directions".

    superior/inferior and anterior/posterior read the *sign* of one component of
    ``delta``, so negating that axis flips them. medial/lateral instead compares
    distance to the centre plane, and a half turn reflects ``x`` through that
    same plane - both distances are preserved, so the token does not move. A
    rewrite that mapped every token to its opposite would corrupt this clause.
    """
    rotation = AxisRotation.about_axis(turn_axis, 180)
    assert directions_of(rewrite(rotation, LATERAL_TARGET, LATERAL_ANCHORS)) == expected


@pytest.mark.parametrize(
    ("turn_axis", "expected"),
    (
        ("x", [INFERIOR, POSTERIOR, MEDIAL]),
        ("y", [INFERIOR, ANTERIOR, MEDIAL]),
        ("z", [SUPERIOR, POSTERIOR, MEDIAL]),
    ),
)
def test_180_degrees_leaves_medial_alone_too(turn_axis, expected):
    rotation = AxisRotation.about_axis(turn_axis, 180)
    assert directions_of(rewrite(rotation, MEDIAL_TARGET, MEDIAL_ANCHORS)) == expected


def test_a_half_turn_is_its_own_inverse_on_the_clauses():
    rotation = AxisRotation.about_axis("x", 180)
    once = rewrite(rotation, LATERAL_TARGET, LATERAL_ANCHORS)
    twice = rewrite_relations(
        rotation,
        relations=once,
        target_centroid_world=rotation.transform_point_world(
            LATERAL_TARGET, volume_shape=CUBE_SHAPE
        ),
        anchor_centroids_world=[
            rotation.transform_point_world(centroid, volume_shape=CUBE_SHAPE)
            for _, centroid in LATERAL_ANCHORS
        ],
        volume_shape=CUBE_SHAPE,
    )
    assert twice == triple(LATERAL_TARGET, LATERAL_ANCHORS)


def test_90_degrees_moves_a_pair_between_the_two_rule_families():
    """Why no token-to-token table can exist: the destination needs new facts.

    Both pairs below are ``superior`` with the *same* ``delta_z``. A quarter turn
    about y brings z onto x, and there the answer is decided by which side of the
    centre plane each centroid landed on - something neither the token nor the
    delta carried. One becomes lateral, the other medial, so no function of the
    source token alone could produce both.
    """
    rotation = AxisRotation.about_axis("y", 90)
    assert rotation.perm[0] == 2, "this turn brings the old z axis onto x"

    pairs = {
        # (target, anchor); both separated by +3 along z
        "near_the_centre": ((CUBE_CENTER, CUBE_CENTER, 18.5), (CUBE_CENTER, CUBE_CENTER, CUBE_CENTER)),
        "far_from_the_centre": ((CUBE_CENTER, CUBE_CENTER, 9.5), (CUBE_CENTER, CUBE_CENTER, 6.5)),
    }
    outcomes = {}
    for name, (target, anchor) in pairs.items():
        assert classify_cubic(target, anchor) == SUPERIOR
        assert target[2] - anchor[2] == 3.0
        outcomes[name] = classify_cubic(
            rotation.transform_point_world(target, volume_shape=CUBE_SHAPE),
            rotation.transform_point_world(anchor, volume_shape=CUBE_SHAPE),
        )
    assert outcomes == {"near_the_centre": LATERAL, "far_from_the_centre": MEDIAL}


def test_the_rotations_reach_every_direction_token():
    """The point of the augmentation: one stored layout yields all six tokens.

    Each clause slot is re-derived independently, so a fixed layout that only
    ever said ``superior`` in slot 1 is seen saying five different things across
    the group, and the two triples together cover the whole vocabulary. Without
    this the augmentation would multiply the data without broadening it.
    """
    reached_per_slot = [set(), set(), set()]
    for target, anchors in (
        (LATERAL_TARGET, LATERAL_ANCHORS),
        (MEDIAL_TARGET, MEDIAL_ANCHORS),
    ):
        for rotation in all_rotations():
            try:
                rewritten = rewrite(rotation, target, anchors)
            except RelationRewriteError:
                continue
            for slot, clause in enumerate(rewritten):
                reached_per_slot[slot].add(clause["direction"])

    assert set().union(*reached_per_slot) == set(DIRECTION_OPPOSITES), (
        "the group must reach all six direction tokens"
    )
    for slot, reached in enumerate(reached_per_slot):
        assert len(reached) >= 5, f"slot {slot} only ever said {sorted(reached)}"


# ---------------------------------------------------------------------------
# Rejection: a direction is never invented
# ---------------------------------------------------------------------------
def test_a_rotation_that_collapses_two_clauses_is_rejected():
    """Two opposite z-pairs can both become medial; that triple is not legal."""
    target = (CUBE_CENTER, CUBE_CENTER, 18.5)
    anchors = [
        ("cube", (CUBE_CENTER, CUBE_CENTER, 5.5)),
        ("sphere", (CUBE_CENTER, CUBE_CENTER, 24.5)),
        ("torus", (CUBE_CENTER, 9.5, 18.5)),
    ]
    relations = clauses(
        *((classify_cubic(target, centroid), name) for name, centroid in anchors)
    )
    assert directions_of(relations) == [SUPERIOR, INFERIOR, ANTERIOR]

    with pytest.raises(RelationRewriteError, match="collapses the clause triple"):
        rewrite_relations(
            AxisRotation.about_axis("y", 90),
            relations=relations,
            target_centroid_world=target,
            anchor_centroids_world=[centroid for _, centroid in anchors],
            volume_shape=CUBE_SHAPE,
        )


def test_a_rotation_onto_an_exact_medial_lateral_tie_is_rejected():
    """A pair symmetric about the centre plane has no direction at all."""
    target = (CUBE_CENTER, CUBE_CENTER, CUBE_CENTER + 3.0)
    anchors = [
        ("cube", (CUBE_CENTER, CUBE_CENTER, CUBE_CENTER - 3.0)),
        ("sphere", (CUBE_CENTER, 9.5, CUBE_CENTER + 3.0)),
        ("torus", (3.5, CUBE_CENTER, CUBE_CENTER + 3.0)),
    ]
    relations = clauses(
        *((classify_cubic(target, centroid), name) for name, centroid in anchors)
    )
    with pytest.raises(RelationRewriteError, match="without a direction"):
        rewrite_relations(
            AxisRotation.about_axis("y", 90),
            relations=relations,
            target_centroid_world=target,
            anchor_centroids_world=[centroid for _, centroid in anchors],
            volume_shape=CUBE_SHAPE,
        )


def test_the_identity_rewrite_reproduces_the_stored_clauses(examples):
    """The fallback must always be legal, or `plan` would have no way out."""
    for record in examples.records:
        metadata = record.metadata
        rewritten = rewrite_relations(
            AxisRotation.identity(),
            relations=metadata.relations,
            target_centroid_world=metadata.target_centroid_world,
            anchor_centroids_world=metadata.anchor_centroids_world,
            volume_shape=metadata.volume_shape,
            spacing=metadata.spacing,
        )
        assert rewritten == [dict(clause) for clause in metadata.relations]
        assert render_prompt(rewritten) == metadata.prompt


def test_the_rewrite_keeps_the_anchors_and_their_order(examples):
    metadata = examples.records[0].metadata
    for rotation in all_rotations():
        try:
            rewritten = rewrite_relations(
                rotation,
                relations=metadata.relations,
                target_centroid_world=metadata.target_centroid_world,
                anchor_centroids_world=metadata.anchor_centroids_world,
                volume_shape=metadata.volume_shape,
                spacing=metadata.spacing,
            )
        except RelationRewriteError:
            continue
        assert [clause["anchor"] for clause in rewritten] == [
            clause["anchor"] for clause in metadata.relations
        ]
        assert len(set(directions_of(rewritten))) == 3


# ---------------------------------------------------------------------------
# Ground truth: the rewrite against directions re-derived from the voxels
# ---------------------------------------------------------------------------
def test_rewritten_directions_match_directions_recomputed_from_rotated_voxels(examples):
    """Rotate the masks, re-derive every direction, and require an exact match.

    This is the test the whole augmentation rests on. The rewrite only ever sees
    centroids from the manifest; the ground truth here comes from the rotated
    label volume, through the same canonical rule the generator used. They agree
    only if the array transform, the world-coordinate map, the centre plane and
    the spacing are all consistent with one another.
    """
    checked = 0
    rejected = 0
    for record in examples.records:
        metadata = record.metadata
        _, instance_labels = examples._scene_arrays(record)
        labels = np.asarray(instance_labels)

        for rotation in all_rotations():
            try:
                rewritten = rewrite_relations(
                    rotation,
                    relations=metadata.relations,
                    target_centroid_world=metadata.target_centroid_world,
                    anchor_centroids_world=metadata.anchor_centroids_world,
                    volume_shape=metadata.volume_shape,
                    spacing=metadata.spacing,
                )
            except RelationRewriteError:
                rejected += 1
                continue

            rotated = rotation.apply(labels)
            target_mask = rotated == metadata.target_instance_id
            truth = [
                direction_between_masks(
                    target_mask,
                    rotated == instance_id,
                    volume_shape=rotation.transform_shape(metadata.volume_shape),
                    spacing=rotation.transform_spacing(metadata.spacing),
                ).direction
                for instance_id in metadata.anchor_instance_ids
            ]
            assert directions_of(rewritten) == truth, (
                f"{metadata.example_id} under {rotation.code}: the rewrite said "
                f"{directions_of(rewritten)} but the rotated voxels say {truth}"
            )
            checked += 1

    assert checked > 0
    # The rewrite is accepted far more often than not; a corpus that rejected
    # almost everything would make the augmentation pointless without failing.
    assert rejected / (checked + rejected) < 0.25


def test_an_accepted_rewrite_is_never_ambiguous_on_the_voxels(examples):
    """Whatever `rewrite_relations` accepts, the voxel rule can also classify."""
    record = examples.records[0]
    metadata = record.metadata
    _, instance_labels = examples._scene_arrays(record)
    labels = np.asarray(instance_labels)

    for rotation in all_rotations():
        try:
            rewrite_relations(
                rotation,
                relations=metadata.relations,
                target_centroid_world=metadata.target_centroid_world,
                anchor_centroids_world=metadata.anchor_centroids_world,
                volume_shape=metadata.volume_shape,
                spacing=metadata.spacing,
            )
        except RelationRewriteError:
            continue
        rotated = rotation.apply(labels)
        target_mask = rotated == metadata.target_instance_id
        for instance_id in metadata.anchor_instance_ids:
            direction_between_masks(  # must not raise
                target_mask,
                rotated == instance_id,
                volume_shape=rotation.transform_shape(metadata.volume_shape),
                spacing=rotation.transform_spacing(metadata.spacing),
            )


def test_the_stored_centroids_agree_with_the_mask_centroids(examples):
    """The rewrite trusts the manifest's centroids; check they are the real ones."""
    for record in examples.records:
        metadata = record.metadata
        _, instance_labels = examples._scene_arrays(record)
        labels = np.asarray(instance_labels)
        assert np.allclose(
            centroid_world(labels == metadata.target_instance_id, metadata.spacing),
            metadata.target_centroid_world,
            atol=1e-9,
        )
        for instance_id, stored in zip(
            metadata.anchor_instance_ids, metadata.anchor_centroids_world
        ):
            assert np.allclose(
                centroid_world(labels == instance_id, metadata.spacing), stored, atol=1e-9
            )


# ---------------------------------------------------------------------------
# Sampling: determinism and coverage
# ---------------------------------------------------------------------------
def make_augmentation(**overrides) -> RotationAugmentation:
    settings = {"volume_shape": CUBE_SHAPE, "spacing": (1.0, 1.0, 1.0), "seed": 4321}
    settings.update(overrides)
    return RotationAugmentation(**settings)


def test_the_pose_is_a_pure_function_of_seed_epoch_and_index():
    augmentation = make_augmentation()
    first = [augmentation.sample(index, epoch=3).code for index in range(16)]
    # Same query again, and out of order: neither may change the answer.
    assert [augmentation.sample(index, epoch=3).code for index in range(16)] == first
    assert [
        augmentation.sample(index, epoch=3).code for index in reversed(range(16))
    ] == list(reversed(first))
    # A fresh object with the same seed reproduces the run exactly.
    assert [make_augmentation().sample(i, epoch=3).code for i in range(16)] == first


def test_a_different_seed_or_epoch_gives_a_different_pose_stream():
    augmentation = make_augmentation()
    epoch_3 = [augmentation.sample(index, epoch=3).code for index in range(24)]
    epoch_4 = [augmentation.sample(index, epoch=4).code for index in range(24)]
    other_seed = [make_augmentation(seed=99).sample(i, epoch=3).code for i in range(24)]
    assert epoch_3 != epoch_4, "the pose must change between epochs"
    assert epoch_3 != other_seed


def test_set_epoch_selects_the_stream_and_is_visible_to_workers():
    augmentation = make_augmentation()
    assert augmentation.epoch == 0
    augmentation.set_epoch(7)
    assert augmentation.epoch == 7
    assert [augmentation.sample(i).code for i in range(8)] == [
        augmentation.sample(i, epoch=7).code for i in range(8)
    ]
    # Shared memory, so a forked dataloader worker sees the main process's epoch
    # instead of freezing on its own copy.
    assert augmentation._epoch.is_shared()


def test_sampling_reaches_every_pose_and_stays_on_the_grid():
    augmentation = make_augmentation()
    seen = set()
    for epoch in range(40):
        for index in range(30):
            rotation = augmentation.sample(index, epoch=epoch)
            assert rotation.preserves_grid(CUBE_SHAPE, (1.0, 1.0, 1.0))
            seen.add(rotation.code)
    assert seen == {rotation.code for rotation in augmentation.rotations}
    assert len(seen) == 24


def test_the_identity_is_drawn_sometimes_so_the_stored_pose_is_still_seen():
    augmentation = make_augmentation()
    codes = [
        augmentation.sample(index, epoch=epoch).code
        for epoch in range(40)
        for index in range(30)
    ]
    assert "+x+y+z" in codes
    assert codes.count("+x+y+z") < len(codes) // 2, "the identity must not dominate"


def test_restricting_the_angles_or_axes_restricts_the_group():
    half_turns = make_augmentation(angles=(0, 180))
    assert len(half_turns.rotations) == 4  # identity plus the three 180s
    assert all(
        all(sign in (-1, 1) for sign in r.signs) and r.perm == (0, 1, 2)
        for r in half_turns.rotations
    )
    one_axis = make_augmentation(axes=("z",))
    assert len(one_axis.rotations) == 4  # the four quarter turns about z
    for index in range(20):
        assert one_axis.sample(index, epoch=1).signs[2] == 1


def test_a_grid_with_three_different_axes_keeps_only_the_half_turns():
    """Half turns never permute axes, so they survive any shape and spacing.

    The twelve quarter turns would reshape the volume and are dropped, leaving
    the identity and the three 180s - still a real augmentation, so this is not
    an error.
    """
    augmentation = RotationAugmentation(volume_shape=ODD_SHAPE, spacing=ODD_SPACING)
    assert {rotation.code for rotation in augmentation.rotations} == {
        "+x+y+z",
        "-x-y+z",
        "-x+y-z",
        "+x-y-z",
    }
    for index in range(20):
        rotation = augmentation.sample(index, epoch=0)
        assert rotation.perm == (0, 1, 2), "a quarter turn would reshape this volume"
        assert rotation.apply(asymmetric_mask()).shape == ODD_SHAPE


def test_an_augmentation_that_could_only_ever_be_the_identity_is_refused():
    """A silent no-op for a whole run is worse than a loud failure."""
    with pytest.raises(AugmentationError, match="reach only the identity"):
        make_augmentation(angles=(0,))
    with pytest.raises(AugmentationError, match="reach only the identity"):
        make_augmentation(angles=(0, 360))
    with pytest.raises(AugmentationError, match="multiples of 90"):
        make_augmentation(angles=(0, 45))
    with pytest.raises(AugmentationError, match="unknown axis"):
        make_augmentation(axes=("w",))
    with pytest.raises(AugmentationError, match="must not be empty"):
        make_augmentation(angles=())


# ---------------------------------------------------------------------------
# Planning one example
# ---------------------------------------------------------------------------
def metadata_stub(target, anchors, *, shape=CUBE_SHAPE, spacing=(1.0, 1.0, 1.0)):
    return SimpleNamespace(
        relations=clauses(
            *((classify_cubic(target, centroid), name) for name, centroid in anchors)
        ),
        target_centroid_world=target,
        anchor_centroids_world=[centroid for _, centroid in anchors],
        anchor_extents_world=[(1.0, 2.0, 3.0) for _ in anchors],
        volume_shape=shape,
        spacing=spacing,
    )


def test_a_plan_is_self_consistent(examples):
    augmentation = make_augmentation()
    for index, record in enumerate(examples.records):
        plan = augmentation.plan(record.metadata, index, epoch=2)
        assert plan.anchors == record.metadata.anchor_shape_names
        assert len(set(plan.directions)) == 3
        assert plan.prompt == render_prompt(plan.relations)
        assert parse_prompt(plan.prompt) == tuple(plan.relations)
        assert plan.volume_shape == record.metadata.volume_shape
        assert plan.spacing == record.metadata.spacing
        assert 1 <= plan.attempts <= augmentation.max_attempts
        # The plan's own centroids must classify to the directions it reports.
        for direction, anchor_centroid in zip(
            plan.directions, plan.anchor_centroids_world
        ):
            assert (
                classify_direction(
                    plan.target_centroid_world,
                    anchor_centroid,
                    volume_shape=plan.volume_shape,
                    spacing=plan.spacing,
                ).direction
                == direction
            )


#: A layout symmetric about the centre on all three axes: 8 of the 24 rotations
#: have no legal clause triple for it, so a short `max_attempts` will sometimes
#: run out and force the identity.
SYMMETRIC_TARGET = (CUBE_CENTER, CUBE_CENTER, CUBE_CENTER + 3.0)
SYMMETRIC_ANCHORS = (
    ("cube", (CUBE_CENTER, CUBE_CENTER, CUBE_CENTER - 3.0)),
    ("sphere", (CUBE_CENTER, CUBE_CENTER - 6.0, CUBE_CENTER + 3.0)),
    ("torus", (CUBE_CENTER - 9.0, CUBE_CENTER, CUBE_CENTER + 3.0)),
)


def test_the_symmetric_layout_really_does_reject_some_rotations():
    """Guards the fixture the fallback tests below depend on."""
    rejected = 0
    for rotation in all_rotations():
        try:
            rewrite(rotation, SYMMETRIC_TARGET, SYMMETRIC_ANCHORS)
        except RelationRewriteError:
            rejected += 1
    assert rejected == 8


def test_planning_falls_back_to_the_identity_when_every_draw_is_rejected():
    """A hopeless draw sequence still yields a trainable example, unaugmented."""
    metadata = metadata_stub(SYMMETRIC_TARGET, SYMMETRIC_ANCHORS)
    augmentation = make_augmentation(max_attempts=3)
    plans = [augmentation.plan(metadata, index, epoch=index) for index in range(50)]

    fallbacks = [plan for plan in plans if plan.fell_back_to_identity]
    assert fallbacks, "expected this layout to exhaust max_attempts at least once"
    for plan in fallbacks:
        assert plan.rotation.is_identity
        assert plan.attempts == augmentation.max_attempts
        assert len(plan.rejected) == augmentation.max_attempts
        # The stored prompt is reproduced exactly, so nothing is lost.
        assert plan.relations == tuple(dict(c) for c in metadata.relations)
        assert plan.prompt == render_prompt(metadata.relations)

    # Every plan, fallback or not, is still a legal triple.
    for plan in plans:
        assert len(set(plan.directions)) == 3
        assert plan.prompt == render_prompt(plan.relations)


def test_a_drawn_identity_is_not_reported_as_a_fallback():
    """`fell_back_to_identity` means exhausted, not merely "pose unchanged".

    The identity is one of the legal draws, so an example whose first draw was
    rejected and whose second was the identity has been augmented exactly as
    intended. Counting it as a fallback would overstate how often the
    augmentation gives up.
    """
    metadata = metadata_stub(SYMMETRIC_TARGET, SYMMETRIC_ANCHORS)
    augmentation = make_augmentation(max_attempts=3)
    plans = [augmentation.plan(metadata, index, epoch=index) for index in range(50)]

    drawn_identity = [
        plan
        for plan in plans
        if plan.rotation.is_identity and len(plan.rejected) < augmentation.max_attempts
    ]
    assert drawn_identity, "expected the identity to be drawn after a rejection"
    for plan in drawn_identity:
        assert not plan.fell_back_to_identity
        assert not plan.exhausted

    # The counter and the per-plan flag must tell the same story.
    assert augmentation.stats["fallbacks"] == sum(
        plan.fell_back_to_identity for plan in plans
    )


def test_rejections_are_counted_rather_than_hidden(examples):
    augmentation = make_augmentation()
    for index, record in enumerate(examples.records):
        for epoch in range(6):
            augmentation.plan(record.metadata, index, epoch=epoch)
    assert augmentation.stats["planned"] == len(examples.records) * 6
    assert augmentation.stats["rejected"] >= 0
    assert "rejected draws per example" in augmentation.describe()


def test_plan_is_deterministic_for_the_same_index_and_epoch(examples):
    metadata = examples.records[0].metadata
    first = make_augmentation().plan(metadata, 5, epoch=2)
    second = make_augmentation().plan(metadata, 5, epoch=2)
    assert first.rotation == second.rotation
    assert first.relations == second.relations
    assert first.prompt == second.prompt


# ---------------------------------------------------------------------------
# Stage B dataset integration
# ---------------------------------------------------------------------------
def stage_b_dataset(**overrides) -> ExampleDataset:
    settings = {"root": SMOKE_ROOT, "split": "train", "limit": 6}
    settings.update(overrides)
    try:
        return ExampleDataset(**settings)
    except DatasetError as error:  # pragma: no cover
        pytest.skip(f"smoke corpus unavailable: {error}")


def test_an_augmented_item_agrees_with_its_own_masks():
    """The end-to-end guarantee: prompt, ids and voxels describe one pose.

    Nothing here consults the plan. The directions are re-derived from the very
    tensors the model is handed, so a mismatch between what the network sees and
    what the prompt says would fail right here.
    """
    dataset = stage_b_dataset(augment=make_augmentation())
    for epoch in range(3):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            item = dataset[index]
            metadata = dataset.records[index].metadata
            target_mask = item["target_mask"][0].numpy() > 0.5
            truth = [
                direction_between_masks(
                    target_mask,
                    item["anchor_masks"][channel].numpy() > 0.5,
                    volume_shape=tuple(target_mask.shape),
                    spacing=metadata.spacing,
                ).direction
                for channel in range(3)
            ]
            assert item["directions"] == truth
            assert parse_prompt(item["prompt"]) == tuple(
                {"direction": direction, "anchor": anchor}
                for direction, anchor in zip(truth, item["anchor_shape_names"])
            )


def test_direction_ids_follow_the_rewritten_directions():
    from src.data.prompt_generator import clause_indices

    dataset = stage_b_dataset(augment=make_augmentation())
    dataset.set_epoch(1)
    for index in range(len(dataset)):
        item = dataset[index]
        expected, shape_ids = clause_indices(
            [
                {"direction": direction, "anchor": anchor}
                for direction, anchor in zip(item["directions"], item["anchor_shape_names"])
            ]
        )
        assert item["direction_ids"].tolist() == expected
        assert item["anchor_shape_ids"].tolist() == shape_ids


def test_augmentation_preserves_anchors_target_and_mask_volumes():
    plain = stage_b_dataset()
    augmented = stage_b_dataset(augment=make_augmentation())
    augmented.set_epoch(2)
    for index in range(len(plain)):
        before, after = plain[index], augmented[index]
        assert after["example_id"] == before["example_id"]
        assert after["target_shape_name"] == before["target_shape_name"]
        # Anchors and their order survive; only the directions move.
        assert after["anchor_shape_names"] == before["anchor_shape_names"]
        assert after["anchor_masks"].shape == before["anchor_masks"].shape
        assert after["target_mask"].sum() == before["target_mask"].sum()
        assert torch.equal(
            after["anchor_masks"].sum(dim=(1, 2, 3)),
            before["anchor_masks"].sum(dim=(1, 2, 3)),
        )
        # The target must stay out of every anchor channel.
        assert float((after["anchor_masks"] * after["target_mask"]).sum()) == 0.0


def test_the_pose_changes_across_epochs_but_not_within_one():
    dataset = stage_b_dataset(augment=make_augmentation())
    codes_by_epoch = []
    for epoch in range(4):
        dataset.set_epoch(epoch)
        codes = [dataset[index]["rotation"] for index in range(len(dataset))]
        assert codes == [dataset[index]["rotation"] for index in range(len(dataset))]
        codes_by_epoch.append(codes)
    assert len({tuple(codes) for codes in codes_by_epoch}) > 1, (
        "the same volume must be seen in a different pose each epoch"
    )


def test_an_unaugmented_dataset_is_untouched():
    dataset = stage_b_dataset()
    assert dataset.augment is None
    dataset.set_epoch(5)  # a no-op, and must not raise
    for index in range(len(dataset)):
        item = dataset[index]
        metadata = dataset.records[index].metadata
        assert item["rotation"] == ""
        assert item["prompt"] == metadata.prompt
        assert item["directions"] == list(metadata.structured_prompt.directions)
        assert item["anchor_shape_names"] == list(metadata.anchor_shape_names)


def test_the_scene_volume_channel_rotates_with_the_masks():
    """Stage B's optional scene input must stay registered with its labels."""
    dataset = stage_b_dataset(limit=2, include_scene_volume=True, augment=make_augmentation())
    dataset.set_epoch(3)
    for index in range(len(dataset)):
        item = dataset[index]
        occupancy = item["scene_volume"][0].numpy() > 0
        union = item["anchor_masks"].numpy().max(axis=0) > 0.5
        target = item["target_mask"][0].numpy() > 0.5
        # Every labelled voxel is occupied in the rotated scene volume.
        assert np.all(occupancy[union])
        assert np.all(occupancy[target])


def test_rotation_is_not_a_model_input():
    from src.data.dataset import STAGE_B_NON_INPUT_KEYS, stage_b_model_inputs

    assert "rotation" in STAGE_B_NON_INPUT_KEYS
    dataset = stage_b_dataset(limit=1, augment=make_augmentation())
    inputs = stage_b_model_inputs(dataset[0])
    assert "rotation" not in inputs


# ---------------------------------------------------------------------------
# Stage A dataset integration
# ---------------------------------------------------------------------------
def test_stage_a_rotates_the_scene_and_every_mask_together():
    """No relations to rewrite, but the masks must stay the masks of their shapes."""
    try:
        plain = SceneDataset(SMOKE_ROOT, "train", limit=2)
    except DatasetError as error:  # pragma: no cover
        pytest.skip(f"smoke corpus unavailable: {error}")
    augmented = SceneDataset(
        SMOKE_ROOT, "train", limit=2, augment=make_augmentation(volume_shape=plain.volume_shape)
    )
    augmented.set_epoch(1)

    for index in range(len(plain)):
        before, after = plain[index], augmented[index]
        rotation = AxisRotation.identity()
        for candidate in all_rotations():
            if candidate.code == after["rotation"]:
                rotation = candidate
                break
        assert rotation.code == after["rotation"]
        assert torch.equal(
            after["scene_volume"], rotation.apply_tensor(before["scene_volume"])
        )
        assert torch.equal(
            after["target_masks"], rotation.apply_tensor(before["target_masks"])
        )
        assert torch.equal(after["prompt_ids"], before["prompt_ids"])


def test_stage_a_reports_its_spacing_and_refuses_a_mixed_split():
    try:
        dataset = SceneDataset(SMOKE_ROOT, "train", limit=2)
    except DatasetError as error:  # pragma: no cover
        pytest.skip(f"smoke corpus unavailable: {error}")
    assert len(dataset.spacing) == 3
    assert all(value > 0 for value in dataset.spacing)
    assert dataset[0]["rotation"] == ""


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------
def augmentation_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "rotation_90": {
            "enabled": True,
            "rewrite_tested": True,
            "angles": [0, 90, 180, -90],
            "axes": ["x", "y", "z"],
            "max_attempts": 8,
            "apply_to": {"stage_a": False, "stage_b": True},
        },
    }
    config["rotation_90"].update(overrides)
    return config


def build(stage="stage_b", **overrides):
    return build_rotation_augmentation(
        augmentation_config(**overrides),
        volume_shape=CUBE_SHAPE,
        spacing=(1.0, 1.0, 1.0),
        seed=7,
        stage=stage,
    )


def test_the_config_block_builds_the_augmentation_it_describes():
    augmentation = build()
    assert isinstance(augmentation, RotationAugmentation)
    assert augmentation.angles == ROTATION_ANGLES
    assert augmentation.axes == ("x", "y", "z")
    assert augmentation.max_attempts == 8
    assert len(augmentation.rotations) == 24


def test_apply_to_gates_each_stage_independently():
    assert build(stage="stage_b") is not None
    assert build(stage="stage_a") is None
    assert build(stage="stage_a", apply_to={"stage_a": True, "stage_b": True}) is not None


def test_the_augmentation_is_off_unless_both_switches_are_on():
    assert build_rotation_augmentation({"enabled": False}, volume_shape=CUBE_SHAPE) is None
    assert build(enabled=False) is None


def test_an_untested_rewrite_cannot_be_enabled_through_the_config():
    with pytest.raises(AugmentationError, match="rewrite_tested"):
        build(rewrite_tested=False)


def test_the_shipped_config_builds_and_targets_stage_b_only():
    config = load_config("train")["augmentations"]
    stage_b = build_rotation_augmentation(
        config, volume_shape=CUBE_SHAPE, spacing=(1.0, 1.0, 1.0), seed=0, stage="stage_b"
    )
    stage_a = build_rotation_augmentation(
        config, volume_shape=CUBE_SHAPE, spacing=(1.0, 1.0, 1.0), seed=0, stage="stage_a"
    )
    assert isinstance(stage_b, RotationAugmentation)
    assert stage_a is None, "Stage A augmentation is opt-in, not the default"
    assert AUGMENTATION_VERSION


# ---------------------------------------------------------------------------
# Dataloader and trainer plumbing
# ---------------------------------------------------------------------------
def test_set_epoch_reaches_persistent_dataloader_workers():
    """The reason the epoch counter lives in shared memory.

    A persistent worker holds its own copy of the dataset. If the counter were a
    plain ``int`` it would freeze at epoch 0 in every worker while the main
    process advanced, and the run would silently train on one pose forever. The
    multiprocess stream must also equal the single-process one, or a run would
    stop being reproducible from its seed the moment ``num_workers`` changed.
    """
    from src.data.dataset import build_example_dataloader
    from src.training.trainer import _set_loader_epoch

    def fresh() -> ExampleDataset:
        return stage_b_dataset(
            limit=4, augment=make_augmentation(volume_shape=(64, 64, 64), seed=5)
        )

    loader = build_example_dataloader(fresh(), batch_size=2, num_workers=2)
    try:
        by_epoch = []
        for epoch in range(3):
            _set_loader_epoch(loader, epoch)
            codes: list[str] = []
            for batch in loader:
                codes.extend(batch["rotation"])
            by_epoch.append(codes)
    finally:
        del loader

    assert len({tuple(codes) for codes in by_epoch}) == 3, (
        "workers kept serving the same pose; the epoch never reached them"
    )

    reference = fresh()
    for epoch, codes in enumerate(by_epoch):
        reference.set_epoch(epoch)
        assert codes == [reference[index]["rotation"] for index in range(len(reference))]


def test_set_loader_epoch_is_a_no_op_without_an_augmentation():
    from src.data.dataset import build_example_dataloader
    from src.training.trainer import _set_loader_epoch

    loader = build_example_dataloader(stage_b_dataset(limit=2), batch_size=1)
    _set_loader_epoch(loader, 3)  # must not raise
    assert all(item["rotation"] == "" for item in (loader.dataset[0], loader.dataset[1]))


def test_the_augmentation_version_is_recorded_in_run_provenance():
    """A checkpoint must say which rewrite produced it, not just which settings."""
    from src.provenance import version_metadata

    assert version_metadata()["augmentation_version"] == AUGMENTATION_VERSION
