"""Tests for nearest-feasible anchor selection and scene generation.

Covers:
* candidates ranked by Euclidean centroid distance, instance ID as tie-break;
* the nearest FEASIBLE triple is chosen, skipping candidates whose direction is
  already used - so the selection is not always the three literally nearest
  objects;
* the three directions are always distinct;
* a target with no feasible triple, or a scene with an ambiguous pair, is
  rejected rather than patched;
* anchor order is ascending distance and is identical in the mask channels, the
  structured prompt and the rendered text;
* packing produces exactly ten in-bounds, non-overlapping objects, and a scene
  is reproducible from its seed alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.direction_rules import AmbiguousDirectionError, centroid_world
from src.data.primitives import SHAPE_VOCABULARY
from src.data.prompt_generator import (
    AnchorSelectionError,
    clauses_from_anchors,
    rank_candidates,
    select_anchors,
)
from src.data.scene_generator import (
    GeneratorSettings,
    build_examples,
    generate_examples,
    generate_scene,
    pack_scene,
    placement_order,
    sample_shape_params,
)
from src.data.validation import RejectionLog, verify_relations_against_masks

VOLUME_SHAPE = (64, 64, 64)
NAMES = {index: SHAPE_VOCABULARY.id_to_name(index) for index in range(1, 11)}


def select(centroids, target=1):
    return select_anchors(target, centroids, NAMES, volume_shape=VOLUME_SHAPE)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def test_candidates_are_ranked_by_centroid_distance():
    centroids = {
        1: (32.0, 32.0, 32.0),
        2: (32.0, 32.0, 52.0),
        3: (32.0, 32.0, 37.0),
        4: (32.0, 42.0, 32.0),
    }
    assert rank_candidates(1, centroids) == (3, 4, 2)


def test_equal_distances_are_broken_by_instance_id():
    centroids = {
        1: (32.0, 32.0, 32.0),
        5: (32.0, 32.0, 42.0),
        2: (32.0, 42.0, 32.0),
        7: (42.0, 32.0, 32.0),
    }
    assert rank_candidates(1, centroids) == (2, 5, 7)


def test_ranking_rejects_an_unknown_target():
    with pytest.raises(AnchorSelectionError):
        rank_candidates(99, {1: (0.0, 0.0, 0.0)})


# ---------------------------------------------------------------------------
# Nearest-feasible selection
# ---------------------------------------------------------------------------
def test_the_selected_set_is_the_nearest_feasible_not_the_three_nearest():
    # Candidates 2, 3 and 4 are all superior to the target and are the three
    # nearest; only the first of them is usable, so 5 and 6 must be pulled in.
    centroids = {
        1: (32.0, 32.0, 32.0),   # target
        2: (32.0, 32.0, 22.0),   # d=10, superior  -> kept
        3: (32.0, 32.0, 21.0),   # d=11, superior  -> skipped
        4: (32.0, 32.0, 19.0),   # d=13, superior  -> skipped
        5: (32.0, 20.0, 32.0),   # d=12, anterior  -> kept
        6: (18.0, 32.0, 32.0),   # d=14, medial    -> kept
    }
    anchors = select(centroids)
    assert [anchor.instance_id for anchor in anchors] == [2, 5, 6]
    assert [anchor.direction for anchor in anchors] == ["superior", "anterior", "medial"]
    # Skipped candidates were strictly nearer than the third selected anchor.
    assert anchors[2].distance > 11.0


def test_anchors_are_returned_in_ascending_distance_order():
    centroids = {
        1: (32.0, 32.0, 32.0),
        2: (32.0, 32.0, 22.0),
        5: (32.0, 20.0, 32.0),
        6: (18.0, 32.0, 32.0),
    }
    anchors = select(centroids)
    distances = [anchor.distance for anchor in anchors]
    assert distances == sorted(distances)


def test_selected_directions_are_always_distinct():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(300):
        centroids = {
            index: tuple(float(v) for v in rng.integers(4, 60, size=3))
            for index in range(1, 11)
        }
        try:
            anchors = select(centroids)
        except (AnchorSelectionError, AmbiguousDirectionError):
            continue
        directions = [anchor.direction for anchor in anchors]
        assert len(set(directions)) == 3
        assert len({anchor.instance_id for anchor in anchors}) == 3
        checked += 1
    assert checked > 150


def test_a_target_without_three_distinct_directions_is_rejected():
    centroids = {
        1: (32.0, 32.0, 32.0),
        2: (32.0, 32.0, 22.0),   # superior
        3: (32.0, 32.0, 20.0),   # superior
        4: (32.0, 20.0, 32.0),   # anterior
    }
    with pytest.raises(AnchorSelectionError):
        select(centroids)


def test_an_ambiguous_pair_anywhere_in_the_scene_is_raised_not_hidden():
    # Instance 9 is exactly as far from the centre plane as the target, on the
    # other side. It would never be selected, but the rule still refuses to
    # invent a direction for it.
    centroids = {
        1: (20.0, 32.0, 32.0),   # target; |20 - 31.5| = 11.5
        2: (20.0, 32.0, 22.0),   # superior
        3: (20.0, 20.0, 32.0),   # anterior
        4: (20.0, 32.0, 42.0),   # inferior
        9: (43.0, 32.0, 32.0),   # |43 - 31.5| = 11.5 -> tie on the x axis
    }
    with pytest.raises(AmbiguousDirectionError):
        select(centroids)


def test_clauses_follow_the_anchor_order():
    centroids = {
        1: (32.0, 32.0, 32.0),
        2: (32.0, 32.0, 22.0),
        5: (32.0, 20.0, 32.0),
        6: (18.0, 32.0, 32.0),
    }
    anchors = select(centroids)
    clauses = clauses_from_anchors(anchors)
    assert [clause["anchor"] for clause in clauses] == [anchor.shape_name for anchor in anchors]
    assert [clause["direction"] for clause in clauses] == [anchor.direction for anchor in anchors]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def settings():
    return GeneratorSettings.from_config()


def test_placement_order_puts_the_largest_classes_first():
    order = placement_order()
    assert len(order) == 10
    volumes = [spec.mean_volume_voxels for spec in order]
    assert volumes == sorted(volumes, reverse=True)


def test_sampled_sizes_stay_inside_the_configured_ranges(settings):
    rng = np.random.default_rng(3)
    for spec in SHAPE_VOCABULARY:
        for _ in range(20):
            params = sample_shape_params(spec, rng)
            for name, value in params.items():
                low, high = spec.param_range(name)
                assert low <= value <= high


def test_anisotropic_classes_never_degenerate_into_their_isotropic_twin():
    rng = np.random.default_rng(5)
    for name in ("cuboid", "ellipsoid"):
        spec = SHAPE_VOCABULARY.by_name(name)
        for _ in range(50):
            values = list(sample_shape_params(spec, rng).values())
            assert max(values) / min(values) >= spec.min_anisotropy_ratio


def test_packing_places_exactly_ten_non_overlapping_in_bounds_objects(settings):
    rng = np.random.default_rng([42, 0, 0])
    labels, params = pack_scene(rng, settings, 1.0)
    assert np.array_equal(np.unique(labels), np.arange(0, 11))
    assert set(params) == set(SHAPE_VOCABULARY.names)
    margin = settings.margin_voxels
    for axis in range(3):
        occupied = np.nonzero((labels != 0).any(axis=tuple(i for i in range(3) if i != axis)))[0]
        assert occupied.min() >= margin
        assert occupied.max() < labels.shape[axis] - margin
    # One label per voxel is overlap-freedom by construction; check the count.
    assert int((labels != 0).sum()) == sum(
        int((labels == spec.id).sum()) for spec in SHAPE_VOCABULARY
    )


def test_a_scene_is_reproducible_from_its_seed(settings):
    first, _ = pack_scene(np.random.default_rng([7, 0, 0]), settings, 1.0)
    second, _ = pack_scene(np.random.default_rng([7, 0, 0]), settings, 1.0)
    assert np.array_equal(first, second)


def test_escalation_grids_are_configured_and_rescale_sizes(settings):
    assert settings.stage_volume_shapes[0] == settings.volume_shape
    assert settings.escalation_volume_shapes == ((72, 72, 72), (80, 80, 80))
    escalated = settings.replace_volume_shape((80, 80, 80))
    assert escalated.volume_shape == (80, 80, 80)
    assert escalated.spacing == settings.spacing
    spec = SHAPE_VOCABULARY.by_name("cube")
    rng = np.random.default_rng(1)
    scaled = sample_shape_params(spec, rng, scale=80 / 64)
    low, high = spec.param_range("side")
    assert low * 80 / 64 <= scaled["side"] <= high * 80 / 64


# ---------------------------------------------------------------------------
# End-to-end examples
# ---------------------------------------------------------------------------
def test_examples_from_a_generated_scene_agree_with_their_masks(settings):
    # generate_scene retries until every one of the ten targets is feasible;
    # build_examples on a raw packing may legitimately raise, which is what
    # drives the scene-level rejection.
    scene = generate_scene(1000000, settings=settings, split="train")
    examples = scene.examples
    assert len(examples) == 10
    seen_targets = set()
    for example in examples:
        metadata = example.metadata
        seen_targets.add(metadata.target_shape_name)
        # Independent re-derivation from the stored arrays.
        verify_relations_against_masks(example)
        assert metadata.anchor_shape_names == metadata.structured_prompt.anchors
        assert metadata.prompt == metadata.structured_prompt.render()
        assert metadata.target_instance_id not in metadata.anchor_instance_ids
        assert len(set(metadata.structured_prompt.directions)) == 3
        target_centroid = centroid_world(example.arrays.target_mask, metadata.spacing)
        distances = [
            float(np.linalg.norm(centroid_world(channel, metadata.spacing) - target_centroid))
            for channel in example.arrays.anchor_masks
        ]
        assert distances == sorted(distances)
    assert seen_targets == set(SHAPE_VOCABULARY.names)


def test_generate_examples_filters_by_target_class(settings):
    log = RejectionLog()
    kept = list(
        generate_examples(
            1000000,
            settings=settings,
            split="val",
            target_classes=["cuboid", "ellipsoid"],
            log=log,
        )
    )
    assert {example.metadata.target_shape_name for example in kept} == {"cuboid", "ellipsoid"}
    assert len(kept) == 2
    assert log.accepted == 1
    assert log.acceptance_rate > 0.0


def test_build_examples_raises_when_a_target_has_no_feasible_triple(settings):
    """A raw packing may be infeasible; that raise is what rejects the scene."""
    labels, _ = pack_scene(np.random.default_rng([42, 0, 0]), settings, 1.0)
    with pytest.raises(AnchorSelectionError):
        build_examples("scene_test", 42, labels, settings, split="train")


def test_generate_scene_retries_until_every_target_is_feasible(settings):
    log = RejectionLog()
    scene = generate_scene(42, settings=settings, split="train", log=log)
    assert len(scene.examples) == 10
    assert scene.attempts >= 1
    assert log.accepted == 1
    # The first packing at this seed is infeasible, so a retry must have happened.
    assert scene.attempts > 1
    assert log.rejected
