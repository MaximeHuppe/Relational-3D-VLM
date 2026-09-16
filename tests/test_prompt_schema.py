"""Tests for the prompt renderer/parser and the dataset schema.

Covers the round-trip guarantee between the structured representation and the
natural-language prompt, the closed vocabularies, anchor-order identity between
channels/prompt/text, and the Stage B input contract.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from src.config import load_config
from src.data.direction_rules import (
    DIRECTION_RULE_VERSION,
    DIRECTIONS,
    bbox_extent_world,
    centroid_world,
)
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY, VOCABULARY_VERSION
from src.data.prompt_generator import (
    PROMPT_TEMPLATE,
    PromptError,
    assert_no_target_leakage,
    iter_all_clause_pairs,
    parse_prompt,
    render_prompt,
    round_trips,
    validate_clauses,
)
from src.data.schema import (
    NUM_ANCHORS,
    REQUIRED_EXAMPLE_FIELDS,
    SCHEMA_VERSION,
    STAGE_B_ALLOWED_FIELDS,
    STAGE_B_FORBIDDEN_FIELDS,
    Example,
    ExampleArrays,
    ExampleMetadata,
    Relation,
    SchemaError,
    StructuredPrompt,
    assert_complete,
    read_manifest,
    stage_b_inputs,
    write_manifest,
)
from src.data.validation import ValidationError, validate_example, validate_split_assignment

CLAUSES = [
    {"direction": "lateral", "anchor": "cube"},
    {"direction": "superior", "anchor": "pyramid"},
    {"direction": "anterior", "anchor": "sphere"},
]


# ---------------------------------------------------------------------------
# Prompt rendering and parsing
# ---------------------------------------------------------------------------
def test_canonical_rendering_matches_the_specified_sentence():
    assert render_prompt(CLAUSES) == (
        "segment the shape that is lateral to the cube, superior to the pyramid, "
        "and anterior to the sphere."
    )
    assert PROMPT_TEMPLATE.startswith("segment the shape that is ")


def test_structured_to_text_to_structured_round_trips_for_every_direction_triple():
    anchors = ("cube", "sphere", "torus")
    checked = 0
    for directions in itertools.permutations(DIRECTIONS, 3):
        clauses = [
            {"direction": direction, "anchor": anchor}
            for direction, anchor in zip(directions, anchors)
        ]
        prompt = render_prompt(clauses)
        assert parse_prompt(prompt) == tuple(clauses)
        assert round_trips(clauses, prompt)
        checked += 1
    assert checked == 120


def test_round_trip_holds_for_every_anchor_name():
    for anchor in SHAPE_NAMES:
        others = [name for name in SHAPE_NAMES if name != anchor][:2]
        clauses = [
            {"direction": "medial", "anchor": anchor},
            {"direction": "superior", "anchor": others[0]},
            {"direction": "posterior", "anchor": others[1]},
        ]
        assert parse_prompt(render_prompt(clauses)) == tuple(clauses)


def test_parser_normalises_the_spaced_form_of_a_multiword_shape_name():
    spaced = (
        "segment the shape that is medial to the triangular prism, "
        "inferior to the torus, and posterior to the capsule."
    )
    parsed = parse_prompt(spaced)
    assert parsed[0]["anchor"] == "triangular_prism"
    assert render_prompt(parsed) == spaced.replace("triangular prism", "triangular_prism")


@pytest.mark.parametrize(
    "prompt",
    [
        # synonyms and ordinals are outside the closed vocabulary
        "segment the shape that is above the cube, superior to the pyramid, and anterior to the sphere.",
        "segment the shape that is lateral to the first cube, superior to the pyramid, and anterior to the sphere.",
        # unknown anchor name
        "segment the shape that is lateral to the blob, superior to the pyramid, and anterior to the sphere.",
        # wrong template
        "Segment the shape lateral to the cube, superior to the pyramid, anterior to the sphere.",
        # missing terminal period
        "segment the shape that is lateral to the cube, superior to the pyramid, and anterior to the sphere",
        # only two clauses
        "segment the shape that is lateral to the cube, and superior to the pyramid.",
    ],
)
def test_parser_rejects_anything_outside_the_canonical_closed_vocabulary(prompt):
    with pytest.raises(PromptError):
        parse_prompt(prompt)


def test_clause_validation_enforces_three_distinct_directions_and_anchors():
    with pytest.raises(PromptError):
        validate_clauses(CLAUSES[:2])
    repeated_direction = [dict(clause) for clause in CLAUSES]
    repeated_direction[1]["direction"] = "lateral"
    with pytest.raises(PromptError):
        validate_clauses(repeated_direction)
    repeated_anchor = [dict(clause) for clause in CLAUSES]
    repeated_anchor[1]["anchor"] = "cube"
    with pytest.raises(PromptError):
        validate_clauses(repeated_anchor)
    with pytest.raises(PromptError):
        validate_clauses([{"direction": "lateral", "anchor": "cube", "slot": 1}] * 3)


def test_target_shape_name_never_leaks_into_the_prompt():
    prompt = render_prompt(CLAUSES)
    assert_no_target_leakage(prompt, "cylinder")
    with pytest.raises(PromptError):
        assert_no_target_leakage(prompt, "cube")


def test_pair_vocabulary_has_one_entry_per_direction_and_shape():
    pairs = list(iter_all_clause_pairs())
    assert len(pairs) == len(DIRECTIONS) * len(SHAPE_NAMES) == 60
    assert len(set(pairs)) == 60


# ---------------------------------------------------------------------------
# Structured prompt objects
# ---------------------------------------------------------------------------
def test_structured_prompt_is_the_primary_interface():
    structured = StructuredPrompt.from_list(CLAUSES)
    assert structured.to_list() == CLAUSES
    assert structured.directions == ("lateral", "superior", "anterior")
    assert structured.anchors == ("cube", "pyramid", "sphere")
    assert StructuredPrompt.from_text(structured.render()) == structured


def test_relation_rejects_tokens_outside_the_vocabularies():
    assert Relation("medial", "triangular prism").anchor == "triangular_prism"
    with pytest.raises(SchemaError):
        Relation("above", "cube")
    with pytest.raises(SchemaError):
        Relation("medial", "blob")


def test_structured_prompt_requires_exactly_three_clauses():
    with pytest.raises(SchemaError):
        StructuredPrompt(relations=tuple(Relation.from_dict(c) for c in CLAUSES[:2]))


def test_clause_permutation_helper_keeps_the_clause_set():
    structured = StructuredPrompt.from_list(CLAUSES)
    permuted = structured.with_permuted_clauses([2, 0, 1])
    assert permuted.anchors == ("sphere", "cube", "pyramid")
    assert set(permuted.to_list()[0].items()) == set(CLAUSES[2].items())
    with pytest.raises(SchemaError):
        structured.with_permuted_clauses([0, 0, 1])


# ---------------------------------------------------------------------------
# Example schema
# ---------------------------------------------------------------------------
SHAPE = (16, 16, 16)


def build_scene():
    """A tiny synthetic scene: ten disjoint 3-voxel cubes, labels 1..10."""
    labels = np.zeros(SHAPE, dtype=np.uint8)
    for index in range(10):
        z = 1 + (index // 4) * 5
        y = 1 + (index % 4) * 3
        x = 1 + (index % 3) * 4
        labels[z : z + 3, y : y + 3, x : x + 3] = index + 1
    scene = (labels > 0).astype(np.uint8)
    return scene, labels


def build_metadata(**overrides):
    _, labels = build_scene()
    anchor_ids = (1, 7, 3)  # cube, pyramid, sphere
    fields = dict(
        scene_id="scene_000000",
        example_id="scene_000000_target_05",
        seed=1000000,
        volume_shape=SHAPE,
        spacing=(1.0, 1.0, 1.0),
        target_instance_id=5,
        target_shape_name="cylinder",
        target_centroid_world=tuple(centroid_world(labels == 5)),
        anchor_instance_ids=anchor_ids,
        anchor_shape_names=("cube", "pyramid", "sphere"),
        anchor_centroids_world=tuple(tuple(centroid_world(labels == i)) for i in anchor_ids),
        anchor_extents_world=tuple(tuple(bbox_extent_world(labels == i)) for i in anchor_ids),
        structured_prompt=StructuredPrompt.from_list(CLAUSES),
        prompt=render_prompt(CLAUSES),
        generator_version="1.0.0",
    )
    fields.update(overrides)
    return ExampleMetadata(**fields)


def build_example(**overrides):
    metadata = build_metadata(**overrides)
    scene, labels = build_scene()
    return Example(metadata=metadata, arrays=ExampleArrays.from_scene(metadata, scene, labels))


def test_a_complete_example_exposes_every_required_field():
    example = build_example()
    assert_complete(example)
    assert set(REQUIRED_EXAMPLE_FIELDS) == set(example.as_dict())
    assert example.field("anchor_masks").shape == (NUM_ANCHORS,) + SHAPE
    assert example.field("structured_prompt") == CLAUSES
    assert example.field("relations") == CLAUSES


def test_versions_are_recorded_on_every_example():
    metadata = build_metadata()
    assert metadata.direction_rule_version == DIRECTION_RULE_VERSION
    assert metadata.vocabulary_version == VOCABULARY_VERSION
    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.generator_version


def test_anchor_order_is_identical_in_channels_prompt_and_text():
    metadata = build_metadata()
    assert metadata.anchor_shape_names == metadata.structured_prompt.anchors
    assert metadata.prompt == metadata.structured_prompt.render()
    with pytest.raises(SchemaError):
        build_metadata(anchor_shape_names=("pyramid", "cube", "sphere"))


def test_a_stored_prompt_that_does_not_round_trip_is_rejected():
    with pytest.raises(SchemaError):
        build_metadata(
            prompt="segment the shape that is lateral to the cube, superior to the "
            "pyramid, and anterior to the torus."
        )


def test_the_target_may_never_be_an_anchor():
    with pytest.raises(SchemaError):
        build_metadata(target_instance_id=1)
    with pytest.raises(SchemaError):
        build_metadata(target_shape_name="cube", target_instance_id=11)


def test_anchor_channels_must_match_their_declared_anchors():
    example = build_example()
    swapped = np.stack(
        [example.arrays.anchor_masks[1], example.arrays.anchor_masks[0], example.arrays.anchor_masks[2]]
    )
    broken = ExampleArrays(
        scene_volume=example.arrays.scene_volume,
        instance_labels=example.arrays.instance_labels,
        target_mask=example.arrays.target_mask,
        anchor_masks=swapped,
        anchor_union_mask=example.arrays.anchor_union_mask,
    )
    with pytest.raises(SchemaError):
        broken.validate(example.metadata)


def test_the_target_must_not_appear_in_any_anchor_channel():
    example = build_example()
    contaminated = example.arrays.anchor_masks.copy()
    contaminated[0] = np.maximum(contaminated[0], example.arrays.target_mask)
    broken = ExampleArrays(
        scene_volume=example.arrays.scene_volume,
        instance_labels=example.arrays.instance_labels,
        target_mask=example.arrays.target_mask,
        anchor_masks=contaminated,
        anchor_union_mask=example.arrays.anchor_union_mask,
    )
    with pytest.raises(SchemaError):
        broken.validate(example.metadata)


def test_a_scene_without_exactly_ten_instances_is_rejected():
    metadata = build_metadata()
    scene, labels = build_scene()
    labels[labels == 10] = 0
    with pytest.raises(SchemaError):
        ExampleArrays.from_scene(metadata, (labels > 0).astype(np.uint8), labels)


def test_validate_example_runs_the_full_contract():
    validate_example(build_example(), margin_voxels=1)
    with pytest.raises(ValidationError):
        validate_example(build_example(), margin_voxels=3)


# ---------------------------------------------------------------------------
# Stage B input contract
# ---------------------------------------------------------------------------
def test_stage_b_receives_occupancy_but_never_the_target():
    inputs = stage_b_inputs(build_example())
    assert set(inputs) == set(STAGE_B_ALLOWED_FIELDS)
    assert "scene_volume" in inputs
    for forbidden in STAGE_B_FORBIDDEN_FIELDS:
        assert forbidden not in inputs
    assert inputs["anchor_masks"].shape == (NUM_ANCHORS,) + SHAPE


def test_the_union_mask_is_only_reachable_through_the_ablation_switch():
    assert "anchor_union_mask" not in stage_b_inputs(build_example())
    ablation = stage_b_inputs(build_example(), use_union_mask=True)
    assert "anchor_masks" not in ablation
    assert ablation["anchor_union_mask"].shape == SHAPE


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------
def test_manifest_round_trips_through_jsonl(tmp_path):
    metadata = build_metadata()
    path = tmp_path / "train.jsonl"
    assert write_manifest(path, [metadata]) == 1
    (loaded,) = list(read_manifest(path))
    assert loaded.to_json_dict() == metadata.to_json_dict()
    assert loaded.prompt == metadata.prompt


def test_a_corrupt_manifest_line_fails_fast(tmp_path):
    metadata = build_metadata()
    record = metadata.to_json_dict()
    record["prompt"] = "segment the shape that is lateral to the cube."
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        list(read_manifest(path))


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------
def test_configured_target_split_is_seven_two_one_and_covers_the_vocabulary():
    targets = load_config("split")["target_classes"]
    validate_split_assignment(targets["train"], targets["val"], targets["test"])
    assert len(targets["train"]) == 7 and len(targets["val"]) == 2 and len(targets["test"]) == 1


def test_all_ten_classes_remain_available_as_anchors_in_every_split():
    split = load_config("split")
    for value in split["anchor_classes"].values():
        assert value == "all"
    assert len(SHAPE_VOCABULARY) == 10


def test_scene_seed_ranges_are_disjoint_and_match_the_scene_counts():
    split = load_config("split")
    counts = split["scenes"]
    assert counts["train"] + counts["val"] + counts["test"] == counts["total"]
    used = set()
    for name, (start, end) in split["scene_seed_ranges"].items():
        assert end - start == counts[name]
        seeds = set(range(start, end))
        assert not (seeds & used)
        used |= seeds


def test_generator_config_versions_match_the_code_constants():
    generator = load_config("generator")
    assert generator["direction_rule_version"] == DIRECTION_RULE_VERSION
    assert generator["vocabulary_version"] == VOCABULARY_VERSION
    assert generator["schema_version"] == SCHEMA_VERSION
    assert tuple(generator["geometry"]["volume_shape"]) == (64, 64, 64)
    assert generator["anchors"]["num_anchors"] == NUM_ANCHORS
    assert generator["packing"]["shapes_per_scene"] == len(SHAPE_VOCABULARY)
    assert generator["packing"]["allow_overlap"] is False
    assert generator["anchors"]["require_distinct_directions"] is True
