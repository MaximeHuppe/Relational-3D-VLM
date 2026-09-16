"""Canonical prompt rendering and parsing.

The structured representation is the primary model interface:

.. code-block:: python

    [
        {"direction": direction_1, "anchor": anchor_1},
        {"direction": direction_2, "anchor": anchor_2},
        {"direction": direction_3, "anchor": anchor_3},
    ]

The natural-language form is a deterministic rendering of it:

.. code-block:: text

    segment the shape that is {direction_1} to the {anchor_1}, {direction_2} to
    the {anchor_2}, and {direction_3} to the {anchor_3}.

``parse_prompt(render_prompt(x)) == x`` holds for every valid clause triple, and
the parser is built from the closed vocabularies, so a prompt containing a
synonym, an ordinal reference or an unknown shape name cannot be parsed.

Clause order is the anchor order: ascending centroid distance from the target,
instance ID as tie-break. That same order is used by the three mask channels.

This module deliberately does not import :mod:`src.data.schema`; the schema
layer builds on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.data.direction_rules import (
    DIRECTIONS,
    DirectionError,
    classify_direction,
    is_valid_direction,
    validate_distinct_directions,
)
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY

PROMPT_VERSION = "1.0.0"

NUM_CLAUSES = 3

PROMPT_TEMPLATE = (
    "segment the shape that is "
    "{direction_1} to the {anchor_1}, "
    "{direction_2} to the {anchor_2}, "
    "and {direction_3} to the {anchor_3}."
)

_DIRECTION_ALTERNATION = "|".join(sorted(DIRECTIONS, key=len, reverse=True))
# The parser accepts a shape name either in its canonical underscore form or
# with spaces ("triangular prism"); both normalise to the canonical name. This
# is formatting tolerance only - no synonyms are accepted.
_SHAPE_ALTERNATION = "|".join(
    sorted(
        {name for name in SHAPE_NAMES} | {name.replace("_", " ") for name in SHAPE_NAMES},
        key=len,
        reverse=True,
    )
)

_PROMPT_PATTERN = re.compile(
    r"^segment the shape that is "
    rf"(?P<direction_1>{_DIRECTION_ALTERNATION}) to the (?P<anchor_1>{_SHAPE_ALTERNATION}), "
    rf"(?P<direction_2>{_DIRECTION_ALTERNATION}) to the (?P<anchor_2>{_SHAPE_ALTERNATION}), "
    rf"and (?P<direction_3>{_DIRECTION_ALTERNATION}) to the (?P<anchor_3>{_SHAPE_ALTERNATION})\.$"
)


class PromptError(ValueError):
    """Raised for malformed prompts or malformed clause triples."""


def _normalize_shape(name: str) -> str:
    """Map an accepted surface form to the canonical vocabulary name."""
    canonical = name.replace(" ", "_")
    if canonical not in SHAPE_NAMES:
        raise PromptError(f"unknown anchor shape {name!r}; vocabulary is {SHAPE_NAMES}")
    return canonical


def validate_clause(
    clause: Mapping[str, str], position: int | None = None
) -> dict[str, str]:
    """Validate one clause in isolation and return it in canonical form.

    Enforces the ``direction``/``anchor`` keys and both closed vocabularies.
    Cross-clause rules (distinct directions, distinct anchors) are checked by
    :func:`validate_clauses`.
    """
    where = "clause" if position is None else f"clause {position}"
    if set(clause) != {"direction", "anchor"}:
        raise PromptError(
            f"{where} must have exactly the keys {{'direction', 'anchor'}}, "
            f"got {sorted(clause)}"
        )
    direction = str(clause["direction"])
    if not is_valid_direction(direction):
        raise PromptError(
            f"{where} has unknown direction {direction!r}; "
            f"accepted tokens are {DIRECTIONS}"
        )
    return {"direction": direction, "anchor": _normalize_shape(str(clause["anchor"]))}


def validate_clauses(clauses: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    """Validate a structured clause triple and return it in canonical form.

    Enforces: exactly three clauses, ``direction``/``anchor`` keys only, closed
    vocabularies, three pairwise-distinct directions and three distinct anchors.
    """
    if len(clauses) != NUM_CLAUSES:
        raise PromptError(f"expected {NUM_CLAUSES} clauses, got {len(clauses)}")

    canonical = [
        validate_clause(clause, position)
        for position, clause in enumerate(clauses, start=1)
    ]

    try:
        validate_distinct_directions(clause["direction"] for clause in canonical)
    except DirectionError as error:  # pragma: no cover - re-raised as PromptError
        raise PromptError(str(error)) from error

    anchors = [clause["anchor"] for clause in canonical]
    if len(set(anchors)) != len(anchors):
        raise PromptError(f"anchors must be pairwise distinct, got {anchors!r}")

    return tuple(canonical)


def render_prompt(clauses: Sequence[Mapping[str, str]]) -> str:
    """Render the canonical natural-language prompt from a clause triple."""
    canonical = validate_clauses(clauses)
    fields: dict[str, str] = {}
    for position, clause in enumerate(canonical, start=1):
        fields[f"direction_{position}"] = clause["direction"]
        fields[f"anchor_{position}"] = clause["anchor"]
    return PROMPT_TEMPLATE.format(**fields)


def parse_prompt(prompt: str) -> tuple[dict[str, str], ...]:
    """Parse a canonical prompt back into its structured clause triple."""
    if not isinstance(prompt, str):
        raise PromptError(f"prompt must be a string, got {type(prompt)}")
    match = _PROMPT_PATTERN.match(prompt.strip())
    if match is None:
        raise PromptError(f"prompt does not match the canonical template: {prompt!r}")
    clauses = [
        {
            "direction": match.group(f"direction_{position}"),
            "anchor": match.group(f"anchor_{position}"),
        }
        for position in range(1, NUM_CLAUSES + 1)
    ]
    return validate_clauses(clauses)


def round_trips(clauses: Sequence[Mapping[str, str]], prompt: str) -> bool:
    """True iff ``prompt`` is exactly the rendering of ``clauses`` and parses back."""
    try:
        canonical = validate_clauses(clauses)
        return render_prompt(canonical) == prompt and parse_prompt(prompt) == canonical
    except PromptError:
        return False


def assert_no_target_leakage(prompt: str, target_shape_name: str) -> None:
    """Fail if the target shape name appears anywhere in the prompt.

    The target shape is never an input: prompts name anchors only.
    """
    canonical = _normalize_shape(target_shape_name)
    anchors = {clause["anchor"] for clause in parse_prompt(prompt)}
    if canonical in anchors:
        raise PromptError(
            f"target shape {canonical!r} appears as an anchor in the prompt"
        )
    surfaces = (canonical, canonical.replace("_", " "))
    if any(surface in prompt for surface in surfaces):
        raise PromptError(
            f"target shape name {canonical!r} leaks into the prompt: {prompt!r}"
        )


@dataclass(frozen=True)
class AnchorCandidate:
    """One ranked candidate anchor for a given target."""

    instance_id: int
    shape_name: str
    distance: float
    direction: str


class AnchorSelectionError(PromptError):
    """Raised when no feasible three-direction anchor set exists for a target."""


def rank_candidates(
    target_instance_id: int,
    centroids_world: Mapping[int, Sequence[float]],
) -> tuple[int, ...]:
    """Rank the other objects by Euclidean centroid distance, ID as tie-break."""
    if target_instance_id not in centroids_world:
        raise AnchorSelectionError(f"target instance {target_instance_id} has no centroid")
    target = np.asarray(centroids_world[target_instance_id], dtype=np.float64)
    ranked = sorted(
        (instance_id for instance_id in centroids_world if instance_id != target_instance_id),
        key=lambda instance_id: (
            float(np.linalg.norm(np.asarray(centroids_world[instance_id], dtype=np.float64) - target)),
            instance_id,
        ),
    )
    return tuple(ranked)


def select_anchors(
    target_instance_id: int,
    centroids_world: Mapping[int, Sequence[float]],
    shape_names: Mapping[int, str],
    *,
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    num_anchors: int = NUM_CLAUSES,
) -> tuple[AnchorCandidate, ...]:
    """Nearest-feasible anchor selection.

    Algorithm (CLAUDE.md, "Prompt and anchor generation"):

    1. compute centroids of the target and the other nine objects;
    2. rank candidates by Euclidean centroid distance, instance ID as tie-break;
    3. compute the target-relative direction of every candidate;
    4. scan in ranked order, keeping a candidate only if its direction is unused;
    5. stop once three anchors with three distinct directions are held;
    6. if no such triple exists, reject and regenerate the whole scene.

    The result is the nearest FEASIBLE set, not necessarily the three nearest
    objects. Anchor order is ascending distance with instance ID as tie-break,
    and that order is shared by the text, the structured prompt and the three
    mask channels.

    Step 3 is computed for every candidate, so an ambiguous pair anywhere in the
    scene raises :class:`~src.data.direction_rules.AmbiguousDirectionError` and
    the caller regenerates; a direction is never invented.

    Raises:
        AnchorSelectionError: no set of ``num_anchors`` distinct directions exists.
        AmbiguousDirectionError: some candidate has no well-defined direction.
    """
    ranked = rank_candidates(target_instance_id, centroids_world)
    target_centroid = centroids_world[target_instance_id]

    # Step 3: every candidate, not only the ones that end up selected.
    directions = {
        instance_id: classify_direction(
            target_centroid,
            centroids_world[instance_id],
            volume_shape=volume_shape,
            spacing=spacing,
        ).direction
        for instance_id in ranked
    }

    target_point = np.asarray(target_centroid, dtype=np.float64)
    selected: list[AnchorCandidate] = []
    used: set[str] = set()
    for instance_id in ranked:
        direction = directions[instance_id]
        if direction in used:
            continue
        used.add(direction)
        distance = float(
            np.linalg.norm(np.asarray(centroids_world[instance_id], dtype=np.float64) - target_point)
        )
        selected.append(
            AnchorCandidate(
                instance_id=int(instance_id),
                shape_name=shape_names[instance_id],
                distance=distance,
                direction=direction,
            )
        )
        if len(selected) == num_anchors:
            break

    if len(selected) < num_anchors:
        raise AnchorSelectionError(
            f"target {target_instance_id} has only {len(selected)} distinct "
            f"direction(s) among {len(ranked)} candidates; regenerate the scene"
        )
    return tuple(selected)


def clauses_from_anchors(anchors: Sequence[AnchorCandidate]) -> list[dict[str, str]]:
    """Turn a selected anchor set into the structured clause triple, in order."""
    return [
        {"direction": anchor.direction, "anchor": anchor.shape_name} for anchor in anchors
    ]


def iter_all_clause_pairs() -> Iterable[tuple[str, str]]:
    """Every ``(direction, anchor_shape)`` pair; sizes the pair-embedding table."""
    for direction in DIRECTIONS:
        for shape in SHAPE_NAMES:
            yield direction, shape


def clause_indices(clauses: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int]]:
    """Structured clauses -> ``(direction_ids, anchor_shape_ids)``.

    Zero-based indices into the two closed vocabularies, in clause order. This
    is the single place the text side of the project turns into integers: the
    model's embedding tables (:class:`src.models.prompt_encoder.RelationPromptEncoder`)
    and the Stage B dataset both go through it, so a prompt cannot mean one
    thing on disk and another in the network.
    """
    canonical = validate_clauses(clauses)
    directions = [DIRECTIONS.index(clause["direction"]) for clause in canonical]
    shapes = [SHAPE_VOCABULARY.index_of(clause["anchor"]) for clause in canonical]
    return directions, shapes


def clauses_from_indices(
    direction_ids: Sequence[int], anchor_shape_ids: Sequence[int]
) -> list[dict[str, str]]:
    """Inverse of :func:`clause_indices`, for logging and counterfactuals."""
    if len(direction_ids) != len(anchor_shape_ids):
        raise PromptError(
            f"got {len(direction_ids)} directions and {len(anchor_shape_ids)} anchors"
        )
    return list(
        validate_clauses(
            [
                {"direction": DIRECTIONS[int(d)], "anchor": SHAPE_NAMES[int(s)]}
                for d, s in zip(direction_ids, anchor_shape_ids)
            ]
        )
    )
