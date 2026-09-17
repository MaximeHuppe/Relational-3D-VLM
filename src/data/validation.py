"""Fail-fast validation of generated scenes and examples.

Every generated example is validated before it is written to disk. Nothing is
silently repaired: a failing scene is rejected, its reason is logged, and the
generator regenerates.

The checks mandated by CLAUDE.md:

* exactly ten shape instances exist;
* no object is empty, out of bounds, or overlapping another;
* the target never occurs in an anchor channel;
* every channel matches its declared anchor;
* anchor order is identical in the channels, the structured prompt and the text;
* the prompt round-trips from the structured metadata.

The array-level subset lives here; the mask/metadata consistency checks are
implemented by :meth:`src.data.schema.ExampleArrays.validate` and are re-run
from :func:`validate_example`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from src.data.primitives import SHAPE_VOCABULARY
from src.data.schema import (
    BACKGROUND_LABEL,
    SHAPES_PER_SCENE,
    Example,
    SchemaError,
    assert_complete,
)


class RejectionReason(str, Enum):
    """Why a candidate scene or example was rejected. Logged, never swallowed."""

    WRONG_INSTANCE_COUNT = "wrong_instance_count"
    EMPTY_OBJECT = "empty_object"
    OUT_OF_BOUNDS = "out_of_bounds"
    OVERLAP = "overlap"
    OUTSIDE_BODY = "outside_body"
    TARGET_IN_ANCHOR_CHANNEL = "target_in_anchor_channel"
    CHANNEL_ANCHOR_MISMATCH = "channel_anchor_mismatch"
    ANCHOR_ORDER_MISMATCH = "anchor_order_mismatch"
    PROMPT_ROUND_TRIP = "prompt_round_trip"
    AMBIGUOUS_DIRECTION = "ambiguous_direction"
    NO_FEASIBLE_ANCHOR_TRIPLE = "no_feasible_anchor_triple"
    PACKING_FAILED = "packing_failed"


class ValidationError(ValueError):
    """Raised when a scene or example violates the generation contract."""

    def __init__(self, reason: RejectionReason, message: str) -> None:
        super().__init__(f"[{reason.value}] {message}")
        self.reason = reason


@dataclass
class RejectionLog:
    """Rejection counters and acceptance rate for one generation run.

    Scene-level and object-level rejections are counted separately: the
    acceptance rate is scenes accepted over scene attempts, so it is not
    diluted by the many per-object placement retries inside one scene.
    """

    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    object_rejected: Counter = field(default_factory=Counter)

    def accept(self) -> None:
        self.accepted += 1

    def reject(self, reason: RejectionReason) -> None:
        """Record a rejected scene attempt."""
        self.rejected[reason.value] += 1

    def reject_object(self, reason: RejectionReason) -> None:
        """Record a rejected object placement inside a scene attempt."""
        self.object_rejected[reason.value] += 1

    @property
    def attempts(self) -> int:
        return self.accepted + sum(self.rejected.values())

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "scene_attempts": self.attempts,
            "scenes_accepted": self.accepted,
            "scene_acceptance_rate": self.acceptance_rate,
            "scene_rejections": dict(self.rejected),
            "object_rejections": dict(self.object_rejected),
        }


def validate_instance_labels(
    instance_labels: np.ndarray,
    *,
    margin_voxels: int = 0,
) -> None:
    """Check a scene's ``instance_labels``: count, emptiness, bounds, overlap.

    Overlap cannot be represented in a single label volume, so the generator
    must additionally call :func:`check_no_overlap` on the candidate mask before
    compositing. This function checks everything the label volume can express.
    """
    labels = np.asarray(instance_labels)
    if labels.ndim != 3:
        raise ValidationError(
            RejectionReason.WRONG_INSTANCE_COUNT,
            f"instance_labels must be 3D (D, H, W), got shape {labels.shape}",
        )

    present = np.unique(labels)
    expected = np.arange(BACKGROUND_LABEL, SHAPES_PER_SCENE + 1)
    if not np.array_equal(present, expected):
        missing = sorted(set(expected.tolist()) - set(present.tolist()))
        extra = sorted(set(present.tolist()) - set(expected.tolist()))
        reason = (
            RejectionReason.EMPTY_OBJECT if missing else RejectionReason.WRONG_INSTANCE_COUNT
        )
        raise ValidationError(
            reason,
            f"expected exactly {SHAPES_PER_SCENE} instances plus background; "
            f"missing {missing}, unexpected {extra}",
        )

    if margin_voxels > 0:
        check_in_bounds(labels != BACKGROUND_LABEL, margin_voxels=margin_voxels)


def check_in_bounds(mask: np.ndarray, *, margin_voxels: int) -> None:
    """Fail if any foreground voxel is closer to a border than ``margin_voxels``."""
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        raise ValidationError(RejectionReason.EMPTY_OBJECT, "mask is empty")
    for axis, size in enumerate(arr.shape):
        occupied = np.nonzero(arr.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        low, high = int(occupied.min()), int(occupied.max())
        if low < margin_voxels or high >= size - margin_voxels:
            axis_name = ("z", "y", "x")[axis]
            raise ValidationError(
                RejectionReason.OUT_OF_BOUNDS,
                f"object spans {axis_name} in [{low}, {high}] of size {size}, "
                f"violating the {margin_voxels}-voxel margin",
            )


def check_inside_body(
    center_world: Sequence[float],
    half_extent_world: Sequence[float],
    body,
    *,
    margin: float = 0.0,
) -> None:
    """Fail if an object's bounding box leaves the simulated head.

    Only used when ``body.constrain_placement`` is on: structures have to sit
    inside tissue for the appearance model to mean anything, so a placement
    that pokes out into the air around the head is rejected like any other
    invalid placement rather than quietly accepted.
    """
    if body is None:
        return
    if not body.contains_box(center_world, half_extent_world, margin=margin):
        raise ValidationError(
            RejectionReason.OUTSIDE_BODY,
            f"object centred at {tuple(round(float(v), 2) for v in center_world)} is not "
            "fully inside the body",
        )


def check_no_overlap(candidate: np.ndarray, occupancy: np.ndarray) -> None:
    """Fail if a candidate object intersects already-placed objects."""
    overlap = np.asarray(candidate, dtype=bool) & np.asarray(occupancy, dtype=bool)
    count = int(overlap.sum())
    if count:
        raise ValidationError(
            RejectionReason.OVERLAP, f"candidate overlaps placed objects in {count} voxels"
        )


def validate_example(example: Example, *, margin_voxels: int = 0) -> None:
    """Run the complete example contract. Raises :class:`ValidationError`."""
    metadata = example.metadata
    arrays = example.arrays

    validate_instance_labels(arrays.instance_labels, margin_voxels=margin_voxels)

    try:
        arrays.validate(metadata)
        assert_complete(example)
    except SchemaError as error:
        raise ValidationError(_classify_schema_error(str(error)), str(error)) from error

    SHAPE_VOCABULARY.require_names(metadata.anchor_shape_names)


def _classify_schema_error(message: str) -> RejectionReason:
    lowered = message.lower()
    if "round-trip" in lowered or "parse back" in lowered:
        return RejectionReason.PROMPT_ROUND_TRIP
    if "anchor order" in lowered:
        return RejectionReason.ANCHOR_ORDER_MISMATCH
    if "target appears inside anchor channel" in lowered:
        return RejectionReason.TARGET_IN_ANCHOR_CHANNEL
    if "does not match its declared anchor" in lowered:
        return RejectionReason.CHANNEL_ANCHOR_MISMATCH
    if "empty" in lowered:
        return RejectionReason.EMPTY_OBJECT
    return RejectionReason.WRONG_INSTANCE_COUNT


def validate_split_assignment(
    train: Sequence[str], val: Sequence[str], test: Sequence[str]
) -> None:
    """Check the seven/two/one target-class assignment from ``configs/split.yaml``."""
    groups = {"train": tuple(train), "val": tuple(val), "test": tuple(test)}
    for name, classes in groups.items():
        SHAPE_VOCABULARY.require_names(classes)
        if len(set(classes)) != len(classes):
            raise ValidationError(
                RejectionReason.WRONG_INSTANCE_COUNT,
                f"{name} target classes contain duplicates: {classes}",
            )
    sizes = {name: len(classes) for name, classes in groups.items()}
    if sizes != {"train": 7, "val": 2, "test": 1}:
        raise ValidationError(
            RejectionReason.WRONG_INSTANCE_COUNT,
            f"target-class split must be 7/2/1, got {sizes}",
        )
    combined = [name for classes in groups.values() for name in classes]
    if len(set(combined)) != len(combined):
        raise ValidationError(
            RejectionReason.WRONG_INSTANCE_COUNT,
            "target-class splits must be disjoint",
        )
    if set(combined) != set(SHAPE_VOCABULARY.names):
        raise ValidationError(
            RejectionReason.WRONG_INSTANCE_COUNT,
            "target-class splits must cover the whole vocabulary exactly once",
        )


def verify_relations_against_masks(example, *, atol: float = 1e-9) -> None:
    """Independently recompute every relation from the stored arrays.

    Checks, straight from ``target_mask`` and ``anchor_masks`` rather than from
    the metadata the generator wrote:

    * each clause direction equals the direction recomputed from the two masks;
    * each anchor channel is the mask of its declared shape name;
    * anchor channels are ordered by ascending centroid distance, instance ID as
      tie-break - the same order as the prompt clauses.
    """
    from src.data.direction_rules import centroid_world, direction_between_masks

    metadata = example.metadata
    arrays = example.arrays
    spacing = metadata.spacing
    target_centroid = centroid_world(arrays.target_mask, spacing)

    distances: list[tuple[float, int]] = []
    for slot, relation in enumerate(metadata.relations):
        channel = arrays.anchor_masks[slot]
        recomputed = direction_between_masks(
            arrays.target_mask,
            channel,
            volume_shape=metadata.volume_shape,
            spacing=spacing,
            atol=atol,
        )
        if recomputed.direction != relation["direction"]:
            raise ValidationError(
                RejectionReason.CHANNEL_ANCHOR_MISMATCH,
                f"clause {slot} says {relation['direction']!r} but the masks give "
                f"{recomputed.direction!r}",
            )
        instance_id = metadata.anchor_instance_ids[slot]
        expected_name = SHAPE_VOCABULARY.id_to_name(instance_id)
        if (
            expected_name != relation["anchor"]
            or relation["anchor"] != metadata.anchor_shape_names[slot]
        ):
            raise ValidationError(
                RejectionReason.CHANNEL_ANCHOR_MISMATCH,
                f"channel {slot} holds instance {instance_id} ({expected_name}) but the "
                f"clause names {relation['anchor']!r}",
            )
        distance = float(
            np.linalg.norm(centroid_world(channel, spacing) - target_centroid)
        )
        distances.append((distance, instance_id))

    if distances != sorted(distances):
        raise ValidationError(
            RejectionReason.ANCHOR_ORDER_MISMATCH,
            f"anchor channels are not ordered by ascending centroid distance: {distances}",
        )
