"""Target-relative direction semantics for Relational3D-VLM.

This module is the code-level source of truth for the six direction tokens and
for how a `(target, anchor)` pair is turned into exactly one of them. Every
other module (prompt generation, validation, augmentation rewrites, evaluation
stratification) must call into here instead of re-deriving the rule.

Conventions (see CLAUDE.md, "Coordinate system and direction semantics"):

* World frame is RAS. World coordinates are always ordered ``(x, y, z)``:
  ``x`` right/lateral, ``y`` anterior, ``z`` superior.
* Voxel arrays are always indexed ``(z, y, x)``, i.e. shape ``(D, H, W)``.
* ``spacing`` is world units per voxel, ordered ``(x, y, z)``.
* For a volume of shape ``(D, H, W)`` the world-space centre is
  ``(((W - 1) / 2) * sx, ((H - 1) / 2) * sy, ((D - 1) / 2) * sz)``.

The rule, applied to ``delta = centroid(target) - centroid(anchor)`` expressed
in world units (so anisotropic spacing is already normalised):

1. the main axis is ``argmax(|delta_x|, |delta_y|, |delta_z|)``, with the
   deterministic tie priority ``z``, then ``y``, then ``x``;
2. axis ``z``: ``delta_z > 0`` -> ``superior``, ``delta_z < 0`` -> ``inferior``;
3. axis ``y``: ``delta_y > 0`` -> ``anterior``, ``delta_y < 0`` -> ``posterior``;
4. axis ``x``: compare distances to the volume centre plane. The target farther
   from the centre is ``lateral``, closer to the centre is ``medial``.

A direction is never invented. Two cases are rejected instead, and the caller
must regenerate the scene:

* the two centroids coincide (``delta`` is the zero vector);
* the x axis is selected and the two centre-plane distances are equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

# Bump on any change to the rule itself (axis selection, tie priority, token
# assignment, rejection conditions). Stored in every generated example.
DIRECTION_RULE_VERSION = "1.0.0"

ANTERIOR = "anterior"
POSTERIOR = "posterior"
SUPERIOR = "superior"
INFERIOR = "inferior"
MEDIAL = "medial"
LATERAL = "lateral"

#: The complete, closed set of accepted direction tokens. No synonyms.
DIRECTIONS: tuple[str, ...] = (
    ANTERIOR,
    POSTERIOR,
    SUPERIOR,
    INFERIOR,
    MEDIAL,
    LATERAL,
)

DIRECTION_OPPOSITES: dict[str, str] = {
    ANTERIOR: POSTERIOR,
    POSTERIOR: ANTERIOR,
    SUPERIOR: INFERIOR,
    INFERIOR: SUPERIOR,
    MEDIAL: LATERAL,
    LATERAL: MEDIAL,
}

#: World axis each direction pair is decided on.
AXIS_OF_DIRECTION: dict[str, str] = {
    ANTERIOR: "y",
    POSTERIOR: "y",
    SUPERIOR: "z",
    INFERIOR: "z",
    MEDIAL: "x",
    LATERAL: "x",
}

#: World-coordinate ordering used everywhere in this project.
AXES: tuple[str, ...] = ("x", "y", "z")
AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}

#: Deterministic tie priority when two or more |delta| components are equal.
AXIS_TIE_PRIORITY: tuple[str, ...] = ("z", "y", "x")

#: Default absolute tolerance for "these two magnitudes are equal".
DEFAULT_ATOL = 1e-9


class DirectionError(ValueError):
    """Base class for direction-rule failures."""


class AmbiguityReason(str, Enum):
    """Why a pair has no well-defined direction."""

    ZERO_DELTA = "zero_delta"
    MEDIAL_LATERAL_TIE = "medial_lateral_tie"


class AmbiguousDirectionError(DirectionError):
    """Raised instead of inventing a direction; the caller must regenerate."""

    def __init__(self, reason: AmbiguityReason, message: str) -> None:
        super().__init__(f"[{reason.value}] {message}")
        self.reason = reason


@dataclass(frozen=True)
class DirectionResult:
    """Outcome of classifying one ``(target, anchor)`` pair."""

    direction: str
    axis: str
    delta: tuple[float, float, float]  # world units, ordered (x, y, z)
    tie_broken: bool  # True when the axis was chosen by the z > y > x priority
    rule_version: str = DIRECTION_RULE_VERSION

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise DirectionError(f"unknown direction token: {self.direction!r}")
        if AXIS_OF_DIRECTION[self.direction] != self.axis:
            raise DirectionError(
                f"direction {self.direction!r} is not decided on axis {self.axis!r}"
            )


def is_valid_direction(token: str) -> bool:
    """True iff ``token`` is one of the six accepted direction tokens."""
    return token in DIRECTIONS


def opposite_direction(direction: str) -> str:
    """Return the opposite token (used by the counterfactual tests)."""
    if direction not in DIRECTION_OPPOSITES:
        raise DirectionError(f"unknown direction token: {direction!r}")
    return DIRECTION_OPPOSITES[direction]


def _as_spacing(spacing: Sequence[float]) -> np.ndarray:
    arr = np.asarray(spacing, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"spacing must have 3 components (x, y, z), got {arr.shape}")
    if np.any(arr <= 0):
        raise ValueError(f"spacing must be strictly positive, got {tuple(arr)}")
    return arr


def _as_point(point: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have 3 components (x, y, z), got {arr.shape}")
    return arr


def volume_center_world(
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """World coordinates ``(x, y, z)`` of the centre of a ``(D, H, W)`` volume."""
    shape = np.asarray(volume_shape, dtype=np.int64)
    if shape.shape != (3,):
        raise ValueError(f"volume_shape must be (D, H, W), got {tuple(shape)}")
    if np.any(shape <= 0):
        raise ValueError(f"volume_shape must be positive, got {tuple(shape)}")
    sp = _as_spacing(spacing)
    depth, height, width = (int(v) for v in shape)
    # (D, H, W) is (z, y, x); the returned point is ordered (x, y, z).
    return np.array(
        [
            ((width - 1) / 2.0) * sp[0],
            ((height - 1) / 2.0) * sp[1],
            ((depth - 1) / 2.0) * sp[2],
        ],
        dtype=np.float64,
    )


def centroid_voxel_zyx(mask: np.ndarray) -> np.ndarray:
    """Centroid of a binary mask in fractional voxel indices ``(z, y, x)``."""
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"mask must be 3D (D, H, W), got shape {arr.shape}")
    indices = np.nonzero(arr)
    if indices[0].size == 0:
        raise ValueError("cannot compute a centroid of an empty mask")
    return np.array([idx.mean() for idx in indices], dtype=np.float64)


def centroid_world(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Centroid of a binary mask in world coordinates ``(x, y, z)``."""
    sp = _as_spacing(spacing)
    z, y, x = centroid_voxel_zyx(mask)
    return np.array([x * sp[0], y * sp[1], z * sp[2]], dtype=np.float64)


def bbox_extent_world(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Inclusive bounding-box extent of a mask in world units ``(x, y, z)``.

    A single occupied voxel has an extent of one spacing unit per axis.
    """
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"mask must be 3D (D, H, W), got shape {arr.shape}")
    indices = np.nonzero(arr)
    if indices[0].size == 0:
        raise ValueError("cannot compute a bounding box of an empty mask")
    sp = _as_spacing(spacing)
    spans_zyx = [int(idx.max()) - int(idx.min()) + 1 for idx in indices]
    span_z, span_y, span_x = spans_zyx
    return np.array(
        [span_x * sp[0], span_y * sp[1], span_z * sp[2]], dtype=np.float64
    )


def select_axis(
    delta: Sequence[float],
    atol: float = DEFAULT_ATOL,
) -> tuple[str, bool]:
    """Pick the main axis of ``delta`` (world units, ordered ``(x, y, z)``).

    Returns ``(axis, tie_broken)``. ``tie_broken`` is True when more than one
    component shared the maximum magnitude and the ``z > y > x`` priority
    decided the outcome.

    Raises:
        AmbiguousDirectionError: if ``delta`` is the zero vector.
    """
    d = _as_point(delta, "delta")
    magnitudes = np.abs(d)
    largest = float(magnitudes.max())
    if largest <= atol:
        raise AmbiguousDirectionError(
            AmbiguityReason.ZERO_DELTA,
            "target and anchor centroids coincide; no direction is defined",
        )
    tied = [
        axis
        for axis in AXIS_TIE_PRIORITY
        if bool(np.isclose(magnitudes[AXIS_INDEX[axis]], largest, rtol=0.0, atol=atol))
    ]
    return tied[0], len(tied) > 1


def classify_direction(
    target_centroid_world: Sequence[float],
    anchor_centroid_world: Sequence[float],
    *,
    volume_shape: Sequence[int] | None = None,
    center_world: Sequence[float] | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    atol: float = DEFAULT_ATOL,
) -> DirectionResult:
    """Classify the target's position relative to the anchor.

    The prompt always describes the TARGET relative to the ANCHOR: the returned
    token is the one that goes into ``"{direction} to the {anchor}"``.

    Exactly one of ``volume_shape`` or ``center_world`` must be given; it fixes
    the centre plane used by the medial/lateral comparison.

    Raises:
        AmbiguousDirectionError: zero delta, or an exact medial/lateral tie.
    """
    target = _as_point(target_centroid_world, "target_centroid_world")
    anchor = _as_point(anchor_centroid_world, "anchor_centroid_world")

    if (volume_shape is None) == (center_world is None):
        raise ValueError("pass exactly one of volume_shape or center_world")
    center = (
        volume_center_world(volume_shape, spacing)
        if center_world is None
        else _as_point(center_world, "center_world")
    )

    delta = target - anchor
    axis, tie_broken = select_axis(delta, atol=atol)

    if axis == "z":
        direction = SUPERIOR if delta[AXIS_INDEX["z"]] > 0 else INFERIOR
    elif axis == "y":
        direction = ANTERIOR if delta[AXIS_INDEX["y"]] > 0 else POSTERIOR
    else:
        target_offset = abs(float(target[AXIS_INDEX["x"]] - center[AXIS_INDEX["x"]]))
        anchor_offset = abs(float(anchor[AXIS_INDEX["x"]] - center[AXIS_INDEX["x"]]))
        if bool(np.isclose(target_offset, anchor_offset, rtol=0.0, atol=atol)):
            raise AmbiguousDirectionError(
                AmbiguityReason.MEDIAL_LATERAL_TIE,
                "target and anchor are equidistant from the centre plane "
                f"(|dx| = {target_offset:.12g}); regenerate the scene",
            )
        direction = LATERAL if target_offset > anchor_offset else MEDIAL

    return DirectionResult(
        direction=direction,
        axis=axis,
        delta=(float(delta[0]), float(delta[1]), float(delta[2])),
        tie_broken=tie_broken,
    )


def direction_between_masks(
    target_mask: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    volume_shape: Sequence[int] | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    atol: float = DEFAULT_ATOL,
) -> DirectionResult:
    """Convenience wrapper: classify directly from two binary masks.

    ``volume_shape`` defaults to the shape of ``target_mask``.
    """
    target = np.asarray(target_mask)
    anchor = np.asarray(anchor_mask)
    if target.shape != anchor.shape:
        raise ValueError(
            f"mask shapes differ: target {target.shape} vs anchor {anchor.shape}"
        )
    shape = tuple(target.shape) if volume_shape is None else tuple(volume_shape)
    return classify_direction(
        centroid_world(target, spacing),
        centroid_world(anchor, spacing),
        volume_shape=shape,
        spacing=spacing,
        atol=atol,
    )


def validate_distinct_directions(directions: Iterable[str]) -> tuple[str, ...]:
    """Check a clause set: all tokens known, and all three directions distinct."""
    tokens = tuple(directions)
    unknown = [token for token in tokens if not is_valid_direction(token)]
    if unknown:
        raise DirectionError(f"unknown direction token(s): {unknown!r}")
    if len(set(tokens)) != len(tokens):
        raise DirectionError(f"directions must be pairwise distinct, got {tokens!r}")
    return tokens
