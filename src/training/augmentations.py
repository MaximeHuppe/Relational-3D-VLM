"""Relation-preserving augmentations: axis-aligned 90-degree rotations.

What this implements
--------------------
For every volume, one angle is drawn independently per world axis from
``(0, 90, 180, -90)`` degrees and the three rotations are composed. The same
volume is therefore seen under a different (but relation-consistent) pose in
every epoch. Only *proper* rotations are produced - no mirror reflections - so
every sampled transform is an element of the 24-element octahedral group, and
the 64 angle triples collapse onto those 24 elements.

Why the directions cannot simply be relabelled
----------------------------------------------
A 180-degree rotation swaps the two tokens of each axis it reverses, so a naive
"map every token to its opposite" rewrite happens to work there. It does **not**
generalise, and the 90-degree cases are where it breaks:

* ``superior``/``inferior`` and ``anterior``/``posterior`` are *signed* rules -
  they read the sign of one component of ``delta``;
* ``medial``/``lateral`` is a *centre-distance* rule - it compares
  ``|target_x - centre_x|`` against ``|anchor_x - centre_x|`` and never looks at
  the sign of ``delta_x``.

A 90-degree rotation moves the main axis of a pair between those two families,
and the information needed by the destination family is not carried by the
source token. Two pairs with the same ``delta_x`` can be ``lateral`` and
``medial``; both become the same anterior/posterior token once ``x`` rotates
onto ``y``. Conversely a single ``anterior`` says nothing about which side of
the centre plane the pair will land on once ``y`` rotates onto ``x``.

So the rewrite is done the only way that is always right: the centroids are
transformed, and the direction of every pair is re-derived with the project's
one canonical rule, :func:`src.data.direction_rules.classify_direction`, against
the rotated volume's own shape, spacing and centre plane.

What the rotation preserves, and what it does not
-------------------------------------------------
Preserved, exactly:

* the three anchors and their order. Anchor order is ascending centroid
  distance, a rotation is an isometry of the grid, so the ranking is untouched;
* the target, the masks' voxel counts, and the target's absence from every
  anchor channel.

Not preserved, and deliberately not repaired:

* **the directions**, which is the whole point;
* **the "nearest feasible set" property**. The generator picks anchors greedily
  by scanning all nine candidates and keeping the first whose direction is still
  unused. Because the rotation re-derives the direction of *every* candidate, a
  candidate that was skipped as a duplicate can become feasible, so re-running
  the selection on the rotated scene could return a different anchor triple.
  This module keeps the stored anchors instead - as specified - which yields a
  prompt that is *valid* (three distinct, correctly-derived directions naming
  three real anchors) but not necessarily the triple the generator would have
  chosen for that pose. Training on those extra anchor sets is a superset of the
  evaluation distribution, never a contradiction of it.

Rejection
---------
Re-deriving the directions can produce a clause triple that is not legal:

* two anchors that sat on the same axis with opposite signs (``superior`` and
  ``inferior``) both become ``lateral`` when that axis rotates onto ``x``,
  because the centre-distance rule is not a sign rule - the three directions are
  no longer distinct;
* a pair can land exactly on the medial/lateral tie that
  :class:`~src.data.direction_rules.AmbiguousDirectionError` exists to reject.

Neither is repaired by inventing a direction. The rotation is rejected and
another is drawn, up to ``max_attempts``; the identity is the guaranteed-valid
fallback, and it reproduces the stored prompt exactly. Rejections are counted in
:attr:`RotationAugmentation.stats` so a corpus that rejects constantly is
visible rather than silent.

Determinism
-----------
The rotation of example ``index`` in epoch ``epoch`` is a pure function of
``(seed, epoch, index)``. It does not depend on shuffling, on the number of
dataloader workers, or on iteration order, so a run is reproducible from its
seed alone. ``epoch`` lives in a shared-memory tensor precisely so that
:meth:`RotationAugmentation.set_epoch` in the main process is visible to
persistent dataloader workers; call it before creating the epoch's iterator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

from src.data.direction_rules import (
    AXES,
    AXIS_INDEX,
    AmbiguousDirectionError,
    DirectionError,
    classify_direction,
    validate_distinct_directions,
)
from src.data.prompt_generator import PromptError, render_prompt, validate_clauses

#: Bump on any change to the sampling or to the rewrite itself.
AUGMENTATION_VERSION = "1.0.0"

#: The angles drawn per axis, in degrees. 0 is kept: the identity is a legal
#: draw, so a fraction of the epochs still sees the stored pose.
ROTATION_ANGLES: tuple[int, ...] = (0, 90, 180, -90)

#: Array axis holding world axis ``j``. Arrays are ``(z, y, x)``, world is
#: ``(x, y, z)``, so the two orders are each other's reverse.
_ARRAY_AXIS_OF_WORLD_AXIS = (2, 1, 0)


class AugmentationError(RuntimeError):
    """Base class for augmentation failures."""


class RelationRewriteError(AugmentationError):
    """A rotation has no legal clause triple; draw another one."""


def _array_axis(world_axis: int) -> int:
    return _ARRAY_AXIS_OF_WORLD_AXIS[world_axis]


def _world_sizes(volume_shape: Sequence[int]) -> tuple[int, int, int]:
    """``(D, H, W)`` -> per-world-axis sizes ``(N_x, N_y, N_z)``."""
    depth, height, width = (int(v) for v in volume_shape)
    if min(depth, height, width) <= 0:
        raise AugmentationError(f"volume_shape must be positive, got {tuple(volume_shape)}")
    return width, height, depth


def _volume_shape(world_sizes: Sequence[int]) -> tuple[int, int, int]:
    """Per-world-axis sizes ``(N_x, N_y, N_z)`` -> ``(D, H, W)``."""
    size_x, size_y, size_z = (int(v) for v in world_sizes)
    return size_z, size_y, size_x


@dataclass(frozen=True)
class AxisRotation:
    """One proper rotation of the grid, as a signed permutation of world axes.

    The rotation is stored in the form it is actually applied in::

        u'[i] = signs[i] * u[perm[i]]

    for ``u`` a point in *centred* world coordinates, i.e. the new ``x`` axis is
    ``signs[0]`` times the old ``AXES[perm[0]]`` axis. Both tuples are indexed by
    world axis, ``(x, y, z)``.

    Only determinant ``+1`` transforms are accepted, so this is an element of the
    octahedral rotation group and never a mirror reflection.
    """

    perm: tuple[int, int, int]
    signs: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "perm", tuple(int(v) for v in self.perm))
        object.__setattr__(self, "signs", tuple(int(v) for v in self.signs))
        if sorted(self.perm) != [0, 1, 2]:
            raise AugmentationError(f"perm must be a permutation of (0, 1, 2), got {self.perm}")
        if any(sign not in (-1, 1) for sign in self.signs):
            raise AugmentationError(f"signs must be +1 or -1, got {self.signs}")
        if int(round(float(np.linalg.det(self.matrix())))) != 1:
            raise AugmentationError(
                f"{self.code} is a reflection (determinant -1), not a rotation"
            )

    # -- construction ------------------------------------------------------
    @classmethod
    def identity(cls) -> "AxisRotation":
        return cls(perm=(0, 1, 2), signs=(1, 1, 1))

    @classmethod
    def about_axis(cls, axis: str | int, degrees: int) -> "AxisRotation":
        """Right-handed rotation of ``degrees`` about one world axis.

        ``degrees`` must be a multiple of 90; 270 and -90 denote the same
        rotation.
        """
        index = AXIS_INDEX[axis] if isinstance(axis, str) else int(axis)
        if index not in (0, 1, 2):
            raise AugmentationError(f"unknown rotation axis {axis!r}")
        if int(degrees) % 90 != 0:
            raise AugmentationError(f"degrees must be a multiple of 90, got {degrees}")
        quarters = (int(degrees) // 90) % 4
        rotation = cls.identity()
        # One quarter turn about `index` sends the next axis in the cyclic order
        # (x -> y -> z -> x) onto the one after it, and the one after it onto the
        # negated first: e.g. about x, y -> z and z -> -y.
        first, second = (index + 1) % 3, (index + 2) % 3
        quarter_perm = list(range(3))
        quarter_signs = [1, 1, 1]
        quarter_perm[first], quarter_signs[first] = second, -1
        quarter_perm[second], quarter_signs[second] = first, 1
        quarter = cls(perm=tuple(quarter_perm), signs=tuple(quarter_signs))
        for _ in range(quarters):
            rotation = quarter.compose(rotation)
        return rotation

    @classmethod
    def from_angles(cls, x_degrees: int, y_degrees: int, z_degrees: int) -> "AxisRotation":
        """Compose the three per-axis rotations as ``Rz . Ry . Rx``.

        The order is fixed and part of the contract: the same angle triple must
        always mean the same pose. ``x`` is applied first.
        """
        rotation = cls.about_axis("x", x_degrees)
        rotation = cls.about_axis("y", y_degrees).compose(rotation)
        return cls.about_axis("z", z_degrees).compose(rotation)

    def compose(self, other: "AxisRotation") -> "AxisRotation":
        """``self . other``: apply ``other`` first, then ``self``."""
        perm = tuple(other.perm[self.perm[i]] for i in range(3))
        signs = tuple(self.signs[i] * other.signs[self.perm[i]] for i in range(3))
        return AxisRotation(perm=perm, signs=signs)  # type: ignore[arg-type]

    def inverse(self) -> "AxisRotation":
        perm = [0, 0, 0]
        signs = [1, 1, 1]
        for i in range(3):
            perm[self.perm[i]] = i
            signs[self.perm[i]] = self.signs[i]
        return AxisRotation(perm=tuple(perm), signs=tuple(signs))  # type: ignore[arg-type]

    # -- identity and display ---------------------------------------------
    @property
    def is_identity(self) -> bool:
        return self.perm == (0, 1, 2) and self.signs == (1, 1, 1)

    @property
    def code(self) -> str:
        """Compact form, e.g. ``"+y-x+z"``: new x is +old y, new y is -old x."""
        return "".join(
            f"{'+' if sign > 0 else '-'}{AXES[axis]}"
            for axis, sign in zip(self.perm, self.signs)
        )

    def matrix(self) -> np.ndarray:
        """The ``3x3`` world-space rotation matrix, acting on ``(x, y, z)``."""
        matrix = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            matrix[i, self.perm[i]] = self.signs[i]
        return matrix

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.code

    # -- geometry ----------------------------------------------------------
    def transform_shape(self, volume_shape: Sequence[int]) -> tuple[int, int, int]:
        """``(D, H, W)`` of the rotated array."""
        sizes = _world_sizes(volume_shape)
        return _volume_shape([sizes[self.perm[i]] for i in range(3)])

    def transform_spacing(
        self, spacing: Sequence[float] = (1.0, 1.0, 1.0)
    ) -> tuple[float, float, float]:
        """World units per voxel ``(x, y, z)`` of the rotated array."""
        values = tuple(float(v) for v in spacing)
        if len(values) != 3:
            raise AugmentationError(f"spacing must have 3 components, got {spacing!r}")
        return tuple(values[self.perm[i]] for i in range(3))  # type: ignore[return-value]

    def preserves_grid(
        self, volume_shape: Sequence[int], spacing: Sequence[float] = (1.0, 1.0, 1.0)
    ) -> bool:
        """True iff the rotated array has the same shape *and* spacing.

        A rotation that permutes two axes of different length, or of different
        spacing, produces a differently-shaped volume. Batching needs one shape
        per split, so those rotations are simply never drawn - on the project's
        cubic, isotropic grids every one of the 24 rotations qualifies.
        """
        return (
            self.transform_shape(volume_shape) == tuple(int(v) for v in volume_shape)
            and self.transform_spacing(spacing) == tuple(float(v) for v in spacing)
        )

    def transform_point_world(
        self,
        point: Sequence[float],
        *,
        volume_shape: Sequence[int],
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> tuple[float, float, float]:
        """Map a world point through the rotation about the volume centre.

        Centres are per-axis ``((N - 1) / 2) * s``, so the rotated centre is just
        the old centre with its components permuted - which makes the map exact
        in floating point: every output component is one input component, offset
        by the difference of two centre coordinates.
        """
        values = tuple(float(v) for v in point)
        if len(values) != 3:
            raise AugmentationError(f"point must have 3 components, got {point!r}")
        sizes = _world_sizes(volume_shape)
        spacings = tuple(float(v) for v in spacing)
        centre = tuple(((sizes[axis] - 1) / 2.0) * spacings[axis] for axis in range(3))
        return tuple(  # type: ignore[return-value]
            centre[self.perm[i]] + self.signs[i] * (values[self.perm[i]] - centre[self.perm[i]])
            for i in range(3)
        )

    def transform_extent_world(self, extent: Sequence[float]) -> tuple[float, float, float]:
        """Bounding-box extents are unsigned, so they only permute."""
        values = tuple(float(v) for v in extent)
        if len(values) != 3:
            raise AugmentationError(f"extent must have 3 components, got {extent!r}")
        return tuple(values[self.perm[i]] for i in range(3))  # type: ignore[return-value]

    # -- arrays ------------------------------------------------------------
    def _transpose_axes(self, ndim: int) -> tuple[int, ...]:
        """Source axis for each destination axis; the last 3 axes are ``(z, y, x)``."""
        if ndim < 3:
            raise AugmentationError(f"expected at least 3 dimensions, got {ndim}")
        lead = ndim - 3
        # Destination array axis `a` is world axis `2 - a`, which takes old world
        # axis `perm[2 - a]`, which lives on old array axis `2 - perm[2 - a]`.
        return tuple(range(lead)) + tuple(
            lead + _array_axis(self.perm[2 - a]) for a in range(3)
        )

    def _flip_axes(self, ndim: int) -> tuple[int, ...]:
        lead = ndim - 3
        return tuple(
            lead + _array_axis(i) for i in range(3) if self.signs[i] < 0
        )

    def apply(self, array: np.ndarray) -> np.ndarray:
        """Rotate a ``(..., D, H, W)`` numpy array. Leading axes are untouched."""
        values = np.asarray(array)
        out = np.transpose(values, self._transpose_axes(values.ndim))
        flip = self._flip_axes(values.ndim)
        if flip:
            out = np.flip(out, axis=flip)
        # np.flip and np.transpose return strided views; torch.from_numpy cannot
        # take a negative stride, so hand back something contiguous. The identity
        # path costs nothing: its view is already contiguous.
        return np.ascontiguousarray(out)

    def apply_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rotate a ``(..., D, H, W)`` torch tensor. Leading axes are untouched."""
        out = tensor.permute(self._transpose_axes(tensor.ndim))
        flip = self._flip_axes(tensor.ndim)
        if flip:
            out = torch.flip(out, dims=flip)
        return out.contiguous()


def all_rotations() -> tuple[AxisRotation, ...]:
    """The 24 elements of the octahedral rotation group, identity first."""
    seen: dict[tuple[tuple[int, ...], tuple[int, ...]], AxisRotation] = {}
    identity = AxisRotation.identity()
    seen[(identity.perm, identity.signs)] = identity
    for x_degrees in (0, 90, 180, 270):
        for y_degrees in (0, 90, 180, 270):
            for z_degrees in (0, 90, 180, 270):
                rotation = AxisRotation.from_angles(x_degrees, y_degrees, z_degrees)
                seen.setdefault((rotation.perm, rotation.signs), rotation)
    return tuple(seen.values())


# ---------------------------------------------------------------------------
# Relation rewrite
# ---------------------------------------------------------------------------
def rewrite_relations(
    rotation: AxisRotation,
    *,
    relations: Sequence[Mapping[str, str]],
    target_centroid_world: Sequence[float],
    anchor_centroids_world: Sequence[Sequence[float]],
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> list[dict[str, str]]:
    """Re-derive one clause triple for the rotated volume.

    The anchors and their order are carried over unchanged; only the directions
    are recomputed, from the rotated centroids, with
    :func:`~src.data.direction_rules.classify_direction` against the rotated
    volume's own shape, spacing and centre plane.

    Args:
        rotation: the pose to rewrite into.
        relations: the stored clause triple, ``{"direction", "anchor"}`` each.
        target_centroid_world: the target centroid before rotation, ``(x, y, z)``.
        anchor_centroids_world: the anchor centroids before rotation, in clause
            order.
        volume_shape: ``(D, H, W)`` before rotation.
        spacing: world units per voxel ``(x, y, z)`` before rotation.

    Returns:
        The rewritten clause triple, in canonical form and in the original
        anchor order.

    Raises:
        RelationRewriteError: the rotated pose has no legal triple - either a
            pair lands on the medial/lateral tie, or two clauses collapse onto
            the same direction. The caller draws another rotation; it must never
            paper over this by inventing a token.
    """
    try:
        canonical = validate_clauses(relations)
    except PromptError as error:
        raise RelationRewriteError(f"the stored clause triple is invalid: {error}") from error
    if len(anchor_centroids_world) != len(canonical):
        raise RelationRewriteError(
            f"got {len(canonical)} clauses but {len(anchor_centroids_world)} anchor centroids"
        )

    rotated_shape = rotation.transform_shape(volume_shape)
    rotated_spacing = rotation.transform_spacing(spacing)
    rotated_target = rotation.transform_point_world(
        target_centroid_world, volume_shape=volume_shape, spacing=spacing
    )

    rewritten: list[dict[str, str]] = []
    for clause, anchor_centroid in zip(canonical, anchor_centroids_world):
        rotated_anchor = rotation.transform_point_world(
            anchor_centroid, volume_shape=volume_shape, spacing=spacing
        )
        try:
            result = classify_direction(
                rotated_target,
                rotated_anchor,
                volume_shape=rotated_shape,
                spacing=rotated_spacing,
            )
        except AmbiguousDirectionError as error:
            raise RelationRewriteError(
                f"rotation {rotation.code} leaves the {clause['anchor']} clause "
                f"without a direction: {error}"
            ) from error
        rewritten.append({"direction": result.direction, "anchor": clause["anchor"]})

    try:
        validate_distinct_directions(clause["direction"] for clause in rewritten)
    except DirectionError as error:
        raise RelationRewriteError(
            f"rotation {rotation.code} collapses the clause triple onto "
            f"{[clause['direction'] for clause in rewritten]}: {error}"
        ) from error
    try:
        return list(validate_clauses(rewritten))
    except PromptError as error:  # pragma: no cover - guarded by the checks above
        raise RelationRewriteError(str(error)) from error


@dataclass(frozen=True)
class RotationPlan:
    """The rotation chosen for one example, with everything it rewrites."""

    rotation: AxisRotation
    relations: tuple[dict[str, str], ...]
    prompt: str
    volume_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    target_centroid_world: tuple[float, float, float]
    anchor_centroids_world: tuple[tuple[float, float, float], ...]
    anchor_extents_world: tuple[tuple[float, float, float], ...]
    attempts: int
    rejected: tuple[str, ...] = ()
    #: True only when `max_attempts` draws were all rejected and the identity was
    #: forced. The identity is also a legal *draw*, so it is not enough to test
    #: that the pose is unchanged: an example whose first draw was rejected and
    #: whose second was the identity has been augmented exactly as intended.
    exhausted: bool = False

    @property
    def fell_back_to_identity(self) -> bool:
        """True when every drawn rotation was rejected and the pose was forced."""
        return self.exhausted

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(clause["direction"] for clause in self.relations)

    @property
    def anchors(self) -> tuple[str, ...]:
        return tuple(clause["anchor"] for clause in self.relations)


class RotationAugmentation:
    """Draws a rotation per (example, epoch) and rewrites the relations with it.

    Args:
        volume_shape: ``(D, H, W)`` of the split being trained on.
        spacing: world units per voxel ``(x, y, z)``.
        seed: base seed; the rotation of ``(index, epoch)`` derives from it.
        angles: the per-axis draw, in degrees. Every entry must be a multiple
            of 90.
        axes: which world axes are rotated about. Dropping one restricts the
            group without changing anything else.
        max_attempts: how many rotations to draw before falling back to the
            identity when the rewrite keeps failing.

    Raises:
        AugmentationError: the configured angles cannot produce anything but the
            identity on this grid - which would make the augmentation a silent
            no-op for the whole run.
    """

    def __init__(
        self,
        *,
        volume_shape: Sequence[int],
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        seed: int = 0,
        angles: Sequence[int] = ROTATION_ANGLES,
        axes: Sequence[str] = AXES,
        max_attempts: int = 8,
    ) -> None:
        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.spacing = tuple(float(v) for v in spacing)
        self.seed = int(seed)
        self.angles = tuple(int(a) for a in angles)
        if not self.angles:
            raise AugmentationError("angles must not be empty")
        for angle in self.angles:
            if angle % 90 != 0:
                raise AugmentationError(f"angles must be multiples of 90, got {angle}")
        self.axes = tuple(str(a) for a in axes)
        for axis in self.axes:
            if axis not in AXIS_INDEX:
                raise AugmentationError(f"unknown axis {axis!r}; world axes are {AXES}")
        self.max_attempts = max(int(max_attempts), 1)

        self.rotations = tuple(
            rotation
            for rotation in self._reachable_rotations()
            if rotation.preserves_grid(self.volume_shape, self.spacing)
        )
        if len(self.rotations) <= 1:
            raise AugmentationError(
                f"the configured angles {self.angles} on axes {self.axes} reach only the "
                f"identity for volume_shape={self.volume_shape} spacing={self.spacing}; "
                "a 90-degree rotation needs a grid whose axes share their length and "
                "spacing. Disable the augmentation or fix the grid."
            )

        # Shared memory, not a plain int: the dataloader hands each persistent
        # worker its own copy of the dataset, and a copied int would freeze at
        # epoch 0 in every worker while the main process advanced on its own.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.stats: dict[str, int] = {"planned": 0, "rejected": 0, "fallbacks": 0}

    # -- group -------------------------------------------------------------
    def _reachable_rotations(self) -> tuple[AxisRotation, ...]:
        """Every distinct pose the configured angles and axes can compose to."""
        seen: dict[tuple[tuple[int, ...], tuple[int, ...]], AxisRotation] = {}
        for angles in self._angle_grid():
            rotation = self._compose(angles)
            seen.setdefault((rotation.perm, rotation.signs), rotation)
        return tuple(seen.values())

    def _angle_grid(self) -> Iterator[tuple[int, int, int]]:
        for x_degrees in self.angles if "x" in self.axes else (0,):
            for y_degrees in self.angles if "y" in self.axes else (0,):
                for z_degrees in self.angles if "z" in self.axes else (0,):
                    yield x_degrees, y_degrees, z_degrees

    def _compose(self, angles: tuple[int, int, int]) -> AxisRotation:
        return AxisRotation.from_angles(*angles)

    # -- epoch -------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        """Advance the pose stream. Call it *before* iterating the dataloader.

        Persistent workers only re-read the dataset between epochs, so setting
        the epoch after the iterator exists would leave the first prefetched
        batches on the previous pose.
        """
        self._epoch.fill_(int(epoch))

    # -- sampling ----------------------------------------------------------
    def _rng(self, index: int, epoch: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, int(epoch), int(index)])
        )

    def _draw(self, rng: np.random.Generator) -> AxisRotation:
        """One draw: an independent angle per configured axis, composed."""
        angles = tuple(
            int(rng.choice(self.angles)) if axis in self.axes else 0 for axis in AXES
        )
        return self._compose(angles)  # type: ignore[arg-type]

    def sample(self, index: int, epoch: int | None = None) -> AxisRotation:
        """The first grid-preserving rotation drawn for ``(index, epoch)``.

        Stage A uses this directly: a scene volume and its per-shape masks rotate
        together, and there are no relations to rewrite.
        """
        rng = self._rng(index, self.epoch if epoch is None else epoch)
        for _ in range(64):
            rotation = self._draw(rng)
            if rotation.preserves_grid(self.volume_shape, self.spacing):
                return rotation
        return AxisRotation.identity()  # pragma: no cover - `rotations` is non-empty

    # -- planning ----------------------------------------------------------
    def plan(
        self,
        metadata,
        index: int,
        epoch: int | None = None,
    ) -> RotationPlan:
        """Choose a rotation for one Stage B example and rewrite its prompt.

        Draws up to ``max_attempts`` rotations and keeps the first whose rewrite
        is legal. The identity is the fallback, and it is always legal: it
        reproduces the stored prompt.

        Args:
            metadata: an :class:`~src.data.schema.ExampleMetadata`, or anything
                exposing ``relations``, ``target_centroid_world``,
                ``anchor_centroids_world``, ``anchor_extents_world``,
                ``volume_shape`` and ``spacing``.
            index: the example's position in the dataset. Stable across epochs
                and independent of shuffling, so the pose stream is too.
            epoch: defaults to the shared epoch counter.
        """
        rng = self._rng(index, self.epoch if epoch is None else epoch)
        rejected: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            rotation = self._draw(rng)
            if not rotation.preserves_grid(metadata.volume_shape, metadata.spacing):
                rejected.append(f"{rotation.code}: changes the grid")
                continue
            try:
                relations = rewrite_relations(
                    rotation,
                    relations=metadata.relations,
                    target_centroid_world=metadata.target_centroid_world,
                    anchor_centroids_world=metadata.anchor_centroids_world,
                    volume_shape=metadata.volume_shape,
                    spacing=metadata.spacing,
                )
            except RelationRewriteError as error:
                rejected.append(str(error))
                continue
            self.stats["planned"] += 1
            self.stats["rejected"] += len(rejected)
            return self._build_plan(rotation, relations, metadata, attempt, rejected)

        # Every draw was rejected. The identity always has a legal rewrite - it
        # is the stored one - so the example is still trained on, unaugmented.
        self.stats["planned"] += 1
        self.stats["rejected"] += len(rejected)
        self.stats["fallbacks"] += 1
        identity = AxisRotation.identity()
        relations = rewrite_relations(
            identity,
            relations=metadata.relations,
            target_centroid_world=metadata.target_centroid_world,
            anchor_centroids_world=metadata.anchor_centroids_world,
            volume_shape=metadata.volume_shape,
            spacing=metadata.spacing,
        )
        return self._build_plan(
            identity, relations, metadata, self.max_attempts, rejected, exhausted=True
        )

    def _build_plan(
        self,
        rotation: AxisRotation,
        relations: Sequence[Mapping[str, str]],
        metadata,
        attempts: int,
        rejected: Sequence[str],
        *,
        exhausted: bool = False,
    ) -> RotationPlan:
        shape, spacing = metadata.volume_shape, metadata.spacing
        return RotationPlan(
            rotation=rotation,
            relations=tuple(dict(clause) for clause in relations),
            prompt=render_prompt(relations),
            volume_shape=rotation.transform_shape(shape),
            spacing=rotation.transform_spacing(spacing),
            target_centroid_world=rotation.transform_point_world(
                metadata.target_centroid_world, volume_shape=shape, spacing=spacing
            ),
            anchor_centroids_world=tuple(
                rotation.transform_point_world(centroid, volume_shape=shape, spacing=spacing)
                for centroid in metadata.anchor_centroids_world
            ),
            anchor_extents_world=tuple(
                rotation.transform_extent_world(extent)
                for extent in metadata.anchor_extents_world
            ),
            attempts=attempts,
            rejected=tuple(rejected),
            exhausted=exhausted,
        )

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        planned = max(self.stats["planned"], 1)
        return (
            f"rotation_90: {len(self.rotations)} poses over axes {''.join(self.axes)} "
            f"at {self.angles} degrees, seed {self.seed}, epoch {self.epoch}; "
            f"{self.stats['rejected'] / planned:.2f} rejected draws per example, "
            f"{self.stats['fallbacks']} identity fallback(s)"
        )


def build_rotation_augmentation(
    config: Mapping[str, object],
    *,
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    seed: int = 0,
    stage: str = "stage_b",
) -> RotationAugmentation | None:
    """Build the augmentation from the ``augmentations`` block of ``train.yaml``.

    Returns ``None`` when the block, the ``rotation_90`` entry, or this stage's
    ``apply_to`` flag is off - the caller then trains unaugmented.
    """
    if not config.get("enabled", False):
        return None
    settings = config.get("rotation_90")
    if not isinstance(settings, Mapping) or not settings.get("enabled", False):
        return None
    if not settings.get("rewrite_tested", False):
        raise AugmentationError(
            "rotation_90 is enabled but rewrite_tested is false; see tests/test_augmentations.py"
        )
    apply_to = settings.get("apply_to", {})
    if isinstance(apply_to, Mapping) and not apply_to.get(stage, False):
        return None
    return RotationAugmentation(
        volume_shape=volume_shape,
        spacing=spacing,
        seed=seed,
        angles=tuple(settings.get("angles", ROTATION_ANGLES)),  # type: ignore[arg-type]
        axes=tuple(settings.get("axes", AXES)),  # type: ignore[arg-type]
        max_attempts=int(settings.get("max_attempts", 8)),  # type: ignore[arg-type]
    )
