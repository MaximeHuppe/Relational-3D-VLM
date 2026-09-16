"""The fixed shape vocabulary.

Ten primitives, one instance of each per scene, stable integer IDs 1..10 with 0
reserved for background. The vocabulary is declared once in
``configs/shapes.yaml``; this module loads it, checks its internal consistency
and exposes it as the only shape-name/ID authority for the whole project.

Because every scene contains exactly one instance of every class, a shape name
is a unique in-scene identifier, which is why prompts can use bare shape names
as anchor identifiers.

Voxelisation of these primitives lives in :mod:`src.data.voxelization`
(Phase 1); this module only owns the vocabulary and its parameter ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.config import load_config
from src.data.direction_rules import DIRECTIONS, DIRECTION_OPPOSITES, AXIS_OF_DIRECTION

#: Canonical ordering of the vocabulary; index + 1 is the class ID.
CANONICAL_SHAPE_ORDER: tuple[str, ...] = (
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

BACKGROUND_LABEL = 0
NUM_SHAPE_CLASSES = len(CANONICAL_SHAPE_ORDER)


class VocabularyError(ValueError):
    """Raised when the configured vocabulary violates the project contract."""


@dataclass(frozen=True)
class ShapeSpec:
    """One primitive: its stable identity and its sampling ranges."""

    id: int
    name: str
    family: str
    params: Mapping[str, tuple[float, float]]
    axis: str | None = None
    min_anisotropy_ratio: float | None = None
    thin_axis_exception: str | None = None
    mean_volume_voxels: float | None = None

    def param_range(self, param: str) -> tuple[float, float]:
        """Inclusive ``(low, high)`` sampling range of one shape parameter."""
        if param not in self.params:
            raise KeyError(
                f"shape {self.name!r} has no parameter {param!r}; "
                f"known parameters: {sorted(self.params)}"
            )
        return self.params[param]

    def scaled_params(
        self, axis_length: int, reference_axis_length: int
    ) -> dict[str, tuple[float, float]]:
        """Parameter ranges rescaled for an escalated grid (72^3, 80^3, ...)."""
        if axis_length <= 0 or reference_axis_length <= 0:
            raise ValueError("axis lengths must be positive")
        factor = axis_length / reference_axis_length
        return {
            name: (low * factor, high * factor) for name, (low, high) in self.params.items()
        }


@dataclass(frozen=True)
class ShapeVocabulary:
    """The complete, validated ten-class vocabulary."""

    version: str
    background_label: int
    reference_axis_length: int
    axis_extent_fraction_range: tuple[float, float]
    shapes: tuple[ShapeSpec, ...]

    # -- lookups -----------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """Shape names in canonical (ID) order."""
        return tuple(shape.name for shape in self.shapes)

    @property
    def ids(self) -> tuple[int, ...]:
        """Class IDs in canonical order."""
        return tuple(shape.id for shape in self.shapes)

    def __len__(self) -> int:
        return len(self.shapes)

    def __iter__(self):
        return iter(self.shapes)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def by_name(self, name: str) -> ShapeSpec:
        for shape in self.shapes:
            if shape.name == name:
                return shape
        raise KeyError(f"unknown shape name {name!r}; vocabulary is {self.names}")

    def by_id(self, shape_id: int) -> ShapeSpec:
        for shape in self.shapes:
            if shape.id == shape_id:
                return shape
        raise KeyError(f"unknown shape id {shape_id!r}; vocabulary ids are {self.ids}")

    def name_to_id(self, name: str) -> int:
        return self.by_name(name).id

    def id_to_name(self, shape_id: int) -> str:
        return self.by_id(shape_id).name

    def index_of(self, name: str) -> int:
        """Zero-based index, for embedding tables (``id - 1``)."""
        return self.names.index(self.by_name(name).name)

    def is_valid_name(self, name: str) -> bool:
        return name in self.names

    def require_names(self, names: Sequence[str]) -> tuple[str, ...]:
        """Validate a sequence of shape names, returning it unchanged."""
        unknown = [name for name in names if not self.is_valid_name(name)]
        if unknown:
            raise VocabularyError(
                f"unknown shape name(s) {unknown!r}; vocabulary is {self.names}"
            )
        return tuple(names)


def _parse_range(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise VocabularyError(f"{context} must be a [low, high] pair, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not low <= high:
        raise VocabularyError(f"{context} must satisfy low <= high, got {value!r}")
    if low <= 0:
        raise VocabularyError(f"{context} must be strictly positive, got {value!r}")
    return low, high


def _build_shape(entry: Mapping[str, Any]) -> ShapeSpec:
    for key in ("id", "name", "family", "params"):
        if key not in entry:
            raise VocabularyError(f"shape entry {entry!r} is missing {key!r}")
    name = str(entry["name"])
    params = entry["params"]
    if not isinstance(params, Mapping) or not params:
        raise VocabularyError(f"shape {name!r} must declare a non-empty params mapping")
    parsed = {
        str(param): _parse_range(value, f"shape {name!r} param {param!r}")
        for param, value in params.items()
    }
    axis = entry.get("axis")
    if axis is not None and axis not in ("x", "y", "z"):
        raise VocabularyError(f"shape {name!r} has invalid axis {axis!r}")
    return ShapeSpec(
        id=int(entry["id"]),
        name=name,
        family=str(entry["family"]),
        params=parsed,
        axis=None if axis is None else str(axis),
        min_anisotropy_ratio=(
            None
            if entry.get("min_anisotropy_ratio") is None
            else float(entry["min_anisotropy_ratio"])
        ),
        thin_axis_exception=(
            None
            if entry.get("thin_axis_exception") is None
            else str(entry["thin_axis_exception"])
        ),
        mean_volume_voxels=(
            None
            if entry.get("mean_volume_voxels") is None
            else float(entry["mean_volume_voxels"])
        ),
    )


def _validate_directions(config: Mapping[str, Any]) -> None:
    """The config must agree with :mod:`src.data.direction_rules`."""
    configured = tuple(config.get("directions", ()))
    if set(configured) != set(DIRECTIONS):
        raise VocabularyError(
            f"configured directions {configured!r} do not match the rule module "
            f"{DIRECTIONS!r}"
        )
    opposites = dict(config.get("direction_opposites", {}))
    if opposites != DIRECTION_OPPOSITES:
        raise VocabularyError(
            f"configured direction_opposites {opposites!r} do not match "
            f"{DIRECTION_OPPOSITES!r}"
        )
    axes = dict(config.get("direction_axis", {}))
    if axes != AXIS_OF_DIRECTION:
        raise VocabularyError(
            f"configured direction_axis {axes!r} do not match {AXIS_OF_DIRECTION!r}"
        )


def build_vocabulary(config: Mapping[str, Any]) -> ShapeVocabulary:
    """Build and fully validate a vocabulary from a parsed ``shapes.yaml``."""
    shapes = tuple(_build_shape(entry) for entry in config.get("shapes", ()))

    if len(shapes) != NUM_SHAPE_CLASSES:
        raise VocabularyError(
            f"expected exactly {NUM_SHAPE_CLASSES} shape classes, got {len(shapes)}"
        )
    declared = config.get("num_classes")
    if declared is not None and int(declared) != NUM_SHAPE_CLASSES:
        raise VocabularyError(
            f"num_classes {declared!r} disagrees with {NUM_SHAPE_CLASSES}"
        )

    names = [shape.name for shape in shapes]
    if len(set(names)) != len(names):
        raise VocabularyError(f"shape names must be unique, got {names!r}")
    if tuple(names) != CANONICAL_SHAPE_ORDER:
        raise VocabularyError(
            f"shape order {tuple(names)!r} does not match the canonical order "
            f"{CANONICAL_SHAPE_ORDER!r}"
        )

    ids = [shape.id for shape in shapes]
    if ids != list(range(1, NUM_SHAPE_CLASSES + 1)):
        raise VocabularyError(
            f"shape ids must be contiguous 1..{NUM_SHAPE_CLASSES} in canonical "
            f"order, got {ids!r}"
        )

    background = int(config.get("background_label", BACKGROUND_LABEL))
    if background != BACKGROUND_LABEL:
        raise VocabularyError(f"background_label must be {BACKGROUND_LABEL}")
    if background in ids:
        raise VocabularyError("background label collides with a shape id")

    _validate_directions(config)

    return ShapeVocabulary(
        version=str(config.get("vocabulary_version", "0.0.0")),
        background_label=background,
        reference_axis_length=int(config.get("reference_axis_length", 64)),
        axis_extent_fraction_range=_parse_range(
            config.get("axis_extent_fraction_range", [0.08, 0.22]),
            "axis_extent_fraction_range",
        ),
        shapes=shapes,
    )


def load_vocabulary() -> ShapeVocabulary:
    """Load the project vocabulary from ``configs/shapes.yaml``."""
    return build_vocabulary(load_config("shapes"))


#: Process-wide vocabulary. Import this rather than re-reading the config.
SHAPE_VOCABULARY: ShapeVocabulary = load_vocabulary()
SHAPE_NAMES: tuple[str, ...] = SHAPE_VOCABULARY.names
SHAPE_IDS: tuple[int, ...] = SHAPE_VOCABULARY.ids
VOCABULARY_VERSION: str = SHAPE_VOCABULARY.version
