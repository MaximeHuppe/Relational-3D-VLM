"""Dataset schema: the on-disk and in-memory contract for one example.

An *example* is one ``(scene, target)`` pair. Every scene yields ten candidate
examples, one per instance; the target-class split filter decides which of them
are actually used for training, validation and testing.

Layering
--------
``ExampleMetadata``
    Everything JSON-serialisable: identities, seeds, geometry descriptors, the
    structured prompt and the rendered prompt. This is what the manifest holds.
``ExampleArrays``
    The five volumes: ``scene_volume``, ``instance_labels``, ``target_mask``,
    ``anchor_masks`` and ``anchor_union_mask``.
``Example``
    Metadata plus arrays; exposes every field named in
    :data:`REQUIRED_EXAMPLE_FIELDS`.

Storage
-------
``scene_volume`` (float MRI-like image) and ``instance_labels`` are identical
for the ten examples of a scene, so they are written once per scene as RAS
NIfTI files under ``scenes/<scene_id>/``. The per-example manifest is a JSONL
file of ``ExampleMetadata``. Masks are materialised from ``instance_labels`` by
the declared instance IDs; inspection copies are also dumped under
``examples/<example_id>/``. :meth:`ExampleArrays.validate` re-runs every
consistency check at load time. See ``docs/dataset_schema.md``.

Stage B contract
----------------
:data:`STAGE_B_FORBIDDEN_FIELDS` names the fields the relational model must
never see: instance ids and every ``target_*`` field. Binary occupancy derived
from ``instance_labels`` (no instance colours, not the intensity image) is the
decoder-side WHAT stream. :func:`stage_b_inputs` returns only the permitted
subset. The intensity ``scene_volume`` is a Stage A input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.data.direction_rules import DIRECTION_RULE_VERSION, validate_distinct_directions
from src.data.nifti_io import load_nifti, save_nifti
from src.data.primitives import SHAPE_VOCABULARY, VOCABULARY_VERSION
from src.data.prompt_generator import (
    NUM_CLAUSES,
    PromptError,
    parse_prompt,
    render_prompt,
    validate_clause,
    validate_clauses,
)

SCHEMA_VERSION = "2.0.0"

NUM_ANCHORS = 3
BACKGROUND_LABEL = SHAPE_VOCABULARY.background_label
SHAPES_PER_SCENE = len(SHAPE_VOCABULARY)

#: Every field a complete example must expose, exactly as specified in CLAUDE.md.
REQUIRED_EXAMPLE_FIELDS: tuple[str, ...] = (
    "scene_id",
    "example_id",
    "seed",
    "volume_shape",
    "spacing",
    "scene_volume",
    "instance_labels",
    "target_instance_id",
    "target_shape_name",
    "target_mask",
    "anchor_instance_ids",
    "anchor_shape_names",
    "anchor_masks",
    "anchor_union_mask",
    "relations",
    "anchor_centroids_world",
    "anchor_extents_world",
    "target_centroid_world",
    "prompt",
    "structured_prompt",
    "generator_version",
)

#: Provenance fields carried in addition to the required set.
VERSION_FIELDS: tuple[str, ...] = (
    "generator_version",
    "direction_rule_version",
    "vocabulary_version",
    "schema_version",
)

#: Stage B must never receive these, in any form. Binary occupancy derived from
#: labels is allowed as the decoder WHAT stream; instance ids and every target
#: identity field are not. The intensity ``scene_volume`` is a Stage A input.
STAGE_B_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "instance_labels",
    "target_mask",
    "target_shape_name",
    "target_instance_id",
    "target_centroid_world",
)

#: The union mask is an ablation-baseline input only, never the main input.
STAGE_B_ABLATION_ONLY_FIELDS: tuple[str, ...] = ("anchor_union_mask",)

#: What the relational model is allowed to consume. ``scene_volume`` here is the
#: binary occupancy stream the decoder still names that way (see
#: :func:`src.data.dataset.stage_b_model_inputs`); it is not the intensity image.
STAGE_B_ALLOWED_FIELDS: tuple[str, ...] = (
    "anchor_masks",
    "scene_volume",
    "structured_prompt",
    "prompt",
    "anchor_shape_names",
    "anchor_centroids_world",
    "anchor_extents_world",
    "volume_shape",
    "spacing",
)

SCENE_VOLUME_FILENAME = "scene_volume.nii.gz"
INSTANCE_LABELS_FILENAME = "instance_labels.nii.gz"
TARGET_MASK_FILENAME = "target_mask.nii.gz"
ANCHOR_UNION_FILENAME = "anchor_union.nii.gz"


class SchemaError(ValueError):
    """Raised when an example violates the dataset contract."""


# --------------------------------------------------------------------------
# Structured prompt
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    """One ``{"direction": ..., "anchor": ...}`` clause."""

    direction: str
    anchor: str

    def __post_init__(self) -> None:
        try:
            canonical = validate_clause({"direction": self.direction, "anchor": self.anchor})
        except PromptError as error:
            raise SchemaError(str(error)) from error
        object.__setattr__(self, "anchor", canonical["anchor"])

    def to_dict(self) -> dict[str, str]:
        return {"direction": self.direction, "anchor": self.anchor}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "Relation":
        if set(data) != {"direction", "anchor"}:
            raise SchemaError(
                f"relation must have exactly the keys {{'direction', 'anchor'}}, "
                f"got {sorted(data)}"
            )
        return cls(direction=str(data["direction"]), anchor=str(data["anchor"]))


@dataclass(frozen=True)
class StructuredPrompt:
    """The three ordered clauses: the primary model interface."""

    relations: tuple[Relation, ...]

    def __post_init__(self) -> None:
        if len(self.relations) != NUM_CLAUSES:
            raise SchemaError(
                f"a structured prompt has exactly {NUM_CLAUSES} clauses, "
                f"got {len(self.relations)}"
            )
        try:
            validate_clauses([relation.to_dict() for relation in self.relations])
        except PromptError as error:
            raise SchemaError(str(error)) from error

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(relation.direction for relation in self.relations)

    @property
    def anchors(self) -> tuple[str, ...]:
        return tuple(relation.anchor for relation in self.relations)

    def to_list(self) -> list[dict[str, str]]:
        """The canonical structured representation."""
        return [relation.to_dict() for relation in self.relations]

    def render(self) -> str:
        """The canonical natural-language prompt."""
        return render_prompt(self.to_list())

    @classmethod
    def from_list(cls, clauses: Sequence[Mapping[str, str]]) -> "StructuredPrompt":
        canonical = validate_clauses(clauses)
        return cls(relations=tuple(Relation.from_dict(clause) for clause in canonical))

    @classmethod
    def from_text(cls, prompt: str) -> "StructuredPrompt":
        return cls.from_list(parse_prompt(prompt))

    def with_permuted_clauses(self, order: Sequence[int]) -> "StructuredPrompt":
        """Reorder clauses (counterfactual test helper)."""
        if sorted(order) != list(range(NUM_CLAUSES)):
            raise SchemaError(f"order must be a permutation of 0..{NUM_CLAUSES - 1}")
        return StructuredPrompt(relations=tuple(self.relations[i] for i in order))


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def _as_int_triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise SchemaError(f"{name} must have 3 components, got {value!r}")
    if any(v <= 0 for v in values):
        raise SchemaError(f"{name} must be positive, got {value!r}")
    return values  # type: ignore[return-value]


def _as_float_triple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise SchemaError(f"{name} must have 3 components, got {value!r}")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class ExampleMetadata:
    """JSON-serialisable part of an example; one line of the manifest."""

    scene_id: str
    example_id: str
    seed: int
    volume_shape: tuple[int, int, int]  # (D, H, W) == (z, y, x)
    spacing: tuple[float, float, float]  # world units per voxel, (x, y, z)
    target_instance_id: int
    target_shape_name: str
    target_centroid_world: tuple[float, float, float]
    anchor_instance_ids: tuple[int, ...]
    anchor_shape_names: tuple[str, ...]
    anchor_centroids_world: tuple[tuple[float, float, float], ...]
    anchor_extents_world: tuple[tuple[float, float, float], ...]
    structured_prompt: StructuredPrompt
    prompt: str
    generator_version: str
    direction_rule_version: str = DIRECTION_RULE_VERSION
    vocabulary_version: str = VOCABULARY_VERSION
    schema_version: str = SCHEMA_VERSION
    split: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "volume_shape", _as_int_triple(self.volume_shape, "volume_shape"))
        spacing = _as_float_triple(self.spacing, "spacing")
        if any(value <= 0 for value in spacing):
            raise SchemaError(f"spacing must be strictly positive, got {spacing!r}")
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(
            self, "target_centroid_world", _as_float_triple(self.target_centroid_world, "target_centroid_world")
        )
        object.__setattr__(self, "anchor_instance_ids", tuple(int(v) for v in self.anchor_instance_ids))
        object.__setattr__(self, "anchor_shape_names", tuple(str(v) for v in self.anchor_shape_names))
        object.__setattr__(
            self,
            "anchor_centroids_world",
            tuple(_as_float_triple(row, "anchor_centroids_world row") for row in self.anchor_centroids_world),
        )
        object.__setattr__(
            self,
            "anchor_extents_world",
            tuple(_as_float_triple(row, "anchor_extents_world row") for row in self.anchor_extents_world),
        )
        self.validate()

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Fail fast on any contract violation. Never repair silently."""
        for name, values in (
            ("anchor_instance_ids", self.anchor_instance_ids),
            ("anchor_shape_names", self.anchor_shape_names),
            ("anchor_centroids_world", self.anchor_centroids_world),
            ("anchor_extents_world", self.anchor_extents_world),
        ):
            if len(values) != NUM_ANCHORS:
                raise SchemaError(f"{name} must have {NUM_ANCHORS} entries, got {len(values)}")

        SHAPE_VOCABULARY.require_names(self.anchor_shape_names)
        SHAPE_VOCABULARY.require_names([self.target_shape_name])

        if len(set(self.anchor_instance_ids)) != NUM_ANCHORS:
            raise SchemaError(f"anchor instance ids must be distinct, got {self.anchor_instance_ids}")
        if self.target_instance_id in self.anchor_instance_ids:
            raise SchemaError(
                f"target instance {self.target_instance_id} appears among the anchors "
                f"{self.anchor_instance_ids}"
            )
        if self.target_shape_name in self.anchor_shape_names:
            raise SchemaError(
                f"target shape {self.target_shape_name!r} appears among the anchor shapes "
                f"{self.anchor_shape_names}"
            )
        if self.target_instance_id == BACKGROUND_LABEL:
            raise SchemaError("target instance id must not be the background label")
        if BACKGROUND_LABEL in self.anchor_instance_ids:
            raise SchemaError("an anchor instance id must not be the background label")

        # Anchor order must be identical in the mask channels, the structured
        # prompt and the rendered text.
        if self.structured_prompt.anchors != self.anchor_shape_names:
            raise SchemaError(
                f"anchor order mismatch: channels {self.anchor_shape_names} vs prompt "
                f"{self.structured_prompt.anchors}"
            )
        validate_distinct_directions(self.structured_prompt.directions)

        # The prompt must round-trip from the structured metadata.
        rendered = self.structured_prompt.render()
        if rendered != self.prompt:
            raise SchemaError(
                f"prompt does not round-trip from structured metadata:\n"
                f"  stored:   {self.prompt!r}\n  rendered: {rendered!r}"
            )
        if StructuredPrompt.from_text(self.prompt).to_list() != self.structured_prompt.to_list():
            raise SchemaError(f"prompt does not parse back to the structured prompt: {self.prompt!r}")

        for name, version in (
            ("direction_rule_version", self.direction_rule_version),
            ("vocabulary_version", self.vocabulary_version),
            ("schema_version", self.schema_version),
            ("generator_version", self.generator_version),
        ):
            if not version:
                raise SchemaError(f"{name} must be recorded")

    # -- convenience -------------------------------------------------------
    @property
    def relations(self) -> list[dict[str, str]]:
        """``relations[3]`` as required by the schema."""
        return self.structured_prompt.to_list()

    @property
    def anchor_shape_ids(self) -> tuple[int, ...]:
        return tuple(SHAPE_VOCABULARY.name_to_id(name) for name in self.anchor_shape_names)

    # -- serialisation -----------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "example_id": self.example_id,
            "seed": int(self.seed),
            "volume_shape": list(self.volume_shape),
            "spacing": list(self.spacing),
            "target_instance_id": int(self.target_instance_id),
            "target_shape_name": self.target_shape_name,
            "target_centroid_world": list(self.target_centroid_world),
            "anchor_instance_ids": list(self.anchor_instance_ids),
            "anchor_shape_names": list(self.anchor_shape_names),
            "anchor_centroids_world": [list(row) for row in self.anchor_centroids_world],
            "anchor_extents_world": [list(row) for row in self.anchor_extents_world],
            "relations": self.relations,
            "prompt": self.prompt,
            "generator_version": self.generator_version,
            "direction_rule_version": self.direction_rule_version,
            "vocabulary_version": self.vocabulary_version,
            "schema_version": self.schema_version,
            "split": self.split,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "ExampleMetadata":
        return cls(
            scene_id=str(data["scene_id"]),
            example_id=str(data["example_id"]),
            seed=int(data["seed"]),
            volume_shape=tuple(data["volume_shape"]),
            spacing=tuple(data["spacing"]),
            target_instance_id=int(data["target_instance_id"]),
            target_shape_name=str(data["target_shape_name"]),
            target_centroid_world=tuple(data["target_centroid_world"]),
            anchor_instance_ids=tuple(data["anchor_instance_ids"]),
            anchor_shape_names=tuple(data["anchor_shape_names"]),
            anchor_centroids_world=tuple(tuple(row) for row in data["anchor_centroids_world"]),
            anchor_extents_world=tuple(tuple(row) for row in data["anchor_extents_world"]),
            structured_prompt=StructuredPrompt.from_list(data["relations"]),
            prompt=str(data["prompt"]),
            generator_version=str(data["generator_version"]),
            direction_rule_version=str(data.get("direction_rule_version", DIRECTION_RULE_VERSION)),
            vocabulary_version=str(data.get("vocabulary_version", VOCABULARY_VERSION)),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            split=data.get("split"),
            extra=dict(data.get("extra", {})),
        )


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExampleArrays:
    """The five volumes of one example."""

    scene_volume: np.ndarray  # (D, H, W) float MRI-like intensities
    instance_labels: np.ndarray  # (D, H, W), 0 = background, 1..10 = instances
    target_mask: np.ndarray  # (D, H, W) binary
    anchor_masks: np.ndarray  # (3, D, H, W) binary, ordered by clause slot
    anchor_union_mask: np.ndarray  # (D, H, W) binary, ablation baseline only

    @property
    def occupancy(self) -> np.ndarray:
        """Binary foreground of all ten shapes, derived from labels not the image."""
        return (np.asarray(self.instance_labels) != BACKGROUND_LABEL).astype(np.uint8)

    def validate(self, metadata: ExampleMetadata) -> None:
        """Check every array against the metadata. Fail fast, never repair."""
        shape = metadata.volume_shape
        for name in ("scene_volume", "instance_labels", "target_mask", "anchor_union_mask"):
            array = getattr(self, name)
            if tuple(array.shape) != shape:
                raise SchemaError(f"{name} has shape {tuple(array.shape)}, expected {shape}")
        if tuple(self.anchor_masks.shape) != (NUM_ANCHORS,) + shape:
            raise SchemaError(
                f"anchor_masks has shape {tuple(self.anchor_masks.shape)}, "
                f"expected {(NUM_ANCHORS,) + shape}"
            )

        labels = np.unique(self.instance_labels)
        expected_labels = np.arange(BACKGROUND_LABEL, SHAPES_PER_SCENE + 1)
        if not np.array_equal(labels, expected_labels):
            raise SchemaError(
                f"instance_labels must contain exactly {SHAPES_PER_SCENE} non-empty "
                f"instances plus background; found labels {labels.tolist()}"
            )

        foreground = np.asarray(self.instance_labels) != BACKGROUND_LABEL
        image = np.asarray(self.scene_volume, dtype=np.float32)
        if foreground.any() and (~foreground).any():
            if float(image[foreground].mean()) == float(image[~foreground].mean()):
                raise SchemaError(
                    "scene_volume mean intensity inside structures must differ from background"
                )

        expected_target = self.instance_labels == metadata.target_instance_id
        if not np.array_equal(np.asarray(self.target_mask, dtype=bool), expected_target):
            raise SchemaError("target_mask does not match its declared target instance id")
        if not expected_target.any():
            raise SchemaError("target_mask is empty")

        for slot, instance_id in enumerate(metadata.anchor_instance_ids):
            expected = self.instance_labels == instance_id
            channel = np.asarray(self.anchor_masks[slot], dtype=bool)
            if not np.array_equal(channel, expected):
                raise SchemaError(
                    f"anchor channel {slot} does not match its declared anchor instance "
                    f"{instance_id} ({metadata.anchor_shape_names[slot]})"
                )
            if not expected.any():
                raise SchemaError(f"anchor channel {slot} is empty")
            if np.any(channel & np.asarray(self.target_mask, dtype=bool)):
                raise SchemaError(f"the target appears inside anchor channel {slot}")

        union = np.any(np.asarray(self.anchor_masks, dtype=bool), axis=0)
        if not np.array_equal(np.asarray(self.anchor_union_mask, dtype=bool), union):
            raise SchemaError("anchor_union_mask is not the union of the three anchor channels")

    @classmethod
    def from_scene(
        cls,
        metadata: ExampleMetadata,
        scene_volume: np.ndarray,
        instance_labels: np.ndarray,
    ) -> "ExampleArrays":
        """Materialise the per-example masks from the shared scene arrays."""
        labels = np.asarray(instance_labels)
        target_mask = (labels == metadata.target_instance_id).astype(np.uint8)
        anchor_masks = np.stack(
            [(labels == instance_id).astype(np.uint8) for instance_id in metadata.anchor_instance_ids]
        )
        union = (anchor_masks.max(axis=0) > 0).astype(np.uint8)
        arrays = cls(
            scene_volume=np.asarray(scene_volume),
            instance_labels=labels,
            target_mask=target_mask,
            anchor_masks=anchor_masks,
            anchor_union_mask=union,
        )
        arrays.validate(metadata)
        return arrays


@dataclass(frozen=True)
class Example:
    """A complete example: metadata plus arrays."""

    metadata: ExampleMetadata
    arrays: ExampleArrays

    def __post_init__(self) -> None:
        self.arrays.validate(self.metadata)

    def field(self, name: str) -> Any:
        """Access any field named in :data:`REQUIRED_EXAMPLE_FIELDS`."""
        if hasattr(self.arrays, name):
            return getattr(self.arrays, name)
        if name == "structured_prompt":
            return self.metadata.structured_prompt.to_list()
        if hasattr(self.metadata, name):
            return getattr(self.metadata, name)
        raise SchemaError(f"unknown example field {name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {name: self.field(name) for name in REQUIRED_EXAMPLE_FIELDS}


def assert_complete(example: Example) -> None:
    """Assert every required field is present and non-empty."""
    for name in REQUIRED_EXAMPLE_FIELDS:
        value = example.field(name)
        if value is None:
            raise SchemaError(f"required field {name!r} is missing")
        if isinstance(value, (str, bytes)) and not value:
            raise SchemaError(f"required field {name!r} is empty")
        if isinstance(value, (list, tuple)) and len(value) == 0:
            raise SchemaError(f"required field {name!r} is empty")


def stage_b_inputs(example: Example, *, use_union_mask: bool = False) -> dict[str, Any]:
    """The only data Stage B may consume.

    With ``use_union_mask=True`` the ordered channels are replaced by the union
    mask; that configuration is the ablation baseline, never the main model.
    """
    inputs: dict[str, Any] = {name: example.field(name) for name in STAGE_B_ALLOWED_FIELDS}
    # Stage B's decoder WHAT stream is binary occupancy, never the intensity image.
    inputs["scene_volume"] = example.arrays.occupancy
    if use_union_mask:
        inputs.pop("anchor_masks")
        inputs["anchor_union_mask"] = example.field("anchor_union_mask")
    leaked = [name for name in STAGE_B_FORBIDDEN_FIELDS if name in inputs]
    if leaked:  # pragma: no cover - guards against future edits
        raise SchemaError(f"forbidden field(s) leaked into Stage B inputs: {leaked}")
    return inputs


# --------------------------------------------------------------------------
# Manifest and scene-array I/O
# --------------------------------------------------------------------------
def write_manifest(path: Path | str, metadatas: Iterable[ExampleMetadata]) -> int:
    """Write a JSONL manifest; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for metadata in metadatas:
            handle.write(json.dumps(metadata.to_json_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_manifest(path: Path | str) -> Iterator[ExampleMetadata]:
    """Read a JSONL manifest, validating every record as it is parsed."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ExampleMetadata.from_json_dict(json.loads(line))
            except (SchemaError, PromptError, KeyError, ValueError) as error:
                raise SchemaError(f"{path}:{line_number}: {error}") from error


def scene_array_dir(root: Path | str, scene_id: str) -> Path:
    """Directory holding the two shared per-scene NIfTIs."""
    return Path(root) / "scenes" / scene_id


def example_array_dir(root: Path | str, example_id: str) -> Path:
    """Directory holding the per-example inspection NIfTIs."""
    return Path(root) / "examples" / example_id


def anchor_mask_filename(slot: int, shape_name: str) -> str:
    return f"anchor_{slot}_{shape_name}.nii.gz"


def save_scene_arrays(
    directory: Path | str,
    scene_volume: np.ndarray,
    instance_labels: np.ndarray,
    spacing: Sequence[float],
) -> Path:
    """Write the two shared per-scene volumes as RAS ``.nii.gz`` files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_nifti(
        directory / SCENE_VOLUME_FILENAME,
        np.asarray(scene_volume, dtype=np.float32),
        spacing,
        dtype=np.float32,
    )
    save_nifti(
        directory / INSTANCE_LABELS_FILENAME,
        np.asarray(instance_labels, dtype=np.uint8),
        spacing,
        dtype=np.uint8,
    )
    return directory


def load_scene_arrays(directory: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read the two shared per-scene volumes from a scene directory."""
    directory = Path(directory)
    if directory.suffix == ".npz":
        raise SchemaError(
            f"scene arrays are NIfTI directories now; refused legacy npz path {directory}"
        )
    scene_volume = load_nifti(directory / SCENE_VOLUME_FILENAME, dtype=np.float32)
    instance_labels = load_nifti(directory / INSTANCE_LABELS_FILENAME, dtype=np.uint8)
    return scene_volume, instance_labels


def save_example_arrays(
    directory: Path | str,
    arrays: ExampleArrays,
    metadata: ExampleMetadata,
) -> Path:
    """Write per-example masks as RAS ``.nii.gz`` files for inspection."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spacing = metadata.spacing
    save_nifti(directory / TARGET_MASK_FILENAME, arrays.target_mask, spacing, dtype=np.uint8)
    for slot, shape_name in enumerate(metadata.anchor_shape_names):
        save_nifti(
            directory / anchor_mask_filename(slot, shape_name),
            arrays.anchor_masks[slot],
            spacing,
            dtype=np.uint8,
        )
    save_nifti(directory / ANCHOR_UNION_FILENAME, arrays.anchor_union_mask, spacing, dtype=np.uint8)
    return directory
