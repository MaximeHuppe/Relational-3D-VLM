"""Synthetic scene generation.

Per scene: rejection-sample the ten primitives (sample parameters, voxelise,
reject empty / out-of-bounds / overlapping candidates) until all ten are placed;
then emit one example per instance via the nearest-feasible anchor selection in
:mod:`src.data.prompt_generator`. Every example is validated by
:mod:`src.data.validation` before it is returned.

Because a scene holds exactly one instance of every class, the instance label of
an object *is* its shape ID: ``instance_labels`` uses 0 for background and
1..10 for the ten shapes.

Escalation: after ``packing.max_scene_attempts`` failed attempts, grow all three
axes together (72^3, then 80^3), rescale the size ranges by the axis ratio and
record the actual ``volume_shape`` on every example. The output contract stays
explicit - the model predicts at the recorded ``volume_shape``.

All randomness derives from the scene seed: attempt ``a`` at escalation stage
``s`` uses ``default_rng([seed, s, a])``, so a scene is bit-for-bit
reproducible from its seed alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from src.config import load_config
from src.data.appearance import AppearanceSettings, render_appearance
from src.data.direction_rules import (
    AmbiguousDirectionError,
    bbox_extent_world,
    centroid_world,
)
from src.data.primitives import SHAPE_VOCABULARY, ShapeSpec
from src.data.prompt_generator import (
    AnchorSelectionError,
    clauses_from_anchors,
    render_prompt,
    select_anchors,
)
from src.data.schema import Example, ExampleArrays, ExampleMetadata, StructuredPrompt
from src.data.validation import (
    RejectionLog,
    RejectionReason,
    ValidationError,
    check_in_bounds,
    check_no_overlap,
    validate_example,
    validate_instance_labels,
)
from src.data.voxelization import (
    VoxelizationError,
    analytic_volume_world,
    half_extent_world,
    voxelize,
)

#: Bounded retry budget when a size draw violates a class-specific constraint
#: (for example the cuboid/ellipsoid anisotropy rule).
MAX_PARAM_DRAWS = 64


class SceneGenerationError(RuntimeError):
    """Raised when a scene cannot be generated within the configured budget."""


@dataclass(frozen=True)
class GeneratorSettings:
    """Everything the generator needs, resolved from ``configs/generator.yaml``."""

    volume_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    margin_voxels: int
    max_object_attempts: int
    max_scene_attempts: int
    escalation_volume_shapes: tuple[tuple[int, int, int], ...]
    rescale_on_escalation: bool
    num_anchors: int
    shapes_per_scene: int
    reference_axis_length: int
    generator_version: str
    appearance: AppearanceSettings

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "GeneratorSettings":
        config = dict(config or load_config("generator"))
        geometry = config["geometry"]
        packing = config["packing"]
        anchors = config["anchors"]
        return cls(
            volume_shape=tuple(int(v) for v in geometry["volume_shape"]),  # type: ignore[arg-type]
            spacing=tuple(float(v) for v in geometry["spacing"]),  # type: ignore[arg-type]
            margin_voxels=int(geometry["in_bounds_margin_voxels"]),
            max_object_attempts=int(packing["max_object_attempts"]),
            max_scene_attempts=int(packing["max_scene_attempts"]),
            escalation_volume_shapes=tuple(
                tuple(int(v) for v in shape) for shape in packing.get("escalation_volume_shapes", ())
            ),  # type: ignore[arg-type]
            rescale_on_escalation=bool(packing.get("rescale_size_ranges_on_escalation", True)),
            num_anchors=int(anchors["num_anchors"]),
            shapes_per_scene=int(packing["shapes_per_scene"]),
            reference_axis_length=int(SHAPE_VOCABULARY.reference_axis_length),
            generator_version=str(config["generator_version"]),
            appearance=AppearanceSettings.from_config(config.get("appearance")),
        )

    def replace_volume_shape(self, volume_shape: Sequence[int]) -> "GeneratorSettings":
        return GeneratorSettings(
            volume_shape=tuple(int(v) for v in volume_shape),  # type: ignore[arg-type]
            spacing=self.spacing,
            margin_voxels=self.margin_voxels,
            max_object_attempts=self.max_object_attempts,
            max_scene_attempts=self.max_scene_attempts,
            escalation_volume_shapes=self.escalation_volume_shapes,
            rescale_on_escalation=self.rescale_on_escalation,
            num_anchors=self.num_anchors,
            shapes_per_scene=self.shapes_per_scene,
            reference_axis_length=self.reference_axis_length,
            generator_version=self.generator_version,
            appearance=self.appearance,
        )

    @property
    def stage_volume_shapes(self) -> tuple[tuple[int, int, int], ...]:
        """The base grid followed by every escalation grid, in order."""
        return (self.volume_shape,) + self.escalation_volume_shapes


@dataclass
class GeneratedScene:
    """One accepted scene and the examples derived from it."""

    scene_id: str
    seed: int
    volume_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    scene_volume: np.ndarray
    instance_labels: np.ndarray
    examples: list[Example]
    attempts: int
    shape_params: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parameter and position sampling
# ---------------------------------------------------------------------------
def placement_order() -> tuple[ShapeSpec, ...]:
    """Largest classes first: placing bulky objects early keeps packing cheap."""
    return tuple(
        sorted(
            SHAPE_VOCABULARY,
            key=lambda spec: (-(spec.mean_volume_voxels or 0.0), spec.id),
        )
    )


def sample_shape_params(
    spec: ShapeSpec, rng: np.random.Generator, scale: float = 1.0
) -> dict[str, float]:
    """Draw one size parameter set, honouring the class-specific constraints."""
    ranges = {name: (low * scale, high * scale) for name, (low, high) in spec.params.items()}
    for _ in range(MAX_PARAM_DRAWS):
        params = {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}
        if spec.min_anisotropy_ratio is not None:
            values = list(params.values())
            if max(values) / min(values) < spec.min_anisotropy_ratio:
                # A cuboid must not be a cube and an ellipsoid must not be a sphere.
                continue
        if spec.name == "torus" and params["minor_radius"] >= params["major_radius"]:
            continue
        return params
    raise SceneGenerationError(
        f"could not draw valid parameters for {spec.name!r} in {MAX_PARAM_DRAWS} attempts"
    )


def sample_center(
    rng: np.random.Generator,
    half_extent: Sequence[float],
    settings: GeneratorSettings,
) -> tuple[float, float, float] | None:
    """Uniformly sample a centre that keeps the object inside the margin.

    Returns ``None`` when the object simply cannot fit, so the caller resamples
    its size instead of looping on an impossible placement.
    """
    depth, height, width = settings.volume_shape
    sizes = (width, height, depth)  # world-axis order (x, y, z)
    center: list[float] = []
    for axis in range(3):
        spacing = settings.spacing[axis]
        low = settings.margin_voxels * spacing + half_extent[axis]
        high = (sizes[axis] - 1 - settings.margin_voxels) * spacing - half_extent[axis]
        if low > high:
            return None
        center.append(float(rng.uniform(low, high)))
    return (center[0], center[1], center[2])


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def pack_scene(
    rng: np.random.Generator,
    settings: GeneratorSettings,
    scale: float,
    log: RejectionLog | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Place all ten objects by rejection sampling.

    Returns ``(instance_labels, shape_params)``.

    Raises:
        ValidationError: packing failed within the per-object attempt budget.
    """
    labels = np.zeros(settings.volume_shape, dtype=np.uint8)
    occupancy = np.zeros(settings.volume_shape, dtype=bool)
    shape_params: dict[str, dict[str, float]] = {}

    for spec in placement_order():
        placed = False
        for _ in range(settings.max_object_attempts):
            params = sample_shape_params(spec, rng, scale=scale)
            half_extent = half_extent_world(spec.name, params)
            center = sample_center(rng, half_extent, settings)
            if center is None:
                continue  # too big for this grid: draw a smaller one
            try:
                candidate = voxelize(spec.name, params, center, settings.volume_shape, settings.spacing)
            except VoxelizationError:  # pragma: no cover - guarded by sample_shape_params
                continue
            if not candidate.any():
                if log is not None:
                    log.reject_object(RejectionReason.EMPTY_OBJECT)
                continue
            try:
                check_in_bounds(candidate, margin_voxels=settings.margin_voxels)
                check_no_overlap(candidate, occupancy)
            except ValidationError as error:
                if log is not None:
                    log.reject_object(error.reason)
                continue
            labels[candidate] = spec.id
            occupancy |= candidate
            shape_params[spec.name] = params
            placed = True
            break
        if not placed:
            raise ValidationError(
                RejectionReason.PACKING_FAILED,
                f"could not place {spec.name!r} in {settings.max_object_attempts} attempts",
            )

    validate_instance_labels(labels, margin_voxels=settings.margin_voxels)
    return labels, shape_params


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def build_examples(
    scene_id: str,
    seed: int,
    instance_labels: np.ndarray,
    settings: GeneratorSettings,
    *,
    scene_volume: np.ndarray | None = None,
    split: str | None = None,
) -> list[Example]:
    """Build the ten candidate examples of an accepted scene.

    Raises:
        AmbiguousDirectionError: some pair has no well-defined direction.
        AnchorSelectionError: some target has no feasible three-direction set.
    """
    if scene_volume is None:
        scene_volume = (instance_labels != 0).astype(np.float32)
    else:
        scene_volume = np.asarray(scene_volume, dtype=np.float32)
    instance_ids = tuple(range(1, settings.shapes_per_scene + 1))
    masks = {instance_id: instance_labels == instance_id for instance_id in instance_ids}
    centroids = {
        instance_id: centroid_world(mask, settings.spacing) for instance_id, mask in masks.items()
    }
    extents = {
        instance_id: bbox_extent_world(mask, settings.spacing) for instance_id, mask in masks.items()
    }
    shape_names = {
        instance_id: SHAPE_VOCABULARY.id_to_name(instance_id) for instance_id in instance_ids
    }

    examples: list[Example] = []
    for target_id in instance_ids:
        anchors = select_anchors(
            target_id,
            centroids,
            shape_names,
            volume_shape=settings.volume_shape,
            spacing=settings.spacing,
            num_anchors=settings.num_anchors,
        )
        clauses = clauses_from_anchors(anchors)
        structured = StructuredPrompt.from_list(clauses)
        metadata = ExampleMetadata(
            scene_id=scene_id,
            example_id=f"{scene_id}_target_{target_id:02d}",
            seed=seed,
            volume_shape=settings.volume_shape,
            spacing=settings.spacing,
            target_instance_id=target_id,
            target_shape_name=shape_names[target_id],
            target_centroid_world=tuple(float(v) for v in centroids[target_id]),
            anchor_instance_ids=tuple(anchor.instance_id for anchor in anchors),
            anchor_shape_names=tuple(anchor.shape_name for anchor in anchors),
            anchor_centroids_world=tuple(
                tuple(float(v) for v in centroids[anchor.instance_id]) for anchor in anchors
            ),
            anchor_extents_world=tuple(
                tuple(float(v) for v in extents[anchor.instance_id]) for anchor in anchors
            ),
            structured_prompt=structured,
            prompt=render_prompt(clauses),
            generator_version=settings.generator_version,
            split=split,
            extra={"anchor_distances": [round(anchor.distance, 6) for anchor in anchors]},
        )
        example = Example(
            metadata=metadata,
            arrays=ExampleArrays.from_scene(metadata, scene_volume, instance_labels),
        )
        validate_example(example, margin_voxels=settings.margin_voxels)
        examples.append(example)
    return examples


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------
def generate_scene(
    seed: int,
    *,
    scene_id: str | None = None,
    settings: GeneratorSettings | None = None,
    split: str | None = None,
    log: RejectionLog | None = None,
) -> GeneratedScene:
    """Generate one validated scene together with its ten candidate examples.

    Retries the whole scene on any rejection, then escalates the grid once the
    per-grid attempt budget is exhausted.

    Raises:
        SceneGenerationError: every grid and attempt was exhausted.
    """
    base = settings or GeneratorSettings.from_config()
    scene_id = scene_id or f"scene_{seed:09d}"
    attempts = 0

    for stage, volume_shape in enumerate(base.stage_volume_shapes):
        stage_settings = base.replace_volume_shape(volume_shape)
        scale = (
            max(volume_shape) / base.reference_axis_length
            if base.rescale_on_escalation
            else 1.0
        )
        for attempt in range(base.max_scene_attempts):
            attempts += 1
            rng = np.random.default_rng([seed, stage, attempt])
            try:
                labels, shape_params = pack_scene(rng, stage_settings, scale, log=log)
                appearance_rng = np.random.default_rng([seed, stage, attempt, 17])
                scene_volume = render_appearance(labels, appearance_rng, stage_settings.appearance)
                examples = build_examples(
                    scene_id,
                    seed,
                    labels,
                    stage_settings,
                    scene_volume=scene_volume,
                    split=split,
                )
            except ValidationError as error:
                if log is not None:
                    log.reject(error.reason)
                continue
            except AmbiguousDirectionError:
                if log is not None:
                    log.reject(RejectionReason.AMBIGUOUS_DIRECTION)
                continue
            except AnchorSelectionError:
                if log is not None:
                    log.reject(RejectionReason.NO_FEASIBLE_ANCHOR_TRIPLE)
                continue

            if log is not None:
                log.accept()
            return GeneratedScene(
                scene_id=scene_id,
                seed=seed,
                volume_shape=stage_settings.volume_shape,
                spacing=stage_settings.spacing,
                scene_volume=scene_volume,
                instance_labels=labels,
                examples=examples,
                attempts=attempts,
                shape_params=shape_params,
            )

    raise SceneGenerationError(
        f"scene {scene_id} (seed {seed}) failed after {attempts} attempts across grids "
        f"{base.stage_volume_shapes}"
    )


def generate_examples(
    seed: int,
    *,
    scene_id: str | None = None,
    settings: GeneratorSettings | None = None,
    split: str | None = None,
    target_classes: Sequence[str] | None = None,
    log: RejectionLog | None = None,
) -> Iterator[Example]:
    """Yield the candidate examples of one scene, optionally target-filtered."""
    scene = generate_scene(
        seed, scene_id=scene_id, settings=settings, split=split, log=log
    )
    allowed = None if target_classes is None else set(SHAPE_VOCABULARY.require_names(target_classes))
    for example in scene.examples:
        if allowed is None or example.metadata.target_shape_name in allowed:
            yield example


def scene_occupancy_report(scene: GeneratedScene) -> dict[str, Any]:
    """Per-class voxel counts, extents and occupancy - used by the smoke report."""
    report: dict[str, Any] = {"scene_id": scene.scene_id, "shapes": {}}
    axis_length = max(scene.volume_shape)
    for spec in SHAPE_VOCABULARY:
        mask = scene.instance_labels == spec.id
        extent = bbox_extent_world(mask, scene.spacing)
        params = scene.shape_params.get(spec.name, {})
        report["shapes"][spec.name] = {
            "voxels": int(mask.sum()),
            "extent_world": [float(v) for v in extent],
            "max_extent_fraction": float(max(extent) / axis_length),
            "analytic_volume": (
                float(analytic_volume_world(spec.name, params)) if params else None
            ),
        }
    report["foreground_fraction"] = float(
        (scene.instance_labels != 0).sum() / scene.instance_labels.size
    )
    return report
