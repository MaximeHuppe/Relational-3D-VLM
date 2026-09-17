"""PyTorch datasets over a generated corpus.

Stage A consumes *scenes*: one sample is a whole scene image plus the ten
per-shape masks derived from ``instance_labels``. The target-class split filter
does not apply to Stage A - it constrains which target classes Stage B may be
supervised on, while Stage A must learn all ten shapes in every split - so
:class:`SceneDataset` takes the set of scenes belonging to a split, read from
that split's manifest.

Stage B consumes *examples*: one sample is one ``(scene, target)`` pair - three
ordered anchor channels, a three-clause prompt, the scene image, and the target
mask that supervises it - read from the same manifests.
:class:`ExampleDataset` is filtered by ``configs/split.yaml`` at generation time,
so it yields only the target classes a split is allowed to supervise.

Augmentation
------------
Both datasets accept an optional ``augment``
(:class:`src.training.augmentations.RotationAugmentation`). When one is given,
every item is drawn in a rotated pose that changes per epoch, and Stage B's
clause triple is re-derived for that pose - see that module for why the
directions cannot be relabelled token-by-token. Augmentation is opt-in per
dataset, so it is attached to the training split and never to validation or
test, and the corpus on disk is never modified.

The scene image
---------------
Both datasets hand the model a ``scene_volume`` tensor. On a corpus generated
with ``configs/appearance.yaml`` enabled that is the simulated MRI-like volume,
normalised per volume by :class:`ImagePolicy`; on an older corpus it is the
binary occupancy, and ``image_source="occupancy"`` forces the binary form on
either. The key stays ``scene_volume`` because that is what the models call
their image argument; nothing about the label contract changes with it, and the
schema's own ``scene_volume`` - the thing validation compares against
``instance_labels`` - remains the binary occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.appearance import AppearanceSettings, normalize_image
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY
from src.data.prompt_generator import clause_indices
from src.data.scene_io import SceneIOError, SceneVolumes, load_scene, scene_path
from src.data.schema import ExampleArrays, ExampleMetadata, read_manifest

if TYPE_CHECKING:  # pragma: no cover - `src.data` must not import `src.training`
    from src.training.augmentations import RotationAugmentation


class DatasetError(RuntimeError):
    """Raised when a corpus on disk cannot back the requested dataset."""


#: How a dataset picks the volume it hands the model as ``scene_volume``.
#: ``auto`` is the simulated MRI-like image when the corpus has one and the
#: binary occupancy when it does not, so a corpus from either milestone loads
#: without a flag.
IMAGE_SOURCES: tuple[str, ...] = ("auto", "intensity", "occupancy")

#: In-memory type of a cached scene image. float16 halves the cache against
#: float32, and its ~5e-4 quantisation over the normalised range is well under
#: the image's own noise, so nothing a model sees changes. Items are cast back
#: to float32 on the way out.
CACHE_DTYPE = np.float16


@dataclass(frozen=True)
class ImagePolicy:
    """Which volume a dataset feeds the model, and how it is normalised.

    The binary volumes of the previous milestone needed no normalisation - they
    were already 0/1. A simulated acquisition does: its units are arbitrary and
    its bias field shifts the level across the volume, exactly as a real one's
    does, so every volume is normalised on its own at load time rather than
    baked into the corpus. The transform depends only on the volume's own
    intensities, so it commutes with the rotation augmentation.
    """

    source: str = "auto"
    mode: str = "percentile"
    percentiles: tuple[float, float] = (0.5, 99.5)
    clip: tuple[float, float] | None = (-0.5, 1.5)

    def __post_init__(self) -> None:
        if self.source not in IMAGE_SOURCES:
            raise DatasetError(
                f"image_source must be one of {IMAGE_SOURCES}, got {self.source!r}"
            )

    @classmethod
    def from_config(cls, source: str = "auto") -> "ImagePolicy":
        settings = AppearanceSettings.from_config()
        return cls(
            source=source,
            mode=settings.normalization_mode,
            percentiles=settings.normalization_percentiles,
            clip=settings.normalization_clip,
        )

    def apply(self, volumes: SceneVolumes) -> np.ndarray:
        """The model-facing volume of one scene, normalised when it is an image."""
        image = volumes.model_image(self.source)
        if self.source == "occupancy" or (self.source == "auto" and not volumes.has_image):
            return np.asarray(image, dtype=np.float32)
        return normalize_image(
            image, mode=self.mode, percentiles=self.percentiles, clip=self.clip
        )

    def describe(self) -> str:
        return f"image={self.source}/{self.mode}"


@dataclass(frozen=True)
class SceneRecord:
    """One scene available to Stage A."""

    scene_id: str
    path: Path
    seed: int
    volume_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]


def scene_records(root: Path | str, split: str) -> list[SceneRecord]:
    """List the scenes of a split, in deterministic scene-ID order.

    Scene identities come from the split manifest, so a scene can never be read
    into the wrong split even though all scenes share one directory.
    """
    root = Path(root)
    manifest = root / "manifests" / f"{split}.jsonl"
    if not manifest.is_file():
        raise DatasetError(f"missing manifest {manifest}")
    records: dict[str, SceneRecord] = {}
    for metadata in read_manifest(manifest):
        if metadata.scene_id in records:
            continue
        try:
            path = scene_path(root, metadata.scene_id)
        except SceneIOError as error:
            raise DatasetError(str(error)) from error
        records[metadata.scene_id] = SceneRecord(
            scene_id=metadata.scene_id,
            path=path,
            seed=metadata.seed,
            volume_shape=metadata.volume_shape,
            spacing=metadata.spacing,
        )
    if not records:
        raise DatasetError(f"no scenes listed in {manifest}")
    return [records[scene_id] for scene_id in sorted(records)]


class SceneDataset(Dataset):
    """Stage A samples: the scene image in, one binary mask per shape name out.

    Each item is a dict with:

    ``scene_volume``  ``[1, D, H, W]`` float32, the scene image - the simulated
                      MRI-like volume when the corpus has one, normalised per
                      volume, otherwise the binary occupancy;
    ``prompt_ids``    ``[N_T]`` int64, zero-based vocabulary indices;
    ``target_masks``  ``[N_T, D, H, W]`` float32, aligned with ``prompt_ids``;
    ``instance_labels`` ``[D, H, W]`` int64, kept for qualitative output;
    ``scene_id``      str.

    Args:
        root: dataset directory written by ``scripts/generate_dataset.py``.
        split: ``train``, ``val`` or ``test``.
        prompt_names: which shape names to request, in order. Defaults to the
            full vocabulary in canonical order.
        limit: keep only the first N scenes (smoke runs).
        image_source: which stored volume backs ``scene_volume`` - ``auto``
            (the simulated image when the corpus has one), ``intensity`` or
            ``occupancy``. ``occupancy`` reproduces the binary input of the
            previous milestone and is the baseline to compare against.
        cache: hold the scenes in memory to keep I/O out of the training loop.
            Only the normalised image (float16) and the uint8 label volume are
            cached, ~0.79 MB per scene; the per-shape masks are derived per
            item. Caching the derived masks instead would cost ~13.6 MB per
            scene - 5.4 GB for a 400-scene split, multiplied again by every
            persistent dataloader worker. float16 halves the image against
            float32 and its ~5e-4 quantisation is under 2% of the image noise,
            so it changes nothing a model can see.
        augment: rotate each scene into a per-epoch pose. Stage A has no
            relations to rewrite - the scene and all ten masks rotate together,
            and a rotated cube is still a cube - so the labels stay correct by
            construction. It is still off by default: the generator only ever
            emits shapes in their canonical orientation (cones point ``+z``, and
            so on), so rotating them is a real change to the training
            distribution that evaluation does not share. The cache is filled
            with the stored pose and rotated afterwards, so one cached scene
            still backs every epoch.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        prompt_names: Sequence[str] | None = None,
        limit: int | None = None,
        cache: bool = True,
        image_source: str = "auto",
        augment: "RotationAugmentation | None" = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_policy = ImagePolicy.from_config(image_source)
        self.records = scene_records(self.root, split)
        if limit is not None:
            self.records = self.records[: max(int(limit), 0)]
        if not self.records:
            raise DatasetError(f"no scenes left for split {split!r} after applying limit={limit}")

        shapes = {record.volume_shape for record in self.records}
        if len(shapes) != 1:
            raise DatasetError(
                f"split {split!r} mixes volume shapes {sorted(shapes)}; batching requires one "
                "shape. Regenerate without grid escalation or train per shape."
            )
        self.volume_shape = next(iter(shapes))
        spacings = {record.spacing for record in self.records}
        if len(spacings) != 1:
            raise DatasetError(f"split {split!r} mixes spacings {sorted(spacings)}")
        self.spacing = next(iter(spacings))

        names = tuple(prompt_names) if prompt_names is not None else SHAPE_NAMES
        self.prompt_names = SHAPE_VOCABULARY.require_names(names)
        self.prompt_ids = torch.tensor(
            [SHAPE_VOCABULARY.index_of(name) for name in self.prompt_names], dtype=torch.long
        )
        self.instance_ids = [SHAPE_VOCABULARY.name_to_id(name) for name in self.prompt_names]
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = {} if cache else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation pose. A no-op when not augmenting."""
        if self.augment is not None:
            self.augment.set_epoch(epoch)

    def _volumes(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A scene's model-facing image and its label volume, cached as loaded.

        Normalisation is done here, once per scene, rather than per item: it
        depends only on the stored volume, so caching the normalised result is
        equivalent to caching the raw one and cheaper per epoch.
        """
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        scene = load_scene(self.records[index].path)
        volumes = (
            torch.from_numpy(
                np.ascontiguousarray(self.image_policy.apply(scene), dtype=CACHE_DTYPE)
            ),
            torch.from_numpy(np.ascontiguousarray(scene.instance_labels, dtype=np.uint8)),
        )
        if self._cache is not None:
            self._cache[index] = volumes
        return volumes

    def __getitem__(self, index: int) -> dict:
        scene_volume, labels = self._volumes(index)
        rotation = None
        if self.augment is not None:
            rotation = self.augment.sample(index)
            # One rotation for both volumes, so every mask stays the mask of the
            # instance it labels.
            scene_volume = rotation.apply_tensor(scene_volume)
            labels = rotation.apply_tensor(labels)
        masks = torch.stack(
            [(labels == instance_id) for instance_id in self.instance_ids]
        ).to(torch.float32)
        return {
            "scene_volume": scene_volume.to(torch.float32).unsqueeze(0),
            "instance_labels": labels,
            "target_masks": masks,
            "prompt_ids": self.prompt_ids.clone(),
            "scene_id": self.records[index].scene_id,
            "rotation": rotation.code if rotation is not None else "",
        }

    def cache_bytes(self) -> int:
        """Resident size of the scene cache, for sizing a run against the machine."""
        if not self._cache:
            return 0
        per_scene = sum(
            tensor.element_size() * tensor.nelement() for tensor in next(iter(self._cache.values()))
        )
        return per_scene * len(self._cache)

    def describe(self) -> str:
        augment = "" if self.augment is None else f", {self.augment.describe()}"
        return (
            f"{self.split}: {len(self)} scenes at {self.volume_shape}, "
            f"{len(self.prompt_names)} prompts, {self.image_policy.describe()}{augment}"
        )


def build_dataloader(
    dataset: SceneDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """A deterministic dataloader: fixed generator, no non-reproducible ordering."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=drop_last,
        pin_memory=False,
        # Workers are respawned every epoch by default, which would throw away
        # the dataset's decoded-scene cache each time and re-read the corpus.
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Stage B: one sample is one (scene, target) example
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExampleRecord:
    """One Stage B example: its metadata plus the scene arrays it derives from."""

    metadata: ExampleMetadata
    path: Path

    @property
    def example_id(self) -> str:
        return self.metadata.example_id

    @property
    def scene_id(self) -> str:
        return self.metadata.scene_id


def example_records(
    root: Path | str,
    split: str,
    *,
    scene_ids: Sequence[str] | None = None,
    target_shapes: Sequence[str] | None = None,
) -> list[ExampleRecord]:
    """Read a split manifest in deterministic example-ID order.

    Every record is validated by :func:`src.data.schema.read_manifest` as it is
    parsed, so a malformed example fails here rather than silently training on a
    broken prompt.
    """
    root = Path(root)
    manifest = root / "manifests" / f"{split}.jsonl"
    if not manifest.is_file():
        raise DatasetError(f"missing manifest {manifest}")
    wanted_scenes = set(scene_ids) if scene_ids is not None else None
    wanted_targets = set(target_shapes) if target_shapes is not None else None

    records: list[ExampleRecord] = []
    for metadata in read_manifest(manifest):
        if wanted_scenes is not None and metadata.scene_id not in wanted_scenes:
            continue
        if wanted_targets is not None and metadata.target_shape_name not in wanted_targets:
            continue
        try:
            path = scene_path(root, metadata.scene_id)
        except SceneIOError as error:
            raise DatasetError(str(error)) from error
        records.append(ExampleRecord(metadata=metadata, path=path))
    if not records:
        raise DatasetError(
            f"no examples in {manifest} for scenes={scene_ids} targets={target_shapes}"
        )
    return sorted(records, key=lambda record: record.example_id)


class ExampleDataset(Dataset):
    """Stage B samples: three ordered anchor channels and a prompt in, one target out.

    Each item is a dict with

    ``anchor_masks``       ``[3, D, H, W]`` float32, ordered by clause slot;
    ``target_mask``        ``[1, D, H, W]`` float32, the supervision signal;
    ``direction_ids``      ``[3]`` int64, closed-vocabulary direction indices;
    ``anchor_shape_ids``   ``[3]`` int64, zero-based shape indices - these double
                           as Stage A's ``prompt_ids`` when anchors are predicted;
    ``anchor_union_mask``  ``[1, D, H, W]`` float32, ablation baseline only;
    ``scene_occupancy``    ``[1, D, H, W]`` float32, the binary foreground of the
                           scene. Analysis only - it is what the occupancy
                           sanity checks count connected components on, which
                           cannot be done on the intensity image. Never an input;
    ``scene_volume``       ``[1, D, H, W]`` float32, the scene image - the
                           simulated MRI-like volume when the corpus has one,
                           normalised per volume, otherwise the binary
                           occupancy. Always present by default. Stage A
                           consumes it to predict anchors; Stage B's decoder
                           consumes the same tensor as the WHAT stream. It is
                           the only appearance the model sees, and it still
                           says nothing about which structure is the target.
    plus ``example_id``, ``scene_id``, ``prompt``, ``target_shape_name``,
    ``anchor_shape_names``, ``directions`` and ``rotation`` for reporting and
    stratification.

    The target mask is the label, never an input: :func:`stage_b_model_inputs`
    is the only sanctioned way to build the model's arguments from an item, and
    it passes the anchor channels, the two index tensors and ``scene_volume``.
    Instance labels and every ``target_*`` field stay behind.

    Args:
        root: dataset directory written by ``scripts/generate_dataset.py``.
        split: ``train``, ``val`` or ``test``.
        limit: keep only the first N examples, after filtering.
        scene_ids: restrict to these scenes (Phase 2 overfits on one).
        target_shapes: restrict to these target classes; the manifest is already
            filtered by ``configs/split.yaml``, so this is a further narrowing.
        include_scene_volume: return the scene image. On by default because
            Stage B's decoder needs it; predicted-anchor runs also feed the
            same tensor to Stage A.
        image_source: which stored volume backs ``scene_volume`` - ``auto``
            (the simulated image when the corpus has one), ``intensity`` or
            ``occupancy``. ``occupancy`` reproduces the binary WHAT stream of
            the previous milestone.
        validate: re-run the full schema validation on the first visit to each
            scene. Cheap (once per scene) and fails fast on a corrupt corpus.
        cache: hold the normalised image (float16) and ``instance_labels`` in
            memory, one copy per scene shared by that scene's ten examples.
        augment: rotate each example into a per-epoch pose and re-derive its
            three directions for that pose. The anchors, their order and the
            target are untouched; ``direction_ids``, ``prompt`` and
            ``directions`` all come from the rewrite, so the tensors and the
            text can never disagree. Attach it to the training split only -
            validation and test must stay in the stored pose, or the metric
            stops being comparable. Validation still runs against the stored
            arrays, before any rotation.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        limit: int | None = None,
        scene_ids: Sequence[str] | None = None,
        target_shapes: Sequence[str] | None = None,
        include_scene_volume: bool = True,
        validate: bool = True,
        cache: bool = True,
        image_source: str = "auto",
        augment: "RotationAugmentation | None" = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_policy = ImagePolicy.from_config(image_source)
        self.records = example_records(
            self.root, split, scene_ids=scene_ids, target_shapes=target_shapes
        )
        if limit is not None:
            self.records = self.records[: max(int(limit), 0)]
        if not self.records:
            raise DatasetError(f"no examples left for split {split!r} after applying limit={limit}")

        shapes = {record.metadata.volume_shape for record in self.records}
        if len(shapes) != 1:
            raise DatasetError(
                f"split {split!r} mixes volume shapes {sorted(shapes)}; batching requires one shape"
            )
        self.volume_shape = next(iter(shapes))
        spacings = {record.metadata.spacing for record in self.records}
        if len(spacings) != 1:
            raise DatasetError(f"split {split!r} mixes spacings {sorted(spacings)}")
        self.spacing = next(iter(spacings))

        self.include_scene_volume = bool(include_scene_volume)
        self.validate = bool(validate)
        self._scene_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = {} if cache else None
        self._validated: set[str] = set()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation pose. A no-op when not augmenting."""
        if self.augment is not None:
            self.augment.set_epoch(epoch)

    def target_shape_counts(self) -> dict[str, int]:
        """How many examples each target class contributes."""
        counts: dict[str, int] = {}
        for record in self.records:
            name = record.metadata.target_shape_name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def scene_ids(self) -> list[str]:
        """The distinct scenes these examples come from, in order."""
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.scene_id, None)
        return list(seen)

    def _scene_arrays(self, record: ExampleRecord) -> tuple[np.ndarray, np.ndarray]:
        """``(model image, instance labels)`` of a scene, cached per scene.

        One copy is shared by that scene's ten examples; the schema
        re-validation below reads the labels, which is where the binary
        occupancy is re-derived from.
        """
        if self._scene_cache is not None and record.scene_id in self._scene_cache:
            return self._scene_cache[record.scene_id]
        scene = load_scene(record.path)
        arrays = (
            np.ascontiguousarray(self.image_policy.apply(scene), dtype=CACHE_DTYPE),
            np.asarray(scene.instance_labels),
        )
        if self._scene_cache is not None:
            self._scene_cache[record.scene_id] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        metadata = record.metadata
        scene_volume, instance_labels = self._scene_arrays(record)

        if self.validate and metadata.example_id not in self._validated:
            # Full schema re-validation on first visit: exactly ten instances,
            # every channel equal to its declared anchor, the target absent from
            # all three. The schema's `scene_volume` is the binary occupancy, so
            # it is re-derived from the labels here - the simulated image is the
            # model's input, never the thing the label contract is checked
            # against. Later epochs take the fast path below.
            ExampleArrays.from_scene(
                metadata, (np.asarray(instance_labels) != 0).astype(np.uint8), instance_labels
            )
            self._validated.add(metadata.example_id)

        labels = np.asarray(instance_labels)
        relations = metadata.relations
        prompt = metadata.prompt
        rotation_code = ""
        if self.augment is not None:
            # The pose and the clause triple are chosen together: the rewrite is
            # what decides whether a drawn rotation is usable at all, so the
            # arrays are rotated only once a legal triple exists for them.
            plan = self.augment.plan(metadata, index)
            labels = plan.rotation.apply(labels)
            if self.include_scene_volume:
                # Same rotation, so predicted anchors line up with the channels
                # they replace.
                scene_volume = plan.rotation.apply(np.asarray(scene_volume))
            relations = list(plan.relations)
            prompt = plan.prompt
            rotation_code = plan.rotation.code

        anchor_masks = np.stack(
            [(labels == instance_id) for instance_id in metadata.anchor_instance_ids]
        ).astype(np.float32)
        target_mask = (labels == metadata.target_instance_id).astype(np.float32)
        # The binary foreground, for analysis only. The occupancy sanity checks
        # count how many scene objects a prediction covers, and they cannot do
        # that on the intensity image: thresholding an acquisition lights up the
        # whole head as one component. It is never a model input - see
        # STAGE_B_NON_INPUT_KEYS.
        scene_occupancy = (labels != 0).astype(np.float32)

        directions, shape_ids = clause_indices(relations)
        item = {
            "anchor_masks": torch.from_numpy(anchor_masks),
            "target_mask": torch.from_numpy(target_mask).unsqueeze(0),
            "anchor_union_mask": torch.from_numpy(
                anchor_masks.max(axis=0)
            ).unsqueeze(0),
            "scene_occupancy": torch.from_numpy(scene_occupancy).unsqueeze(0),
            "direction_ids": torch.tensor(directions, dtype=torch.long),
            "anchor_shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "example_id": metadata.example_id,
            "scene_id": metadata.scene_id,
            "prompt": prompt,
            "target_shape_name": metadata.target_shape_name,
            "anchor_shape_names": [clause["anchor"] for clause in relations],
            "directions": [clause["direction"] for clause in relations],
            # Empty when unaugmented; otherwise the pose this item was drawn in,
            # so a stratified report can separate them.
            "rotation": rotation_code,
        }
        if self.include_scene_volume:
            item["scene_volume"] = torch.from_numpy(
                np.asarray(scene_volume, dtype=np.float32)
            ).unsqueeze(0)
        return item

    def describe(self) -> str:
        counts = ", ".join(f"{name} {count}" for name, count in self.target_shape_counts().items())
        augment = "" if self.augment is None else f", {self.augment.describe()}"
        return (
            f"{self.split}: {len(self)} examples from {len(self.scene_ids)} scenes "
            f"at {self.volume_shape} [{counts}], {self.image_policy.describe()}{augment}"
        )


#: Keys of a Stage B item that are labels or provenance, never model inputs.
STAGE_B_NON_INPUT_KEYS: tuple[str, ...] = (
    "target_mask",
    "anchor_union_mask",
    "scene_occupancy",
    "example_id",
    "scene_id",
    "prompt",
    "target_shape_name",
    "anchor_shape_names",
    "directions",
    "rotation",
)


def stage_b_model_inputs(
    batch: Mapping[str, Any], anchor_masks: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """The only sanctioned way to call Stage B from a batch.

    Returns exactly ``{anchor_masks, direction_ids, anchor_shape_ids, scene_volume}``.
    ``anchor_masks`` may be overridden with predicted channels; everything else
    in the batch - the target mask above all - stays behind.
    """
    masks = batch["anchor_masks"] if anchor_masks is None else anchor_masks
    return {
        "anchor_masks": masks,
        "direction_ids": batch["direction_ids"],
        "anchor_shape_ids": batch["anchor_shape_ids"],
        "scene_volume": batch["scene_volume"],
    }


def collate_examples(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep metadata as plain lists.

    The default collate would transpose ``anchor_shape_names`` into three lists
    of batch entries, which silently breaks per-sample stratification; here one
    list entry stays one sample.
    """
    batch: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return batch


def build_example_dataloader(
    dataset: ExampleDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """A deterministic Stage B dataloader."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=drop_last,
        pin_memory=False,
        collate_fn=collate_examples,
        persistent_workers=num_workers > 0,
    )
