"""PyTorch datasets over a generated corpus.

Stage A consumes *scenes*: one sample is a whole ``scene_volume`` plus the ten
per-shape masks derived from ``instance_labels``. The target-class split filter
does not apply to Stage A - it constrains which target classes Stage B may be
supervised on, while Stage A must learn all ten shapes in every split - so
:class:`SceneDataset` takes the set of scenes belonging to a split, read from
that split's manifest.

Stage B consumes *examples*: one sample is one ``(scene, target)`` pair - three
ordered anchor channels, a three-clause prompt and the target mask that
supervises it - read from the same manifests. :class:`ExampleDataset` is filtered
by ``configs/split.yaml`` at generation time, so it yields only the target
classes a split is allowed to supervise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY
from src.data.prompt_generator import clause_indices
from src.data.schema import ExampleArrays, ExampleMetadata, load_scene_arrays, read_manifest


class DatasetError(RuntimeError):
    """Raised when a corpus on disk cannot back the requested dataset."""


@dataclass(frozen=True)
class SceneRecord:
    """One scene available to Stage A."""

    scene_id: str
    path: Path
    seed: int
    volume_shape: tuple[int, int, int]


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
        path = root / "scenes" / f"{metadata.scene_id}.npz"
        if not path.is_file():
            raise DatasetError(f"missing scene arrays {path}")
        records[metadata.scene_id] = SceneRecord(
            scene_id=metadata.scene_id,
            path=path,
            seed=metadata.seed,
            volume_shape=metadata.volume_shape,
        )
    if not records:
        raise DatasetError(f"no scenes listed in {manifest}")
    return [records[scene_id] for scene_id in sorted(records)]


class SceneDataset(Dataset):
    """Stage A samples: ``scene_volume`` in, one binary mask per shape name out.

    Each item is a dict with:

    ``scene_volume``  ``[1, D, H, W]`` float32, the binary scene;
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
        cache: hold the scenes in memory to keep I/O out of the training loop.
            Only the two compact uint8 volumes are cached (~0.52 MB per scene);
            the float tensors and the per-shape masks are derived per item.
            Caching the derived tensors instead would cost ~13.6 MB per scene -
            5.4 GB for a 400-scene split, multiplied again by every persistent
            dataloader worker.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        prompt_names: Sequence[str] | None = None,
        limit: int | None = None,
        cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
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

        names = tuple(prompt_names) if prompt_names is not None else SHAPE_NAMES
        self.prompt_names = SHAPE_VOCABULARY.require_names(names)
        self.prompt_ids = torch.tensor(
            [SHAPE_VOCABULARY.index_of(name) for name in self.prompt_names], dtype=torch.long
        )
        self.instance_ids = [SHAPE_VOCABULARY.name_to_id(name) for name in self.prompt_names]
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _volumes(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The two compact uint8 volumes of a scene, cached as stored on disk."""
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        scene_volume, instance_labels = load_scene_arrays(self.records[index].path)
        volumes = (
            torch.from_numpy(np.ascontiguousarray(scene_volume, dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(instance_labels, dtype=np.uint8)),
        )
        if self._cache is not None:
            self._cache[index] = volumes
        return volumes

    def __getitem__(self, index: int) -> dict:
        scene_volume, labels = self._volumes(index)
        masks = torch.stack(
            [(labels == instance_id) for instance_id in self.instance_ids]
        ).to(torch.float32)
        return {
            "scene_volume": scene_volume.to(torch.float32).unsqueeze(0),
            "instance_labels": labels,
            "target_masks": masks,
            "prompt_ids": self.prompt_ids.clone(),
            "scene_id": self.records[index].scene_id,
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
        return (
            f"{self.split}: {len(self)} scenes at {self.volume_shape}, "
            f"{len(self.prompt_names)} prompts"
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
        path = root / "scenes" / f"{metadata.scene_id}.npz"
        if not path.is_file():
            raise DatasetError(f"missing scene arrays {path}")
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
    ``scene_volume``       ``[1, D, H, W]`` float32, **only** when
                           ``include_scene_volume=True``;
    plus ``example_id``, ``scene_id``, ``prompt``, ``target_shape_name``,
    ``anchor_shape_names`` and ``directions`` for reporting and stratification.

    The target mask is the label, never an input: :func:`stage_b_model_inputs`
    is the only sanctioned way to build the model's arguments from an item, and
    it passes exactly the anchor channels and the two index tensors.

    ``scene_volume`` is off by default and exists for one purpose: feeding
    Stage A when anchors are *predicted* (Phase 4). It goes to the anchor
    provider, never to Stage B - see :mod:`src.models.anchor_provider`.

    Args:
        root: dataset directory written by ``scripts/generate_dataset.py``.
        split: ``train``, ``val`` or ``test``.
        limit: keep only the first N examples, after filtering.
        scene_ids: restrict to these scenes (Phase 2 overfits on one).
        target_shapes: restrict to these target classes; the manifest is already
            filtered by ``configs/split.yaml``, so this is a further narrowing.
        include_scene_volume: also return the scene volume, for predicted anchors.
        validate: re-run the full schema validation on the first visit to each
            scene. Cheap (once per scene) and fails fast on a corrupt corpus.
        cache: hold decoded ``instance_labels`` in memory, one copy per scene
            shared by that scene's examples.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        limit: int | None = None,
        scene_ids: Sequence[str] | None = None,
        target_shapes: Sequence[str] | None = None,
        include_scene_volume: bool = False,
        validate: bool = True,
        cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
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

    def __len__(self) -> int:
        return len(self.records)

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
        if self._scene_cache is not None and record.scene_id in self._scene_cache:
            return self._scene_cache[record.scene_id]
        arrays = load_scene_arrays(record.path)
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
            # all three. Later epochs take the fast path below.
            ExampleArrays.from_scene(metadata, scene_volume, instance_labels)
            self._validated.add(metadata.example_id)

        labels = np.asarray(instance_labels)
        anchor_masks = np.stack(
            [(labels == instance_id) for instance_id in metadata.anchor_instance_ids]
        ).astype(np.float32)
        target_mask = (labels == metadata.target_instance_id).astype(np.float32)

        directions, shape_ids = clause_indices(metadata.relations)
        item = {
            "anchor_masks": torch.from_numpy(anchor_masks),
            "target_mask": torch.from_numpy(target_mask).unsqueeze(0),
            "anchor_union_mask": torch.from_numpy(
                anchor_masks.max(axis=0)
            ).unsqueeze(0),
            "direction_ids": torch.tensor(directions, dtype=torch.long),
            "anchor_shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "example_id": metadata.example_id,
            "scene_id": metadata.scene_id,
            "prompt": metadata.prompt,
            "target_shape_name": metadata.target_shape_name,
            "anchor_shape_names": list(metadata.anchor_shape_names),
            "directions": list(metadata.structured_prompt.directions),
        }
        if self.include_scene_volume:
            item["scene_volume"] = torch.from_numpy(
                np.asarray(scene_volume, dtype=np.float32)
            ).unsqueeze(0)
        return item

    def describe(self) -> str:
        counts = ", ".join(f"{name} {count}" for name, count in self.target_shape_counts().items())
        return (
            f"{self.split}: {len(self)} examples from {len(self.scene_ids)} scenes "
            f"at {self.volume_shape} [{counts}]"
        )


#: Keys of a Stage B item that are labels or provenance, never model inputs.
STAGE_B_NON_INPUT_KEYS: tuple[str, ...] = (
    "target_mask",
    "scene_volume",
    "anchor_union_mask",
    "example_id",
    "scene_id",
    "prompt",
    "target_shape_name",
    "anchor_shape_names",
    "directions",
)


def stage_b_model_inputs(
    batch: Mapping[str, Any], anchor_masks: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """The only sanctioned way to call Stage B from a batch.

    Returns exactly ``{anchor_masks, direction_ids, anchor_shape_ids}``.
    ``anchor_masks`` may be overridden with predicted channels; everything else
    in the batch - the target mask above all - stays behind.
    """
    masks = batch["anchor_masks"] if anchor_masks is None else anchor_masks
    return {
        "anchor_masks": masks,
        "direction_ids": batch["direction_ids"],
        "anchor_shape_ids": batch["anchor_shape_ids"],
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
