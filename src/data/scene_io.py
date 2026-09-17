"""On-disk layout of a generated corpus.

The tree matches ``exp/realistic-appearance`` so ``data/processed`` can be
copied into that worktree as-is::

    scenes/<scene_id>/scene_volume.nii.gz
    scenes/<scene_id>/instance_labels.nii.gz
    examples/<example_id>/target_mask.nii.gz
    examples/<example_id>/anchor_{slot}_{shape}.nii.gz
    examples/<example_id>/anchor_union.nii.gz
    manifests/<split>.jsonl
    manifests/<split>_candidates.jsonl
    run_metadata.json

``scene_volume.nii.gz`` is the simulated MRI-like image (uint16 with a header
scale factor by default). ``instance_labels.nii.gz`` is uint8 labels 0 and
1..10. Occupancy, per-structure masks and every example channel are derivable
from the labels; they are written under ``examples/`` for inspection, and the
loader rematerialises them either way.

An older nested layout (``image.nii.gz`` / ``labels.nii.gz``, ``occupancy``,
``masks/``, ``scenes/<id>/examples/``) is still readable. The previous
single-file ``scenes/<scene_id>.npz`` form is too.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.nifti_io import NIFTI_SUFFIX, load_nifti, save_nifti, scene_affine
from src.data.primitives import SHAPE_VOCABULARY

#: Format tokens accepted by ``storage.scene_array_format``.
NIFTI_FORMAT = "nifti"
NPZ_FORMAT = "npz_compressed"

#: Canonical names, matching ``exp/realistic-appearance``.
SCENE_VOLUME_FILE = "scene_volume" + NIFTI_SUFFIX
INSTANCE_LABELS_FILE = "instance_labels" + NIFTI_SUFFIX
TARGET_MASK_FILE = "target_mask" + NIFTI_SUFFIX
ANCHOR_UNION_FILE = "anchor_union" + NIFTI_SUFFIX
#: Older short names; still read, no longer written.
IMAGE_FILE = "image" + NIFTI_SUFFIX
LABELS_FILE = "labels" + NIFTI_SUFFIX
OCCUPANCY_FILE = "occupancy" + NIFTI_SUFFIX
TISSUE_FILE = "tissue" + NIFTI_SUFFIX
BIAS_FILE = "bias_field" + NIFTI_SUFFIX
FRACTION_FILE = "structure_fraction" + NIFTI_SUFFIX
BODY_FILE = "body_mask" + NIFTI_SUFFIX
SCENE_RECORD = "scene.json"
MASK_DIR = "masks"
EXAMPLE_DIR = "examples"


class SceneIOError(RuntimeError):
    """Raised when a scene cannot be written or read back."""


@dataclass(frozen=True)
class SceneVolumes:
    """The volumes of one scene, as a consumer sees them.

    ``image`` is the MRI-like volume and is ``None`` for an appearance-free
    corpus; ``occupancy`` is the binary foreground the previous milestone used
    as the image, and stays available as a baseline and for validation.
    """

    instance_labels: np.ndarray
    occupancy: np.ndarray
    image: np.ndarray | None = None
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def scene_volume(self) -> np.ndarray:
        """The binary foreground, under the name the schema uses for it."""
        return self.occupancy

    @property
    def has_image(self) -> bool:
        return self.image is not None

    def model_image(self, prefer: str = "auto") -> np.ndarray:
        """The volume a model should be fed.

        ``auto`` takes the simulated image when the corpus has one and falls
        back to the binary occupancy when it does not; ``intensity`` and
        ``occupancy`` force one or the other.
        """
        if prefer == "occupancy":
            return self.occupancy
        if prefer == "intensity":
            if self.image is None:
                raise SceneIOError(
                    "this corpus has no simulated image; regenerate it with "
                    "`enabled: true` in configs/appearance.yaml, or ask for "
                    "image_source='occupancy'"
                )
            return self.image
        if prefer != "auto":
            raise SceneIOError(
                f"image_source must be 'auto', 'intensity' or 'occupancy', got {prefer!r}"
            )
        return self.occupancy if self.image is None else self.image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def scene_directory(root: Path | str, scene_id: str) -> Path:
    """Directory a NIfTI scene lives in."""
    return Path(root) / "scenes" / scene_id


def legacy_scene_path(root: Path | str, scene_id: str) -> Path:
    """Path of the single-file ``.npz`` form of a scene."""
    return Path(root) / "scenes" / f"{scene_id}.npz"


def scene_path(root: Path | str, scene_id: str) -> Path:
    """Where this scene actually is, whichever format the corpus uses.

    Raises:
        SceneIOError: neither form exists.
    """
    directory = scene_directory(root, scene_id)
    if _labels_path(directory) is not None:
        return directory
    legacy = legacy_scene_path(root, scene_id)
    if legacy.is_file():
        return legacy
    raise SceneIOError(f"missing scene arrays for {scene_id!r} under {Path(root) / 'scenes'}")


def mask_filename(instance_id: int) -> str:
    """Older ``masks/`` entry of one structure, named by ID and shape."""
    return f"{int(instance_id):02d}_{SHAPE_VOCABULARY.id_to_name(int(instance_id))}{NIFTI_SUFFIX}"


def example_directory(root: Path | str, example_id: str) -> Path:
    """Directory holding one example's inspection NIfTIs."""
    return Path(root) / EXAMPLE_DIR / example_id


def anchor_mask_filename(slot: int, shape_name: str) -> str:
    """``examples/<example_id>/anchor_{slot}_{shape}.nii.gz``."""
    return f"anchor_{int(slot)}_{shape_name}{NIFTI_SUFFIX}"


def _first_file(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _labels_path(directory: Path) -> Path | None:
    return _first_file(directory, (INSTANCE_LABELS_FILE, LABELS_FILE))


def _image_path(directory: Path) -> Path | None:
    return _first_file(directory, (SCENE_VOLUME_FILE, IMAGE_FILE))


def _link_alias(canonical: Path, alias: Path) -> None:
    """Make ``alias`` refer to the same bytes as ``canonical``."""
    if alias == canonical:
        return
    if alias.exists() or alias.is_symlink():
        if canonical.exists() and alias.exists() and alias.samefile(canonical):
            return
        alias.unlink()
    try:
        os.link(canonical, alias)
    except OSError:
        shutil.copy2(canonical, alias)


def ensure_scene_name_aliases(directory: Path | str, *, replace: bool = False) -> None:
    """Fill in ``scene_volume`` / ``instance_labels`` from older short names.

    Used when relayouting a corpus that still has ``image.nii.gz`` /
    ``labels.nii.gz`` so the shared names exist before extras are stripped.
    """
    directory = Path(directory)
    pairs = (
        (SCENE_VOLUME_FILE, IMAGE_FILE),
        (INSTANCE_LABELS_FILE, LABELS_FILE),
    )
    for canonical_name, alias_name in pairs:
        canonical = directory / canonical_name
        alias = directory / alias_name
        if replace:
            if canonical.is_file():
                _link_alias(canonical, alias)
            elif alias.is_file():
                _link_alias(alias, canonical)
            continue
        if canonical.is_file() and not alias.is_file():
            _link_alias(canonical, alias)
        elif alias.is_file() and not canonical.is_file():
            _link_alias(alias, canonical)


def alias_scene_corpus(root: Path | str) -> int:
    """Ensure every scene directory has the shared volume names.

    Returns the number of scene directories visited.
    """
    count = 0
    for directory in iter_scene_directories(root):
        ensure_scene_name_aliases(directory, replace=False)
        count += 1
    return count


def _strip_scene_extras(directory: Path) -> None:
    """Leave only ``scene_volume.nii.gz`` and ``instance_labels.nii.gz``."""
    keep = {SCENE_VOLUME_FILE, INSTANCE_LABELS_FILE}
    for child in list(directory.iterdir()):
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists() or child.is_symlink():
            child.unlink()


def relayout_corpus(root: Path | str) -> dict[str, int]:
    """Rewrite an older nested corpus into the shared on-disk layout.

    Ensures ``scene_volume.nii.gz`` / ``instance_labels.nii.gz``, writes
    ``examples/<example_id>/{target_mask,anchor_*,anchor_union}.nii.gz`` from
    the manifests, and strips occupancy, masks, nested examples and short
    aliases from each scene directory.

    Returns counts of scenes and examples rewritten.
    """
    from src.data.schema import read_manifest

    root = Path(root)
    n_scenes = alias_scene_corpus(root)
    by_scene: dict[str, list[Any]] = {}
    for split in ("train", "val", "test"):
        for name in (f"{split}.jsonl", f"{split}_candidates.jsonl"):
            path = root / "manifests" / name
            if not path.is_file():
                continue
            for metadata in read_manifest(path):
                by_scene.setdefault(metadata.scene_id, []).append(metadata)
    n_examples = 0
    seen: set[str] = set()
    for scene_id, metadatas in by_scene.items():
        volumes = load_scene(scene_directory(root, scene_id))
        for metadata in metadatas:
            if metadata.example_id in seen:
                continue
            seen.add(metadata.example_id)
            save_example_masks(
                root,
                metadata.example_id,
                instance_labels=volumes.instance_labels,
                target_instance_id=metadata.target_instance_id,
                anchor_instance_ids=metadata.anchor_instance_ids,
                anchor_shape_names=metadata.anchor_shape_names,
                spacing=volumes.spacing,
            )
            n_examples += 1
        _strip_scene_extras(scene_directory(root, scene_id))
    return {"scenes": n_scenes, "examples": n_examples}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def save_scene(
    root: Path | str,
    scene_id: str,
    *,
    instance_labels: np.ndarray,
    occupancy: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    image: np.ndarray | None = None,
    record: Mapping[str, Any] | None = None,
    extra_volumes: Mapping[str, np.ndarray] | None = None,
    write_structure_masks: bool = False,
    inspection_extras: bool = False,
    image_dtype: np.dtype | str = np.uint16,
) -> Path:
    """Write one scene as ``scene_volume.nii.gz`` and ``instance_labels.nii.gz``.

    Args:
        root: the dataset root; the scene lands in ``<root>/scenes/<scene_id>``.
        scene_id: the scene's identifier.
        instance_labels: ``(D, H, W)`` labels, 0 background and 1..10 instances.
        occupancy: ``(D, H, W)`` binary foreground. Written only with
            ``inspection_extras``.
        spacing: world units per voxel, ordered ``(x, y, z)``.
        image: the simulated MRI-like volume, when the corpus has one.
        image_dtype: on-disk type of ``scene_volume.nii.gz``. An integer type
            makes nibabel store a header scale factor, which is how scanner
            data is written and halves the file; the volume still reads back
            in its simulated units.
        record: JSON-serialisable scene record written to ``scene.json`` when
            ``inspection_extras`` is on.
        extra_volumes: optional diagnostics keyed by file name, for example
            ``{"tissue.nii.gz": ...}``.
        write_structure_masks: also write one binary mask per structure.
        inspection_extras: also write occupancy, short-name aliases, per-structure
            masks and ``scene.json``. Off by default so a copy of the corpus
            matches ``exp/realistic-appearance``.

    Returns:
        The scene directory.
    """
    directory = scene_directory(root, scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(instance_labels, dtype=np.uint8)

    save_nifti(
        directory / INSTANCE_LABELS_FILE,
        labels,
        spacing,
        dtype=np.uint8,
        description="instance labels 0-10",
    )
    if image is not None:
        save_nifti(
            directory / SCENE_VOLUME_FILE,
            np.asarray(image, dtype=np.float32),
            spacing,
            dtype=np.dtype(image_dtype),
            description="simulated MRI-like volume",
        )
    else:
        leftover = directory / SCENE_VOLUME_FILE
        if leftover.exists() or leftover.is_symlink():
            leftover.unlink()

    if inspection_extras:
        save_nifti(
            directory / OCCUPANCY_FILE,
            np.asarray(occupancy, dtype=np.uint8),
            spacing,
            dtype=np.uint8,
            description="binary foreground",
        )
        ensure_scene_name_aliases(directory, replace=True)
        write_structure_masks = True
        if record is not None:
            (directory / SCENE_RECORD).write_text(
                json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    for name, volume in (extra_volumes or {}).items():
        array = np.asarray(volume)
        dtype = np.uint8 if array.dtype == np.uint8 else np.float32
        save_nifti(directory / name, array.astype(dtype), spacing, dtype=dtype)

    if write_structure_masks:
        mask_root = directory / MASK_DIR
        mask_root.mkdir(parents=True, exist_ok=True)
        for spec in SHAPE_VOCABULARY:
            save_nifti(
                mask_root / mask_filename(spec.id),
                (labels == spec.id).astype(np.uint8),
                spacing,
                dtype=np.uint8,
                description=f"{spec.name} (instance {spec.id})",
            )
    return directory


def save_example_masks(
    root: Path | str,
    example_id: str,
    *,
    instance_labels: np.ndarray,
    target_instance_id: int,
    anchor_instance_ids: Sequence[int],
    anchor_shape_names: Sequence[str],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Path:
    """Write one example under ``examples/<example_id>/``.

    Files::

        target_mask.nii.gz
        anchor_{slot}_{shape}.nii.gz   one per clause, prompt order
        anchor_union.nii.gz
    """
    if len(anchor_instance_ids) != len(anchor_shape_names):
        raise SceneIOError(
            f"anchor_instance_ids ({len(anchor_instance_ids)}) and "
            f"anchor_shape_names ({len(anchor_shape_names)}) must be the same length"
        )
    directory = example_directory(root, example_id)
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(instance_labels)
    save_nifti(
        directory / TARGET_MASK_FILE,
        (labels == int(target_instance_id)).astype(np.uint8),
        spacing,
        dtype=np.uint8,
        description="target mask (label, never a model input)",
    )
    union = np.zeros(labels.shape, dtype=np.uint8)
    for slot, (instance_id, shape_name) in enumerate(zip(anchor_instance_ids, anchor_shape_names)):
        mask = (labels == int(instance_id)).astype(np.uint8)
        union = np.maximum(union, mask)
        save_nifti(
            directory / anchor_mask_filename(slot, shape_name),
            mask,
            spacing,
            dtype=np.uint8,
            description=f"anchor slot {slot} ({shape_name})",
        )
    save_nifti(
        directory / ANCHOR_UNION_FILE,
        union,
        spacing,
        dtype=np.uint8,
        description="union of the three ordered anchor channels",
    )
    return directory


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load_scene(path: Path | str) -> SceneVolumes:
    """Read a scene from either on-disk format.

    Args:
        path: a scene directory, or a legacy ``<scene_id>.npz`` file.
    """
    path = Path(path)
    if path.is_dir():
        labels_path = _labels_path(path)
        if labels_path is None:
            raise SceneIOError(f"missing instance labels under {path}")
        labels, spacing = load_nifti(labels_path, dtype=np.uint8)
        occupancy_path = path / OCCUPANCY_FILE
        if occupancy_path.is_file():
            occupancy, _ = load_nifti(occupancy_path, dtype=np.uint8)
        else:
            occupancy = (labels != 0).astype(np.uint8)
        image = None
        image_path = _image_path(path)
        if image_path is not None:
            image, _ = load_nifti(image_path, dtype=np.float32)
        return SceneVolumes(
            instance_labels=labels, occupancy=occupancy, image=image, spacing=spacing
        )
    if path.is_file() and path.suffix == ".npz":
        with np.load(path) as data:
            return SceneVolumes(
                instance_labels=np.asarray(data["instance_labels"]),
                occupancy=np.asarray(data["scene_volume"]),
                image=np.asarray(data["image"]) if "image" in data else None,
            )
    raise SceneIOError(f"{path} is neither a scene directory nor a scene .npz")


def load_scene_record(path: Path | str) -> dict[str, Any]:
    """Read ``scene.json`` of a NIfTI scene directory."""
    record = Path(path) / SCENE_RECORD
    if not record.is_file():
        raise SceneIOError(f"missing {record}")
    return json.loads(record.read_text(encoding="utf-8"))


def load_structure_mask(path: Path | str, instance_id: int) -> np.ndarray:
    """Read one structure's stored binary mask."""
    mask, _ = load_nifti(Path(path) / MASK_DIR / mask_filename(instance_id), dtype=np.uint8)
    return mask


def iter_scene_directories(root: Path | str) -> Iterable[Path]:
    """Every scene directory under a dataset root, in scene-ID order."""
    scenes = Path(root) / "scenes"
    if not scenes.is_dir():
        return []
    return sorted(path for path in scenes.iterdir() if _labels_path(path) is not None)


__all__ = [
    "ANCHOR_UNION_FILE",
    "BODY_FILE",
    "BIAS_FILE",
    "EXAMPLE_DIR",
    "FRACTION_FILE",
    "IMAGE_FILE",
    "INSTANCE_LABELS_FILE",
    "LABELS_FILE",
    "MASK_DIR",
    "NIFTI_FORMAT",
    "NPZ_FORMAT",
    "OCCUPANCY_FILE",
    "SCENE_RECORD",
    "SCENE_VOLUME_FILE",
    "TARGET_MASK_FILE",
    "TISSUE_FILE",
    "SceneIOError",
    "SceneVolumes",
    "alias_scene_corpus",
    "anchor_mask_filename",
    "ensure_scene_name_aliases",
    "example_directory",
    "iter_scene_directories",
    "legacy_scene_path",
    "load_scene",
    "load_scene_record",
    "load_structure_mask",
    "mask_filename",
    "relayout_corpus",
    "save_example_masks",
    "save_scene",
    "scene_affine",
    "scene_directory",
    "scene_path",
]
