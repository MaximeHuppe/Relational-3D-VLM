"""On-disk layout of a generated scene.

A scene is a directory of NIfTI volumes plus one JSON record::

    scenes/<scene_id>/
        image.nii.gz                 the simulated MRI-like volume; uint16
                                     with a header scale factor by default, so
                                     it reads back in its simulated units
        labels.nii.gz                uint8    instance labels, 0 and 1..10
        occupancy.nii.gz             uint8    binary union of the ten structures
        masks/<id>_<name>.nii.gz     uint8    one binary mask per structure
        tissue.nii.gz                float32  noise-free tissue map    (optional)
        bias_field.nii.gz            float32  receive-coil gain        (optional)
        structure_fraction.nii.gz    float32  total partial volume     (optional)
        body_mask.nii.gz             uint8    head outline             (optional)
        examples/<example_id>_target.nii.gz   uint8    (optional)
        examples/<example_id>_anchors.nii.gz  uint8 4D (optional)
        examples/<example_id>.json            the example's prompt record
        scene.json                   seeds, versions, geometry, appearance draws

Everything is redundant on purpose. ``labels.nii.gz`` alone determines
``occupancy``, every per-structure mask and every example's target and anchor
channels, but writing them out means any scene can be opened in a viewer, or
loaded by an analysis script, without re-deriving anything or importing this
package.

The previous single-file format (``scenes/<scene_id>.npz`` holding
``scene_volume`` and ``instance_labels``) is still readable: :func:`load_scene`
dispatches on what is actually on disk, so a corpus generated before this
change keeps working and simply has no ``image``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.nifti_io import NIFTI_SUFFIX, load_nifti, save_nifti, scene_affine
from src.data.primitives import SHAPE_VOCABULARY

#: Format tokens accepted by ``storage.scene_array_format``.
NIFTI_FORMAT = "nifti"
NPZ_FORMAT = "npz_compressed"

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
    if (directory / LABELS_FILE).is_file():
        return directory
    legacy = legacy_scene_path(root, scene_id)
    if legacy.is_file():
        return legacy
    raise SceneIOError(f"missing scene arrays for {scene_id!r} under {Path(root) / 'scenes'}")


def mask_filename(instance_id: int) -> str:
    """``masks/`` entry of one structure, named by ID and shape."""
    return f"{int(instance_id):02d}_{SHAPE_VOCABULARY.id_to_name(int(instance_id))}{NIFTI_SUFFIX}"


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
    write_structure_masks: bool = True,
    image_dtype: np.dtype | str = np.uint16,
) -> Path:
    """Write one scene as a directory of NIfTI volumes plus ``scene.json``.

    Args:
        root: the dataset root; the scene lands in ``<root>/scenes/<scene_id>``.
        scene_id: the scene's identifier.
        instance_labels: ``(D, H, W)`` labels, 0 background and 1..10 instances.
        occupancy: ``(D, H, W)`` binary foreground.
        spacing: world units per voxel, ordered ``(x, y, z)``.
        image: the simulated MRI-like volume, when the corpus has one.
        image_dtype: on-disk type of ``image.nii.gz``. An integer type makes
            nibabel store a header scale factor, which is how scanner data is
            written and halves the file; the volume still reads back in its
            simulated units.
        record: JSON-serialisable scene record written to ``scene.json``.
        extra_volumes: optional diagnostics keyed by file name, for example
            ``{"tissue.nii.gz": ...}``.
        write_structure_masks: also write one binary mask per structure.

    Returns:
        The scene directory.
    """
    directory = scene_directory(root, scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(instance_labels, dtype=np.uint8)

    save_nifti(directory / LABELS_FILE, labels, spacing, dtype=np.uint8, description="instance labels 0-10")
    save_nifti(
        directory / OCCUPANCY_FILE,
        np.asarray(occupancy, dtype=np.uint8),
        spacing,
        dtype=np.uint8,
        description="binary foreground",
    )
    if image is not None:
        save_nifti(
            directory / IMAGE_FILE,
            np.asarray(image, dtype=np.float32),
            spacing,
            dtype=np.dtype(image_dtype),
            description="simulated MRI-like volume",
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

    if record is not None:
        (directory / SCENE_RECORD).write_text(
            json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return directory


def save_example_masks(
    root: Path | str,
    scene_id: str,
    example_id: str,
    *,
    instance_labels: np.ndarray,
    target_instance_id: int,
    anchor_instance_ids: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    record: Mapping[str, Any] | None = None,
) -> Path:
    """Write one example's target mask, ordered anchor stack and prompt record.

    The anchor stack is a 4D volume whose fourth dimension is the clause slot,
    so opening it shows the three anchors in prompt order as a volume series.
    """
    directory = scene_directory(root, scene_id) / EXAMPLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(instance_labels)
    save_nifti(
        directory / f"{example_id}_target{NIFTI_SUFFIX}",
        (labels == int(target_instance_id)).astype(np.uint8),
        spacing,
        dtype=np.uint8,
        description="target mask (label, never a model input)",
    )
    anchors = np.stack(
        [(labels == int(instance_id)).astype(np.uint8) for instance_id in anchor_instance_ids]
    )
    save_nifti(
        directory / f"{example_id}_anchors{NIFTI_SUFFIX}",
        anchors,
        spacing,
        dtype=np.uint8,
        description="ordered anchor channels, 4th dim = clause slot",
    )
    if record is not None:
        (directory / f"{example_id}.json").write_text(
            json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        labels, spacing = load_nifti(path / LABELS_FILE, dtype=np.uint8)
        occupancy_path = path / OCCUPANCY_FILE
        if occupancy_path.is_file():
            occupancy, _ = load_nifti(occupancy_path, dtype=np.uint8)
        else:  # pragma: no cover - occupancy is always written
            occupancy = (labels != 0).astype(np.uint8)
        image = None
        if (path / IMAGE_FILE).is_file():
            image, _ = load_nifti(path / IMAGE_FILE, dtype=np.float32)
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
    return sorted(path for path in scenes.iterdir() if (path / LABELS_FILE).is_file())


__all__ = [
    "BODY_FILE",
    "BIAS_FILE",
    "EXAMPLE_DIR",
    "FRACTION_FILE",
    "IMAGE_FILE",
    "LABELS_FILE",
    "MASK_DIR",
    "NIFTI_FORMAT",
    "NPZ_FORMAT",
    "OCCUPANCY_FILE",
    "SCENE_RECORD",
    "TISSUE_FILE",
    "SceneIOError",
    "SceneVolumes",
    "iter_scene_directories",
    "legacy_scene_path",
    "load_scene",
    "load_scene_record",
    "load_structure_mask",
    "mask_filename",
    "save_example_masks",
    "save_scene",
    "scene_affine",
    "scene_directory",
    "scene_path",
]
