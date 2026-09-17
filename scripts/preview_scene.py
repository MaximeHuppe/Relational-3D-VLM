#!/usr/bin/env python3
"""Render a generated scene to PNG, so a corpus can be eyeballed without a viewer.

Writes one montage per scene: the three orthogonal mid-planes of the simulated
image, the same planes with the instance labels overlaid, and - when an example
is named - the target and the three ordered anchor channels of that example.

The point is the first row. If the scene still looks like white blocks on black,
the appearance model is off and the corpus is the previous milestone's.

Usage::

    .venv/bin/python scripts/preview_scene.py --root data/smoke --limit 4
    .venv/bin/python scripts/preview_scene.py --scene scene_009000000 \\
        --example scene_009000000_target_01
    .venv/bin/python scripts/preview_scene.py --root data/processed --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.data.primitives import SHAPE_VOCABULARY  # noqa: E402
from src.data.scene_io import (  # noqa: E402
    EXAMPLE_DIR,
    SceneVolumes,
    iter_scene_directories,
    load_scene,
    scene_directory,
)
from src.data.schema import read_manifest  # noqa: E402
from src.evaluation.qualitative import write_png  # noqa: E402

#: Green for the supervised target, so it never collides with an anchor slot.
TARGET_COLOR: tuple[int, int, int] = (0, 230, 64)

#: Anchor slots 1-3, chosen to stay apart from the target green.
ANCHOR_COLORS: tuple[tuple[int, int, int], ...] = (
    (228, 26, 28),
    (55, 126, 184),
    (255, 127, 0),
)

#: Ten visually separable label colours, one per instance ID.
LABEL_COLORS: tuple[tuple[int, int, int], ...] = (
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (255, 217, 47),
    (166, 86, 40),
    (247, 129, 191),
    (0, 206, 209),
    (153, 153, 153),
)


def window(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Robust percentile windowing to ``uint8``, the usual way MRI is displayed."""
    values = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(values, [low, high])
    span = float(hi - lo)
    scaled = (values - float(lo)) / span if span > 0 else values - float(lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def plane_indices(
    volume_shape: tuple[int, ...], focus: np.ndarray | None = None
) -> tuple[int, int, int]:
    """``(z, y, x)`` planes to cut. Through ``focus`` when given, else mid-volume.

    The image rows and the combined overlay share one set of indices, so the
    overlay really is the configuration at those planes; each single-channel row
    is then cut through its own structure, because a row that shows nothing
    tells you nothing about where that anchor is.
    """
    depth, height, width = volume_shape
    if focus is None or not np.asarray(focus).any():
        return (depth // 2, height // 2, width // 2)
    mask = np.asarray(focus)
    return tuple(
        int(mask.sum(axis=tuple(other for other in range(3) if other != axis)).argmax())
        for axis in range(3)
    )  # type: ignore[return-value]


def orthogonal_planes(
    volume: np.ndarray, indices: tuple[int, int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axial, coronal and sagittal planes of a ``(z, y, x)`` array.

    Each plane is returned with superior (or anterior) up, which is how a viewer
    shows it, so the montage and ITK-SNAP agree on which way is up.
    """
    z, y, x = plane_indices(volume.shape[:3], None) if indices is None else indices
    return (
        np.flipud(volume[z, :, :]),
        np.flipud(volume[:, y, :]),
        np.flipud(volume[:, :, x]),
    )


def pad_to(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad a plane with zeros so planes of different sizes can be tiled."""
    height, width = shape
    out = np.zeros((height, width) + plane.shape[2:], dtype=plane.dtype)
    out[: plane.shape[0], : plane.shape[1]] = plane
    return out


def row(planes: Sequence[np.ndarray], gap: int = 4) -> np.ndarray:
    """Tile planes horizontally with a thin separator."""
    height = max(plane.shape[0] for plane in planes)
    width = max(plane.shape[1] for plane in planes)
    padded = [pad_to(plane, (height, width)) for plane in planes]
    separator = np.zeros((height, gap, 3), dtype=np.uint8)
    tiles: list[np.ndarray] = []
    for index, plane in enumerate(padded):
        if index:
            tiles.append(separator)
        tiles.append(plane)
    return np.concatenate(tiles, axis=1)


def to_rgb(plane: np.ndarray) -> np.ndarray:
    return np.stack([plane] * 3, axis=-1)


def overlay_labels(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend the instance colours over a windowed grey plane."""
    rgb = to_rgb(gray).astype(np.float32)
    for instance_id in range(1, len(SHAPE_VOCABULARY) + 1):
        mask = labels == instance_id
        if not mask.any():
            continue
        colour = np.array(LABEL_COLORS[instance_id - 1], dtype=np.float32)
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colour
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_planes(
    coloured: np.ndarray,
    indices: tuple[int, int, int],
    background: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Cut an already-coloured ``(z, y, x, 3)`` volume over a dimmed image."""
    z, y, x = indices
    cuts = (
        np.flipud(coloured[z, :, :]),
        np.flipud(coloured[:, y, :]),
        np.flipud(coloured[:, :, x]),
    )
    out: list[np.ndarray] = []
    for cut, grey_plane in zip(cuts, background):
        rgb = 0.35 * to_rgb(grey_plane).astype(np.float32)
        painted = cut.sum(axis=-1) > 0
        rgb[painted] = cut[painted]
        out.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return out


def mask_planes(
    mask: np.ndarray,
    colour: tuple[int, int, int],
    indices: tuple[int, int, int],
    grey: np.ndarray,
) -> list[np.ndarray]:
    """One coloured plane per orthogonal view of a binary mask, over a dim image."""
    coloured = np.zeros(np.asarray(mask).shape + (3,), dtype=np.float32)
    coloured[np.asarray(mask) > 0] = np.array(colour, dtype=np.float32)
    return overlay_planes(coloured, indices, list(orthogonal_planes(grey, indices)))


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour zoom, so a 64-voxel plane is legible on screen."""
    if factor <= 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def render_scene(
    volumes: SceneVolumes,
    *,
    example_volumes: tuple[np.ndarray, np.ndarray] | None = None,
    zoom: int = 3,
) -> np.ndarray:
    """Build the montage for one scene, optionally with one example's channels.

    Row 1 is the image as a viewer would window it, row 2 the same planes with
    the ten instance labels over them. With an example, row 3 is the target
    (green) and the three anchors together at the target's planes, and rows 4-7
    are the target and then each anchor in clause order, each at its own planes.
    """
    image = volumes.model_image("auto")
    grey = window(image)
    focus = None if example_volumes is None else example_volumes[0]
    indices = plane_indices(grey.shape, focus)

    grey_planes = list(orthogonal_planes(grey, indices))
    label_planes = list(orthogonal_planes(volumes.instance_labels, indices))
    rows = [
        row([to_rgb(plane) for plane in grey_planes]),
        row(
            [
                overlay_labels(gray_plane, label_plane)
                for gray_plane, label_plane in zip(grey_planes, label_planes)
            ]
        ),
    ]

    if example_volumes is not None:
        target, anchors = example_volumes
        # One combined view at the target's planes: the relation configuration.
        combined = np.zeros(grey.shape + (3,), dtype=np.float32)
        combined[np.asarray(target) > 0] = TARGET_COLOR
        for slot in range(anchors.shape[0]):
            combined[np.asarray(anchors[slot]) > 0] = ANCHOR_COLORS[slot]
        rows.append(row(overlay_planes(combined, indices, grey_planes)))
        # Then one row per channel, each cut through its own structure.
        rows.append(row(mask_planes(target, TARGET_COLOR, plane_indices(grey.shape, target), grey)))
        for slot in range(anchors.shape[0]):
            rows.append(
                row(
                    mask_planes(
                        anchors[slot],
                        ANCHOR_COLORS[slot],
                        plane_indices(grey.shape, anchors[slot]),
                        grey,
                    )
                )
            )

    width = max(block.shape[1] for block in rows)
    gap = np.zeros((4, width, 3), dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for index, block in enumerate(rows):
        if index:
            stacked.append(gap)
        stacked.append(pad_to(block, (block.shape[0], width)))
    return upscale(np.concatenate(stacked, axis=0), zoom)


def example_arrays(root: Path, scene_id: str, example_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one example's stored target and anchor stack, else derive them."""
    from src.data.nifti_io import load_nifti

    directory = scene_directory(root, scene_id) / EXAMPLE_DIR
    target_path = directory / f"{example_id}_target.nii.gz"
    anchors_path = directory / f"{example_id}_anchors.nii.gz"
    if target_path.is_file() and anchors_path.is_file():
        target, _ = load_nifti(target_path, dtype=np.uint8)
        anchors, _ = load_nifti(anchors_path, dtype=np.uint8)
        return target, anchors

    labels = load_scene(scene_directory(root, scene_id)).instance_labels
    for split in ("train", "val", "test"):
        manifest = root / "manifests" / f"{split}.jsonl"
        if not manifest.is_file():
            continue
        for metadata in read_manifest(manifest):
            if metadata.example_id != example_id:
                continue
            target = (labels == metadata.target_instance_id).astype(np.uint8)
            anchors = np.stack(
                [(labels == i).astype(np.uint8) for i in metadata.anchor_instance_ids]
            )
            return target, anchors
    raise SystemExit(f"example {example_id!r} is not in {root}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "smoke")
    parser.add_argument("--scene", default=None, help="one scene id (default: the first few)")
    parser.add_argument("--example", default=None, help="also draw this example's channels")
    parser.add_argument("--limit", type=int, default=3, help="how many scenes to render")
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: <root>/previews)"
    )
    parser.add_argument("--zoom", type=int, default=3, help="nearest-neighbour zoom factor")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out is not None else root / "previews"
    out.mkdir(parents=True, exist_ok=True)

    if args.scene is not None:
        directories = [scene_directory(root, args.scene)]
    else:
        directories = list(iter_scene_directories(root))[: max(args.limit, 0)]
    if not directories:
        raise SystemExit(f"no scenes found under {root / 'scenes'}")

    for directory in directories:
        volumes = load_scene(directory)
        pair = None
        if args.example is not None:
            pair = example_arrays(root, directory.name, args.example)
        montage = render_scene(volumes, example_volumes=pair, zoom=args.zoom)
        name = directory.name if args.example is None else f"{directory.name}_{args.example}"
        path = write_png(out / f"{name}.png", montage)
        print(f"wrote {path}  ({montage.shape[1]}x{montage.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
