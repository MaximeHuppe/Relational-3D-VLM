"""Occupancy sanity slices: is the prediction one remaining object, or the union?

Each dump is a mid-volume (or most-occupied) axial slice of occupancy, the
prediction, the target, and an RGB overlay. Occupancy is the decoder WHAT
stream — the scene's objects with the three anchors removed — so a correct
mask should light up one of those remaining components, not all of them. The
volume passed in is the label-derived binary occupancy, not the intensity image
the decoder sees, because components cannot be counted on an acquisition.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path | str, image: np.ndarray) -> Path:
    """Write an ``H x W`` or ``H x W x 3`` uint8 PNG with no extra dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxW or HxWx3, got {array.shape}")
    pixels = np.clip(array, 0, 255).astype(np.uint8)
    height, width, _ = pixels.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return path


def pick_slice_index(occupancy: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> int:
    """Axial plane with the most occupied voxels across occupancy / pred / target."""
    energy = occupancy.sum(axis=(1, 2)) + prediction.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    if float(energy.max()) == 0.0:
        return occupancy.shape[0] // 2
    return int(energy.argmax())


def overlay_rgb(occupancy: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gray occupancy, red prediction, green target. Yellow where pred and target agree."""
    occupancy_f = np.clip(occupancy.astype(np.float32), 0.0, 1.0)
    prediction_f = np.clip(prediction.astype(np.float32), 0.0, 1.0)
    target_f = np.clip(target.astype(np.float32), 0.0, 1.0)
    rgb = np.zeros((*occupancy.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(0.35 * occupancy_f + prediction_f, 0.0, 1.0)
    rgb[..., 1] = np.clip(0.35 * occupancy_f + target_f, 0.0, 1.0)
    rgb[..., 2] = 0.35 * occupancy_f
    return (rgb * 255.0).astype(np.uint8)


def save_occupancy_slice(
    path: Path | str,
    occupancy: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    slice_index: int | None = None,
) -> dict[str, int | str]:
    """Write a 2x2 montage PNG: occupancy, prediction, target, overlay."""
    if occupancy.shape != prediction.shape or occupancy.shape != target.shape:
        raise ValueError(
            f"volumes must match, got occupancy {occupancy.shape}, "
            f"prediction {prediction.shape}, target {target.shape}"
        )
    index = pick_slice_index(occupancy, prediction, target) if slice_index is None else int(slice_index)
    occ = occupancy[index]
    pred = prediction[index]
    tgt = target[index]

    def gray(plane: np.ndarray) -> np.ndarray:
        return (np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8)

    pred_gray = gray(pred)
    tgt_gray = gray(tgt)
    zeros = np.zeros_like(pred_gray)
    red = np.stack([pred_gray, zeros, zeros], axis=-1)
    green = np.stack([np.zeros_like(tgt_gray), tgt_gray, np.zeros_like(tgt_gray)], axis=-1)
    occ_rgb = np.stack([gray(occ)] * 3, axis=-1)
    overlay = overlay_rgb(occ, pred, tgt)
    top = np.concatenate([occ_rgb, red], axis=1)
    bottom = np.concatenate([green, overlay], axis=1)
    montage = np.concatenate([top, bottom], axis=0)
    written = write_png(path, montage)
    return {"path": str(written), "slice_index": index}
