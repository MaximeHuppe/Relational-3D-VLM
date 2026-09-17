"""MRI-like intensities painted onto packed instance labels.

Geometry stays in ``instance_labels`` (hard 0/1..10). This module only produces
the float ``scene_volume`` that Stage A sees: noisy background, overlapping
per-instance structure means, intra-object noise, and a light Gaussian blur so
edges are partial-volume rather than a 0/1 cut.

Structure means are drawn from one shared range so Stage A cannot classify by
brightness. A small deterministic per-class bias (T1-like) is allowed; it is
not a unique lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.data.primitives import SHAPE_VOCABULARY

BACKGROUND_LABEL = SHAPE_VOCABULARY.background_label


@dataclass(frozen=True)
class AppearanceSettings:
    """Resolved ``appearance:`` block from ``configs/generator.yaml``."""

    enabled: bool
    background_mean: float
    background_std: float
    structure_mean_range: tuple[float, float]
    structure_mean_jitter: float
    intra_object_std: float
    class_bias: float
    partial_volume_sigma_voxels: float
    clip: tuple[float, float]

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "AppearanceSettings":
        block = dict(config or {})
        mean_range = tuple(float(v) for v in block.get("structure_mean_range", (0.45, 0.75)))
        clip = tuple(float(v) for v in block.get("clip", (0.0, 1.0)))
        if len(mean_range) != 2 or mean_range[0] >= mean_range[1]:
            raise ValueError(f"structure_mean_range must be a low < high pair, got {mean_range}")
        if len(clip) != 2 or clip[0] >= clip[1]:
            raise ValueError(f"clip must be a low < high pair, got {clip}")
        return cls(
            enabled=bool(block.get("enabled", True)),
            background_mean=float(block.get("background_mean", 0.12)),
            background_std=float(block.get("background_std", 0.04)),
            structure_mean_range=(mean_range[0], mean_range[1]),
            structure_mean_jitter=float(block.get("structure_mean_jitter", 0.05)),
            intra_object_std=float(block.get("intra_object_std", 0.03)),
            class_bias=float(block.get("class_bias", 0.04)),
            partial_volume_sigma_voxels=float(block.get("partial_volume_sigma_voxels", 0.6)),
            clip=(clip[0], clip[1]),
        )


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_blur_3d(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur; ``sigma <= 0`` is a no-op."""
    if sigma <= 0:
        return np.asarray(volume, dtype=np.float32)
    kernel = _gaussian_kernel1d(sigma)
    out = np.asarray(volume, dtype=np.float32)
    for axis in range(out.ndim):
        out = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="same"), axis, out)
    return out.astype(np.float32, copy=False)


def class_intensity_bias(shape_id: int, amplitude: float, num_classes: int = 10) -> float:
    """Small deterministic offset in roughly ``[-amplitude, +amplitude]``."""
    if amplitude == 0:
        return 0.0
    centre = (num_classes + 1) / 2.0
    span = (num_classes - 1) / 2.0
    return float(amplitude * (shape_id - centre) / span)


def render_appearance(
    instance_labels: np.ndarray,
    rng: np.random.Generator,
    settings: AppearanceSettings,
) -> np.ndarray:
    """Paint a float32 intensity volume from packed labels.

    Labels are not modified. When ``settings.enabled`` is false the binary
    occupancy ``(labels != 0)`` is returned as float32.
    """
    labels = np.asarray(instance_labels)
    foreground = labels != BACKGROUND_LABEL
    if not settings.enabled:
        return foreground.astype(np.float32)

    volume = rng.normal(
        loc=settings.background_mean,
        scale=settings.background_std,
        size=labels.shape,
    ).astype(np.float32)

    low, high = settings.structure_mean_range
    mid = 0.5 * (low + high)
    for shape_id in range(1, len(SHAPE_VOCABULARY) + 1):
        mask = labels == shape_id
        if not mask.any():
            continue
        mean = mid + class_intensity_bias(shape_id, settings.class_bias)
        mean += float(rng.uniform(-settings.structure_mean_jitter, settings.structure_mean_jitter))
        mean = float(np.clip(mean, low, high))
        noise = rng.normal(0.0, settings.intra_object_std, size=int(mask.sum())).astype(np.float32)
        volume[mask] = mean + noise

    volume = gaussian_blur_3d(volume, settings.partial_volume_sigma_voxels)
    low_clip, high_clip = settings.clip
    np.clip(volume, low_clip, high_clip, out=volume)
    return volume
