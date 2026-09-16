"""Analytic voxelisers for the ten primitives.

Contract for every voxeliser in this module:

* returns a boolean array of shape ``(D, H, W)`` indexed ``(z, y, x)``;
* takes its centre and dimensions in world units ``(x, y, z)`` and the voxel
  ``spacing``;
* is axis-aligned - orientation is fixed per class in ``configs/shapes.yaml``
  and is not sampled in this milestone;
* is deterministic: same parameters, same voxels.

A voxel belongs to a shape when its centre lies inside the analytic solid, so
the voxelised extent of a solid of continuous size ``s`` is ``s`` or ``s + 1``
voxels depending on where the centre falls between two voxel centres. Sampling
ranges in ``configs/shapes.yaml`` describe the continuous size; the measured
voxel extents are reported by the smoke run.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from src.data.primitives import SHAPE_VOCABULARY

#: Parameter names each primitive expects, in ``configs/shapes.yaml`` order.
SHAPE_PARAMS: dict[str, tuple[str, ...]] = {
    "cube": ("side",),
    "cuboid": ("size_x", "size_y", "size_z"),
    "sphere": ("radius",),
    "ellipsoid": ("radius_x", "radius_y", "radius_z"),
    "cylinder": ("radius", "height"),
    "cone": ("radius", "height"),
    "pyramid": ("base_side", "height"),
    "triangular_prism": ("base_width", "height", "depth"),
    "torus": ("major_radius", "minor_radius"),
    "capsule": ("radius", "segment_length"),
}


class VoxelizationError(ValueError):
    """Raised for unknown shapes, missing parameters or degenerate sizes."""


def world_coordinate_grids(
    volume_shape: Sequence[int], spacing: Sequence[float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcastable world coordinates of every voxel centre.

    Returns ``(z, y, x)`` with shapes ``(D, 1, 1)``, ``(1, H, 1)``, ``(1, 1, W)``
    so that expressions over them broadcast to ``(D, H, W)`` without ever
    materialising three full grids.
    """
    depth, height, width = (int(v) for v in volume_shape)
    sx, sy, sz = (float(v) for v in spacing)
    z = (np.arange(depth, dtype=np.float64) * sz)[:, None, None]
    y = (np.arange(height, dtype=np.float64) * sy)[None, :, None]
    x = (np.arange(width, dtype=np.float64) * sx)[None, None, :]
    return z, y, x


def normalized_world_coordinates(
    volume_shape: Sequence[int], spacing: Sequence[float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """``(3, D, H, W)`` world coordinates ``(x, y, z)`` scaled to ``[-1, 1]``.

    Computed from world geometry, not from tensor indices, so it stays correct
    after a resize or a crop. Stage B consumes this at every scale.
    """
    z, y, x = world_coordinate_grids(volume_shape, spacing)
    depth, height, width = (int(v) for v in volume_shape)
    sx, sy, sz = (float(v) for v in spacing)
    spans = (max(width - 1, 1) * sx, max(height - 1, 1) * sy, max(depth - 1, 1) * sz)
    grids = [
        np.broadcast_to(2.0 * x / spans[0] - 1.0, (depth, height, width)),
        np.broadcast_to(2.0 * y / spans[1] - 1.0, (depth, height, width)),
        np.broadcast_to(2.0 * z / spans[2] - 1.0, (depth, height, width)),
    ]
    return np.stack(grids).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-primitive solids, expressed on centred offsets (dx, dy, dz)
# ---------------------------------------------------------------------------
def _cube(dx, dy, dz, p):
    half = p["side"] / 2.0
    return (np.abs(dx) <= half) & (np.abs(dy) <= half) & (np.abs(dz) <= half)


def _cuboid(dx, dy, dz, p):
    return (
        (np.abs(dx) <= p["size_x"] / 2.0)
        & (np.abs(dy) <= p["size_y"] / 2.0)
        & (np.abs(dz) <= p["size_z"] / 2.0)
    )


def _sphere(dx, dy, dz, p):
    radius = p["radius"]
    return (dx**2 + dy**2 + dz**2) <= radius**2


def _ellipsoid(dx, dy, dz, p):
    return (
        (dx / p["radius_x"]) ** 2
        + (dy / p["radius_y"]) ** 2
        + (dz / p["radius_z"]) ** 2
    ) <= 1.0


def _cylinder(dx, dy, dz, p):
    """Circular cross-section in x-y, extruded along z."""
    return ((dx**2 + dy**2) <= p["radius"] ** 2) & (np.abs(dz) <= p["height"] / 2.0)


def _cone(dx, dy, dz, p):
    """Circular base at low z tapering to an apex at high z."""
    height, radius = p["height"], p["radius"]
    half = height / 2.0
    inside_z = np.abs(dz) <= half
    # t = 0 at the base, 1 at the apex.
    t = np.clip((dz + half) / height, 0.0, 1.0)
    radius_at_z = radius * (1.0 - t)
    return inside_z & ((dx**2 + dy**2) <= radius_at_z**2)


def _pyramid(dx, dy, dz, p):
    """Square base at low z tapering to an apex at high z."""
    height, base = p["height"], p["base_side"]
    half = height / 2.0
    inside_z = np.abs(dz) <= half
    t = np.clip((dz + half) / height, 0.0, 1.0)
    half_side = (base / 2.0) * (1.0 - t)
    return inside_z & (np.abs(dx) <= half_side) & (np.abs(dy) <= half_side)


def _triangular_prism(dx, dy, dz, p):
    """Isoceles triangle in the x-z plane, extruded along y."""
    height, width, depth = p["height"], p["base_width"], p["depth"]
    half = height / 2.0
    inside_z = np.abs(dz) <= half
    t = np.clip((dz + half) / height, 0.0, 1.0)
    half_width = (width / 2.0) * (1.0 - t)
    return inside_z & (np.abs(dx) <= half_width) & (np.abs(dy) <= depth / 2.0)


def _torus(dx, dy, dz, p):
    """Ring in the x-y plane; the hole runs along z."""
    major, minor = p["major_radius"], p["minor_radius"]
    radial = np.sqrt(dx**2 + dy**2)
    return ((radial - major) ** 2 + dz**2) <= minor**2


def _capsule(dx, dy, dz, p):
    """Cylinder segment along z with two hemispherical caps."""
    radius, half_segment = p["radius"], p["segment_length"] / 2.0
    # Distance to the axial segment: clamp dz into the straight part.
    dz_axis = dz - np.clip(dz, -half_segment, half_segment)
    return (dx**2 + dy**2 + dz_axis**2) <= radius**2


_VOXELIZERS: dict[str, Callable[..., np.ndarray]] = {
    "cube": _cube,
    "cuboid": _cuboid,
    "sphere": _sphere,
    "ellipsoid": _ellipsoid,
    "cylinder": _cylinder,
    "cone": _cone,
    "pyramid": _pyramid,
    "triangular_prism": _triangular_prism,
    "torus": _torus,
    "capsule": _capsule,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_params(shape_name: str, params: Mapping[str, float]) -> dict[str, float]:
    """Validate the parameter set of one primitive and return it as floats."""
    SHAPE_VOCABULARY.require_names([shape_name])
    expected = SHAPE_PARAMS[shape_name]
    if set(params) != set(expected):
        raise VoxelizationError(
            f"{shape_name} expects parameters {expected}, got {tuple(sorted(params))}"
        )
    values = {name: float(params[name]) for name in expected}
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise VoxelizationError(f"{shape_name}.{name} must be positive, got {value}")
    if shape_name == "torus" and values["minor_radius"] >= values["major_radius"]:
        raise VoxelizationError(
            "torus minor_radius must be smaller than major_radius "
            f"({values['minor_radius']} >= {values['major_radius']})"
        )
    return values


def half_extent_world(shape_name: str, params: Mapping[str, float]) -> tuple[float, float, float]:
    """Half of the analytic bounding box, in world units ``(x, y, z)``.

    Used to sample a centre that keeps the object inside the in-bounds margin.
    """
    p = check_params(shape_name, params)
    if shape_name == "cube":
        half = p["side"] / 2.0
        return (half, half, half)
    if shape_name == "cuboid":
        return (p["size_x"] / 2.0, p["size_y"] / 2.0, p["size_z"] / 2.0)
    if shape_name == "sphere":
        return (p["radius"],) * 3
    if shape_name == "ellipsoid":
        return (p["radius_x"], p["radius_y"], p["radius_z"])
    if shape_name in ("cylinder", "cone"):
        return (p["radius"], p["radius"], p["height"] / 2.0)
    if shape_name == "pyramid":
        return (p["base_side"] / 2.0, p["base_side"] / 2.0, p["height"] / 2.0)
    if shape_name == "triangular_prism":
        return (p["base_width"] / 2.0, p["depth"] / 2.0, p["height"] / 2.0)
    if shape_name == "torus":
        outer = p["major_radius"] + p["minor_radius"]
        return (outer, outer, p["minor_radius"])
    if shape_name == "capsule":
        return (p["radius"], p["radius"], p["segment_length"] / 2.0 + p["radius"])
    raise VoxelizationError(f"no half-extent rule for {shape_name!r}")  # pragma: no cover


def analytic_volume_world(shape_name: str, params: Mapping[str, float]) -> float:
    """Closed-form volume of the solid, for balancing checks and reporting."""
    p = check_params(shape_name, params)
    if shape_name == "cube":
        return p["side"] ** 3
    if shape_name == "cuboid":
        return p["size_x"] * p["size_y"] * p["size_z"]
    if shape_name == "sphere":
        return 4.0 / 3.0 * np.pi * p["radius"] ** 3
    if shape_name == "ellipsoid":
        return 4.0 / 3.0 * np.pi * p["radius_x"] * p["radius_y"] * p["radius_z"]
    if shape_name == "cylinder":
        return np.pi * p["radius"] ** 2 * p["height"]
    if shape_name == "cone":
        return np.pi * p["radius"] ** 2 * p["height"] / 3.0
    if shape_name == "pyramid":
        return p["base_side"] ** 2 * p["height"] / 3.0
    if shape_name == "triangular_prism":
        return p["base_width"] * p["height"] * p["depth"] / 2.0
    if shape_name == "torus":
        return 2.0 * np.pi**2 * p["major_radius"] * p["minor_radius"] ** 2
    if shape_name == "capsule":
        radius = p["radius"]
        return np.pi * radius**2 * p["segment_length"] + 4.0 / 3.0 * np.pi * radius**3
    raise VoxelizationError(f"no volume rule for {shape_name!r}")  # pragma: no cover


def voxelize(
    shape_name: str,
    params: Mapping[str, float],
    center_world: Sequence[float],
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Voxelise one primitive by name. Dispatches on the fixed vocabulary.

    Args:
        shape_name: a name from the ten-class vocabulary.
        params: the primitive's size parameters, in world units.
        center_world: centre of the analytic solid, ordered ``(x, y, z)``.
        volume_shape: ``(D, H, W)``.
        spacing: world units per voxel, ordered ``(x, y, z)``.

    Returns:
        A boolean ``(D, H, W)`` array indexed ``(z, y, x)``.
    """
    checked = check_params(shape_name, params)
    center = np.asarray(center_world, dtype=np.float64)
    if center.shape != (3,):
        raise VoxelizationError(f"center_world must be (x, y, z), got {center.shape}")

    z, y, x = world_coordinate_grids(volume_shape, spacing)
    mask = _VOXELIZERS[shape_name](x - center[0], y - center[1], z - center[2], checked)
    return np.broadcast_to(mask, tuple(int(v) for v in volume_shape)).copy()
