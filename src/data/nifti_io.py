"""NIfTI (``.nii.gz``) reading and writing for generated scenes.

Everything in this repository indexes arrays ``(z, y, x)`` and orders world
coordinates ``(x, y, z)`` in a RAS frame. NIfTI stores the data the other way
round - ``data[i, j, k]`` with an affine that maps ``(i, j, k)`` to RAS
millimetres - so every array is transposed on the way out and on the way in.
Doing it in one place is the point of this module: a viewer (ITK-SNAP, FSLeyes,
3D Slicer) then shows the same left/right, anterior/posterior and
superior/inferior as :mod:`src.data.direction_rules` reasons about.

The affine is a plain diagonal ``diag(sx, sy, sz, 1)`` with no translation, so
a voxel's NIfTI world coordinate is exactly the world coordinate this codebase
computes for it (``x = i * sx`` and so on) and a centroid printed by a manifest
can be typed straight into a viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - exercised by the import error path only
    import nibabel as nib
except ImportError as error:  # pragma: no cover
    nib = None  # type: ignore[assignment]
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

#: Suffix every volume this module writes carries.
NIFTI_SUFFIX = ".nii.gz"


class NiftiError(RuntimeError):
    """Raised when NIfTI I/O is requested but unavailable or inconsistent."""


def require_nibabel() -> Any:
    """Return the ``nibabel`` module, with an actionable error when it is absent."""
    if nib is None:  # pragma: no cover - depends on the environment
        raise NiftiError(
            "nibabel is required to read or write .nii.gz scenes; install it with "
            "`pip install nibabel` (it is in requirements.txt), or set "
            "`storage.scene_array_format: npz_compressed` in configs/generator.yaml"
        ) from _IMPORT_ERROR
    return nib


def scene_affine(spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """RAS affine for a volume with the given ``(x, y, z)`` spacing."""
    sx, sy, sz = (float(v) for v in spacing)
    if min(sx, sy, sz) <= 0:
        raise NiftiError(f"spacing must be strictly positive, got {spacing!r}")
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = sx, sy, sz
    return affine


def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """The ``(x, y, z)`` voxel spacing encoded in an affine."""
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    return tuple(float(np.linalg.norm(matrix[:, axis])) for axis in range(3))  # type: ignore[return-value]


def to_nifti_order(array: np.ndarray) -> np.ndarray:
    """``(..., z, y, x)`` to NIfTI ``(x, y, z, ...)``.

    A leading channel axis (for example the three ordered anchor masks) becomes
    the fourth NIfTI dimension, which is how a viewer shows a volume series.
    """
    values = np.asarray(array)
    if values.ndim == 3:
        return np.ascontiguousarray(np.transpose(values, (2, 1, 0)))
    if values.ndim == 4:
        return np.ascontiguousarray(np.transpose(values, (3, 2, 1, 0)))
    raise NiftiError(f"expected a 3D (z, y, x) or 4D (c, z, y, x) array, got {values.shape}")


def from_nifti_order(array: np.ndarray) -> np.ndarray:
    """NIfTI ``(x, y, z, ...)`` back to ``(..., z, y, x)``."""
    values = np.asarray(array)
    if values.ndim == 3:
        return np.ascontiguousarray(np.transpose(values, (2, 1, 0)))
    if values.ndim == 4:
        return np.ascontiguousarray(np.transpose(values, (3, 2, 1, 0)))
    raise NiftiError(f"expected a 3D or 4D NIfTI array, got {values.shape}")


def save_nifti(
    path: Path | str,
    array: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    dtype: np.dtype | str | None = None,
    description: str = "",
) -> Path:
    """Write ``(..., z, y, x)`` data as a gzipped NIfTI-1 volume.

    Args:
        path: destination; ``.nii.gz`` is appended when missing.
        array: 3D ``(z, y, x)`` or 4D ``(c, z, y, x)`` data.
        spacing: world units per voxel, ordered ``(x, y, z)``.
        dtype: on-disk data type. Integer types make nibabel store a scale
            factor, which is how real scanner data is written; the default
            keeps the array's own type.
        description: free text placed in the header's ``descrip`` field.
    """
    module = require_nibabel()
    path = Path(path)
    if not str(path).endswith(NIFTI_SUFFIX):
        path = path.with_name(path.name + NIFTI_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = to_nifti_order(array)
    image = module.Nifti1Image(data, scene_affine(spacing))
    if dtype is not None:
        image.set_data_dtype(np.dtype(dtype))
    header = image.header
    header.set_xyzt_units("mm")
    if description:
        header["descrip"] = description.encode("utf-8")[:79]
    # sform and qform both aligned-to-anatomy, so a viewer uses the affine
    # above instead of guessing an orientation.
    image.set_sform(scene_affine(spacing), code=2)
    image.set_qform(scene_affine(spacing), code=2)
    module.save(image, str(path))
    return path


def load_nifti(
    path: Path | str, *, dtype: np.dtype | str | None = None
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a NIfTI volume back as ``(..., z, y, x)`` plus its ``(x, y, z)`` spacing."""
    module = require_nibabel()
    path = Path(path)
    if not path.is_file():
        raise NiftiError(f"missing NIfTI volume {path}")
    image = module.load(str(path))
    data = np.asanyarray(image.dataobj)
    if dtype is not None:
        data = data.astype(np.dtype(dtype))
    return from_nifti_order(data), spacing_from_affine(image.affine)
