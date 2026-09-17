"""RAS NIfTI read/write for this project's ``(z, y, x)`` arrays.

In-memory volumes are ``(D, H, W) == (z, y, x)``. nibabel and Slicer store
``(x, y, z)`` with an affine that maps voxel indices to millimetres. Every
write transposes ``(2, 1, 0)``; every read transposes back.

World convention matches the rest of the repo: voxel index ``(x, y, z)`` maps
to ``(x * sx, y * sy, z * sz)``, origin at index zero. The affine is therefore
``diag(sx, sy, sz, 1)`` with ``sform``/``qform`` code 1 so viewers do not guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image

_XYZ_FROM_ZYX = (2, 1, 0)
_ZYX_FROM_XYZ = (2, 1, 0)


class NiftiIOError(ValueError):
    """Raised when a NIfTI volume cannot be written or read as this project expects."""


def ras_affine(spacing: Sequence[float]) -> np.ndarray:
    """4x4 RAS affine for spacing ``(sx, sy, sz)`` with origin at voxel (0, 0, 0)."""
    values = tuple(float(v) for v in spacing)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise NiftiIOError(f"spacing must be three positive (x, y, z) values, got {spacing!r}")
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = values[0]
    affine[1, 1] = values[1]
    affine[2, 2] = values[2]
    return affine


def to_xyz(volume: np.ndarray) -> np.ndarray:
    """``(z, y, x)`` -> ``(x, y, z)``."""
    if volume.ndim != 3:
        raise NiftiIOError(f"expected a 3D volume, got shape {tuple(volume.shape)}")
    return np.transpose(volume, _XYZ_FROM_ZYX)


def to_zyx(volume: np.ndarray) -> np.ndarray:
    """``(x, y, z)`` -> ``(z, y, x)``."""
    if volume.ndim != 3:
        raise NiftiIOError(f"expected a 3D volume, got shape {tuple(volume.shape)}")
    return np.transpose(volume, _ZYX_FROM_XYZ)


def save_nifti(
    path: Path | str,
    volume: np.ndarray,
    spacing: Sequence[float],
    *,
    dtype: np.dtype | type | None = None,
) -> Path:
    """Write ``volume`` ``(z, y, x)`` as a RAS ``.nii.gz`` (or ``.nii``)."""
    path = Path(path)
    if path.suffix not in {".nii", ".gz"} and not path.name.endswith(".nii.gz"):
        raise NiftiIOError(f"NIfTI path must end in .nii or .nii.gz, got {path}")
    array = np.ascontiguousarray(volume)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Nifti1Image(to_xyz(array), ras_affine(spacing))
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    nib.save(image, path)
    return path


def load_nifti(path: Path | str, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Read a RAS NIfTI and return a ``(z, y, x)`` array.

    Integer dtypes are loaded from the on-disk payload (no float round-trip).
    Float dtypes go through ``get_fdata``.
    """
    path = Path(path)
    if not path.is_file():
        raise NiftiIOError(f"missing NIfTI volume {path}")
    image = nib.load(str(path))
    if image.ndim != 3:
        raise NiftiIOError(f"{path} has {image.ndim} axes; expected a 3D volume")
    wanted = np.dtype(dtype) if dtype is not None else np.dtype(image.get_data_dtype())
    if np.issubdtype(wanted, np.integer):
        payload = np.asanyarray(image.dataobj)
        volume = to_zyx(np.rint(payload).astype(wanted, copy=False))
    else:
        volume = to_zyx(np.ascontiguousarray(image.get_fdata(dtype=wanted)))
    return np.ascontiguousarray(volume)


def voxel_world_xyz(index_zyx: Sequence[int], spacing: Sequence[float]) -> tuple[float, float, float]:
    """World ``(x, y, z)`` of the voxel at array index ``(z, y, x)``."""
    z, y, x = (int(v) for v in index_zyx)
    sx, sy, sz = (float(v) for v in spacing)
    return (x * sx, y * sy, z * sz)
