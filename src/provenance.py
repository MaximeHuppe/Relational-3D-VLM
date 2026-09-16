"""Run provenance: versions, git revision and hardware.

Every artefact this project writes - generated corpora, checkpoints, evaluation
reports - carries the same provenance block, so any result can be traced back to
the exact code and data that produced it.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

_UNKNOWN = None


def git_revision(root: Path | str = PROJECT_ROOT) -> str | None:
    """Current ``HEAD``, or ``None`` outside a repository / before the first commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return _UNKNOWN
    return result.stdout.strip() or _UNKNOWN


def git_is_dirty(root: Path | str = PROJECT_ROOT) -> bool | None:
    """True when the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return _UNKNOWN
    return bool(result.stdout.strip())


def version_metadata() -> dict[str, str]:
    """The four contract versions that must agree between code, data and runs."""
    from src.config import load_config
    from src.data.direction_rules import DIRECTION_RULE_VERSION
    from src.data.primitives import VOCABULARY_VERSION
    from src.data.schema import SCHEMA_VERSION

    return {
        "generator_version": str(load_config("generator")["generator_version"]),
        "direction_rule_version": DIRECTION_RULE_VERSION,
        "vocabulary_version": VOCABULARY_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def hardware_metadata() -> dict[str, Any]:
    """Platform and accelerator description, recorded on every run."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(torch.backends.mps.is_available())
        if torch.cuda.is_available():  # pragma: no cover - no CUDA here
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def environment_metadata() -> dict[str, Any]:
    """Git revision plus hardware, the block embedded in every artefact."""
    return {
        "git_revision": git_revision(),
        "git_dirty": git_is_dirty(),
        **hardware_metadata(),
    }
