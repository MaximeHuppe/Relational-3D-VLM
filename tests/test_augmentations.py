"""Augmentation relation-rewrite tests. Added when augmentations are enabled.

No augmentation may be turned on in ``configs/train.yaml`` before its rewrite is
tested here: a flip, translation, crop or axis permutation must consistently
rewrite the scene and instance masks, the three anchor channels, world
coordinates, centroids and extents, the direction labels and the prompt clauses.
"""

from __future__ import annotations

import pytest

from src.config import load_config


def test_no_augmentation_is_enabled_before_its_rewrite_is_tested():
    """This guard runs today: config must not enable an untested rewrite."""
    augmentations = load_config("train")["augmentations"]
    for name, settings in augmentations.items():
        if not isinstance(settings, dict):
            continue
        if settings.get("enabled"):
            assert settings.get("rewrite_tested") is True, (
                f"augmentation {name!r} is enabled but its relation rewrite is untested"
            )
    assert augmentations["arbitrary_rotation"]["allowed_in_milestone"] is False


@pytest.mark.skip(reason="relation rewrites are implemented after Phase 3")
def test_flip_rewrites_directions_and_prompt():  # pragma: no cover - placeholder
    ...
