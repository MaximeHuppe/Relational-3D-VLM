"""Relation-preserving augmentations. Not implemented yet.

Any flip, translation, crop or axis permutation must rewrite, consistently:
scene and instance masks, the three anchor channels, world coordinates,
centroids and extents, direction labels and prompt clauses.

An augmentation stays disabled in ``configs/train.yaml`` until its relation
rewrite has a passing unit test (``tests/test_augmentations.py``). Arbitrary
rotations are out of scope for this milestone.
"""

from __future__ import annotations


def rewrite_relations(*args, **kwargs):  # pragma: no cover - post Phase 3
    raise NotImplementedError("augmentation relation rewrites are implemented later")
