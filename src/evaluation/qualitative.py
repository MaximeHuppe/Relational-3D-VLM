"""Qualitative 3D prediction dumps. Not implemented yet - Phase 4 report.

Stage B already returns everything these need: probabilities, a thresholded
binary mask and, with ``return_evidence=True``, the three 8^3 relation-specific
evidence maps - which are the interesting picture, because they show *where each
clause votes* before the intersection.
"""

from __future__ import annotations


def save_qualitative(*args, **kwargs):  # pragma: no cover - Phase 4 report
    raise NotImplementedError("qualitative outputs land with the Phase 4 report")
