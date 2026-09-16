"""Counterfactual probes. Not implemented yet - planned with the Phase 4 report.

Mandatory set: permute anchor channels with the prompt fixed; permute prompt
clauses with the channels fixed; replace one direction with its opposite;
replace one anchor shape name; remove one anchor; reorder channels and prompt
clauses together (which must be a no-op).

The Stage B architecture is already *sensitive* to the first five - each changes
the prediction, which ``tests/test_stage_b_contract.py`` asserts - and the model
takes the permutations directly, since a counterfactual is just a reordered
``anchor_masks`` / ``direction_ids`` / ``anchor_shape_ids`` triple. What is
missing is the scored battery over a trained checkpoint.

A high Dice is insufficient: the model must be measurably sensitive to
direction-anchor correspondence and must beat the union-mask baseline here.
"""

from __future__ import annotations


def permute_anchor_channels(*args, **kwargs):  # pragma: no cover - Phase 4 report
    raise NotImplementedError("the scored counterfactual battery lands with the Phase 4 report")
