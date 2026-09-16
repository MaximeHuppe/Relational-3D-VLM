"""Correspondence probes with occupancy held fixed.

A high Dice on occupancy is not enough: the decoder can copy a remaining blob
without reading the prompt. These probes keep ``scene_volume`` unchanged and
only rewrite the WHERE inputs. If the binary mask barely moves, the model is
prompt-invariant.

The full Phase 4 battery (union-mask baseline, joint reorder as a no-op) is
still reserved for ``scripts/evaluate.py``. This module is the occupancy
sanity subset: permute channels with the prompt fixed, and replace one
direction with its opposite.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

from src.data.direction_rules import DIRECTIONS, opposite_direction

CHANNEL_PERMUTATION: tuple[int, ...] = (2, 0, 1)


def permute_anchor_channels(
    anchor_masks: Tensor, order: Sequence[int] = CHANNEL_PERMUTATION
) -> Tensor:
    """Reorder the three mask channels; the prompt and occupancy stay put."""
    permutation = tuple(int(index) for index in order)
    if sorted(permutation) != list(range(anchor_masks.shape[1])):
        raise ValueError(
            f"order must be a permutation of 0..{anchor_masks.shape[1] - 1}, got {permutation}"
        )
    return anchor_masks[:, list(permutation)]


def flip_one_direction(direction_ids: Tensor, slot: int = 0) -> Tensor:
    """Replace clause ``slot`` with its opposite token; occupancy stays put."""
    if slot < 0 or slot >= direction_ids.shape[-1]:
        raise ValueError(f"slot {slot} is out of range for {tuple(direction_ids.shape)}")
    flipped = direction_ids.clone()
    for batch_index in range(flipped.shape[0]):
        name = DIRECTIONS[int(flipped[batch_index, slot])]
        flipped[batch_index, slot] = DIRECTIONS.index(opposite_direction(name))
    return flipped
