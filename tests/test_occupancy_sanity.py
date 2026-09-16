"""Occupancy sanity: prompt use, train/val localisation, one object not the union."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.data.direction_rules import DIRECTIONS
from src.evaluation.counterfactuals import (
    CHANNEL_PERMUTATION,
    flip_one_direction,
    permute_anchor_channels,
)
from src.evaluation.occupancy_sanity import (
    HIGH_DICE,
    LOCALISATION_GAP,
    last_train_dice,
    label_components,
    localisation_verdict,
    occupancy_carving,
    run_occupancy_sanity,
)
from src.evaluation.qualitative import overlay_rgb, pick_slice_index, save_occupancy_slice, write_png


SIZE = 8
POSTERIOR = DIRECTIONS.index("posterior")


def _blob(volume: Tensor, z: int, y: int, x: int, width: int = 2) -> Tensor:
    volume[..., z : z + width, y : y + width, x : x + width] = 1
    return volume


def remaining_occupancy() -> tuple[Tensor, Tensor, Tensor]:
    """Three disjoint remaining objects plus three anchors. Target is object 0."""
    anchors = torch.zeros(1, 3, SIZE, SIZE, SIZE)
    _blob(anchors[:, 0], 0, 0, 0)   # x-band 0
    _blob(anchors[:, 1], 0, 6, 0)   # x-band 0
    _blob(anchors[:, 2], 0, 0, 6)   # x-band 6
    occupancy = torch.zeros(1, 1, SIZE, SIZE, SIZE)
    _blob(occupancy, 6, 0, 0)  # remaining object 0 / target
    _blob(occupancy, 6, 0, 6)  # remaining object 1
    _blob(occupancy, 6, 6, 6)  # remaining object 2
    scene = occupancy + anchors.amax(dim=1, keepdim=True)
    target = occupancy.clone()
    target[..., 6:8, 0:2, 6:8] = 0
    target[..., 6:8, 6:8, 6:8] = 0
    return scene, anchors, target


class _DummyVLM(nn.Module):
    """Shared occupancy helpers; subclasses choose what to paint."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def prepare_masks(self, anchor_masks: Tensor) -> Tensor:
        return anchor_masks.to(torch.float32)

    def prepare_occupancy(self, scene_volume: Tensor, masks: Tensor) -> Tensor:
        return scene_volume.to(torch.float32) * (1.0 - masks.amax(dim=1, keepdim=True))

    def _logits(self, mask: Tensor) -> Tensor:
        high = torch.full_like(mask, 10.0)
        low = torch.full_like(mask, -10.0)
        return torch.where(mask >= 0.5, high, low) + 0.0 * self.bias


class OccupancySnappingModel(_DummyVLM):
    """Ignores the prompt and paints every remaining occupancy voxel."""

    def forward(
        self,
        anchor_masks: Tensor,
        direction_ids: Tensor,
        anchor_shape_ids: Tensor,
        scene_volume: Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        occupancy = self.prepare_occupancy(scene_volume, self.prepare_masks(anchor_masks))
        return SimpleNamespace(logits=self._logits(occupancy))


class PromptSensitiveModel(_DummyVLM):
    """Moves with channel order and collapses when the first direction flips."""

    def forward(
        self,
        anchor_masks: Tensor,
        direction_ids: Tensor,
        anchor_shape_ids: Tensor,
        scene_volume: Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        occupancy = self.prepare_occupancy(scene_volume, self.prepare_masks(anchor_masks))
        collapsed = (direction_ids[:, 0] == POSTERIOR).view(-1, 1, 1, 1, 1)
        x_mass = anchor_masks[:, 0].sum(dim=(1, 2))
        x_peak = x_mass.argmax(dim=1).view(-1, 1, 1, 1, 1)
        grid_x = torch.arange(occupancy.shape[-1], device=occupancy.device).view(1, 1, 1, 1, -1)
        selected = occupancy * (grid_x // 4 == x_peak // 4).to(occupancy.dtype)
        logits = self._logits(selected)
        logits = torch.where(collapsed, torch.full_like(logits, -10.0), logits)
        return SimpleNamespace(logits=logits)


def _batch() -> dict[str, object]:
    scene, anchors, target = remaining_occupancy()
    return {
        "anchor_masks": anchors,
        "direction_ids": torch.tensor([[0, 2, 4]], dtype=torch.long),
        "anchor_shape_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "scene_volume": scene,
        "target_mask": target,
        "example_id": ["scene000_target_00"],
    }


# ---------------------------------------------------------------------------
# Counterfactuals hold occupancy fixed
# ---------------------------------------------------------------------------
def test_permuting_channels_does_not_change_the_occupancy_union():
    scene, anchors, _ = remaining_occupancy()
    occupancy = scene * (1.0 - anchors.amax(dim=1, keepdim=True))
    permuted = permute_anchor_channels(anchors)
    occupancy_after = scene * (1.0 - permuted.amax(dim=1, keepdim=True))
    assert torch.equal(occupancy, occupancy_after)
    assert tuple(permuted[0, :, 0, 0, 0].tolist()) != tuple(anchors[0, :, 0, 0, 0].tolist())
    assert CHANNEL_PERMUTATION == (2, 0, 1)


def test_flip_one_direction_replaces_the_slot_with_its_opposite():
    ids = torch.tensor([[0, 2, 4]], dtype=torch.long)  # anterior, superior, medial
    flipped = flip_one_direction(ids, slot=0)
    assert int(flipped[0, 0]) == POSTERIOR
    assert torch.equal(flipped[0, 1:], ids[0, 1:])
    with pytest.raises(ValueError):
        permute_anchor_channels(torch.zeros(1, 3, 2, 2, 2), order=(0, 0, 1))
    with pytest.raises(ValueError):
        flip_one_direction(ids, slot=3)


# ---------------------------------------------------------------------------
# Train vs val
# ---------------------------------------------------------------------------
def test_localisation_verdict_splits_where_from_occupancy():
    both_high = localisation_verdict(0.95, 0.91)
    assert both_high["verdict"] == "localisation_generalising"
    where = localisation_verdict(0.95, 0.50)
    assert where["verdict"] == "where_failure"
    assert where["gap"] == pytest.approx(0.45)
    assert localisation_verdict(0.40, 0.35)["verdict"] == "underfit"
    assert localisation_verdict(HIGH_DICE + 0.02, HIGH_DICE - 0.05)["verdict"] == "mixed"
    assert localisation_verdict(None, 0.9)["verdict"] == "skipped"
    assert LOCALISATION_GAP == 0.20


def test_last_train_dice_reads_history_json(tmp_path):
    assert last_train_dice(tmp_path) is None
    (tmp_path / "history.json").write_text('[{"train_dice": 0.12}, {"train_dice": 0.88}]\n')
    assert last_train_dice(tmp_path) == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# One remaining object, not the union
# ---------------------------------------------------------------------------
def test_label_components_counts_disjoint_blobs():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[3, 3, 3] = True
    labels, count = label_components(mask)
    assert count == 2
    assert labels[0, 0, 0] != labels[3, 3, 3]
    assert labels[0, 0, 0] != 0


def test_occupancy_carving_one_object_versus_the_union():
    scene, anchors, target = remaining_occupancy()
    occupancy = scene * (1.0 - anchors.amax(dim=1, keepdim=True))
    one = occupancy_carving(occupancy, target, target, from_logits=False)
    assert one["verdict"] == "one_object"
    assert one["passed"]
    assert one["occupancy_components"] == 3
    assert one["components_touched"] == 1

    union = occupancy_carving(occupancy, occupancy, target, from_logits=False)
    assert union["verdict"] == "union"
    assert not union["passed"]
    assert union["components_touched"] == 3

    empty = occupancy_carving(occupancy, torch.zeros_like(occupancy), target, from_logits=False)
    assert empty["verdict"] == "empty"
    outside = occupancy_carving(occupancy, anchors[:, 0:1], target, from_logits=False)
    assert outside["verdict"] == "outside_occupancy"


# ---------------------------------------------------------------------------
# Qualitative slice
# ---------------------------------------------------------------------------
def test_occupancy_slice_is_a_png_montage(tmp_path):
    occupancy = np.zeros((4, 4, 4), dtype=np.float32)
    occupancy[1] = 1.0
    prediction = np.zeros_like(occupancy)
    prediction[1, 0:2, 0:2] = 1.0
    target = np.zeros_like(occupancy)
    target[1, 0:2, 2:4] = 1.0
    assert pick_slice_index(occupancy, prediction, target) == 1
    rgb = overlay_rgb(occupancy[1], prediction[1], target[1])
    assert rgb.shape == (4, 4, 3)
    path = write_png(tmp_path / "gray.png", (occupancy[1] * 255).astype(np.uint8))
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    dump = save_occupancy_slice(tmp_path / "slice.png", occupancy, prediction, target)
    assert dump["slice_index"] == 1
    assert Path(dump["path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# End-to-end probes on dummy models
# ---------------------------------------------------------------------------
def test_snapping_to_occupancy_fails_prompt_and_union_checks(tmp_path):
    report = run_occupancy_sanity(
        OccupancySnappingModel().eval(),
        [_batch()],
        output_dir=tmp_path,
        train_dice=0.96,
        val_dice=0.94,
        num_slices=1,
    )
    assert not report["correspondence"]["permute_channels"]["passed"]
    assert not report["correspondence"]["flip_direction"]["passed"]
    assert report["correspondence"]["permute_channels"]["dice_vs_original"] == pytest.approx(1.0)
    assert report["carving"]["union_fraction"] == pytest.approx(1.0)
    assert not report["carving"]["passed"]
    assert report["localisation"]["verdict"] == "localisation_generalising"
    assert not report["passed"]
    assert (tmp_path / "occupancy_sanity.json").is_file()
    assert (tmp_path / "occupancy_sanity" / "slices" / "scene000_target_00.png").is_file()
    assert "FAIL" in report["table"]


def test_prompt_sensitive_dummy_moves_or_collapses_with_occupancy_fixed(tmp_path):
    report = run_occupancy_sanity(
        PromptSensitiveModel().eval(),
        [_batch()],
        output_dir=tmp_path,
        train_dice=0.9,
        val_dice=0.4,
        num_slices=1,
    )
    assert report["correspondence"]["permute_channels"]["passed"]
    assert report["correspondence"]["flip_direction"]["passed"]
    assert report["correspondence"]["flip_direction"]["dice_vs_original"] < 0.5
    assert report["localisation"]["verdict"] == "where_failure"
    assert not report["passed"]
    assert "where_failure" in report["table"]
