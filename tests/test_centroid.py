"""Centroid localisation: the metric, the auxiliary loss, and the model head.

Why this exists. Dice conflates two independent failures - *looking in the wrong
place* and *looking in the right place at the wrong size* - and Stage B's
measured 0.22 turned out to be almost entirely the second. Across four
checkpoints spanning three objectives, the predicted/GT volume ratio moved
2.05x -> 1.31x while the centroid error did not move at all (7.72 -> 7.86 vox,
against 18.4 for an anchor-union predictor and 26.6 for the volume centre).
Nothing but this metric can see that, and it is what justifies the loss term and
the head.

The three pieces share one definition of "centroid of a weight map"
(:func:`src.evaluation.metrics.soft_centroid`), so the number reported, the
number optimised and the number the head predicts cannot drift apart. The first
test pins that definition against the project's own independent implementation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.data.direction_rules import centroid_world
from src.evaluation.metrics import (
    MetricAccumulator,
    StratifiedMetrics,
    centroid_baselines,
    centroid_distance,
    format_stratified_table,
    soft_centroid,
    volume_diagonal,
)
from src.models.relational_vlm import RelationalVLM, RelationalVLMConfig, build_relational_vlm
from src.training.losses import (
    centroid_head_loss,
    centroid_loss,
    segmentation_loss,
)
from src.training.trainer import StageBTrainer, TrainingSettings

SMALL = RelationalVLMConfig(
    encoder_channels=(4, 8, 16, 32), decoder_channels=(16, 8, 4),
    input_resolution=64, bottleneck_resolution=8, token_dim=16, embedding_dim=16,
    num_heads=2, intersection_hidden_channels=16,
)


def blob(shape=(16, 16, 16), at=(4, 4, 4), side=2) -> torch.Tensor:
    mask = torch.zeros(1, 1, *shape)
    z, y, x = at
    mask[0, 0, z:z + side, y:y + side, x:x + side] = 1.0
    return mask


def sharp_logits(mask: torch.Tensor, magnitude: float = 20.0) -> torch.Tensor:
    """Logits a confident model would emit for ``mask``."""
    return mask * 2 * magnitude - magnitude


# ---------------------------------------------------------------------------
# The shared definition
# ---------------------------------------------------------------------------
def test_soft_centroid_matches_the_projects_own_centroid():
    """The one place the (z, y, x) array order and (x, y, z) spacing can be swapped.

    A cubic isotropic grid hides that mistake; this one does not.
    """
    mask = torch.zeros(1, 1, 6, 5, 4)
    mask[0, 0, 1:4, 0:2, 2:4] = 1.0
    spacing = (1.0, 2.0, 3.0)  # (x, y, z)

    measured = soft_centroid(mask, spacing=spacing)[0, 0].numpy()
    reference = centroid_world(mask[0, 0].numpy() > 0.5, spacing)  # (x, y, z)
    assert np.allclose(measured, reference[::-1], atol=1e-5)


def test_soft_centroid_is_mass_weighted_and_differentiable():
    weights = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    with torch.no_grad():
        weights[0, 0, 0, 0, 0] = 3.0
        weights[0, 0, 2, 0, 0] = 1.0
    centroid = soft_centroid(weights)[0, 0]
    # 3 parts at z=0, 1 part at z=2 -> 0.5
    assert float(centroid[0].detach()) == pytest.approx(0.5, abs=1e-5)
    centroid.sum().backward()
    assert torch.isfinite(weights.grad).all()


def test_volume_diagonal():
    assert volume_diagonal((64, 64, 64)) == pytest.approx(63 * 3 ** 0.5)
    assert volume_diagonal((2, 2, 2), (1.0, 2.0, 3.0)) == pytest.approx(
        (1 + 4 + 9) ** 0.5
    )


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
def test_centroid_distance_is_zero_for_a_perfect_prediction_and_grows_with_offset():
    target = blob(at=(4, 4, 4))
    assert centroid_distance(target, target, soft=False)[0] == pytest.approx(0.0)
    for offset in (1, 3, 6):
        moved = blob(at=(4, 4, 4 + offset))
        assert centroid_distance(moved, target, soft=False)[0] == pytest.approx(
            float(offset), abs=1e-4
        )


def test_centroid_distance_is_spacing_aware():
    target, moved = blob(at=(4, 4, 4)), blob(at=(4, 4, 6))
    # two voxels along x, at 3 world units per voxel
    assert centroid_distance(moved, target, soft=False, spacing=(3.0, 1.0, 1.0))[0] == (
        pytest.approx(6.0, abs=1e-4)
    )


def test_soft_and_hard_agree_for_a_confident_prediction():
    """They agreed to 0.01 vox on the real checkpoints; that is what says the
    prediction is one coherent blob rather than scattered mass."""
    target = blob(at=(4, 4, 4))
    logits = sharp_logits(blob(at=(6, 4, 4)))
    soft = centroid_distance(logits, target, from_logits=True, soft=True)[0]
    hard = centroid_distance(logits, target, from_logits=True, soft=False)[0]
    assert soft == pytest.approx(hard, abs=1e-3)
    assert soft == pytest.approx(2.0, abs=1e-3)


def test_hard_centroid_is_undefined_for_an_empty_prediction_but_soft_is_not():
    target = blob(at=(4, 4, 4))
    empty = torch.full_like(target, -20.0)
    assert np.isnan(centroid_distance(empty, target, from_logits=True, soft=False)[0])
    assert np.isfinite(centroid_distance(empty, target, from_logits=True, soft=True)[0])


def test_centroid_baselines_are_the_two_trivial_predictors():
    target = blob(shape=(8, 8, 8), at=(0, 0, 0), side=1)   # centroid (0, 0, 0)
    anchors = blob(shape=(8, 8, 8), at=(4, 0, 0), side=1)  # centroid (4, 0, 0)
    baselines = centroid_baselines(target, anchor_union=anchors)
    assert baselines["volume_centre"][0] == pytest.approx(3.5 * 3 ** 0.5, abs=1e-4)
    assert baselines["anchor_union"][0] == pytest.approx(4.0, abs=1e-4)
    # Without an anchor union only the centre baseline is reported.
    assert set(centroid_baselines(target)) == {"volume_centre"}


def test_accumulator_reports_centroid_and_drops_undefined_samples():
    accumulator = MetricAccumulator()
    accumulator.add(1.0, 1.0, centroid=2.0)
    accumulator.add(1.0, 1.0, centroid=4.0)
    accumulator.add(1.0, 1.0, centroid=float("nan"))
    summary = accumulator.summary()
    assert summary["centroid"] == pytest.approx(3.0)
    assert summary["centroid_undefined"] == 1.0
    # A run that never supplies one does not grow the key.
    bare = MetricAccumulator()
    bare.add(1.0, 1.0)
    assert "centroid" not in bare.summary()


def test_stratified_metrics_break_centroid_down_and_report_the_baselines():
    target = torch.cat([blob(at=(4, 4, 4)), blob(at=(8, 8, 8))])
    logits = sharp_logits(torch.cat([blob(at=(4, 4, 6)), blob(at=(8, 8, 8))]))
    metrics = StratifiedMetrics()
    metrics.update(
        logits, target, from_logits=True,
        directions=[["medial", "superior", "anterior"], ["lateral", "inferior", "posterior"]],
        target_shapes=["cube", "sphere"],
        anchor_union=torch.cat([blob(at=(0, 0, 0)), blob(at=(0, 0, 0))]),
    )
    summary = metrics.summary()
    assert summary["overall"]["centroid"] == pytest.approx(1.0, abs=1e-3)
    # sample 0 is 2 vox off, sample 1 is exact
    assert summary["strata"]["direction"]["medial"]["centroid"] == pytest.approx(2.0, abs=1e-3)
    assert summary["strata"]["direction"]["lateral"]["centroid"] == pytest.approx(0.0, abs=1e-3)
    assert set(summary["centroid_baselines"]) == {"anchor_union", "volume_centre"}

    table = format_stratified_table(summary)
    assert "cdist" in table and "cdist baselines" in table


def test_the_metric_can_be_switched_off():
    target = blob()
    metrics = StratifiedMetrics()
    metrics.update(sharp_logits(target), target, from_logits=True, with_centroid=False)
    assert "centroid" not in metrics.summary()["overall"]


# ---------------------------------------------------------------------------
# The auxiliary loss
# ---------------------------------------------------------------------------
def test_centroid_loss_vanishes_for_a_sharp_perfect_prediction():
    """The background-mass caveat, pinned: the floor decays as the model sharpens."""
    target = blob(shape=(64, 64, 64), at=(20, 20, 20), side=8)
    losses = [float(centroid_loss(sharp_logits(target, m), target)) for m in (6, 12, 20)]
    assert losses[0] > losses[1] > losses[2]
    assert losses[-1] == pytest.approx(0.0, abs=1e-6)


def test_centroid_loss_grows_with_displacement_and_is_diagonal_normalised():
    target = blob(shape=(64, 64, 64), at=(20, 20, 20), side=8)
    near = float(centroid_loss(sharp_logits(blob((64, 64, 64), (20, 20, 24), 8)), target))
    far = float(centroid_loss(sharp_logits(blob((64, 64, 64), (20, 20, 40), 8)), target))
    assert 0 < near < far
    # 4 voxels of offset, normalised by the 64^3 diagonal
    assert near == pytest.approx(4.0 / volume_diagonal((64, 64, 64)), rel=1e-2)


def test_centroid_loss_pulls_probability_toward_the_target():
    """The whole point: a usable direction even where soft Dice has none.

    Raising ``p`` at the target's location must lower the loss, and raising it on
    the far side must raise it - that is the signal Dice cannot give when the
    prediction and the target do not overlap at all.
    """
    target = blob(shape=(32, 32, 32), at=(24, 16, 16), side=4)
    # Deliberately unsaturated logits: at +/-20 the sigmoid derivative p(1 - p)
    # underflows to zero in float32 and would hide the term's own direction. The
    # signal below is present at every magnitude; only its scale changes.
    logits = sharp_logits(blob((32, 32, 32), (4, 16, 16), 4), magnitude=4.0)
    logits = logits.clone().requires_grad_(True)
    centroid_loss(logits, target).backward()
    grad = logits.grad[0, 0]

    assert float(grad[25, 17, 17]) < 0, "more mass on the target must reduce the loss"
    assert float(grad[5, 17, 17]) > 0, "more mass on the far side must increase it"
    # The prediction and the target do not overlap at all, which is precisely
    # the regime soft Dice cannot give a direction in and this term exists for.
    predicted = torch.sigmoid(logits.detach()) > 0.5
    assert int((predicted & (target > 0.5)).sum()) == 0


def test_centroid_loss_is_silenced_rather_than_nan_for_a_dead_prediction():
    target = blob(shape=(32, 32, 32), at=(8, 8, 8), side=4)
    logits = torch.full((1, 1, 32, 32, 32), -30.0, requires_grad=True)
    loss = centroid_loss(logits, target)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(logits.grad).all()


def test_centroid_loss_alone_is_degenerate():
    """Documented, and pinned so nobody promotes it to the primary term.

    A centroid is invariant to scale and to shape: a single correctly-placed
    voxel and a correctly-centred blob a hundred times too large score the same.
    Dice has to stay dominant.
    """
    target = blob(shape=(32, 32, 32), at=(14, 14, 14), side=4)
    one_voxel = blob(shape=(32, 32, 32), at=(15, 15, 15), side=2)
    huge = blob(shape=(32, 32, 32), at=(8, 8, 8), side=16)
    assert float(centroid_loss(sharp_logits(one_voxel), target)) == pytest.approx(
        float(centroid_loss(sharp_logits(huge), target)), abs=1e-5
    )
    # ... while Dice tells them apart decisively.
    from src.training.losses import dice_loss
    assert float(dice_loss(sharp_logits(one_voxel), target)) != pytest.approx(
        float(dice_loss(sharp_logits(huge), target)), abs=0.1
    )


def test_segmentation_loss_reports_centroid_always_and_adds_it_only_when_weighted():
    target = blob(shape=(32, 32, 32), at=(8, 8, 8), side=4)
    logits = sharp_logits(blob((32, 32, 32), (16, 8, 8), 4))
    off, parts = segmentation_loss(logits, target)
    on, _ = segmentation_loss(logits, target, lambda_centroid=3.0)
    assert "centroid" in parts and parts["centroid"] > 0
    assert float(on - off) == pytest.approx(3.0 * parts["centroid"], rel=1e-5)


# ---------------------------------------------------------------------------
# The model head
# ---------------------------------------------------------------------------
def test_the_head_is_off_by_default_and_cheap_when_on():
    assert build_relational_vlm().centroid_head is None
    assert build_relational_vlm(centroid_head=True).centroid_head is not None

    without = RelationalVLM(SMALL)
    with_head = RelationalVLM(dataclasses.replace(SMALL, centroid_head=True))
    added = sum(p.numel() for p in with_head.parameters()) - sum(
        p.numel() for p in without.parameters()
    )
    assert added == SMALL.bottleneck_channels + 1
    # Parameter-identical by default, so existing checkpoints still load.
    assert set(without.state_dict()) < set(with_head.state_dict())


def test_the_soft_argmax_maps_bottleneck_cells_to_full_resolution_coordinates():
    """Cell ``i`` covers ``scale`` voxels, so its centre is ``i * scale + (scale - 1) / 2``."""
    model = RelationalVLM(dataclasses.replace(SMALL, centroid_head=True)).eval()
    channels = SMALL.bottleneck_channels
    for cell, expected in (((0, 0, 0), (3.5, 3.5, 3.5)), ((7, 7, 7), (59.5, 59.5, 59.5)),
                           ((0, 0, 7), (3.5, 3.5, 59.5))):
        features = torch.zeros(1, channels, 8, 8, 8)
        with torch.no_grad():
            # Drive one cell through the head's own weights.
            model.centroid_head.weight.zero_()
            model.centroid_head.bias.zero_()
            model.centroid_head.weight[0, 0] = 1.0
            features[0, 0, cell[0], cell[1], cell[2]] = 100.0
            predicted = model.predict_centroid(features, (64, 64, 64))[0]
        assert predicted.tolist() == pytest.approx(list(expected), abs=1e-3)


def test_the_head_produces_a_finite_differentiable_centroid_in_the_forward_pass():
    model = RelationalVLM(dataclasses.replace(SMALL, centroid_head=True))
    masks = torch.zeros(2, 3, 64, 64, 64)
    masks[:, :, 8:12, 8:12, 8:12] = 1.0
    output = model(masks, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    assert output.centroid is not None and output.centroid.shape == (2, 3)
    assert torch.isfinite(output.centroid).all()
    # Inside the volume it is reading from.
    assert (output.centroid >= 0).all() and (output.centroid <= 63).all()
    output.centroid.sum().backward()
    assert torch.isfinite(model.centroid_head.weight.grad).all()

    plain = RelationalVLM(SMALL)
    assert plain(masks, torch.zeros(2, 3, dtype=torch.long),
                 torch.zeros(2, 3, dtype=torch.long)).centroid is None


def test_centroid_head_loss_is_zero_on_target_and_grows_with_error():
    target = blob(shape=(32, 32, 32), at=(8, 8, 8), side=4)
    exact = soft_centroid(target)[:, 0]
    assert float(centroid_head_loss(exact, target)) == pytest.approx(0.0, abs=1e-9)
    near = float(centroid_head_loss(exact + 1.0, target))
    far = float(centroid_head_loss(exact + 6.0, target))
    assert 0 < near < far
    with pytest.raises(ValueError, match=r"\[B, 3\]"):
        centroid_head_loss(exact[:, :2], target)


def test_the_trainer_reports_the_head_and_weights_it_only_when_asked():
    """`_loss` takes the whole output so the head can be a probe at weight 0."""
    from types import SimpleNamespace

    target = blob(shape=(16, 16, 16), at=(4, 4, 4), side=2)
    batch = {"target_mask": target}
    output = SimpleNamespace(
        logits=sharp_logits(blob(shape=(16, 16, 16), at=(8, 4, 4), side=2)),
        centroid=soft_centroid(target)[:, 0] + 3.0,
    )

    probe = SimpleNamespace(
        settings=TrainingSettings(lambda_centroid_head=0.0), spacing=(1.0, 1.0, 1.0)
    )
    total, parts = StageBTrainer._loss(probe, output, batch)
    assert "centroid_head" in parts and parts["centroid_head"] > 0

    weighted = SimpleNamespace(
        settings=TrainingSettings(lambda_centroid_head=2.0), spacing=(1.0, 1.0, 1.0)
    )
    heavier, _ = StageBTrainer._loss(weighted, output, batch)
    assert float(heavier - total) == pytest.approx(2.0 * parts["centroid_head"], rel=1e-5)

    # No head on the model -> nothing reported, nothing added.
    headless = SimpleNamespace(logits=output.logits, centroid=None)
    plain_total, plain_parts = StageBTrainer._loss(probe, headless, batch)
    assert "centroid_head" not in plain_parts
    assert float(plain_total) == pytest.approx(float(total))


def test_settings_validate_the_centroid_weights():
    assert TrainingSettings().lambda_centroid == 0.0
    assert TrainingSettings().lambda_centroid_head == 0.0
    with pytest.raises(ValueError, match="lambda_centroid must be"):
        TrainingSettings(lambda_centroid=-1.0)
    with pytest.raises(ValueError, match="lambda_centroid_head must be"):
        TrainingSettings(lambda_centroid_head=-1.0)
