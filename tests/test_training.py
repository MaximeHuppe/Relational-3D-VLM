"""Losses, metrics, the Stage A dataset and the training loop.

The loop test trains the tiny model for a handful of steps on a two-scene smoke
dataset and asserts that the loss falls and Dice rises - enough to catch a
broken optimiser, a detached graph or a target/prediction misalignment, without
depending on any particular accuracy.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.data.dataset import DatasetError, SceneDataset, build_dataloader
from src.data.primitives import SHAPE_NAMES
from src.evaluation.metrics import (
    PerClassMetrics,
    batch_hausdorff,
    dice_score,
    format_per_class_table,
    hausdorff_distance,
    iou_score,
    surface_voxels,
)
from src.models.shape_segmenter import ShapeSegmenter
from src.training.losses import (
    bce_loss,
    deep_supervision_loss,
    dice_loss,
    downsample_targets,
    segmentation_loss,
)
from src.training.trainer import (
    StageATrainer,
    TrainingSettings,
    build_scheduler,
    resolve_device,
    seed_everything,
)
from tests.test_model_contract import TINY

SMOKE_ROOT = "data/smoke"


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_dice_loss_is_zero_for_a_perfect_prediction():
    target = torch.zeros(2, 3, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1
    logits = (target * 2 - 1) * 50.0  # saturated, correct
    assert float(dice_loss(logits, target)) < 1e-3


def test_dice_loss_is_near_one_for_an_inverted_prediction():
    target = torch.zeros(1, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1
    assert float(dice_loss((1 - target) * 100 - 50, target)) > 0.99


def test_losses_reject_shape_mismatches():
    with pytest.raises(ValueError):
        dice_loss(torch.zeros(1, 2, 4, 4, 4), torch.zeros(1, 3, 4, 4, 4))


def test_segmentation_loss_reports_its_components():
    target = torch.zeros(1, 2, 4, 4, 4)
    target[:, :, 1:3] = 1
    total, parts = segmentation_loss(torch.zeros_like(target), target, lambda_dice=1.0, lambda_bce=1.0)
    assert set(parts) == {"dice", "bce"}
    assert pytest.approx(float(total), rel=1e-5) == parts["dice"] + parts["bce"]
    assert pytest.approx(parts["bce"], abs=1e-4) == float(bce_loss(torch.zeros_like(target), target))


def test_deep_supervision_weights_each_scale():
    target = torch.zeros(1, 2, 16, 16, 16)
    target[:, :, 4:8, 4:8, 4:8] = 1
    predictions = [
        torch.zeros(1, 2, 4, 4, 4),
        torch.zeros(1, 2, 8, 8, 8),
        torch.zeros(1, 2, 16, 16, 16),
    ]
    total, parts = deep_supervision_loss(predictions, target, [0.1, 0.3, 0.6])
    assert {"dice@4", "bce@4", "dice@8", "bce@8", "dice@16", "bce@16", "total"} == set(parts)
    expected = sum(
        weight * (parts[f"dice@{resolution}"] + parts[f"bce@{resolution}"])
        for weight, resolution in zip([0.1, 0.3, 0.6], [4, 8, 16])
    )
    assert pytest.approx(float(total), rel=1e-5) == expected
    with pytest.raises(ValueError):
        deep_supervision_loss(predictions, target, [0.5, 0.5])


def test_target_downsampling_preserves_thin_structures():
    """A one-voxel-thick sheet must survive to the coarsest supervision scale."""
    target = torch.zeros(1, 1, 16, 16, 16)
    target[:, :, 7, :, :] = 1
    for size in ((8, 8, 8), (4, 4, 4)):
        coarse = downsample_targets(target, size)
        assert tuple(coarse.shape[2:]) == size
        assert coarse.sum() > 0
    assert torch.equal(downsample_targets(target, (16, 16, 16)), target)
    with pytest.raises(ValueError):
        downsample_targets(target, (5, 5, 5))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_dice_and_iou_on_a_known_overlap():
    prediction = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    prediction[:, :, 0, 0, :2] = 1   # 2 voxels
    target[:, :, 0, 0, 1:3] = 1      # 2 voxels, 1 shared
    assert pytest.approx(float(dice_score(prediction, target))) == 2 * 1 / (2 + 2)
    assert pytest.approx(float(iou_score(prediction, target))) == 1 / 3


def test_empty_masks_follow_the_documented_convention():
    empty = torch.zeros(1, 1, 4, 4, 4)
    full = torch.ones(1, 1, 4, 4, 4)
    assert float(dice_score(empty, empty)) == 1.0
    assert float(iou_score(empty, empty)) == 1.0
    assert float(dice_score(full, empty)) == 0.0
    assert float(iou_score(full, empty)) == 0.0


def test_metrics_accept_logits():
    target = torch.zeros(1, 1, 4, 4, 4)
    target[:, :, 0] = 1
    logits = (target * 2 - 1) * 10
    assert pytest.approx(float(dice_score(logits, target, from_logits=True))) == 1.0


def test_per_class_metrics_follow_the_requested_prompt_order():
    target = torch.zeros(2, 3, 4, 4, 4)
    target[:, :, 0] = 1
    logits = (target * 2 - 1) * 10
    metrics = PerClassMetrics(SHAPE_NAMES)
    class_ids = torch.tensor([[5, 0, 9], [5, 0, 9]])
    metrics.update(logits, target, class_ids=class_ids)
    per_class = metrics.per_class()
    assert set(per_class) == {"cone", "cube", "capsule"}
    assert per_class["cone"]["count"] == 2
    assert pytest.approx(metrics.mean()["dice"]) == 1.0
    assert "mean" in format_per_class_table(metrics)


def test_hausdorff_is_zero_for_identical_masks_and_grows_with_displacement():
    mask = torch.zeros(8, 8, 8)
    mask[2:5, 2:5, 2:5] = 1
    assert hausdorff_distance(mask, mask) == 0.0
    shifted = torch.zeros(8, 8, 8)
    shifted[3:6, 2:5, 2:5] = 1          # one voxel along z
    near = hausdorff_distance(mask, shifted)
    further = torch.zeros(8, 8, 8)
    further[4:7, 2:5, 2:5] = 1          # two voxels along z
    assert 0 < near < hausdorff_distance(mask, further)


def test_hausdorff_is_symmetric_and_spacing_aware():
    a = torch.zeros(8, 8, 8); a[2:5, 2:5, 2:5] = 1
    b = torch.zeros(8, 8, 8); b[2:5, 2:5, 4:7] = 1
    assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a))
    # The offset is along x, so only the x spacing may scale the distance.
    isotropic = hausdorff_distance(a, b)
    assert hausdorff_distance(a, b, spacing=(2.0, 1.0, 1.0)) == pytest.approx(2 * isotropic)


def test_hausdorff_degenerate_cases_follow_the_documented_convention():
    empty = torch.zeros(8, 8, 8)
    full = torch.zeros(8, 8, 8); full[2:5, 2:5, 2:5] = 1
    assert hausdorff_distance(empty, empty) == 0.0
    import math
    assert math.isnan(hausdorff_distance(full, empty))
    assert math.isnan(hausdorff_distance(empty, full))


def test_hausdorff_percentile_is_not_larger_than_the_maximum():
    a = torch.zeros(12, 12, 12); a[2:8, 2:8, 2:8] = 1
    b = torch.zeros(12, 12, 12); b[3:9, 3:9, 3:9] = 1
    b[10, 10, 10] = 1  # one stray voxel
    assert hausdorff_distance(a, b, percentile=95) < hausdorff_distance(a, b)


def test_surface_voxels_excludes_the_interior():
    mask = torch.zeros(9, 9, 9)
    mask[2:7, 2:7, 2:7] = 1           # 5^3 = 125 voxels, 3^3 = 27 interior
    surface = surface_voxels(mask)
    assert surface.shape[1] == 3
    assert 0 < surface.shape[0] < int(mask.sum())
    assert surface_voxels(torch.zeros(4, 4, 4)).shape == (0, 3)


def test_batch_hausdorff_reports_one_value_per_sample():
    a = torch.zeros(2, 1, 8, 8, 8); a[:, :, 2:5, 2:5, 2:5] = 1
    b = a.clone(); b[1, 0] = torch.roll(b[1, 0], shifts=2, dims=0)
    values = batch_hausdorff(a, b)
    assert len(values) == 2
    assert values[0] == 0.0 and values[1] > 0.0
    with pytest.raises(ValueError):
        batch_hausdorff(torch.zeros(2, 3, 4, 4, 4), torch.zeros(2, 3, 4, 4, 4))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_dataset():
    try:
        return SceneDataset(SMOKE_ROOT, "train", limit=2)
    except DatasetError as error:  # pragma: no cover - dataset not generated
        pytest.skip(f"smoke corpus unavailable: {error}")


def test_dataset_items_match_the_stage_a_contract(smoke_dataset):
    item = smoke_dataset[0]
    depth, height, width = smoke_dataset.volume_shape
    assert item["scene_volume"].shape == (1, depth, height, width)
    assert item["target_masks"].shape == (10, depth, height, width)
    assert item["prompt_ids"].tolist() == list(range(10))
    assert set(item["target_masks"].unique().tolist()) <= {0.0, 1.0}
    # Every class is present exactly once per scene, so no mask may be empty.
    assert (item["target_masks"].sum(dim=(1, 2, 3)) > 0).all()
    # The masks partition the foreground.
    assert torch.equal(item["target_masks"].sum(dim=0), item["scene_volume"][0])


def test_dataset_masks_follow_the_requested_prompt_order():
    dataset = SceneDataset(SMOKE_ROOT, "train", prompt_names=["torus", "cube"], limit=1)
    item = dataset[0]
    assert item["prompt_ids"].tolist() == [8, 0]
    labels = item["instance_labels"]
    assert torch.equal(item["target_masks"][0], (labels == 9).float())
    assert torch.equal(item["target_masks"][1], (labels == 1).float())


def test_dataset_rejects_unknown_splits_and_names():
    with pytest.raises(DatasetError):
        SceneDataset(SMOKE_ROOT, "nope")
    with pytest.raises(Exception):
        SceneDataset(SMOKE_ROOT, "train", prompt_names=["blob"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def test_scheduler_warms_up_then_decays():
    settings = TrainingSettings(epochs=10, warmup_epochs=2, learning_rate=1.0)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(optimizer, settings)
    rates = []
    for _ in range(10):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < rates[1]          # warmup
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] < rates[2]         # cosine decay
    assert rates[-1] < 0.05


def test_settings_resolve_from_the_hardware_profiles():
    laptop = TrainingSettings.for_stage_a(hardware_profile="laptop_mps")
    rtx = TrainingSettings.for_stage_a(hardware_profile="rtx5090")
    # Both profiles autocast in bf16; neither needs a gradient scaler.
    assert laptop.precision == "bf16" and laptop.model_profile == "default"
    assert rtx.precision == "bf16" and rtx.batch_size > laptop.batch_size
    assert not laptop.needs_grad_scaler and not rtx.needs_grad_scaler
    smoke = TrainingSettings.for_stage_a(smoke=True)
    assert smoke.model_profile == "smoke" and smoke.epochs < laptop.epochs
    assert TrainingSettings.for_stage_a(overrides={"epochs": 3}).epochs == 3


def test_resolve_device_accepts_an_explicit_name():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}


def test_a_few_training_steps_reduce_the_loss_and_raise_dice(smoke_dataset, tmp_path):
    seed_everything(0)
    dataset = SceneDataset(SMOKE_ROOT, "train", limit=2)
    loader = build_dataloader(dataset, batch_size=1)
    config = type(TINY)(**{**TINY.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    settings = TrainingSettings(
        epochs=3, batch_size=1, device="cpu", learning_rate=5e-3, warmup_epochs=0, seed=0
    )
    trainer = StageATrainer(
        ShapeSegmenter(config),
        settings,
        loader,
        loader,
        class_names=SHAPE_NAMES,
        output_dir=tmp_path,
        verbose=False,
    )
    before = trainer.evaluate(loader)
    history = trainer.fit()
    after = trainer.evaluate(loader)

    assert len(history) == 3
    assert history[-1].train_loss < history[0].train_loss
    assert after["mean"]["dice"] >= before["mean"]["dice"]
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.is_file()
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert len(records) == 3
    assert "train_loss" in records[-1] and "val_dice" in records[-1]
    assert "train_dice" in records[-1]
    metadata = json.loads((tmp_path / "best.json").read_text())
    assert metadata["stage"] == "stage_a"
    assert metadata["selection_metric"] == "val_mean_dice"
    assert metadata["versions"]["schema_version"]
    assert "git_revision" in metadata["environment"]


# ---------------------------------------------------------------------------
# Autocast, precision and gradient scaling
# ---------------------------------------------------------------------------
def test_precision_maps_to_an_autocast_dtype_and_a_scaler_decision():
    assert TrainingSettings(precision="fp32").autocast_dtype is None
    assert TrainingSettings(precision="bf16").autocast_dtype is torch.bfloat16
    assert TrainingSettings(precision="fp16").autocast_dtype is torch.float16
    # Only fp16 needs loss scaling; bf16 has float32's exponent range.
    assert TrainingSettings(precision="fp16").needs_grad_scaler
    assert not TrainingSettings(precision="bf16").needs_grad_scaler
    assert not TrainingSettings(precision="fp32").needs_grad_scaler
    with pytest.raises(ValueError):
        TrainingSettings(precision="int8")


def test_losses_are_computed_in_float32_whatever_the_input_dtype():
    """Autocast must not change the objective: 262k-voxel sums need fp32."""
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 16, 16, 16) * 3 - 6
    target = torch.zeros(1, 2, 16, 16, 16)
    target[:, :, 4:8, 4:8, 4:8] = 1
    reference = segmentation_loss(logits, target)[0]
    for dtype in (torch.bfloat16, torch.float16):
        cast = segmentation_loss(logits.to(dtype), target)[0]
        assert cast.dtype == torch.float32
        assert torch.allclose(cast, reference, atol=2e-2), dtype
    assert dice_loss(logits.half(), target).dtype == torch.float32
    assert bce_loss(logits.half(), target).dtype == torch.float32


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_training_runs_under_every_precision_on_cpu(smoke_dataset, tmp_path, precision):
    """The autocast and gradient-scaler paths must both actually step."""
    seed_everything(0)
    dataset = SceneDataset(SMOKE_ROOT, "train", limit=1)
    loader = build_dataloader(dataset, batch_size=1)
    config = type(TINY)(**{**TINY.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    settings = TrainingSettings(
        epochs=1, batch_size=1, device="cpu", precision=precision, warmup_epochs=0, seed=0
    )
    model = ShapeSegmenter(config)
    before = [p.detach().clone() for p in model.parameters()]
    trainer = StageATrainer(
        model, settings, loader, loader,
        class_names=SHAPE_NAMES, output_dir=tmp_path / precision, verbose=False,
    )
    assert trainer.scaler.is_enabled() == (precision == "fp16")
    trainer.fit()
    moved = [
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters())
    ]
    assert any(moved), "no parameter was updated"


def test_gradient_accumulation_reduces_the_number_of_optimiser_steps(smoke_dataset, tmp_path):
    dataset = SceneDataset(SMOKE_ROOT, "train", limit=4)
    loader = build_dataloader(dataset, batch_size=1)
    config = type(TINY)(**{**TINY.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    steps: list[int] = []
    for accumulation in (1, 4):
        seed_everything(0)
        settings = TrainingSettings(
            epochs=1, batch_size=1, device="cpu", warmup_epochs=0, seed=0,
            gradient_accumulation_steps=accumulation,
        )
        trainer = StageATrainer(
            ShapeSegmenter(config), settings, loader, None,
            class_names=SHAPE_NAMES, output_dir=tmp_path / str(accumulation), verbose=False,
        )
        counter = {"n": 0}
        original = trainer.optimizer.step

        def counted(*a, _orig=original, _c=counter, **k):
            _c["n"] += 1
            return _orig(*a, **k)

        trainer.optimizer.step = counted
        trainer.train_epoch(0)
        steps.append(counter["n"])
    assert steps == [4, 1]
