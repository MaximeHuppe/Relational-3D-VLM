"""The Stage B dataset, anchor sources, metrics and training loops.

The loop tests run a handful of steps on the smoke corpus and assert that the
loss falls - enough to catch a detached graph, a broken optimiser or a
target/prediction misalignment, without depending on any particular accuracy.

The anchor-source tests are the important ones: they check that the ground-truth
and Stage A-predicted paths are interchangeable, that the predicted path really
does swap the channels, and that the scene volume it needs never reaches
Stage B.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import (
    DatasetError,
    ExampleDataset,
    build_example_dataloader,
    collate_examples,
    stage_b_model_inputs,
)
from src.data.schema import STAGE_B_FORBIDDEN_FIELDS, read_manifest
from src.evaluation.metrics import (
    StratifiedMetrics,
    batch_hausdorff,
    format_stratified_table,
    hausdorff_distance,
    surface_voxels,
)
from src.models.anchor_provider import (
    AnchorProviderError,
    OracleAnchorProvider,
    PredictedAnchorProvider,
    build_anchor_provider,
    load_stage_a,
)
from src.models.relational_vlm import RelationalVLM, RelationalVLMConfig
from src.models.shape_segmenter import ShapeSegmenter
from src.training.trainer import (
    StageBOverfitRunner,
    StageBTrainer,
    TrainingSettings,
    resolve_device,
    seed_everything,
)
from tests.test_model_contract import TINY as TINY_STAGE_A

SMOKE_ROOT = "data/smoke"

SMALL = RelationalVLMConfig(
    encoder_channels=(4, 8, 16, 32),
    decoder_channels=(16, 8, 4),
    input_resolution=64,
    bottleneck_resolution=8,
    token_dim=16,
    embedding_dim=16,
    num_heads=2,
    intersection_hidden_channels=16,
)


@pytest.fixture(scope="module")
def dataset() -> ExampleDataset:
    try:
        return ExampleDataset(SMOKE_ROOT, "train", limit=4)
    except DatasetError as error:  # pragma: no cover - corpus not generated
        pytest.skip(f"smoke corpus unavailable: {error}")


@pytest.fixture(scope="module")
def scene_dataset() -> ExampleDataset:
    try:
        return ExampleDataset(SMOKE_ROOT, "train", limit=2, include_scene_volume=True)
    except DatasetError as error:  # pragma: no cover
        pytest.skip(f"smoke corpus unavailable: {error}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_items_match_the_stage_b_contract(dataset):
    item = dataset[0]
    depth, height, width = dataset.volume_shape
    assert item["anchor_masks"].shape == (3, depth, height, width)
    assert item["target_mask"].shape == (1, depth, height, width)
    assert item["direction_ids"].shape == (3,) and item["anchor_shape_ids"].shape == (3,)
    assert set(item["anchor_masks"].unique().tolist()) <= {0.0, 1.0}
    # Three non-empty, disjoint anchors, none of which is the target.
    assert (item["anchor_masks"].sum(dim=(1, 2, 3)) > 0).all()
    assert float(item["anchor_masks"].sum(0).max()) == 1.0
    assert float((item["anchor_masks"].sum(0) * item["target_mask"][0]).sum()) == 0.0
    assert "scene_volume" not in item


def test_the_channels_are_the_three_anchor_structures_of_the_scene(dataset):
    """Oracle anchors = the .npz labels with everything but the anchors dropped."""
    record = dataset.records[0]
    with np.load(record.path) as arrays:
        labels = arrays["instance_labels"]
    item = dataset[0]
    for slot, instance_id in enumerate(record.metadata.anchor_instance_ids):
        expected = torch.from_numpy((labels == instance_id).astype("float32"))
        assert torch.equal(item["anchor_masks"][slot], expected)
    target = record.metadata.target_instance_id
    assert torch.equal(item["target_mask"][0], torch.from_numpy((labels == target).astype("float32")))


def test_the_prompt_indices_match_the_manifest_clauses(dataset):
    from src.data.prompt_generator import clauses_from_indices

    item = dataset[0]
    relations = dataset.records[0].metadata.relations
    assert clauses_from_indices(item["direction_ids"], item["anchor_shape_ids"]) == relations
    assert item["anchor_shape_names"] == [clause["anchor"] for clause in relations]
    assert item["directions"] == [clause["direction"] for clause in relations]


def test_the_scene_volume_is_opt_in_and_only_for_predicted_anchors(dataset, scene_dataset):
    assert "scene_volume" not in dataset[0]
    item = scene_dataset[0]
    assert item["scene_volume"].shape == (1, *scene_dataset.volume_shape)
    # ...and it is not a Stage B input even when the dataset carries it.
    assert "scene_volume" not in stage_b_model_inputs(item)


def test_split_filtering_and_scene_selection():
    train = ExampleDataset(SMOKE_ROOT, "train")
    val = ExampleDataset(SMOKE_ROOT, "val")
    # configs/split.yaml holds two classes out of training, and they are exactly
    # what validation supervises.
    assert set(train.target_shape_counts()) == {
        "cube", "sphere", "cylinder", "cone", "pyramid", "torus", "capsule"
    }
    assert set(val.target_shape_counts()) == {"cuboid", "ellipsoid"}
    assert not set(train.target_shape_counts()) & set(val.target_shape_counts())

    one_scene = ExampleDataset(SMOKE_ROOT, "train", scene_ids=train.scene_ids[:1])
    assert one_scene.scene_ids == train.scene_ids[:1]
    assert len(one_scene) < len(train)
    with pytest.raises(DatasetError):
        ExampleDataset(SMOKE_ROOT, "train", scene_ids=["no_such_scene"])
    with pytest.raises(DatasetError):
        ExampleDataset(SMOKE_ROOT, "nope")


def test_collation_keeps_one_list_entry_per_sample(dataset):
    batch = collate_examples([dataset[0], dataset[1]])
    assert batch["anchor_masks"].shape[0] == 2
    assert len(batch["anchor_shape_names"]) == 2
    assert len(batch["anchor_shape_names"][0]) == 3
    assert batch["target_shape_name"] == [dataset[0]["target_shape_name"], dataset[1]["target_shape_name"]]


def test_a_corrupt_example_fails_fast_rather_than_training_on_it(tmp_path):
    """Validation runs on first visit; a mismatched channel must not slip through."""
    import shutil

    root = tmp_path / "corpus"
    shutil.copytree(SMOKE_ROOT, root)
    manifest = root / "manifests" / "train.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records[0]["anchor_instance_ids"][0] = records[0]["target_instance_id"]
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(Exception):
        ExampleDataset(root, "train")[0]


# ---------------------------------------------------------------------------
# Anchor sources
# ---------------------------------------------------------------------------
def test_the_oracle_provider_returns_the_ground_truth_channels(dataset):
    batch = collate_examples([dataset[0]])
    provider = build_anchor_provider("oracle")
    assert provider.source == "oracle"
    assert torch.equal(provider(batch), batch["anchor_masks"])
    with pytest.raises(AnchorProviderError):
        provider({"direction_ids": torch.zeros(1, 3, dtype=torch.long)})


def test_the_predicted_provider_segments_the_named_anchors_in_clause_order(scene_dataset):
    torch.manual_seed(0)
    stage_a = ShapeSegmenter(
        type(TINY_STAGE_A)(**{**TINY_STAGE_A.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    )
    provider = PredictedAnchorProvider(model=stage_a, threshold=0.5)
    batch = collate_examples([scene_dataset[0], scene_dataset[1]])
    masks = provider(batch)

    assert masks.shape == batch["anchor_masks"].shape
    assert set(masks.unique().tolist()) <= {0.0, 1.0}
    # Channel i is Stage A's answer for the shape named in clause i.
    reference = stage_a(batch["scene_volume"], batch["anchor_shape_ids"], deep_supervision=False)
    assert torch.equal(masks, (torch.sigmoid(reference.logits) >= 0.5).float())
    # The provider scores its own anchors so a Stage B drop can be attributed.
    quality = provider.anchor_quality()
    assert set(quality) >= {"anchor_dice", "anchor_iou", "empty_anchor_fraction"}
    assert quality["channels"] == 6.0
    provider.reset()
    assert provider.anchor_quality() == {}


def test_predicted_anchors_need_the_scene_volume_and_a_checkpoint(dataset):
    torch.manual_seed(0)
    stage_a = ShapeSegmenter(
        type(TINY_STAGE_A)(**{**TINY_STAGE_A.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    )
    provider = PredictedAnchorProvider(model=stage_a)
    with pytest.raises(AnchorProviderError):
        provider(collate_examples([dataset[0]]))       # dataset built without it
    with pytest.raises(AnchorProviderError):
        build_anchor_provider("predicted")             # no checkpoint
    with pytest.raises(AnchorProviderError):
        build_anchor_provider("nonsense")
    with pytest.raises(AnchorProviderError):
        load_stage_a("runs/does_not_exist.pt")


def test_the_config_spellings_of_the_two_sources_are_accepted():
    assert build_anchor_provider("ground_truth").source == "oracle"


def test_stage_b_gets_the_same_interface_from_either_source(scene_dataset):
    """The model call is identical; only the three channels differ."""
    torch.manual_seed(0)
    stage_a = ShapeSegmenter(
        type(TINY_STAGE_A)(**{**TINY_STAGE_A.__dict__, "input_resolution": 64, "bottleneck_resolution": 8})
    )
    model = RelationalVLM(SMALL).eval()
    batch = collate_examples([scene_dataset[0]])
    for provider in (OracleAnchorProvider(), PredictedAnchorProvider(model=stage_a)):
        inputs = stage_b_model_inputs(batch, provider(batch))
        assert set(inputs) == {"anchor_masks", "direction_ids", "anchor_shape_ids"}
        for forbidden in STAGE_B_FORBIDDEN_FIELDS:
            assert forbidden not in inputs
        with torch.no_grad():
            logits = model(**inputs).logits
        assert logits.shape == (1, 1, 64, 64, 64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_surface_voxels_are_the_shell_not_the_interior():
    mask = torch.zeros(6, 6, 6)
    mask[1:5, 1:5, 1:5] = 1
    assert float(mask.sum()) == 64
    assert surface_voxels(mask).shape == (64 - 8, 3)   # 4^3 block minus its 2^3 core
    assert surface_voxels(torch.zeros(4, 4, 4)).shape == (0, 3)


def test_hausdorff_measures_both_directions():
    a = torch.zeros(10, 10, 10)
    a[2:5, 2:5, 2:5] = 1
    b = a.clone()
    b[2:5, 2:5, 7] = 1                      # a spur 3 voxels away from a's surface
    assert hausdorff_distance(a, a) == 0.0
    assert hausdorff_distance(a, b) == pytest.approx(3.0)
    assert hausdorff_distance(b, a) == pytest.approx(3.0)   # symmetric


def test_hausdorff_degenerate_cases_follow_the_dice_convention():
    empty = torch.zeros(4, 4, 4)
    full = torch.ones(4, 4, 4)
    assert hausdorff_distance(empty, empty) == 0.0
    assert np.isnan(hausdorff_distance(empty, full))
    assert np.isnan(hausdorff_distance(full, empty))


def test_hausdorff_accepts_logits_a_percentile_and_spacing():
    target = torch.zeros(8, 8, 8)
    target[2:5, 2:5, 2:5] = 1
    logits = (target * 2 - 1) * 10
    assert hausdorff_distance(logits, target, from_logits=True) == 0.0
    prediction = target.clone()
    prediction[2:5, 2:5, 7] = 1
    # Spacing is world units per voxel, so stretching x stretches the distance.
    assert hausdorff_distance(prediction, target, spacing=(2.0, 1.0, 1.0)) == pytest.approx(
        2 * hausdorff_distance(prediction, target)
    )
    # A single stray voxel dominates the max but not the 95th percentile.
    stray = target.clone()
    stray[7, 7, 7] = 1
    assert hausdorff_distance(stray, target, percentile=95) < hausdorff_distance(stray, target)


def test_batch_hausdorff_is_per_sample():
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1
    prediction = target.clone()
    prediction[1, 0, 2:5, 2:5, 7] = 1
    distances = batch_hausdorff(prediction, target)
    assert distances[0] == 0.0 and distances[1] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        batch_hausdorff(torch.zeros(1, 3, 4, 4, 4), torch.zeros(1, 3, 4, 4, 4))


def test_stratified_metrics_bucket_by_target_anchor_direction_and_slot():
    prediction = torch.zeros(2, 1, 8, 8, 8)
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1
    prediction[0] = (target[0] * 2 - 1) * 10          # perfect
    prediction[1] = -10                               # empty
    metrics = StratifiedMetrics()
    metrics.update(
        prediction,
        target,
        target_shapes=["cube", "sphere"],
        anchor_shapes=[["torus", "cone", "cube"], ["torus", "cone", "cube"]],
        directions=[["medial", "superior", "anterior"], ["lateral", "inferior", "posterior"]],
        with_hausdorff=True,
    )
    summary = metrics.summary()
    assert summary["overall"]["dice"] == pytest.approx(0.5)
    assert summary["strata"]["target_shape"]["cube"]["dice"] == pytest.approx(1.0)
    assert summary["strata"]["target_shape"]["sphere"]["dice"] == pytest.approx(0.0)
    # An anchor seen by both samples averages them.
    assert summary["strata"]["anchor_shape"]["torus"]["count"] == 2.0
    assert summary["strata"]["direction"]["medial"]["count"] == 1.0
    assert summary["strata"]["clause_slot"]["slot_1"]["count"] == 2.0
    # The empty prediction has no surface, so its distance is undefined, not 0.
    assert summary["overall"]["hausdorff_undefined"] == 1.0
    assert "target_shape" in format_stratified_table(metrics)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def test_stage_b_settings_resolve_from_the_config():
    overfit = TrainingSettings.for_stage_b(phase="overfit")
    oracle = TrainingSettings.for_stage_b(phase="oracle")
    assert overfit.steps > 0 and overfit.target_train_dice == 0.95
    # Resolved from the config, not pinned to a literal: `epochs` is a tuning
    # knob and a hardcoded value here fails every time an experiment changes it.
    configured = load_config("train")["stage_b_oracle"]
    assert oracle.epochs == int(configured["epochs"]) and oracle.epochs > 0
    assert oracle.anchor_source == "oracle"
    assert oracle.learning_rate == float(configured["optimizer"]["lr"])
    smoke = TrainingSettings.for_stage_b(phase="oracle", smoke=True)
    assert smoke.model_profile == "smoke" and smoke.epochs < oracle.epochs
    assert TrainingSettings.for_stage_b(phase="oracle", overrides={"epochs": 2}).epochs == 2
    with pytest.raises(ValueError):
        TrainingSettings.for_stage_b(phase="predicted")


def test_autocast_is_disabled_on_cpu_where_it_is_a_30x_regression():
    settings = TrainingSettings(precision="bf16")
    assert settings.autocast_dtype == torch.bfloat16
    assert settings.autocast_dtype_on(torch.device("cpu")) is None
    assert settings.autocast_dtype_on(torch.device("mps")) == torch.bfloat16
    assert TrainingSettings(precision="fp32").autocast_dtype_on(torch.device("mps")) is None


def test_a_few_stage_b_steps_reduce_the_loss(dataset, tmp_path):
    seed_everything(0)
    loader = build_example_dataloader(dataset, batch_size=2)
    settings = TrainingSettings(
        epochs=2, batch_size=2, device="cpu", learning_rate=5e-3, warmup_epochs=0, seed=0
    )
    trainer = StageBTrainer(
        RelationalVLM(SMALL), settings, loader, loader, output_dir=tmp_path, verbose=False
    )
    history = trainer.fit()
    assert len(history) == 2
    assert history[-1].train_loss < history[0].train_loss
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "history.json").is_file()
    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.is_file()
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert len(records) == 2
    assert {"train_loss", "train_dice", "val_loss", "val_dice"} <= set(records[-1])
    metadata = json.loads((tmp_path / "best.json").read_text())
    assert metadata["stage"] == "stage_b_oracle"
    assert metadata["selection_metric"] == "val_dice"
    assert metadata["versions"]["schema_version"]
    assert metadata["extra"]["anchors"]["anchor_source"] == "oracle"


def test_the_evaluation_report_is_stratified_and_names_its_anchor_source(dataset, tmp_path):
    seed_everything(0)
    loader = build_example_dataloader(dataset, batch_size=2)
    trainer = StageBTrainer(
        RelationalVLM(SMALL),
        TrainingSettings(epochs=1, device="cpu"),
        loader,
        loader,
        output_dir=tmp_path,
        verbose=False,
    )
    report = trainer.evaluate(loader, with_hausdorff=True)
    assert report["anchor_source"] == "oracle"
    assert set(report["strata"]) == {"target_shape", "anchor_shape", "direction", "clause_slot"}
    assert report["overall"]["count"] == float(len(dataset))
    assert "hausdorff" in report["overall"]
    assert "table" in report


def test_the_overfit_runner_reports_pass_or_fail_against_its_target(dataset, tmp_path):
    """Two steps cannot reach Dice 0.95, and the runner must say so."""
    seed_everything(0)
    loader = build_example_dataloader(ExampleDataset(SMOKE_ROOT, "train", limit=2), batch_size=1)
    settings = TrainingSettings(
        epochs=1, batch_size=1, device="cpu", learning_rate=1e-3, steps=2,
        target_train_dice=0.95, warmup_epochs=0, seed=0,
    )
    runner = StageBOverfitRunner(
        RelationalVLM(SMALL), settings, loader, None, output_dir=tmp_path, verbose=False
    )
    report = runner.run()
    assert report["passed"] is False
    assert report["steps_run"] == 2
    assert report["stage"] == "stage_b_overfit"
    assert report["anchor_source"] == "oracle"
    assert 0.0 <= report["final_train_dice"] <= 1.0
    assert (tmp_path / "overfit_report.json").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()


def test_the_trainer_never_hands_the_target_mask_to_the_model(dataset, tmp_path):
    """The loop's only model call goes through the input helper."""
    seen: list[set[str]] = []

    class Spy(RelationalVLM):
        def forward(self, *args, **kwargs):  # type: ignore[override]
            seen.append(set(kwargs))
            return super().forward(*args, **kwargs)

    loader = build_example_dataloader(dataset, batch_size=2)
    trainer = StageBTrainer(
        Spy(SMALL), TrainingSettings(epochs=1, device="cpu"), loader, None,
        output_dir=tmp_path, verbose=False,
    )
    trainer.train_epoch(0)
    assert seen and all(call == {"anchor_masks", "direction_ids", "anchor_shape_ids"} for call in seen)


def test_resolve_device_and_manifest_reading_still_agree(dataset):
    assert resolve_device("cpu").type == "cpu"
    metadata = list(read_manifest(f"{SMOKE_ROOT}/manifests/train.jsonl"))
    assert {record.example_id for record in dataset.records} <= {m.example_id for m in metadata}
