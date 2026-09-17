"""The mri-like campaign: five hexagons, data/dataset_mri, shared W&B projects."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_mri_experiments import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROFILE,
    EXPERIMENTS_BY_ID,
    MRI_EXPERIMENTS,
    PYTHON_DOC,
    RunContext,
    documented_evaluate_command,
    documented_training_command,
    evaluate_argv,
    main,
    selected_experiments,
    training_argv,
)
from scripts.run_realistic_experiments import EXPERIMENTS_BY_ID as REALISTIC_BY_ID


def test_the_five_hexagons_match_the_experiments_table():
    assert [item.id for item in MRI_EXPERIMENTS] == [
        "EX-6", "EX-8", "EX-7", "EX-9", "EX-10",
    ]
    assert EXPERIMENTS_BY_ID["EX-6"].run == "dataset_mri_like"
    assert EXPERIMENTS_BY_ID["EX-6"].output.as_posix() == "runs/shape_segmenter/dataset_mri_like"
    assert EXPERIMENTS_BY_ID["EX-7"].run == "pred_dataset_mri_like_augB"
    assert EXPERIMENTS_BY_ID["EX-7"].stage_a_id == "EX-6"
    assert EXPERIMENTS_BY_ID["EX-8"].run == "dataset_mri_like_aug"
    assert EXPERIMENTS_BY_ID["EX-9"].stage_a_id == "EX-8"
    assert EXPERIMENTS_BY_ID["EX-10"].anchors == "oracle"
    assert EXPERIMENTS_BY_ID["EX-10"].augment is False
    assert all(item.phase == "a" for item in MRI_EXPERIMENTS[:2])
    assert all(item.phase == "b" for item in MRI_EXPERIMENTS[2:])
    assert all(item.dataset == "mri-like" for item in MRI_EXPERIMENTS)


def test_wandb_project_splits_on_phase_not_dataset():
    assert EXPERIMENTS_BY_ID["EX-6"].wandb_project == REALISTIC_BY_ID["EX-1"].wandb_project
    assert EXPERIMENTS_BY_ID["EX-8"].wandb_project == "relational-3d-vlm-phase-a"
    assert EXPERIMENTS_BY_ID["EX-7"].wandb_project == REALISTIC_BY_ID["EX-2"].wandb_project
    assert EXPERIMENTS_BY_ID["EX-10"].wandb_project == "relational-3d-vlm-phase-b"
    assert "dataset-mri-like" in EXPERIMENTS_BY_ID["EX-6"].wandb_tags()
    assert "dataset-realistic" not in EXPERIMENTS_BY_ID["EX-6"].wandb_tags()


def test_wandb_tags_are_derived_from_the_table_row():
    assert EXPERIMENTS_BY_ID["EX-6"].wandb_tags() == (
        "EX-6", "dataset_mri_like", "phase-a", "dataset-mri-like",
        "epochs-50", "patience-0", "no-aug-a",
    )
    assert EXPERIMENTS_BY_ID["EX-7"].wandb_tags() == (
        "EX-7", "pred_dataset_mri_like_augB", "phase-b", "dataset-mri-like",
        "epochs-30", "patience-0", "aug-b", "anchors-predicted",
        "stage-a-dataset_mri_like", "no-aug-a",
    )
    assert EXPERIMENTS_BY_ID["EX-10"].wandb_tags() == (
        "EX-10", "oracle_dataset_mri_like", "phase-b", "dataset-mri-like",
        "epochs-30", "patience-0", "no-aug-b", "anchors-oracle",
    )


def test_stage_a_flags_are_explicit_and_point_at_dataset_mri():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-6"])
    assert command == """\
.venv/bin/python scripts/train_shape_segmenter.py \\
    --data-root data/dataset_mri \\
    --output runs/shape_segmenter/dataset_mri_like \\
    --profile rtx5090 \\
    --epochs 50 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-a \\
    --wandb-tag EX-6 \\
    --wandb-tag dataset_mri_like \\
    --wandb-tag phase-a \\
    --wandb-tag dataset-mri-like \\
    --wandb-tag epochs-50 \\
    --wandb-tag patience-0 \\
    --wandb-tag no-aug-a \\
    --no-augment"""
    command_aug = documented_training_command(EXPERIMENTS_BY_ID["EX-8"])
    assert "--augment" in command_aug
    assert "--no-augment" not in command_aug
    assert DEFAULT_DATA_ROOT.as_posix() == "data/dataset_mri"


def test_predicted_stage_b_names_the_stage_a_checkpoint_and_turns_aug_on():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-7"])
    assert command == """\
.venv/bin/python scripts/train_relational_model.py \\
    --phase oracle \\
    --anchor-source predicted \\
    --stage-a-checkpoint runs/shape_segmenter/dataset_mri_like/best.pt \\
    --data-root data/dataset_mri \\
    --output runs/relational_model/predicted/pred_dataset_mri_like_augB \\
    --profile rtx5090 \\
    --epochs 30 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-b \\
    --wandb-tag EX-7 \\
    --wandb-tag pred_dataset_mri_like_augB \\
    --wandb-tag phase-b \\
    --wandb-tag dataset-mri-like \\
    --wandb-tag epochs-30 \\
    --wandb-tag patience-0 \\
    --wandb-tag aug-b \\
    --wandb-tag anchors-predicted \\
    --wandb-tag stage-a-dataset_mri_like \\
    --wandb-tag no-aug-a \\
    --augment"""


def test_oracle_stage_b_has_no_stage_a_and_turns_aug_off():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-10"])
    assert "--stage-a-checkpoint" not in command
    assert "--anchor-source oracle" in command
    assert "--no-augment" in command
    assert "--data-root data/dataset_mri" in command


def test_test_set_evaluation_is_a_second_command_on_stage_b_only():
    assert documented_evaluate_command(EXPERIMENTS_BY_ID["EX-6"]) is None
    command = documented_evaluate_command(EXPERIMENTS_BY_ID["EX-7"])
    assert command == """\
.venv/bin/python scripts/evaluate.py \\
    --stage-b-checkpoint runs/relational_model/predicted/pred_dataset_mri_like_augB/best.pt \\
    --stage-a-checkpoint runs/shape_segmenter/dataset_mri_like/best.pt \\
    --data-root data/dataset_mri \\
    --split test \\
    --anchor-source predicted"""
    oracle = documented_evaluate_command(EXPERIMENTS_BY_ID["EX-10"])
    assert "--anchor-source oracle" in oracle
    assert "--stage-a-checkpoint" not in oracle
    assert "--data-root data/dataset_mri" in oracle


def test_schedule_flags_are_on_every_training_command():
    ctx = RunContext(python=PYTHON_DOC, data_root=DEFAULT_DATA_ROOT, profile=DEFAULT_PROFILE)
    for experiment in MRI_EXPERIMENTS:
        argv = training_argv(experiment, ctx)
        assert argv[argv.index("--epochs") + 1] == str(experiment.epochs)
        assert argv[argv.index("--data-root") + 1] == "data/dataset_mri"
        assert argv[argv.index("--wandb-project") + 1] == experiment.wandb_project
        for tag in experiment.wandb_tags():
            assert tag in argv
        assert "--wandb-project" not in (evaluate_argv(experiment, ctx) or [])


def test_experiments_markdown_embeds_the_runner_commands():
    page = Path(__file__).resolve().parents[1].joinpath("docs/EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "scripts/run_mri_experiments.py" in page
    for experiment in MRI_EXPERIMENTS:
        assert documented_training_command(experiment) in page
        eval_command = documented_evaluate_command(experiment)
        if eval_command is not None:
            assert eval_command in page


def test_only_selects_a_subset_in_the_requested_order():
    chosen = selected_experiments(["EX-10", "EX-6"])
    assert [item.id for item in chosen] == ["EX-10", "EX-6"]
    with pytest.raises(SystemExit, match="unknown experiment"):
        selected_experiments(["EX-1"])


def test_dry_run_prints_every_command_and_does_not_need_the_corpus(capsys):
    code = main(["--dry-run", "--only", "EX-6", "EX-10"])
    assert code == 0
    output = capsys.readouterr().out
    assert "MRI-LIKE EXPERIMENTS" in output
    assert "scripts/train_shape_segmenter.py" in output
    assert "scripts/train_relational_model.py" in output
    assert "scripts/evaluate.py" in output
    assert "--data-root data/dataset_mri" in output
    assert "campaign log:" in output
