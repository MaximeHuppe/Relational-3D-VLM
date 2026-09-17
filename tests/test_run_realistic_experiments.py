"""The realistic-appearance campaign: five hexagons, explicit CLI, no silent defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_realistic_experiments import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROFILE,
    EXPERIMENTS_BY_ID,
    PYTHON_DOC,
    REALISTIC_EXPERIMENTS,
    RunContext,
    documented_evaluate_command,
    documented_training_command,
    evaluate_argv,
    format_command,
    main,
    selected_experiments,
    training_argv,
)


def test_the_five_hexagons_match_the_experiments_table():
    assert [item.id for item in REALISTIC_EXPERIMENTS] == [
        "EX-1", "EX-3", "EX-2", "EX-4", "EX-5",
    ]
    assert EXPERIMENTS_BY_ID["EX-1"].run == "dataset_realistic"
    assert EXPERIMENTS_BY_ID["EX-1"].output.as_posix() == "runs/shape_segmenter/dataset_realistic"
    assert EXPERIMENTS_BY_ID["EX-2"].run == "pred_dataset_realistic_augB"
    assert EXPERIMENTS_BY_ID["EX-2"].stage_a_id == "EX-1"
    assert EXPERIMENTS_BY_ID["EX-3"].run == "dataset_realistic_aug"
    assert EXPERIMENTS_BY_ID["EX-4"].stage_a_id == "EX-3"
    assert EXPERIMENTS_BY_ID["EX-5"].anchors == "oracle"
    assert EXPERIMENTS_BY_ID["EX-5"].augment is False
    assert all(item.phase == "a" for item in REALISTIC_EXPERIMENTS[:2])
    assert all(item.phase == "b" for item in REALISTIC_EXPERIMENTS[2:])


def test_wandb_project_splits_on_phase_not_dataset():
    assert EXPERIMENTS_BY_ID["EX-1"].wandb_project == "relational-3d-vlm-phase-a"
    assert EXPERIMENTS_BY_ID["EX-3"].wandb_project == "relational-3d-vlm-phase-a"
    assert EXPERIMENTS_BY_ID["EX-2"].wandb_project == "relational-3d-vlm-phase-b"
    assert EXPERIMENTS_BY_ID["EX-5"].wandb_project == "relational-3d-vlm-phase-b"
    from scripts.run_mri_experiments import EXPERIMENTS_BY_ID as MRI_BY_ID

    # mri-like shares those two projects; dataset is only a tag.
    assert MRI_BY_ID["EX-6"].wandb_project == EXPERIMENTS_BY_ID["EX-1"].wandb_project
    assert "dataset-mri-like" in MRI_BY_ID["EX-6"].wandb_tags()
    assert "dataset-realistic" in EXPERIMENTS_BY_ID["EX-1"].wandb_tags()


def test_wandb_tags_are_derived_from_the_table_row():
    assert EXPERIMENTS_BY_ID["EX-1"].wandb_tags() == (
        "EX-1", "dataset_realistic", "phase-a", "dataset-realistic",
        "epochs-50", "patience-0", "no-aug-a",
    )
    assert EXPERIMENTS_BY_ID["EX-2"].wandb_tags() == (
        "EX-2", "pred_dataset_realistic_augB", "phase-b", "dataset-realistic",
        "epochs-30", "patience-0", "aug-b", "anchors-predicted",
        "stage-a-dataset_realistic", "no-aug-a",
    )
    assert EXPERIMENTS_BY_ID["EX-5"].wandb_tags() == (
        "EX-5", "oracle_dataset_realistic", "phase-b", "dataset-realistic",
        "epochs-30", "patience-0", "no-aug-b", "anchors-oracle",
    )


def test_stage_a_flags_are_explicit_and_do_not_rely_on_config_defaults():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-1"])
    assert command == """\
.venv/bin/python scripts/train_shape_segmenter.py \\
    --data-root data/processed \\
    --output runs/shape_segmenter/dataset_realistic \\
    --profile rtx5090 \\
    --epochs 50 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-a \\
    --wandb-tag EX-1 \\
    --wandb-tag dataset_realistic \\
    --wandb-tag phase-a \\
    --wandb-tag dataset-realistic \\
    --wandb-tag epochs-50 \\
    --wandb-tag patience-0 \\
    --wandb-tag no-aug-a \\
    --no-augment"""
    command_aug = documented_training_command(EXPERIMENTS_BY_ID["EX-3"])
    assert "--augment" in command_aug
    assert "--no-augment" not in command_aug
    assert EXPERIMENTS_BY_ID["EX-1"].epochs == 50
    assert EXPERIMENTS_BY_ID["EX-1"].early_stopping_patience == 0


def test_predicted_stage_b_names_the_stage_a_checkpoint_and_turns_aug_on():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-2"])
    assert command == """\
.venv/bin/python scripts/train_relational_model.py \\
    --phase oracle \\
    --anchor-source predicted \\
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic/best.pt \\
    --data-root data/processed \\
    --output runs/relational_model/predicted/pred_dataset_realistic_augB \\
    --profile rtx5090 \\
    --epochs 30 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-b \\
    --wandb-tag EX-2 \\
    --wandb-tag pred_dataset_realistic_augB \\
    --wandb-tag phase-b \\
    --wandb-tag dataset-realistic \\
    --wandb-tag epochs-30 \\
    --wandb-tag patience-0 \\
    --wandb-tag aug-b \\
    --wandb-tag anchors-predicted \\
    --wandb-tag stage-a-dataset_realistic \\
    --wandb-tag no-aug-a \\
    --augment"""


def test_augA_predicted_stage_b_points_at_the_augmented_stage_a():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-4"])
    assert command == """\
.venv/bin/python scripts/train_relational_model.py \\
    --phase oracle \\
    --anchor-source predicted \\
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic_aug/best.pt \\
    --data-root data/processed \\
    --output runs/relational_model/predicted/pred_dataset_realistic_augA_augB \\
    --profile rtx5090 \\
    --epochs 30 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-b \\
    --wandb-tag EX-4 \\
    --wandb-tag pred_dataset_realistic_augA_augB \\
    --wandb-tag phase-b \\
    --wandb-tag dataset-realistic \\
    --wandb-tag epochs-30 \\
    --wandb-tag patience-0 \\
    --wandb-tag aug-b \\
    --wandb-tag anchors-predicted \\
    --wandb-tag stage-a-dataset_realistic_aug \\
    --wandb-tag aug-a \\
    --augment"""


def test_oracle_stage_b_has_no_stage_a_and_turns_aug_off():
    command = documented_training_command(EXPERIMENTS_BY_ID["EX-5"])
    assert command == """\
.venv/bin/python scripts/train_relational_model.py \\
    --phase oracle \\
    --anchor-source oracle \\
    --data-root data/processed \\
    --output runs/relational_model/oracle/oracle_dataset_realistic \\
    --profile rtx5090 \\
    --epochs 30 \\
    --early-stopping-patience 0 \\
    --early-stopping-min-delta 0.005 \\
    --wandb-project relational-3d-vlm-phase-b \\
    --wandb-tag EX-5 \\
    --wandb-tag oracle_dataset_realistic \\
    --wandb-tag phase-b \\
    --wandb-tag dataset-realistic \\
    --wandb-tag epochs-30 \\
    --wandb-tag patience-0 \\
    --wandb-tag no-aug-b \\
    --wandb-tag anchors-oracle \\
    --no-augment"""
    assert "--stage-a-checkpoint" not in command


def test_test_set_evaluation_is_a_second_command_on_stage_b_only():
    assert documented_evaluate_command(EXPERIMENTS_BY_ID["EX-1"]) is None
    command = documented_evaluate_command(EXPERIMENTS_BY_ID["EX-2"])
    assert command == """\
.venv/bin/python scripts/evaluate.py \\
    --stage-b-checkpoint runs/relational_model/predicted/pred_dataset_realistic_augB/best.pt \\
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic/best.pt \\
    --data-root data/processed \\
    --split test \\
    --anchor-source predicted"""
    oracle = documented_evaluate_command(EXPERIMENTS_BY_ID["EX-5"])
    assert "--anchor-source oracle" in oracle
    assert "--stage-a-checkpoint" not in oracle


def test_schedule_flags_are_on_every_training_command():
    ctx = RunContext(python=PYTHON_DOC, data_root=DEFAULT_DATA_ROOT, profile=DEFAULT_PROFILE)
    for experiment in REALISTIC_EXPERIMENTS:
        argv = training_argv(experiment, ctx)
        assert argv[argv.index("--epochs") + 1] == str(experiment.epochs)
        assert argv[argv.index("--early-stopping-patience") + 1] == str(experiment.early_stopping_patience)
        assert argv[argv.index("--early-stopping-min-delta") + 1] == str(experiment.early_stopping_min_delta)
        assert argv[argv.index("--wandb-project") + 1] == experiment.wandb_project
        for tag in experiment.wandb_tags():
            assert tag in argv
            assert argv[argv.index("--wandb-tag") :].count(tag) >= 1
        assert "--wandb-project" not in (evaluate_argv(experiment, ctx) or [])


def test_profile_override_lands_on_every_training_command():
    ctx = RunContext(python=PYTHON_DOC, data_root=DEFAULT_DATA_ROOT, profile="laptop_mps")
    for experiment in REALISTIC_EXPERIMENTS:
        argv = training_argv(experiment, ctx)
        assert argv[argv.index("--profile") + 1] == "laptop_mps"


def test_experiments_markdown_embeds_the_runner_commands():
    page = Path(__file__).resolve().parents[1].joinpath("docs/EXPERIMENTS.md").read_text(encoding="utf-8")
    for experiment in REALISTIC_EXPERIMENTS:
        assert documented_training_command(experiment) in page
        eval_command = documented_evaluate_command(experiment)
        if eval_command is not None:
            assert eval_command in page


def test_only_selects_a_subset_in_the_requested_order():
    chosen = selected_experiments(["EX-5", "EX-1"])
    assert [item.id for item in chosen] == ["EX-5", "EX-1"]
    with pytest.raises(SystemExit, match="unknown experiment"):
        selected_experiments(["EX-99"])


def test_dry_run_prints_every_command_and_does_not_need_the_corpus(capsys):
    code = main(["--dry-run", "--only", "EX-1", "EX-5"])
    assert code == 0
    output = capsys.readouterr().out
    assert "scripts/train_shape_segmenter.py" in output
    assert "scripts/train_relational_model.py" in output
    assert "scripts/evaluate.py" in output
    assert "--no-augment" in output
    assert "campaign log:" in output


def test_dry_run_no_evaluate_omits_the_test_set_dump(capsys):
    code = main(["--dry-run", "--no-evaluate", "--only", "EX-5"])
    assert code == 0
    output = capsys.readouterr().out
    assert "scripts/evaluate.py" not in output
    assert "scripts/train_relational_model.py" in output


def test_format_command_keeps_flag_and_value_on_one_line():
    wrapped = format_command(
        [PYTHON_DOC, "scripts/train_shape_segmenter.py", "--profile", DEFAULT_PROFILE, "--no-augment"]
    )
    assert "\\\n    --profile rtx5090 \\\n    --no-augment" in wrapped
