#!/usr/bin/env python3
"""Run the mri-like experiment matrix (EX-6 … EX-10).

Same Phase A / Phase B split as ``scripts/run_realistic_experiments.py``, against
the ``data/dataset_mri`` corpus. The five hexagons in ``docs/EXPERIMENTS.md``:

    EX-6   Stage A, no aug          dataset_mri_like
    EX-8   Stage A, with aug        dataset_mri_like_aug
    EX-7   Stage B, predicted+augB  pred_dataset_mri_like_augB   ← EX-6
    EX-9   Stage B, predicted+augA+augB  pred_dataset_mri_like_augA_augB  ← EX-8
    EX-10  Stage B, oracle, no aug  oracle_dataset_mri_like

Every flag that distinguishes an arm is on the command line, including the W&B
project, epoch budget and early-stopping rule. Learning rate, seed and the rest
come from ``configs/train.yaml`` at the git revision recorded in each run's
``best.json``. Re-running the same command at the same commit, against the same
corpus (``data/dataset_mri/run_metadata.json``), is the reproduction recipe.

W&B projects stay ``relational-3d-vlm-phase-a`` / ``relational-3d-vlm-phase-b``;
dataset is a tag (``dataset-mri-like``), not a third project.

Examples::

    .venv/bin/python scripts/run_mri_experiments.py
    .venv/bin/python scripts/run_mri_experiments.py --profile laptop_mps
    .venv/bin/python scripts/run_mri_experiments.py --dry-run
    .venv/bin/python scripts/run_mri_experiments.py --only EX-6 EX-10
    .venv/bin/python scripts/run_mri_experiments.py --skip-existing --no-evaluate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.experiment_campaign import (  # noqa: E402
    PYTHON_DOC,
    STAGE_A_EPOCHS,
    STAGE_B_EPOCHS,
    CampaignConfig,
    Experiment,
    RunContext,
    documented_evaluate_command as _documented_evaluate_command,
    documented_training_command as _documented_training_command,
    evaluate_argv,
    format_command,
    register_experiments,
    run_campaign,
    selected_experiments as _selected_experiments,
    training_argv,
)

DEFAULT_DATA_ROOT = Path("data/dataset_mri")
DEFAULT_PROFILE = "rtx5090"
CAMPAIGN_DIR = Path("runs/experiments/mri-like")

__all__ = [
    "CAMPAIGN_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_PROFILE",
    "EXPERIMENTS_BY_ID",
    "MRI_EXPERIMENTS",
    "PYTHON_DOC",
    "Experiment",
    "RunContext",
    "documented_evaluate_command",
    "documented_training_command",
    "evaluate_argv",
    "format_command",
    "main",
    "selected_experiments",
    "training_argv",
]


MRI_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        id="EX-6",
        run="dataset_mri_like",
        model="shape",
        output=Path("runs/shape_segmenter/dataset_mri_like"),
        augment=False,
        epochs=STAGE_A_EPOCHS,
        dataset="mri-like",
    ),
    Experiment(
        id="EX-8",
        run="dataset_mri_like_aug",
        model="shape",
        output=Path("runs/shape_segmenter/dataset_mri_like_aug"),
        augment=True,
        epochs=STAGE_A_EPOCHS,
        dataset="mri-like",
    ),
    Experiment(
        id="EX-7",
        run="pred_dataset_mri_like_augB",
        model="relational",
        output=Path("runs/relational_model/predicted/pred_dataset_mri_like_augB"),
        augment=True,
        epochs=STAGE_B_EPOCHS,
        dataset="mri-like",
        anchors="predicted",
        stage_a_id="EX-6",
        depends_on=("EX-6",),
    ),
    Experiment(
        id="EX-9",
        run="pred_dataset_mri_like_augA_augB",
        model="relational",
        output=Path("runs/relational_model/predicted/pred_dataset_mri_like_augA_augB"),
        augment=True,
        epochs=STAGE_B_EPOCHS,
        dataset="mri-like",
        anchors="predicted",
        stage_a_id="EX-8",
        depends_on=("EX-8",),
    ),
    Experiment(
        id="EX-10",
        run="oracle_dataset_mri_like",
        model="relational",
        output=Path("runs/relational_model/oracle/oracle_dataset_mri_like"),
        augment=False,
        epochs=STAGE_B_EPOCHS,
        dataset="mri-like",
        anchors="oracle",
    ),
)

EXPERIMENTS_BY_ID = register_experiments(MRI_EXPERIMENTS)

CAMPAIGN = CampaignConfig(
    name="mri-like",
    dataset="mri-like",
    data_root=DEFAULT_DATA_ROOT,
    campaign_dir=CAMPAIGN_DIR,
    experiments=MRI_EXPERIMENTS,
    generate_hint=(
        "no mri-like corpus at {data_root}. Expected manifests/ under "
        f"{DEFAULT_DATA_ROOT}."
    ),
    description=__doc__ or "",
    banner="MRI-LIKE EXPERIMENTS",
    data_root_help=f"mri-like corpus (default: {DEFAULT_DATA_ROOT})",
)


def documented_training_command(experiment: Experiment, *, profile: str = DEFAULT_PROFILE) -> str:
    return _documented_training_command(experiment, data_root=DEFAULT_DATA_ROOT, profile=profile)


def documented_evaluate_command(experiment: Experiment, *, profile: str = DEFAULT_PROFILE) -> str | None:
    return _documented_evaluate_command(experiment, data_root=DEFAULT_DATA_ROOT, profile=profile)


def selected_experiments(only: Sequence[str] | None) -> list[Experiment]:
    return _selected_experiments(only, EXPERIMENTS_BY_ID, experiments=MRI_EXPERIMENTS)


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(argv, CAMPAIGN)


if __name__ == "__main__":
    sys.exit(main())
