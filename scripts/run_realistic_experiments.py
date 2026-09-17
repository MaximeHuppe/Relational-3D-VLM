#!/usr/bin/env python3
"""Run the realistic-appearance experiment matrix (EX-1 … EX-5).

The five hexagons in ``docs/EXPERIMENTS.md`` / ``docs/experiments/experiment_flowchart.drawio``:

    EX-1  Stage A, no aug          dataset_realistic
    EX-2  Stage B, predicted+augB  pred_dataset_realistic_augB   ← EX-1
    EX-3  Stage A, with aug        dataset_realistic_aug
    EX-4  Stage B, predicted+augA+augB  pred_dataset_realistic_augA_augB  ← EX-3
    EX-5  Stage B, oracle, no aug  oracle_dataset_realistic

Every flag that distinguishes an arm is on the command line. Epochs, learning
rate, seed, early stopping and the rest come from ``configs/train.yaml`` at the
git revision recorded in each run's ``best.json``. Re-running the same command
at the same commit, against the same corpus (``data/processed/run_metadata.json``),
is the reproduction recipe.

Examples::

    .venv/bin/python scripts/run_realistic_experiments.py
    .venv/bin/python scripts/run_realistic_experiments.py --profile laptop_mps
    .venv/bin/python scripts/run_realistic_experiments.py --dry-run
    .venv/bin/python scripts/run_realistic_experiments.py --only EX-1 EX-5
    .venv/bin/python scripts/run_realistic_experiments.py --skip-existing --no-evaluate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_all_configs  # noqa: E402
from src.provenance import git_is_dirty, git_revision  # noqa: E402

DEFAULT_DATA_ROOT = Path("data/processed")
DEFAULT_PROFILE = "rtx5090"
CAMPAIGN_DIR = Path("runs/experiments/realistic-appearance")
CAMPAIGN_FILENAME = "campaign.json"

PYTHON_DOC = ".venv/bin/python"


@dataclass(frozen=True)
class Experiment:
    """One hexagon: the CLI that produces its run folder."""

    id: str
    run: str
    model: str                      # "shape" or "relational"
    output: Path                    # relative to the project root
    augment: bool
    anchors: str | None = None      # None for Stage A; "oracle" / "predicted" for Stage B
    stage_a_id: str | None = None   # predicted Stage B: which Stage A experiment supplies anchors
    depends_on: tuple[str, ...] = ()

    @property
    def checkpoint(self) -> Path:
        return self.output / "best.pt"


# Table order in docs/EXPERIMENTS.md. EX-2 needs EX-1; EX-4 needs EX-3.
REALISTIC_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        id="EX-1",
        run="dataset_realistic",
        model="shape",
        output=Path("runs/shape_segmenter/dataset_realistic"),
        augment=False,
    ),
    Experiment(
        id="EX-2",
        run="pred_dataset_realistic_augB",
        model="relational",
        output=Path("runs/relational_model/predicted/pred_dataset_realistic_augB"),
        augment=True,
        anchors="predicted",
        stage_a_id="EX-1",
        depends_on=("EX-1",),
    ),
    Experiment(
        id="EX-3",
        run="dataset_realistic_aug",
        model="shape",
        output=Path("runs/shape_segmenter/dataset_realistic_aug"),
        augment=True,
    ),
    Experiment(
        id="EX-4",
        run="pred_dataset_realistic_augA_augB",
        model="relational",
        output=Path("runs/relational_model/predicted/pred_dataset_realistic_augA_augB"),
        augment=True,
        anchors="predicted",
        stage_a_id="EX-3",
        depends_on=("EX-3",),
    ),
    Experiment(
        id="EX-5",
        run="oracle_dataset_realistic",
        model="relational",
        output=Path("runs/relational_model/oracle/oracle_dataset_realistic"),
        augment=False,
        anchors="oracle",
    ),
)

EXPERIMENTS_BY_ID = {item.id: item for item in REALISTIC_EXPERIMENTS}


@dataclass(frozen=True)
class RunContext:
    python: str
    data_root: Path
    profile: str


def _posix(path: Path) -> str:
    return path.as_posix()


def _augment_flag(enabled: bool) -> str:
    return "--augment" if enabled else "--no-augment"


def stage_a_checkpoint(experiment: Experiment) -> Path:
    """Path of the Stage A ``best.pt`` a predicted Stage B run consumes."""
    if experiment.stage_a_id is None:
        raise ValueError(f"{experiment.id} has no Stage A dependency")
    return EXPERIMENTS_BY_ID[experiment.stage_a_id].checkpoint


def training_argv(experiment: Experiment, ctx: RunContext) -> list[str]:
    """Exact argv for one training hexagon, including the interpreter."""
    data_root = _posix(ctx.data_root)
    output = _posix(experiment.output)
    if experiment.model == "shape":
        return [
            ctx.python, "scripts/train_shape_segmenter.py",
            "--data-root", data_root,
            "--output", output,
            "--profile", ctx.profile,
            _augment_flag(experiment.augment),
        ]
    argv = [
        ctx.python, "scripts/train_relational_model.py",
        "--phase", "oracle",
        "--anchor-source", experiment.anchors or "oracle",
    ]
    if experiment.anchors == "predicted":
        argv.extend(["--stage-a-checkpoint", _posix(stage_a_checkpoint(experiment))])
    argv.extend(
        [
            "--data-root", data_root,
            "--output", output,
            "--profile", ctx.profile,
            _augment_flag(experiment.augment),
        ]
    )
    return argv


def evaluate_argv(experiment: Experiment, ctx: RunContext) -> list[str] | None:
    """Test-set dump for a Stage B run; ``None`` for Stage A."""
    if experiment.model != "relational":
        return None
    argv = [
        ctx.python, "scripts/evaluate.py",
        "--stage-b-checkpoint", _posix(experiment.checkpoint),
    ]
    if experiment.anchors == "predicted":
        argv.extend(["--stage-a-checkpoint", _posix(stage_a_checkpoint(experiment))])
    argv.extend(
        [
            "--data-root", _posix(ctx.data_root),
            "--split", "test",
            "--anchor-source", experiment.anchors or "oracle",
        ]
    )
    return argv


def format_command(argv: Sequence[str], *, python_alias: str | None = None) -> str:
    """Copy-pasteable multi-line bash, with ``\\`` continuations after the script."""
    parts = list(argv)
    if python_alias:
        parts[0] = python_alias
    if len(parts) < 3:
        return " ".join(parts)
    lines = [f"{parts[0]} {parts[1]} \\"]
    rest = parts[2:]
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("--") and index + 1 < len(rest) and not rest[index + 1].startswith("--"):
            chunk = f"    {token} {rest[index + 1]}"
            index += 2
        else:
            chunk = f"    {token}"
            index += 1
        lines.append(chunk + (" \\" if index < len(rest) else ""))
    return "\n".join(lines)


def documented_training_command(experiment: Experiment, *, profile: str = DEFAULT_PROFILE) -> str:
    """The command written in ``docs/EXPERIMENTS.md`` for this hexagon."""
    ctx = RunContext(python=PYTHON_DOC, data_root=DEFAULT_DATA_ROOT, profile=profile)
    return format_command(training_argv(experiment, ctx))


def documented_evaluate_command(experiment: Experiment, *, profile: str = DEFAULT_PROFILE) -> str | None:
    ctx = RunContext(python=PYTHON_DOC, data_root=DEFAULT_DATA_ROOT, profile=profile)
    argv = evaluate_argv(experiment, ctx)
    return format_command(argv) if argv else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+", metavar="EX-N", default=None,
        help="run these ids only (default: EX-1 … EX-5 in table order)",
    )
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE, choices=("laptop_mps", "rtx5090"),
        help="hardware profile from configs/train.yaml (default: rtx5090)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=f"realistic corpus (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip an experiment whose output/best.pt already exists",
    )
    parser.add_argument(
        "--no-evaluate", dest="evaluate", action="store_false",
        help="train only; do not dump test-set predictions after Stage B",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the commands and exit without running them",
    )
    parser.add_argument(
        "--overwrite-eval", action="store_true",
        help="pass --overwrite to evaluate.py if a prediction folder already exists",
    )
    return parser.parse_args(argv)


def selected_experiments(only: Sequence[str] | None) -> list[Experiment]:
    if not only:
        return list(REALISTIC_EXPERIMENTS)
    unknown = [item for item in only if item not in EXPERIMENTS_BY_ID]
    if unknown:
        known = ", ".join(EXPERIMENTS_BY_ID)
        raise SystemExit(f"unknown experiment id(s): {', '.join(unknown)}. Known: {known}")
    return [EXPERIMENTS_BY_ID[item] for item in only]


def resolve_data_root(path: Path) -> Path:
    root = path if path.is_absolute() else PROJECT_ROOT / path
    return root


def require_dataset(data_root: Path) -> None:
    if not (data_root / "manifests").is_dir():
        raise SystemExit(
            f"no realistic corpus at {data_root}. Generate one first:\n"
            f"  {PYTHON_DOC} scripts/generate_dataset.py --output-root {DEFAULT_DATA_ROOT}"
        )


def require_dependency(experiment: Experiment, *, skip_existing: bool) -> None:
    for dep_id in experiment.depends_on:
        dep = EXPERIMENTS_BY_ID[dep_id]
        if (PROJECT_ROOT / dep.checkpoint).is_file():
            continue
        if skip_existing:
            raise SystemExit(
                f"{experiment.id} needs {dep.checkpoint}, which does not exist. "
                f"Run {dep.id} first, or drop --skip-existing so this campaign trains it."
            )
        raise SystemExit(
            f"{experiment.id} needs {dep.id} ({dep.checkpoint}), which has not been trained yet"
        )


@dataclass
class StepRecord:
    id: str
    kind: str
    argv: list[str]
    status: str
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    skipped_reason: str | None = None


@dataclass
class Campaign:
    records: list[StepRecord] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self, args: argparse.Namespace, ctx: RunContext) -> dict:
        return {
            "created_at": self.created_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "realistic",
            "git_revision": git_revision(),
            "git_dirty": git_is_dirty(),
            "runner_command": [sys.argv[0], *sys.argv[1:]],
            "profile": ctx.profile,
            "data_root": str(resolve_data_root(args.data_root)),
            "evaluate": args.evaluate,
            "configs": load_all_configs(),
            "steps": [
                {
                    "id": rec.id,
                    "kind": rec.kind,
                    "command": rec.argv,
                    "status": rec.status,
                    "returncode": rec.returncode,
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                    "skipped_reason": rec.skipped_reason,
                }
                for rec in self.records
            ],
        }

    def write(self, args: argparse.Namespace, ctx: RunContext) -> Path:
        directory = PROJECT_ROOT / CAMPAIGN_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CAMPAIGN_FILENAME
        path.write_text(json.dumps(self.to_dict(args, ctx), indent=2, sort_keys=True) + "\n")
        return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_step(argv: Sequence[str], *, dry_run: bool) -> int:
    print()
    print(format_command(argv))
    print()
    if dry_run:
        return 0
    completed = subprocess.run(list(argv), cwd=PROJECT_ROOT)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = selected_experiments(args.only)
    ctx = RunContext(
        python=sys.executable,
        data_root=args.data_root,
        profile=args.profile,
    )
    if not args.dry_run:
        require_dataset(resolve_data_root(args.data_root))

    revision = git_revision()
    dirty = git_is_dirty()
    print("=" * 78)
    print("REALISTIC-APPEARANCE EXPERIMENTS")
    print("=" * 78)
    print(f"git         : {revision or 'unknown'}{' (dirty working tree)' if dirty else ''}")
    print(f"data root   : {args.data_root}")
    print(f"profile     : {args.profile}")
    print(f"experiments : {', '.join(item.id for item in experiments)}")
    print(f"evaluate    : {'yes' if args.evaluate else 'no'}")
    if dirty:
        print()
        print(
            "WARNING: the working tree has uncommitted changes. The recorded git "
            "SHA is not enough to reproduce this run; commit (or stash) first, or "
            "note the diff under docs/EXPERIMENTS.md Notes."
        )
    if args.dry_run:
        print()
        print("dry-run: printing commands only")

    campaign = Campaign()
    for experiment in experiments:
        train_cmd = training_argv(experiment, ctx)
        checkpoint = PROJECT_ROOT / experiment.checkpoint
        print()
        print("-" * 78)
        print(f"{experiment.id}  {experiment.run}  ->  {experiment.output}")
        print("-" * 78)

        if args.skip_existing and checkpoint.is_file() and not args.dry_run:
            reason = f"{experiment.checkpoint} already exists"
            print(f"skip: {reason}")
            campaign.records.append(
                StepRecord(
                    id=experiment.id, kind="train", argv=train_cmd,
                    status="skipped", skipped_reason=reason,
                )
            )
            continue

        if not args.dry_run:
            require_dependency(experiment, skip_existing=args.skip_existing)
        record = StepRecord(
            id=experiment.id, kind="train", argv=train_cmd,
            status="running", started_at=_now(),
        )
        campaign.records.append(record)
        code = run_step(train_cmd, dry_run=args.dry_run)
        record.finished_at = _now()
        record.returncode = code
        record.status = "dry-run" if args.dry_run else ("ok" if code == 0 else "failed")
        if code != 0:
            return _finish(campaign, args, ctx, code)

        eval_cmd = evaluate_argv(experiment, ctx) if args.evaluate else None
        if eval_cmd is None:
            continue
        if args.overwrite_eval:
            eval_cmd = [*eval_cmd, "--overwrite"]
        eval_record = StepRecord(
            id=experiment.id, kind="evaluate", argv=eval_cmd,
            status="running", started_at=_now(),
        )
        campaign.records.append(eval_record)
        code = run_step(eval_cmd, dry_run=args.dry_run)
        eval_record.finished_at = _now()
        eval_record.returncode = code
        eval_record.status = "dry-run" if args.dry_run else ("ok" if code == 0 else "failed")
        if code != 0:
            return _finish(campaign, args, ctx, code)

    return _finish(campaign, args, ctx, 0)


def _finish(campaign: Campaign, args: argparse.Namespace, ctx: RunContext, code: int) -> int:
    path = PROJECT_ROOT / CAMPAIGN_DIR / CAMPAIGN_FILENAME
    if args.dry_run:
        print()
        print(f"campaign log: {path.relative_to(PROJECT_ROOT)} (not written)")
        return code
    written = campaign.write(args, ctx)
    print()
    print(f"campaign log: {written.relative_to(PROJECT_ROOT)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
