"""Shared campaign runner for the EXPERIMENTS.md hexagon matrix.

Each appearance arm (realistic, mri-like) is a ``CampaignConfig`` plus a tuple
of ``Experiment`` rows. Training flags that distinguish an arm stay on the CLI;
learning rate, seed and the rest come from ``configs/train.yaml``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.config import load_all_configs
from src.provenance import git_is_dirty, git_revision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "rtx5090"
WANDB_PROJECT_PHASE_A = "relational-3d-vlm-phase-a"
WANDB_PROJECT_PHASE_B = "relational-3d-vlm-phase-b"
STAGE_A_EPOCHS = 50
STAGE_B_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 0
EARLY_STOPPING_MIN_DELTA = 0.005
CAMPAIGN_FILENAME = "campaign.json"
PYTHON_DOC = ".venv/bin/python"


@dataclass(frozen=True)
class Experiment:
    """One hexagon: the CLI that produces its run folder."""

    id: str
    run: str
    model: str                      # "shape" (Phase A) or "relational" (Phase B)
    output: Path                    # relative to the project root
    augment: bool
    epochs: int
    dataset: str
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA
    anchors: str | None = None      # None for Stage A; "oracle" / "predicted" for Stage B
    stage_a_id: str | None = None   # predicted Stage B: which Stage A experiment supplies anchors
    depends_on: tuple[str, ...] = ()

    @property
    def checkpoint(self) -> Path:
        return self.output / "best.pt"

    @property
    def phase(self) -> str:
        return "a" if self.model == "shape" else "b"

    @property
    def wandb_project(self) -> str:
        """Phase A and Phase B are two W&B projects; dataset is a tag, not the split."""
        return WANDB_PROJECT_PHASE_A if self.phase == "a" else WANDB_PROJECT_PHASE_B

    def wandb_tags(self) -> tuple[str, ...]:
        """W&B tags derived from the EXPERIMENTS.md row, not from train.yaml leftovers."""
        tags = [
            self.id,
            self.run,
            f"phase-{self.phase}",
            f"dataset-{self.dataset}",
            f"epochs-{self.epochs}",
            f"patience-{self.early_stopping_patience}",
        ]
        if self.model == "shape":
            tags.append("aug-a" if self.augment else "no-aug-a")
            return tuple(tags)
        tags.append("aug-b" if self.augment else "no-aug-b")
        if self.anchors:
            tags.append(f"anchors-{self.anchors}")
        if self.stage_a_id is not None:
            stage_a = _EXPERIMENT_REGISTRY[self.stage_a_id]
            tags.append(f"stage-a-{stage_a.run}")
            tags.append("aug-a" if stage_a.augment else "no-aug-a")
        return tuple(tags)


_EXPERIMENT_REGISTRY: dict[str, Experiment] = {}


def register_experiments(experiments: Sequence[Experiment]) -> dict[str, Experiment]:
    """Index a campaign's rows and publish them for Stage A tag/checkpoint lookup."""
    by_id: dict[str, Experiment] = {}
    for item in experiments:
        existing = _EXPERIMENT_REGISTRY.get(item.id)
        if existing is not None and existing != item:
            raise ValueError(f"duplicate experiment id {item.id} with a different row")
        _EXPERIMENT_REGISTRY[item.id] = item
        by_id[item.id] = item
    return by_id


@dataclass(frozen=True)
class CampaignConfig:
    """One appearance arm: corpus, ids, and where the campaign log is written."""

    name: str
    dataset: str
    data_root: Path
    campaign_dir: Path
    experiments: tuple[Experiment, ...]
    generate_hint: str
    description: str
    banner: str
    data_root_help: str

    @property
    def by_id(self) -> dict[str, Experiment]:
        return {item.id: item for item in self.experiments}


@dataclass(frozen=True)
class RunContext:
    python: str
    data_root: Path
    profile: str


def _posix(path: Path) -> str:
    return path.as_posix()


def _augment_flag(enabled: bool) -> str:
    return "--augment" if enabled else "--no-augment"


def _schedule_flags(experiment: Experiment) -> list[str]:
    """Epochs, early stopping, W&B project and tags derived from the table row."""
    flags = [
        "--epochs", str(experiment.epochs),
        "--early-stopping-patience", str(experiment.early_stopping_patience),
        "--early-stopping-min-delta", str(experiment.early_stopping_min_delta),
        "--wandb-project", experiment.wandb_project,
    ]
    for tag in experiment.wandb_tags():
        flags.extend(["--wandb-tag", tag])
    return flags


def stage_a_checkpoint(experiment: Experiment, by_id: Mapping[str, Experiment] | None = None) -> Path:
    """Path of the Stage A ``best.pt`` a predicted Stage B run consumes."""
    if experiment.stage_a_id is None:
        raise ValueError(f"{experiment.id} has no Stage A dependency")
    registry = by_id if by_id is not None else _EXPERIMENT_REGISTRY
    return registry[experiment.stage_a_id].checkpoint


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
            *_schedule_flags(experiment),
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
            *_schedule_flags(experiment),
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


def documented_training_command(
    experiment: Experiment,
    *,
    data_root: Path,
    profile: str = DEFAULT_PROFILE,
) -> str:
    """The command written in ``docs/EXPERIMENTS.md`` for this hexagon."""
    ctx = RunContext(python=PYTHON_DOC, data_root=data_root, profile=profile)
    return format_command(training_argv(experiment, ctx))


def documented_evaluate_command(
    experiment: Experiment,
    *,
    data_root: Path,
    profile: str = DEFAULT_PROFILE,
) -> str | None:
    ctx = RunContext(python=PYTHON_DOC, data_root=data_root, profile=profile)
    argv = evaluate_argv(experiment, ctx)
    return format_command(argv) if argv else None


def parse_args(argv: Sequence[str] | None, config: CampaignConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=config.description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+", metavar="EX-N", default=None,
        help="run these ids only (default: Phase A then Phase B)",
    )
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE, choices=("laptop_mps", "rtx5090"),
        help="hardware profile from configs/train.yaml (default: rtx5090)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=config.data_root,
        help=config.data_root_help,
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


def selected_experiments(
    only: Sequence[str] | None,
    by_id: Mapping[str, Experiment],
    *,
    experiments: Sequence[Experiment] | None = None,
) -> list[Experiment]:
    if not only:
        return list(experiments if experiments is not None else by_id.values())
    unknown = [item for item in only if item not in by_id]
    if unknown:
        known = ", ".join(by_id)
        raise SystemExit(f"unknown experiment id(s): {', '.join(unknown)}. Known: {known}")
    return [by_id[item] for item in only]


def resolve_data_root(path: Path) -> Path:
    root = path if path.is_absolute() else PROJECT_ROOT / path
    return root


def require_dataset(data_root: Path, *, generate_hint: str) -> None:
    if not (data_root / "manifests").is_dir():
        raise SystemExit(generate_hint.format(data_root=data_root))


def require_dependency(
    experiment: Experiment,
    by_id: Mapping[str, Experiment],
    *,
    skip_existing: bool,
) -> None:
    for dep_id in experiment.depends_on:
        dep = by_id[dep_id]
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

    def to_dict(self, args: argparse.Namespace, ctx: RunContext, config: CampaignConfig) -> dict:
        return {
            "created_at": self.created_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": config.dataset,
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

    def write(self, args: argparse.Namespace, ctx: RunContext, config: CampaignConfig) -> Path:
        directory = PROJECT_ROOT / config.campaign_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CAMPAIGN_FILENAME
        path.write_text(json.dumps(self.to_dict(args, ctx, config), indent=2, sort_keys=True) + "\n")
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


def run_campaign(argv: Sequence[str] | None, config: CampaignConfig) -> int:
    args = parse_args(argv, config)
    experiments = selected_experiments(
        args.only, config.by_id, experiments=config.experiments
    )
    ctx = RunContext(
        python=sys.executable,
        data_root=args.data_root,
        profile=args.profile,
    )
    if not args.dry_run:
        require_dataset(resolve_data_root(args.data_root), generate_hint=config.generate_hint)

    revision = git_revision()
    dirty = git_is_dirty()
    print("=" * 78)
    print(config.banner)
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
            require_dependency(experiment, config.by_id, skip_existing=args.skip_existing)
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
            return _finish(campaign, args, ctx, config, code)

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
            return _finish(campaign, args, ctx, config, code)

    return _finish(campaign, args, ctx, config, 0)


def _finish(
    campaign: Campaign,
    args: argparse.Namespace,
    ctx: RunContext,
    config: CampaignConfig,
    code: int,
) -> int:
    path = PROJECT_ROOT / config.campaign_dir / CAMPAIGN_FILENAME
    if args.dry_run:
        print()
        print(f"campaign log: {path.relative_to(PROJECT_ROOT)} (not written)")
        return code
    written = campaign.write(args, ctx, config)
    print()
    print(f"campaign log: {written.relative_to(PROJECT_ROOT)}")
    return code
