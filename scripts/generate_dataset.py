#!/usr/bin/env python3
"""Generate and validate the synthetic corpus.

Writes, under ``--output-root``::

    scenes/<scene_id>/scene_volume.nii.gz
    scenes/<scene_id>/instance_labels.nii.gz
    examples/<example_id>/target_mask.nii.gz
    examples/<example_id>/anchor_{slot}_{shape}.nii.gz
    examples/<example_id>/anchor_union.nii.gz
    manifests/<split>.jsonl      one ExampleMetadata per retained example
    manifests/<split>_candidates.jsonl   all ten candidates (--keep-all-candidates)
    run_metadata.json            configs, seeds, versions, git revision, hardware,
                                 exact counts and the rejection log

Training rematerialises masks from ``instance_labels``; the example NIfTIs are
for inspection in ITK-SNAP / 3D Slicer.

Scene seeds come from ``configs/split.yaml``; target-class filtering comes from
the same file. Every example is validated before it is written, and a scene that
fails any check is regenerated rather than repaired.

Examples::

    .venv/bin/python scripts/generate_dataset.py --smoke
    .venv/bin/python scripts/generate_dataset.py --splits train val test
    .venv/bin/python scripts/generate_dataset.py --limit 10 --output-root /tmp/try
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.config import load_all_configs, load_config  # noqa: E402
from src.data.primitives import SHAPE_VOCABULARY, VOCABULARY_VERSION  # noqa: E402
from src.data.direction_rules import DIRECTION_RULE_VERSION  # noqa: E402
from src.data.scene_generator import (  # noqa: E402
    GeneratorSettings,
    generate_scene,
)
from src.data.schema import (  # noqa: E402
    SCHEMA_VERSION,
    ExampleMetadata,
    example_array_dir,
    save_example_arrays,
    save_scene_arrays,
    scene_array_dir,
    write_manifest,
)
from src.data.validation import RejectionLog, validate_split_assignment  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitReport:
    """Exact counts for one split, reported after filtering."""

    name: str
    seed_range: tuple[int, int]
    scenes: int = 0
    candidates: int = 0
    retained: int = 0
    target_classes: tuple[str, ...] = ()
    per_target_counts: dict[str, int] = field(default_factory=dict)
    scene_attempts: int = 0
    escalated_scenes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed_range": list(self.seed_range),
            "scenes": self.scenes,
            "candidate_examples": self.candidates,
            "retained_examples": self.retained,
            "target_classes": list(self.target_classes),
            "per_target_counts": dict(sorted(self.per_target_counts.items())),
            "mean_scene_attempts": (
                self.scene_attempts / self.scenes if self.scenes else 0.0
            ),
            "escalated_scenes": self.escalated_scenes,
        }


@dataclass
class RunReport:
    """Everything a run produces, mirrored into ``run_metadata.json``."""

    output_root: Path
    settings: GeneratorSettings
    splits: dict[str, SplitReport]
    rejection_log: RejectionLog
    elapsed_seconds: float
    smoke: bool
    kept_all_candidates: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "smoke": self.smoke,
            "kept_all_candidates": self.kept_all_candidates,
            "output_root": str(self.output_root),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "versions": {
                "generator_version": self.settings.generator_version,
                "direction_rule_version": DIRECTION_RULE_VERSION,
                "vocabulary_version": VOCABULARY_VERSION,
                "schema_version": SCHEMA_VERSION,
            },
            "geometry": {
                "volume_shape": list(self.settings.volume_shape),
                "spacing": list(self.settings.spacing),
                "in_bounds_margin_voxels": self.settings.margin_voxels,
                "escalation_volume_shapes": [
                    list(shape) for shape in self.settings.escalation_volume_shapes
                ],
            },
            "splits": {name: report.to_dict() for name, report in self.splits.items()},
            "totals": {
                "scenes": sum(report.scenes for report in self.splits.values()),
                "candidate_examples": sum(report.candidates for report in self.splits.values()),
                "retained_examples": sum(report.retained for report in self.splits.values()),
            },
            "rejection_log": self.rejection_log.summary(),
            "environment": environment_metadata(),
            "configs": load_all_configs(),
        }


def environment_metadata() -> dict[str, Any]:
    """Git revision, platform and library versions, for reproducibility."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return {
        "git_revision": revision,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def resolve_plan(
    smoke: bool,
    splits: Sequence[str],
    limit: int | None,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[str, ...]], Path, bool]:
    """Resolve seed ranges, target classes, default output root and candidate flag."""
    split_config = load_config("split")
    generator_config = load_config("generator")
    targets = split_config["target_classes"]
    validate_split_assignment(targets["train"], targets["val"], targets["test"])

    if smoke:
        smoke_config = generator_config["smoke"]
        seed_ranges = {
            name: tuple(int(v) for v in smoke_config["scene_seed_ranges"][name])
            for name in splits
        }
        default_root = PROJECT_ROOT / smoke_config["output_root"]
        keep_all = bool(smoke_config.get("keep_all_candidates", False))
    else:
        seed_ranges = {
            name: tuple(int(v) for v in split_config["scene_seed_ranges"][name])
            for name in splits
        }
        default_root = PROJECT_ROOT / generator_config["storage"]["root"]
        keep_all = False

    if limit is not None:
        seed_ranges = {
            name: (start, min(end, start + limit)) for name, (start, end) in seed_ranges.items()
        }

    target_classes = {name: tuple(targets[name]) for name in splits}
    return seed_ranges, target_classes, default_root, keep_all  # type: ignore[return-value]


def prepare_output(root: Path, overwrite: bool) -> None:
    """Create the output tree, refusing to clobber an existing dataset."""
    scenes, examples, manifests = root / "scenes", root / "examples", root / "manifests"
    existing = [path for path in (scenes, examples, manifests) if path.exists() and any(path.iterdir())]
    if existing and not overwrite:
        raise SystemExit(
            f"{root} already contains a dataset ({', '.join(str(p) for p in existing)}). "
            "Pass --overwrite to replace it."
        )
    for path in (scenes, examples, manifests):
        if path.exists() and overwrite:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def build_split(
    name: str,
    seed_range: tuple[int, int],
    target_classes: Sequence[str],
    settings: GeneratorSettings,
    output_root: Path,
    log: RejectionLog,
    *,
    keep_all_candidates: bool,
    verbose: bool = True,
) -> SplitReport:
    """Generate one split: scenes to disk, metadata to the manifests."""
    start, end = seed_range
    allowed = set(SHAPE_VOCABULARY.require_names(target_classes))
    report = SplitReport(name=name, seed_range=seed_range, target_classes=tuple(target_classes))
    retained: list[ExampleMetadata] = []
    candidates: list[ExampleMetadata] = []

    for seed in range(start, end):
        scene = generate_scene(seed, settings=settings, split=name, log=log)
        save_scene_arrays(
            scene_array_dir(output_root, scene.scene_id),
            scene.scene_volume,
            scene.instance_labels,
            scene.spacing,
        )
        report.scenes += 1
        report.scene_attempts += scene.attempts
        if scene.volume_shape != settings.volume_shape:
            report.escalated_scenes += 1
        for example in scene.examples:
            metadata = example.metadata
            report.candidates += 1
            candidates.append(metadata)
            dump = keep_all_candidates or metadata.target_shape_name in allowed
            if dump:
                save_example_arrays(
                    example_array_dir(output_root, metadata.example_id),
                    example.arrays,
                    metadata,
                )
            if metadata.target_shape_name in allowed:
                retained.append(metadata)
                report.per_target_counts[metadata.target_shape_name] = (
                    report.per_target_counts.get(metadata.target_shape_name, 0) + 1
                )
        if verbose and report.scenes % 50 == 0:
            print(f"  [{name}] {report.scenes} scenes, {len(retained)} retained examples")

    report.retained = len(retained)
    write_manifest(output_root / "manifests" / f"{name}.jsonl", retained)
    if keep_all_candidates:
        write_manifest(output_root / "manifests" / f"{name}_candidates.jsonl", candidates)
    return report


def build_dataset(
    *,
    splits: Sequence[str] = SPLIT_NAMES,
    output_root: Path | str | None = None,
    smoke: bool = False,
    limit: int | None = None,
    overwrite: bool = False,
    keep_all_candidates: bool | None = None,
    verbose: bool = True,
) -> RunReport:
    """Generate the corpus and write it under ``output_root``."""
    seed_ranges, target_classes, default_root, keep_all = resolve_plan(smoke, splits, limit)
    if keep_all_candidates is not None:
        keep_all = keep_all_candidates
    root = Path(output_root) if output_root is not None else default_root
    prepare_output(root, overwrite)

    settings = GeneratorSettings.from_config()
    log = RejectionLog()
    started = time.perf_counter()
    reports: dict[str, SplitReport] = {}
    for name in splits:
        if verbose:
            print(f"[{name}] seeds {seed_ranges[name][0]}..{seed_ranges[name][1] - 1}")
        reports[name] = build_split(
            name,
            seed_ranges[name],
            target_classes[name],
            settings,
            root,
            log,
            keep_all_candidates=keep_all,
            verbose=verbose,
        )
    elapsed = time.perf_counter() - started

    run = RunReport(
        output_root=root,
        settings=settings,
        splits=reports,
        rejection_log=log,
        elapsed_seconds=elapsed,
        smoke=smoke,
        kept_all_candidates=keep_all,
    )
    (root / "run_metadata.json").write_text(
        json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def print_report(run: RunReport) -> None:
    """Human-readable summary, including the exact counts after filtering."""
    print()
    print(f"output root        : {run.output_root}")
    print(f"volume shape       : {run.settings.volume_shape}  spacing {run.settings.spacing}")
    print(f"elapsed            : {run.elapsed_seconds:.1f}s")
    print()
    header = f"{'split':<6} {'scenes':>7} {'candidates':>11} {'retained':>9}  target classes"
    print(header)
    print("-" * len(header))
    for name, report in run.splits.items():
        print(
            f"{name:<6} {report.scenes:>7} {report.candidates:>11} {report.retained:>9}  "
            + ", ".join(report.target_classes)
        )
    totals = run.to_dict()["totals"]
    print("-" * len(header))
    print(
        f"{'total':<6} {totals['scenes']:>7} {totals['candidate_examples']:>11} "
        f"{totals['retained_examples']:>9}"
    )
    print()
    summary = run.rejection_log.summary()
    print(
        f"scene acceptance   : {summary['scenes_accepted']}/{summary['scene_attempts']} "
        f"({summary['scene_acceptance_rate']:.1%})"
    )
    print(f"scene rejections   : {summary['scene_rejections'] or '{}'}")
    print(f"object rejections  : {summary['object_rejections'] or '{}'}")
    escalated = sum(report.escalated_scenes for report in run.splits.values())
    if escalated:
        print(f"escalated scenes   : {escalated}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES),
        help="which splits to generate (default: all three)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="destination directory (default: configs/generator.yaml storage.root)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="small end-to-end run using the smoke seed ranges and output root",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of scenes per split",
    )
    parser.add_argument(
        "--keep-all-candidates", dest="keep_all_candidates", action="store_true", default=None,
        help="also write every unfiltered candidate example to <split>_candidates.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = build_dataset(
        splits=args.splits,
        output_root=args.output_root,
        smoke=args.smoke,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_all_candidates=args.keep_all_candidates,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print_report(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
