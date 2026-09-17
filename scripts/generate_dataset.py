#!/usr/bin/env python3
"""Generate and validate the synthetic corpus.

Writes, under ``--output-root``::

    scenes/<scene_id>/image.nii.gz       the simulated MRI-like volume (float32)
    scenes/<scene_id>/labels.nii.gz      instance labels, 0 and 1..10 (uint8)
    scenes/<scene_id>/occupancy.nii.gz   binary foreground (uint8)
    scenes/<scene_id>/masks/*.nii.gz     one binary mask per structure
    scenes/<scene_id>/examples/*         per-example target, anchor stack, prompt
    scenes/<scene_id>/scene.json         seeds, placed parameters, appearance draws
    manifests/<split>.jsonl      one ExampleMetadata per retained example
    manifests/<split>_candidates.jsonl   all ten candidates (--keep-all-candidates)
    run_metadata.json            configs, seeds, versions, git revision, hardware,
                                 exact counts, appearance statistics and the
                                 rejection log

With ``storage.scene_array_format: npz_compressed`` a scene is instead the
single ``scenes/<scene_id>.npz`` file of the previous milestone.

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
    scene_record,
)
from src.data.scene_io import (  # noqa: E402
    BIAS_FILE,
    BODY_FILE,
    FRACTION_FILE,
    NIFTI_FORMAT,
    NPZ_FORMAT,
    TISSUE_FILE,
    save_example_masks,
    save_scene,
)
from src.data.schema import (  # noqa: E402
    SCHEMA_VERSION,
    ExampleMetadata,
    save_scene_arrays,
    write_manifest,
)
from src.data.validation import RejectionLog, validate_split_assignment  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class StorageOptions:
    """How much of a scene is written to disk, from ``generator.storage``."""

    scene_array_format: str
    write_example_masks: bool
    write_appearance_diagnostics: bool
    image_dtype: str

    @classmethod
    def from_config(cls, config: Any | None = None) -> "StorageOptions":
        storage = dict((config or load_config("generator"))["storage"])
        fmt = str(storage.get("scene_array_format", NIFTI_FORMAT))
        if fmt not in (NIFTI_FORMAT, NPZ_FORMAT):
            raise SystemExit(
                f"storage.scene_array_format must be {NIFTI_FORMAT!r} or {NPZ_FORMAT!r}, "
                f"got {fmt!r}"
            )
        return cls(
            scene_array_format=fmt,
            write_example_masks=bool(storage.get("write_example_masks", True)),
            write_appearance_diagnostics=bool(
                storage.get("write_appearance_diagnostics", False)
            ),
            image_dtype=str(storage.get("image_dtype", "uint16")),
        )


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
    storage: StorageOptions
    appearance: dict[str, Any] = field(default_factory=dict)

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
                "appearance_version": self.settings.appearance.version,
            },
            "storage": {
                "scene_array_format": self.storage.scene_array_format,
                "write_example_masks": self.storage.write_example_masks,
                "write_appearance_diagnostics": self.storage.write_appearance_diagnostics,
                "image_dtype": self.storage.image_dtype,
            },
            "appearance": {
                "enabled": self.settings.appearance.enabled,
                **self.appearance,
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
    scenes, manifests = root / "scenes", root / "manifests"
    existing = [path for path in (scenes, manifests) if path.exists() and any(path.iterdir())]
    if existing and not overwrite:
        raise SystemExit(
            f"{root} already contains a dataset ({', '.join(str(p) for p in existing)}). "
            "Pass --overwrite to replace it."
        )
    for path in (scenes, manifests):
        if path.exists() and overwrite:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def write_scene(
    scene,
    split: str,
    output_root: Path,
    storage: StorageOptions,
    *,
    retained_example_ids: Sequence[str] = (),
) -> None:
    """Write one accepted scene in the configured on-disk format."""
    if storage.scene_array_format == NPZ_FORMAT:
        save_scene_arrays(
            output_root / "scenes" / f"{scene.scene_id}.npz",
            scene.scene_volume,
            scene.instance_labels,
            image=scene.appearance.image if scene.appearance is not None else None,
        )
        return

    diagnostics: dict[str, np.ndarray] = {}
    if storage.write_appearance_diagnostics and scene.appearance is not None:
        diagnostics = {
            TISSUE_FILE: scene.appearance.tissue_image,
            BIAS_FILE: scene.appearance.bias,
            FRACTION_FILE: scene.appearance.structure_fraction,
            BODY_FILE: (scene.appearance.body_fraction >= 0.5).astype(np.uint8),
        }
    save_scene(
        output_root,
        scene.scene_id,
        instance_labels=scene.instance_labels,
        occupancy=scene.scene_volume,
        spacing=scene.spacing,
        image=scene.appearance.image if scene.appearance is not None else None,
        record=scene_record(scene, split=split),
        extra_volumes=diagnostics,
        image_dtype=storage.image_dtype,
    )
    if not storage.write_example_masks:
        return
    wanted = set(retained_example_ids)
    for example in scene.examples:
        metadata = example.metadata
        if wanted and metadata.example_id not in wanted:
            continue
        save_example_masks(
            output_root,
            scene.scene_id,
            metadata.example_id,
            instance_labels=scene.instance_labels,
            target_instance_id=metadata.target_instance_id,
            anchor_instance_ids=metadata.anchor_instance_ids,
            spacing=scene.spacing,
            record=metadata.to_json_dict(),
        )


def build_split(
    name: str,
    seed_range: tuple[int, int],
    target_classes: Sequence[str],
    settings: GeneratorSettings,
    output_root: Path,
    log: RejectionLog,
    *,
    keep_all_candidates: bool,
    storage: StorageOptions,
    appearance_stats: list[dict[str, Any]] | None = None,
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
        scene_retained: list[str] = []
        for example in scene.examples:
            metadata = example.metadata
            report.candidates += 1
            candidates.append(metadata)
            if metadata.target_shape_name in allowed:
                retained.append(metadata)
                scene_retained.append(metadata.example_id)
                report.per_target_counts[metadata.target_shape_name] = (
                    report.per_target_counts.get(metadata.target_shape_name, 0) + 1
                )
        # Every candidate is written when the manifest keeps every candidate,
        # so an inspected batch can open any of the ten.
        write_scene(
            scene,
            name,
            output_root,
            storage,
            retained_example_ids=() if keep_all_candidates else scene_retained,
        )
        report.scenes += 1
        report.scene_attempts += scene.attempts
        if scene.volume_shape != settings.volume_shape:
            report.escalated_scenes += 1
        if appearance_stats is not None and scene.appearance is not None:
            appearance_stats.append(scene.appearance.parameters)
        if verbose and report.scenes % 50 == 0:
            print(f"  [{name}] {report.scenes} scenes, {len(retained)} retained examples")

    report.retained = len(retained)
    write_manifest(output_root / "manifests" / f"{name}.jsonl", retained)
    if keep_all_candidates:
        write_manifest(output_root / "manifests" / f"{name}_candidates.jsonl", candidates)
    return report


def summarize_appearance(draws: Sequence[Any]) -> dict[str, Any]:
    """Aggregate the per-scene appearance draws for ``run_metadata.json``.

    The measured numbers are the interesting ones: they say how hard the corpus
    actually is, rather than what the configuration asked for.
    """
    if not draws:
        return {}

    def spread(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray([v for v in values if v is not None], dtype=np.float64)
        if array.size == 0:
            return {}
        return {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    measured = [draw["measured"] for draw in draws]
    return {
        "scenes": len(draws),
        "structure_polarity": draws[0]["structure_polarity"],
        "class_conditioned_intensities": draws[0]["class_conditioned_intensities"],
        "drawn": {
            "noise_sigma": spread([draw["noise_sigma"] for draw in draws]),
            "kspace_fraction": spread([draw["kspace_fraction"] for draw in draws]),
            "bias_inhomogeneity": spread([draw["bias_inhomogeneity"] for draw in draws]),
            "parenchyma_intensity": spread([draw["parenchyma_intensity"] for draw in draws]),
        },
        "measured": {
            "mean_contrast": spread([entry["mean_contrast"] for entry in measured]),
            "tissue_background_std": spread(
                [entry["tissue_background_std"] for entry in measured]
            ),
            "contrast_to_noise": spread([entry["contrast_to_noise"] for entry in measured]),
            "contrast_to_background_variation": spread(
                [entry["contrast_to_background_variation"] for entry in measured]
            ),
        },
    }


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
    storage = StorageOptions.from_config()
    log = RejectionLog()
    started = time.perf_counter()
    reports: dict[str, SplitReport] = {}
    appearance_stats: list[dict[str, Any]] = []
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
            storage=storage,
            appearance_stats=appearance_stats,
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
        storage=storage,
        appearance=summarize_appearance(appearance_stats),
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
    print(f"scene format       : {run.storage.scene_array_format}")
    appearance = run.appearance
    if not run.settings.appearance.enabled:
        print("appearance         : disabled (binary volumes)")
    elif appearance:
        measured = appearance["measured"]
        drawn = appearance["drawn"]
        print(
            f"appearance         : polarity {appearance['structure_polarity']}, "
            f"class-conditioned {appearance['class_conditioned_intensities']}"
        )
        print(
            f"  contrast         : {measured['mean_contrast']['mean']:.3f} "
            f"[{measured['mean_contrast']['min']:.3f}, {measured['mean_contrast']['max']:.3f}]"
        )
        print(
            f"  noise sigma      : {drawn['noise_sigma']['mean']:.4f} "
            f"[{drawn['noise_sigma']['min']:.4f}, {drawn['noise_sigma']['max']:.4f}]"
        )
        print(
            f"  CNR / contrast-to-background-variation : "
            f"{measured['contrast_to_noise']['mean']:.2f} / "
            f"{measured['contrast_to_background_variation']['mean']:.2f}"
        )


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
