#!/usr/bin/env python3
"""Generate a small dataset and validate it end to end (Phase 0 data verification).

This is the "inspected batch" step: it regenerates the smoke corpus, re-reads it
from disk as a consumer would, and independently re-derives every claim the
manifest makes. Nothing is taken on trust from the generator.

Checks:

* every manifest record parses, and its prompt round-trips from the structured
  clauses;
* every example revalidates against the full schema contract (exactly ten
  instances, no empty object, in-bounds, target absent from every anchor
  channel, each channel equal to its declared anchor);
* every clause direction is recomputed from the stored masks and must match;
* anchor channels are ordered by ascending centroid distance, in the same order
  as the prompt clauses;
* the target shape name never appears in the prompt;
* coverage: all ten shapes appear as targets, all ten appear as anchors and all
  six directions occur;
* retained target classes match ``configs/split.yaml``, and the smoke seed
  ranges are disjoint from the real corpus.

Usage::

    .venv/bin/python scripts/run_smoke_test.py            # generate, then validate
    .venv/bin/python scripts/run_smoke_test.py --no-generate
    .venv/bin/python scripts/run_smoke_test.py --show 5   # more inspected examples
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from generate_dataset import SPLIT_NAMES, build_dataset, print_report  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.direction_rules import (  # noqa: E402
    DIRECTION_RULE_VERSION,
    DIRECTIONS,
    centroid_world,
)
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY, VOCABULARY_VERSION  # noqa: E402
from src.data.prompt_generator import assert_no_target_leakage  # noqa: E402
from src.data.schema import (  # noqa: E402
    SCHEMA_VERSION,
    Example,
    ExampleArrays,
    load_scene_arrays,
    read_manifest,
)
from src.data.validation import (  # noqa: E402
    validate_example,
    verify_relations_against_masks,
)


class SmokeFailure(AssertionError):
    """Raised when the generated dataset violates a Phase 0 guarantee."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def load_examples(root: Path, manifest: Path, margin_voxels: int) -> list[Example]:
    """Re-read a manifest and rebuild every example from the scene arrays on disk."""
    scene_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    examples: list[Example] = []
    for metadata in read_manifest(manifest):
        if metadata.scene_id not in scene_cache:
            scene_cache[metadata.scene_id] = load_scene_arrays(
                root / "scenes" / f"{metadata.scene_id}.npz"
            )
        scene_volume, instance_labels = scene_cache[metadata.scene_id]
        example = Example(
            metadata=metadata,
            arrays=ExampleArrays.from_scene(metadata, scene_volume, instance_labels),
        )
        validate_example(example, margin_voxels=margin_voxels)
        verify_relations_against_masks(example)
        assert_no_target_leakage(metadata.prompt, metadata.target_shape_name)
        examples.append(example)
    return examples


def check_conditioning_channels(example: Example) -> None:
    """The target must not be present anywhere in the Stage B conditioning input."""
    target = np.asarray(example.arrays.target_mask, dtype=bool)
    for slot in range(example.arrays.anchor_masks.shape[0]):
        channel = np.asarray(example.arrays.anchor_masks[slot], dtype=bool)
        overlap = int((channel & target).sum())
        check(
            overlap == 0,
            f"{example.metadata.example_id}: target leaks into anchor channel {slot} "
            f"({overlap} voxels)",
        )
    union = np.asarray(example.arrays.anchor_union_mask, dtype=bool)
    check(
        int((union & target).sum()) == 0,
        f"{example.metadata.example_id}: target leaks into the anchor union mask",
    )


def describe_example(example: Example) -> str:
    """One inspected example: prompt, ordered channels and recomputed geometry."""
    metadata = example.metadata
    spacing = metadata.spacing
    target_centroid = centroid_world(example.arrays.target_mask, spacing)
    lines = [
        f"  {metadata.example_id}  (split={metadata.split}, seed={metadata.seed})",
        f"    target      : {metadata.target_shape_name} "
        f"(instance {metadata.target_instance_id}, "
        f"{int(example.arrays.target_mask.sum())} voxels, "
        f"centroid {np.round(target_centroid, 2).tolist()})",
        f"    prompt      : {metadata.prompt}",
        f"    structured  : {metadata.structured_prompt.to_list()}",
    ]
    for slot, relation in enumerate(metadata.relations):
        channel = example.arrays.anchor_masks[slot]
        centroid = centroid_world(channel, spacing)
        distance = float(np.linalg.norm(centroid - target_centroid))
        lines.append(
            f"    channel {slot}   : {relation['anchor']:<17} "
            f"instance {metadata.anchor_instance_ids[slot]:>2}  "
            f"direction {relation['direction']:<9} "
            f"d={distance:6.2f}  voxels={int(channel.sum()):>4}  "
            f"centroid={np.round(centroid, 2).tolist()}"
        )
    return "\n".join(lines)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=None, help="dataset directory")
    parser.add_argument(
        "--no-generate", dest="generate", action="store_false",
        help="validate an existing smoke dataset instead of regenerating it",
    )
    parser.add_argument("--show", type=int, default=3, help="inspected examples to print")
    args = parser.parse_args(argv)

    generator_config = load_config("generator")
    split_config = load_config("split")
    smoke_config = generator_config["smoke"]
    margin = int(generator_config["geometry"]["in_bounds_margin_voxels"])
    root = args.output_root or (PROJECT_ROOT / smoke_config["output_root"])

    print("=" * 78)
    print("PHASE 0 SMOKE RUN")
    print("=" * 78)

    if args.generate:
        run_report = build_dataset(smoke=True, output_root=args.output_root, overwrite=True)
        print_report(run_report)
    check(root.is_dir(), f"no dataset at {root}; run without --no-generate first")

    metadata_path = root / "run_metadata.json"
    check(metadata_path.is_file(), f"missing {metadata_path}")
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    versions = run_metadata["versions"]
    check(versions["direction_rule_version"] == DIRECTION_RULE_VERSION, "direction rule version drift")
    check(versions["vocabulary_version"] == VOCABULARY_VERSION, "vocabulary version drift")
    check(versions["schema_version"] == SCHEMA_VERSION, "schema version drift")
    check(run_metadata["environment"]["platform"] != "", "missing hardware metadata")

    print()
    print("-" * 78)
    print("VALIDATING EVERY EXAMPLE FROM DISK")
    print("-" * 78)

    target_classes = split_config["target_classes"]
    all_candidates: list[Example] = []
    inspected: list[Example] = []
    total_retained = 0

    for split in SPLIT_NAMES:
        manifest = root / "manifests" / f"{split}.jsonl"
        check(manifest.is_file(), f"missing manifest {manifest}")
        retained = load_examples(root, manifest, margin)
        for example in retained:
            check_conditioning_channels(example)
            check(
                example.metadata.target_shape_name in target_classes[split],
                f"{example.metadata.example_id}: target "
                f"{example.metadata.target_shape_name!r} is not a {split} target class",
            )
            check(example.metadata.split == split, "split tag mismatch")
        total_retained += len(retained)

        candidates_path = root / "manifests" / f"{split}_candidates.jsonl"
        if candidates_path.is_file():
            candidates = load_examples(root, candidates_path, margin)
            for example in candidates:
                check_conditioning_channels(example)
            all_candidates.extend(candidates)
        else:  # pragma: no cover - only when --keep-all-candidates is off
            all_candidates.extend(retained)

        inspected.extend(retained[:1])
        print(f"  {split:<6} {len(retained):>4} retained examples validated")

    print(f"  {'total':<6} {total_retained:>4} retained examples validated")
    print(f"  {'':<6} {len(all_candidates):>4} candidate examples validated")

    # -- coverage -----------------------------------------------------------
    print()
    print("-" * 78)
    print("COVERAGE")
    print("-" * 78)
    target_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    slot_directions: dict[int, Counter[str]] = defaultdict(Counter)
    for example in all_candidates:
        target_counts[example.metadata.target_shape_name] += 1
        for slot, relation in enumerate(example.metadata.relations):
            anchor_counts[relation["anchor"]] += 1
            direction_counts[relation["direction"]] += 1
            slot_directions[slot][relation["direction"]] += 1

    missing_targets = [name for name in SHAPE_NAMES if not target_counts[name]]
    missing_anchors = [name for name in SHAPE_NAMES if not anchor_counts[name]]
    missing_directions = [name for name in DIRECTIONS if not direction_counts[name]]
    check(not missing_targets, f"shapes never used as a target: {missing_targets}")
    check(not missing_anchors, f"shapes never used as an anchor: {missing_anchors}")
    check(not missing_directions, f"directions never generated: {missing_directions}")
    print(f"  targets   : all {len(SHAPE_NAMES)} shapes  {dict(sorted(target_counts.items()))}")
    print(f"  anchors   : all {len(SHAPE_NAMES)} shapes  {dict(sorted(anchor_counts.items()))}")
    print(f"  directions: all {len(DIRECTIONS)}         {dict(sorted(direction_counts.items()))}")
    for slot in sorted(slot_directions):
        print(f"  slot {slot}    : {dict(sorted(slot_directions[slot].items()))}")

    # -- measured geometry --------------------------------------------------
    print()
    print("-" * 78)
    print("MEASURED GEOMETRY PER CLASS (design band: 8-22% of the axis)")
    print("-" * 78)
    low, high = SHAPE_VOCABULARY.axis_extent_fraction_range
    scene_paths = sorted((root / "scenes").glob("*.npz"))
    voxel_counts: dict[str, list[int]] = defaultdict(list)
    extent_fractions: dict[str, list[float]] = defaultdict(list)
    for path in scene_paths:
        _, labels = load_scene_arrays(path)
        axis_length = max(labels.shape)
        for spec in SHAPE_VOCABULARY:
            mask = labels == spec.id
            voxel_counts[spec.name].append(int(mask.sum()))
            indices = np.nonzero(mask)
            spans = [int(idx.max() - idx.min() + 1) for idx in indices]  # (z, y, x)
            for axis_name, span in zip(("z", "y", "x"), spans):
                if axis_name == spec.thin_axis_exception:
                    continue
                extent_fractions[spec.name].append(span / axis_length)
    print(f"  {'shape':<18} {'voxels mean':>12} {'min ext':>8} {'max ext':>8}  band")
    for spec in SHAPE_VOCABULARY:
        counts = voxel_counts[spec.name]
        fractions = extent_fractions[spec.name]
        smallest, largest = min(fractions), max(fractions)
        status = "ok" if smallest >= low - 0.02 and largest <= high + 0.02 else "OUT OF BAND"
        note = f" (thin axis {spec.thin_axis_exception} excluded)" if spec.thin_axis_exception else ""
        print(
            f"  {spec.name:<18} {np.mean(counts):>12.0f} {smallest:>8.3f} {largest:>8.3f}  {status}{note}"
        )
    means = [float(np.mean(voxel_counts[spec.name])) for spec in SHAPE_VOCABULARY]
    print(f"  occupied volumes are comparable: max/min of class means = {max(means) / min(means):.2f}")

    # -- inspected batch ----------------------------------------------------
    print()
    print("-" * 78)
    print("INSPECTED EXAMPLES")
    print("-" * 78)
    for example in inspected[: args.show]:
        print(describe_example(example))
        print()

    print("=" * 78)
    print(
        f"SMOKE RUN PASSED - {total_retained} retained and {len(all_candidates)} candidate "
        f"examples validated from {len(scene_paths)} scenes"
    )
    print("=" * 78)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except SmokeFailure as error:
        print(f"\nSMOKE RUN FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
