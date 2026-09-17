#!/usr/bin/env python3
"""Phase 4: run a chosen Stage A / Stage B pair on a split and keep the masks.

Pick the two checkpoints, point at a split (``test`` by default), and every
prediction is written next to the target it was scored against, in a folder that
records which weights produced it::

    .venv/bin/python scripts/evaluate.py \
        --stage-b-checkpoint runs/relational_model/predicted/dataset_custom/best.pt \
        --stage-a-checkpoint runs/shape_segmenter/shapeSeg_dataset_custom/best.pt

    predictions/<stage_b name>__anchors-predicted-<stage_a name>__test/
      run_metadata.json          both checkpoint paths, both run names, provenance
      metrics.json / metrics.txt aggregate + stratified Dice / IoU / Hausdorff
      predictions.jsonl          one row per example: prompt, scores, file paths
      predictions/<example_id>/prediction_mask.nii.gz, target_mask.nii.gz

The model *name* defaults to the run directory the checkpoint sits in - every
run writes a ``best.pt``, so the directory is what tells two of them apart - and
``--stage-a-name`` / ``--stage-b-name`` override it.

Anchor source
-------------
``--anchor-source predicted`` (the default) is the end-to-end setting: Stage A
segments the three named anchors from the scene and Stage B consumes those
channels. The run then also reports the anchor masks' own Dice/IoU, so a drop
against the oracle run can be attributed to Stage A rather than guessed at.
``--anchor-source oracle`` uses the ground-truth channels instead and needs no
Stage A checkpoint; run both to get the Phase 4 oracle-versus-predicted delta.

Still reserved for later: the four-way baseline sweep and the full counterfactual
battery (``src/evaluation/counterfactuals.py``). The occupancy-sanity subset runs
today from ``scripts/train_relational_model.py``.

Examples::

    # test set, end-to-end anchors, probabilities kept as well
    .venv/bin/python scripts/evaluate.py \
        --stage-b-checkpoint runs/relational_model/predicted/dataset_custom/best.pt \
        --stage-a-checkpoint runs/shape_segmenter/shapeSeg_dataset_custom/best.pt \
        --save-probabilities

    # the same weights on ground-truth anchors, for the delta
    .venv/bin/python scripts/evaluate.py --anchor-source oracle \
        --stage-b-checkpoint runs/relational_model/predicted/dataset_custom/best.pt

    # smoke corpus, metrics only
    .venv/bin/python scripts/evaluate.py --smoke --no-volumes \
        --stage-b-checkpoint runs/stage_b_oracle_smoke/best.pt --anchor-source oracle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.dataset import ExampleDataset, build_example_dataloader  # noqa: E402
from src.evaluation.predictions import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    PredictionError,
    default_run_name,
    describe_checkpoint,
    run_prediction,
    write_prediction_folder,
)
from src.models.anchor_provider import ANCHOR_SOURCES, build_anchor_provider  # noqa: E402
from src.models.relational_vlm import VARIANTS, load_relational_vlm  # noqa: E402
from src.training.trainer import resolve_device  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    checkpoints = parser.add_argument_group("checkpoints")
    checkpoints.add_argument(
        "--stage-b-checkpoint", type=Path, required=True,
        help="Stage B (relational) weights, e.g. runs/relational_model/oracle/<run>/best.pt",
    )
    checkpoints.add_argument(
        "--stage-a-checkpoint", type=Path, default=None,
        help="Stage A (shape segmenter) weights; required unless --anchor-source oracle",
    )
    checkpoints.add_argument(
        "--stage-a-name", default=None,
        help="name recorded for Stage A (default: the checkpoint's run directory)",
    )
    checkpoints.add_argument(
        "--stage-b-name", default=None,
        help="name recorded for Stage B (default: the checkpoint's run directory)",
    )
    checkpoints.add_argument(
        "--variant", default=None, choices=VARIANTS,
        help="Stage B variant (default: whatever the checkpoint was trained as)",
    )
    checkpoints.add_argument(
        "--model-profile", default=None, choices=("default", "smoke"),
        help="override the model width profile (default: the checkpoint's own)",
    )
    checkpoints.add_argument(
        "--no-checksum", dest="checksum", action="store_false",
        help="skip the SHA-256 of each checkpoint",
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--split", default="test", choices=("train", "val", "test"),
        help="split to evaluate (default: test)",
    )
    data.add_argument(
        "--data-root", type=Path, default=None,
        help="dataset directory (default: data/smoke with --smoke, else data/processed)",
    )
    data.add_argument("--smoke", action="store_true", help="use the smoke corpus")
    data.add_argument(
        "--limit-examples", type=int, default=None, help="cap the examples evaluated",
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--anchor-source", default="predicted", choices=ANCHOR_SOURCES,
        help="predicted = Stage A's masks (default); oracle = ground-truth channels",
    )
    run.add_argument(
        "--anchor-threshold", type=float, default=0.5,
        help="probability cut applied to Stage A's anchor masks",
    )
    run.add_argument(
        "--threshold", type=float, default=0.5,
        help="probability cut applied to Stage B's target mask",
    )
    run.add_argument(
        "--hausdorff-percentile", type=float, default=None,
        help="report the Nth-percentile Hausdorff distance (e.g. 95) instead of the max",
    )
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--device", default=None, help="cpu, mps, cuda or auto")

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output", type=Path, default=None,
        help="prediction folder (default: predictions/<stage_b>__anchors-<...>__<split>)",
    )
    output.add_argument(
        "--name", default=None, help="folder name under --output-root",
    )
    output.add_argument(
        "--output-root", type=Path, default=None,
        help=f"parent of the prediction folder (default: {DEFAULT_OUTPUT_ROOT}/)",
    )
    output.add_argument(
        "--overwrite", action="store_true",
        help="write into an existing prediction folder instead of refusing",
    )
    output.add_argument(
        "--no-volumes", dest="save_volumes", action="store_false",
        help="score only; write no NIfTI volumes",
    )
    output.add_argument(
        "--no-targets", dest="save_targets", action="store_false",
        help="do not copy each target mask next to its prediction",
    )
    output.add_argument(
        "--save-probabilities", action="store_true",
        help="also write the float32 probability volume per example",
    )
    output.add_argument(
        "--save-anchors", action="store_true",
        help="also write the three anchor channels Stage B actually consumed",
    )
    output.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def resolve_data_root(args: argparse.Namespace) -> Path:
    generator_config = load_config("generator")
    default_root = (
        PROJECT_ROOT / generator_config["smoke"]["output_root"]
        if args.smoke
        else PROJECT_ROOT / generator_config["storage"]["root"]
    )
    root = args.data_root or default_root
    if not (root / "manifests").is_dir():
        raise SystemExit(
            f"no dataset at {root}. Generate one first:\n"
            f"  .venv/bin/python scripts/generate_dataset.py"
            f"{' --smoke' if args.smoke else ''}"
        )
    return root


def resolve_output_dir(args: argparse.Namespace, name: str) -> Path:
    """Pick the prediction folder and refuse to clobber an existing one."""
    if args.output is not None:
        output_dir = args.output
    else:
        root = args.output_root or (PROJECT_ROOT / DEFAULT_OUTPUT_ROOT)
        output_dir = root / (args.name or name)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"{output_dir} already exists and is not empty. Pass --overwrite to write "
            f"into it, or choose another folder with --name / --output."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet
    command = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]

    if args.anchor_source == "predicted" and args.stage_a_checkpoint is None:
        raise SystemExit(
            "--anchor-source predicted needs --stage-a-checkpoint (Phase 1 weights); "
            "pass --anchor-source oracle to score the ground-truth channels instead"
        )

    data_root = resolve_data_root(args)
    try:
        stage_b_identity = describe_checkpoint(
            args.stage_b_checkpoint,
            role="stage_b",
            name=args.stage_b_name,
            checksum=args.checksum,
        )
        stage_a_identity = (
            describe_checkpoint(
                args.stage_a_checkpoint,
                role="stage_a",
                name=args.stage_a_name,
                checksum=args.checksum,
            )
            if args.stage_a_checkpoint is not None
            else None
        )
    except PredictionError as error:
        raise SystemExit(str(error)) from error

    output_dir = resolve_output_dir(
        args,
        default_run_name(stage_b_identity, stage_a_identity, args.split, args.anchor_source),
    )

    dataset = ExampleDataset(
        data_root,
        args.split,
        limit=args.limit_examples,
        include_scene_volume=args.anchor_source == "predicted",
    )
    loader = build_example_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    device = resolve_device(args.device or "auto")
    model = load_relational_vlm(
        args.stage_b_checkpoint,
        device=device,
        profile=args.model_profile,
        variant=args.variant,
        input_resolution=max(dataset.volume_shape),
        spacing=dataset.spacing,
    )
    anchor_provider = build_anchor_provider(
        args.anchor_source,
        checkpoint=args.stage_a_checkpoint,
        device=device,
        threshold=args.anchor_threshold,
        input_resolution=max(dataset.volume_shape),
    )

    if verbose:
        print("=" * 78)
        print(f"PHASE 4 - EVALUATION on {args.split} ({args.anchor_source} anchors)")
        print("=" * 78)
        print(f"data root   : {data_root}")
        print(f"{args.split:<12}: {dataset.describe()}")
        print(f"stage A     : {stage_a_identity.name if stage_a_identity else '-'}"
              f" <- {args.stage_a_checkpoint or 'not used (oracle anchors)'}")
        print(f"stage B     : {stage_b_identity.name} <- {args.stage_b_checkpoint}")
        print(f"output      : {output_dir}")
        print(f"device      : {device}  threshold {args.threshold}")
        print()

    report = run_prediction(
        model,
        loader,
        output_dir=output_dir,
        anchor_provider=anchor_provider,
        spacing=dataset.spacing,
        device=device,
        threshold=args.threshold,
        hausdorff_percentile=args.hausdorff_percentile,
        save_volumes=args.save_volumes,
        save_targets=args.save_targets,
        save_probabilities=args.save_probabilities,
        save_anchors=args.save_anchors,
        verbose=verbose,
    )

    written = write_prediction_folder(
        output_dir,
        report=report,
        stage_b=stage_b_identity,
        stage_a=stage_a_identity,
        split=args.split,
        data_root=data_root,
        anchor_source=args.anchor_source,
        settings={
            "threshold": args.threshold,
            "anchor_threshold": args.anchor_threshold,
            "hausdorff_percentile": args.hausdorff_percentile,
            "batch_size": args.batch_size,
            "device": str(device),
            "limit_examples": args.limit_examples,
            "variant": args.variant,
            "model_profile": args.model_profile,
            "saved": {
                "volumes": args.save_volumes,
                "targets": args.save_targets and args.save_volumes,
                "probabilities": args.save_probabilities and args.save_volumes,
                "anchors": args.save_anchors and args.save_volumes,
            },
        },
        command=command,
    )

    if verbose:
        print()
        print("-" * 78)
        print(
            f"{args.split.upper()} - stage B {stage_b_identity.name}, anchors "
            f"{args.anchor_source}"
            + (f" from stage A {stage_a_identity.name}" if stage_a_identity else "")
        )
        print("-" * 78)
        print(report["table"])
        if "anchor_quality" in report:
            quality = report["anchor_quality"]
            print(
                f"\n  anchor masks: dice {quality['anchor_dice']:.4f} "
                f"iou {quality['anchor_iou']:.4f} "
                f"empty {quality['empty_anchor_fraction']:.1%}"
            )
        print()
        print(f"predictions : {output_dir}")
        for key in ("run_metadata", "metrics", "metrics_table", "rows"):
            print(f"{key:<12}: {written[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
