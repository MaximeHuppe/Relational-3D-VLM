#!/usr/bin/env python3
"""Phases 2-4: train and evaluate the Stage B relational target segmenter.

One sample is one ``(scene, target)`` example: three ordered anchor-mask
channels, a three-clause prompt and the binary scene occupancy go in, one
target logit volume comes out. The target shape is never an input - the model
has to find the region that satisfies all three relations at once.

Phases
------
``--phase overfit`` (Phase 2)
    Drive the model onto a single scene until its training Dice passes
    ``stage_b_overfit.target_train_dice``. A bug catcher for channel order,
    world coordinates, prompt indices and the decoder; exits non-zero if the
    target is not reached.

``--phase oracle`` (Phase 3)
    Train on the full corpus and evaluate every epoch. Validation targets are
    the two held-out shape classes, so the selection metric measures
    compositional transfer, not fit. This is the primary proof-of-concept
    measurement.

``--eval-only`` (Phase 4 and re-scoring)
    Load a trained checkpoint and only evaluate it.

Anchor source
-------------
Both phases accept either anchor source, and it is the same model either way::

    --anchor-source oracle       ground-truth masks: instance_labels under
                                 data/processed/scenes/<scene_id>/ is loaded
                                 and only the three named anchors are kept
    --anchor-source predicted    Stage A segments the scene and its masks for
                                 those same three names are used instead
                                 (needs --stage-a-checkpoint)

With predicted anchors the run also reports the anchor masks' own Dice/IoU, so a
drop against the oracle run can be attributed to Stage A rather than guessed at.
The intensity scene volume is loaded for Stage A (predicted anchors). Binary
occupancy derived from labels is the Stage B decoder WHAT stream. Instance
labels never reach Stage B.

After oracle training, overfit, and ``--eval-only``, the run also prints an
occupancy sanity report (prompt permutation / direction flip with occupancy
held fixed, train vs val localisation, one remaining object vs the union)
unless ``--skip-occupancy-sanity`` is set.

Examples::

    .venv/bin/python scripts/train_relational_model.py --phase overfit --smoke
    .venv/bin/python scripts/train_relational_model.py --phase oracle --smoke
    .venv/bin/python scripts/train_relational_model.py --phase oracle
    .venv/bin/python scripts/train_relational_model.py --eval-only \
        --checkpoint runs/stage_b_oracle/best.pt \
        --anchor-source predicted --stage-a-checkpoint runs/stage_a/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_all_configs, load_config  # noqa: E402
from src.data.dataset import (  # noqa: E402
    ExampleDataset,
    build_example_dataloader,
    example_records,
)
from src.evaluation.occupancy_sanity import last_train_dice  # noqa: E402
from src.models.anchor_provider import ANCHOR_SOURCES, build_anchor_provider  # noqa: E402
from src.models.relational_vlm import (  # noqa: E402
    VARIANTS,
    build_relational_vlm,
    load_relational_vlm,
)
from src.training.augmentations import build_rotation_augmentation  # noqa: E402
from src.training.logger import logging_config  # noqa: E402
from src.training.trainer import (  # noqa: E402
    StageBOverfitRunner,
    StageBTrainer,
    TrainingSettings,
    resolve_device,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase", default="oracle", choices=("overfit", "oracle"),
        help="overfit = Phase 2 single-scene bug catcher; oracle = Phase 3 training",
    )
    parser.add_argument(
        "--anchor-source", default=None, choices=ANCHOR_SOURCES,
        help="oracle = ground-truth anchor masks (default); predicted = Stage A's masks",
    )
    parser.add_argument(
        "--stage-a-checkpoint", type=Path, default=None,
        help="Stage A weights, required by --anchor-source predicted",
    )
    parser.add_argument(
        "--anchor-threshold", type=float, default=0.5,
        help="probability cut applied to Stage A's anchor masks",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="short run on the smoke corpus with the reduced model profile",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="dataset directory (default: data/smoke with --smoke, else data/processed)",
    )
    parser.add_argument(
        "--variant", default=None, choices=VARIANTS,
        help="model variant (default: full, or whatever --checkpoint was trained as); "
             "the three non-`full` ones are the mandatory baselines",
    )
    parser.add_argument(
        "--profile", default="laptop_mps", choices=("laptop_mps", "rtx5090"),
        help="hardware profile from configs/train.yaml",
    )
    parser.add_argument(
        "--model-profile", default=None, choices=("default", "smoke"),
        help="override the model width profile",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Phase 2 step budget")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--scene-id", default=None,
        help="Phase 2: which scene to overfit (default: the first training scene)",
    )
    parser.add_argument(
        "--limit-examples", type=int, default=None, help="cap the examples used per split",
    )
    parser.add_argument(
        "--eval-split", default="val", choices=("train", "val", "test"),
        help="split used for evaluation",
    )
    parser.add_argument(
        "--augment", dest="augment", action="store_true", default=None,
        help="rotate the training examples per epoch and rewrite their directions "
             "(default: on for --phase oracle, off for --phase overfit)",
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="train on the stored pose only",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="skip training; evaluate --checkpoint",
    )
    parser.add_argument(
        "--skip-occupancy-sanity",
        action="store_true",
        help="skip prompt-invariance, train/val localisation, and one-object-vs-union checks",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Stage B weights to evaluate or to resume the model from",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="run directory (default: runs/stage_b_<phase>[_smoke])",
    )
    parser.add_argument("--quiet", action="store_true")
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


def build_settings(args: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings.for_stage_b(
        phase=args.phase,
        hardware_profile=args.profile,
        smoke=args.smoke,
        overrides={
            "epochs": args.epochs,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "seed": args.seed,
            "model_profile": args.model_profile,
            "anchor_source": args.anchor_source,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    data_root = resolve_data_root(args)
    settings = build_settings(args)
    anchor_source = args.anchor_source or settings.anchor_source
    needs_scene_volume = True
    suffix = "_smoke" if args.smoke else ""
    output_dir = args.output or (PROJECT_ROOT / f"runs/stage_b_{args.phase}{suffix}")
    log_cfg = logging_config(
        extra_tags=[
            f"stage_b_{args.phase}",
            f"anchors_{anchor_source}",
            "occupancy-decoder",
            "smoke" if args.smoke else "",
            args.variant or "",
        ]
    )
    run_name = output_dir.name
    full_config = load_all_configs()

    # The augmentation is built against the split's grid, so read it from the
    # manifest before any dataset is constructed.
    probe_record = example_records(data_root, "train")[0].metadata
    volume_shape, spacing = probe_record.volume_shape, probe_record.spacing

    # -- data --------------------------------------------------------------
    # Phase 2 is a wiring bug-catcher measured by how far one scene can be
    # overfit, so it runs on the stored pose unless --augment is asked for
    # explicitly; augmenting it would move the goalposts rather than find bugs.
    # Never on the evaluation split: the metric has to stay comparable.
    augment_wanted = args.phase == "oracle" if args.augment is None else args.augment
    train_augmentation = (
        build_rotation_augmentation(
            load_config("train")["augmentations"],
            volume_shape=volume_shape,
            spacing=spacing,
            seed=settings.seed,
            stage="stage_b",
        )
        if augment_wanted
        else None
    )

    if args.phase == "overfit":
        scene_ids = [args.scene_id] if args.scene_id else None
        if scene_ids is None:
            probe = ExampleDataset(data_root, "train")
            scene_ids = probe.scene_ids[: load_config("train")["stage_b_overfit"]["scenes"]]
        train_dataset = ExampleDataset(
            data_root, "train", scene_ids=scene_ids,
            include_scene_volume=needs_scene_volume, augment=train_augmentation,
        )
        val_dataset = None
    else:
        train_dataset = ExampleDataset(
            data_root, "train", limit=args.limit_examples,
            include_scene_volume=needs_scene_volume, augment=train_augmentation,
        )
        val_dataset = ExampleDataset(
            data_root, args.eval_split, limit=args.limit_examples,
            include_scene_volume=needs_scene_volume,
        )

    train_loader = build_example_dataloader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=args.phase != "overfit",
        num_workers=settings.num_workers,
        seed=settings.seed,
    )
    val_loader = (
        build_example_dataloader(
            val_dataset, batch_size=settings.batch_size, num_workers=settings.num_workers
        )
        if val_dataset is not None
        else None
    )

    # -- model and anchors -------------------------------------------------
    device = resolve_device(settings.device)
    if args.checkpoint is not None:
        # The checkpoint knows its own width profile, variant and resolution.
        model = load_relational_vlm(
            args.checkpoint,
            profile=args.model_profile,
            variant=args.variant,          # None keeps the checkpoint's own variant
            input_resolution=max(train_dataset.volume_shape),
            spacing=train_dataset.spacing,
        )
    elif args.eval_only:
        raise SystemExit("--eval-only needs --checkpoint")
    else:
        model = build_relational_vlm(
            settings.model_profile,
            variant=args.variant or "full",
            input_resolution=max(train_dataset.volume_shape),
            spacing=train_dataset.spacing,
        )

    anchor_provider = build_anchor_provider(
        anchor_source,
        checkpoint=args.stage_a_checkpoint,
        device=device,
        threshold=args.anchor_threshold,
        input_resolution=max(train_dataset.volume_shape),
    )

    if verbose:
        phase_number = 2 if args.phase == "overfit" else 3
        print("=" * 78)
        print(f"PHASE {phase_number} - STAGE B: relational target segmenter ({args.phase})")
        print("=" * 78)
        print(f"data root   : {data_root}")
        print(f"train       : {train_dataset.describe()}")
        if val_dataset is not None:
            print(f"{args.eval_split:<12}: {val_dataset.describe()}")
        print(f"output      : {output_dir}")
        print(f"device      : {device}  profile {args.profile}")
        print(
            f"anchors     : {anchor_source}"
            + (f" from {args.stage_a_checkpoint}" if anchor_source == "predicted" else "")
        )
        if args.phase == "overfit":
            print(f"schedule    : {settings.steps} steps, lr {settings.learning_rate}, seed {settings.seed}")
        else:
            print(
                f"schedule    : {settings.epochs} epochs, batch {settings.batch_size}, "
                f"lr {settings.learning_rate}, seed {settings.seed}"
            )
        print()

    # -- run ---------------------------------------------------------------
    if args.eval_only:
        trainer = StageBTrainer(
            model, settings, train_loader, val_loader,
            output_dir=output_dir, anchor_provider=anchor_provider,
            stage=f"stage_b_{args.phase}", spacing=train_dataset.spacing, verbose=verbose,
        )
        loader = val_loader or train_loader
        report = trainer.evaluate(loader, with_hausdorff=True)
        if verbose:
            print(f"EVALUATION on {args.eval_split} ({anchor_source} anchors)")
            print("-" * 78)
            print(report["table"])
            if "anchor_quality" in report:
                quality = report["anchor_quality"]
                print(
                    f"\n  anchor masks: dice {quality['anchor_dice']:.4f} "
                    f"iou {quality['anchor_iou']:.4f} "
                    f"empty {quality['empty_anchor_fraction']:.1%}"
                )
        write_report(output_dir / f"evaluation_{args.eval_split}_{anchor_source}.json", report)
        if not args.skip_occupancy_sanity:
            sanity = trainer.occupancy_sanity(
                loader,
                train_dice=last_train_dice(output_dir),
                val_dice=float(report["overall"]["dice"]),
            )
            print_occupancy_sanity(sanity, output_dir, verbose)
        return 0

    if args.phase == "overfit":
        runner = StageBOverfitRunner(
            model, settings, train_loader, None,
            output_dir=output_dir, anchor_provider=anchor_provider,
            spacing=train_dataset.spacing, verbose=verbose,
            log_cfg=log_cfg, run_name=run_name, full_config=full_config,
        )
        report = runner.run()
        if verbose:
            print()
            print("-" * 78)
            status = "PASS" if report["passed"] else "FAIL"
            print(
                f"PHASE 2 {status}: train dice {report['final_train_dice']:.4f} "
                f"(target {report['target_train_dice']}) after {report['steps_run']} steps "
                f"on {report['examples']} examples of {', '.join(train_dataset.scene_ids)}"
            )
            print(report["metrics"].get("table", ""))
            print(f"report      : {output_dir}/overfit_report.json")
            print(f"metrics     : {output_dir}/metrics.jsonl")
        if not args.skip_occupancy_sanity:
            sanity = runner.occupancy_sanity(
                train_loader,
                train_dice=float(report["final_train_dice"]),
                val_dice=None,
            )
            print_occupancy_sanity(sanity, output_dir, verbose)
        return 0 if report["passed"] else 1

    trainer = StageBTrainer(
        model, settings, train_loader, val_loader,
        output_dir=output_dir, anchor_provider=anchor_provider,
        stage=f"stage_b_{anchor_source}", spacing=train_dataset.spacing, verbose=verbose,
        log_cfg=log_cfg, run_name=run_name, full_config=full_config,
    )
    history = trainer.fit()
    final = trainer.evaluate(val_loader, with_hausdorff=True)
    if verbose:
        print()
        print("-" * 78)
        print(
            f"FINAL {args.eval_split.upper()} (best epoch {trainer.best_epoch}, "
            f"best dice {trainer.best_dice:.4f}, {anchor_source} anchors)"
        )
        print("-" * 78)
        print(final["table"])
        print()
        print(f"checkpoints : {output_dir}/best.pt, {output_dir}/last.pt")
        print(f"history     : {output_dir}/history.json")
        print(f"metrics     : {output_dir}/metrics.jsonl")
    write_report(output_dir / f"final_{args.eval_split}_{anchor_source}.json", final)
    if not args.skip_occupancy_sanity:
        train_dice = history[-1].train_dice if history else last_train_dice(output_dir)
        sanity = trainer.occupancy_sanity(
            val_loader,
            train_dice=train_dice,
            val_dice=float(final["overall"]["dice"]),
        )
        print_occupancy_sanity(sanity, output_dir, verbose)
    return 0 if history else 1


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v for k, v in report.items() if k != "table"}, indent=2) + "\n",
        encoding="utf-8",
    )


def print_occupancy_sanity(report: dict, output_dir: Path, verbose: bool) -> None:
    if not verbose:
        return
    print()
    print(report["table"])
    print(f"occupancy sanity : {output_dir}/occupancy_sanity.json")


if __name__ == "__main__":
    sys.exit(main())
