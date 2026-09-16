#!/usr/bin/env python3
"""Phase 1: train the Stage A promptable shape segmenter.

One sample is a whole scene: the binary ``scene_volume`` goes in, one mask logit
volume comes out per requested shape name. The target-class split filter does
not apply here - it constrains which classes Stage B may be *supervised* on,
while Stage A must learn all ten shapes so it can supply anchors by name in
Phase 4.

Reported per epoch: training loss (with its Dice/BCE components at every
deep-supervision scale) and validation Dice and IoU, per class and macro
averaged. The checkpoint is selected on macro-average validation Dice.

Examples::

    .venv/bin/python scripts/train_shape_segmenter.py --smoke
    .venv/bin/python scripts/train_shape_segmenter.py --data-root data/processed
    .venv/bin/python scripts/train_shape_segmenter.py --epochs 80 --profile rtx5090
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
from src.data.dataset import SceneDataset, build_dataloader  # noqa: E402
from src.data.primitives import SHAPE_NAMES  # noqa: E402
from src.models.shape_segmenter import build_shape_segmenter  # noqa: E402
from src.training.augmentations import build_rotation_augmentation  # noqa: E402
from src.training.logger import logging_config  # noqa: E402
from src.training.trainer import StageATrainer, TrainingSettings, resolve_device  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--profile", default="laptop_mps", choices=("laptop_mps", "rtx5090"),
        help="hardware profile from configs/train.yaml",
    )
    parser.add_argument(
        "--model-profile", default=None, choices=("default", "smoke"),
        help="override the model width profile",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    parser.add_argument(
        "--precision", default=None, choices=("fp32", "bf16", "fp16"),
        help="override the profile's autocast precision (fp16 adds a gradient scaler)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None, dest="gradient_accumulation_steps",
        help="override the gradient-accumulation factor",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--limit-scenes", type=int, default=None, help="cap the scenes used per split",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="checkpoint directory (default: runs/stage_a[_smoke])",
    )
    parser.add_argument(
        "--augment", dest="augment", action="store_true", default=None,
        help="rotate the training scenes per epoch (default: whatever "
             "augmentations.rotation_90.apply_to.stage_a says in configs/train.yaml)",
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="train on the stored pose only",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    generator_config = load_config("generator")
    default_root = (
        PROJECT_ROOT / generator_config["smoke"]["output_root"]
        if args.smoke
        else PROJECT_ROOT / generator_config["storage"]["root"]
    )
    data_root = args.data_root or default_root
    output_dir = args.output or (PROJECT_ROOT / ("runs/stage_a_smoke" if args.smoke else "runs/stage_a"))

    if not (data_root / "manifests").is_dir():
        raise SystemExit(
            f"no dataset at {data_root}. Generate one first:\n"
            f"  .venv/bin/python scripts/generate_dataset.py"
            f"{' --smoke' if args.smoke else ''}"
        )

    settings = TrainingSettings.for_stage_a(
        hardware_profile=args.profile,
        smoke=args.smoke,
        overrides={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "precision": args.precision,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "model_profile": args.model_profile,
        },
    )

    # Built against the training split's own grid, and attached to that split
    # alone: the validation scenes stay in the stored pose, or the metric stops
    # being comparable with an unaugmented run.
    probe = SceneDataset(data_root, "train", limit=args.limit_scenes)
    augmentations = dict(load_config("train")["augmentations"])
    if args.augment is not None:
        # An explicit flag overrides the per-stage switch. It does not override
        # the `rewrite_tested` gate, which stays the one hard requirement.
        rotation = dict(augmentations.get("rotation_90", {}))
        rotation["apply_to"] = {**rotation.get("apply_to", {}), "stage_a": args.augment}
        augmentations = {**augmentations, "rotation_90": rotation}
        if args.augment:
            augmentations["enabled"] = True
    train_augmentation = build_rotation_augmentation(
        augmentations,
        volume_shape=probe.volume_shape,
        spacing=probe.spacing,
        seed=settings.seed,
        stage="stage_a",
    )

    train_dataset = SceneDataset(
        data_root, "train", limit=args.limit_scenes, augment=train_augmentation
    )
    val_dataset = SceneDataset(data_root, "val", limit=args.limit_scenes)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=settings.batch_size, num_workers=settings.num_workers
    )

    model = build_shape_segmenter(
        settings.model_profile, input_resolution=max(train_dataset.volume_shape)
    )

    if verbose:
        print("=" * 78)
        print("PHASE 1 - STAGE A: promptable shape segmenter")
        print("=" * 78)
        print(f"data root   : {data_root}")
        print(f"train       : {train_dataset.describe()}")
        print(f"val         : {val_dataset.describe()}")
        print(f"output      : {output_dir}")
        print(f"device      : {resolve_device(settings.device)}  profile {args.profile}")
        print(
            f"schedule    : {settings.epochs} epochs, batch {settings.batch_size}, "
            f"lr {settings.learning_rate}, seed {settings.seed}"
        )
        print()

    trainer = StageATrainer(
        model,
        settings,
        train_loader,
        val_loader,
        class_names=SHAPE_NAMES,
        output_dir=output_dir,
        verbose=verbose,
        log_cfg=logging_config(extra_tags=["stage_a", "smoke" if args.smoke else ""]),
        run_name=output_dir.name,
        full_config=load_all_configs(),
    )
    history = trainer.fit()

    final = trainer.evaluate(val_loader)
    if verbose:
        print()
        print("-" * 78)
        print(f"FINAL VALIDATION (best epoch {trainer.best_epoch}, best dice {trainer.best_dice:.4f})")
        print("-" * 78)
        print(final["table"])
        print()
        print(f"checkpoints : {output_dir}/best.pt, {output_dir}/last.pt")
        print(f"history     : {output_dir}/history.json")
        print(f"metrics     : {output_dir}/metrics.jsonl")

    (output_dir / "final_validation.json").write_text(
        json.dumps({k: v for k, v in final.items() if k != "table"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if history else 1


if __name__ == "__main__":
    sys.exit(main())
