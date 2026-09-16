"""Occupancy sanity checks: prompt use, train/val split, one object not the union.

High Dice after adding decoder occupancy is expected — WHAT can copy a connected
component once WHERE is roughly right. These checks catch the failure mode that
Dice hides: the model ignoring the prompt and painting a remaining blob, or
painting every remaining blob.

1. Flip one direction / permute channels with occupancy **fixed**. The binary
   mask should move or collapse. If it barely changes, the model is prompt-
   invariant.
2. Compare train Dice to val Dice. Both high → localisation is generalising.
   Train high / val low → WHERE is still the problem, not occupancy.
3. A qualitative slice plus a component count: the prediction should overlap
   **one** remaining occupancy object, not the union of the seven.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.data.dataset import stage_b_model_inputs
from src.evaluation.counterfactuals import flip_one_direction, permute_anchor_channels
from src.evaluation.metrics import dice_score
from src.evaluation.qualitative import save_occupancy_slice

#: Dice(original pred, counterfactual pred) above this → occupancy is being
#: copied without reading the prompt.
PROMPT_INVARIANT_DICE = 0.85
#: Train/val Dice at or above this counts as "high" under occupancy.
HIGH_DICE = 0.80
#: Train minus val at or above this, with train high, is a WHERE failure.
LOCALISATION_GAP = 0.20
#: Dice(pred, occupancy) above this means the prediction is the remaining union
#: (one of seven similar-sized objects scores around 0.25).
UNION_OCCUPANCY_DICE = 0.50
#: Fraction of predicted voxels that must sit inside occupancy.
INSIDE_OCCUPANCY = 0.80


def localisation_verdict(
    train_dice: float | None,
    val_dice: float | None,
    *,
    high: float = HIGH_DICE,
    gap: float = LOCALISATION_GAP,
) -> dict[str, Any]:
    """Classify a train/val Dice pair.

    ``localisation_generalising``
        both high — occupancy is supplying shape and WHERE transfers.
    ``where_failure``
        train high, val not — occupancy cannot explain the gap; relations fail
        on held-out layouts / shapes.
    ``underfit``
        train not high yet.
    ``skipped``
        one of the scores is missing (eval-only without a history file).
    """
    if train_dice is None or val_dice is None:
        return {
            "verdict": "skipped",
            "reason": "need both train_dice and val_dice",
            "train_dice": train_dice,
            "val_dice": val_dice,
        }
    train = float(train_dice)
    val = float(val_dice)
    payload: dict[str, Any] = {
        "train_dice": train,
        "val_dice": val,
        "gap": train - val,
        "high": high,
        "gap_threshold": gap,
    }
    if train >= high and val >= high:
        payload["verdict"] = "localisation_generalising"
        payload["reason"] = (
            "train and val Dice are both high; occupancy is supplying shape and "
            "WHERE is transferring"
        )
        return payload
    if train >= high and (train - val) >= gap:
        payload["verdict"] = "where_failure"
        payload["reason"] = (
            "train Dice is high but val is not; occupancy is not the remaining "
            "problem — relations (WHERE) are"
        )
        return payload
    if train < high:
        payload["verdict"] = "underfit"
        payload["reason"] = "train Dice is not high yet"
        return payload
    payload["verdict"] = "mixed"
    payload["reason"] = "train is high but the val drop is smaller than the WHERE gap"
    return payload


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """6-connected labels on a 3D binary mask. Label 0 is background."""
    occupied = np.asarray(mask, dtype=bool)
    labels = np.zeros(occupied.shape, dtype=np.int32)
    count = 0
    depth, height, width = occupied.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for z, y, x in np.argwhere(occupied):
        if labels[z, y, x]:
            continue
        count += 1
        stack = [(int(z), int(y), int(x))]
        labels[z, y, x] = count
        while stack:
            cz, cy, cx = stack.pop()
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                    if occupied[nz, ny, nx] and labels[nz, ny, nx] == 0:
                        labels[nz, ny, nx] = count
                        stack.append((nz, ny, nx))
    return labels, count


def _volume(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    while array.ndim > 3:
        array = array[0]
    return array


def occupancy_carving(
    occupancy: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, Any]:
    """Is the prediction one remaining object, the union, empty, or off occupancy?"""
    pred_tensor = torch.sigmoid(prediction) if from_logits else prediction
    pred = _volume(pred_tensor) >= threshold
    occ = _volume(occupancy) >= 0.5
    tgt = _volume(target) >= 0.5
    pred_count = int(pred.sum())
    inside = float((pred & occ).sum() / pred_count) if pred_count else 0.0
    dice_occ = float(
        dice_score(
            torch.from_numpy(pred.astype(np.float32))[None, None],
            torch.from_numpy(occ.astype(np.float32))[None, None],
        )
    )
    dice_tgt = float(
        dice_score(
            torch.from_numpy(pred.astype(np.float32))[None, None],
            torch.from_numpy(tgt.astype(np.float32))[None, None],
        )
    )
    labels, n_components = label_components(occ)
    touched = sorted({int(label) for label in labels[pred] if label > 0}) if pred_count else []
    if pred_count == 0:
        verdict = "empty"
    elif inside < 0.5:
        verdict = "outside_occupancy"
    elif len(touched) == 1 and inside >= INSIDE_OCCUPANCY:
        verdict = "one_object"
    elif len(touched) >= 3 or dice_occ >= UNION_OCCUPANCY_DICE:
        verdict = "union"
    else:
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "predicted_voxels": pred_count,
        "inside_occupancy": inside,
        "dice_vs_occupancy": dice_occ,
        "dice_vs_target": dice_tgt,
        "occupancy_components": n_components,
        "components_touched": len(touched),
        "passed": verdict == "one_object",
    }


def format_sanity_table(report: Mapping[str, Any]) -> str:
    """Compact stdout block for one occupancy-sanity run."""
    lines = ["occupancy sanity"]
    correspondence = report.get("correspondence") or {}
    for name, probe in correspondence.items():
        if not isinstance(probe, Mapping) or "dice_vs_original" not in probe:
            continue
        mark = "PASS" if probe.get("passed") else "FAIL"
        lines.append(
            f"  {name:<28} {mark}  dice(pred, cf)={probe['dice_vs_original']:.4f}  "
            f"dice(cf, target)={probe['dice_vs_target']:.4f}"
        )
    localisation = report.get("localisation") or {}
    if localisation:
        def _score(value: Any) -> str:
            return "-" if value is None else f"{float(value):.4f}"

        lines.append(
            f"  {'train vs val':<28} {localisation.get('verdict', '')}  "
            f"train={_score(localisation.get('train_dice'))}  "
            f"val={_score(localisation.get('val_dice'))}"
        )
    carving = report.get("carving") or {}
    if carving:
        mark = "PASS" if carving.get("passed") else "FAIL"
        lines.append(
            f"  {'one object vs union':<28} {mark}  "
            f"one-object={carving.get('one_object_fraction', 0):.1%}  "
            f"union={carving.get('union_fraction', 0):.1%}"
        )
    slices = report.get("slices") or []
    if slices:
        lines.append(f"  slices: {len(slices)} written under occupancy_sanity/slices/")
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _predict(model: nn.Module, batch: Mapping[str, Any], anchor_masks: Tensor | None = None) -> Tensor:
    return model(**stage_b_model_inputs(batch, anchor_masks)).logits


def _dice_between_logits(prediction: Tensor, other: Tensor, *, threshold: float) -> Tensor:
    """Dice of two logit volumes after both have been sigmoided and thresholded."""
    return dice_score(
        torch.sigmoid(prediction),
        torch.sigmoid(other),
        threshold=threshold,
        from_logits=False,
    )


def _dice_vs_mask(prediction: Tensor, mask: Tensor, *, threshold: float) -> Tensor:
    return dice_score(prediction, mask, threshold=threshold, from_logits=True)


def last_train_dice(output_dir: Path | str) -> float | None:
    """Read the last epoch's train Dice from a run's ``history.json``, if any."""
    path = Path(output_dir) / "history.json"
    if not path.is_file():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return None
    value = history[-1].get("train_dice")
    return None if value is None else float(value)


@torch.no_grad()
def run_occupancy_sanity(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path | str,
    train_dice: float | None = None,
    val_dice: float | None = None,
    threshold: float = 0.5,
    max_batches: int | None = None,
    num_slices: int = 8,
    prepare_batch: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the three occupancy checks over ``loader`` and write a JSON report.

    Occupancy is held fixed on the correspondence probes: only the ordered
    anchor channels or one direction token change. ``prepare_batch`` is how
    predicted-anchor evaluation swaps the channels before those probes run.
    """
    model.eval()
    output = Path(output_dir)
    slice_dir = output / "occupancy_sanity" / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    def _move(batch: Mapping[str, Any]) -> dict[str, Any]:
        prepared = prepare_batch(batch) if prepare_batch is not None else batch
        return {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in prepared.items()
        }

    permute_vs_orig: list[float] = []
    permute_vs_target: list[float] = []
    orig_vs_target: list[float] = []
    flip_vs_orig: list[float] = []
    flip_vs_target: list[float] = []
    carving_verdicts: list[str] = []
    slices: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = _move(batch)
        original = _predict(model, moved)
        permuted = _predict(model, moved, permute_anchor_channels(moved["anchor_masks"]))
        flipped = model(
            **{
                **stage_b_model_inputs(moved),
                "direction_ids": flip_one_direction(moved["direction_ids"]),
            }
        ).logits

        permute_vs_orig.extend(
            _dice_between_logits(permuted, original, threshold=threshold).flatten().tolist()
        )
        flip_vs_orig.extend(
            _dice_between_logits(flipped, original, threshold=threshold).flatten().tolist()
        )
        permute_vs_target.extend(
            _dice_vs_mask(permuted, moved["target_mask"], threshold=threshold).flatten().tolist()
        )
        flip_vs_target.extend(
            _dice_vs_mask(flipped, moved["target_mask"], threshold=threshold).flatten().tolist()
        )
        orig_vs_target.extend(
            _dice_vs_mask(original, moved["target_mask"], threshold=threshold).flatten().tolist()
        )

        masks = model.prepare_masks(moved["anchor_masks"])
        occupancy = model.prepare_occupancy(moved["scene_volume"], masks)
        ids = moved.get("example_id")
        if isinstance(ids, str):
            ids = [ids]
        for sample in range(original.shape[0]):
            carving = occupancy_carving(
                occupancy[sample],
                original[sample],
                moved["target_mask"][sample],
                threshold=threshold,
                from_logits=True,
            )
            carving_verdicts.append(str(carving["verdict"]))
            if len(slices) < num_slices:
                example_id = (
                    str(ids[sample])
                    if ids is not None and sample < len(ids)
                    else f"batch{batch_index:03d}_sample{sample}"
                )
                dump = save_occupancy_slice(
                    slice_dir / f"{example_id}.png",
                    _volume(occupancy[sample]) >= 0.5,
                    _volume(torch.sigmoid(original[sample])) >= threshold,
                    _volume(moved["target_mask"][sample]) >= 0.5,
                )
                dump["example_id"] = example_id
                dump["carving"] = carving
                slices.append(dump)

    def _probe(vs_original: list[float], vs_target: list[float]) -> dict[str, Any]:
        dice_original = _mean(vs_original)
        return {
            "dice_vs_original": dice_original,
            "dice_vs_target": _mean(vs_target),
            "count": len(vs_original),
            "invariant_threshold": PROMPT_INVARIANT_DICE,
            "passed": bool(vs_original) and dice_original < PROMPT_INVARIANT_DICE,
        }

    n = len(carving_verdicts)
    one_object_fraction = carving_verdicts.count("one_object") / n if n else 0.0
    union_fraction = carving_verdicts.count("union") / n if n else 0.0
    correspondence = {
        "permute_channels": _probe(permute_vs_orig, permute_vs_target),
        "flip_direction": _probe(flip_vs_orig, flip_vs_target),
        "reference_dice_vs_target": _mean(orig_vs_target),
    }
    correspondence["passed"] = all(
        correspondence[name]["passed"] for name in ("permute_channels", "flip_direction")
    )
    carving = {
        "one_object_fraction": one_object_fraction,
        "union_fraction": union_fraction,
        "empty_fraction": carving_verdicts.count("empty") / n if n else 0.0,
        "outside_fraction": carving_verdicts.count("outside_occupancy") / n if n else 0.0,
        "ambiguous_fraction": carving_verdicts.count("ambiguous") / n if n else 0.0,
        "count": n,
        "passed": n > 0 and one_object_fraction >= 0.5 and union_fraction < 0.5,
    }
    localisation = localisation_verdict(train_dice, val_dice)
    report: dict[str, Any] = {
        "correspondence": correspondence,
        "localisation": localisation,
        "carving": carving,
        "slices": slices,
        "passed": bool(correspondence["passed"] and carving["passed"]),
    }
    if localisation["verdict"] == "where_failure":
        report["passed"] = False
    report["table"] = format_sanity_table(report)
    (output / "occupancy_sanity.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "table"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return report
