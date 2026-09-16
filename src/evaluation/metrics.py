"""Segmentation metrics.

Dice and IoU are what Stage A is judged on (per class, plus the mean). Stage B
adds symmetric Hausdorff distance (:func:`hausdorff_distance`) and stratified
reporting (:class:`StratifiedMetrics`): aggregate numbers hide exactly the
failure this project cares about, so every Stage B result is also broken down by
target shape, anchor shape and direction.

Both metrics are computed on thresholded predictions, per sample and per
channel, with the standard degenerate-case convention: if prediction and target
are both empty the score is 1.0, and if exactly one is empty it is 0.0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _binarize(x: Tensor, threshold: float) -> Tensor:
    return (x >= threshold) if x.dtype.is_floating_point else (x > 0)


def _flatten(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], x.shape[1], -1)


def dice_score(
    prediction: Tensor, target: Tensor, *, threshold: float = 0.5, from_logits: bool = False
) -> Tensor:
    """Per-sample, per-channel Dice. Returns ``[B, C]``."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
    if from_logits:
        prediction = torch.sigmoid(prediction)
    predicted = _flatten(_binarize(prediction, threshold)).to(torch.float32)
    reference = _flatten(_binarize(target, 0.5)).to(torch.float32)
    intersection = (predicted * reference).sum(dim=-1)
    denominator = predicted.sum(dim=-1) + reference.sum(dim=-1)
    both_empty = denominator == 0
    scores = torch.where(
        both_empty, torch.ones_like(denominator), 2.0 * intersection / denominator.clamp(min=1e-8)
    )
    return scores


def iou_score(
    prediction: Tensor, target: Tensor, *, threshold: float = 0.5, from_logits: bool = False
) -> Tensor:
    """Per-sample, per-channel IoU (Jaccard). Returns ``[B, C]``."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
    if from_logits:
        prediction = torch.sigmoid(prediction)
    predicted = _flatten(_binarize(prediction, threshold)).to(torch.float32)
    reference = _flatten(_binarize(target, 0.5)).to(torch.float32)
    intersection = (predicted * reference).sum(dim=-1)
    union = predicted.sum(dim=-1) + reference.sum(dim=-1) - intersection
    both_empty = union == 0
    return torch.where(
        both_empty, torch.ones_like(union), intersection / union.clamp(min=1e-8)
    )


def surface_voxels(mask: Tensor) -> Tensor:
    """Indices ``(z, y, x)`` of the surface voxels of one binary mask.

    A voxel is on the surface when it is occupied and at least one of its six
    face neighbours is not. Comparing surfaces rather than whole volumes is the
    standard formulation and is what keeps the distance computation small: a
    64^3 shape here has a few hundred surface voxels against a few hundred
    thousand background ones.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D (D, H, W), got {tuple(mask.shape)}")
    occupied = (mask > 0.5).to(torch.float32)
    if occupied.sum() == 0:
        return occupied.new_zeros((0, 3))
    padded = F.pad(occupied[None, None], (1, 1, 1, 1, 1, 1), value=0.0)
    eroded = -F.max_pool3d(-padded, kernel_size=3, stride=1, padding=0)[0, 0]
    return torch.nonzero(occupied - eroded > 0, as_tuple=False).to(torch.float32)


def hausdorff_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    percentile: float | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Symmetric Hausdorff distance between two binary masks, in world units.

    Both directions are measured (``max`` of the two directed distances), on the
    mask surfaces, with the exact pairwise distances - no approximation, which a
    64^3 volume can afford.

    ``percentile=95`` gives the 95th-percentile variant, which is what to use if
    a single stray voxel should not dominate the report.

    Degenerate cases follow the same convention as Dice and IoU: two empty masks
    are a perfect match (0.0), and one empty mask against a non-empty one is
    undefined and returned as ``nan`` so it can be counted rather than silently
    averaged into the result.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
    if from_logits:
        prediction = torch.sigmoid(prediction)
    predicted = _binarize(prediction.squeeze(), threshold).to(torch.float32)
    reference = _binarize(target.squeeze(), 0.5).to(torch.float32)
    if predicted.ndim != 3:
        raise ValueError(
            f"hausdorff_distance takes one mask at a time, got {tuple(prediction.shape)}"
        )

    a = surface_voxels(predicted)
    b = surface_voxels(reference)
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")

    # Surfaces are indexed (z, y, x); spacing is (x, y, z).
    scale = torch.tensor(
        [float(spacing[2]), float(spacing[1]), float(spacing[0])], dtype=a.dtype, device=a.device
    )
    distances = torch.cdist(a * scale, b * scale)
    forward = distances.amin(dim=1)
    backward = distances.amin(dim=0)
    if percentile is None:
        return float(torch.maximum(forward.max(), backward.max()))
    quantile = float(percentile) / 100.0
    return float(
        torch.maximum(
            torch.quantile(forward, quantile), torch.quantile(backward, quantile)
        )
    )


def batch_hausdorff(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    percentile: float | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> list[float]:
    """Per-sample Hausdorff distance for a ``[B, 1, D, H, W]`` pair."""
    if prediction.shape[1] != 1:
        raise ValueError(f"expected a single output channel, got {prediction.shape[1]}")
    return [
        hausdorff_distance(
            prediction[index, 0],
            target[index, 0],
            threshold=threshold,
            from_logits=from_logits,
            percentile=percentile,
            spacing=spacing,
        )
        for index in range(prediction.shape[0])
    ]


@dataclass
class PerClassMetrics:
    """Accumulates per-class Dice and IoU across batches."""

    class_names: tuple[str, ...]
    _dice: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _iou: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        class_ids: Tensor | None = None,
        threshold: float = 0.5,
        from_logits: bool = True,
    ) -> None:
        """Accumulate one batch.

        Args:
            prediction: ``[B, C, D, H, W]`` logits (or probabilities).
            target: ``[B, C, D, H, W]`` binary targets.
            class_ids: ``[B, C]`` zero-based vocabulary indices naming each
                channel; defaults to channel order.
        """
        dice = dice_score(prediction, target, threshold=threshold, from_logits=from_logits)
        iou = iou_score(prediction, target, threshold=threshold, from_logits=from_logits)
        batch, channels = dice.shape
        if class_ids is None:
            class_ids = torch.arange(channels).unsqueeze(0).expand(batch, -1)
        for sample in range(batch):
            for channel in range(channels):
                name = self.class_names[int(class_ids[sample, channel])]
                self._dice[name].append(float(dice[sample, channel]))
                self._iou[name].append(float(iou[sample, channel]))

    def per_class(self) -> dict[str, dict[str, float]]:
        """Mean Dice and IoU for every class that was seen."""
        return {
            name: {
                "dice": sum(self._dice[name]) / len(self._dice[name]),
                "iou": sum(self._iou[name]) / len(self._iou[name]),
                "count": len(self._dice[name]),
            }
            for name in self.class_names
            if self._dice[name]
        }

    def mean(self) -> dict[str, float]:
        """Macro-average over classes (each class weighted equally)."""
        per_class = self.per_class()
        if not per_class:
            return {"dice": 0.0, "iou": 0.0}
        return {
            "dice": sum(v["dice"] for v in per_class.values()) / len(per_class),
            "iou": sum(v["iou"] for v in per_class.values()) / len(per_class),
        }

    def summary(self) -> dict[str, object]:
        return {"mean": self.mean(), "per_class": self.per_class()}


def format_per_class_table(metrics: PerClassMetrics, class_order: Sequence[str] | None = None) -> str:
    """Render a per-class Dice/IoU table for the console."""
    per_class = metrics.per_class()
    names = [name for name in (class_order or metrics.class_names) if name in per_class]
    lines = [f"  {'class':<18} {'dice':>7} {'iou':>7} {'n':>5}"]
    for name in names:
        entry = per_class[name]
        lines.append(
            f"  {name:<18} {entry['dice']:>7.4f} {entry['iou']:>7.4f} {entry['count']:>5}"
        )
    average = metrics.mean()
    lines.append(f"  {'mean':<18} {average['dice']:>7.4f} {average['iou']:>7.4f}")
    return "\n".join(lines)


@dataclass
class MetricAccumulator:
    """Running Dice / IoU / Hausdorff over a set of samples."""

    dice: list[float] = field(default_factory=list)
    iou: list[float] = field(default_factory=list)
    hausdorff: list[float] = field(default_factory=list)

    def add(self, dice: float, iou: float, hausdorff: float | None = None) -> None:
        self.dice.append(float(dice))
        self.iou.append(float(iou))
        if hausdorff is not None:
            self.hausdorff.append(float(hausdorff))

    def summary(self) -> dict[str, float]:
        finite = [value for value in self.hausdorff if not math.isnan(value)]
        result = {
            "dice": sum(self.dice) / len(self.dice) if self.dice else 0.0,
            "iou": sum(self.iou) / len(self.iou) if self.iou else 0.0,
            "count": float(len(self.dice)),
        }
        if self.hausdorff:
            result["hausdorff"] = sum(finite) / len(finite) if finite else float("nan")
            result["hausdorff_undefined"] = float(len(self.hausdorff) - len(finite))
        return result


@dataclass
class StratifiedMetrics:
    """Stage B metrics, aggregate and stratified.

    Strata follow ``configs/train.yaml: evaluation.stratify_by``: target shape,
    anchor shape (every anchor of an example contributes to its own bucket),
    direction, and the clause slot each ``(direction, anchor)`` pair sat in.
    CLAUDE.md is explicit that a high aggregate Dice proves nothing on its own -
    the headline claim is about held-out target classes, which only the target
    stratum can show.
    """

    overall: MetricAccumulator = field(default_factory=MetricAccumulator)
    strata: dict[str, dict[str, MetricAccumulator]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(MetricAccumulator))
    )

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        target_shapes: Sequence[str] | None = None,
        anchor_shapes: Sequence[Sequence[str]] | None = None,
        directions: Sequence[Sequence[str]] | None = None,
        threshold: float = 0.5,
        from_logits: bool = True,
        with_hausdorff: bool = False,
        percentile: float | None = None,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Accumulate one batch of ``[B, 1, D, H, W]`` predictions."""
        dice = dice_score(prediction, target, threshold=threshold, from_logits=from_logits)
        iou = iou_score(prediction, target, threshold=threshold, from_logits=from_logits)
        distances: list[float] | list[None]
        if with_hausdorff:
            distances = batch_hausdorff(
                prediction,
                target,
                threshold=threshold,
                from_logits=from_logits,
                percentile=percentile,
                spacing=spacing,
            )
        else:
            distances = [None] * prediction.shape[0]  # type: ignore[assignment]

        for index in range(prediction.shape[0]):
            sample_dice = float(dice[index].mean())
            sample_iou = float(iou[index].mean())
            distance = distances[index]
            self.overall.add(sample_dice, sample_iou, distance)
            if target_shapes is not None:
                self._add("target_shape", target_shapes[index], sample_dice, sample_iou, distance)
            if anchor_shapes is not None:
                for name in anchor_shapes[index]:
                    self._add("anchor_shape", name, sample_dice, sample_iou, distance)
            if directions is not None:
                for slot, direction in enumerate(directions[index]):
                    self._add("direction", direction, sample_dice, sample_iou, distance)
                    self._add(
                        "clause_slot", f"slot_{slot + 1}", sample_dice, sample_iou, distance
                    )

    def _add(
        self, stratum: str, key: str, dice: float, iou: float, hausdorff: float | None
    ) -> None:
        self.strata[stratum][key].add(dice, iou, hausdorff)

    def summary(self) -> dict[str, Any]:
        return {
            "overall": self.overall.summary(),
            "strata": {
                stratum: {key: acc.summary() for key, acc in sorted(buckets.items())}
                for stratum, buckets in sorted(self.strata.items())
            },
        }


def format_stratified_table(
    metrics: StratifiedMetrics | Mapping[str, Any], strata: Iterable[str] | None = None
) -> str:
    """Render the Stage B report for the console."""
    summary = metrics.summary() if isinstance(metrics, StratifiedMetrics) else dict(metrics)
    overall = summary["overall"]
    lines = [f"  {'stratum':<22} {'dice':>7} {'iou':>7} {'hd':>7} {'n':>5}"]

    def row(label: str, entry: Mapping[str, float]) -> str:
        distance = entry.get("hausdorff")
        rendered = "      -" if distance is None else f"{distance:>7.2f}"
        return (
            f"  {label:<22} {entry['dice']:>7.4f} {entry['iou']:>7.4f} "
            f"{rendered} {int(entry['count']):>5}"
        )

    lines.append(row("overall", overall))
    for stratum, buckets in summary.get("strata", {}).items():
        if strata is not None and stratum not in strata:
            continue
        lines.append(f"  -- {stratum} " + "-" * (62 - len(stratum)))
        for key, entry in buckets.items():
            lines.append(row(f"  {key}", entry))
    return "\n".join(lines)
