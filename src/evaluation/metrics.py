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


def soft_centroid(
    weights: Tensor,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    eps: float = 1e-6,
) -> Tensor:
    """Mass-weighted centroid of ``[B, C, D, H, W]`` weights, in world units.

    This is the single definition of "centroid of a prediction" in the project:
    :func:`centroid_distance` reports it and
    :func:`src.training.losses.centroid_loss` optimises it, so the metric and the
    loss can never disagree about what they mean.

    Returns ``[B, C, 3]`` ordered ``(z, y, x)`` - array order, matching the
    volume's own axes - while ``spacing`` is ``(x, y, z)`` as everywhere else in
    the project. Passing a binary mask gives its ordinary centroid; passing
    probabilities gives the expected position under the prediction, which is
    always defined and differentiable even when nothing crosses the threshold.

    ``eps`` keeps the denominator finite for an all-zero map; the caller decides
    what an empty prediction means (``centroid_distance`` returns ``nan``,
    ``centroid_loss`` masks the sample out).
    """
    if weights.ndim != 5:
        raise ValueError(f"expected [B, C, D, H, W] weights, got {tuple(weights.shape)}")
    batch, channels, depth, height, width = weights.shape
    values = weights.to(torch.float32)
    sx, sy, sz = (float(v) for v in spacing)
    axes = (
        torch.arange(depth, device=values.device, dtype=values.dtype) * sz,
        torch.arange(height, device=values.device, dtype=values.dtype) * sy,
        torch.arange(width, device=values.device, dtype=values.dtype) * sx,
    )
    flat = values.reshape(batch * channels, -1)
    mass = flat.sum(dim=1, keepdim=True)
    coordinates = torch.stack(
        torch.meshgrid(*axes, indexing="ij")
    ).reshape(3, -1)                                   # [3, D*H*W], ordered (z, y, x)
    centroid = (flat @ coordinates.T) / (mass + eps)   # [B*C, 3]
    return centroid.reshape(batch, channels, 3)


def volume_diagonal(
    volume_shape: Sequence[int], spacing: Sequence[float] = (1.0, 1.0, 1.0)
) -> float:
    """Longest distance in a ``(D, H, W)`` volume, the natural scale for an error."""
    depth, height, width = (int(v) for v in volume_shape)
    sx, sy, sz = (float(v) for v in spacing)
    return math.sqrt(
        ((depth - 1) * sz) ** 2 + ((height - 1) * sy) ** 2 + ((width - 1) * sx) ** 2
    )


def centroid_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    soft: bool = True,
) -> list[float]:
    """Per-sample distance between predicted and target centroids, in world units.

    Dice conflates *looking in the wrong place* with *looking in the right place
    at the wrong size*; this separates them. It is the metric that shows whether
    the relational part of the task is being solved at all, which matters here
    because the target shape is never an input - Stage B's whole job is to find
    the right region.

    Args:
        soft: weight by probability (the default). Always defined, so no
            empty-prediction special case, and it uses the full confidence map
            instead of discarding it at a threshold. ``soft=False`` thresholds
            first and returns ``nan`` for an empty prediction, which
            :class:`MetricAccumulator` drops exactly as it drops an undefined
            Hausdorff distance.

    A large soft-vs-hard divergence is itself a signal: it means the probability
    mass is spread or multi-modal rather than sitting on one coherent blob.
    """
    if prediction.shape[1] != 1:
        raise ValueError(f"expected a single output channel, got {prediction.shape[1]}")
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)} vs target "
            f"{tuple(target.shape)}"
        )
    scores = prediction.to(torch.float32)
    if from_logits:
        scores = torch.sigmoid(scores)
    weights = scores if soft else _binarize(scores, threshold).to(torch.float32)
    reference = _binarize(target, 0.5).to(torch.float32)

    predicted = soft_centroid(weights, spacing=spacing)[:, 0]
    expected = soft_centroid(reference, spacing=spacing)[:, 0]
    distances = torch.linalg.vector_norm(predicted - expected, dim=1)

    predicted_mass = weights.reshape(weights.shape[0], -1).sum(dim=1)
    target_mass = reference.reshape(reference.shape[0], -1).sum(dim=1)
    undefined = (predicted_mass <= 0) | (target_mass <= 0)
    return [
        float("nan") if bad else float(value)
        for value, bad in zip(distances.tolist(), undefined.tolist())
    ]


def centroid_baselines(
    target: Tensor,
    *,
    anchor_union: Tensor | None = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, list[float]]:
    """Reference centroid errors, without which the model's number means nothing.

    ``volume_centre`` is what a model that learned nothing scores.
    ``anchor_union`` is what a model that ignores the *directions* and simply
    points at the anchors scores - beating it is the minimum evidence that the
    prompt is being used at all, and a model sitting at it is a relational
    failure however good its Dice looks.

    Neither changes during training, so they are computed from the batch itself
    rather than being carried around as constants.
    """
    reference = _binarize(target, 0.5).to(torch.float32)
    expected = soft_centroid(reference, spacing=spacing)[:, 0]

    depth, height, width = target.shape[2:]
    sx, sy, sz = (float(v) for v in spacing)
    centre = torch.tensor(
        [((depth - 1) / 2) * sz, ((height - 1) / 2) * sy, ((width - 1) / 2) * sx],
        device=expected.device,
        dtype=expected.dtype,
    )
    out = {
        "volume_centre": torch.linalg.vector_norm(
            centre.unsqueeze(0) - expected, dim=1
        ).tolist()
    }
    if anchor_union is not None:
        union = _binarize(anchor_union, 0.5).to(torch.float32)
        anchors = soft_centroid(union, spacing=spacing)[:, 0]
        out["anchor_union"] = torch.linalg.vector_norm(anchors - expected, dim=1).tolist()
    return out


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
    centroid: list[float] = field(default_factory=list)

    def add(
        self,
        dice: float,
        iou: float,
        hausdorff: float | None = None,
        centroid: float | None = None,
    ) -> None:
        self.dice.append(float(dice))
        self.iou.append(float(iou))
        if hausdorff is not None:
            self.hausdorff.append(float(hausdorff))
        if centroid is not None:
            self.centroid.append(float(centroid))

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
        if self.centroid:
            defined = [value for value in self.centroid if not math.isnan(value)]
            result["centroid"] = sum(defined) / len(defined) if defined else float("nan")
            result["centroid_undefined"] = float(len(self.centroid) - len(defined))
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
    #: Centroid errors of the two trivial predictors, accumulated alongside the
    #: model's own. Reported once at the top level rather than per stratum: they
    #: do not change during training, and the model's centroid error is
    #: uninterpretable without them.
    centroid_baselines: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
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
        with_centroid: bool = True,
        anchor_union: Tensor | None = None,
    ) -> None:
        """Accumulate one batch of ``[B, 1, D, H, W]`` predictions.

        ``with_centroid`` is on by default - unlike Hausdorff it costs one
        weighted sum per sample, and it is the only number here that separates a
        misplaced prediction from a badly-sized one. Pass ``anchor_union`` to
        also accumulate the baselines that make it interpretable.
        """
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

        centroids: list[float] | list[None]
        if with_centroid:
            centroids = centroid_distance(
                prediction, target, threshold=threshold, from_logits=from_logits,
                spacing=spacing, soft=True,
            )
            for name, values in centroid_baselines(
                target, anchor_union=anchor_union, spacing=spacing
            ).items():
                self.centroid_baselines[name].extend(values)
        else:
            centroids = [None] * prediction.shape[0]  # type: ignore[assignment]

        for index in range(prediction.shape[0]):
            sample_dice = float(dice[index].mean())
            sample_iou = float(iou[index].mean())
            distance = distances[index]
            centroid = centroids[index]
            self.overall.add(sample_dice, sample_iou, distance, centroid)
            if target_shapes is not None:
                self._add(
                    "target_shape", target_shapes[index], sample_dice, sample_iou,
                    distance, centroid,
                )
            if anchor_shapes is not None:
                for name in anchor_shapes[index]:
                    self._add(
                        "anchor_shape", name, sample_dice, sample_iou, distance, centroid
                    )
            if directions is not None:
                for slot, direction in enumerate(directions[index]):
                    self._add(
                        "direction", direction, sample_dice, sample_iou, distance, centroid
                    )
                    self._add(
                        "clause_slot", f"slot_{slot + 1}", sample_dice, sample_iou,
                        distance, centroid,
                    )

    def _add(
        self,
        stratum: str,
        key: str,
        dice: float,
        iou: float,
        hausdorff: float | None,
        centroid: float | None = None,
    ) -> None:
        self.strata[stratum][key].add(dice, iou, hausdorff, centroid)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "overall": self.overall.summary(),
            "strata": {
                stratum: {key: acc.summary() for key, acc in sorted(buckets.items())}
                for stratum, buckets in sorted(self.strata.items())
            },
        }
        if self.centroid_baselines:
            result["centroid_baselines"] = {
                name: sum(values) / len(values)
                for name, values in sorted(self.centroid_baselines.items())
                if values
            }
        return result


def format_stratified_table(
    metrics: StratifiedMetrics | Mapping[str, Any], strata: Iterable[str] | None = None
) -> str:
    """Render the Stage B report for the console."""
    summary = metrics.summary() if isinstance(metrics, StratifiedMetrics) else dict(metrics)
    overall = summary["overall"]
    lines = [
        f"  {'stratum':<22} {'dice':>7} {'iou':>7} {'hd':>7} {'cdist':>7} {'n':>5}"
    ]

    def row(label: str, entry: Mapping[str, float]) -> str:
        distance = entry.get("hausdorff")
        rendered = "      -" if distance is None else f"{distance:>7.2f}"
        centroid = entry.get("centroid")
        centroid_rendered = "      -" if centroid is None else f"{centroid:>7.2f}"
        return (
            f"  {label:<22} {entry['dice']:>7.4f} {entry['iou']:>7.4f} "
            f"{rendered} {centroid_rendered} {int(entry['count']):>5}"
        )

    lines.append(row("overall", overall))
    baselines = summary.get("centroid_baselines") or {}
    if baselines:
        # Without these the centroid column is a number with no scale.
        rendered = "  ".join(f"{name} {value:.2f}" for name, value in baselines.items())
        lines.append(f"  {'cdist baselines':<22} {rendered}")
    for stratum, buckets in summary.get("strata", {}).items():
        if strata is not None and stratum not in strata:
            continue
        lines.append(f"  -- {stratum} " + "-" * (62 - len(stratum)))
        for key, entry in buckets.items():
            lines.append(row(f"  {key}", entry))
    return "\n".join(lines)
