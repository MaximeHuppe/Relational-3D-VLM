"""Losses.

Stage A optimises, per requested prompt, a soft Dice term plus
``BCEWithLogits``, summed over three deep-supervision scales with the weights
from ``configs/model.yaml`` (0.1 / 0.3 / 0.6, coarse to fine).

CLAUDE.md asks for "multiclass cross-entropy plus Dice loss, or a documented
equivalent". The documented equivalent used here is per-prompt sigmoid Dice +
BCE, because the Stage A head is promptable: it emits one independent mask logit
per requested shape name rather than a softmax over a fixed class axis. That
independence is what lets Phase 4 query only the three anchors named in a
prompt. The shapes never overlap, so the two formulations supervise the same
partition; only the normalisation differs, and per-class Dice/IoU - the mandated
Stage A metrics - are unaffected.

Stage B uses the objective stated in CLAUDE.md::

    L = lambda_dice * DiceLoss(logits, target_mask)
      + lambda_bce  * BCEWithLogitsLoss(logits, target_mask)

which is :func:`segmentation_loss` with a single logit channel, at full
resolution only - Stage B has no deep supervision.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_SMOOTH = 1.0


def _flatten_spatial(x: Tensor) -> Tensor:
    """``[B, C, D, H, W] -> [B, C, D*H*W]``."""
    return x.reshape(x.shape[0], x.shape[1], -1)


def _as_loss_dtype(x: Tensor) -> Tensor:
    """Upcast to float32 for the loss.

    Losses are always computed in float32, even inside an ``autocast`` region.
    The Dice denominator sums one sigmoid value per voxel - 262,144 of them at
    64^3 - and float16 has neither the range nor the resolution for that sum:
    measured under fp16 autocast the loss drifts further from its float32 value
    the larger the batch (1.11 at batch 1, 1.33 at batch 8, against a stable
    1.10). bfloat16 is closer but still loses precision in the reduction.
    Upcasting costs one cast of the logits and makes the objective independent
    of the autocast dtype.
    """
    return x.float() if x.dtype != torch.float32 else x


def dice_loss(
    logits: Tensor, targets: Tensor, *, smooth: float = DEFAULT_SMOOTH
) -> Tensor:
    """Soft Dice loss on sigmoid probabilities, averaged over batch and channel.

    Args:
        logits: ``[B, C, D, H, W]`` raw logits.
        targets: ``[B, C, D, H, W]`` binary targets in ``{0, 1}``.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")
    probabilities = _flatten_spatial(torch.sigmoid(_as_loss_dtype(logits)))
    reference = _flatten_spatial(targets.to(probabilities.dtype))
    intersection = (probabilities * reference).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + reference.sum(dim=-1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """``BCEWithLogitsLoss``, averaged over every element, computed in float32."""
    logits = _as_loss_dtype(logits)
    return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype))


def segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    lambda_dice: float = 1.0,
    lambda_bce: float = 1.0,
    smooth: float = DEFAULT_SMOOTH,
) -> tuple[Tensor, dict[str, float]]:
    """``lambda_dice * DiceLoss + lambda_bce * BCEWithLogitsLoss``.

    Returns the total and a dict of the detached components, for logging.
    """
    dice = dice_loss(logits, targets, smooth=smooth)
    bce = bce_loss(logits, targets)
    total = lambda_dice * dice + lambda_bce * bce
    return total, {"dice": float(dice.detach()), "bce": float(bce.detach())}


def downsample_targets(targets: Tensor, size: Sequence[int]) -> Tensor:
    """Downsample binary targets for deep supervision, preserving presence.

    Max pooling is used rather than nearest or average sampling: at 1/4
    resolution a torus is only about one voxel thick, and both alternatives can
    delete it entirely, which would supervise the coarse head towards an empty
    mask for a structure that is really there.
    """
    target_size = tuple(int(v) for v in size)
    if tuple(targets.shape[2:]) == target_size:
        return targets
    factors = [source // wanted for source, wanted in zip(targets.shape[2:], target_size)]
    if any(factor < 1 for factor in factors) or [
        wanted * factor for wanted, factor in zip(target_size, factors)
    ] != list(targets.shape[2:]):
        raise ValueError(
            f"cannot max-pool {tuple(targets.shape[2:])} down to {target_size} with an "
            "integer factor"
        )
    return F.max_pool3d(targets.to(torch.float32), kernel_size=factors, stride=factors)


def deep_supervision_loss(
    predictions: Sequence[Tensor],
    targets: Tensor,
    weights: Sequence[float],
    *,
    lambda_dice: float = 1.0,
    lambda_bce: float = 1.0,
    smooth: float = DEFAULT_SMOOTH,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted sum of :func:`segmentation_loss` over the deep-supervision maps.

    Args:
        predictions: logits, coarse to fine; the last one is full resolution.
        targets: full-resolution binary targets, downsampled per scale here.
        weights: one weight per prediction, in the same order.
    """
    if len(predictions) != len(weights):
        raise ValueError(f"got {len(predictions)} predictions and {len(weights)} weights")
    total = torch.zeros((), dtype=torch.float32, device=targets.device)

    components: dict[str, float] = {}
    for index, (prediction, weight) in enumerate(zip(predictions, weights)):
        scaled_targets = downsample_targets(targets, prediction.shape[2:])
        loss, parts = segmentation_loss(
            prediction,
            scaled_targets,
            lambda_dice=lambda_dice,
            lambda_bce=lambda_bce,
            smooth=smooth,
        )
        total = total + weight * loss
        resolution = int(prediction.shape[-1])
        components[f"dice@{resolution}"] = parts["dice"]
        components[f"bce@{resolution}"] = parts["bce"]
    components["total"] = float(total.detach())
    return total, components
