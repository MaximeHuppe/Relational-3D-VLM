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

The ``BCEWithLogits`` term may optionally be replaced by its focal variant
(``bce_variant="focal"``, see :func:`focal_bce_loss`). That is a deliberate
departure from the objective as literally written above, motivated in
``docs/stage_b_tuning_plan.md`` §3.3: at Stage B's 1:552 foreground imbalance the
plain term contributes ~2% of the loss regardless of ``lambda_bce``. The default
is ``plain``, so nothing changes unless a config asks for it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_SMOOTH = 1.0

#: Cross-entropy variants `segmentation_loss` accepts. ``plain`` is the default
#: and reproduces every run made before focal loss existed, bit for bit.
BCE_VARIANTS: tuple[str, ...] = ("plain", "focal")
DEFAULT_BCE_VARIANT = "plain"

#: Focusing exponent. 0 reduces focal BCE exactly to plain BCE.
DEFAULT_FOCAL_GAMMA = 2.0
#: Weight on the POSITIVE class; ``1 - alpha`` weights the negative one.
#: ``None`` disables the class balancing and keeps only the focusing term.
DEFAULT_FOCAL_ALPHA: float | None = 0.25


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


def focal_bce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    alpha: float | None = DEFAULT_FOCAL_ALPHA,
) -> Tensor:
    """Focal binary cross-entropy (Lin et al., 2017), averaged over every element.

    Plain BCE weights every voxel equally, which is why it is worth so little
    here: a Stage B target occupies ~474 of 262,144 voxels (0.18%, a 1:552
    imbalance), so the mean is dominated by background that the model already
    predicts correctly, and the whole term lands at ~2% of the objective however
    large ``lambda_bce`` is set. Focal loss fixes that by construction rather
    than by multiplying through a constant::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where ``p_t`` is the probability assigned to the *correct* class. The
    ``(1 - p_t)^gamma`` factor is what does the work - at ``gamma=2`` and the
    confidence levels this model actually reaches:

    ===========================================  =====  =============
    voxel                                        p_t    (1 - p_t)^2
    ===========================================  =====  =============
    easy background (predicted 0.001, true 0)    0.999  1e-6
    confidently-wrong FP (predicted 0.996,       0.004  0.992
    true 0)
    ===========================================  =====  =============

    so the ~261,000 easy background voxels are effectively discarded and the
    gradient concentrates on the mistakes.

    Args:
        logits: ``[B, C, D, H, W]`` raw logits.
        targets: ``[B, C, D, H, W]`` binary targets in ``{0, 1}``.
        gamma: focusing exponent, ``>= 0``. ``0`` reduces this to plain BCE
            (modulated by ``alpha``), which is the property the tests pin.
        alpha: weight on the positive class, in ``[0, 1]``; ``1 - alpha`` weights
            the negative one. ``None`` disables class balancing entirely. The
            usual default of 0.25 therefore emphasises *negatives*, which suits a
            model whose errors are predominantly false positives; raise it toward
            0.5 if recall collapses.

    Note:
        Focal BCE is numerically much smaller than plain BCE - everything easy is
        down-weighted toward zero - so it needs its own ``lambda_bce``. Reusing
        the plain-BCE weight silently recreates the dilution this exists to fix.
        The magnitude is reported in the loss components so it can be calibrated
        against the Dice term after one epoch.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1] or be None, got {alpha}")
    if logits.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
        )
    logits = _as_loss_dtype(logits)
    targets = targets.to(logits.dtype)
    # `binary_cross_entropy_with_logits` is the log-sum-exp form, so a large
    # positive logit is never exponentiated and the term stays finite.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


def cross_entropy_term(
    logits: Tensor,
    targets: Tensor,
    *,
    variant: str = DEFAULT_BCE_VARIANT,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    focal_alpha: float | None = DEFAULT_FOCAL_ALPHA,
) -> Tensor:
    """Dispatch to :func:`bce_loss` or :func:`focal_bce_loss` by name.

    Unknown names are rejected rather than silently falling back to plain BCE: a
    typo in ``configs/train.yaml`` must not quietly train the wrong objective.
    """
    if variant not in BCE_VARIANTS:
        raise ValueError(f"bce variant must be one of {BCE_VARIANTS}, got {variant!r}")
    if variant == "focal":
        return focal_bce_loss(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
    return bce_loss(logits, targets)


def segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    lambda_dice: float = 1.0,
    lambda_bce: float = 1.0,
    smooth: float = DEFAULT_SMOOTH,
    bce_variant: str = DEFAULT_BCE_VARIANT,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    focal_alpha: float | None = DEFAULT_FOCAL_ALPHA,
) -> tuple[Tensor, dict[str, float]]:
    """``lambda_dice * DiceLoss + lambda_bce * CrossEntropy``.

    ``bce_variant`` selects the cross-entropy term: ``plain`` (the default, and
    the objective stated in CLAUDE.md) or ``focal``. The component is reported
    under the key ``"bce"`` either way, so the logged series stays comparable
    across runs; which variant produced it is recorded in the run's saved
    configuration.

    Returns the total and a dict of the detached components, for logging.
    """
    dice = dice_loss(logits, targets, smooth=smooth)
    bce = cross_entropy_term(
        logits,
        targets,
        variant=bce_variant,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
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
    bce_variant: str = DEFAULT_BCE_VARIANT,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    focal_alpha: float | None = DEFAULT_FOCAL_ALPHA,
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
            bce_variant=bce_variant,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        total = total + weight * loss
        resolution = int(prediction.shape[-1])
        components[f"dice@{resolution}"] = parts["dice"]
        components[f"bce@{resolution}"] = parts["bce"]
    components["total"] = float(total.detach())
    return total, components
