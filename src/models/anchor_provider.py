"""Where Stage B's three anchor channels come from.

Stage B is deliberately agnostic about this. It consumes three ordered binary
channels and measures its own geometry from them, so the same trained model can
be run two ways:

``oracle``
    the ground-truth masks, materialised from ``instance_labels`` in
    ``data/processed/scenes/*.npz`` by keeping only the three anchor structures
    the prompt names. This isolates the relational architecture from Stage A's
    errors and is the primary proof-of-concept measurement (Phase 3).

``predicted``
    Stage A is run on the scene volume and asked for exactly the three shape
    names in the prompt, in prompt order. This is the end-to-end setting and the
    future MRI setting, where an anatomy segmenter supplies the anchors
    (Phase 4).

The two providers are interchangeable at the call site::

    provider = build_anchor_provider("oracle")            # or "predicted"
    masks = provider(batch)                               # [B, 3, D, H, W]
    output = model(**stage_b_model_inputs(batch, masks))

Note what the predicted provider does *not* do: the scene volume it consumes
goes into Stage A and stops there. Stage B still receives only three mask
channels, so the forbidden-field contract holds in both modes - which is the
whole point of routing the choice through a provider instead of widening Stage
B's inputs.

The predicted provider also scores its own output against the ground-truth
channels when the batch carries them, so a run can report *why* Stage B degraded
(:meth:`PredictedAnchorProvider.anchor_quality`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor, nn

from src.evaluation.metrics import dice_score, iou_score
from src.models.shape_segmenter import ShapeSegmenter, build_shape_segmenter
from src.training.checkpointing import load_checkpoint

#: The two anchor sources, as named in ``configs/train.yaml``.
ANCHOR_SOURCES: tuple[str, ...] = ("oracle", "predicted")

#: Config spelling -> CLI spelling.
_CONFIG_ALIASES = {"stage_a_prediction": "predicted", "ground_truth": "oracle"}


class AnchorProviderError(RuntimeError):
    """Raised when an anchor source cannot be built or applied."""


class AnchorProvider(Protocol):
    """Anything that can turn a batch into three ordered anchor channels."""

    source: str

    def __call__(self, batch: Mapping[str, Any]) -> Tensor: ...


@dataclass
class OracleAnchorProvider:
    """Ground-truth anchor channels, straight from the dataset.

    ``ExampleDataset`` already keeps only the three anchor structures of the
    example, in clause order, so this is the identity on ``anchor_masks``.
    """

    source: str = "oracle"

    def __call__(self, batch: Mapping[str, Any]) -> Tensor:
        if "anchor_masks" not in batch:
            raise AnchorProviderError("the batch carries no ground-truth anchor_masks")
        return batch["anchor_masks"].to(torch.float32)

    def summary(self) -> dict[str, Any]:
        return {"anchor_source": self.source}


@dataclass
class PredictedAnchorProvider:
    """Anchor channels segmented by Stage A from the scene volume.

    Args:
        model: a trained Stage A segmenter, used in eval mode under ``no_grad``.
        threshold: probability cut for the binary anchor masks.
        checkpoint: path the weights came from, recorded in the run metadata.
        track_quality: accumulate Dice/IoU of the predicted channels against the
            ground-truth ones whenever the batch carries them.
    """

    model: ShapeSegmenter
    threshold: float = 0.5
    checkpoint: Path | None = None
    track_quality: bool = True
    source: str = "predicted"
    _dice: list[float] = field(default_factory=list, repr=False)
    _iou: list[float] = field(default_factory=list, repr=False)
    _empty: int = field(default=0, repr=False)
    _channels: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def to(self, device: torch.device | str) -> "PredictedAnchorProvider":
        self.model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, batch: Mapping[str, Any]) -> Tensor:
        if "scene_volume" not in batch:
            raise AnchorProviderError(
                "predicted anchors need the scene volume; build the dataset with "
                "include_scene_volume=True"
            )
        scene_volume = batch["scene_volume"]
        # anchor_shape_ids are already Stage A's zero-based prompt ids, in
        # clause order, so the predicted channels come back in that same order.
        prompt_ids = batch["anchor_shape_ids"].to(scene_volume.device)
        output = self.model(scene_volume, prompt_ids, deep_supervision=False)
        masks = (torch.sigmoid(output.logits.float()) >= self.threshold).to(torch.float32)
        if self.track_quality and "anchor_masks" in batch:
            self._accumulate(masks, batch["anchor_masks"].to(masks.device))
        return masks

    def _accumulate(self, predicted: Tensor, reference: Tensor) -> None:
        dice = dice_score(predicted, reference)
        iou = iou_score(predicted, reference)
        self._dice.extend(dice.flatten().tolist())
        self._iou.extend(iou.flatten().tolist())
        self._empty += int((predicted.flatten(2).sum(-1) == 0).sum())
        self._channels += int(predicted.shape[0] * predicted.shape[1])

    def anchor_quality(self) -> dict[str, float]:
        """How good the anchors themselves were, per channel.

        Reported next to the Stage B metrics: a drop against the oracle run is
        only interpretable if the anchor error that caused it is on the record.
        """
        if not self._dice:
            return {}
        return {
            "anchor_dice": sum(self._dice) / len(self._dice),
            "anchor_iou": sum(self._iou) / len(self._iou),
            "empty_anchor_fraction": self._empty / max(self._channels, 1),
            "channels": float(self._channels),
        }

    def reset(self) -> None:
        self._dice.clear()
        self._iou.clear()
        self._empty = 0
        self._channels = 0

    def summary(self) -> dict[str, Any]:
        return {
            "anchor_source": self.source,
            "stage_a_checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "threshold": self.threshold,
            **self.anchor_quality(),
        }


def load_stage_a(
    checkpoint: Path | str,
    *,
    device: torch.device | str = "cpu",
    profile: str | None = None,
    input_resolution: int | None = None,
) -> ShapeSegmenter:
    """Rebuild Stage A from a checkpoint written by ``train_shape_segmenter.py``.

    The width profile is read from the checkpoint's own metadata when it is not
    given, so a smoke-profile checkpoint cannot be loaded into a default-profile
    skeleton and fail with an opaque shape error.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise AnchorProviderError(
            f"no Stage A checkpoint at {path}. Train one first:\n"
            f"  .venv/bin/python scripts/train_shape_segmenter.py"
        )
    payload = load_checkpoint(path, map_location=device)
    metadata = payload.get("metadata", {})
    settings = metadata.get("settings", {}) if isinstance(metadata, Mapping) else {}
    resolved_profile = profile or str(settings.get("model_profile", "default"))
    model_config = payload.get("model_config") or {}
    resolution = int(
        input_resolution
        or model_config.get("input_resolution")
        or 64
    )
    model = build_shape_segmenter(resolved_profile, input_resolution=resolution)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def build_anchor_provider(
    source: str = "oracle",
    *,
    checkpoint: Path | str | None = None,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    profile: str | None = None,
    input_resolution: int | None = None,
    model: nn.Module | None = None,
) -> AnchorProvider:
    """Build the anchor source named on the command line or in the config.

    Args:
        source: ``oracle`` (ground-truth masks) or ``predicted`` (Stage A).
            The config spellings ``ground_truth`` and ``stage_a_prediction`` are
            accepted as aliases.
        checkpoint: Stage A weights; required for ``predicted`` unless ``model``
            is given.
    """
    name = _CONFIG_ALIASES.get(str(source), str(source))
    if name not in ANCHOR_SOURCES:
        raise AnchorProviderError(
            f"anchor source must be one of {ANCHOR_SOURCES}, got {source!r}"
        )
    if name == "oracle":
        return OracleAnchorProvider()
    if model is None:
        if checkpoint is None:
            raise AnchorProviderError(
                "predicted anchors need --stage-a-checkpoint (a Phase 1 checkpoint)"
            )
        model = load_stage_a(
            checkpoint, device=device, profile=profile, input_resolution=input_resolution
        )
    return PredictedAnchorProvider(
        model=model,  # type: ignore[arg-type]
        threshold=threshold,
        checkpoint=Path(checkpoint) if checkpoint else None,
    ).to(device)
