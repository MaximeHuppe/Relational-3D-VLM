"""Stage A: full-volume promptable shape segmenter.

A 3D residual U-Net that takes the scene image ``scene_volume`` and a set of
shape-name prompts, and returns one mask logit volume per prompt. This is the
synthetic analogue of the future MRI anatomy segmenter that will supply anchor
masks, and it is what Phase 4 uses to extract the three requested anchors by
name.

Architecture (adapted from ``docs/flowchart/phase1_encoder_decoder.drawio``)::

    scene_volume [B, 1, 64, 64, 64]
      STEM       Conv-IN-ReLU + ResBlock  1 -> 32, s=1     skip1  32 x 64^3
      Stage 1    Conv-IN-ReLU + ResBlock 32 -> 64, s=2     skip2  64 x 32^3
      Stage 2    Conv-IN-ReLU + ResBlock 64 ->128, s=2     skip3 128 x 16^3
      Bottleneck Conv-IN-ReLU + ResBlock 128->256, s=2           256 x  8^3

    prompt_ids [B, N_T] -> ShapeNamePromptEncoder -> Q [B, N_T, 256]
      PromptDecoder: Q attends over the flattened bottleneck
        Q: no positional encoding (what)
        K: bottleneck + decomposed 3D positional encoding (where)
        V: bottleneck, unmodified
      aligned_queries = LayerNorm(MHA(Q, K, V) + Q)          [B, N_T, 256]

      Decode 1  8^3 -> 16^3, concat skip3, 256+128 -> 128    pred[0] DS 0.1
      Decode 2 16^3 -> 32^3, concat skip2, 128+ 64 ->  64    pred[1] DS 0.3
      Decode 3 32^3 -> 64^3, concat skip1,  64+ 32 ->  32    pred[2] DS 0.6 *

    * inference uses the full-resolution map only.

The same aligned queries are reused at all three decoder scales. At each scale a
:class:`StageVLFusionBlock` produces the mask logits by a scaled dot product
between the projected queries and the (unmodulated) visual features, plus a
per-query logit bias.

Two deliberate departures from the reference figure, both documented in
``docs/stage_a_architecture.md``:

1. the volume is 64^3 rather than 128^3, so the bottleneck grid is 8^3 - the
   512-token budget CLAUDE.md sets for global cross-attention;
2. the frozen PubMedBERT name embeddings are replaced by this project's
   closed-vocabulary shape-name table (:mod:`src.models.prompt_encoder`).

Because queries never interact with each other and the visual features are never
gated by the prompt set, the logits of a given prompt do not depend on which
other prompts are in the batch. Stage A can therefore be trained on all ten
names and queried with any subset, which is exactly what anchor extraction
needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.config import load_config
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY
from src.models.blocks import (
    DecomposedPositionalEncoding3D,
    EncoderStage,
    UpBlock,
    flatten_spatial,
    prior_logit,
)
from src.models.prompt_encoder import ShapeNamePromptEncoder


@dataclass(frozen=True)
class ShapeSegmenterConfig:
    """Resolved Stage A architecture, from ``configs/model.yaml``."""

    in_channels: int = 1
    num_prompts: int = 10
    stem_channels: int = 32
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2)
    resblocks: tuple[int, ...] = (1, 1, 1)
    kernel_size: int = 3
    input_resolution: int = 64
    bottleneck_resolution: int = 8
    decoder_channels: tuple[int, ...] = (128, 64, 32)
    align_corners: bool = True
    text_dim: int = 256
    embed_dim: int = 256
    heads: int = 4
    head_dim: int = 64
    per_query_bias: bool = True
    bias_init: str = "prior"
    prior_foreground_fraction: float = 0.0016
    deep_supervision_weights: tuple[float, ...] = (0.1, 0.3, 0.6)

    def __post_init__(self) -> None:
        if len(self.encoder_channels) != 4:
            raise ValueError(f"expected 4 encoder channel widths, got {self.encoder_channels}")
        if self.encoder_channels[0] != self.stem_channels:
            raise ValueError(
                f"stem_channels {self.stem_channels} must equal encoder_channels[0] "
                f"{self.encoder_channels[0]}"
            )
        if len(self.decoder_channels) != 3:
            raise ValueError(f"expected 3 decoder channel widths, got {self.decoder_channels}")
        if self.heads * self.head_dim != self.embed_dim:
            raise ValueError(
                f"heads * head_dim ({self.heads} * {self.head_dim}) must equal embed_dim "
                f"{self.embed_dim}"
            )
        expected_bottleneck = self.input_resolution // (2 ** len(self.strides))
        if expected_bottleneck != self.bottleneck_resolution:
            raise ValueError(
                f"{len(self.strides)} stride-2 stages take {self.input_resolution}^3 to "
                f"{expected_bottleneck}^3, but bottleneck_resolution is "
                f"{self.bottleneck_resolution}"
            )
        if len(self.deep_supervision_weights) != 3:
            raise ValueError("deep supervision needs one weight per decoder stage")

    @property
    def decoder_resolutions(self) -> tuple[int, int, int]:
        """Output grid of each decoder stage, coarse to fine."""
        base = self.bottleneck_resolution
        return (base * 2, base * 4, base * 8)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        profile: str = "default",
        input_resolution: int | None = None,
    ) -> "ShapeSegmenterConfig":
        """Build from ``configs/model.yaml``; ``profile='smoke'`` applies the overrides."""
        model_config = dict(config or load_config("model"))
        stage = _deep_merge(
            dict(model_config["stage_a"]),
            dict(model_config.get("smoke", {}).get("stage_a", {})) if profile == "smoke" else {},
        )
        prompt_encoder = stage["prompt_encoder"]
        prompt_decoder = stage["prompt_decoder"]
        deep_supervision = stage["deep_supervision"]
        resolution = int(input_resolution or stage["resolutions"][0])
        strides = tuple(int(v) for v in stage["strides"])
        return cls(
            in_channels=int(stage["in_channels"]),
            num_prompts=int(stage["num_prompts"]),
            stem_channels=int(stage["stem_channels"]),
            encoder_channels=tuple(int(v) for v in stage["encoder_channels"]),
            strides=strides,
            resblocks=tuple(int(v) for v in stage["resblocks"]),
            kernel_size=int(stage["kernel_size"]),
            input_resolution=resolution,
            bottleneck_resolution=resolution // (2 ** len(strides)),
            decoder_channels=tuple(int(v) for v in stage["decoder_channels"]),
            align_corners=bool(stage["align_corners"]),
            text_dim=int(prompt_encoder["text_dim"]),
            embed_dim=int(prompt_encoder["embed_dim"]),
            heads=int(prompt_decoder["heads"]),
            head_dim=int(prompt_decoder["head_dim"]),
            per_query_bias=bool(stage["fusion"]["per_query_bias"]),
            bias_init=str(stage["fusion"].get("bias_init", "prior")),
            prior_foreground_fraction=float(
                stage["fusion"].get("prior_foreground_fraction", 0.0016)
            ),
            deep_supervision_weights=tuple(float(v) for v in deep_supervision["weights"]),
        )


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` onto ``base`` (used for the smoke profile)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


class PromptDecoder(nn.Module):
    """Language queries attend over the flattened bottleneck (panel d).

    The asymmetry is intentional and is what makes the queries *spatially*
    aligned: the query says *what*, the key says *where*, the value carries the
    unmodified visual content.
    """

    def __init__(
        self,
        embed_dim: int,
        visual_channels: int,
        grid_shape: Sequence[int],
        heads: int,
    ) -> None:
        super().__init__()
        if visual_channels != embed_dim:
            raise ValueError(
                f"the bottleneck width ({visual_channels}) must equal embed_dim ({embed_dim})"
            )
        self.embed_dim = embed_dim
        self.positional_encoding = DecomposedPositionalEncoding3D(grid_shape, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: Tensor, visual: Tensor) -> Tensor:
        """``Q [B, N_T, C]``, ``visual [B, C, D, H, W]`` -> aligned ``[B, N_T, C]``."""
        values = flatten_spatial(visual)  # [B, DHW, C]
        keys = values + self.positional_encoding(visual.shape[2:])
        attended, _ = self.attention(queries, keys, values, need_weights=False)
        return self.norm(attended + queries)


class StageVLFusionBlock(nn.Module):
    """Per-prompt mask head: scaled dot product plus a per-query logit bias (panel e).

    Visual features are returned untouched - the block only *reads* them - so the
    logits of one prompt are independent of the other prompts in the request.

    ``bias_init`` is this project's one departure from the reference figure's
    mask head. The figure initialises the bias to zero, which starts every voxel
    at p = 0.5. Here a single shape covers about 0.16% of the volume, so a
    zero-init head spends its early training pushing 262,000 background logits
    down before the Dice term carries any usable gradient; measured on the smoke
    corpus that plateau costs roughly an order of magnitude in convergence
    (train Dice 0.18 vs 0.66 after the same 150 steps). Initialising the bias to
    ``prior_logit(expected foreground fraction)`` - the standard focal-loss prior
    trick - starts the head at the right base rate instead. The bias stays
    learnable and its weight is still zero-initialised, so nothing else changes.
    """

    def __init__(
        self,
        embed_dim: int,
        visual_channels: int,
        per_query_bias: bool = True,
        *,
        bias_init: str = "prior",
        prior_foreground_fraction: float = 0.0016,
    ) -> None:
        super().__init__()
        if bias_init not in ("prior", "zeros"):
            raise ValueError(f"bias_init must be 'prior' or 'zeros', got {bias_init!r}")
        self.visual_channels = visual_channels
        self.bias_init = bias_init
        self.project = nn.Sequential(
            nn.Linear(embed_dim, visual_channels),
            nn.ReLU(inplace=True),
            nn.Linear(visual_channels, visual_channels),
        )
        self.bias_head = nn.Linear(visual_channels, 1) if per_query_bias else None
        if self.bias_head is not None:
            nn.init.zeros_(self.bias_head.weight)
            nn.init.constant_(
                self.bias_head.bias,
                prior_logit(prior_foreground_fraction) if bias_init == "prior" else 0.0,
            )

    def forward(self, queries: Tensor, visual: Tensor) -> Tensor:
        """``Q [B, N_T, E]``, ``visual [B, C, D, H, W]`` -> logits ``[B, N_T, D, H, W]``."""
        projected = F.layer_norm(self.project(queries), (self.visual_channels,))
        batch, channels, depth, height, width = visual.shape
        flat = visual.reshape(batch, channels, -1)  # [B, C, DHW]
        logits = torch.bmm(projected, flat) / math.sqrt(self.visual_channels)
        if self.bias_head is not None:
            logits = logits + self.bias_head(projected)
        return logits.reshape(batch, -1, depth, height, width)


@dataclass
class ShapeSegmenterOutput:
    """Forward result: deep-supervision maps plus the inference map."""

    logits: Tensor  # full resolution, [B, N_T, D, H, W]
    deep_supervision: list[Tensor] = field(default_factory=list)  # coarse -> fine

    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits)

    def binary_mask(self, threshold: float = 0.5) -> Tensor:
        return (self.probabilities() >= threshold).to(torch.uint8)


class ShapeSegmenter(nn.Module):
    """The Stage A network."""

    def __init__(self, config: ShapeSegmenterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ShapeSegmenterConfig()
        cfg = self.config
        c1, c2, c3, c4 = cfg.encoder_channels

        self.stem = EncoderStage(cfg.in_channels, c1, stride=1, num_blocks=1, kernel_size=cfg.kernel_size)
        self.stage1 = EncoderStage(c1, c2, stride=cfg.strides[0], num_blocks=cfg.resblocks[0], kernel_size=cfg.kernel_size)
        self.stage2 = EncoderStage(c2, c3, stride=cfg.strides[1], num_blocks=cfg.resblocks[1], kernel_size=cfg.kernel_size)
        self.bottleneck = EncoderStage(c3, c4, stride=cfg.strides[2], num_blocks=cfg.resblocks[2], kernel_size=cfg.kernel_size)

        self.prompt_encoder = ShapeNamePromptEncoder(
            num_shapes=cfg.num_prompts, text_dim=cfg.text_dim, embed_dim=cfg.embed_dim
        )
        grid = (cfg.bottleneck_resolution,) * 3
        self.prompt_decoder = PromptDecoder(cfg.embed_dim, c4, grid, cfg.heads)

        d1, d2, d3 = cfg.decoder_channels
        self.decode1 = UpBlock(c4, c3, d1, cfg.kernel_size, cfg.align_corners)
        self.decode2 = UpBlock(d1, c2, d2, cfg.kernel_size, cfg.align_corners)
        self.decode3 = UpBlock(d2, c1, d3, cfg.kernel_size, cfg.align_corners)

        fusion_kwargs = {
            "bias_init": cfg.bias_init,
            "prior_foreground_fraction": cfg.prior_foreground_fraction,
        }
        self.fusion1 = StageVLFusionBlock(cfg.embed_dim, d1, cfg.per_query_bias, **fusion_kwargs)
        self.fusion2 = StageVLFusionBlock(cfg.embed_dim, d2, cfg.per_query_bias, **fusion_kwargs)
        self.fusion3 = StageVLFusionBlock(cfg.embed_dim, d3, cfg.per_query_bias, **fusion_kwargs)

    # -- helpers -----------------------------------------------------------
    def default_prompt_ids(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        """All ten shape names, in canonical vocabulary order."""
        ids = torch.arange(self.config.num_prompts, dtype=torch.long, device=device)
        return ids.unsqueeze(0).expand(batch_size, -1).contiguous()

    def prompt_ids_for_names(
        self, names: Sequence[str], batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> Tensor:
        """Build ``prompt_ids`` for a named subset, e.g. the three anchors of a prompt."""
        ids = torch.tensor(
            ShapeNamePromptEncoder.ids_for_names(list(names)), dtype=torch.long, device=device
        )
        return ids.unsqueeze(0).expand(batch_size, -1).contiguous()

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        scene_volume: Tensor,
        prompt_ids: Tensor | None = None,
        *,
        deep_supervision: bool = True,
    ) -> ShapeSegmenterOutput:
        """Segment the requested shape names from the scene image.

        Args:
            scene_volume: ``[B, 1, D, H, W]`` float tensor.
            prompt_ids: ``[B, N_T]`` zero-based shape indices; defaults to all ten.
            deep_supervision: also return the 1/4 and 1/2 resolution maps.
        """
        if scene_volume.ndim != 5:
            raise ValueError(f"scene_volume must be [B, C, D, H, W], got {tuple(scene_volume.shape)}")
        if prompt_ids is None:
            prompt_ids = self.default_prompt_ids(scene_volume.shape[0], scene_volume.device)
        if prompt_ids.shape[0] != scene_volume.shape[0]:
            raise ValueError(
                f"batch mismatch: scene_volume {scene_volume.shape[0]} vs prompt_ids "
                f"{prompt_ids.shape[0]}"
            )

        skip1 = self.stem(scene_volume)       # C1 @ full resolution
        skip2 = self.stage1(skip1)            # C2 @ 1/2
        skip3 = self.stage2(skip2)            # C3 @ 1/4
        bottleneck = self.bottleneck(skip3)   # C4 @ 1/8

        queries = self.prompt_encoder(prompt_ids)
        aligned = self.prompt_decoder(queries, bottleneck)

        features1 = self.decode1(bottleneck, skip3)  # 1/4
        features2 = self.decode2(features1, skip2)   # 1/2
        features3 = self.decode3(features2, skip1)   # full

        logits = self.fusion3(aligned, features3)
        maps: list[Tensor] = []
        if deep_supervision:
            maps = [
                self.fusion1(aligned, features1),
                self.fusion2(aligned, features2),
                logits,
            ]
        return ShapeSegmenterOutput(logits=logits, deep_supervision=maps)

    @torch.no_grad()
    def predict_anchor_masks(
        self,
        scene_volume: Tensor,
        anchor_shape_names: Sequence[str],
        *,
        threshold: float = 0.5,
    ) -> Tensor:
        """Phase 4 entry point: extract named anchor masks from a scene.

        Returns ``[B, len(anchor_shape_names), D, H, W]`` binary masks, in the
        order the names were given - which is the prompt's anchor order.
        """
        self.eval()
        prompt_ids = self.prompt_ids_for_names(
            anchor_shape_names, scene_volume.shape[0], scene_volume.device
        )
        output = self(scene_volume, prompt_ids, deep_supervision=False)
        return output.binary_mask(threshold)


def build_shape_segmenter(
    profile: str = "default", *, input_resolution: int | None = None
) -> ShapeSegmenter:
    """Construct Stage A from ``configs/model.yaml``."""
    config = ShapeSegmenterConfig.from_config(profile=profile, input_resolution=input_resolution)
    if config.num_prompts != len(SHAPE_NAMES):
        raise ValueError(
            f"num_prompts {config.num_prompts} must match the {len(SHAPE_VOCABULARY)}-shape vocabulary"
        )
    return ShapeSegmenter(config)
