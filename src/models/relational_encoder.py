"""Stage B visual encoder over the ordered anchor-mask channels.

``3 x 64^3 -> 32^3 -> 16^3 -> 8^3``. The 8^3 bottleneck holds 512 spatial
tokens, which is what makes global cross-attention affordable; full-resolution
attention over 262,144 tokens is explicitly forbidden by CLAUDE.md and never
happens here.

The encoder sees *only* the anchor channels. There is no scene volume, no
instance labels and no target mask anywhere in this file - the target is not an
input to be encoded, it is the region the relations intersect.

Normalised world coordinates ``(x, y, z)`` are concatenated to the features at
every scale (64, 32, 16, 8), recomputed from the world frame at each resolution
by :func:`src.models.blocks.normalized_world_grid` rather than read off local
tensor indices. Without them the network would have no way to tell "anterior"
from "posterior": the anchor channels alone are translation-ambiguous, and
`medial`/`lateral` is defined against the volume centre plane, which only exists
in world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from src.models.blocks import EncoderStage, normalized_world_grid

#: Number of coordinate channels appended at every scale: (x, y, z).
NUM_COORDINATE_CHANNELS = 3


@dataclass
class EncoderFeatures:
    """The four feature maps Stage B's decoder and grounding branches consume."""

    stem: Tensor        # [B, C1, D, H, W]
    stage1: Tensor      # [B, C2, D/2, H/2, W/2]
    stage2: Tensor      # [B, C3, D/4, H/4, W/4]
    bottleneck: Tensor  # [B, C4, D/8, H/8, W/8]

    @property
    def skips(self) -> tuple[Tensor, Tensor, Tensor]:
        """Skips in decoder order: 1/4, 1/2, full."""
        return (self.stage2, self.stage1, self.stem)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Bottleneck grid ``(D', H', W')``."""
        return tuple(int(v) for v in self.bottleneck.shape[2:])  # type: ignore[return-value]


class RelationalEncoder(nn.Module):
    """Encode the anchor channels into a feature pyramid with world coordinates.

    Args:
        in_channels: anchor-mask channels - 3 ordered channels, or 1 for the
            union-mask ablation. The prompt-only baseline keeps three channels
            and zeroes them, so the encoder sees the coordinate grid alone
            without a second code path.
        channels: widths of the four scales, ``(C1, C2, C3, C4)``.
        input_resolution: the resolution the world frame is defined on.
        spacing: world units per voxel, ordered ``(x, y, z)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        *,
        kernel_size: int = 3,
        resblocks: Sequence[int] = (1, 1, 1),
        activation: str = "leaky_relu",
        input_resolution: int = 64,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        coordinate_features: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(int(v) for v in channels)
        if len(widths) != 4:
            raise ValueError(f"expected 4 encoder widths (C1..C4), got {widths}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        blocks = tuple(int(v) for v in resblocks)
        if len(blocks) != 3:
            raise ValueError(f"expected one resblock count per stride-2 stage, got {resblocks}")

        self.in_channels = int(in_channels)
        self.channels = widths
        self.input_resolution = int(input_resolution)
        self.spacing = tuple(float(v) for v in spacing)
        self.coordinate_features = bool(coordinate_features)
        self.extra = NUM_COORDINATE_CHANNELS if self.coordinate_features else 0

        c1, c2, c3, c4 = widths
        self.stem = EncoderStage(
            self.in_channels + self.extra, c1, stride=1, num_blocks=1,
            kernel_size=kernel_size, activation=activation,
        )
        self.stage1 = EncoderStage(
            c1 + self.extra, c2, stride=2, num_blocks=blocks[0],
            kernel_size=kernel_size, activation=activation,
        )
        self.stage2 = EncoderStage(
            c2 + self.extra, c3, stride=2, num_blocks=blocks[1],
            kernel_size=kernel_size, activation=activation,
        )
        self.bottleneck = EncoderStage(
            c3 + self.extra, c4, stride=2, num_blocks=blocks[2],
            kernel_size=kernel_size, activation=activation,
        )

    # -- coordinates -------------------------------------------------------
    def world_grid(self, features: Tensor, full_shape: Sequence[int]) -> Tensor:
        """Normalised world ``(x, y, z)`` for the grid ``features`` lives on."""
        grid = normalized_world_grid(
            features.shape[2:],
            full_shape,
            self.spacing,
            device=features.device,
            dtype=features.dtype,
        )
        return grid.expand(features.shape[0], -1, -1, -1, -1)

    def _with_coordinates(self, features: Tensor, full_shape: Sequence[int]) -> Tensor:
        if not self.coordinate_features:
            return features
        return torch.cat([features, self.world_grid(features, full_shape)], dim=1)

    # -- forward -----------------------------------------------------------
    def forward(
        self, anchor_masks: Tensor, full_shape: Sequence[int] | None = None
    ) -> EncoderFeatures:
        """``[B, in_channels, D, H, W]`` anchor channels -> :class:`EncoderFeatures`.

        ``full_shape`` names the volume the world frame is tied to; it defaults
        to the input's own shape and only differs if a crop is ever introduced.
        """
        if anchor_masks.ndim != 5:
            raise ValueError(f"anchor_masks must be [B, C, D, H, W], got {tuple(anchor_masks.shape)}")
        if anchor_masks.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} anchor channel(s), got {anchor_masks.shape[1]}"
            )
        shape = tuple(full_shape) if full_shape is not None else tuple(anchor_masks.shape[2:])

        stem = self.stem(self._with_coordinates(anchor_masks, shape))
        stage1 = self.stage1(self._with_coordinates(stem, shape))
        stage2 = self.stage2(self._with_coordinates(stage1, shape))
        bottleneck = self.bottleneck(self._with_coordinates(stage2, shape))
        return EncoderFeatures(stem=stem, stage1=stage1, stage2=stage2, bottleneck=bottleneck)
