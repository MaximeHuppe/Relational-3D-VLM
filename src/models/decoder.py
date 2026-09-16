"""Stage B decoder: ``8^3 -> 16^3 -> 32^3 -> 64^3``.

At each stage: trilinear upsample, concatenate the matching encoder skip,
inject the relation-fused context, and refine with 3D convolutions. The head
returns one target logit volume at the input resolution; evaluation also asks
for sigmoid probabilities and a thresholded binary mask.

Conditioning at 16^3 and 32^3 is FiLM - a per-channel affine modulation
predicted from the three fused clause tokens, concatenated in clause order so
the context keeps the correspondence between slot, direction and anchor. FiLM is
what CLAUDE.md asks for here ("lightweight FiLM/gating or cross-attention
conditioning at the 16^3 and 32^3 decoder stages. Do not use expensive
full-resolution global attention"): attention at 32^3 would mean 32,768 query
tokens per sample, sixty-four times the bottleneck budget.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from src.models.blocks import (
    UpBlock,
    conv3d,
    downsample_masks,
    make_activation,
    norm3d,
    normalized_world_grid,
    prior_logit,
)


class FiLM3d(nn.Module):
    """Per-channel affine modulation of a 3D feature map from a context vector.

    ``y = (1 + gamma(context)) * x + beta(context)``. Both projections are
    zero-initialised, so the module starts as the identity and the decoder is
    never handed a randomly scrambled feature map at step 0.
    """

    def __init__(self, context_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.project = nn.Linear(context_dim, 2 * channels)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        gamma, beta = self.project(context).chunk(2, dim=-1)
        shape = (features.shape[0], self.channels, 1, 1, 1)
        return (1.0 + gamma.reshape(shape)) * features + beta.reshape(shape)


class RelationalDecoder(nn.Module):
    """The Stage B decoder and target head.

    Args:
        bottleneck_channels: width of the conditioned bottleneck grid.
        skip_channels: encoder widths at 1/4, 1/2 and full resolution.
        decoder_channels: output widths of the three decoder stages.
        context_dim: width of the flattened clause context.
        condition_at: decoder output resolutions that get FiLM (16 and 32).
        occupancy_at: decoder output resolutions that concatenate the occupancy
            channel after the skip merge. Default 16/32/64; never 8, which is
            the unconditioned bottleneck.
        spacing: world frame for the coordinate features injected per stage.
        bias_init: ``prior`` starts the head at the base rate of a single
            structure (see :func:`src.models.blocks.prior_logit`); ``zeros``
            starts every voxel at p = 0.5, which wastes the early schedule
            suppressing 262,144 background logits. Stage A measured roughly an
            order of magnitude in convergence on this choice.
    """

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: Sequence[int],
        decoder_channels: Sequence[int],
        context_dim: int,
        *,
        stage_resolutions: Sequence[int] | None = None,
        kernel_size: int = 3,
        align_corners: bool = True,
        activation: str = "leaky_relu",
        condition_at: Sequence[int] = (16, 32),
        occupancy_at: Sequence[int] = (16, 32, 64),
        out_channels: int = 1,
        coordinate_features: bool = True,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        bias_init: str = "prior",
        prior_foreground_fraction: float = 0.0016,
    ) -> None:
        super().__init__()
        skips = tuple(int(v) for v in skip_channels)
        widths = tuple(int(v) for v in decoder_channels)
        if len(skips) != 3 or len(widths) != 3:
            raise ValueError(
                f"expected three skips and three decoder widths, got {skips} and {widths}"
            )
        self.condition_at = tuple(int(v) for v in condition_at)
        self.occupancy_at = tuple(int(v) for v in occupancy_at)
        # Output resolution of each decoder stage, coarse to fine. Used to
        # decide which stages get FiLM; a stage that is never conditioned must
        # not own a FiLM block at all, or its parameters would sit in the
        # optimiser forever without a gradient. Occupancy concat uses the same
        # list so WHAT is never injected at an unknown scale.
        self.stage_resolutions = (
            tuple(int(v) for v in stage_resolutions) if stage_resolutions is not None else ()
        )
        self.coordinate_features = bool(coordinate_features)
        self.spacing = tuple(float(v) for v in spacing)
        extra = 3 if self.coordinate_features else 0

        self.up1 = UpBlock(bottleneck_channels + extra, skips[0], widths[0], kernel_size, align_corners, activation)
        self.up2 = UpBlock(widths[0] + extra, skips[1], widths[1], kernel_size, align_corners, activation)
        self.up3 = UpBlock(widths[1] + extra, skips[2], widths[2], kernel_size, align_corners, activation)
        self.refine = nn.ModuleList(
            nn.Sequential(
                conv3d(width, width, kernel_size),
                norm3d(width),
                make_activation(activation),
            )
            for width in widths
        )
        conditioned = {
            index
            for index, resolution in enumerate(self.stage_resolutions)
            if resolution in self.condition_at
        }
        if not self.stage_resolutions:
            conditioned = set(range(len(widths)))
        self.film = nn.ModuleDict(
            {str(index): FiLM3d(context_dim, widths[index]) for index in sorted(conditioned)}
        )
        occupancy_stages = {
            index
            for index, resolution in enumerate(self.stage_resolutions)
            if resolution in self.occupancy_at
        }
        self.occ_proj = nn.ModuleDict(
            {
                str(index): nn.Conv3d(widths[index] + 1, widths[index], kernel_size=1)
                for index in sorted(occupancy_stages)
            }
        )
        if bias_init not in ("prior", "zeros"):
            raise ValueError(f"bias_init must be 'prior' or 'zeros', got {bias_init!r}")
        self.head = nn.Conv3d(widths[2], out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(
            self.head.bias,
            prior_logit(prior_foreground_fraction) if bias_init == "prior" else 0.0,
        )

    def _with_coordinates(self, features: Tensor, full_shape: Sequence[int]) -> Tensor:
        if not self.coordinate_features:
            return features
        grid = normalized_world_grid(
            features.shape[2:], full_shape, self.spacing,
            device=features.device, dtype=features.dtype,
        )
        return torch.cat([features, grid.expand(features.shape[0], -1, -1, -1, -1)], dim=1)

    def forward(
        self,
        bottleneck: Tensor,
        skips: Sequence[Tensor],
        context: Tensor,
        *,
        occupancy: Tensor | None = None,
        full_shape: Sequence[int] | None = None,
    ) -> Tensor:
        """``-> [B, out_channels, D, H, W]`` target logits.

        Args:
            bottleneck: the intersection-conditioned ``[B, C, 8, 8, 8]`` grid.
            skips: encoder features at 1/4, 1/2 and full resolution.
            context: ``[B, context_dim]`` relation-fused clause context.
            occupancy: full-resolution binary scene ``[B, 1, D, H, W]``. Max-
                pooled onto each named decoder grid after the skip merge.
                ``None`` skips the concat (ablation).
        """
        if len(skips) != 3:
            raise ValueError(f"expected three skip features, got {len(skips)}")
        shape = tuple(full_shape) if full_shape is not None else tuple(skips[-1].shape[2:])
        if occupancy is not None:
            if occupancy.ndim != 5 or occupancy.shape[1] != 1:
                raise ValueError(
                    f"occupancy must be [B, 1, D, H, W], got {tuple(occupancy.shape)}"
                )
            if occupancy.shape[0] != bottleneck.shape[0]:
                raise ValueError(
                    f"batch mismatch: occupancy {occupancy.shape[0]} vs bottleneck "
                    f"{bottleneck.shape[0]}"
                )

        features = bottleneck
        for index, (up, skip) in enumerate(zip((self.up1, self.up2, self.up3), skips)):
            features = up(self._with_coordinates(features, shape), skip)
            key = str(index)
            if occupancy is not None and key in self.occ_proj:
                occ = downsample_masks(occupancy, features.shape[2:], mode="max")
                features = torch.cat([features, occ], dim=1)
                features = self.occ_proj[key](features)
            if key in self.film:
                film = self.film[key]
                resolution = int(features.shape[-1])
                if self.stage_resolutions and resolution != self.stage_resolutions[index]:
                    raise ValueError(
                        f"decoder stage {index} produced {resolution}^3 but the model was "
                        f"built for {self.stage_resolutions[index]}^3; rebuild it for this "
                        "input resolution"
                    )
                features = film(features, context)
            features = self.refine[index](features)
        return self.head(features)
