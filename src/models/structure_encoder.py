"""Per-anchor structure tokens.

One token per anchor channel, never pooled together:

* masked pooled encoder features, read from the 8^3 bottleneck under the
  anchor's own occupancy;
* normalised world centroid ``(x, y, z)`` in ``[-1, 1]``;
* normalised bounding-box extent ``(x, y, z)``;
* normalised voxel volume;
* a learned slot embedding for slots 1-3;
* a learned shape embedding for the anchor's name.

Every geometric feature is measured **from the mask channel itself**, not read
out of the manifest. That is deliberate and is what makes the oracle and
predicted anchor sources interchangeable: when Stage A supplies a slightly wrong
mask, its centroid and extent are the centroid and extent of *that* mask, so the
relational model is never handed ground-truth geometry the anchor channel does
not support. The manifest values are then an independent check on this code
(``tests/test_stage_b_contract.py``), not an input.

A predicted anchor channel can also be empty. An empty channel yields zeroed
geometry, the globally pooled features and a ``present`` flag of 0, so the
network can tell "small anchor at the origin" from "no anchor at all" instead of
consuming a NaN.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from src.data.primitives import SHAPE_NAMES
from src.models.blocks import downsample_masks

#: centroid (3) + extent (3) + volume (1) + presence flag (1).
NUM_GEOMETRY_FEATURES = 8

_EPSILON = 1e-6


def mask_geometry_features(
    masks: Tensor,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    volume_shape: Sequence[int] | None = None,
) -> Tensor:
    """Geometry of every mask channel, in normalised world units.

    Args:
        masks: ``[B, A, D, H, W]`` binary (or soft) masks.
        spacing: world units per voxel, ordered ``(x, y, z)``.
        volume_shape: the ``(D, H, W)`` the world frame is defined on; defaults
            to the mask's own shape.

    Returns:
        ``[B, A, 8]``: normalised centroid ``(x, y, z)`` in ``[-1, 1]``,
        normalised bounding-box extent ``(x, y, z)`` as a fraction of the volume,
        the cube root of the occupied volume fraction (a normalised linear size,
        which keeps the feature O(0.1) instead of O(0.001)), and a presence flag.

    The extent convention matches
    :func:`src.data.direction_rules.bbox_extent_world`: a single occupied voxel
    has an extent of one spacing unit per axis.
    """
    if masks.ndim != 5:
        raise ValueError(f"masks must be [B, A, D, H, W], got {tuple(masks.shape)}")
    occupancy = (masks > 0.5).to(torch.float32)
    device, dtype = masks.device, torch.float32
    shape = tuple(int(v) for v in (volume_shape if volume_shape is not None else masks.shape[2:]))
    spacing_xyz = tuple(float(v) for v in spacing)

    counts = occupancy.sum(dim=(2, 3, 4))                     # [B, A]
    present = (counts > 0).to(dtype)
    safe_counts = counts.clamp(min=1.0)

    centroids: list[Tensor] = []
    extents: list[Tensor] = []
    # Array axes are (z, y, x); world coordinates are ordered (x, y, z).
    for axis, size in enumerate(shape):
        step = spacing_xyz[2 - axis]
        indices = torch.arange(size, device=device, dtype=dtype)
        other = tuple(a for a in (2, 3, 4) if a != axis + 2)
        profile = occupancy.amax(dim=other)                    # [B, A, size]
        weights = occupancy.sum(dim=other)                     # [B, A, size]

        centre_index = (weights * indices).sum(dim=-1) / safe_counts
        half_extent = ((size - 1) / 2.0) * step
        centroid = (centre_index * step - half_extent) / max(half_extent, _EPSILON)
        centroids.append(centroid * present)

        big = torch.full_like(profile, float(size))
        lowest = torch.where(profile > 0, indices.expand_as(profile), big).amin(dim=-1)
        highest = torch.where(profile > 0, indices.expand_as(profile), torch.zeros_like(big)).amax(dim=-1)
        span = (highest - lowest + 1.0) * present
        extents.append(span * step / max(size * step, _EPSILON))

    # centroids/extents were built in (z, y, x) order; emit (x, y, z).
    centroid_xyz = torch.stack(centroids[::-1], dim=-1)
    extent_xyz = torch.stack(extents[::-1], dim=-1)
    total_voxels = float(shape[0] * shape[1] * shape[2])
    volume_fraction = counts / total_voxels
    linear_size = volume_fraction.clamp(min=0.0) ** (1.0 / 3.0)
    return torch.cat(
        [centroid_xyz, extent_xyz, linear_size.unsqueeze(-1), present.unsqueeze(-1)], dim=-1
    )


def masked_pool(features: Tensor, masks: Tensor) -> Tensor:
    """Pool ``features`` under each mask channel.

    Args:
        features: ``[B, C, D', H', W']`` encoder features (the bottleneck).
        masks: ``[B, A, D, H, W]`` masks at the input resolution.

    Returns:
        ``[B, A, C]``. Masks are average-pooled to the feature grid, so a
        structure smaller than one bottleneck cell still contributes a
        fractional weight instead of vanishing. A channel that is empty at the
        feature resolution falls back to the globally pooled features.
    """
    weights = downsample_masks(masks.to(features.dtype), features.shape[2:], mode="avg")
    weights = weights.flatten(2)                                      # [B, A, N]
    flat = features.flatten(2)                                        # [B, C, N]
    totals = weights.sum(dim=-1, keepdim=True)                        # [B, A, 1]
    pooled = torch.bmm(weights, flat.transpose(1, 2)) / totals.clamp(min=_EPSILON)
    globally = flat.mean(dim=-1).unsqueeze(1).expand_as(pooled)
    return torch.where(totals > _EPSILON, pooled, globally)


class StructureEncoder(nn.Module):
    """Turn the anchor channels and the bottleneck into three structure tokens.

    Args:
        visual_channels: width of the bottleneck feature map.
        token_dim: width of a structure token (must match the relation tokens).
        num_slots: clause slots, always 3.
        num_shapes: shape vocabulary size, always 10.
        spacing / volume_shape: the world frame the geometry is normalised in.
    """

    def __init__(
        self,
        visual_channels: int,
        token_dim: int = 256,
        *,
        num_slots: int = 3,
        num_shapes: int = len(SHAPE_NAMES),
        slot_embedding: bool = True,
        shape_embedding: bool = True,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        volume_shape: Sequence[int] = (64, 64, 64),
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.token_dim = int(token_dim)
        self.num_slots = int(num_slots)
        self.spacing = tuple(float(v) for v in spacing)
        self.volume_shape = tuple(int(v) for v in volume_shape)

        self.feature_projection = nn.Linear(visual_channels, token_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(NUM_GEOMETRY_FEATURES, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.slot_embedding = nn.Embedding(num_slots, token_dim) if slot_embedding else None
        self.shape_embedding = nn.Embedding(num_shapes, token_dim) if shape_embedding else None
        for table in (self.slot_embedding, self.shape_embedding):
            if table is not None:
                nn.init.trunc_normal_(table.weight, std=0.02)
        self.norm = nn.LayerNorm(token_dim)

    def geometry(self, masks: Tensor) -> Tensor:
        """The raw ``[B, A, 8]`` geometry block, exposed for tests and logging."""
        return mask_geometry_features(
            masks, spacing=self.spacing, volume_shape=self.volume_shape
        )

    def forward(
        self,
        anchor_masks: Tensor,
        bottleneck: Tensor,
        anchor_shape_ids: Tensor,
    ) -> Tensor:
        """``-> [B, num_slots, token_dim]``, one token per clause slot.

        Args:
            anchor_masks: ``[B, A, D, H, W]``. ``A`` is 3 for the ordered
                representation and 1 for the union-mask ablation, in which case
                the single channel is shared by all three slots.
            bottleneck: ``[B, C, D', H', W']`` encoder bottleneck.
            anchor_shape_ids: ``[B, num_slots]`` zero-based shape indices.
        """
        if anchor_masks.ndim != 5:
            raise ValueError(f"anchor_masks must be [B, A, D, H, W], got {tuple(anchor_masks.shape)}")
        if anchor_shape_ids.shape[1] != self.num_slots:
            raise ValueError(
                f"expected {self.num_slots} anchor shape ids, got {anchor_shape_ids.shape[1]}"
            )
        channels = anchor_masks.shape[1]
        if channels not in (1, self.num_slots):
            raise ValueError(
                f"anchor_masks must have {self.num_slots} channels (ordered) or 1 (union), "
                f"got {channels}"
            )
        if channels == 1:
            anchor_masks = anchor_masks.expand(-1, self.num_slots, -1, -1, -1)

        pooled = self.feature_projection(masked_pool(bottleneck, anchor_masks))
        tokens = pooled + self.geometry_projection(self.geometry(anchor_masks))
        if self.slot_embedding is not None:
            slots = torch.arange(self.num_slots, device=anchor_masks.device)
            tokens = tokens + self.slot_embedding(slots).unsqueeze(0)
        if self.shape_embedding is not None:
            tokens = tokens + self.shape_embedding(anchor_shape_ids)
        return self.norm(tokens)
