"""Explicit intersection of the three evidence maps.

``[H_1, H_2, H_3, H_1 * H_2 * H_3]`` through a pointwise fusion block. The
product term is the point of the module: it is high only where *all three*
relations are simultaneously satisfied, which is the definition of the target,
and a sum of the three maps cannot express that. The three maps are kept
alongside it so the block can also learn softer combinations.

The three ``(direction, anchor)`` pairs are never collapsed into one pooled text
vector before this point - by the time anything is pooled, each clause has
already produced its own spatial map.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.blocks import conv3d, make_activation, norm3d

#: The four inputs declared in ``configs/model.yaml``.
INTERSECTION_INPUTS: tuple[str, ...] = ("H1", "H2", "H3", "H1_mul_H2_mul_H3")


class IntersectionFusion(nn.Module):
    """Pointwise fusion of the evidence maps into one conditioned feature grid.

    Args:
        evidence_channels: width of each ``H_i``.
        hidden_channels: width inside the fusion block.
        out_channels: width of the grid handed to the decoder (the bottleneck
            width, so the result can be added to the visual features).
        num_clauses: 3.
        normalize_product: scale each map to ``[0, 1]`` with a sigmoid before
            multiplying. Without it the product of three unbounded activations
            is numerically brutal and its gradient vanishes or explodes; with it
            the product is a genuine soft AND.
    """

    def __init__(
        self,
        evidence_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_clauses: int = 3,
        activation: str = "leaky_relu",
        normalize_product: bool = True,
    ) -> None:
        super().__init__()
        self.num_clauses = int(num_clauses)
        self.normalize_product = bool(normalize_product)
        inputs = (self.num_clauses + 1) * evidence_channels
        self.fuse = nn.Sequential(
            conv3d(inputs, hidden_channels, kernel_size=1),
            norm3d(hidden_channels),
            make_activation(activation),
            conv3d(hidden_channels, hidden_channels, kernel_size=1),
            norm3d(hidden_channels),
            make_activation(activation),
            conv3d(hidden_channels, out_channels, kernel_size=1),
        )
        self.refine = nn.Sequential(
            conv3d(out_channels, out_channels, kernel_size=3),
            norm3d(out_channels),
            make_activation(activation),
        )

    def product(self, evidence: list[Tensor]) -> Tensor:
        """The soft-AND term ``H_1 * H_2 * H_3``."""
        maps = [torch.sigmoid(h) for h in evidence] if self.normalize_product else list(evidence)
        term = maps[0]
        for other in maps[1:]:
            term = term * other
        return term

    def forward(self, evidence: list[Tensor]) -> Tensor:
        """``[H_1, H_2, H_3] -> [B, out_channels, D', H', W']``."""
        if len(evidence) != self.num_clauses:
            raise ValueError(f"expected {self.num_clauses} evidence maps, got {len(evidence)}")
        shapes = {tuple(h.shape) for h in evidence}
        if len(shapes) != 1:
            raise ValueError(f"evidence maps must share a shape, got {sorted(shapes)}")
        stacked = torch.cat([*evidence, self.product(evidence)], dim=1)
        return self.refine(self.fuse(stacked))
