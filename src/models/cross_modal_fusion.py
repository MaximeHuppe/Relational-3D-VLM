"""Relation-conditioned grounding: one spatial evidence map per clause.

The target is not inside any anchor - it is the region that satisfies all three
target-relative constraints at once. So each clause is grounded *separately*
first, and only then intersected (:mod:`src.models.intersection_fusion`).

For clause ``i``:

1. :class:`ClauseFusion` fuses relation token ``i`` with structure token ``i``;
2. the fused clause token is broadcast over the encoder's 16^3 stage2 grid
   through cross-attention where the **visual locations are the queries** and
   the clause's three tokens (fused, relation, structure) are the keys/values;
3. a small convolutional head turns the attended grid into the relation-specific
   evidence map ``H_i``.

Grounding happens at 16^3: 4,096 queries against three keys, one cell per four
voxels. That is the grid the target has to be placed on - the 8^3 bottleneck
gives one cell per eight voxels, which cannot hold a peak for a structure of
radius ~5 voxels, and the decoder cannot move a peak afterwards (FiLM is a
per-channel affine, the same modulation everywhere). Anything finer is what
CLAUDE.md rules out: 32^3 is 32,768 queries and 64^3 is 262,144. The bottleneck
keeps its other two jobs - global context for the decoder and the structure
encoder's input.

Branch weights are shared across the three clauses by default
(``share_branch_weights``). The branches are independent in the sense that
matters - each ``H_i`` is computed from its own clause's tokens and nothing
else - but sharing the parameters means a clause cannot be grounded by a
slot-specific shortcut: the only thing that distinguishes branch 2 from branch 1
is its tokens, which carry the direction, the anchor shape, the pair and the
slot embedding. Set ``share_branch_weights=False`` for per-slot parameters.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from src.models.blocks import (
    DecomposedPositionalEncoding3D,
    conv3d,
    flatten_spatial,
    make_activation,
    norm3d,
)


class ClauseFusion(nn.Module):
    """Fuse one relation token with its structure token.

    ``cross_attention``: the relation token is the query over ``[relation,
    structure]``, followed by a residual MLP. ``gated_mlp``: a sigmoid gate
    computed from the concatenation, the cheaper alternative named in
    ``configs/model.yaml``.
    """

    def __init__(self, token_dim: int, *, heads: int = 4, mode: str = "cross_attention") -> None:
        super().__init__()
        if mode not in ("cross_attention", "gated_mlp"):
            raise ValueError(f"branch_fusion must be 'cross_attention' or 'gated_mlp', got {mode!r}")
        if mode == "cross_attention" and token_dim % heads != 0:
            raise ValueError(f"token_dim {token_dim} must be divisible by heads {heads}")
        self.mode = mode
        self.token_dim = int(token_dim)
        if mode == "cross_attention":
            self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(token_dim)
        else:
            self.gate = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.Sigmoid())
            self.value = nn.Linear(2 * token_dim, token_dim)
            self.attention_norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim),
            nn.GELU(),
            nn.Linear(2 * token_dim, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, relation_token: Tensor, structure_token: Tensor) -> Tensor:
        """``[B, C]`` + ``[B, C]`` -> ``[B, C]`` fused clause token."""
        if self.mode == "cross_attention":
            query = relation_token.unsqueeze(1)                      # [B, 1, C]
            memory = torch.stack([relation_token, structure_token], dim=1)  # [B, 2, C]
            attended, _ = self.attention(query, memory, memory, need_weights=False)
            fused = self.attention_norm(attended.squeeze(1) + relation_token)
        else:
            pair = torch.cat([relation_token, structure_token], dim=-1)
            fused = self.attention_norm(self.gate(pair) * self.value(pair) + relation_token)
        return self.norm(fused + self.mlp(fused))


class EvidenceHead(nn.Module):
    """Attended stage2 grid -> one relation-specific evidence map ``H_i``.

    ``grid_shape`` is the grounding grid the positional encoding is built for;
    the runtime grid must match it, so a model is tied to one input resolution.
    """

    def __init__(
        self,
        visual_channels: int,
        token_dim: int,
        evidence_channels: int,
        *,
        heads: int = 4,
        grid_shape: Sequence[int] = (16, 16, 16),
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(f"token_dim {token_dim} must be divisible by heads {heads}")
        self.token_dim = int(token_dim)
        self.query_projection = nn.Linear(visual_channels, token_dim)
        self.positional_encoding = DecomposedPositionalEncoding3D(grid_shape, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(token_dim)
        self.project = nn.Sequential(
            conv3d(visual_channels + token_dim, evidence_channels, kernel_size=1),
            norm3d(evidence_channels),
            make_activation(activation),
            conv3d(evidence_channels, evidence_channels, kernel_size=3),
            norm3d(evidence_channels),
            make_activation(activation),
        )

    def forward(self, visual: Tensor, tokens: Tensor) -> Tensor:
        """``visual [B, C, D', H', W']``, ``tokens [B, T, E]`` -> ``[B, Ce, D', H', W']``."""
        batch, _, depth, height, width = visual.shape
        queries = self.query_projection(flatten_spatial(visual))
        queries = queries + self.positional_encoding((depth, height, width))
        attended, _ = self.attention(queries, tokens, tokens, need_weights=False)
        attended = self.attention_norm(attended + queries)
        grid = attended.transpose(1, 2).reshape(batch, self.token_dim, depth, height, width)
        return self.project(torch.cat([visual, grid], dim=1))


class CrossModalFusion(nn.Module):
    """The three relation-conditioned branches.

    Returns the three evidence maps ``H_1, H_2, H_3`` and the three fused clause
    tokens (the latter condition the decoder; they are never pooled before the
    evidence maps exist).
    """

    def __init__(
        self,
        visual_channels: int,
        token_dim: int,
        evidence_channels: int,
        *,
        num_clauses: int = 3,
        heads: int = 4,
        grid_shape: Sequence[int] = (16, 16, 16),
        branch_fusion: str = "cross_attention",
        share_branch_weights: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.num_clauses = int(num_clauses)
        self.share_branch_weights = bool(share_branch_weights)
        branches = 1 if share_branch_weights else num_clauses

        def clause_fusion() -> ClauseFusion:
            return ClauseFusion(token_dim, heads=heads, mode=branch_fusion)

        def evidence_head() -> EvidenceHead:
            return EvidenceHead(
                visual_channels,
                token_dim,
                evidence_channels,
                heads=heads,
                grid_shape=grid_shape,
                activation=activation,
            )

        self.clause_fusion = nn.ModuleList(clause_fusion() for _ in range(branches))
        self.evidence_heads = nn.ModuleList(evidence_head() for _ in range(branches))

    def _branch(self, module_list: nn.ModuleList, index: int) -> nn.Module:
        return module_list[0 if self.share_branch_weights else index]

    def forward(
        self, visual: Tensor, relation_tokens: Tensor, structure_tokens: Tensor
    ) -> tuple[list[Tensor], Tensor]:
        """``-> ([H_1, H_2, H_3], fused_tokens [B, 3, E])``.

        Args:
            visual: ``[B, C, D', H', W']`` stage2 features, the grounding grid.
            relation_tokens: ``[B, 3, E]`` from the prompt encoder.
            structure_tokens: ``[B, 3, E]`` from the structure encoder.
        """
        if relation_tokens.shape != structure_tokens.shape:
            raise ValueError(
                f"relation tokens {tuple(relation_tokens.shape)} and structure tokens "
                f"{tuple(structure_tokens.shape)} must align clause by clause"
            )
        if relation_tokens.shape[1] != self.num_clauses:
            raise ValueError(
                f"expected {self.num_clauses} clauses, got {relation_tokens.shape[1]}"
            )

        evidence: list[Tensor] = []
        fused: list[Tensor] = []
        for index in range(self.num_clauses):
            relation = relation_tokens[:, index]
            structure = structure_tokens[:, index]
            clause = self._branch(self.clause_fusion, index)(relation, structure)
            tokens = torch.stack([clause, relation, structure], dim=1)
            evidence.append(self._branch(self.evidence_heads, index)(visual, tokens))
            fused.append(clause)
        return evidence, torch.stack(fused, dim=1)
