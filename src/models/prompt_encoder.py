"""Closed-vocabulary prompt encoder.

CLAUDE.md fixes the text side of this project to a closed vocabulary: six
directions, ten shape names, three clause slots and the ``(direction, shape)``
pairs. Nothing else may appear in a prompt, so learned embedding tables are the
exact and deterministic representation - there is no open-vocabulary text to
generalise over, and the natural-language path maps onto the same tokens through
:mod:`src.data.prompt_generator`.

This is also where this project departs from
``docs/flowchart/phase1_encoder_decoder.drawio``, which queries a frozen
PubMedBERT with anatomical names. Ten fixed names need no language model.

Stage A prompts with shape names alone: one query per requested name
(:class:`ShapeNamePromptEncoder`). Stage B adds direction, slot and pair
embeddings (:class:`RelationPromptEncoder`) to build

    relation_token_i = direction_embedding_i
                      + shape_embedding_i
                      + pair_embedding(direction_i, shape_i)
                      + slot_embedding_i

and reuses :class:`ShapeNamePromptEncoder` for the shape term, so a shape name
means the same thing to both stages. Text and structured prompts meet at
:func:`src.data.prompt_generator.clause_indices`, the project's single
text-to-index mapping.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from src.data.direction_rules import DIRECTIONS
from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY
from src.data.prompt_generator import (
    NUM_CLAUSES,
    clause_indices,
    clauses_from_indices,
    parse_prompt,
)


class ShapeNamePromptEncoder(nn.Module):
    """Embed shape-name prompts, then project them to the query width.

    Args:
        num_shapes: vocabulary size (ten).
        text_dim: width of the learned name embedding; the stand-in for the
            frozen sentence-embedding width of the reference architecture.
        embed_dim: query width consumed by the prompt decoder.
    """

    def __init__(self, num_shapes: int = len(SHAPE_NAMES), text_dim: int = 256, embed_dim: int = 256) -> None:
        super().__init__()
        if num_shapes != len(SHAPE_NAMES):
            raise ValueError(
                f"the vocabulary is fixed at {len(SHAPE_NAMES)} shapes, got {num_shapes}"
            )
        self.num_shapes = num_shapes
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.name_embedding = nn.Embedding(num_shapes, text_dim)
        self.projection = nn.Linear(text_dim, embed_dim)
        nn.init.trunc_normal_(self.name_embedding.weight, std=0.02)

    def forward(self, prompt_ids: Tensor) -> Tensor:
        """``[B, N_T]`` zero-based shape indices -> ``[B, N_T, embed_dim]`` queries."""
        if prompt_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"prompt_ids must be integer indices, got {prompt_ids.dtype}")
        if prompt_ids.ndim != 2:
            raise ValueError(f"prompt_ids must be [B, N_T], got shape {tuple(prompt_ids.shape)}")
        if int(prompt_ids.min()) < 0 or int(prompt_ids.max()) >= self.num_shapes:
            raise ValueError(
                f"prompt_ids must index 0..{self.num_shapes - 1}, got range "
                f"[{int(prompt_ids.min())}, {int(prompt_ids.max())}]"
            )
        return self.projection(self.name_embedding(prompt_ids))

    @staticmethod
    def ids_for_names(names: list[str] | tuple[str, ...]) -> list[int]:
        """Map vocabulary names to the zero-based indices this module expects."""
        return [SHAPE_VOCABULARY.index_of(name) for name in SHAPE_VOCABULARY.require_names(names)]

    @staticmethod
    def names_for_ids(ids: list[int] | tuple[int, ...]) -> list[str]:
        """Inverse of :meth:`ids_for_names`."""
        return [SHAPE_VOCABULARY.id_to_name(int(index) + 1) for index in ids]


class RelationPromptEncoder(nn.Module):
    """Stage B: the three ordered clauses of a prompt, as three tokens.

    ``relation_token_i = direction_embedding_i + shape_embedding_i
                        + pair_embedding(direction_i, shape_i) + slot_embedding_i``

    Four tables, no pooling: the clauses stay separate all the way into the
    grounding branches, which is what CLAUDE.md requires ("Preserve all three
    clauses; do not average the prompt into a single vector before grounding").

    The pair table has ``6 x 10 = 60`` rows and is what lets the model represent
    a relation that is not the sum of its parts (``lateral to the torus`` need
    not behave like ``lateral`` plus ``torus``).

    The shape term reuses :class:`ShapeNamePromptEncoder`, so a shape name means
    the same thing to Stage A and to Stage B.

    Args:
        embedding_dim: width of a clause token.
        num_directions/num_shapes/num_slots: table sizes; all three are fixed by
            the closed vocabulary and are validated against it.
        pair_embedding: keep the ``(direction, shape)`` table. Disabling it is an
            ablation, not a supported configuration.
        use_prompt: when ``False`` the direction, shape and pair terms are
            dropped and only the slot embedding survives - the "anchor masks
            only, no prompt" baseline. The module still returns three distinct
            tokens so the rest of the network is unchanged.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        *,
        num_directions: int = len(DIRECTIONS),
        num_shapes: int = len(SHAPE_NAMES),
        num_slots: int = NUM_CLAUSES,
        pair_embedding: bool = True,
        use_prompt: bool = True,
    ) -> None:
        super().__init__()
        if num_directions != len(DIRECTIONS):
            raise ValueError(
                f"the direction vocabulary is fixed at {len(DIRECTIONS)} tokens, got {num_directions}"
            )
        if num_shapes != len(SHAPE_NAMES):
            raise ValueError(f"the shape vocabulary is fixed at {len(SHAPE_NAMES)}, got {num_shapes}")
        if num_slots != NUM_CLAUSES:
            raise ValueError(f"a prompt has exactly {NUM_CLAUSES} clauses, got {num_slots}")

        self.embedding_dim = int(embedding_dim)
        self.num_directions = num_directions
        self.num_shapes = num_shapes
        self.num_slots = num_slots
        self.use_prompt = bool(use_prompt)

        self.direction_embedding = nn.Embedding(num_directions, embedding_dim)
        self.shape_encoder = ShapeNamePromptEncoder(
            num_shapes=num_shapes, text_dim=embedding_dim, embed_dim=embedding_dim
        )
        self.pair_embedding = (
            nn.Embedding(num_directions * num_shapes, embedding_dim) if pair_embedding else None
        )
        self.slot_embedding = nn.Embedding(num_slots, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        for table in (self.direction_embedding, self.slot_embedding):
            nn.init.trunc_normal_(table.weight, std=0.02)
        if self.pair_embedding is not None:
            nn.init.trunc_normal_(self.pair_embedding.weight, std=0.02)

    def forward(self, direction_ids: Tensor, shape_ids: Tensor) -> Tensor:
        """``[B, 3]`` direction and shape indices -> ``[B, 3, embedding_dim]``.

        Clause ``i`` of the output is clause ``i`` of the prompt; the caller must
        keep that order identical to the anchor-mask channel order.
        """
        self._validate(direction_ids, "direction_ids", self.num_directions)
        self._validate(shape_ids, "shape_ids", self.num_shapes)
        if direction_ids.shape != shape_ids.shape:
            raise ValueError(
                f"direction_ids {tuple(direction_ids.shape)} and shape_ids "
                f"{tuple(shape_ids.shape)} must have the same shape"
            )
        if direction_ids.shape[1] != self.num_slots:
            raise ValueError(
                f"expected {self.num_slots} clauses, got {direction_ids.shape[1]}"
            )

        batch, slots = direction_ids.shape
        slot_ids = torch.arange(slots, device=direction_ids.device).unsqueeze(0).expand(batch, -1)
        tokens = self.slot_embedding(slot_ids)
        if self.use_prompt:
            tokens = tokens + self.direction_embedding(direction_ids)
            tokens = tokens + self.shape_encoder(shape_ids)
            if self.pair_embedding is not None:
                tokens = tokens + self.pair_embedding(
                    direction_ids * self.num_shapes + shape_ids
                )
        return self.norm(tokens)

    @staticmethod
    def _validate(ids: Tensor, name: str, size: int) -> None:
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be integer indices, got {ids.dtype}")
        if ids.ndim != 2:
            raise ValueError(f"{name} must be [B, 3], got shape {tuple(ids.shape)}")
        if int(ids.min()) < 0 or int(ids.max()) >= size:
            raise ValueError(
                f"{name} must index 0..{size - 1}, got range [{int(ids.min())}, {int(ids.max())}]"
            )

    # -- vocabulary helpers -------------------------------------------------
    @staticmethod
    def direction_index(direction: str) -> int:
        """Zero-based index of a direction token."""
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}; vocabulary is {DIRECTIONS}")
        return DIRECTIONS.index(direction)

    @staticmethod
    def direction_name(index: int) -> str:
        return DIRECTIONS[int(index)]

    @classmethod
    def encode_clauses(
        cls, clauses: Sequence[Mapping[str, str]]
    ) -> tuple[list[int], list[int]]:
        """Structured prompt -> ``(direction_ids, shape_ids)``.

        Thin wrapper over :func:`src.data.prompt_generator.clause_indices`, which
        is the project's single text-to-index mapping; an invalid direction or
        anchor name is rejected there rather than silently indexed.
        """
        return clause_indices(clauses)

    @classmethod
    def encode_prompt(cls, prompt: str) -> tuple[list[int], list[int]]:
        """Natural-language prompt -> ``(direction_ids, shape_ids)``.

        The text path is a parse into the structured representation followed by
        :meth:`encode_clauses`, so both interfaces provably produce the same
        three clause tokens.
        """
        return cls.encode_clauses(parse_prompt(prompt))

    @classmethod
    def decode_clauses(
        cls, direction_ids: Sequence[int], shape_ids: Sequence[int]
    ) -> list[dict[str, str]]:
        """Inverse of :meth:`encode_clauses`, for logging and counterfactuals."""
        return clauses_from_indices(direction_ids, shape_ids)
