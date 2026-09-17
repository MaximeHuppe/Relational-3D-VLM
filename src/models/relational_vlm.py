"""Stage B end to end: the relational target segmenter.

Inputs are the ordered binary anchor-mask channels, the structured three-clause
prompt (as closed-vocabulary indices), geometry features **derived from those
masks**, and the binary scene occupancy. Occupancy is a decoder-side WHAT
stream: the encoder, fusion and intersection never see it. The model still must
not receive ``instance_labels``, the target mask, the target class, the target
centroid or the target instance ID - see
:data:`src.data.schema.STAGE_B_FORBIDDEN_FIELDS`.

Pipeline::

    anchor_masks [B, 3, 64, 64, 64]  ->  RelationalEncoder   (WHERE)
        64^3 -> 32^3 -> 16^3 -> 8^3, world (x, y, z) injected at every scale

    direction_ids, anchor_shape_ids   ->  RelationPromptEncoder  -> R [B, 3, E]
    anchor_masks + bottleneck         ->  StructureEncoder       -> S [B, 3, E]

    per clause i:  ClauseFusion(R_i, S_i) -> C_i
                   cross-attention (visual queries, {C_i, R_i, S_i} as K/V)
                   -> evidence map H_i at 8^3
    IntersectionFusion([H_1, H_2, H_3, H_1*H_2*H_3]) -> conditioned bottleneck

    scene_volume [B, 1, 64, 64, 64]  ->  occupancy, anchors masked out by default
    RelationalDecoder: 8^3 -> 16^3 -> 32^3 -> 64^3, skips + FiLM at 16^3/32^3
        occupancy concatenated at 16^3 / 32^3 / 64^3 after the skip merge (WHAT)
    head -> one target logit volume [B, 1, 64, 64, 64]

Anchor source is *not* this model's concern. It consumes whatever three ordered
channels it is handed: ground-truth masks from ``instance_labels`` (Phase 3,
oracle) or Stage A's predictions for the three named anchors (Phase 4). Because
every geometric feature is measured from the channels themselves
(:mod:`src.models.structure_encoder`), the two sources are interchangeable and
the oracle-vs-predicted delta measures Stage A's error, not a change of
interface. Stage A segments the intensity ``scene_volume``; this decoder
receives binary occupancy derived from labels (still named ``scene_volume`` on
the forward signature). See :mod:`src.models.anchor_provider`.

Baseline variants (``configs/model.yaml: baselines``) are selected by
:class:`RelationalVLMConfig`: ``full`` (the main model), ``anchor_masks_only``
(no prompt), ``prompt_only`` (coordinates, no masks) and
``prompt_plus_union_mask`` (the union ablation CLAUDE.md allows only as a
baseline). Occupancy stays on for all of them unless ``use_occupancy`` is
turned off.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.config import load_config
from src.data.primitives import SHAPE_NAMES
from src.data.prompt_generator import NUM_CLAUSES
from src.data.schema import STAGE_B_FORBIDDEN_FIELDS  # noqa: F401  (contract reference)
from src.models.cross_modal_fusion import CrossModalFusion
from src.models.decoder import RelationalDecoder
from src.models.intersection_fusion import IntersectionFusion
from src.models.prompt_encoder import RelationPromptEncoder
from src.models.relational_encoder import RelationalEncoder
from src.models.structure_encoder import StructureEncoder

#: The baseline names declared in ``configs/model.yaml``.
VARIANTS: tuple[str, ...] = (
    "full",
    "anchor_masks_only",
    "prompt_only",
    "prompt_plus_union_mask",
)

ANCHOR_REPRESENTATIONS: tuple[str, ...] = ("ordered_channels", "union_mask")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class RelationalVLMConfig:
    """Resolved Stage B architecture, from ``configs/model.yaml``."""

    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    decoder_channels: tuple[int, ...] = (128, 64, 32)
    resblocks: tuple[int, ...] = (1, 1, 1)
    kernel_size: int = 3
    activation: str = "leaky_relu"
    align_corners: bool = True
    input_resolution: int = 64
    bottleneck_resolution: int = 8
    out_channels: int = 1
    num_clauses: int = NUM_CLAUSES
    token_dim: int = 256
    embedding_dim: int = 256
    num_heads: int = 4
    branch_fusion: str = "cross_attention"
    share_branch_weights: bool = True
    evidence_channels: int | None = None
    intersection_hidden_channels: int = 256
    condition_at: tuple[int, ...] = (16, 32)
    coordinate_features: bool = True
    pair_embedding: bool = True
    slot_embedding: bool = True
    shape_embedding: bool = True
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    binary_threshold: float = 0.5
    bias_init: str = "prior"
    prior_foreground_fraction: float = 0.0016
    # -- baseline switches -------------------------------------------------
    variant: str = "full"
    anchor_representation: str = "ordered_channels"
    use_prompt: bool = True
    use_anchor_masks: bool = True
    use_occupancy: bool = True
    mask_occupancy_anchors: bool = True
    occupancy_at: tuple[int, ...] = (16, 32, 64)

    def __post_init__(self) -> None:
        if len(self.encoder_channels) != 4:
            raise ValueError(f"expected 4 encoder widths, got {self.encoder_channels}")
        if len(self.decoder_channels) != 3:
            raise ValueError(f"expected 3 decoder widths, got {self.decoder_channels}")
        if self.token_dim != self.embedding_dim:
            raise ValueError(
                f"structure token_dim ({self.token_dim}) must equal the prompt embedding_dim "
                f"({self.embedding_dim}); relation and structure tokens are fused in one space"
            )
        if self.token_dim % self.num_heads != 0:
            raise ValueError(
                f"token_dim {self.token_dim} must be divisible by num_heads {self.num_heads}"
            )
        if self.num_clauses != NUM_CLAUSES:
            raise ValueError(f"a prompt has exactly {NUM_CLAUSES} clauses, got {self.num_clauses}")
        if self.anchor_representation not in ANCHOR_REPRESENTATIONS:
            raise ValueError(
                f"anchor_representation must be one of {ANCHOR_REPRESENTATIONS}, "
                f"got {self.anchor_representation!r}"
            )
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        expected = self.input_resolution // 8
        if expected != self.bottleneck_resolution:
            raise ValueError(
                f"three stride-2 stages take {self.input_resolution}^3 to {expected}^3, but "
                f"bottleneck_resolution is {self.bottleneck_resolution}"
            )
        if self.bottleneck_resolution**3 > 4096:
            raise ValueError(
                f"a {self.bottleneck_resolution}^3 bottleneck is "
                f"{self.bottleneck_resolution ** 3} attention tokens; CLAUDE.md budgets 512"
            )

    @property
    def in_channels(self) -> int:
        """Anchor-mask channels fed to the encoder."""
        return 1 if self.anchor_representation == "union_mask" else self.num_clauses

    @property
    def bottleneck_channels(self) -> int:
        return int(self.encoder_channels[-1])

    @property
    def evidence_width(self) -> int:
        return int(self.evidence_channels or self.bottleneck_channels)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (self.bottleneck_resolution,) * 3

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        return (self.input_resolution,) * 3

    @property
    def decoder_resolutions(self) -> tuple[int, int, int]:
        """Output grid of each decoder stage, coarse to fine (16, 32, 64)."""
        base = self.bottleneck_resolution
        return (base * 2, base * 4, base * 8)

    @classmethod
    def for_variant(cls, variant: str, **changes: Any) -> "RelationalVLMConfig":
        """Apply one of the four declared baselines on top of the defaults."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        switches: dict[str, Any] = {"variant": variant}
        if variant == "anchor_masks_only":
            switches["use_prompt"] = False
        elif variant == "prompt_only":
            switches["use_anchor_masks"] = False
        elif variant == "prompt_plus_union_mask":
            switches["anchor_representation"] = "union_mask"
        return cls(**{**switches, **changes})

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        profile: str = "default",
        variant: str = "full",
        input_resolution: int | None = None,
        spacing: Sequence[float] | None = None,
    ) -> "RelationalVLMConfig":
        """Build from ``configs/model.yaml``; ``profile='smoke'`` applies the overrides."""
        model_config = dict(config or load_config("model"))
        stage = _deep_merge(
            dict(model_config["stage_b"]),
            dict(model_config.get("smoke", {}).get("stage_b", {})) if profile == "smoke" else {},
        )
        fusion = stage["fusion"]
        structure = stage["structure_encoder"]
        prompt = stage["prompt_encoder"]
        occupancy = dict(stage.get("occupancy") or {})
        resolution = int(input_resolution or stage["resolutions"][0])
        return cls.for_variant(
            variant,
            encoder_channels=tuple(int(v) for v in stage["encoder_channels"]),
            decoder_channels=tuple(int(v) for v in stage["decoder_channels"]),
            activation=str(stage["activation"]),
            input_resolution=resolution,
            bottleneck_resolution=resolution // 8,
            out_channels=int(stage["out_channels"]),
            token_dim=int(structure["token_dim"]),
            embedding_dim=int(prompt["embedding_dim"]),
            num_heads=int(fusion["num_heads"]),
            branch_fusion=str(fusion["branch_fusion"]),
            share_branch_weights=bool(fusion.get("share_branch_weights", True)),
            intersection_hidden_channels=int(fusion["intersection"]["hidden_channels"]),
            condition_at=tuple(int(v) for v in fusion["decoder_conditioning"]["at_resolutions"]),
            pair_embedding=bool(prompt["pair_embedding"]),
            slot_embedding=bool(structure["slot_embedding"]),
            shape_embedding=bool(structure["shape_embedding"]),
            binary_threshold=float(stage["output"]["binary_threshold"]),
            bias_init=str(stage["output"].get("bias_init", "prior")),
            prior_foreground_fraction=float(
                stage["output"].get("prior_foreground_fraction", 0.0016)
            ),
            spacing=tuple(float(v) for v in (spacing or (1.0, 1.0, 1.0))),
            use_occupancy=bool(occupancy.get("enabled", True)),
            mask_occupancy_anchors=bool(occupancy.get("mask_anchors", True)),
            occupancy_at=tuple(
                int(v) for v in occupancy.get("inject_at_resolutions", (16, 32, 64))
            ),
        )


@dataclass
class RelationalVLMOutput:
    """Forward result: the target logit volume, plus optional diagnostics."""

    logits: Tensor                       # [B, 1, D, H, W]
    evidence: list[Tensor] | None = None  # three 8^3 maps, when asked for
    clause_tokens: Tensor | None = None   # [B, 3, E] fused clause tokens

    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits)

    def binary_mask(self, threshold: float = 0.5) -> Tensor:
        return (self.probabilities() >= threshold).to(torch.uint8)


class RelationalVLM(nn.Module):
    """The Stage B network."""

    def __init__(self, config: RelationalVLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or RelationalVLMConfig()
        cfg = self.config

        self.encoder = RelationalEncoder(
            cfg.in_channels,
            cfg.encoder_channels,
            kernel_size=cfg.kernel_size,
            resblocks=cfg.resblocks,
            activation=cfg.activation,
            input_resolution=cfg.input_resolution,
            spacing=cfg.spacing,
            coordinate_features=cfg.coordinate_features,
        )
        self.prompt_encoder = RelationPromptEncoder(
            cfg.embedding_dim,
            num_slots=cfg.num_clauses,
            pair_embedding=cfg.pair_embedding,
            use_prompt=cfg.use_prompt,
        )
        self.structure_encoder = StructureEncoder(
            cfg.bottleneck_channels,
            cfg.token_dim,
            num_slots=cfg.num_clauses,
            slot_embedding=cfg.slot_embedding,
            shape_embedding=cfg.shape_embedding,
            spacing=cfg.spacing,
            volume_shape=cfg.volume_shape,
        )
        self.fusion = CrossModalFusion(
            cfg.bottleneck_channels,
            cfg.token_dim,
            cfg.evidence_width,
            num_clauses=cfg.num_clauses,
            heads=cfg.num_heads,
            grid_shape=cfg.grid_shape,
            branch_fusion=cfg.branch_fusion,
            share_branch_weights=cfg.share_branch_weights,
            activation=cfg.activation,
        )
        self.intersection = IntersectionFusion(
            cfg.evidence_width,
            cfg.intersection_hidden_channels,
            cfg.bottleneck_channels,
            num_clauses=cfg.num_clauses,
            activation=cfg.activation,
        )
        self.decoder = RelationalDecoder(
            cfg.bottleneck_channels,
            (cfg.encoder_channels[2], cfg.encoder_channels[1], cfg.encoder_channels[0]),
            cfg.decoder_channels,
            cfg.num_clauses * cfg.token_dim,
            stage_resolutions=cfg.decoder_resolutions,
            kernel_size=cfg.kernel_size,
            align_corners=cfg.align_corners,
            activation=cfg.activation,
            condition_at=cfg.condition_at,
            occupancy_at=cfg.occupancy_at,
            out_channels=cfg.out_channels,
            coordinate_features=cfg.coordinate_features,
            spacing=cfg.spacing,
            bias_init=cfg.bias_init,
            prior_foreground_fraction=cfg.prior_foreground_fraction,
        )

    # -- inputs ------------------------------------------------------------
    def prepare_masks(self, anchor_masks: Tensor) -> Tensor:
        """Apply the variant's anchor representation to the ordered channels.

        ``ordered_channels`` passes the three channels through unchanged;
        ``union_mask`` collapses them (the ablation baseline); ``prompt_only``
        zeroes them, which leaves the encoder with the coordinate grid alone and
        the structure tokens with a ``present`` flag of 0.
        """
        cfg = self.config
        if anchor_masks.ndim != 5:
            raise ValueError(f"anchor_masks must be [B, 3, D, H, W], got {tuple(anchor_masks.shape)}")
        if anchor_masks.shape[1] != cfg.num_clauses:
            raise ValueError(
                f"expected {cfg.num_clauses} ordered anchor channels, got {anchor_masks.shape[1]}"
            )
        masks = anchor_masks.to(torch.float32)
        if not cfg.use_anchor_masks:
            return torch.zeros_like(masks)
        if cfg.anchor_representation == "union_mask":
            return masks.amax(dim=1, keepdim=True)
        return masks

    def prepare_occupancy(self, scene_volume: Tensor, masks: Tensor) -> Tensor:
        """Binary occupancy at the mask grid, with the three anchors zeroed.

        ``masks`` is the encoder input after :meth:`prepare_masks`, so the
        prompt-only baseline (zeroed anchors) leaves the full scene and the
        union baseline subtracts the single union channel.
        """
        if scene_volume.ndim != 5 or scene_volume.shape[1] != 1:
            raise ValueError(
                f"scene_volume must be [B, 1, D, H, W], got {tuple(scene_volume.shape)}"
            )
        if scene_volume.shape[0] != masks.shape[0]:
            raise ValueError(
                f"batch mismatch: scene_volume {scene_volume.shape[0]} vs masks "
                f"{masks.shape[0]}"
            )
        if tuple(scene_volume.shape[2:]) != tuple(masks.shape[2:]):
            raise ValueError(
                f"scene_volume spatial size {tuple(scene_volume.shape[2:])} must match "
                f"anchor_masks {tuple(masks.shape[2:])}"
            )
        occupancy = scene_volume.to(dtype=torch.float32)
        if self.config.mask_occupancy_anchors:
            occupancy = occupancy * (1.0 - masks.amax(dim=1, keepdim=True))
        return occupancy

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        anchor_masks: Tensor,
        direction_ids: Tensor,
        anchor_shape_ids: Tensor,
        scene_volume: Tensor,
        *,
        return_evidence: bool = False,
    ) -> RelationalVLMOutput:
        """Segment the target described by three relations to three anchors.

        Args:
            anchor_masks: ``[B, 3, D, H, W]`` binary channels, ordered by clause
                slot. Channel ``i`` must be the anchor named in clause ``i``.
            direction_ids: ``[B, 3]`` zero-based direction indices.
            anchor_shape_ids: ``[B, 3]`` zero-based shape indices.
            scene_volume: ``[B, 1, D, H, W]`` binary occupancy of the whole
                scene (derived from labels, not the intensity image). Passed to
                the decoder only; the encoder never sees it.
            return_evidence: also return the three 8^3 evidence maps.
        """
        if direction_ids.shape != anchor_shape_ids.shape:
            raise ValueError(
                f"direction_ids {tuple(direction_ids.shape)} and anchor_shape_ids "
                f"{tuple(anchor_shape_ids.shape)} must align clause by clause"
            )
        if direction_ids.shape[0] != anchor_masks.shape[0]:
            raise ValueError(
                f"batch mismatch: anchor_masks {anchor_masks.shape[0]} vs prompt "
                f"{direction_ids.shape[0]}"
            )

        masks = self.prepare_masks(anchor_masks)
        volume_shape = tuple(anchor_masks.shape[2:])
        occupancy = (
            self.prepare_occupancy(scene_volume, masks) if self.config.use_occupancy else None
        )
        features = self.encoder(masks, volume_shape)

        relation_tokens = self.prompt_encoder(direction_ids, anchor_shape_ids)
        structure_tokens = self.structure_encoder(masks, features.bottleneck, anchor_shape_ids)
        evidence, clause_tokens = self.fusion(
            features.bottleneck, relation_tokens, structure_tokens
        )
        conditioned = features.bottleneck + self.intersection(evidence)
        context = clause_tokens.flatten(1)
        logits = self.decoder(
            conditioned,
            features.skips,
            context,
            occupancy=occupancy,
            full_shape=volume_shape,
        )
        return RelationalVLMOutput(
            logits=logits,
            evidence=evidence if return_evidence else None,
            clause_tokens=clause_tokens,
        )

    @torch.no_grad()
    def predict(
        self,
        anchor_masks: Tensor,
        direction_ids: Tensor,
        anchor_shape_ids: Tensor,
        scene_volume: Tensor,
        *,
        threshold: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``-> (probabilities, binary_mask)`` at the configured threshold."""
        self.eval()
        output = self(anchor_masks, direction_ids, anchor_shape_ids, scene_volume)
        cutoff = self.config.binary_threshold if threshold is None else threshold
        return output.probabilities(), output.binary_mask(cutoff)


def build_relational_vlm(
    profile: str = "default",
    *,
    variant: str = "full",
    input_resolution: int | None = None,
    spacing: Sequence[float] | None = None,
) -> RelationalVLM:
    """Construct Stage B from ``configs/model.yaml``."""
    config = RelationalVLMConfig.from_config(
        profile=profile, variant=variant, input_resolution=input_resolution, spacing=spacing
    )
    if len(SHAPE_NAMES) != 10:  # pragma: no cover - the vocabulary is fixed
        raise ValueError(f"the shape vocabulary must have ten names, got {len(SHAPE_NAMES)}")
    return RelationalVLM(config)


def load_relational_vlm(
    checkpoint: Path | str,
    *,
    device: torch.device | str = "cpu",
    profile: str | None = None,
    variant: str | None = None,
    input_resolution: int | None = None,
    spacing: Sequence[float] | None = None,
) -> RelationalVLM:
    """Rebuild Stage B from a checkpoint written by ``train_relational_model.py``.

    The width profile, variant, resolution and spacing come from the
    checkpoint's own ``model_config`` unless overridden, so evaluating a
    smoke-profile checkpoint without remembering to pass ``--smoke`` rebuilds the
    right skeleton instead of failing with an opaque shape mismatch.
    """
    from src.training.checkpointing import load_checkpoint

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"no Stage B checkpoint at {path}")
    payload = load_checkpoint(path, map_location=device)
    stored = payload.get("model_config") or {}
    settings = (payload.get("metadata") or {}).get("settings") or {}
    config = RelationalVLMConfig.from_config(
        profile=profile or str(settings.get("model_profile", "default")),
        variant=variant or str(stored.get("variant", "full")),
        input_resolution=int(input_resolution or stored.get("input_resolution") or 64),
        spacing=tuple(spacing or stored.get("spacing") or (1.0, 1.0, 1.0)),
    )
    model = RelationalVLM(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)
