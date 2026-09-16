"""Stage B model I/O contract.

Covers the promises the relational architecture makes, not its accuracy:

* the target logit volume comes back at the input resolution, from a 512-token
  bottleneck - full-resolution attention is never built;
* the forward signature takes the three ordered anchors, the prompt indices
  and ``scene_volume`` occupancy, and cannot accept instance labels, the
  target mask or any target identity; the dataset's input helper hands over
  those four tensors and nothing else;
* the natural-language and structured prompt paths produce the same three
  clause tokens;
* anchor geometry is measured from the mask channels, and matches the manifest
  values for oracle masks;
* correspondence matters: permuting the mask channels while the prompt stays
  fixed changes the prediction, and so does flipping a direction or renaming an
  anchor. A model that ignores the prompt would pass a Dice test and fail here;
* an empty anchor channel (which Stage A can produce) is handled, not NaN;
* gradients reach every parameter.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.data.direction_rules import DIRECTIONS, bbox_extent_world, centroid_world
from src.data.prompt_generator import clause_indices, clauses_from_indices
from src.data.schema import STAGE_B_FORBIDDEN_FIELDS
from src.models.blocks import normalized_world_grid
from src.models.cross_modal_fusion import ClauseFusion, CrossModalFusion
from src.models.decoder import FiLM3d, RelationalDecoder
from src.models.intersection_fusion import IntersectionFusion
from src.models.prompt_encoder import RelationPromptEncoder
from src.models.relational_encoder import RelationalEncoder
from src.models.relational_vlm import (
    VARIANTS,
    RelationalVLM,
    RelationalVLMConfig,
    build_relational_vlm,
)
from src.models.structure_encoder import StructureEncoder, mask_geometry_features

TINY = RelationalVLMConfig(
    encoder_channels=(4, 8, 16, 32),
    decoder_channels=(16, 8, 4),
    input_resolution=16,
    bottleneck_resolution=2,
    token_dim=16,
    embedding_dim=16,
    num_heads=2,
    intersection_hidden_channels=16,
    condition_at=(4, 8),
)

PROMPT = (
    "segment the shape that is anterior to the ellipsoid, inferior to the pyramid, "
    "and lateral to the cylinder."
)


@pytest.fixture(scope="module")
def model() -> RelationalVLM:
    torch.manual_seed(0)
    net = RelationalVLM(TINY)
    net.eval()
    return net


@pytest.fixture(scope="module")
def probe() -> RelationalVLM:
    """An untrained model whose head actually varies with its input.

    The mask head is zero-initialised on purpose (every logit starts at the
    foreground prior), so a pristine model returns a constant volume and no
    input change can move it. The correspondence probes below need a head that
    reads its features, so this fixture randomises that one 1x1 convolution and
    leaves everything else at its initial value.
    """
    torch.manual_seed(0)
    net = RelationalVLM(TINY)
    torch.nn.init.normal_(net.decoder.head.weight, std=0.5)
    net.eval()
    return net


def anchors(batch: int = 2, size: int = 16) -> torch.Tensor:
    """Three disjoint blocks, one per clause slot."""
    masks = torch.zeros(batch, 3, size, size, size)
    masks[:, 0, 2:5, 2:5, 2:5] = 1
    masks[:, 1, 8:11, 3:6, 9:12] = 1
    masks[:, 2, 4:7, 10:13, 2:5] = 1
    return masks


def prompt_ids(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    directions, shapes = clause_indices(
        [
            {"direction": "anterior", "anchor": "ellipsoid"},
            {"direction": "inferior", "anchor": "pyramid"},
            {"direction": "lateral", "anchor": "cylinder"},
        ]
    )
    return (
        torch.tensor([directions] * batch, dtype=torch.long),
        torch.tensor([shapes] * batch, dtype=torch.long),
    )


def scene(batch: int = 2, size: int = 16) -> torch.Tensor:
    """Matching occupancy grid. Occupancy-specific tests build a richer volume."""
    return torch.zeros(batch, 1, size, size, size)


def occupied(batch: int = 1, size: int = 16) -> torch.Tensor:
    """Anchors plus a disjoint blob the relations did not name."""
    volume = anchors(batch, size).amax(dim=1, keepdim=True).clone()
    volume[:, :, size - 4 : size - 2, size - 4 : size - 2, size - 4 : size - 2] = 1
    return volume


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_the_output_is_one_target_logit_volume_at_the_input_resolution(model):
    output = model(anchors(), *prompt_ids(), scene())
    assert output.logits.shape == (2, 1, 16, 16, 16)


def test_probabilities_and_binary_mask(model):
    output = model(anchors(1), *prompt_ids(1), scene(1))
    probabilities = output.probabilities()
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    mask = output.binary_mask(0.5)
    assert mask.dtype == torch.uint8
    assert set(mask.unique().tolist()) <= {0, 1}


def test_the_bottleneck_is_the_512_token_budget_claude_md_sets():
    config = RelationalVLMConfig.from_config(profile="smoke")
    assert config.input_resolution == 64
    assert config.bottleneck_resolution == 8
    assert config.bottleneck_resolution**3 == 512
    with pytest.raises(ValueError):
        RelationalVLMConfig(input_resolution=64, bottleneck_resolution=16)


def test_evidence_maps_live_at_the_bottleneck_not_at_full_resolution(model):
    output = model(anchors(1), *prompt_ids(1), scene(1), return_evidence=True)
    assert len(output.evidence) == 3
    for evidence in output.evidence:
        assert tuple(evidence.shape[2:]) == (2, 2, 2)  # the TINY bottleneck
    assert output.clause_tokens.shape == (1, 3, TINY.token_dim)


def test_the_head_starts_at_the_foreground_prior_not_at_one_half(model):
    """A zero-init head would predict p = 0.5 for a quarter of a million voxels."""
    probabilities = model(anchors(1), *prompt_ids(1), scene(1)).probabilities()
    assert probabilities.mean() < 0.01


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------
def test_the_forward_signature_takes_occupancy_but_not_the_target():
    parameters = list(inspect.signature(RelationalVLM.forward).parameters)
    assert parameters == [
        "self",
        "anchor_masks",
        "direction_ids",
        "anchor_shape_ids",
        "scene_volume",
        "return_evidence",
    ]
    for forbidden in STAGE_B_FORBIDDEN_FIELDS:
        assert forbidden not in parameters


def test_the_encoder_still_takes_three_channels_when_occupancy_is_on(model):
    assert model.config.use_occupancy
    assert model.encoder.in_channels == 3
    parameters = list(inspect.signature(model.encoder.forward).parameters)
    assert "scene_volume" not in parameters
    assert "occupancy" not in parameters


def test_the_dataset_helper_passes_only_the_permitted_inputs():
    from src.data.dataset import stage_b_model_inputs

    batch = {
        "anchor_masks": anchors(1),
        "direction_ids": prompt_ids(1)[0],
        "anchor_shape_ids": prompt_ids(1)[1],
        "target_mask": torch.ones(1, 1, 16, 16, 16),
        "scene_volume": torch.ones(1, 1, 16, 16, 16),
        "target_shape_name": ["cube"],
    }
    inputs = stage_b_model_inputs(batch)
    assert set(inputs) == {
        "anchor_masks",
        "direction_ids",
        "anchor_shape_ids",
        "scene_volume",
    }
    assert "target_mask" not in inputs
    assert "target_shape_name" not in inputs
    for forbidden in STAGE_B_FORBIDDEN_FIELDS:
        assert forbidden not in inputs
    swapped = stage_b_model_inputs(batch, torch.zeros_like(batch["anchor_masks"]))
    assert set(swapped) == set(inputs)
    assert float(swapped["anchor_masks"].sum()) == 0.0
    assert torch.equal(swapped["scene_volume"], batch["scene_volume"])


def test_prepare_occupancy_zeros_anchor_voxels_when_masking_is_on(model):
    masks = anchors(1)
    volume = occupied(1)
    prepared = model.prepare_occupancy(volume, masks)
    union = masks.amax(dim=1, keepdim=True)
    expected = volume * (1.0 - union)
    assert torch.equal(prepared, expected)
    assert float(prepared[union.bool()].abs().sum()) == 0.0
    leftover = (volume > 0) & (union == 0)
    assert torch.equal(prepared[leftover], volume[leftover])


def test_mutating_occupancy_of_a_non_anchor_object_changes_logits(probe):
    masks = anchors(1)
    volume = occupied(1)
    flipped = volume.clone()
    flipped[:, :, 12:14, 12:14, 12:14] = 0
    directions, shapes = prompt_ids(1)
    with torch.no_grad():
        reference = probe(masks, directions, shapes, volume).logits
        mutated = probe(masks, directions, shapes, flipped).logits
    assert reference.shape == (1, 1, 16, 16, 16)
    assert not torch.allclose(reference, mutated, atol=1e-4)


def test_mutating_occupancy_inside_an_anchor_does_not_change_logits_when_masked(probe):
    masks = anchors(1)
    volume = occupied(1)
    mutated = volume.clone()
    mutated[:, 0] = torch.where(masks[:, 0] > 0.5, 1.0 - mutated[:, 0], mutated[:, 0])
    directions, shapes = prompt_ids(1)
    with torch.no_grad():
        reference = probe(masks, directions, shapes, volume).logits
        inside = probe(masks, directions, shapes, mutated).logits
    assert torch.equal(reference, inside)


def test_scene_volume_must_match_the_anchor_mask_grid(model):
    with pytest.raises(ValueError, match="spatial size"):
        model(anchors(1), *prompt_ids(1), scene(1, size=8))


def test_malformed_inputs_are_rejected(model):
    directions, shapes = prompt_ids(2)
    with pytest.raises(ValueError):
        model(anchors(2)[:, :2], directions, shapes, scene(2))          # two channels
    with pytest.raises(ValueError):
        model(anchors(1), directions, shapes, scene(1))                 # batch mismatch
    with pytest.raises(ValueError):
        model(anchors(2), directions[:, :2], shapes[:, :2], scene(2))   # two clauses
    with pytest.raises(ValueError):
        model(anchors(2), directions, shapes + 100, scene(2))           # out of vocabulary


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------
def test_the_text_and_structured_paths_give_the_same_clause_tokens():
    structured = [
        {"direction": "anterior", "anchor": "ellipsoid"},
        {"direction": "inferior", "anchor": "pyramid"},
        {"direction": "lateral", "anchor": "cylinder"},
    ]
    assert RelationPromptEncoder.encode_prompt(PROMPT) == clause_indices(structured)
    directions, shapes = clause_indices(structured)
    assert clauses_from_indices(directions, shapes) == structured


def test_a_relation_token_is_the_sum_of_its_four_embeddings():
    torch.manual_seed(0)
    encoder = RelationPromptEncoder(8)
    directions = torch.tensor([[0, 1, 2]])
    shapes = torch.tensor([[3, 4, 5]])
    tokens = encoder(directions, shapes)
    expected = encoder.norm(
        encoder.slot_embedding(torch.arange(3)).unsqueeze(0)
        + encoder.direction_embedding(directions)
        + encoder.shape_encoder(shapes)
        + encoder.pair_embedding(directions * encoder.num_shapes + shapes)
    )
    assert torch.allclose(tokens, expected, atol=1e-6)
    assert tokens.shape == (1, 3, 8)


def test_the_same_clause_in_a_different_slot_gets_a_different_token():
    """Slot embeddings are what keep clause order meaningful."""
    torch.manual_seed(0)
    encoder = RelationPromptEncoder(8)
    tokens = encoder(torch.tensor([[2, 2, 2]]), torch.tensor([[7, 7, 7]]))
    assert not torch.allclose(tokens[0, 0], tokens[0, 1])


def test_the_no_prompt_baseline_drops_the_direction_and_shape_terms():
    torch.manual_seed(0)
    encoder = RelationPromptEncoder(8, use_prompt=False)
    a = encoder(torch.tensor([[0, 1, 2]]), torch.tensor([[3, 4, 5]]))
    b = encoder(torch.tensor([[5, 4, 3]]), torch.tensor([[9, 8, 7]]))
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Geometry from the masks
# ---------------------------------------------------------------------------
def test_geometry_features_match_the_reference_implementation():
    masks = anchors(1, 16)
    features = mask_geometry_features(masks)
    assert features.shape == (1, 3, 8)
    for slot in range(3):
        reference = masks[0, slot].numpy()
        centroid = centroid_world(reference)
        extent = bbox_extent_world(reference)
        assert np.allclose(features[0, slot, :3].numpy(), (centroid - 7.5) / 7.5, atol=1e-5)
        assert np.allclose(features[0, slot, 3:6].numpy(), extent / 16.0, atol=1e-5)
        assert features[0, slot, 7] == 1.0


def test_an_empty_anchor_channel_is_handled_rather_than_producing_nan():
    """Stage A can return an empty mask; the geometry must stay finite."""
    masks = anchors(1, 16)
    masks[:, 1] = 0
    features = mask_geometry_features(masks)
    assert torch.isfinite(features).all()
    assert float(features[0, 1, 7]) == 0.0      # present flag
    assert float(features[0, 1, :7].abs().sum()) == 0.0

    torch.manual_seed(0)
    net = RelationalVLM(TINY)
    logits = net(masks, *prompt_ids(1), scene(1)).logits
    assert torch.isfinite(logits).all()


def test_structure_tokens_stay_separate_per_anchor():
    torch.manual_seed(0)
    encoder = StructureEncoder(8, 16, volume_shape=(16, 16, 16))
    tokens = encoder(anchors(1, 16), torch.randn(1, 8, 2, 2, 2), torch.tensor([[3, 6, 4]]))
    assert tokens.shape == (1, 3, 16)
    assert not torch.allclose(tokens[0, 0], tokens[0, 1])


def test_world_coordinates_are_recomputed_per_scale_not_read_off_indices():
    full = normalized_world_grid((8, 8, 8), (8, 8, 8))
    coarse = normalized_world_grid((2, 2, 2), (8, 8, 8))
    assert full.shape == (1, 3, 8, 8, 8) and coarse.shape == (1, 3, 2, 2, 2)
    # Corners of the full grid are the corners of the world frame.
    assert float(full[0, 0, 0, 0, 0]) == -1.0 and float(full[0, 0, -1, -1, -1]) == 1.0
    # A coarse cell sits at the centre of the voxels it covers, so its
    # coordinates are inset - reading local indices would give +-1 again.
    assert abs(float(coarse[0, 0, 0, 0, 0]) + 0.5714) < 1e-3
    assert abs(float(coarse.mean())) < 1e-6
    # Channel order is (x, y, z) against array axes (z, y, x).
    assert float(coarse[0, 0, 0, 0, -1]) > 0 and float(coarse[0, 2, -1, 0, 0]) > 0


# ---------------------------------------------------------------------------
# Correspondence: the point of the architecture
# ---------------------------------------------------------------------------
def test_permuting_the_channels_with_the_prompt_fixed_changes_the_prediction(probe):
    masks = anchors(1)
    directions, shapes = prompt_ids(1)
    with torch.no_grad():
        reference = probe(masks, directions, shapes, scene(1)).logits
        permuted = probe(masks[:, [2, 0, 1]], directions, shapes, scene(1)).logits
    assert not torch.allclose(reference, permuted, atol=1e-4)


def test_permuting_the_clauses_with_the_channels_fixed_changes_the_prediction(probe):
    masks = anchors(1)
    directions, shapes = prompt_ids(1)
    with torch.no_grad():
        reference = probe(masks, directions, shapes, scene(1)).logits
        permuted = probe(masks, directions[:, [2, 0, 1]], shapes[:, [2, 0, 1]], scene(1)).logits
    assert not torch.allclose(reference, permuted, atol=1e-4)


def test_flipping_one_direction_or_renaming_one_anchor_changes_the_prediction(probe):
    masks = anchors(1)
    directions, shapes = prompt_ids(1)
    opposite = directions.clone()
    opposite[0, 0] = DIRECTIONS.index("posterior")
    renamed = shapes.clone()
    renamed[0, 2] = (renamed[0, 2] + 1) % 10
    with torch.no_grad():
        reference = probe(masks, directions, shapes, scene(1)).logits
        flipped = probe(masks, opposite, shapes, scene(1)).logits
        substituted = probe(masks, directions, renamed, scene(1)).logits
    assert not torch.allclose(reference, flipped, atol=1e-4)
    assert not torch.allclose(reference, substituted, atol=1e-4)


def test_removing_one_anchor_changes_the_prediction(probe):
    masks = anchors(1)
    dropped = masks.clone()
    dropped[:, 1] = 0
    directions, shapes = prompt_ids(1)
    with torch.no_grad():
        reference = probe(masks, directions, shapes, scene(1)).logits
        without = probe(dropped, directions, shapes, scene(1)).logits
    assert not torch.allclose(reference, without, atol=1e-4)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def test_the_union_baseline_collapses_the_channels_and_the_prompt_only_one_drops_them():
    tiny = {k: v for k, v in TINY.__dict__.items() if k not in ("variant", "anchor_representation", "use_prompt", "use_anchor_masks")}
    union = RelationalVLM(RelationalVLMConfig.for_variant("prompt_plus_union_mask", **tiny))
    masks = anchors(1)
    prepared = union.prepare_masks(masks)
    assert prepared.shape == (1, 1, 16, 16, 16)
    assert torch.equal(prepared[0, 0], masks[0].amax(0))

    blind = RelationalVLM(RelationalVLMConfig.for_variant("prompt_only", **tiny))
    assert float(blind.prepare_masks(masks).sum()) == 0.0


def test_every_declared_variant_builds_and_runs():
    for variant in VARIANTS:
        net = build_relational_vlm("smoke", variant=variant)
        with torch.no_grad():
            logits = net(
                torch.zeros(1, 3, 64, 64, 64), *prompt_ids(1), scene(1, 64)
            ).logits
        assert logits.shape == (1, 1, 64, 64, 64)
    with pytest.raises(ValueError):
        RelationalVLMConfig.for_variant("nonsense")


def test_the_smoke_profile_is_smaller_than_the_default():
    default = sum(p.numel() for p in build_relational_vlm("default").parameters())
    smoke = sum(p.numel() for p in build_relational_vlm("smoke").parameters())
    assert smoke < default


def test_configuration_errors_are_caught_at_construction():
    with pytest.raises(ValueError):
        RelationalVLMConfig(token_dim=128, embedding_dim=256)   # tokens must share a space
    with pytest.raises(ValueError):
        RelationalVLMConfig(num_heads=5)                        # 256 % 5
    with pytest.raises(ValueError):
        RelationalVLMConfig(encoder_channels=(32, 64, 128))
    with pytest.raises(ValueError):
        RelationalVLMConfig(anchor_representation="mean")


def test_from_config_reads_the_occupancy_block():
    config = RelationalVLMConfig.from_config(profile="smoke")
    assert config.use_occupancy
    assert config.mask_occupancy_anchors
    assert config.occupancy_at == (16, 32, 64)
    assert config.in_channels == 3


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    net = RelationalVLM(TINY)
    net(anchors(1), *prompt_ids(1), scene(1)).logits.square().mean().backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_the_encoder_halves_the_grid_three_times_and_keeps_the_world_frame():
    encoder = RelationalEncoder(3, (4, 8, 16, 32), input_resolution=16)
    features = encoder(anchors(1, 16))
    assert [tuple(f.shape) for f in (features.stem, features.stage1, features.stage2, features.bottleneck)] == [
        (1, 4, 16, 16, 16), (1, 8, 8, 8, 8), (1, 16, 4, 4, 4), (1, 32, 2, 2, 2)
    ]
    assert features.grid_shape == (2, 2, 2)
    assert [tuple(s.shape[2:]) for s in features.skips] == [(4, 4, 4), (8, 8, 8), (16, 16, 16)]


def test_the_intersection_module_multiplies_the_three_evidence_maps():
    fusion = IntersectionFusion(4, 8, 6)
    maps = [torch.randn(1, 4, 2, 2, 2) for _ in range(3)]
    product = fusion.product(maps)
    expected = torch.sigmoid(maps[0]) * torch.sigmoid(maps[1]) * torch.sigmoid(maps[2])
    assert torch.allclose(product, expected)
    assert fusion(maps).shape == (1, 6, 2, 2, 2)
    with pytest.raises(ValueError):
        fusion(maps[:2])


def test_a_branch_depends_only_on_its_own_clause_tokens():
    torch.manual_seed(0)
    fusion = CrossModalFusion(8, 16, 12, heads=2, grid_shape=(2, 2, 2))
    visual = torch.randn(1, 8, 2, 2, 2)
    relations = torch.randn(1, 3, 16)
    structures = torch.randn(1, 3, 16)
    evidence, _ = fusion(visual, relations, structures)
    changed = relations.clone()
    changed[:, 2] = torch.randn(1, 16)
    other, _ = fusion(visual, changed, structures)
    assert torch.allclose(evidence[0], other[0], atol=1e-6)     # clause 1 untouched
    assert not torch.allclose(evidence[2], other[2], atol=1e-4)  # clause 3 moved


def test_clause_fusion_supports_both_configured_modes():
    for mode in ("cross_attention", "gated_mlp"):
        block = ClauseFusion(8, heads=2, mode=mode)
        assert block(torch.randn(2, 8), torch.randn(2, 8)).shape == (2, 8)
    with pytest.raises(ValueError):
        ClauseFusion(8, mode="concat")


def test_film_starts_as_the_identity():
    film = FiLM3d(6, 4)
    features = torch.randn(2, 4, 3, 3, 3)
    assert torch.allclose(film(features, torch.randn(2, 6)), features)


def test_occupancy_is_concatenated_at_named_decoder_scales_not_the_bottleneck():
    decoder = RelationalDecoder(
        4,
        (8, 4, 2),
        (8, 4, 2),
        6,
        stage_resolutions=(16, 32, 64),
        occupancy_at=(16, 32, 64),
        condition_at=(16, 32),
        coordinate_features=False,
    )
    assert set(decoder.occ_proj) == {"0", "1", "2"}
    assert 8 not in decoder.occupancy_at
    bottleneck = torch.randn(1, 4, 8, 8, 8)
    skips = (
        torch.randn(1, 8, 16, 16, 16),
        torch.randn(1, 4, 32, 32, 32),
        torch.randn(1, 2, 64, 64, 64),
    )
    context = torch.randn(1, 6)
    occupancy = torch.ones(1, 1, 64, 64, 64)
    with torch.no_grad():
        with_occ = decoder(
            bottleneck, skips, context, occupancy=occupancy, full_shape=(64, 64, 64)
        )
        without = decoder(
            bottleneck, skips, context, occupancy=None, full_shape=(64, 64, 64)
        )
    assert with_occ.shape == (1, 1, 64, 64, 64)
    assert without.shape == with_occ.shape


def test_tiny_occupancy_is_injected_at_full_res_not_the_bottleneck(model):
    assert model.config.decoder_resolutions == (4, 8, 16)
    assert set(model.decoder.occ_proj) == {"2"}
    assert model.encoder.in_channels == 3
