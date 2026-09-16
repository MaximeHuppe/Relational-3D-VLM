"""Stage A model I/O contract.

Covers the promises the architecture makes, not its accuracy:

* the output returns to the input resolution, with deep-supervision maps at
  1/4 and 1/2;
* prompts are independent - the logits of a shape do not depend on which other
  shapes were requested, which is what lets Phase 4 query three anchors by name;
* permuting the requested prompts permutes the output channels and nothing else;
* the bottleneck is 8^3 for a 64^3 input (512 attention tokens, the budget
  CLAUDE.md sets), and the positional encoding refuses a mismatched grid;
* gradients reach every parameter;
* the forward signature cannot accept the scene's labels, the target mask or
  any target identity.

Stage B's contract is tested in Phase 3.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.data.primitives import SHAPE_NAMES, SHAPE_VOCABULARY
from src.models.blocks import DecomposedPositionalEncoding3D, ResidualBlock, UpBlock
from src.models.prompt_encoder import ShapeNamePromptEncoder
from src.models.shape_segmenter import (
    ShapeSegmenter,
    ShapeSegmenterConfig,
    StageVLFusionBlock,
    build_shape_segmenter,
    prior_logit,
)

TINY = ShapeSegmenterConfig(
    stem_channels=4,
    encoder_channels=(4, 8, 16, 32),
    decoder_channels=(16, 8, 4),
    input_resolution=16,
    bottleneck_resolution=2,
    text_dim=16,
    embed_dim=32,
    heads=2,
    head_dim=16,
)


@pytest.fixture(scope="module")
def model() -> ShapeSegmenter:
    torch.manual_seed(0)
    net = ShapeSegmenter(TINY)
    net.eval()
    return net


def scene(batch: int = 2, size: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return (torch.rand(batch, 1, size, size, size) > 0.8).float()


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_output_returns_to_the_input_resolution(model):
    output = model(scene())
    assert output.logits.shape == (2, 10, 16, 16, 16)


def test_deep_supervision_maps_are_quarter_half_and_full_resolution(model):
    output = model(scene())
    assert [tuple(t.shape[2:]) for t in output.deep_supervision] == [
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
    ]
    assert torch.equal(output.deep_supervision[-1], output.logits)


def test_deep_supervision_can_be_switched_off_for_inference(model):
    output = model(scene(), deep_supervision=False)
    assert output.deep_supervision == []
    assert output.logits.shape[2:] == (16, 16, 16)


def test_probabilities_and_binary_mask(model):
    output = model(scene(1))
    probabilities = output.probabilities()
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    mask = output.binary_mask(threshold=0.5)
    assert mask.dtype == torch.uint8
    assert set(mask.unique().tolist()) <= {0, 1}


def test_a_64_cubed_input_gives_the_512_token_bottleneck_the_spec_requires():
    config = ShapeSegmenterConfig.from_config(profile="smoke")
    assert config.input_resolution == 64
    assert config.bottleneck_resolution == 8
    assert config.bottleneck_resolution**3 == 512
    assert config.decoder_resolutions == (16, 32, 64)


# ---------------------------------------------------------------------------
# Prompt contract
# ---------------------------------------------------------------------------
def test_default_prompts_are_the_ten_shapes_in_canonical_order(model):
    ids = model.default_prompt_ids(3)
    assert ids.shape == (3, 10)
    assert ids[0].tolist() == list(range(10))
    assert ShapeNamePromptEncoder.names_for_ids(ids[0].tolist()) == list(SHAPE_NAMES)


def test_a_prompt_subset_gets_the_same_logits_as_in_the_full_request(model):
    """Adding prompts must not change the logits of a shared subset."""
    volume = scene(1)
    names = ("sphere", "torus", "capsule")
    with torch.no_grad():
        full = model(volume, model.default_prompt_ids(1), deep_supervision=False).logits
        subset = model(
            volume, model.prompt_ids_for_names(names, 1), deep_supervision=False
        ).logits
    indices = [SHAPE_VOCABULARY.index_of(name) for name in names]
    assert torch.allclose(subset, full[:, indices], atol=1e-5)


def test_permuting_prompts_permutes_the_output_channels(model):
    volume = scene(1)
    order = [3, 0, 7, 1, 9, 2, 8, 4, 6, 5]
    with torch.no_grad():
        canonical = model(volume, model.default_prompt_ids(1), deep_supervision=False).logits
        permuted = model(
            volume, torch.tensor([order], dtype=torch.long), deep_supervision=False
        ).logits
    assert torch.allclose(permuted, canonical[:, order], atol=1e-5)


def test_repeating_a_prompt_repeats_its_logits(model):
    volume = scene(1)
    with torch.no_grad():
        logits = model(volume, torch.tensor([[2, 2, 5]]), deep_supervision=False).logits
    assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)


def test_distinct_names_get_distinct_queries(model):
    """Separation of the output maps is learned; separation of the queries is not."""
    with torch.no_grad():
        queries = model.prompt_encoder(torch.tensor([[2, 2, 5]]))
    assert torch.allclose(queries[:, 0], queries[:, 1])
    assert not torch.allclose(queries[:, 0], queries[:, 2])


def test_prompt_ids_are_validated(model):
    volume = scene(1)
    with pytest.raises(ValueError):
        model(volume, torch.tensor([[10]]))          # out of vocabulary
    with pytest.raises(ValueError):
        model(volume, torch.tensor([[0, 1], [2, 3]]))  # batch mismatch
    with pytest.raises(TypeError):
        model(volume, torch.tensor([[0.0, 1.0]]))    # not indices


def test_predict_anchor_masks_returns_the_named_masks_in_order(model):
    names = ("pyramid", "cube", "torus")
    masks = model.predict_anchor_masks(scene(2), names)
    assert masks.shape == (2, 3, 16, 16, 16)
    assert masks.dtype == torch.uint8


def test_name_and_id_helpers_round_trip():
    ids = ShapeNamePromptEncoder.ids_for_names(list(SHAPE_NAMES))
    assert ids == list(range(10))
    assert ShapeNamePromptEncoder.names_for_ids(ids) == list(SHAPE_NAMES)


# ---------------------------------------------------------------------------
# Stage A input contract
# ---------------------------------------------------------------------------
def test_the_forward_signature_accepts_only_the_scene_and_the_prompt():
    parameters = list(inspect.signature(ShapeSegmenter.forward).parameters)
    assert parameters == ["self", "scene_volume", "prompt_ids", "deep_supervision"]
    # Stage A is allowed the scene volume; it must never be handed labels.
    assert "instance_labels" not in parameters
    assert "target_mask" not in parameters


def test_the_model_rejects_a_non_volumetric_input(model):
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1, 16, 16))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    net = ShapeSegmenter(TINY)
    output = net(scene(1), deep_supervision=True)
    loss = sum(tensor.square().mean() for tensor in output.deep_supervision)
    loss.backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_residual_block_preserves_shape_and_is_an_identity_at_zero_weights():
    block = ResidualBlock(4)
    for parameter in block.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(1, 4, 6, 6, 6)
    out = block(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.relu(x))


def test_up_block_matches_the_skip_resolution():
    block = UpBlock(8, 4, 6)
    out = block(torch.randn(1, 8, 4, 4, 4), torch.randn(1, 4, 8, 8, 8))
    assert out.shape == (1, 6, 8, 8, 8)


def test_positional_encoding_is_decomposed_and_grid_locked():
    encoding = DecomposedPositionalEncoding3D((8, 8, 8), 16)
    codes = encoding()
    assert codes.shape == (1, 512, 16)
    # (D + H + W) * C parameters, not D*H*W*C.
    assert sum(p.numel() for p in encoding.parameters()) == 3 * 8 * 16
    with pytest.raises(ValueError):
        encoding((4, 4, 4))


def test_mask_head_bias_starts_at_the_foreground_prior():
    head = StageVLFusionBlock(8, 4, bias_init="prior", prior_foreground_fraction=0.0016)
    assert pytest.approx(prior_logit(0.0016), abs=1e-5) == float(head.bias_head.bias.detach())
    zeros = StageVLFusionBlock(8, 4, bias_init="zeros")
    assert float(zeros.bias_head.bias.detach()) == 0.0
    with pytest.raises(ValueError):
        StageVLFusionBlock(8, 4, bias_init="uniform")
    with pytest.raises(ValueError):
        prior_logit(0.0)


def test_mask_head_leaves_visual_features_untouched():
    head = StageVLFusionBlock(8, 4)
    visual = torch.randn(1, 4, 3, 3, 3)
    reference = visual.clone()
    head(torch.randn(1, 5, 8), visual)
    assert torch.equal(visual, reference)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_both_profiles_build_and_the_smoke_profile_is_smaller():
    default = build_shape_segmenter("default")
    smoke = build_shape_segmenter("smoke")
    default_params = sum(p.numel() for p in default.parameters())
    smoke_params = sum(p.numel() for p in smoke.parameters())
    assert smoke_params < default_params
    assert default.config.num_prompts == len(SHAPE_NAMES)


def test_configuration_errors_are_caught_at_construction():
    with pytest.raises(ValueError):
        ShapeSegmenterConfig(heads=3)  # 3 * 64 != 256
    with pytest.raises(ValueError):
        ShapeSegmenterConfig(input_resolution=64, bottleneck_resolution=4)
    with pytest.raises(ValueError):
        ShapeSegmenterConfig(stem_channels=16)  # disagrees with encoder_channels[0]
    with pytest.raises(ValueError):
        ShapeSegmenterConfig(decoder_channels=(64, 32))
    with pytest.raises(ValueError):
        ShapeSegmenterConfig(deep_supervision_weights=(0.5, 0.5))


def test_deep_supervision_weights_match_the_reference_figure():
    config = ShapeSegmenterConfig.from_config()
    assert config.deep_supervision_weights == (0.1, 0.3, 0.6)
