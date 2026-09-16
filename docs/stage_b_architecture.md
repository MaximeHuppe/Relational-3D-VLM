# Stage B architecture

Phases 2-4 of the training procedure: the **relational target segmenter**. It
takes three ordered binary anchor-mask channels and a three-clause prompt, and
returns one target logit volume. The target shape is never an input — the model
has to find the region that satisfies all three target-relative constraints at
once.

Source: `src/models/relational_vlm.py` and the six modules it composes
(`relational_encoder`, `structure_encoder`, `prompt_encoder`,
`cross_modal_fusion`, `intersection_fusion`, `decoder`). Configuration:
`configs/model.yaml` (`stage_b`), `configs/train.yaml` (`stage_b_*`).
Figure: `docs/flowchart/phase2_model.drawio` (Architecture + Training-Data-Eval).

## What the model may see

```python
model(anchor_masks, direction_ids, anchor_shape_ids)   # and nothing else
```

The forward signature is the contract. `scene_volume`, `instance_labels`,
`target_mask`, `target_shape_name`, `target_instance_id` and
`target_centroid_world` — `src.data.schema.STAGE_B_FORBIDDEN_FIELDS` — have
nowhere to go. The training loop's only model call goes through
`src.data.dataset.stage_b_model_inputs`, which returns exactly those three
tensors; `tests/test_stage_b_contract.py` pins both ends down.

## Pipeline

```text
anchor_masks [B, 3, 64, 64, 64]        direction_ids, anchor_shape_ids [B, 3]
  RelationalEncoder                      RelationPromptEncoder
    + world (x,y,z) at every scale         direction + shape + pair + slot
    64^3 ->  32^3 ->  16^3 ->  8^3         -> R [B, 3, E]
     C1      C2       C3       C4
                    |
                    +-- StructureEncoder (per anchor channel)
                          masked pooled bottleneck features
                          + normalised centroid / extent / volume / presence
                          + slot embedding + shape embedding
                          -> S [B, 3, E]

per clause i (weights shared across i):
  ClauseFusion(R_i, S_i) -> C_i
  EvidenceHead: queries = 512 bottleneck locations (+ PE3D)
                keys/values = {C_i, R_i, S_i}
                -> H_i [B, Ce, 8, 8, 8]

IntersectionFusion([H_1, H_2, H_3, sigmoid(H_1)*sigmoid(H_2)*sigmoid(H_3)])
  1x1 conv -> hidden -> 1x1 conv -> C4,  added to the bottleneck

RelationalDecoder  8^3 -> 16^3 -> 32^3 -> 64^3
  upsample, concat encoder skip, FiLM(context) at 16^3 and 32^3, 3D conv refine
  head (1x1) -> target logits [B, 1, 64, 64, 64]
```

`context` is the three fused clause tokens concatenated **in clause order**
(`[B, 3E]`). Pooling happens only here, after every clause has produced its own
spatial map — never before grounding.

## Design decisions

### Geometry is measured from the masks, not read from the manifest

`StructureEncoder` computes each anchor's centroid, bounding-box extent and
occupied volume from its own mask channel (`mask_geometry_features`). The
manifest carries the same numbers, and a test asserts they agree for oracle
masks — but the model never reads them.

This is what makes the two anchor sources interchangeable. When Stage A supplies
a slightly wrong mask, its centroid is the centroid of *that* mask, so the
oracle-versus-predicted delta measures Stage A's error rather than a change of
interface. It also means an empty predicted channel is representable: geometry
zeroes out, the `present` flag goes to 0 and the pooled features fall back to the
global mean, instead of producing a NaN.

### World coordinates, recomputed per scale

`normalized_world_grid` annotates every feature map with continuous normalised
world `(x, y, z)`, derived from the world frame at that resolution: a cell of a
grid `f` times coarser covers input indices `f·i … f·i + f − 1`, so it sits at
`(f·i + (f−1)/2)·spacing`. Local tensor indices would give the same value to
different anatomical positions at different scales.

Without them the task is unsolvable: anchor channels alone are
translation-ambiguous, and `medial`/`lateral` is defined against the volume
centre plane, which only exists in world coordinates.

### An explicit product term

The intersection module receives `[H_1, H_2, H_3, H_1·H_2·H_3]`. The product is
the only term that is high *only* where all three relations hold, which is the
definition of the target; a sum cannot express it. Each map is passed through a
sigmoid first, so the product is a soft AND with a usable gradient rather than a
product of three unbounded activations.

### Shared branch weights

The three grounding branches share their parameters by default
(`fusion.share_branch_weights`). They stay independent in the sense that matters
— `H_i` is computed from clause `i`'s tokens and nothing else — but a shared
branch cannot ground clause 2 through a slot-specific shortcut: the only thing
distinguishing the branches is their tokens, which carry the direction, the
anchor shape, the `(direction, shape)` pair and the slot. Set the flag to
`false` for per-slot parameters.

### FiLM, not attention, in the decoder

Conditioning at 16³ and 32³ is a per-channel affine modulation predicted from the
clause context, zero-initialised so it starts as the identity. Attention at 32³
would mean 32,768 query tokens per sample — sixty-four times the bottleneck
budget CLAUDE.md sets. Only the stages named in
`fusion.decoder_conditioning.at_resolutions` own a FiLM block; a stage that is
never conditioned does not carry dead parameters.

### The head starts at the foreground prior

The 1×1 output convolution has zero weights and a bias of
`log(p/(1−p))` for `p = 0.0016`, the measured share of a 64³ volume one
structure occupies. A zero bias would start every one of 262,144 voxels at
`p = 0.5`, and training would begin by suppressing background instead of
locating the target — the same effect measured for Stage A, where it cost about
an order of magnitude in convergence.

One consequence worth knowing when reading tests: an untrained model returns a
*constant* logit volume, so correspondence probes have to randomise that one
convolution before they can observe any input sensitivity.

## Anchor sources

`src/models/anchor_provider.py` is the only place the choice lives.

| Source | What the three channels are | Used by |
| --- | --- | --- |
| `oracle` | ground-truth masks: `data/processed/scenes/<scene>.npz` is loaded and only the three anchor structures the prompt names are kept | Phases 2-3 |
| `predicted` | Stage A segments `scene_volume` and returns its masks for those same three names, in clause order | Phase 4 |

`anchor_shape_ids` *is* Stage A's `prompt_ids`, so the predicted channels come
back already aligned with the clauses. The predicted provider also scores its
own output against the ground-truth channels (`anchor_dice`, `anchor_iou`,
`empty_anchor_fraction`) so a Stage B drop can be attributed rather than guessed
at. The scene volume goes into Stage A and stops there.

## Baseline variants

`configs/model.yaml: baselines`, selected with `--variant`:

| Variant | Anchor channels | Prompt |
| --- | --- | --- |
| `full` | three ordered | full |
| `anchor_masks_only` | three ordered | slot embeddings only |
| `prompt_only` | zeroed (coordinates survive) | full |
| `prompt_plus_union_mask` | one union channel | full |

The union mask is an ablation baseline only, never the main representation.

## Objective

```text
L = lambda_dice * DiceLoss(logits, target_mask)
  + lambda_bce  * BCEWithLogitsLoss(logits, target_mask)
```

Weights in `configs/train.yaml`. No deep supervision (CLAUDE.md specifies a
single-scale objective for Stage B) and no target-shape classification loss in
this milestone.

## Sizes

| Profile | Encoder / decoder widths | Token width | Parameters |
| --- | --- | --- | --- |
| `default` | `[32, 64, 128, 256]` / `[128, 64, 32]` | 256 | 13.7 M |
| `smoke` | `[8, 16, 32, 64]` / `[32, 16, 8]` | 64 | 0.88 M |

## Portability notes, measured

* **`adaptive_avg_pool3d` has no MPS kernel.** Masked pooling therefore uses
  fixed-kernel `avg_pool3d`/`max_pool3d` (every ratio here is a power of two)
  with an interpolation fallback. The adaptive variants raise
  `NotImplementedError` on Apple Silicon, which is one of the two target devices.
* **bf16 autocast is a 33× regression on CPU** for this model: 0.93 s in float32
  against 30.75 s under autocast for one forward+backward at batch 2, because
  there is no fused bf16 `conv3d` path on CPU. The trainers disable autocast on
  CPU whatever the hardware profile says, and report the precision they actually
  ran in.
