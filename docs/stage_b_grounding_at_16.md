# Stage B: ground relations at 16³

Implementation spec for a **single architecture change** on the `origin/main`
baseline. Do not mix in loss-weight experiments (focal BCE, mask centroid,
centroid head) from other branches.

The goal is to stop localising the target on an 8-voxel grid. Measured Phase 3
centroid error is ~7 vox, which is one bottleneck cell. Objects in this corpus
have radius ~4.8 vox; they cannot appear as a peak on an 8-vox cell. Decoder
FiLM cannot move that peak: it is a per-channel scale-and-shift, the same
`(γ, β)` at every voxel.

## Why this change, and why not the alternatives

| Change | Do it? | Why |
| --- | --- | --- |
| Wider encoder / new backbone | No | Phase 2 overfit already reaches Dice 0.95, Hausdorff 1.25. Capacity is not the limit. |
| Different attention mechanism (Deformable, gated MLP, more heads) | No | Keys are already three clause tokens. The query **grid** is the limit. |
| Start the decoder at 16³ while still computing `H_i` at 8³ | No | That upsamples the same coarse field. |
| Full-resolution attention at 32³ or 64³ | No | 32,768 / 262,144 tokens. Forbidden by `docs/CLAUDE.md`. |
| **Evidence maps at 16³** | **Yes** | 4,096 queries, still 3 keys, **4 vox cells**. Cheap, and fine enough to place a 5-vox-radius object. |

Keep the encoder `64 → 32 → 16 → 8`. Do not delete the 8³ bottleneck. Structure
tokens still come from it. What moves is **where each clause is grounded**.

## Current pipeline (`origin/main`)

```text
encoder:  64³ → 32³ → 16³ → 8³     stem, stage1, stage2, bottleneck
                                      C1     C2     C3     C4
                                      32     64    128    256     (default)
                                       8     16     32     64     (smoke)

structure tokens S: pooled from bottleneck (8³)

EvidenceHead queries: bottleneck, 8³, 512 tokens
H_i:                  [B, Ce, 8, 8, 8]
intersection:         added onto the bottleneck
decoder:              8³ → 16³ → 32³ → 64³
                      concat skips (stage2, stage1, stem)
                      FiLM at 16³ and 32³
```

Relevant wiring today, in `RelationalVLM.forward`:

```python
evidence, clause_tokens = self.fusion(
    features.bottleneck, relation_tokens, structure_tokens
)
conditioned = features.bottleneck + self.intersection(evidence)
logits = self.decoder(conditioned, features.skips, context, full_shape=volume_shape)
```

`CrossModalFusion` is constructed with `visual_channels = bottleneck_channels`
and `grid_shape = (8, 8, 8)`. `IntersectionFusion.out_channels` is also
`bottleneck_channels`, so the residual can be added to the 8³ grid.

FiLM at 16³/32³ stays. It still cannot relocate mass; after this change it only
has to refine a peak that already lives on a 4-vox grid.

## Target pipeline

```text
encoder:              unchanged (64³ → 32³ → 16³ → 8³)
structure tokens S:   still pooled from the 8³ bottleneck
EvidenceHead queries: stage2, 16³, 4096 tokens  (+ PE3D at 16³)
H_i:                  [B, Ce, 16, 16, 16]
intersection:         added onto stage2 (16³)
decoder:              still 8³ → 16³ → 32³ → 64³
                      first upsample of the (unconditioned) bottleneck
                      concatenates the conditioned stage2 skip
                      FiLM at 16³ and 32³, unchanged
```

New wiring:

```python
evidence, clause_tokens = self.fusion(
    features.stage2, relation_tokens, structure_tokens
)
conditioned_skip = features.stage2 + self.intersection(evidence)
skips = (conditioned_skip, features.stage1, features.stem)
logits = self.decoder(
    features.bottleneck, skips, context, full_shape=volume_shape
)
```

The relational peak therefore enters the decoder at 16³ through the skip, which
is how a U-Net carries high-resolution signal. The 8³ bottleneck stays as
global visual context and as the structure-encoder input.

Attention cost: 4096 queries × 3 keys, once per clause. Eight times the current
512-token grounding, still eight times cheaper than 32³, 64 times cheaper than
64³.

## What must not change

- Encoder depths, strides, widths, world-coordinate injection.
- `bottleneck_resolution: 8`. The test
  `test_the_bottleneck_is_the_512_token_budget_claude_md_sets` must still pass.
  Do **not** set `bottleneck_resolution: 16`.
- StructureEncoder still reads the 8³ bottleneck.
- Decoder topology: three up-stages `8 → 16 → 32 → 64`, FiLM at 16 and 32.
- Forward signature: `(anchor_masks, direction_ids, anchor_shape_ids)`.
- Forbidden fields. No scene volume, no target mask, no target centroid as
  input.
- Loss, optimiser, augmentation, centroid head, focal BCE. Those are other
  experiments.
- `forbid_full_resolution_attention: true`. Reject 32³ and 64³ grounding.

## Config

In `configs/model.yaml`, `stage_b.fusion`:

```yaml
attention_resolution: 16   # was 8. Queries live at input/4, i.e. encoder stage2.
forbid_full_resolution_attention: true
```

`attention_resolution` is in **full-volume voxels**. On a 64³ input it is 16.
On the 16³ TINY test fixture it is 4. The invariant is:

```text
attention_resolution == input_resolution // 4 == encoder stage2 grid
attention_resolution > bottleneck_resolution
attention_resolution < input_resolution
attention_resolution ** 3 <= 4096
```

Read this key in `RelationalVLMConfig.from_config`. Default it to
`input_resolution // 4` if omitted, so smoke and TINY keep working.

Do not change `bottleneck_resolution`. Do not change smoke encoder widths.

## Code changes, file by file

### `src/models/relational_vlm.py`

- Add `attention_resolution: int` to `RelationalVLMConfig` (default
  `input_resolution // 4` after `__post_init__` resolution is known; for the
  dataclass default on the 64³ model, `16`).
- `grid_shape` becomes `(attention_resolution,) * 3`, **not**
  `(bottleneck_resolution,) * 3`.
- Add `grounding_channels` (encoder stage2 width, `encoder_channels[-2]`).
- Keep the existing check that `bottleneck_resolution == input_resolution // 8`
  and that `bottleneck_resolution ** 3 <= 512` in spirit (the current code
  uses a 4096 cap on the *bottleneck*; leave that as a bottleneck cap, and add
  a **separate** cap `attention_resolution ** 3 <= 4096`).
- Reject `attention_resolution` in `{32, 64}` when `input_resolution == 64`.
- Construct `CrossModalFusion` with `visual_channels=grounding_channels` and
  `grid_shape=(attention_resolution,) * 3`.
- Construct `IntersectionFusion` with `out_channels=grounding_channels` so the
  residual adds to stage2.
- `forward`: query `features.stage2`; add intersection to `features.stage2`;
  pass the **unconditioned** `features.bottleneck` into the decoder; pass
  skips `(conditioned_stage2, stage1, stem)`.
- Docstrings / `RelationalVLMOutput.evidence` comment: three **16³** maps on
  the default model (four-times-coarser-than-input in general).

### `src/models/cross_modal_fusion.py`

- Module docstring: queries are the 16³ stage2 grid (4096 tokens), not 8³.
- `EvidenceHead` already takes `grid_shape` and builds PE from it. Pass 16³.
  No new attention class.
- `visual` in `forward` is stage2, so `visual_channels` must match C3.

### `src/models/intersection_fusion.py`

- `out_channels` is now the stage2 width, not the bottleneck width.
- Spatial size of `H_i` becomes 16³. The product term is unchanged.

### `src/models/decoder.py`

- No topology change. Update comments that call the incoming grid
  "intersection-conditioned 8³": the bottleneck entering the decoder is now
  **unconditioned**; the relational residual arrives via the first skip.

### `src/models/relational_encoder.py`

- No structural change. `EncoderFeatures.stage2` is the grounding grid.
  Optionally document that.

### `docs/stage_b_architecture.md` and `docs/CLAUDE.md`

- Replace "visual locations at 8³ / 512 tokens" as the **grounding** grid with
  16³ / 4096 tokens.
- Keep "do not apply 64³ global attention". Explicitly allow 16³, still forbid
  32³ and 64³.
- Keep the 8³ bottleneck as the structure-encoder / global-context grid.

## Tests

Update `tests/test_stage_b_contract.py`:

- `test_the_bottleneck_is_the_512_token_budget_claude_md_sets` **stays**.
  Bottleneck remains 8³. Building `bottleneck_resolution=16` at input 64 must
  still raise.
- Rename and rewrite
  `test_evidence_maps_live_at_the_bottleneck_not_at_full_resolution`:
  evidence lives at **stage2**, not at the bottleneck and not at full
  resolution. On TINY (`input_resolution=16`) that is `(4, 4, 4)`, not
  `(2, 2, 2)`.
- Add: evidence spatial size equals `input_resolution // 4`.
- Add: `attention_resolution=32` or `64` on a 64³ config raises.
- Add: intersection output channels equal stage2 channels, and
  `stage2 + intersection(evidence)` is a valid add.
- Existing correspondence tests (permute channels, flip a direction) must
  still pass. Grounding at 16³ must remain clause-specific.
- `tests/test_model_contract.py` / smoke build: default model evidence
  `[B, Ce, 16, 16, 16]`, logits still `[B, 1, 64, 64, 64]`.

Run at least:

```text
.venv/bin/python -m pytest tests/test_stage_b_contract.py tests/test_model_contract.py tests/test_stage_b_training.py -q
```

A smoke train is optional and not required to land the architecture change.

## Acceptance

The change is done when:

1. Default model: `H_i` is `[B, Ce, 16, 16, 16]`; logits are `[B, 1, 64, 64, 64]`.
2. Bottleneck is still 8³ / 512 tokens.
3. Full-resolution attention is still impossible to construct.
4. Clause independence still holds (mutating clause 3 does not change `H_1`).
5. No training hyperparameter in `configs/train.yaml` has been edited.

This is not expected to produce Dice 0.75 by itself. It is the prerequisite:
placement has to leave the 8-vox grid before extent losses can help. The
baseline to beat after this lands is the `origin/main` Phase 3 oracle run, not
later loss-tuning runs.
