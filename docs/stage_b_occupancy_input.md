# Stage B: give the model WHAT (occupancy), keep WHERE from relations

Implementation spec for a **single architecture change** on the current
`feat-flip-augment` branch (16³ grounding + `rotation_90` already landed). Do
not mix in loss, scheduler, half-space, or decoder-FiLM experiments from
[`docs/stage_b_upgrade_catalog.md`](stage_b_upgrade_catalog.md).

The goal is to stop asking the decoder to **invent** a shape it has never
seen. Relations can say *where* the target is. Only the scene occupancy can
say *which voxels* that object actually occupies.

This **amends** the Stage B input contract in [`docs/CLAUDE.md`](CLAUDE.md).
It is the synthetic analogue of giving Stage B the MRI volume. It is not a
licence to pass `instance_labels`, the target mask, the target class, or the
target centroid.

## Why this is the right change

Stage B today is called as:

```python
model(anchor_masks, direction_ids, anchor_shape_ids)   # and nothing else
```

The encoder sees three binary landmark masks. At the true target location
those channels are **empty**. World XYZ and the prompt can at best light up a
region. The 64³ head then has to hallucinate a cube / torus / cuboid of the
right volume from:

- a coarse 16³ relational peak;
- high-res **anchor** skips (the wrong objects);
- a learned size prior.

That is **WHERE without WHAT**. Dice 0.8 on a 5-voxel-radius object is not a
reasonable ask of that interface. Phase 2 overfit still works because seven
prompts on one frozen layout can be memorised as “prompt k → blob k.” Phase 3
cannot: every layout is new, val shapes are held out, and there is no occupancy
to copy.

| Change | Do it? | Why |
| --- | --- | --- |
| Concatenate `scene_volume` into the **relational encoder** as a 4th channel | **No** | Grounding queries would see the target’s own voxels. The model can ignore the prompt and pick “a blob that is not an anchor.” WHERE and WHAT collapse into one stream. |
| Pass `instance_labels` or the target class | **No** | That is the answer key. Still forbidden. |
| Read GT centroids from the manifest | **No** | Breaks oracle vs predicted interchangeability. |
| Multiply the prediction by occupancy and stop | **No** | A 7-vox offset times occupancy is empty or a neighbour. The WHERE path still has to land on the right object; occupancy must *inform* the decoder, not replace it. |
| **Occupancy as a decoder-side WHAT stream** | **Yes** | Encoder / fusion / intersection stay the WHERE path (anchors + prompt only). The binary scene is concatenated into the decoder so the head can carve the object that the relations pointed at. |

MRI reading: Stage A (or a clinical segmenter) still supplies the three named
anchors. The image itself — here the binary `scene_volume` — is what the
relational model is allowed to look at in order to recover shape. Later that
channel is T1/T2. Instance maps and the target identity stay out.

## What the extra channel is (and is not)

`scene_volume` is the binary foreground of **all ten** shapes, already stored
per scene in `data/processed/scenes/<scene_id>.npz`. Schema check: it equals
`instance_labels != 0`. It has **no instance ids**, no class colours, no
target highlight.

The model still must use the three relations to decide *which* of the ten
connected components is the target. Occupancy only tells it the voxel set of
whatever object sits at the place the relations named.

Zero the three anchor voxels before the decoder sees occupancy
(`occupancy = scene * (1 - union(anchors))`). Then WHAT is “the other seven
objects” and the landmarks are not duplicated. That is the default. A config
flag turns the masking off for an ablation.

## Current pipeline (`feat-flip-augment`)

```text
anchor_masks [B, 3, 64, 64, 64]
  RelationalEncoder  (anchors + XYZ only)
    64³ → 32³ → 16³ → 8³
  StructureEncoder from 8³ + masks
  PromptEncoder from direction_ids, anchor_shape_ids
  EvidenceHead on stage2 → H_i at 16³
  IntersectionFusion → residual on stage2 skip
  RelationalDecoder  8³ → 16³ → 32³ → 64³
    skips = (conditioned stage2, raw stage1, raw stem)
    FiLM at 16³ / 32³
    1×1 head → logits [B, 1, 64, 64, 64]

scene_volume exists on disk and in the dataset only when
include_scene_volume=True (Phase 4 → Stage A). It is listed in
STAGE_B_FORBIDDEN_FIELDS and stripped by stage_b_model_inputs.
```

`ExampleDataset` already loads and rotates `scene_volume` when asked
([`src/data/dataset.py`](../src/data/dataset.py)). The missing piece is
wiring it into Stage B.

## Target pipeline

```text
anchor_masks [B, 3, 64, 64, 64]          scene_volume [B, 1, 64, 64, 64]
        │ WHERE                                     │ WHAT
        ▼                                           ▼
  RelationalEncoder (unchanged:                    occupancy = scene
    3 ordered anchors + XYZ)                       occupancy *= (1 - union(anchors))   # default
        │                                          downsample with max-pool
        │                                          to 16³, 32³, 64³
  fusion / intersection on stage2                  │
        │                                          │
        └──────────────► RelationalDecoder ◄───────┘
                         concat occupancy at 16³, 32³, 64³
                         (after skip merge, before FiLM/refine)
                         head unchanged
                         logits [B, 1, 64, 64, 64]
```

`forward` becomes:

```python
def forward(
    self,
    anchor_masks: Tensor,
    direction_ids: Tensor,
    anchor_shape_ids: Tensor,
    scene_volume: Tensor,
    *,
    return_evidence: bool = False,
) -> RelationalVLMOutput:
    masks = self.prepare_masks(anchor_masks)
    volume_shape = tuple(anchor_masks.shape[2:])
    occupancy = self.prepare_occupancy(scene_volume, masks)

    features = self.encoder(masks, volume_shape)          # still 3 (+XYZ) channels
    relation_tokens = self.prompt_encoder(direction_ids, anchor_shape_ids)
    structure_tokens = self.structure_encoder(masks, features.bottleneck, anchor_shape_ids)
    evidence, clause_tokens = self.fusion(features.stage2, relation_tokens, structure_tokens)
    conditioned_stage2 = features.stage2 + self.intersection(evidence)
    skips = (conditioned_stage2, features.stage1, features.stem)
    logits = self.decoder(
        features.bottleneck,
        skips,
        clause_tokens.flatten(1),
        occupancy=occupancy,          # [B, 1, D, H, W] at full res; decoder downsamples
        full_shape=volume_shape,
    )
```

The relational encoder **must not** see `scene_volume`. If a test concatenates
it into `RelationalEncoder`, reject the PR.

## What must not change

- Encoder depths, strides, widths, world-XYZ injection, 16³ grounding, 8³
  bottleneck, FiLM locations, loss, optimiser, augmentation.
- Structure tokens still pooled from the **anchor** bottleneck, never from
  occupancy.
- Evidence maps still computed from stage2 of the **anchor** encoder.
- `instance_labels`, `target_mask`, `target_shape_name`, `target_instance_id`,
  `target_centroid_world` stay forbidden. They must not appear on `forward`.
- Oracle vs predicted anchors still go through `AnchorProvider`. Predicted
  mode already loads `scene_volume` for Stage A; that same tensor is now
  *also* passed to Stage B. Stage A does not start seeing anything new.
- Baselines (`anchor_masks_only`, `prompt_only`, `prompt_plus_union_mask`)
  still exist. Occupancy stays on for all of them unless a new ablation
  `occupancy=false` is added (see below). Correspondence tests must still
  fail a model that ignores the prompt: occupancy alone is seven remaining
  objects, not one.

## Config

In [`configs/model.yaml`](../configs/model.yaml), under `stage_b`:

```yaml
stage_b:
  in_channels: 3                 # encoder still sees only ordered anchors
  occupancy:
    enabled: true                # the WHAT stream
    mask_anchors: true           # occupancy *= (1 - union of the 3 channels)
    inject_at_resolutions: [16, 32, 64]
    # Do not inject at 8³: that would put WHAT on the unconditioned bottleneck
    # and undo the WHERE/WHAT split.
```

`in_channels` stays 3. Do not bump it to 4.

Smoke profile: no extra widths. Occupancy is one channel; the decoder concat
adds +1 at the named scales only.

## Code changes, file by file

### `src/data/schema.py`

- Remove `"scene_volume"` from `STAGE_B_FORBIDDEN_FIELDS`.
- Keep it in `REQUIRED_EXAMPLE_FIELDS`.
- Add it to `STAGE_B_ALLOWED_FIELDS`.
- Leave `instance_labels` and all `target_*` fields forbidden.
- Update the module docstring: occupancy is allowed, instance ids are not.

### `src/data/dataset.py`

- Stage B dataset: `include_scene_volume=True` by default (today it is False
  except Phase 4).
- Always put `scene_volume` on the item as `[1, D, H, W]` float32.
- Remove `"scene_volume"` from `STAGE_B_NON_INPUT_KEYS`.
- `stage_b_model_inputs` returns four keys:

```python
return {
    "anchor_masks": masks,
    "direction_ids": batch["direction_ids"],
    "anchor_shape_ids": batch["anchor_shape_ids"],
    "scene_volume": batch["scene_volume"],
}
```

- Rotation already rewrites `scene_volume` when it is loaded. Keep that. A
  test must still assert the occupancy rotates with the masks
  (`tests/test_augmentations.py` already covers this).
- Docstring: `scene_volume` is now a Stage B input, not only a Stage A feed.

### `src/models/relational_vlm.py`

- `RelationalVLMConfig`: `use_occupancy: bool = True`,
  `mask_occupancy_anchors: bool = True`,
  `occupancy_at: tuple[int, ...] = (16, 32, 64)`.
- `from_config` reads `stage_b.occupancy`.
- `in_channels` **unchanged** (still 3 or 1 for the union ablation).
- Add `prepare_occupancy(scene_volume, masks) -> [B, 1, D, H, W]`:
  float32, same spatial size as the masks; if `mask_occupancy_anchors`,
  multiply by `(1 - masks.amax(dim=1, keepdim=True))`.
- `forward` gains a required `scene_volume` argument (see signature above).
  If `use_occupancy` is false (ablation), pass `None` through and the decoder
  skips the concat — needed so the four baselines can still run without a
  second code path in the trainer.
- For `prompt_only`, anchors are already zeroed: occupancy masking then
  leaves the full scene, which is correct (no landmarks to subtract).
- Do **not** feed occupancy into `self.encoder`.

### `src/models/decoder.py`

- `RelationalDecoder.__init__`: know `occupancy_at` and current stage
  resolutions. For each stage whose output resolution is in `occupancy_at`,
  the refine/up path must accept `channels + 1` at the concat point.
- Simplest wiring, one extra conv per injected stage:

```text
features = up(coords(features), skip)
if this stage injects occupancy:
    occ = max_pool_or_identity(occupancy, to features.spatial)
    features = cat([features, occ], dim=1)
    features = occ_proj[stage](features)   # 1×1, C+1 → C, so FiLM/refine widths stay
features = film?(features, context)
features = refine(features)
```

- Downsample occupancy with **max-pool** (same helper as
  `downsample_targets` / `downsample_masks(..., mode="max")`) so a thin torus
  does not vanish at 16³.
- Occupancy concat happens **after** the skip merge so WHAT does not sit
  inside the anchor skip; it is an extra channel the decoder can use to carve.
- Head, prior bias, FiLM zero-init: unchanged.
- If `occupancy is None`, skip the concat (ablation / tests that have not
  been updated yet should not be the long-term path; production `forward`
  always passes the tensor).

Do not change `UpBlock` globally in `blocks.py` unless a dedicated
`+1 channel` path is cleaner there. Prefer keeping occupancy logic in
`decoder.py` so the encoder pyramid stays untouched.

### `src/models/relational_encoder.py`

No structural change. Add a comment that this module is the WHERE encoder
and must not receive occupancy.

### `src/models/anchor_provider.py`

Predicted provider already requires `batch["scene_volume"]` for Stage A.
After this change the same tensor is forwarded to Stage B via
`stage_b_model_inputs`. Update the module docstring: the scene now has two
consumers (Stage A → anchors, Stage B decoder → occupancy). It still must
not become `instance_labels`.

### `src/training/trainer.py` / `scripts/train_relational_model.py`

- `StageBTrainer._forward` already goes through `stage_b_model_inputs`. Once
  that helper includes `scene_volume`, the trainer picks it up. No extra
  unpacking.
- `needs_scene_volume = True` for **both** oracle and predicted Stage B
  loaders (today: only predicted). Oracle Phase 3 must load occupancy.
- Overfit (Phase 2) must load it too.

### `configs/model.yaml` and `configs/train.yaml`

- Model: add the `occupancy` block quoted above. Update the stage_b comment
  that says the model never receives `scene_volume`.
- Train: no loss / lr / epoch change. Optionally tag wandb
  `occupancy-decoder`.

### Docs

- [`docs/CLAUDE.md`](CLAUDE.md) Stage B paragraph: inputs are three ordered
  anchor channels, the structured prompt, geometry derived from those masks,
  **and the binary scene occupancy**. Still forbidden: instance labels,
  target mask, target class, target centroid, target instance id.
- [`docs/stage_b_architecture.md`](stage_b_architecture.md): add a WHERE /
  WHAT subsection and redraw the pipeline text. Occupancy is a decoder skip
  of one channel, not an encoder input.
- This file is the implementation spec; the architecture doc is the lasting
  description.

## Tests

Update [`tests/test_stage_b_contract.py`](../tests/test_stage_b_contract.py):

- `test_the_forward_signature_cannot_take_the_scene_or_the_target`:
  rename. Signature **does** take `scene_volume`. It still must not take
  `instance_labels`, `target_mask`, `target_shape_name`,
  `target_instance_id`, `target_centroid_world`.
- `test_the_dataset_helper_passes_only_the_permitted_inputs`:
  `stage_b_model_inputs` returns the four keys including `scene_volume`,
  and still strips `target_mask` / `target_shape_name`.
- Add: encoder input channels stay 3 when occupancy is on
  (`model.encoder.in_channels == 3`).
- Add: `prepare_occupancy` with `mask_anchors=True` is zero wherever any
  anchor channel is 1, and equals `scene * (1 - union)` elsewhere.
- Add: mutating occupancy (flip a non-anchor object on/off) changes logits
  at 64³; mutating occupancy **inside an anchor** with masking on does not.
- Add: mutating a direction still changes logits (correspondence). Occupancy
  must not make the model prompt-invariant. Use the existing randomised-head
  probe.
- Add: `scene_volume` spatial size must match `anchor_masks`; mismatch
  raises.
- Grounding tests stay: evidence still `[B, Ce, 4, 4, 4]` on TINY,
  bottleneck still 2³, no 32³/64³ attention.

Update [`tests/test_stage_b_training.py`](../tests/test_stage_b_training.py):

- Oracle dataset items include `scene_volume` by default.
- `stage_b_model_inputs(item)` contains `scene_volume`.
- Trainer / overfit still never pass `target_mask` into the model.
- Predicted-anchor test: the same `scene_volume` goes to Stage A **and**
  Stage B; `instance_labels` still does not.

Add a small unit test on the decoder: occupancy concat at 16/32/64, not at
the bottleneck; `occupancy=None` keeps old shapes.

Run at least:

```text
.venv/bin/python -m pytest \
  tests/test_stage_b_contract.py \
  tests/test_stage_b_training.py \
  tests/test_model_contract.py \
  tests/test_augmentations.py \
  -q
```

## Acceptance

The change is done when:

1. `forward` is
   `(anchor_masks, direction_ids, anchor_shape_ids, scene_volume)`.
2. `RelationalEncoder` still takes 3 channels (1 for the union baseline).
3. Occupancy is concatenated in the decoder at 16³, 32³, 64³ only, max-pooled
   to each grid, anchors masked out by default.
4. `instance_labels` and every `target_*` field remain absent from `forward`
   and from `stage_b_model_inputs`.
5. Direction / channel permutation probes still move the prediction.
6. No training hyperparameter in `configs/train.yaml` has been edited
   except wandb tags if desired.
7. Phase 2 overfit still reaches Dice 0.95 (it should become easier, not
   harder).

This is expected to raise Phase 3 **train** Dice first: once WHERE is
roughly right, WHAT can copy the connected component. Val on cuboid /
ellipsoid should follow if localisation generalises, because occupancy
supplies the held-out shape instead of a cube-trained prior. If train jumps
and val does not, the remaining problem is WHERE (relations), not WHAT —
then return to catalog items A1 / D4, not to another occupancy trick.

## Ablations to keep runnable (not in this PR)

| Flag | Meaning |
| --- | --- |
| `occupancy.enabled: false` | restore today’s interface (WHERE only), for a delta vs this branch |
| `occupancy.mask_anchors: false` | pass the full ten-object occupancy, including landmarks |
| inject at `[64]` only | weaker WHAT, useful if 16³ concat leaks occupancy into grounding via skip add |

Do not implement those as extra default behaviour. Ship `enabled: true`,
`mask_anchors: true`, `inject_at_resolutions: [16, 32, 64]`.
