# Relational3D-VLM

## Project identity

**Relational3D-VLM** is a proof-of-concept 3D vision-language segmentation system. Its purpose is to segment an unknown target structure from a prompt containing only spatial relations to three named anchor structures.

The target shape is never provided as an input. The model must infer the target from the intersection of the three relations, for example:

```text
segment the shape that is lateral to the cube, superior to the pyramid, and anterior to the sphere.
```

The project must remain useful for a later MRI setting in which anchor masks are predicted by a separate anatomy/structure segmenter rather than manually supplied.

## Non-negotiable constraints

- Framework: PyTorch. Keep the implementation compatible with both an Apple Silicon laptop and an NVIDIA RTX 5090.
- Input/output resolution: default input and output are `64 x 64 x 64` voxels. The encoder may reduce the spatial resolution internally, but the final prediction must return to `64 x 64 x 64`.
- Each synthetic scene contains exactly one instance of each of ten fixed shape classes.
- Shapes must be fully contained and must not overlap.
- Scenes carry a simulated MRI-like image: partial-volume edges, per-structure intensities on a tissue background, texture, a receive-coil bias field and Rician noise from a k-space magnitude reconstruction. Structure intensities must be drawn independently of shape class, so appearance can never identify a structure — the target stays reachable only through the three relations. `configs/appearance.yaml` + `docs/mri_appearance.md`; `enabled: false` there restores the earlier binary volumes (one foreground intensity on a zero background) as a baseline.
- The **labels** are not blurred. `instance_labels` and every mask derived from it stay crisp centre-in-solid masks, as a manual segmentation does over a real scan.
- Shapes are axis-aligned in the first version. Do not introduce rotations until a later experiment.
- The relational model receives three ordered anchor-mask channels, the prompt representation, geometry derived from those masks, and the scene image. It must not receive instance labels, the target mask, the target class, the target centroid, or the target instance ID.
- Anchor channels must remain separate. A union of the three anchor masks is allowed only for an ablation baseline, never as the main representation.
- The three prompt directions must always be different.
- All random operations must use explicit, saved seeds.
- Every generated example must be validated before it is written to disk.

## Coordinate system and direction semantics

Use RAS world coordinates with voxel spacing `(1, 1, 1)` unless a configuration explicitly changes spacing:

- `x`: right/lateral axis;
- `y`: anterior axis;
- `z`: superior axis.

Use array indexing `(z, y, x)`. For a `64^3` volume, the world-space center is:

```text
c = ((D - 1) / 2, (H - 1) / 2, (W - 1) / 2)
```

For a target and an anchor, compute:

```text
delta = centroid(target) - centroid(anchor)
```

The prompt describes the target relative to the anchor. Therefore, if the target is to the right of the cube, the relation is `lateral to the cube`; if it is closer to the center plane than the cube, the relation is `medial to the cube`.

Choose the main axis using the largest absolute component of `delta` (physical-spacing normalized if spacing is not isotropic):

```text
axis = argmax(abs(delta_x), abs(delta_y), abs(delta_z))
```

Use deterministic tie priority `z`, then `y`, then `x`.

- `z > 0` => `superior`; `z < 0` => `inferior`.
- `y > 0` => `anterior`; `y < 0` => `posterior`.
- When `x` is selected, compare distance to the volume center plane: `abs(target_x - center_x)` versus `abs(anchor_x - center_x)`. Target farther from the center is `lateral`; target closer to the center is `medial`.

If the x-axis is selected but the two distances are equal, apply a deterministic rejection rule and regenerate the scene; never invent a direction. Keep the direction-generation version in metadata.

Accepted direction tokens are exactly:

```text
anterior, posterior, superior, inferior, medial, lateral
```

No synonyms, ordinal references, or target-shape names may appear in generated prompts.

## Fixed shape vocabulary

Every scene contains exactly one of each of these ten primitives:

1. `cube`
2. `cuboid`
3. `sphere`
4. `ellipsoid`
5. `cylinder`
6. `cone`
7. `pyramid`
8. `triangular_prism`
9. `torus`
10. `capsule`

The vocabulary must be defined once in configuration and referenced by stable integer IDs. Shape names are the only anchor identifiers in the prompt because there is exactly one instance of every shape in a scene.

## Synthetic scene generation

### Geometry

Generate a scene on a `64 x 64 x 64` grid. Use simple analytic voxelizers for the ten primitives. Randomize shape centers and shape-specific dimensions within configured ranges, but do not randomize orientation in this milestone.

Use an initial occupancy target of approximately 8–22% of each axis per object, with class-specific ranges adjusted so occupied volumes are broadly comparable. Enforce a configurable in-bounds margin.

Use rejection sampling:

1. sample all shape parameters;
2. voxelize the candidate;
3. reject if any object is empty, out of bounds, or overlaps an existing object;
4. continue until all ten objects are present.

There is no required minimum separation beyond non-overlap. If packing repeatedly fails at `64^3`, regenerate the scene with new parameters first. Do not silently omit or duplicate an object. If a configurable maximum number of scene attempts is exhausted, increase all three dimensions together (for example `72^3`, then `80^3`) and record the actual `volume_shape`; the output/crop contract must still be explicit.

Save `image` (the simulated MRI-like volume, written as `scene_volume.nii.gz`), the binary foreground (derived from labels at load time) and `instance_labels` (written as `instance_labels.nii.gz`). Per-example inspection files are `examples/<example_id>/{target_mask,anchor_{slot}_{shape},anchor_union}.nii.gz`. See `docs/dataset_schema.md`.

Structures must be placed inside the simulated head, not in the air around it.

### Prompt and anchor generation

For every scene, generate one prompt-target example for each of the ten instances. The complete generated scene therefore has ten candidate examples. Split filtering later determines which target classes are used for training, validation, and testing.

For each target:

1. compute centroids of the target and the other nine objects;
2. rank candidates by Euclidean centroid distance, then instance ID as a deterministic tie-break;
3. compute the target-relative direction for every candidate;
4. scan candidates in ranked order and keep the nearest candidate whose direction has not already been used;
5. stop after three anchors with three distinct directions;
6. if no three-direction set exists, reject and regenerate the entire scene.

The selected set is the nearest feasible set, not necessarily the literal three nearest objects. Anchor order is ascending distance with instance ID as tie-break, and this order must be identical in the text, structured prompt, and three mask channels.

Canonical natural-language rendering:

```text
segment the shape that is {direction_1} to the {anchor_1}, {direction_2} to the {anchor_2}, and {direction_3} to the {anchor_3}.
```

Canonical structured representation:

```python
[
    {"direction": direction_1, "anchor": anchor_1},
    {"direction": direction_2, "anchor": anchor_2},
    {"direction": direction_3, "anchor": anchor_3},
]
```

The structured representation is the primary model interface. A deterministic parser/renderer must guarantee round-trip equivalence between it and the natural-language prompt.

Each saved example must contain at least:

```text
scene_id, example_id, seed, volume_shape, spacing,
scene_volume, instance_labels, target_instance_id, target_shape_name,
target_mask, anchor_instance_ids[3], anchor_shape_names[3],
anchor_masks[3,D,H,W], anchor_union_mask, relations[3],
anchor_centroids_world[3,3], anchor_extents_world[3,3],
target_centroid_world, prompt, structured_prompt, generator_version
```

Validation must fail fast if exactly ten shape instances do not exist; any object is empty, out of bounds, or overlapping; the target occurs in an anchor channel; a channel does not match its declared anchor; anchor order differs from prompt order; or the prompt does not round-trip from structured metadata.

## Dataset split and generalization protocol

Generate 500 scenes with disjoint scene seeds:

- 400 scenes for training;
- 50 scenes for validation;
- 50 scenes for testing.

Use target-class filtering to test compositional transfer:

- training targets: seven shape classes;
- validation targets: two different shape classes;
- test targets: one different shape class.

Choose the seven/two/one assignment once, deterministically, to maximize geometric diversity. Store it in the experiment configuration. All ten shape classes remain available as anchor names and may appear as anchors in every split; only the supervised target class is held out.

The first milestone therefore contains approximately 2,800 training examples, 100 validation examples, and 50 test examples if every eligible target is retained. Report exact counts after filtering. Treat 500 scenes as a proof-of-concept size; scale later for stronger scientific claims.

Use independent layout/seed ranges for every split. Add a secondary evaluation protocol with held-out size ranges and held-out spatial packing patterns when the baseline is working.

## Two-stage model architecture

The implementation has two distinct networks and two evaluation modes.

### Stage A: full-volume shape segmenter

Train a multiclass 3D U-Net-style segmenter on the complete synthetic scene image. It predicts ten shape masks from that image. This is the synthetic analogue of the future anatomy segmenter that will produce anchor masks from MRI, which is why its input has to be an acquisition rather than a binary volume.

Stage A must be evaluated independently with per-class Dice and IoU. Save its checkpoints and support inference that extracts the three requested anchor masks by shape name.

### Stage B: relational target segmenter

Stage B receives exactly three ordered binary anchor-mask channels, the structured three-clause prompt (or equivalent closed-vocabulary embedding), geometry features derived from those masks, and the scene image. The image is a decoder-side WHAT stream: the encoder never sees it. Still forbidden: instance labels, target mask, target class, target centroid, target instance id.

Use a 3D encoder-decoder with skip connections:

```text
input: 3 x 64 x 64 x 64 anchor-mask tensor
encoder: 64^3 -> 32^3 -> 16^3 -> 8^3
bottleneck: 8^3 visual feature grid
decoder: 8^3 -> 16^3 -> 32^3 -> 64^3
output: one target logit volume at 64^3
```

The 8³ bottleneck contains only 512 spatial tokens, making global cross-attention practical. Do not apply full-resolution 64³ global attention, which would create 262,144 spatial tokens. Start with configurable channels such as `32, 64, 128, 256`, with a smaller laptop smoke configuration.

### Anchor structure encoder

For each anchor channel, compute a separate structure token containing masked pooled encoder features, normalized world centroid, normalized bounding-box extent, normalized voxel volume, a learned slot embedding for slots 1–3, and a learned shape embedding. The three tokens must not be pooled into one undifferentiated token.

### Closed-vocabulary prompt encoder

Use learned embeddings for the six directions, ten shape names, three clause slots, and each `(direction, anchor-shape)` pair. The natural-language path must map deterministically to the same three clause tokens. Preserve all three clauses; do not average the prompt into a single vector before grounding.

```text
relation_token_i = direction_embedding_i
                  + shape_embedding_i
                  + pair_embedding(direction_i, shape_i)
                  + slot_embedding_i
```

### Relational grounding and target-between-anchors representation

The target is not inside an anchor. It is the object satisfying all three target-relative constraints. Use three independent relation-conditioned branches:

1. fuse relation token `i` with structure token `i` using a small cross-attention block or gated MLP;
2. broadcast the resulting relation token to the bottleneck visual grid;
3. compute a relation-specific spatial evidence map `H_i`;
4. combine `H_1`, `H_2`, and `H_3` using an explicit intersection module.

The intersection module may receive `[H_1, H_2, H_3, H_1*H_2*H_3]` through a pointwise fusion block. It must not collapse the three `(direction, anchor)` pairs into one pooled text vector before spatial grounding.

Use bottleneck visual locations as queries and relation/structure tokens as keys/values for the main cross-attention. Add lightweight FiLM/gating or cross-attention conditioning at the 16³ and 32³ decoder stages. Do not use expensive full-resolution global attention.

### Position embeddings and decoder

Add continuous normalized world-coordinate features `(x, y, z)` to visual features at every scale. Recompute them correctly after resizing or cropping; local tensor indices are insufficient. Add centroid, extent, volume, and slot embeddings to each anchor token.

At each decoder stage, upsample, fuse the matching encoder skip feature, concatenate the occupancy channel at 16³, 32³ and 64³, inject relation-fused context, and apply 3D convolutional refinement. The final head returns one target logit volume. During evaluation, also return sigmoid probabilities and a thresholded binary mask.

## Training procedure

### Phase 0: data verification

Run geometry and prompt tests before model training. Generate a small inspected batch showing every shape, direction, prompt order, and anchor channel. Verify that the target is never present in conditioning channels.

### Phase 1: Stage A pretraining

Train the full-volume shape segmenter on all ten shapes with multiclass cross-entropy plus Dice loss, or a documented equivalent. Confirm that all ten shapes can be segmented before using predicted anchors.

### Phase 2: relational overfit

Train Stage B on one scene until it reaches near-perfect training Dice. This catches channel-order, coordinate, prompt, and decoder bugs.

### Phase 3: oracle-anchor training

Train and evaluate Stage B using ground-truth anchor masks. This isolates relational architecture quality from Stage A errors and is the primary proof-of-concept measurement.

### Phase 4: predicted-anchor evaluation

Run Stage A on the scene, extract the three anchor masks by names in the prompt, and pass them to Stage B. Report the performance difference between oracle and predicted anchors.

Use this Stage B objective:

```text
L = lambda_dice * DiceLoss(logits, target_mask)
  + lambda_bce  * BCEWithLogitsLoss(logits, target_mask)
```

Save loss weights, optimizer, scheduler, precision mode, and batch size in configuration. Do not add target-shape classification loss in this milestone.

Use mixed precision and gradient accumulation only when needed for the selected hardware. Keep separate CPU/MPS smoke and RTX 5090 training configurations.

### Augmentation rules

Use only relation-preserving augmentations in the primary experiment. Any flip, translation, crop, or axis permutation must update scene/instance masks, anchor channels, world coordinates, centroids/extents, direction labels, and prompt clauses. Do not enable an augmentation until its relation rewrite has a unit test. Do not add arbitrary rotations in this milestone.

## Evaluation

Report target-mask Dice, IoU, and Hausdorff distance, both aggregate and stratified by target shape, anchor shape, direction, successful generation count, and oracle versus predicted anchors.

The main generalization result is performance on the two validation-only target shapes and one test-only target shape. Also evaluate held-out layouts, sizes, and positions.

Mandatory baselines:

1. anchor-mask-only model with no prompt;
2. prompt-only model with coordinates but no anchor masks;
3. prompt plus anchor-union-mask model;
4. full ordered three-anchor relational model.

Mandatory counterfactual tests:

- permute anchor-mask channels while keeping the prompt fixed;
- permute relation clauses while keeping channels fixed;
- replace one direction with its opposite;
- replace one anchor shape name;
- remove one anchor;
- reorder channels and correspondingly reorder prompt clauses.

The full model should be sensitive to correspondence and should outperform the union-mask baseline on relational counterfactuals. A high Dice score is insufficient if shuffled prompts and masks produce nearly identical predictions.

Occupancy sanity (run after Stage B training and `--eval-only`): permute the three anchor channels and flip one direction with occupancy held fixed — the mask must move or collapse; compare train vs val Dice (both high means localisation is generalising; train high / val low means WHERE is still the problem); a qualitative slice plus a component count must show one remaining object, not the union of the seven.

## Reproducibility and engineering requirements

- Save configuration, random seeds, generator version, direction-rule version, vocabulary version, git revision, hardware, and checkpoint metadata for every run.
- Keep data generation, geometry, prompt logic, models, training, evaluation, and visualization in separate modules. No notebook-only production logic.
- Add tests for voxelization, containment, overlap rejection, centroid computation, direction classification, tie-breaking, unique-anchor selection, prompt round-trip, tensor/prompt alignment, and augmentation rewrites.
- Log scene rejection reasons and acceptance rate.
- Fail fast on malformed examples rather than silently repairing them.
- Store large generated volumes outside git; commit manifests, schemas, configs, and small fixtures.

## Recommended repository layout

```text
configs/
  shapes.yaml
  generator.yaml
  split.yaml
  model.yaml
  train.yaml
src/
  data/
    primitives.py
    voxelization.py
    appearance.py
    scene_generator.py
    direction_rules.py
    prompt_generator.py
    schema.py
    nifti_io.py
    scene_io.py
    validation.py
  models/
    shape_segmenter.py
    relational_encoder.py
    structure_encoder.py
    prompt_encoder.py
    cross_modal_fusion.py
    intersection_fusion.py
    decoder.py
    relational_vlm.py
  training/
    losses.py
    trainer.py
    augmentations.py
    checkpointing.py
  evaluation/
    metrics.py
    counterfactuals.py
    qualitative.py
    occupancy_sanity.py
scripts/
  generate_dataset.py
  train_shape_segmenter.py
  train_relational_model.py
  evaluate.py
  run_smoke_test.py
  preview_scene.py
tests/
  test_primitives.py
  test_directions.py
  test_anchor_selection.py
  test_prompt_schema.py
  test_model_contract.py
  test_augmentations.py
docs/
  dataset_schema.md
  mri_appearance.md
  experiment_protocol.md
```

## Definition of done

The first milestone is complete only when a clean run can generate and validate the 500-scene corpus, train and evaluate Stage A on all ten shapes, overfit Stage B on a tiny fixture, train the ordered-anchor model with oracle masks, evaluate predicted-anchor inference, evaluate the seven/two/one target split, report Dice/IoU/Hausdorff plus per-shape/per-direction results, run all baselines and counterfactuals, and save qualitative 3D predictions with reproducibility metadata.

Do not claim successful spatial grounding from aggregate Dice alone. The model must demonstrate that it uses direction–anchor correspondence and can segment target classes withheld from supervised training.
