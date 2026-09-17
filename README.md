# Relational3D-VLM

Proof-of-concept 3D vision-language segmentation: segment an unknown target
structure from a prompt that contains **only** spatial relations to three named
anchor structures.

```text
segment the shape that is lateral to the cube, superior to the pyramid, and anterior to the sphere.
```

The target shape is never an input. The model must infer it from the
intersection of the three relations. `docs/CLAUDE.md` is the full specification.

Scenes are **MRI-like**: structures sit inside a simulated head at a few percent
contrast, with partial-volume edges, tissue texture, a receive-coil bias field
and Rician noise, and are written as NIfTI (`.nii.gz`). See
`docs/mri_appearance.md`.

## Status — Phases 0-3 implemented

Implemented:

- repository scaffold and the five configs (`configs/`);
- the fixed ten-shape vocabulary (`configs/shapes.yaml`, `src/data/primitives.py`);
- the direction rules and their rejection cases (`src/data/direction_rules.py`);
- the MRI-like appearance model and the NIfTI corpus layout
  (`configs/appearance.yaml`, `src/data/{appearance,nifti_io,scene_io}.py`) —
  see `docs/mri_appearance.md`;
- the canonical prompt renderer/parser and nearest-feasible anchor selection
  (`src/data/prompt_generator.py`);
- analytic voxelisers for all ten primitives (`src/data/voxelization.py`);
- scene generation with rejection sampling, grid escalation and per-seed
  reproducibility (`src/data/scene_generator.py`);
- the dataset schema, Stage B input contract and manifest I/O (`src/data/schema.py`);
- fail-fast validation, the rejection log and an independent mask-level relation
  verifier (`src/data/validation.py`);
- `scripts/generate_dataset.py` and the Phase 0 inspected batch,
  `scripts/run_smoke_test.py`.

Phase 1 (Stage A), implemented:

- promptable residual 3D U-Net (`src/models/{blocks,prompt_encoder,shape_segmenter}.py`),
  documented in `docs/flowchart/phase1_encoder_decoder.drawio` — see
  `docs/stage_a_architecture.md`;
- Dice + BCE with deep supervision (`src/training/losses.py`);
- per-class Dice and IoU (`src/evaluation/metrics.py`);
- scene dataset, trainer and checkpointing with full provenance;
- `scripts/train_shape_segmenter.py`, with a smoke run that trains in ~105 s on MPS.

Phases 2–3 (Stage B), implemented:

- the relational target segmenter and its six modules
  (`src/models/{relational_encoder,structure_encoder,prompt_encoder,cross_modal_fusion,intersection_fusion,decoder,relational_vlm}.py`)
  — see `docs/flowchart/phase2_model.drawio` and `docs/stage_b_architecture.md`;
- the example dataset, the anchor-source switch (`src/models/anchor_provider.py`),
  the Phase 2 overfit runner and the Phase 3 trainer;
- Hausdorff distance and stratified Stage B metrics (`src/evaluation/metrics.py`);
- `scripts/train_relational_model.py` — Phase 2 passes on the smoke corpus
  (clean train Dice 0.953 after 864 steps), and a 40-epoch smoke Phase 3 reaches
  validation Dice 0.186 on the two never-supervised target classes.

Not started: the Phase 4 report proper — the baseline sweep, the scored
counterfactual battery and the qualitative 3D dumps (`scripts/evaluate.py`,
`src/evaluation/{counterfactuals,qualitative}.py`).

## Generating data

```bash
.venv/bin/python scripts/run_smoke_test.py                 # 24 scenes, generate + validate
.venv/bin/python scripts/generate_dataset.py --smoke       # smoke corpus only
.venv/bin/python scripts/generate_dataset.py               # full 500-scene corpus
.venv/bin/python scripts/generate_dataset.py --limit 10 --output-root /tmp/try
```

The full run writes `data/processed/{scenes,manifests,run_metadata.json}` and
takes roughly four minutes; generated volumes stay out of git.

Each scene is a directory of NIfTI volumes plus a JSON record:

```text
scenes/<scene_id>/
  image.nii.gz                 the simulated MRI-like volume
  labels.nii.gz                instance labels, 0 and 1..10
  occupancy.nii.gz             binary foreground
  masks/<id>_<name>.nii.gz     one binary mask per structure
  examples/<example_id>_{target,anchors}.nii.gz + .json
  scene.json                   seeds, placed parameters, appearance draws
```

Redundant on purpose: any scene, or any single example, opens in ITK-SNAP,
FSLeyes or 3D Slicer without deriving anything. The frame is RAS with a
`diag(spacing, 1)` affine, so a centroid printed by a manifest is the coordinate
a viewer shows. `docs/dataset_schema.md` has the full layout.

To look at what was generated:

```bash
.venv/bin/python scripts/preview_scene.py --root data/smoke --limit 4
.venv/bin/python scripts/preview_scene.py --root data/smoke \
    --scene scene_009000000 --example scene_009000000_target_01
```

PNG montages, no matplotlib needed: the three orthogonal planes of the image,
the same planes with the labels over them, and the target and three ordered
anchor channels of an example.

### The appearance model

`configs/appearance.yaml` decides what a scene looks like once its geometry is
fixed: partial-volume edges, per-structure intensities on a tissue background, a
scalp-like rim, two scales of texture, a bias field, and a k-space magnitude
reconstruction that supplies Gibbs ringing and Rician noise. Two properties are
load-bearing:

- **intensity carries no class information.** Every structure's mean is drawn
  from one shared distribution, so a model cannot recognise a class from its
  grey level; the target stays reachable only through the three relations. Real
  deep grey nuclei are near-isointense with each other too, so this is also the
  faithful choice. A test guards it.
- **geometry is untouched.** The appearance draws from a disjoint random stream,
  so the same seed gives the same layout with or without an image. `enabled:
  false` restores the previous milestone's binary volumes, and
  `image_source="occupancy"` on either dataset gives the binary input without
  regenerating anything — that is the baseline to compare against.

## Training Stage A

```bash
.venv/bin/python scripts/train_shape_segmenter.py --smoke   # 40 epochs, ~110s on MPS
.venv/bin/python scripts/train_shape_segmenter.py           # full corpus
```

Stage A segments any requested shape name from the scene image, which is how
Phase 4 extracts the three anchors a prompt names. It is scored with per-class
Dice and IoU; checkpoints land in `runs/stage_a[_smoke]/`.

## Training Stage B

```bash
# Phase 2: overfit one scene - the bug catcher. Exits non-zero if it fails.
.venv/bin/python scripts/train_relational_model.py --phase overfit --smoke
.venv/bin/python scripts/train_relational_model.py --phase overfit

# Phase 3: oracle-anchor training - the primary measurement.
.venv/bin/python scripts/train_relational_model.py --phase oracle --smoke
.venv/bin/python scripts/train_relational_model.py --phase oracle

# Phase 4: same weights, Stage A's anchors instead of the ground-truth ones.
.venv/bin/python scripts/train_relational_model.py --eval-only \
    --checkpoint runs/stage_b_oracle/best.pt \
    --anchor-source predicted --stage-a-checkpoint runs/stage_a/best.pt
```

Stage B receives only three ordered anchor-mask channels and the three-clause
prompt — never the scene, the labels or the target. Where those channels come
from is one flag:

| `--anchor-source` | Three channels are |
| --- | --- |
| `oracle` (default) | the ground-truth masks: `data/processed/scenes/<scene>/labels.nii.gz` with everything but the three named anchor structures dropped |
| `predicted` | Stage A's segmentation of those same three shape names, in prompt order |

A predicted run also reports the anchor masks' own Dice/IoU, so a drop against
the oracle run can be attributed to Stage A instead of guessed at. Results,
stratified by target shape, anchor shape, direction and clause slot, land in
`runs/stage_b_*/`.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

Torch is only needed from Phase 1 onwards; install the build that matches the
target hardware (MPS on Apple Silicon, CUDA on the RTX 5090).

## Layout

```text
configs/     shapes, generator, appearance, split, model, train
src/data/    primitives, voxelization, appearance, scene_generator,
             direction_rules, prompt_generator, schema, nifti_io, scene_io,
             validation
src/models/  Stage A segmenter, Stage B encoder/decoder, prompt and structure
             encoders, cross-modal and intersection fusion
src/training/  losses, trainer, augmentations, checkpointing
src/evaluation/  metrics, counterfactuals, qualitative
scripts/     generate_dataset, train_shape_segmenter, train_relational_model,
             evaluate, run_smoke_test, preview_scene
tests/       direction rules, primitives, anchor selection, prompt/schema,
             Stage A and Stage B model contracts, training, augmentations
docs/        CLAUDE.md, dataset_schema.md, mri_appearance.md,
             experiment_protocol.md, stage_a_architecture.md,
             stage_b_architecture.md
```

## Conventions

Arrays are indexed `(z, y, x)`; world coordinates are ordered `(x, y, z)` in a
RAS frame (`x` right/lateral, `y` anterior, `z` superior). NIfTI files store the
transpose, `(i, j, k) = (x, y, z)`, with a `diag(spacing, 1)` affine. Directions come from
a closed six-token vocabulary and are always described target-relative-to-anchor.
See `docs/dataset_schema.md`.
