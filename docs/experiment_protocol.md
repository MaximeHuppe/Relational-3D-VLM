# Experiment protocol

## Phases

| Phase | What runs | Status |
| --- | --- | --- |
| 0 | Data verification: geometry and prompt tests, inspected batch | **done** — `scripts/run_smoke_test.py` |
| 1 | Stage A pretraining on all ten shapes | **implemented** — `scripts/train_shape_segmenter.py`; smoke run done, full-corpus run pending |
| 2 | Stage B overfit on one scene | **implemented** — `scripts/train_relational_model.py --phase overfit`; smoke run passes (train Dice 0.95) |
| 3 | Stage B oracle-anchor training (primary measurement) | **implemented** — `scripts/train_relational_model.py --phase oracle`; full-corpus run pending |
| 4 | Predicted-anchor evaluation | anchor path implemented (`--anchor-source predicted`); the full report, baselines and counterfactuals are not |

## Splits

500 scenes with disjoint seed ranges: 400 train (`1000000..1000399`), 50 val
(`2000000..2000049`), 50 test (`3000000..3000049`).

Target-class filtering (seven / two / one), fixed once in `configs/split.yaml`:

| Split | Target classes |
| --- | --- |
| train | `cube`, `sphere`, `cylinder`, `cone`, `pyramid`, `torus`, `capsule` |
| val | `cuboid`, `ellipsoid` |
| test | `triangular_prism` |

Rationale — maximise the geometric diversity of the supervised set. The seven
training targets cover every family in the vocabulary: isotropic box, isotropic
ellipsoidal, constant-section swept, apex swept, apex box, genus-1 toroidal and
capped swept. Each held-out class then has a close geometric parent inside
training (`cuboid` ← `cube`, `ellipsoid` ← `sphere`, `triangular_prism` ←
`cube`/`pyramid`), so transfer is compositional rather than a jump to an unseen
family.

All ten classes remain available as anchor names in every split. Expected counts
if every eligible target is retained: 2,800 / 100 / 50. Exact counts are
reported after generation and filtering.

## Generation

```bash
.venv/bin/python scripts/run_smoke_test.py            # 8 scenes: generate + validate
.venv/bin/python scripts/generate_dataset.py          # full 500-scene corpus
```

Scenes are packed by rejection sampling, largest classes first. A scene is
rejected and regenerated whenever an object cannot be placed, a pair of
centroids has no well-defined direction, or some target has no feasible
three-direction anchor set; the reason is logged and the acceptance rate is
reported. Measured acceptance is roughly 45% of scene attempts at `64^3`, at
about 0.07 s per accepted scene, so the full corpus takes well under a minute.
Grid escalation to `72^3` then `80^3` exists for packing failures and has not
been needed at the configured sizes.

## Stage A (Phase 1)

A promptable residual 3D U-Net: intensity `scene_volume` in, one mask logit volume
per requested shape name out. Architecture, training, loss, metrics and dataset
are in `docs/flowchart/phase1_encoder_decoder.drawio` and
`docs/stage_a_architecture.md`.

```bash
.venv/bin/python scripts/train_shape_segmenter.py --smoke      # 40 epochs on 16 scenes
.venv/bin/python scripts/train_shape_segmenter.py              # full corpus
```

One sample is a whole scene, so the target-class split filter does not apply to
Stage A: it must learn all ten shapes in every split, since any of the ten may
be named as an anchor. Checkpoints are selected on macro-average validation
Dice and carry the full provenance block.

Smoke result (0.50 M-parameter profile, 16 train / 4 val scenes, 40 epochs,
~105 s on MPS): mean validation Dice **0.437**, IoU 0.300, with per-class Dice
spanning roughly 0.21 to 0.69. Which classes lead varies between runs - four
validation scenes and a half-million-parameter model is far too little to rank
classes - so read this as evidence that the loop learns, not as a Stage A
result. The full-corpus run with the `default` profile is what Phase 1 is judged
on, and the gate before Phase 4 is a per-class Dice floor across all ten
classes.

## Stage B (Phases 2-4)

Three ordered anchor-mask channels and a three-clause prompt in, one target logit
volume out. Architecture, training, loss, metrics and dataset are in
`docs/flowchart/phase2_model.drawio` and `docs/stage_b_architecture.md`; the
input contract is the forward signature, which has nowhere to put the scene, the
labels or the target.

### Anchor source

Every Stage B run picks where its three channels come from, and the model is
identical either way:

```bash
--anchor-source oracle      # ground-truth: load instance_labels.nii.gz
                            # and keep only the three named anchors
--anchor-source predicted   # Stage A segments the scene and returns its masks for
                            # those same three names (needs --stage-a-checkpoint)
```

`anchor_shape_ids` is simultaneously Stage B's shape embedding index and Stage
A's `prompt_ids`, so predicted channels arrive already aligned with the clauses.
A predicted run also reports `anchor_dice` / `anchor_iou` /
`empty_anchor_fraction` for the anchors themselves, so the oracle-versus-
predicted delta can be attributed to Stage A rather than guessed at. The scene
volume is read only to feed Stage A — Stage B still receives three mask channels
and nothing else, which `tests/test_stage_b_training.py` asserts.

### Phase 2 — relational overfit

```bash
.venv/bin/python scripts/train_relational_model.py --phase overfit --smoke
.venv/bin/python scripts/train_relational_model.py --phase overfit
```

One scene, every example whose target class the split allows (7 in the smoke
corpus), until training Dice passes `stage_b_overfit.target_train_dice` (0.95) or
the step budget runs out. The run exits non-zero on failure, and the success flag
is confirmed on a clean evaluation pass rather than on the running mean.

This is a bug catcher, not a result: it is what catches a channel/clause
mis-ordering, a coordinate frame that disagrees with the direction rules, or a
decoder that cannot place a mask where the evidence is.

Measured (smoke profile, 0.87 M parameters, 7 examples of one scene, oracle
anchors, CPU, ~10 min): training Dice sits at 0 for the first ~150 steps while
the prior-initialised head finds the target scale, then climbs — 0.90 by step
520, **0.95 at step 864**, clean Dice 0.953 (IoU 0.911, Hausdorff 1.31 voxels).
Phase 2 passes. The per-example spread is narrow (0.91-0.97), so no single
prompt is being ignored.

### Phase 3 — oracle-anchor training

```bash
.venv/bin/python scripts/train_relational_model.py --phase oracle --smoke
.venv/bin/python scripts/train_relational_model.py --phase oracle
```

The full corpus with ground-truth anchors: 2,800 training examples over seven
target classes, validated on the 100 examples of the two held-out classes. The
checkpoint is selected on validation Dice, which here is a *transfer* metric —
`cuboid` and `ellipsoid` are never supervised. This is the primary
proof-of-concept measurement.

The `--smoke` schedule (6 epochs, 112 examples, reduced widths) exists to
exercise the loop end to end in a couple of minutes; it is far too short to
learn anything and its Dice is expected to be 0. Judge Phase 3 on the
full-corpus run.

A longer smoke run does show the loop learning, which is the point of quoting it
here: 40 epochs on the same 16-scene corpus (0.87 M parameters, MPS, ~17 min)
reaches train Dice 0.23 and **validation Dice 0.186 at its best epoch** — on
`cuboid` and `ellipsoid`, which it never saw supervised. That is a signal, not a
result: eight validation examples and a half-million-parameter model cannot
support a claim about spatial grounding.

### Phase 4 — predicted-anchor evaluation

```bash
.venv/bin/python scripts/train_relational_model.py --eval-only \
    --checkpoint runs/stage_b_oracle/best.pt \
    --anchor-source predicted --stage-a-checkpoint runs/stage_a/best.pt
```

No retraining: the Phase 3 weights are re-run with Stage A's anchors in place of
the ground-truth ones, and the same report is written for both sources so the
delta is directly comparable.

Demonstrated end to end on the smoke artefacts — one Stage B checkpoint, seven
examples, two anchor sources:

| Anchors | Anchor Dice | Target Dice | Target IoU | Hausdorff |
| --- | --- | --- | --- | --- |
| oracle | 1.000 | 0.953 | 0.911 | 1.31 |
| predicted (smoke Stage A) | 0.436 | 0.610 | 0.483 | 37.9 |

The anchor column is what makes the target column interpretable: this Stage A
was itself a 40-epoch smoke run, so the drop is an anchor-quality result, not a
relational one. A full-corpus Stage A is the prerequisite for reading the delta
as anything else. The wider Phase 4 deliverables — the baseline
sweep, the counterfactual battery and the qualitative dumps — are not
implemented yet (`src/evaluation/counterfactuals.py`,
`src/evaluation/qualitative.py`, `scripts/evaluate.py`).

## Reports

Every Stage B run writes, next to its checkpoints:

* `history.json` — per-epoch loss, train Dice, validation report;
* `final_<split>_<anchor_source>.json` (or `evaluation_…` for `--eval-only`) —
  Dice, IoU and Hausdorff, aggregate and stratified by target shape, anchor
  shape, direction and clause slot;
* `overfit_report.json` for Phase 2 — pass/fail, the step at which the target was
  reached, and the full training trace;
* `best.json` / `last.json` — the provenance block, including which anchor source
  produced the numbers.

## Objectives

Stage A: per-prompt Dice + BCE over three deep-supervision scales, the
documented equivalent of multiclass cross-entropy + Dice; reported per class.

Stage B: `L = lambda_dice * DiceLoss(logits, target_mask) + lambda_bce *
BCEWithLogitsLoss(logits, target_mask)`, weights in `configs/train.yaml`. Single
scale — no deep supervision — and no target-shape classification loss in this
milestone.

## Evaluation

Dice, IoU and Hausdorff distance, aggregate and stratified by target shape,
anchor shape, direction, successful generation count, and oracle vs predicted
anchors.

Baselines: (1) anchor masks only, no prompt; (2) prompt only with coordinates,
no anchor masks; (3) prompt + anchor union mask; (4) full ordered three-anchor
model. All four are built and runnable today via `--variant`; the sweep that
trains and compares them is not written yet.

Counterfactuals: permute anchor channels with the prompt fixed; permute prompt
clauses with the channels fixed; replace one direction with its opposite;
replace one anchor shape name; remove one anchor; reorder channels and prompt
clauses together (expected to be a no-op). The architecture is *sensitive* to
each of the first five by construction — `tests/test_stage_b_contract.py` checks
that each changes the prediction — but sensitivity is not correctness; the
scored battery on a trained model is still to be written.

The headline result is performance on the two validation-only and one test-only
target classes. Aggregate Dice alone does not demonstrate spatial grounding: if
shuffled prompts and shuffled channels produce nearly identical predictions, the
model is not using direction–anchor correspondence.

## Reproducibility

Every run records the config snapshot, seeds, `generator_version`,
`direction_rule_version`, `vocabulary_version`, `schema_version`, git revision,
hardware and checkpoint metadata. All random operations use explicit saved
seeds; scene rejection reasons and the acceptance rate are logged
(`src.data.validation.RejectionLog`).
