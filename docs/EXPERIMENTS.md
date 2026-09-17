# Experiments

Run log for the matrix in [`docs/experiments/experiment_flowchart.drawio`](experiments/experiment_flowchart.drawio).
Fill **path**, **commit**, **status**, and **val dice** as each hexagon completes.

```text
Baseline → Update Dataset → realistic-appearance | mri-like
                              ├─ Phase A (no aug)  → Phase B predicted
                              ├─ Phase A (with aug) → Phase B predicted
                              └─ Phase B oracle
```

## Reproducibility

Yes: **git commit + the command below + the config snapshot** is the recipe.

| Piece | Where it lives | What it pins |
| --- | --- | --- |
| git commit | table **commit** column; also `best.json` → `environment.git_revision` | the code, including `configs/*.yaml` |
| command line | this page, and `runs/experiments/realistic-appearance/campaign.json` | which arm, which flags, which checkpoint paths |
| config snapshot | each run's `best.json` → `configs` | resolved hyperparameters (epochs, lr, seed, early stopping, appearance, split) |
| dataset | `data/processed/run_metadata.json` | the corpus those weights were trained on |

The CLI only selects the arm (`--output`, `--augment` / `--no-augment`, `--anchor-source`, `--stage-a-checkpoint`, `--profile`). Epochs, learning rate, seed and the rest are whatever `configs/train.yaml` said at that commit; the sidecar stores the snapshot so a later edit of the YAML cannot rewrite history.

A dirty working tree breaks the recipe: the SHA no longer describes the code that ran. The runner warns; put the diff in [Notes](#notes) or commit first.

## How to fill a run

1. Find the hexagon in the flowchart; its label is the **run** name.
2. Fill the matching row. Use `—` when a field does not apply (Stage A never has anchors or aug B).
3. Extra context goes under [Notes](#notes) with the same **id**.

To add an arm that is not in the flowchart, copy a row **and** add a hexagon so the two stay in sync.

## Legend

| Field | Values |
| --- | --- |
| **id** | `EX-NNN`, stable once assigned |
| **model** | `shape` (Phase A) · `relational` (Phase B) |
| **run** | hexagon label in the flowchart |
| **path** | run folder or checkpoint, e.g. `runs/shape_segmenter/<run>/best.pt` |
| **commit** | git SHA of the code that produced the run (`git rev-parse --short HEAD`); `—` until started |
| **dataset** | `realistic` · `mri-like` |
| **aug A** | Phase A trained with augmentations: `yes` · `no` · `—` (oracle B has no Stage A) |
| **aug B** | Phase B trained with augmentations: `yes` · `no` · `—` (shape has no Stage B) |
| **anchors** | `—` (shape) · `oracle` · `predicted` |
| **stage A** | `—` unless **anchors** is `predicted`, then the Phase A **run** that produced the anchors |
| **status** | `todo` · `running` · `done` |
| **val dice** | scalar, or `—` if not trained yet |

`augA` / `augB` in a Phase B name match these two flags: `_augB` means Phase B used aug; `_augA_augB` means both stages did.

## Index

### realistic-appearance

Run all five (then dump each Stage B test set) with `scripts/run_realistic_experiments.py`. That script is the source of the commands in this section.

```bash
.venv/bin/python scripts/run_realistic_experiments.py              # rtx5090
.venv/bin/python scripts/run_realistic_experiments.py --profile laptop_mps
.venv/bin/python scripts/run_realistic_experiments.py --dry-run    # print, do not train
.venv/bin/python scripts/run_realistic_experiments.py --only EX-1 EX-5
.venv/bin/python scripts/run_realistic_experiments.py --skip-existing --no-evaluate
```

`--phase oracle` on a Stage B command is the full-corpus training loop, not “oracle anchors”. Anchors are `--anchor-source`. Augmentation is always explicit (`--augment` / `--no-augment`) so a later change of the YAML default cannot silently flip an arm.

The corpus is `data/processed` (generate with `scripts/generate_dataset.py` if it is missing). After each Stage B run the campaign also runs `scripts/evaluate.py` on the test split; pass `--no-evaluate` to skip that.

| id | model | run | path | commit | dataset | aug A | aug B | anchors | stage A | status | val dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EX-1 | shape | `dataset_realistic` | `runs/shape_segmenter/dataset_realistic/` | — | realistic | no | — | — | — | todo | — |
| EX-2 | relational | `pred_dataset_realistic_augB` | `runs/relational_model/predicted/pred_dataset_realistic_augB/` | — | realistic | no | yes | predicted | `dataset_realistic` | todo | — |
| EX-3 | shape | `dataset_realistic_aug` | `runs/shape_segmenter/dataset_realistic_aug/` | — | realistic | yes | — | — | — | todo | — |
| EX-4 | relational | `pred_dataset_realistic_augA_augB` | `runs/relational_model/predicted/pred_dataset_realistic_augA_augB/` | — | realistic | yes | yes | predicted | `dataset_realistic_aug` | todo | — |
| EX-5 | relational | `oracle_dataset_realistic` | `runs/relational_model/oracle/oracle_dataset_realistic/` | — | realistic | — | no | oracle | — | todo | — |

#### EX-1 — `dataset_realistic`

Stage A, stored pose only.

```bash
.venv/bin/python scripts/train_shape_segmenter.py \
    --data-root data/processed \
    --output runs/shape_segmenter/dataset_realistic \
    --profile rtx5090 \
    --no-augment
```

#### EX-2 — `pred_dataset_realistic_augB`

Stage B trained on EX-1's predicted anchors, with Stage B rotation. Needs EX-1 `best.pt`.

```bash
.venv/bin/python scripts/train_relational_model.py \
    --phase oracle \
    --anchor-source predicted \
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic/best.pt \
    --data-root data/processed \
    --output runs/relational_model/predicted/pred_dataset_realistic_augB \
    --profile rtx5090 \
    --augment
```

```bash
.venv/bin/python scripts/evaluate.py \
    --stage-b-checkpoint runs/relational_model/predicted/pred_dataset_realistic_augB/best.pt \
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic/best.pt \
    --data-root data/processed \
    --split test \
    --anchor-source predicted
```

#### EX-3 — `dataset_realistic_aug`

Stage A, rotation on.

```bash
.venv/bin/python scripts/train_shape_segmenter.py \
    --data-root data/processed \
    --output runs/shape_segmenter/dataset_realistic_aug \
    --profile rtx5090 \
    --augment
```

#### EX-4 — `pred_dataset_realistic_augA_augB`

Stage B trained on EX-3's predicted anchors, with Stage B rotation. Needs EX-3 `best.pt`.

```bash
.venv/bin/python scripts/train_relational_model.py \
    --phase oracle \
    --anchor-source predicted \
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic_aug/best.pt \
    --data-root data/processed \
    --output runs/relational_model/predicted/pred_dataset_realistic_augA_augB \
    --profile rtx5090 \
    --augment
```

```bash
.venv/bin/python scripts/evaluate.py \
    --stage-b-checkpoint runs/relational_model/predicted/pred_dataset_realistic_augA_augB/best.pt \
    --stage-a-checkpoint runs/shape_segmenter/dataset_realistic_aug/best.pt \
    --data-root data/processed \
    --split test \
    --anchor-source predicted
```

#### EX-5 — `oracle_dataset_realistic`

Stage B on ground-truth anchors, stored pose only. Independent of Stage A.

```bash
.venv/bin/python scripts/train_relational_model.py \
    --phase oracle \
    --anchor-source oracle \
    --data-root data/processed \
    --output runs/relational_model/oracle/oracle_dataset_realistic \
    --profile rtx5090 \
    --no-augment
```

```bash
.venv/bin/python scripts/evaluate.py \
    --stage-b-checkpoint runs/relational_model/oracle/oracle_dataset_realistic/best.pt \
    --data-root data/processed \
    --split test \
    --anchor-source oracle
```

Swap `--profile rtx5090` for `--profile laptop_mps` on Apple Silicon. That is the only hardware flag; batch size, precision and device then come from `configs/train.yaml` `hardware_profiles`.

### mri-like

Not wired into the runner yet. Same shape as the realistic arm once that corpus exists.

| id | model | run | path | commit | dataset | aug A | aug B | anchors | stage A | status | val dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EX-6 | shape | `dataset_mri_like` | `runs/shape_segmenter/dataset_mri_like/` | — | mri-like | no | — | — | — | todo | — |
| EX-7 | relational | `pred_dataset_mri_like_augB` | `runs/relational_model/predicted/pred_dataset_mri_like_augB/` | — | mri-like | no | yes | predicted | `dataset_mri_like` | todo | — |
| EX-8 | shape | `dataset_mri_like_aug` | `runs/shape_segmenter/dataset_mri_like_aug/` | — | mri-like | yes | — | — | — | todo | — |
| EX-9 | relational | `pred_dataset_mri_like_augA_augB` | `runs/relational_model/predicted/pred_dataset_mri_like_augA_augB/` | — | mri-like | yes | yes | predicted | `dataset_mri_like_aug` | todo | — |
| EX-10 | relational | `oracle_dataset_mri_like` | `runs/relational_model/oracle/oracle_dataset_mri_like/` | — | mri-like | — | no | oracle | — | todo | — |

## Notes

Use this section only when a row needs context (config deltas, why it failed, what to try next).

### EX-NNN — `run_name`

- what changed:
- result:
- next:
