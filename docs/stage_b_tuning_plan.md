# Stage B (Phase 3) tuning plan

Written against run `runs/stage_b_oracle` (wandb `nlyw22l1`), measured at epoch
30 of 100, and against the checkpoint `runs/stage_b_oracle/best.pt` evaluated on
the validation split. Every number below is measured, not estimated; the command
that produced each one is named so it can be re-run after any change.

The plan is ordered. Step 1 is a gate: if it fails, steps 2-4 are wasted effort
and the problem is elsewhere.

---

## 0. Diagnosis: the model is underfitting, not overfitting

This is the single fact that determines everything else.


| epoch | train_dice | val_dice | gap (val - train) |
| ----- | ---------- | -------- | ----------------- |
| 10    | 0.1688     | 0.1649   | -0.0039           |
| 20    | 0.1974     | 0.2052   | +0.0078           |
| 30    | 0.2189     | 0.2196   | +0.0007           |


Validation *equals* training across 30 epochs, and validation is the **held-out
target classes** (cuboid, ellipsoid) - the hardest split in the project. The
generalization gap is zero, oscillating around it with no trend.

Three consequences follow immediately, and they overturn some earlier reasoning:

- **More scenes will not help.** Adding data addresses a train/val gap. There
isn't one. A model that cannot fit 2800 examples will not be rescued by 5600.
- **More augmentation will not help.** `rotation_90` is a *regularizer*: it makes
the effective input distribution ~22x larger and harder. Regularizing an
underfitting model makes it strictly worse. This also rules out repeated
augmentation (multiple poses per sample per batch) for now - it is the right
tool for an overfitting model, and we do not have one.
- **The constraint is optimization and loss shaping**, which is what the rest of
this document is about.



### Convergence rate

Over the last 10 logged epochs the slope is **+0.00175 train_dice/epoch**. Linear
extrapolation to epoch 100 gives **~0.34** - and that is optimistic, because the
cosine schedule decays the learning rate toward zero over exactly that interval
(see §2.2). The current configuration does not reach a good Dice by running it
out.

### What the model is actually doing

Measured on `best.pt`, validation split, n=24, threshold 0.5:


| measurement                     | value        | reading                                            |
| ------------------------------- | ------------ | -------------------------------------------------- |
| **centroid error (soft)**       | **7.72 vox** | median 7.46                                        |
| centroid error (hard, tau=0.5)  | 7.72 vox     | identical -> prediction is unimodal, not scattered |
| baseline: anchor-union centroid | 18.43 vox    | the strongest trivial predictor                    |
| baseline: volume centre         | 26.59 vox    | the naive predictor                                |
| precision                       | 0.165        | 83.5% of predicted voxels are wrong                |
| recall                          | 0.294        | 70.6% of the target is missed                      |
| predicted volume / GT volume    | **2.05x**    | systematic over-prediction                         |
| max sigmoid                     | 0.996        | confident; not a dead or saturated head            |
| empty predictions               | 0 / 24       | never collapses to background                      |


Reproduce with the diagnostic in §4.6.

**Interpretation.** Localization is being learned - 7.72 vox against 18.43 for
the best trivial baseline is a real signal, not noise. But the prediction is a
**diffuse blob at twice the target's volume, displaced by ~1.6 object radii**
(mean GT object radius ~4.8 vox, from 474 foreground voxels). Dice follows
arithmetically: `2 x 0.294 / (2.05 + 1) = 0.19`, against the 0.22 logged. The
failure is **extent and precision of placement**, not "looking in the wrong
place".

This distinction is invisible in Dice alone, and it is the entire argument for
§4.

### Error budget

From `precision`, `recall` and the volume ratio, with `G` = GT foreground voxels:

```
TP  = 0.294 G
FP  = 2.05 G - 0.294 G = 1.756 G
FN  = 1.000 G - 0.294 G = 0.706 G
```

**False positives outnumber false negatives 2.5 to 1.** Remember this in §3 - it
inverts the textbook remedy for class imbalance.

### Ceiling check

Even with a *perfect* centroid, a concentric prediction at 2.05x volume caps Dice
at `2G / (2.05G + G) = 0.66`. So sharpening the extent is worth more than
perfecting the localization. Budget effort accordingly: §2 and §3 before §4.

---



## 1. Gate: run Phase 2 (overfit) first - it has never been run

`runs/` contains only `stage_b_oracle`. The project's own design gates Phase 3
behind Phase 2, and that gate was skipped.

```bash
.venv/bin/python scripts/train_relational_model.py --phase overfit --profile rtx5090
```

`configs/train.yaml: stage_b_overfit` sets `steps: 2000` and
`target_train_dice: 0.95`, and the script exits non-zero if the target is not
reached. At the measured ~0.42 s/step this is **~14 minutes**.

**Why this first.** It answers the one question that decides whether anything
else is worth doing: *can this architecture, loss and decoder produce a sharp,
correctly-placed mask at all, given unlimited capacity to memorize one scene?*

- **Reaches 0.95** -> the architecture is sound. The Phase 3 shortfall is
optimization and loss shaping. Proceed to §2.
- **Plateaus well below 0.95** -> no learning-rate or loss-weight change will fix
Phase 3. The limit is capacity, decoder resolution, or a wiring fault, and that
is where the effort belongs. Note the plateau value: a Phase 2 ceiling near
0.66 would point straight at the same 2x over-prediction seen in Phase 3,
implicating the loss rather than the architecture.

Run it with `--no-augment` (the default for `--phase overfit`) so the gate
measures the model, not the augmentation.

### 1.1 RESULT: PASS (run `yg3e7fkl`)

```
PHASE 2 PASS: train dice 0.9543 (target 0.95) after 161 steps on 7 examples
```

| measurement            | value                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| steps to target        | **161 of 2000** (8% of budget)                                                                     |
| wall clock             | **36.2 s**                                                                                         |
| final train / val dice | 0.9543                                                                                             |
| val IoU                | 0.9140                                                                                             |
| **val Hausdorff**      | **1.25 vox**                                                                                       |
| per-class dice         | sphere 0.989, cone 0.979, pyramid 0.973, capsule 0.959, cylinder 0.957, cube 0.914, **torus 0.910** |
| learning rate used     | **1e-3, constant**                                                                                 |

**(a) Architecture and capacity are ruled out.** Dice 0.954 at a Hausdorff
distance of **1.25 voxels** - boundaries essentially exact, including the torus,
the only shape with a hole. The 2.05x over-prediction and 7.72 vox centroid error
measured in Phase 3 are therefore *not* architectural limits. **Delete the
"bigger model" option**; spare VRAM should not be spent on capacity.

**(b) The §3.1 BCE dilution is confirmed at the opposite extreme.** At
near-perfect fit the components are `dice: 0.1458`, `bce: 0.00123` - BCE is
**0.8%** of the objective, against 2.1% at Phase 3's poor fit. Across the entire
range from useless to near-perfect, plain BCE never exceeds ~2% of the loss.
Structural dilution, not a tuning subtlety.

**What this does NOT prove.** One scene, seven examples: the ten anchor objects
sit in fixed positions and the model need only map 7 prompts to 7 blobs, which is
achievable by memorization rather than by computing the relation. That is exactly
the gate's stated remit - "a Phase 2 failure should mean *Stage B is wired
wrong*". **Relational generalization across layouts remains untested; Phase 3 is
still the real measurement.**

---



## 2. Learning rate and schedule

### 2.1 Raise the base learning rate: `3e-4` -> `1e-3`

**Evidence.** The train-loss curve is smooth, monotone, and slow across all 30
epochs, with no oscillation, no spike, and no divergence. That is the signature
of an under-driven optimizer, not one at its stability limit. There is headroom
being left unused.

**Corroboration.** `stage_a` already runs AdamW at `1e-3` on a comparable model
(8.02M params vs Stage B's 13.69M) on the same data and hardware, stably.

```yaml
# configs/train.yaml
stage_b_oracle:
  optimizer: {name: adamw, lr: 0.001, weight_decay: 0.00001}   # was lr: 0.0003
  scheduler: {name: cosine, warmup_epochs: 8}                  # was warmup_epochs: 5
```

Warmup goes 5 -> 8 epochs because the peak is 3.3x higher; the warmup exists to
get past the early phase where the Dice denominator is dominated by the `smooth`
constant.

**Do not change** `batch_size` **at the same time.** It is 16, `gradient_accumulation_steps`
is 1, and that gives 175 steps/epoch and 17,500 updates over the run. Raising the
batch would halve the update count *and* invalidate this learning rate, confounding
the experiment. Spare VRAM is not a reason to change an optimization
hyperparameter.

### 2.1b RESULT: the increase did NOT help, and was reverted

**§2.1 was wrong.** It was tested (100-epoch run at `lr 1e-3`, `warmup 8`) and
compared against the 3e-4 baseline at matched epochs:

| epoch                   | OLD (3e-4) train | NEW (1e-3) train | NEW lr             |
| ----------------------- | ---------------- | ---------------- | ------------------ |
| 5                       | 0.1381           | 0.1375           | **7.5e-4 (2.5x)**  |
| 10                      | 0.1688           | 0.1596           | 1.0e-3             |
| 16                      | 0.1887           | **0.1693**       | 9.8e-4             |
| 20                      | 0.1974           | 0.1981           | 9.6e-4             |
| 23                      | 0.2016           | **0.1765**       | 9.4e-4             |
| 24                      | 0.2060           | 0.2045           | 9.3e-4             |
| **mean train ep 15-24** | **0.1967**       | **0.1903**       | slightly *worse*   |

At epoch 5 the new run had **2.5x the learning rate and produced -0.0006 Dice**.
Flat. And instability did appear: `train_loss` upticks over ep 1-24 went 3 -> 5,
with the largest growing from +0.0038 to **+0.0189 (5x)**. Epochs 16 and 23 are
visible partial divergence-and-recover; the 3e-4 run never did that.

**Both arguments in §2.1 were flawed:**

- *The Phase 2 -> Phase 3 transfer does not hold.* Phase 2 overfits 7 examples
  from one scene, so every step sees essentially the same near-full-batch
  gradient and gradient noise is tiny. Phase 3 has 2800 examples and far higher
  minibatch variance, and the maximum stable learning rate scales with that
  noise. The same gradient-noise argument used to advise *against* raising
  `batch_size` applies symmetrically here, and was not applied.
- *`corr(train slope, lr) = 0.813` was confounded by time.* Under cosine decay
  the learning rate falls monotonically, and the improvement slope also falls as
  any model converges, for reasons unrelated to the learning rate. Two declining
  series correlate almost automatically. That was not causal evidence.

**Action: `lr` stays at `3e-4`, `warmup_epochs` at `5`.** The axis is flat
between 3e-4 and 1e-3 with instability at the top; do not spend further runs
there, and skip the §2.3 probe.

**What it positively establishes.** Together with §1.1, two branches are closed:
capacity is not the bottleneck, and **step size is not the bottleneck**. If
tripling the step changes nothing, the problem is the gradient's *direction and
conditioning*, not its magnitude - which is exactly what a diluted objective
looks like, and materially strengthens §3.

### 2.2 Do not let cosine decay to exactly zero

`build_scheduler` in `src/training/trainer.py` implements

```python
progress = (epoch - warmup) / max(total - warmup, 1)
return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))
```

At `epoch == total` this is exactly `0.0`. There is no `eta_min` floor. With the
model still improving linearly at epoch 30, the last ~30 epochs are spent at a
learning rate too small to make progress - the schedule is throwing away a third
of the run.

Two options:

- **Config only, no code change:** `scheduler: {name: constant, warmup_epochs: 8}`.
`build_scheduler` returns `1.0` after warmup for any non-cosine name. Crude but
it costs nothing and removes the confound while you tune.
- **A floor (3-line change), preferred once tuning settles:** clamp the cosine to
a fraction of the peak, e.g. `floor + (1 - floor) * cosine` with `floor = 0.05`.
Keeps the annealing benefit without stalling the tail.



### 2.3 Cheapest possible calibration: a 5-epoch learning-rate probe

At 74.2 s/epoch, five epochs is **~6 minutes**. Three arms is ~20 minutes total,
and it replaces guesswork with a measurement:

```bash
for lr in 3e-4 1e-3 3e-3; do
  .venv/bin/python scripts/train_relational_model.py --phase oracle --profile rtx5090 \
    --epochs 5 --learning-rate $lr --no-augment \
    --output runs/lrprobe_$lr
done
```

Compare `train_dice` at epoch 5. Pick the largest learning rate that is still
smooth and monotone. `--no-augment` removes a confound and makes the fitting
signal cleaner - this probe is about optimizer speed, nothing else.

### 2.4 Price the augmentation while you are here

Not to abandon it - orientation-invariance is the point of the feature - but to
know what it costs. With a zero generalization gap it is currently pure expense:

```bash
.venv/bin/python scripts/train_relational_model.py --phase oracle --profile rtx5090 \
  --no-augment --output runs/stage_b_oracle_noaug
```

If `--no-augment` reaches a materially higher Dice, that is the augmentation's
price, and the right sequence is: fix convergence first, re-enable augmentation
once the model can actually fit, and expect it to start paying off only when a
train/val gap appears.

---



## 3. The loss



### 3.1 The bug in the intent: `lambda_bce: 1.0` does not mean equal weight

At epoch 30 the logged components are `dice: 0.7872`, `bce: 0.0166`. With both
lambdas at 1.0, **BCE is 2.1% of the total loss**. Dice carries ~98% of the
gradient.

This is arithmetic, not a bug in the code. `bce_loss` averages over every
element. With ~474 foreground voxels in 262,144 (**0.18%, a 1:552 imbalance**),
the mean is dominated by easy background. The observed value reconstructs
exactly:

```
~809 confidently-wrong FP voxels x ~5.5 nats / 262,144 voxels = 0.017   (logged: 0.0166)
```

So the configured objective `1.0 * Dice + 1.0 * BCE` is in practice
`1.0 * Dice + 0.02 * BCE`. The config says one thing and the optimizer sees
another.

### 3.2 Option A - reweight BCE (simplest, do this first)

Calibrate from the measured components. For BCE to contribute a fraction `f` of
the Dice term at the current operating point:

```
lambda_bce = f * 0.7872 / 0.0166
```


| target share of Dice term | `lambda_bce` |
| ------------------------- | ------------ |
| 20%                       | 9.5          |
| 25%                       | 11.9         |
| 50%                       | 23.7         |


```yaml
# configs/train.yaml
stage_b_oracle:
  loss:
    lambda_dice: 1.0
    lambda_bce: 10.0     # was 1.0 -> in effect ~0.02
```

**Why this is the right direction here.** Unweighted BCE penalizes each wrong
voxel equally, and from §0 the wrong voxels are overwhelmingly **false
positives** (1.756 G vs 0.706 G). Scaling BCE up therefore pushes hardest against
the exact error that dominates: the 2x over-prediction.

**What to watch.** Recall is already only 0.294. If `lambda_bce: 10` drives
recall below ~0.2 while precision climbs, you have overshot - back off to 5, or
move to Option B, which targets the same errors more surgically.

### 3.3 Option B - focal BCE (better targeted)

Focal loss (Lin et al., 2017) replaces the flat per-voxel weight with
`(1 - p_t)^gamma`, where `p_t` is the probability assigned to the *correct*
class:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

**Why it fits this failure mode precisely.** Take the two voxel populations we
actually have, at the measured confidence levels:


| voxel                                     | `p_t` | `(1-p_t)^2` | effect       |
| ----------------------------------------- | ----- | ----------- | ------------ |
| easy background (pred 0.001, true 0)      | 0.999 | 1e-6        | ~ignored     |
| confidently-wrong FP (pred 0.996, true 0) | 0.004 | 0.992       | ~full weight |


With `max sigmoid = 0.996` measured, the model *is* confidently wrong on its
false positives. Focal concentrates essentially the whole gradient there and
discards the 261,000 easy background voxels that currently dilute BCE to 2%. It
solves the dilution problem structurally rather than by multiplying through a
constant.

Reference implementation, matching the float32-upcast convention already used in
`src/training/losses.py`:

```python
def focal_bce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
) -> Tensor:
    """Focal BCE. `alpha` weights the POSITIVE class; None disables balancing."""
    logits = _as_loss_dtype(logits)
    targets = targets.to(logits.dtype)
    # Numerically stable: never exponentiate a large positive logit.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()
```

**Calibration warning.** Focal's magnitude is far *smaller* than plain BCE
(everything easy is down-weighted toward zero), so it needs its own lambda -
reusing `lambda_bce: 10` would silently re-create the 2% problem. Log the
component for one epoch and apply the same `f * dice / component` arithmetic as
in §3.2. Expect a considerably larger lambda than for plain BCE.

`alpha`**.** `alpha` weights the positive class, `1 - alpha` the negative. The
RetinaNet default `0.25` therefore emphasizes *negatives* - which matches the
FP-dominated error budget here. If recall collapses, raise `alpha` toward 0.5.

### 3.3b IMPLEMENTED - how to switch it on

Focal BCE is merged. `src/training/losses.py` gains `focal_bce_loss` and
`cross_entropy_term`; `segmentation_loss` and `deep_supervision_loss` take
`bce_variant` / `focal_gamma` / `focal_alpha`; `TrainingSettings` carries them
and validates them; and `configs/train.yaml` exposes them under
`stage_b_oracle.loss`. The default is `plain`, so every previous run is
reproduced bit for bit - a test pins that.

Config:

```yaml
stage_b_oracle:
  loss:
    lambda_dice: 1.0
    lambda_bce: 24.0      # recalibrated for focal - see below
    bce: focal            # "plain" (default) or "focal"
    focal_gamma: 2.0
    focal_alpha: 0.25
```

Or without touching the config, which is preferable when screening arms:

```bash
.venv/bin/python scripts/train_relational_model.py --phase oracle --profile rtx5090 \
  --bce focal --lambda-bce 24 --output runs/b_focal
```

The run header now prints the objective, e.g.
`loss        : 1.0 dice + 24.0 focal bce (gamma 2.0, alpha 0.25)`, so the
variant that produced a checkpoint is visible without digging into the saved
config. `train_version` is bumped to `1.3.0`.

**Measured calibration.** §3.3 warned focal would need its own lambda and
estimated the ratio from random logits. Measured instead on the validation split
with a real checkpoint:

| quantity | value |
|---|---:|
| dice component | 0.8497 |
| plain BCE | 0.019828 |
| focal BCE (gamma 2, alpha 0.25) | 0.008860 |
| **focal / plain** | **0.447** |

| target share of the dice term | `lambda_bce` (focal) | `lambda_bce` (plain) |
|---|---:|---:|
| 20% | 19 | 9 |
| 25% | **24** | 11 |
| 50% | 48 | 21 |

The plain column reproduces §3.2's predicted table (11.9 there, 11 measured), so
the two calibrations are consistent. **Use `lambda_bce ~ 24` when switching to
focal**, not the 10 that suits plain BCE - carrying 10 across would put focal at
~10% of the objective and waste the run.

### 3.4 Option C - `pos_weight`, and why it is the wrong direction *here*

`BCEWithLogitsLoss(pos_weight=w)` multiplies only the **positive** term:

```python
F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(w))
```

The reflex for a 1:552 imbalance is `pos_weight = neg/pos = 552` (or
`sqrt(552) = 23.5` as the common compromise). **Do not start here.** The three
knobs do different things:


| knob          | acts on                                 | pushes toward                                           |
| ------------- | --------------------------------------- | ------------------------------------------------------- |
| `lambda_bce`  | FP and FN equally                       | suppressing whichever dominates by count - here, **FP** |
| `pos_weight`  | FN only                                 | **recall**, i.e. *more* predicted volume                |
| focal `gamma` | confidently-wrong voxels of either sign | the actual mistakes                                     |


`pos_weight` would push the model to predict *more* foreground. It is already
predicting **2.05x too much**. The textbook imbalance remedy is aimed at a model
that collapses to background - which is the opposite of the measured failure, and
we only know that because precision and recall were measured separately.

**When to revisit.** If §3.2 or §3.3 overshoots and the model swings to
under-prediction (recall falling, volume ratio dropping below ~1), `pos_weight`
in the range 5-25 is then the correct corrective. Log the volume ratio every
epoch so the swing is visible.

### 3.5 Recommended loss sequence

1. `lambda_bce: 10.0`, everything else unchanged. One run, compare at epoch 30
  against the current run's 0.219 / 0.220.
2. If it helps but plateaus, swap plain BCE for focal (`gamma=2`, `alpha=0.25`),
  recalibrate its lambda from the logged component.
3. Only if the model swings to under-prediction, add `pos_weight`.

Change one thing per run. At ~2 hours for a full 100-epoch run and ~6 minutes for
a 5-epoch probe, most of these can be screened cheaply before committing.

---



## 4. The centroid metric



### 4.1 What it measures that Dice cannot

Dice conflates two independent failures: *looking in the wrong place* and
*looking in the right place with the wrong extent*. A Dice of 0.22 is consistent
with either. The measurement in §0 resolves it in one number - 7.72 vox against
an 18.43 baseline - and the conclusion (localization works, extent does not)
changes which fix is worth doing.

Without it you would be tuning blind, and the obvious guess - "0.22 means it
hasn't learned the relation" - is wrong.

This matters most for **this** project specifically, because the target shape is
never an input. Stage B's entire job is *find the right region from three anchors
and three directions*. Centroid error measures that job directly; Dice measures
it only after convolving it with the segmentation quality.

### 4.2 Soft vs hard centroid

Use the **soft (probability-weighted) centroid** as the primary:

```
c_pred = sum_v p_v * x_v / (sum_v p_v + eps)
c_gt   = sum_v g_v * x_v / sum_v g_v
```

where `x_v` is the voxel coordinate `(z, y, x)` scaled by `spacing`.

- always defined - no empty-prediction special case (a thresholded centroid is
undefined when the mask is empty, which happens for most of early training);
- differentiable, so the identical quantity can become a loss term (§5) with no
second implementation;
- uses the full confidence map rather than discarding it at a threshold.

Report the thresholded one alongside. On the current checkpoint the two agree to
0.01 vox, which is itself informative: it says the prediction is a single
coherent blob rather than scattered mass. A large soft/hard divergence would
indicate multi-modal or diffuse predictions - a genuinely useful alarm.

### 4.3 Always log the baselines

**This is the most important implementation note.** "7.72 voxels" alone is
uninterpretable. Two baselines make it meaningful, and both are free to compute:


| baseline                  | value         | meaning                                                                          |
| ------------------------- | ------------- | -------------------------------------------------------------------------------- |
| volume centre             | 26.59 vox     | what a model that learned nothing scores                                         |
| **anchor-union centroid** | **18.43 vox** | what a model that ignores the *directions* and just points at the anchors scores |


The anchor-union baseline is the important one: beating it is the minimum
evidence that the **prompt** is being used rather than the anchor geometry alone.
A model at 18 vox would be a relational failure no matter what its Dice said.

Also report the error **in object radii** (`7.72 / 4.8 = 1.6`). That is
scale-free and comparable across shape classes whose sizes differ severalfold,
where raw voxels are not.

### 4.4 Where it goes in the code

The repository already has exactly the right seam: `hausdorff` is an optional
per-sample scalar that flows through `MetricAccumulator` and stratifies
automatically. Mirror it.

1. `src/evaluation/metrics.py` - add alongside `batch_hausdorff`:

```python
def centroid_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    soft: bool = True,
) -> list[float]:
    """Euclidean distance between predicted and GT centroids, in world units.

    `soft=True` weights by probability (always defined); `soft=False` thresholds
    first and returns nan for an empty prediction, which the accumulator drops
    the same way it drops an undefined Hausdorff distance.
    """
```

1. `MetricAccumulator` - add a `centroid: list[float]` field, extend `add()`
  with a `centroid: float | None = None` argument, and emit `centroid` plus
   `centroid_undefined` in `summary()`, exactly as `hausdorff` already does.
2. `StratifiedMetrics.update` - compute the per-sample list next to
  `distances` and pass it through the existing `self._add(...)` calls. Because
   `_add` already fans out to `target_shape`, `anchor_shape`, `direction` and
   `clause_slot`, **the stratification comes for free**.
3. `src/training/trainer.py` - no call-site change needed. `update()` already
  receives `prediction`, `target` and `spacing`.
4. `format_stratified_table` - add a column so it appears in the per-epoch
  report.

The baselines of §4.3 need the anchor union, which is not currently passed to
`update()`. Either add an optional `anchor_union: Tensor | None = None`
parameter, or - simpler, and enough - compute both baselines once in the
standalone evaluation script rather than every epoch, since neither changes
during training.

### 4.5 What it would change in practice

Concrete decision table, to be read after the §2/§3 changes land:


| centroid error        | Dice       | reading                                              | what to do                                                                         |
| --------------------- | ---------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| ~18 vox (at baseline) | low        | prompt is being ignored; model points at the anchors | relational failure - architecture/fusion, not tuning                               |
| falls to ~2-3 vox     | still ~0.3 | localization solved, extent is the whole problem     | Tversky / boundary loss / decoder sharpness. **A centroid loss would add nothing** |
| stalls ~7-8 vox       | rises      | extent improving, placement stuck                    | §5 becomes worth doing                                                             |
| falls and Dice rises  | -          | both improving                                       | keep going; no new loss term needed                                                |


That third row is the case where §5 earns its place, and there is currently no
way to know which row you are in.

Stratifying by **direction token** is the highest-value cut. medial/lateral is
decided by distance to the centre plane while the other four read the sign of one
delta component - a genuinely different rule. "Does medial/lateral localize
worse?" is a real, testable hypothesis, and `val_strata.direction` already exists
to answer it.

### 4.6 Standalone diagnostic

The measurements in §0 came from a script of this shape; keep it as a reusable
probe rather than waiting on a full training run:

```python
m = load_relational_vlm('runs/stage_b_oracle/best.pt', device='cpu').eval()
ds = ExampleDataset('data/processed', 'val', limit=24)
loader = build_example_dataloader(ds, batch_size=4)
with torch.no_grad():
    for b in loader:
        p = m(**stage_b_model_inputs(b)).probabilities()[:, 0].numpy()
        # soft/hard centroid vs GT; volume-centre and anchor-union baselines;
        # precision, recall, predicted/GT volume ratio, max sigmoid
```

Run it on CPU so it does not contend with a training run for the GPU.

---



## 5. Centroid in the loss - worth doing, but auxiliary only, and not yet



### 5.1 The genuine argument for it

Soft Dice has a **vanishing gradient when prediction and ground truth do not
overlap**. The numerator `sum(p * g)` is ~0 and stays ~0 under small
perturbations, so the loss reports *that* the prediction is wrong but carries
almost no information about *which way to move it*. At a 1:552 imbalance this
regime is common early in training and for hard examples throughout.

A centroid term has no such dead zone. It produces a well-defined direction
vector at any separation:

```
L_cent = || c_pred - c_gt ||_2 / diag         diag = ||(D-1, H-1, W-1) * spacing||_2
                                                   = 63 * sqrt(3) = 109.1 vox at 64^3
```

Differentiating through the soft centroid gives, per voxel,

```
dL/dp_v  ~  (x_v - c_pred) . (c_pred - c_gt) / (sum_v p_v * diag * ||c_pred - c_gt||)
```

which raises `p_v` for voxels lying on the ground-truth side of the current
predicted centroid and lowers it for those on the far side. Dense, bounded, and
informative regardless of overlap. This is the same rationale as distance-based
segmentation losses (Kervadec et al., *Boundary loss for highly unbalanced
segmentation*), which exist precisely because regional losses degrade under
extreme imbalance.

There is measurable headroom: 7.72 vox is ~1.6 object radii, so placement is not
already solved.

### 5.2 Three ways it bites, and the mitigation for each

**It is degenerate on its own.** The centroid is invariant to both scale and
shape. A uniform prediction over the entire volume has centroid exactly at the
volume centre; a single correctly-placed voxel scores perfectly. Optimizing it
alone produces nonsense. *Mitigation: never let it be the primary term; Dice
stays dominant.*

**It is gameable by bimodality.** The expected centroid of two symmetric false
blobs sits midway between them - in empty space - scoring well while being
entirely wrong. *Mitigation: Dice must remain present to punish the false
positives, and the soft/hard centroid divergence of §4.2 detects the condition
directly.*

**It is unstable as** `sum_v p_v -> 0`**.** Early in training, total predicted mass
is tiny and the denominator blows up. *Mitigation: an* `eps` *in the denominator,
plus a mass floor - skip or down-weight the term for samples whose predicted mass
is below, say, 10 voxels - and a ramp-in over the first few epochs.*

A fourth, subtler risk: **the cheapest way to move a centroid is to add mass on
one side, not to remove wrong mass on the other.** A centroid term can therefore
*increase* the over-prediction that is already the dominant error. This is the
strongest reason to fix §3 first and revisit §5 afterwards.

### 5.3 Weight calibration - and a correction

The normalized distance is currently `7.72 / 109.1 = 0.071`, against a Dice term
of 0.787. So:

```
lambda_cent = f * 0.787 / 0.071
```


| target share of Dice term | `lambda_cent` |
| ------------------------- | ------------- |
| 10%                       | 1.1           |
| 25%                       | 2.8           |
| 50%                       | 5.5           |


`lambda_cent ~ 3` for a meaningful but clearly secondary term. Note that a
small-sounding weight like 0.1-0.3 would make this term **~1% of the loss** - i.e.
exactly the same dilution trap as `lambda_bce: 1.0` in §3.1. Normalize first,
then calibrate against the measured component; never pick the weight by
intuition.

If you normalize by object radius instead of the diagonal the raw value is ~1.6
and the equivalent weight is ~0.12. Same term, different bookkeeping - just be
explicit about which normalizer is in force.

### 5.4 Reference implementation

```python
def soft_centroid(
    probabilities: Tensor,           # [B, 1, D, H, W], already sigmoid
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    eps: float = 1e-6,
) -> Tensor:                          # [B, 3] in world units, ordered (z, y, x)
    b, _, d, h, w = probabilities.shape
    device, dtype = probabilities.device, probabilities.dtype
    sz, sy, sx = float(spacing[2]), float(spacing[1]), float(spacing[0])
    coords = torch.stack(torch.meshgrid(
        torch.arange(d, device=device, dtype=dtype) * sz,
        torch.arange(h, device=device, dtype=dtype) * sy,
        torch.arange(w, device=device, dtype=dtype) * sx,
        indexing="ij",
    )).reshape(3, -1)                                  # [3, N]
    flat = probabilities.reshape(b, -1)                # [B, N]
    mass = flat.sum(dim=1, keepdim=True)               # [B, 1]
    return (flat @ coords.T) / (mass + eps)            # [B, 3]


def centroid_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    min_mass: float = 10.0,
) -> Tensor:
    """Normalized distance between soft predicted and GT centroids."""
    logits = _as_loss_dtype(logits)
    probabilities = torch.sigmoid(logits)
    targets = targets.to(probabilities.dtype)

    c_pred = soft_centroid(probabilities, spacing=spacing)
    c_gt = soft_centroid(targets, spacing=spacing)

    _, _, d, h, w = logits.shape
    diag = math.sqrt(
        ((d - 1) * spacing[2]) ** 2
        + ((h - 1) * spacing[1]) ** 2
        + ((w - 1) * spacing[0]) ** 2
    )
    distance = torch.linalg.vector_norm(c_pred - c_gt, dim=1) / diag

    # Silence the term while there is almost no predicted mass to move.
    mass = probabilities.reshape(logits.shape[0], -1).sum(dim=1)
    return (distance * (mass > min_mass).to(distance.dtype)).mean()
```

Wire it into `segmentation_loss` behind a `lambda_centroid: float = 0.0` keyword
so it is **off by default** and every existing run stays bit-identical, and add
`"centroid"` to the returned components dict so it is logged from the first epoch
and can be calibrated per §5.3.

### 5.5 A cleaner alternative: an auxiliary centroid-regression head

Rather than bending the segmentation map's centre of mass, predict the centroid
directly: a small MLP head on the bottleneck emitting `(z, y, x)`, supervised with
smooth-L1 against the ground-truth centroid, sharing the encoder.

**Advantages.** It gives the encoder an explicit, dense "where" signal without
distorting the probability map; it cannot be gamed by bimodality; it is stable
regardless of predicted mass; and at inference it is a free, directly
interpretable localization readout that can be compared against the mask's own
centroid as a consistency check.

**Disadvantage.** It does not directly improve the mask - it shapes the
representation and hopes the decoder benefits. The in-loss version acts on the
output you actually care about.

Given the §0 finding that localization is *already* substantially learned, the
regression head is the lower-risk of the two, and the one I would reach for if
§3 fixes the extent problem but placement stays stuck around 7-8 vox.

### 5.6 Sequencing - the point to hold onto

**Add the metric (§4) before the loss (§5).** The metric costs almost nothing and
determines whether the loss term is justified at all. Concretely: if after §2 and
§3 the centroid error falls to ~2-3 vox while Dice sits near 0.3, then
localization is solved, a centroid loss would buy **nothing**, and the remaining
problem is entirely extent - pointing at Tversky, boundary loss, or decoder
resolution instead.

On current evidence I would rank §5 **below** the learning rate and the BCE
reweighting, and would not implement it until §1-§3 have been run and measured.

### 5.7 The reference implementations above were executed, not sketched

Both snippets (`focal_bce_loss` in §3.3, `soft_centroid` / `centroid_loss` in
§5.4) were run before this document was written:

- `soft_centroid` on a binary mask reproduces the project's own
`src.data.direction_rules.centroid_world` **exactly** (to 1e-4) on a non-cubic
grid with anisotropic spacing `(1, 2, 3)` - so the `(z, y, x)` array order and
the `(x, y, z)` spacing order are the right way round, which is the one place
this is easy to get wrong;
- `diag` for 64^3 at unit spacing evaluates to 109.119;
- `centroid_loss` backpropagates: gradients finite and non-zero;
- the `min_mass` guard works - a saturated-negative prediction yields loss 0.0
with finite gradients rather than a NaN from the empty denominator;
- focal BCE measured **0.32x** the magnitude of plain BCE on the same logits,
confirming §3.3's warning that it needs its own lambda rather than inheriting
`lambda_bce`;
- the two-population table in §3.3 checks out numerically: easy background gets
weight 1.0e-06, a confidently-wrong false positive gets 0.992.

They are still reference implementations, not merged code - nothing in `src/` has
been changed.

---



## 6. Summary


| #   | change                      | concrete value                            | cost            | rationale                                       |
| --- | --------------------------- | ----------------------------------------- | --------------- | ----------------------------------------------- |
| 1   | Run Phase 2 overfit         | `--phase overfit`                         | **done, 36 s**  | **PASS 0.9543 in 161/2000 steps** (§1.1); capacity ruled out |
| 2   | Learning rate               | ~~`3e-4` -> `1e-3`~~ **keep `3e-4`**      | **done, tested**| **FAILED** (§2.1b): flat, and less stable       |
| 3   | Cosine floor                | `constant`, or floor at 5% of peak        | free            | cosine hits exactly 0 while still improving     |
| 4   | BCE weight                  | `lambda_bce: 1.0` -> `10.0`               | 2 h/run         | BCE is 2.1% of the loss, not 50%                |
| 5   | Focal BCE (alt.)            | `gamma=2, alpha=0.25`, recalibrate lambda | 2 h/run         | targets the confidently-wrong FPs structurally  |
| 6   | `pos_weight`                | **not now**                               | -               | pushes toward *more* volume; already 2.05x over |
| 7   | Centroid metric             | soft + hard, with both baselines          | ~1 h to write   | separates "wrong place" from "wrong extent"     |
| 8   | Centroid loss               | `lambda_cent ~ 3` (diag-normalized)       | after 1-4       | only if placement stalls once extent is fixed   |
| 9   | More scenes                 | **no**                                    | -               | zero train/val gap                              |
| 10  | More augmentation / repeats | **no**                                    | -               | regularizing an underfitting model              |


Do not change `batch_size` (16) or `gradient_accumulation_steps` (1) while tuning
any of the above.