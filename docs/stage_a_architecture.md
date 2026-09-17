# Stage A architecture

Phase 1 of the training procedure: a full-volume **promptable** shape segmenter.
It takes the scene image `scene_volume` — the simulated MRI-like volume, see
[`mri_appearance.md`](mri_appearance.md) — and a set of shape names, and returns one mask
logit volume per requested name. It is the synthetic analogue of the MRI anatomy
segmenter that will later supply anchor masks, and it is what Phase 4 calls to
extract the three anchors a prompt names.

Source: `src/models/shape_segmenter.py`, `src/models/blocks.py`,
`src/models/prompt_encoder.py`. Configuration: `configs/model.yaml` (`stage_a`).

## Reference

The design is documented in `docs/flowchart/phase1_encoder_decoder.drawio`
(Architecture + Training-Data-Eval tabs): a residual 3D U-Net whose name
queries attend over the bottleneck once and are then reused as a mask head at
three decoder scales, with deep supervision. It is adapted from VoxDense
(128³ patches, frozen PubMedBERT, zero-init bias) with the three changes below.

```text
scene_volume [B, 1, 64, 64, 64]
  STEM        Conv-IN-ReLU + ResBlock   1 -> 32, s=1      skip1   32 x 64^3
  Stage 1     Conv-IN-ReLU + ResBlock  32 -> 64, s=2      skip2   64 x 32^3
  Stage 2     Conv-IN-ReLU + ResBlock  64 ->128, s=2      skip3  128 x 16^3
  Bottleneck  Conv-IN-ReLU + ResBlock 128 ->256, s=2             256 x  8^3

prompt_ids [B, N_T]
  ShapeNamePromptEncoder      Embedding(10, 256) -> Linear -> Q [B, N_T, 256]
  PromptDecoder               Q (no PE)  ·  K = bottleneck + PE3D  ·  V = bottleneck
                              MHA(4 heads, d_h = 64), then LayerNorm(attn + Q)
                              -> aligned_queries [B, N_T, 256]

  Decode 1   8^3 -> 16^3, concat skip3, 256+128 -> 128   pred[0]  DS 0.1
  Decode 2  16^3 -> 32^3, concat skip2, 128+ 64 ->  64   pred[1]  DS 0.3
  Decode 3  32^3 -> 64^3, concat skip1,  64+ 32 ->  32   pred[2]  DS 0.6  <- inference
```

At each decoder scale a `StageVLFusionBlock` computes

```text
q~     = LayerNorm(Linear(C_vis -> C_vis)(ReLU(Linear(256 -> C_vis)(q))))
logits = (q~ @ F_vis) / sqrt(C_vis) + bias_head(q~)
```

Visual features are read, never modulated. Queries never interact. Two
consequences the tests pin down: the logits of a shape do not depend on which
other shapes were requested, and permuting the requested prompts permutes the
output channels and nothing else. That is exactly the property anchor extraction
by name needs.

## Three adaptations

### 1. 64³ input, 8³ bottleneck

The reference uses 128³ patches and a 16³ bottleneck. This project's volumes are
64³, so the same three stride-2 stages give `64 -> 32 -> 16 -> 8` and a **512-token**
bottleneck — precisely the budget CLAUDE.md sets for global cross-attention (and
well clear of the 262,144 tokens that full-resolution attention would create).
Decoder scales become 16³ / 32³ / 64³; the positional encoding is built for 8³
and refuses a mismatched runtime grid rather than silently interpolating.

### 2. Closed-vocabulary prompts instead of frozen PubMedBERT

The reference queries a frozen sentence encoder with anatomical names. This
project's vocabulary is closed and fixed at ten names (CLAUDE.md, "Fixed shape
vocabulary"), and no synonyms may appear anywhere, so a learned
`Embedding(10, text_dim)` is the exact equivalent — there is no open-vocabulary
text to generalise over. `ShapeNamePromptEncoder` owns that table; Phase 3 adds
the direction, slot and pair embeddings on top of it to build the Stage B
relation tokens.

This keeps the interface identical to the reference: a prompt is `[B, N_T]`
indices, the model returns `[B, N_T, D, H, W]` logits, and any subset of names
in any order is a valid request.

### 3. Foreground-prior bias initialisation

The reference zero-initialises the per-query logit bias, which starts every voxel
at p = 0.5. One shape occupies about **0.16%** of a 64³ scene here, so a
zero-init head spends its early training pushing 262,000 background logits down
before the Dice term produces usable gradient. Measured on the smoke corpus,
both heads trained identically for 150 steps:

| bias init | train Dice @ step 150 | soft Dice loss @ 64³ |
| --- | --- | --- |
| `zeros` (reference) | 0.179 | 0.978 |
| `prior` (this project) | 0.661 | 0.403 |

`bias_init: prior` sets the bias to `log(p / (1 - p))` for the configured
expected foreground fraction — the standard focal-loss prior — and leaves it
learnable; the bias *weight* is still zero-initialised, so nothing else about the
head changes. Set `bias_init: zeros` in `configs/model.yaml` to reproduce the
reference exactly.

## Objective

```text
L = sum_s w_s * [ lambda_dice * DiceLoss(pred_s, target_s)
                + lambda_bce  * BCEWithLogits(pred_s, target_s) ]
    for s in {16^3, 32^3, 64^3},  w = (0.1, 0.3, 0.6)
```

CLAUDE.md asks for "multiclass cross-entropy plus Dice loss, **or a documented
equivalent**". The equivalent used here is per-prompt sigmoid Dice + BCE, because
the head is promptable: it emits one independent mask logit per requested name
rather than a softmax over a fixed class axis, and that independence is what
makes anchor extraction by name possible. The shapes never overlap, so both
formulations supervise the same partition.

Deep-supervision targets are downsampled with **max pooling**, not nearest or
average sampling: at 1/4 resolution a torus is about one voxel thick and the
alternatives can delete it, which would supervise the coarse head towards an
empty mask for a structure that is really there.

## Metrics

Per-class Dice and IoU on the thresholded full-resolution map, plus the macro
average over classes, which is the checkpoint-selection metric. Degenerate cases
follow the usual convention: both masks empty scores 1.0, exactly one empty
scores 0.0. Hausdorff distance is required for the Stage B target-mask report and
lands with Phase 3.

## Precision and batch size

Measured on the 8.02 M-parameter model at 64³, 17 GB Apple Silicon, every arm
interleaved in one process, median of 6 steps:

| batch | fp32 ms/sample | bf16 ms/sample | bf16 speedup | driver memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 666 | 576 | 1.15× | 2.28 GB |
| 2 | 668 | 539 | 1.24× | 3.39 GB |
| 4 | 851 | 573 | 1.48× | 5.60 GB |
| 8 | 1475 | 600 | 2.46× | 10.05 GB |

**Enable autocast.** bf16 wins at every batch size, and wins by more as the
batch grows — fp32 hits memory pressure sooner, so by batch 8 it is 2.5× slower.

**A larger batch does not buy throughput.** bf16 per-sample time is flat from
batch 1 to 8 (539–600 ms). Batch size here buys a larger *effective* batch — less
gradient noise per step — and costs memory. Choose it for optimisation reasons,
then scale the learning rate with it.

**Autocast does not save memory on MPS.** The cast copies live alongside the
fp32 masters, so a given batch costs *more* driver memory in bf16 than in fp32.
On CUDA the opposite holds; this is a property of the MPS backend. Memory is
linear in batch size, and batch 8 at ~10 GB of 17 GB is close enough to the edge
that one fp32 run degraded to 49 s/step.

A caution on all of the above: timings on a shared laptop move by ~20% between
runs, and measuring configurations in separate processes gives misleading
results — an earlier cross-process run of this same table showed a batch-size
throughput trend that vanished once the arms were interleaved. Only the ordering
(bf16 > fp32, memory linear in batch) survived every repetition.

### Data pipeline

Not a bottleneck: ~1 ms per sample to build an item from the cached volumes,
against ~550 ms of compute. `num_workers` therefore does not affect throughput;
it costs a one-off ~0.6 s of process spawn and one scene cache per worker.

The dataset caches only the two compact `uint8` volumes as stored on disk
(~0.52 MB per scene) and derives the float tensors and the ten per-shape masks
per item. Caching the derived tensors instead would cost ~13.6 MB per scene —
5.4 GB for a 400-scene split, and again for every persistent worker.

### bf16 rather than fp16

Both are numerically fine in the *forward* pass: against an fp32 reference, one
forward at batch 2 gives Δloss = +2.5e-5 for both, and fp16 is actually the more
accurate of the two (max |Δlogit| 0.0099 vs bf16's 0.0837 — fp16 has 10 mantissa
bits, bf16 has 7).

The difference is in the *backward* pass. One backward from identical
initialisation:

| precision | ‖g‖₂ | median \|g\| | gradient elements exactly zero |
| --- | ---: | ---: | ---: |
| fp32 | 1.532e-01 | 5.60e-06 | 0.5% |
| bf16 | 1.532e-01 | 5.60e-06 | 0.5% |
| fp16 | 2.252e-01 | 7.21e-06 | **3.2%** |

fp16 underflows 254,000 gradient elements, because a loss dominated by 262,144
background voxels produces gradients below fp16's denormal threshold. Left
unscaled that visibly corrupts training (the loss after five steps drifts
further from fp32 the larger the batch). bf16 shares fp32's exponent range and
matches its gradient norm to three digits.

`--precision fp16` is still supported and the trainer enables a
`torch.amp.GradScaler` for it automatically (`TrainingSettings.needs_grad_scaler`),
which is what makes fp16 usable on hardware without bf16. bf16 is the default
because it needs no scaler at all.

Losses are always computed in float32 regardless of the autocast dtype: the Dice
denominator sums one sigmoid value per voxel, and that reduction is exactly
where half precision loses range.

## Profiles

| profile | params | encoder channels | embed dim |
| --- | --- | --- | --- |
| `default` | 8.02 M | 32, 64, 128, 256 | 256 |
| `smoke` | 0.50 M | 8, 16, 32, 64 | 64 |

The hardware profile (`laptop_mps`, `rtx5090`) sets device, batch size,
precision and accumulation; the model profile sets width. They are independent —
the laptop can train the full model, just slowly.
