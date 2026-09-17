# MRI-like appearance model

`configs/appearance.yaml` + `src/data/appearance.py`

## Why

The first milestone stored one foreground intensity on a zero background. That
volume is segmentable by a threshold, so nothing a model learned on it had to
be robust to anything an acquisition does. The project's destination is
segmenting dense subcortical structures — thalamus, caudate, putamen — in a
brain MRI, where:

- a structure differs from the tissue around it by a few percent, not by 1.0;
- neighbouring structures are close to isointense with **each other**, which is
  exactly why spatial and relational priors are needed at all;
- boundaries are smeared across a voxel by partial volume and by the point
  spread of the acquisition;
- a smooth receive-coil gain tilts the whole volume;
- everything sits under Rician noise, with a Rayleigh floor in air.

This model reproduces all five. The image is the only thing that changes: the
label volume stays the crisp centre-in-solid mask it always was, exactly as a
manual segmentation of a real scan stays crisp while the image under it does
not.

## Pipeline

Ordered the way the physics happens.

### 1. Partial volume

Each analytic solid is supersampled inside its own bounding box (4 sub-samples
per voxel edge, so 64 per voxel), giving the fraction of each voxel that lies
inside the solid. `src.data.voxelization.partial_volume`. With `subdivisions=1`
it reproduces `voxelize` exactly, which is how the two are kept consistent.

Windowing to the bounding box is what makes this cheap: an object occupies
roughly `16³` voxels, so its supersampled grid is `64³` samples rather than the
`256³` a whole-volume evaluation would need.

Where two adjacent objects share a boundary voxel the fractions are scaled back
so the compartments still partition the voxel.

### 2. Tissue map

Compartments: air, parenchyma, a scalp-like rim just inside the head boundary,
and the ten structures. Intensities are drawn per scene, and structures
displace whatever background was at their voxels:

```text
tissue = sum_k f_k * mu_k  +  (1 - sum_k f_k) * (f_air * mu_air
                                                 + f_rim * mu_rim
                                                 + f_core * mu_parenchyma)
```

The head is a superellipsoid — `|x/a|^p + |y/b|^p + |z/c|^p <= 1` with `p`
around 3 — filling most of the field of view. Structures are constrained to sit
inside it during packing (`body.constrain_placement`), because a structure
floating in the air outside the head would defeat the point. The semi-axes and
the exponent are tuned so this constraint costs nothing measurable: the scene
acceptance rate with it is the same as without it, because what rejects a scene
is the anchor-triple rule, not the head outline.

### 3. Texture

Two zero-mean Gaussian random fields — one slow (correlation ~14 voxels), one
fine (~3 voxels) — multiplied into the tissue map, plus an independent field per
structure so two adjacent structures do not vary in lockstep. Each field is
white noise filtered by a Gaussian in the Fourier domain and rescaled to unit
variance, so the configured amplitude means what it says.

### 4. Bias field

One very smooth multiplicative field, `exp(a * field)` normalised so that
`max / min - 1` equals the configured inhomogeneity and the mean gain is 1.
0.15–0.30 here brackets the 20 % and 40 % non-uniformity BrainWeb offers.

### 5. Acquisition

```text
fft -> truncate k-space -> add complex Gaussian noise -> ifft -> magnitude
```

- **Truncation** samples a fraction of the Nyquist band per axis. It blurs the
  image and rings at sharp edges: that is the Gibbs artefact, and doing it in
  k-space is why it comes out right rather than being painted on.
- **Complex Gaussian noise** in k-space is complex Gaussian noise in the image;
  the magnitude of that is Rician, and Rayleigh where there is no signal. This
  is why the air around the head is grainy rather than black, and it is the
  reason to simulate in k-space at all rather than adding Gaussian noise to the
  image.
- `noise_sigma` is the standard deviation of each **image-domain** component,
  not of the k-space samples: with `ifftn` scaling by `1/N` and a fraction `rho`
  of samples retained, k-space noise of standard deviation `sigma * sqrt(N/rho)`
  lands at `sigma` in the image.

## Intensities must not encode the shape class

`intensities.class_conditioned` is **false** by default and should stay false.

If each of the ten classes had its own intensity band, a model could name a
structure from its grey level, and the target would no longer be reachable only
through the three relations — the benchmark would be measuring the wrong thing.
It is also the faithful choice: real deep grey nuclei are close to isointense
with each other and with the white matter around them.

So every structure's mean is drawn from one shared distribution
(`parenchyma ± |N(contrast, jitter)|`), independent of class.
`tests/test_appearance.py::test_structure_intensities_are_not_conditioned_on_the_shape_class`
is the guard: over 400 draws the spread *between* class means must stay inside
sampling noise of the spread *within* a class.

Setting it to `true` gives each class a deterministic offset. That exists only
as an ablation — a way to measure how much a model would gain from an
appearance leak — never as the default corpus.

## How hard is the result

Every run reports both numbers, measured from the image rather than taken from
the configuration (`run_metadata.json` → `appearance.measured`, and per scene in
`scene.json` → `appearance.measured`):

| number | meaning |
| --- | --- |
| `contrast_to_noise` | structure-to-surroundings contrast over the noise standard deviation, recovered from the air region (Rayleigh, `sigma = median / sqrt(2 ln 2)`). The classical CNR; ~5–10 at the defaults. |
| `contrast_to_background_variation` | the same contrast over the standard deviation of the surrounding tissue, which also holds the texture and the bias field. The honest difficulty; ~1.5–2 at the defaults. |

`tests/test_appearance.py::test_no_single_threshold_recovers_the_structures`
sweeps every global threshold and requires the best achievable Dice to stay
below 0.75. If a future configuration change made the image easy again, that
test fails rather than the corpus quietly reverting to the old milestone.

## Determinism

Everything is drawn from `default_rng([seed, APPEARANCE_STREAM, stage,
substream])`, where `APPEARANCE_STREAM = 7919`. The packer uses
`default_rng([seed, stage, attempt])`. The two streams never meet, so:

- a scene's appearance does not shift when packing needs one more attempt;
- with `body.constrain_placement: false`, geometry is bit-identical to an
  appearance-free corpus generated from the same seeds — which is what makes an
  A/B comparison between the two corpora meaningful.

`tests/test_appearance.py::test_the_appearance_does_not_change_the_geometry`
checks the second point directly.

## Turning it off

`enabled: false` in `configs/appearance.yaml` restores the previous milestone:
binary volumes, no `scene_volume.nii.gz`, and the datasets fall back to the occupancy
automatically. On a corpus that *does* have an image, `image_source="occupancy"`
on `SceneDataset` / `ExampleDataset` gives the same binary input without
regenerating anything — that is the baseline to compare an appearance-trained
model against.

## Looking at it

```bash
.venv/bin/python scripts/preview_scene.py --root data/smoke --limit 4
.venv/bin/python scripts/preview_scene.py --root data/smoke \
    --scene scene_009000000 --example scene_009000000_target_01
```

Writes one PNG montage per scene (no matplotlib needed): the three orthogonal
planes of the image, the same planes with the instance labels over them, and —
with `--example` — the target and the three ordered anchor channels.

## Not simulated

Deliberately out of scope for this milestone, and each would be a separate,
testable addition:

- motion ghosting and other phase-encode artefacts;
- anisotropic voxels and slice-direction blur (the spacing machinery supports
  them, but no configuration uses them yet);
- multiple contrasts per scene (T1 + T2 + FLAIR as channels);
- anatomically plausible packing — real subcortical structures are clustered
  near the midline and roughly symmetric, while these are scattered through the
  head. That is a geometry change, not an appearance one.
