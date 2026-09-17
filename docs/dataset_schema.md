# Dataset schema

Schema version `1.0.0` (`src.data.schema.SCHEMA_VERSION`).

An **example** is one `(scene, target)` pair. Every scene yields ten candidate
examples, one per instance; the target-class filter in `configs/split.yaml`
decides which of them are used for training, validation and testing.

Scenes are written as NIfTI (`.nii.gz`) directories and carry a simulated
MRI-like image alongside the labels — see [Storage layout](#storage-layout) and
[`mri_appearance.md`](mri_appearance.md).

## Coordinate conventions

| Quantity | Order | Notes |
| --- | --- | --- |
| Array indexing | `(z, y, x)` | every volume has shape `(D, H, W)` |
| World coordinates | `(x, y, z)` | RAS: `x` right/lateral, `y` anterior, `z` superior |
| `spacing` | `(x, y, z)` | world units per voxel, `(1, 1, 1)` by default |
| `volume_shape` | `(D, H, W)` | `64 x 64 x 64` unless escalated during packing |

World-space centre of a `(D, H, W)` volume:

```text
c = (((W - 1) / 2) * sx, ((H - 1) / 2) * sy, ((D - 1) / 2) * sz)
```

Bounding-box extents are **inclusive**: a single occupied voxel has an extent of
one spacing unit per axis.

A voxel belongs to a shape when its centre lies inside the analytic solid, so a
solid of continuous size `s` occupies `s` or `s + 1` voxels along an axis
depending on where its centre falls. The ranges in `configs/shapes.yaml`
describe the continuous size; `scripts/run_smoke_test.py` reports the measured
voxel extents against the 8–22% design band.

## Fields

`src.data.schema.REQUIRED_EXAMPLE_FIELDS` is the machine-readable form of this
table; `assert_complete()` checks it.

| Field | Type | Meaning |
| --- | --- | --- |
| `scene_id` | `str` | scene identifier, e.g. `scene_000123` |
| `example_id` | `str` | `<scene_id>_target_<instance_id>` |
| `seed` | `int` | the scene seed that produced this example |
| `volume_shape` | `(int, int, int)` | `(D, H, W)` actually used |
| `spacing` | `(float, float, float)` | `(x, y, z)` world units per voxel |
| `scene_volume` | `uint8 (D, H, W)` | binary foreground of the whole scene |
| `image`¹ | `float32 (D, H, W)` | the simulated MRI-like volume |
| `instance_labels` | `uint8 (D, H, W)` | `0` background, `1..10` shape IDs |
| `target_instance_id` | `int` | `1..10` |
| `target_shape_name` | `str` | vocabulary name; never appears in the prompt |
| `target_mask` | `uint8 (D, H, W)` | `instance_labels == target_instance_id` |
| `anchor_instance_ids` | `(int, int, int)` | ordered by ascending centroid distance |
| `anchor_shape_names` | `(str, str, str)` | same order |
| `anchor_masks` | `uint8 (3, D, H, W)` | channel `i` is anchor `i`; never unioned |
| `anchor_union_mask` | `uint8 (D, H, W)` | ablation baseline input only |
| `relations` | `list[{direction, anchor}]` | the three ordered clauses |
| `anchor_centroids_world` | `(3, 3) float` | per anchor, `(x, y, z)` |
| `anchor_extents_world` | `(3, 3) float` | per anchor, inclusive bbox `(x, y, z)` |
| `target_centroid_world` | `(3,) float` | `(x, y, z)` |
| `prompt` | `str` | canonical rendering of `relations` |
| `structured_prompt` | `list[{direction, anchor}]` | primary model interface |
| `generator_version` | `str` | plus `direction_rule_version`, `vocabulary_version`, `schema_version` |

¹ `image` is a per-scene volume, not a field of `ExampleMetadata`: it is read
through `src.data.scene_io.load_scene`, and it is what the datasets hand a model
as `scene_volume`. The schema's own `scene_volume` stays the binary occupancy,
because that is the thing validation compares against `instance_labels`.

**Labels are not blurred.** The image has partial-volume edges, texture, a bias
field and Rician noise; `instance_labels` and every mask derived from it stay
the crisp centre-in-solid masks. That is the same relationship a manual
segmentation has with the scan it was drawn on.

## Prompt

Structured form (primary interface):

```python
[
    {"direction": "lateral",  "anchor": "cube"},
    {"direction": "superior", "anchor": "pyramid"},
    {"direction": "anterior", "anchor": "sphere"},
]
```

Canonical rendering:

```text
segment the shape that is lateral to the cube, superior to the pyramid, and anterior to the sphere.
```

`render_prompt` and `parse_prompt` are exact inverses over valid clause triples.
The parser is built from the two closed vocabularies, so a synonym, an ordinal
reference or an unknown shape name cannot be parsed. The only formatting
tolerance is that `triangular prism` is accepted and normalised to
`triangular_prism`.

Clause order **is** anchor order: ascending centroid distance from the target,
instance ID as tie-break. The same order is used by the three mask channels.

## Direction rules

Implemented in `src.data.direction_rules`, version `1.0.0`. With
`delta = centroid(target) - centroid(anchor)` in world units:

1. main axis = `argmax(|delta_x|, |delta_y|, |delta_z|)`, ties resolved `z > y > x`;
2. `z`: `+` → `superior`, `−` → `inferior`;
3. `y`: `+` → `anterior`, `−` → `posterior`;
4. `x`: compare `|target_x − c_x|` with `|anchor_x − c_x|`; farther → `lateral`,
   closer → `medial`.

Two cases are **rejected**, never guessed: coincident centroids, and an exact
tie of the two centre-plane distances when `x` wins. The generator regenerates
the scene and logs the rejection reason.

## Storage layout

```text
data/processed/
  scenes/<scene_id>/
    image.nii.gz                     # the simulated MRI-like volume
    labels.nii.gz                    # instance labels, 0 and 1..10 (uint8)
    occupancy.nii.gz                 # binary foreground (uint8)
    masks/<id>_<name>.nii.gz         # one binary mask per structure (10)
    examples/<example_id>_target.nii.gz    # that example's target mask
    examples/<example_id>_anchors.nii.gz   # its 3 anchor channels, 4D, clause order
    examples/<example_id>.json             # its prompt record
    scene.json                       # seeds, placed parameters, appearance draws
    tissue.nii.gz, bias_field.nii.gz,      # --write-appearance-diagnostics
    structure_fraction.nii.gz, body_mask.nii.gz
  manifests/<split>.jsonl            # one ExampleMetadata per retained example
  manifests/<split>_candidates.jsonl # all ten candidates (--keep-all-candidates)
  run_metadata.json                  # configs, seeds, versions, git revision,
                                     # hardware, exact counts, appearance
                                     # statistics, rejection log
data/smoke/                          # same layout, written by the smoke run
```

Everything under a scene directory except `labels.nii.gz` and `image.nii.gz` is
derivable from those two. It is written anyway: the point is that any scene, or
any single example, opens in ITK-SNAP, FSLeyes or 3D Slicer, or loads with three
lines of `nibabel`, without re-deriving anything or importing this package.
`storage.write_example_masks: false` drops the per-example volumes if the file
count becomes a nuisance.

**NIfTI orientation.** Arrays are `(z, y, x)` in memory and `(i, j, k)` in a
NIfTI file, so `src.data.nifti_io` transposes on the way out and back on the way
in. The affine is `diag(sx, sy, sz, 1)` with no translation, which makes a
voxel's NIfTI world coordinate exactly the world coordinate this codebase
computes for it — a centroid printed by a manifest can be typed straight into a
viewer. Both `sform` and `qform` are set, and `aff2axcodes` gives `(R, A, S)`.

`image.nii.gz` is stored as `uint16` with a header scale factor, which is how
scanner data is written: it halves the file and reads back in the simulated
units to within ~1e-5, three orders of magnitude below the noise.
`storage.image_dtype: float32` stores it bit-exactly instead.

`storage.scene_array_format: npz_compressed` restores the previous single-file
form (`scenes/<scene_id>.npz` holding `scene_volume`, `instance_labels` and
`image`). `src.data.scene_io.load_scene` reads either, so a corpus generated
before this change still loads — it simply has no image.

`<split>.jsonl` holds only the examples that survive the target-class filter;
`<split>_candidates.jsonl` holds all ten candidates per scene and is what the
smoke run uses to check that every shape occurs as a target and as an anchor.

The image, the occupancy and `instance_labels` are identical across the ten
examples of a scene, so they are stored once per scene. `target_mask`, `anchor_masks` and
`anchor_union_mask` are materialised from `instance_labels` by the declared
instance IDs (`ExampleArrays.from_scene`), and every consistency check is re-run
at load time: exactly ten instances, no empty object, target not in any anchor
channel, each channel equal to its declared anchor, anchor order identical to
prompt order, and prompt round-trip. A mismatch raises rather than being
repaired. Large arrays stay out of git; manifests, schemas, configs and the
small fixture batch are committed.

## Stage B input contract

`src.data.schema.stage_b_inputs()` returns the only fields the relational model
may consume: `anchor_masks`, `scene_volume`, `structured_prompt`, `prompt`,
`anchor_shape_names`, `anchor_centroids_world`, `anchor_extents_world`,
`volume_shape`, `spacing`.

`STAGE_B_FORBIDDEN_FIELDS` — `instance_labels`, `target_mask`,
`target_shape_name`, `target_instance_id`, `target_centroid_world` — must never
reach Stage B. `scene_volume` carries no instance ids and is a decoder input:
with the appearance model it is the simulated acquisition, whose intensities are
drawn independently of shape class precisely so that it cannot identify one.
`anchor_union_mask` is reachable only through
`stage_b_inputs(..., use_union_mask=True)`, which is the ablation baseline.
