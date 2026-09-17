# Dataset schema

Schema version `2.0.0` (`src.data.schema.SCHEMA_VERSION`).

An **example** is one `(scene, target)` pair. Every scene yields ten candidate
examples, one per instance; the target-class filter in `configs/split.yaml`
decides which of them are used for training, validation and testing.

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
| `scene_volume` | `float32 (D, H, W)` | MRI-like intensities of the whole scene |
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
  scenes/<scene_id>/scene_volume.nii.gz
  scenes/<scene_id>/instance_labels.nii.gz
  examples/<example_id>/target_mask.nii.gz
  examples/<example_id>/anchor_{slot}_{shape}.nii.gz
  examples/<example_id>/anchor_union.nii.gz
  manifests/<split>.jsonl            # one ExampleMetadata per retained example
  manifests/<split>_candidates.jsonl # all ten candidates (--keep-all-candidates)
  run_metadata.json                  # configs, seeds, versions, git revision,
                                     # hardware, exact counts, rejection log
data/smoke/                          # same layout, written by the smoke run
```

`<split>.jsonl` holds only the examples that survive the target-class filter;
`<split>_candidates.jsonl` holds all ten candidates per scene and is what the
smoke run uses to check that every shape occurs as a target and as an anchor.

`scene_volume` and `instance_labels` are identical across the ten examples of a
scene, so they are stored once per scene as RAS NIfTI. `target_mask`,
`anchor_masks` and `anchor_union_mask` are materialised from `instance_labels`
by the declared instance IDs (`ExampleArrays.from_scene`); inspection copies of
those masks are also written under `examples/<example_id>/`. Training loaders
do not read the example NIfTIs. Every consistency check is re-run at load time:
exactly ten instances, no empty object, target not in any anchor channel, each
channel equal to its declared anchor, structure mean intensity above background,
anchor order identical to prompt order, and prompt round-trip. A mismatch
raises rather than being repaired. Large arrays stay out of git; manifests,
schemas, configs and the small fixture batch are committed.

NIfTI arrays are stored `(x, y, z)` RAS with affine `diag(sx, sy, sz, 1)`.
In-memory arrays remain `(z, y, x)`. `src.data.nifti_io` is the only allowed
transpose.

`scene_volume` is a float image. Binary occupancy for Stage B is
`(instance_labels != 0)` at load time.

## Stage B input contract

`src.data.schema.stage_b_inputs()` returns the only fields the relational model
may consume: `anchor_masks`, `scene_volume`, `structured_prompt`, `prompt`,
`anchor_shape_names`, `anchor_centroids_world`, `anchor_extents_world`,
`volume_shape`, `spacing`.

`STAGE_B_FORBIDDEN_FIELDS` — `instance_labels`, `target_mask`,
`target_shape_name`, `target_instance_id`, `target_centroid_world` — must never
reach Stage B. Binary occupancy derived from `instance_labels` (no instance
ids) is the decoder WHAT stream; `stage_b_model_inputs` passes it as
`scene_volume`. The intensity image is a Stage A input. `anchor_union_mask` is
reachable only through `stage_b_inputs(..., use_union_mask=True)`, which is the
ablation baseline.
