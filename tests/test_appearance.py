"""The MRI-like appearance model.

Four things have to hold, and the rest of the file is the detail of each:

* the image is an *acquisition* - partial volume, texture, bias, Rician noise -
  and not a relabelled binary mask;
* it carries no information about which shape class a structure is, or the
  relational task would have a shortcut;
* it does not touch the geometry, so a corpus with and without it differs only
  in appearance and results stay comparable;
* it is reproducible from the scene seed alone.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.config import load_config
from src.data.appearance import (
    APPEARANCE_STREAM,
    AppearanceError,
    AppearanceSettings,
    appearance_rng,
    bias_field,
    draw_body,
    kspace_window,
    measure_image,
    normalize_image,
    simulate_acquisition,
    simulate_scene_appearance,
    smooth_random_field,
    structure_intensities,
)
from src.data.primitives import SHAPE_VOCABULARY
from src.data.scene_generator import GeneratorSettings, generate_scene
from src.data.voxelization import (
    analytic_volume_world,
    half_extent_world,
    partial_volume,
    supersampled_fraction,
    voxelize,
)

VOLUME_SHAPE = (64, 64, 64)
SPACING = (1.0, 1.0, 1.0)


@pytest.fixture(scope="module")
def settings() -> AppearanceSettings:
    return AppearanceSettings.from_config()


def settings_with(**body: object) -> AppearanceSettings:
    """Appearance settings with ``body`` keys overridden."""
    config = copy.deepcopy(load_config("appearance"))
    config["body"] = {**config["body"], **body}
    return AppearanceSettings.from_config(config)


@pytest.fixture(scope="module")
def scene():
    return generate_scene(1000000)


# ---------------------------------------------------------------------------
# Partial volume
# ---------------------------------------------------------------------------
def test_one_subdivision_reproduces_the_voxeliser_exactly():
    """With one sample per voxel the sample *is* the voxel centre."""
    params = {"radius": 5.0}
    center = (31.0, 30.0, 33.0)
    fractions = partial_volume(
        "sphere", params, center, VOLUME_SHAPE, SPACING, subdivisions=1
    )
    binary = voxelize("sphere", params, center, VOLUME_SHAPE, SPACING)
    assert np.array_equal(fractions > 0.5, binary)
    assert set(np.unique(fractions).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "name,params",
    [
        ("cube", {"side": 8.0}),
        ("sphere", {"radius": 5.0}),
        ("cylinder", {"radius": 4.0, "height": 10.0}),
        ("cone", {"radius": 5.0, "height": 12.0}),
        ("torus", {"major_radius": 4.0, "minor_radius": 2.0}),
        ("capsule", {"radius": 4.0, "segment_length": 4.0}),
    ],
)
def test_partial_volume_is_a_fraction_that_integrates_to_the_analytic_volume(name, params):
    center = (32.0, 32.0, 32.0)
    fractions = partial_volume(name, params, center, VOLUME_SHAPE, SPACING, subdivisions=4)
    assert fractions.dtype == np.float32
    assert float(fractions.min()) >= 0.0 and float(fractions.max()) <= 1.0
    # The summed fraction is a quadrature of the solid's volume.
    expected = analytic_volume_world(name, params)
    assert float(fractions.sum()) == pytest.approx(expected, rel=0.03)


def test_partial_volume_gives_soft_edges_a_binary_mask_does_not():
    """Boundary voxels must be genuinely partial: that is the whole point."""
    params = {"radius": 5.0}
    center = (31.5, 31.5, 31.5)
    fractions = partial_volume("sphere", params, center, VOLUME_SHAPE, SPACING)
    binary = voxelize("sphere", params, center, VOLUME_SHAPE, SPACING)
    partial = (fractions > 0.0) & (fractions < 1.0)
    assert int(partial.sum()) > int(binary.sum()) * 0.3
    # The interior is still fully occupied.
    assert float(fractions[np.asarray(binary)].max()) == 1.0


def test_partial_volume_is_zero_outside_the_solids_bounding_box():
    params = {"side": 6.0}
    center = (20.0, 20.0, 20.0)
    fractions = partial_volume("cube", params, center, VOLUME_SHAPE, SPACING)
    half = half_extent_world("cube", params)
    indices = np.nonzero(fractions)
    for axis, index in zip((2, 1, 0), (indices[2], indices[1], indices[0])):
        assert index.min() >= center[axis] - half[axis] - 1.5
        assert index.max() <= center[axis] + half[axis] + 1.5


def test_supersampled_fraction_rejects_a_degenerate_subdivision():
    with pytest.raises(Exception):
        supersampled_fraction(lambda x, y, z: x > 0, VOLUME_SHAPE, SPACING, subdivisions=0)


# ---------------------------------------------------------------------------
# Random fields, bias and acquisition
# ---------------------------------------------------------------------------
def test_a_smooth_field_is_standardised_and_smoother_than_white_noise():
    rng = np.random.default_rng(0)
    field = smooth_random_field(rng, (32, 32, 32), correlation_voxels=6.0)
    assert field.shape == (32, 32, 32)
    assert float(field.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(field.std()) == pytest.approx(1.0, abs=1e-5)
    # Neighbouring voxels of a correlated field agree; of white noise they do not.
    neighbour = float(np.corrcoef(field[:-1].ravel(), field[1:].ravel())[0, 1])
    white = np.random.default_rng(0).normal(size=(32, 32, 32))
    white_neighbour = float(np.corrcoef(white[:-1].ravel(), white[1:].ravel())[0, 1])
    assert neighbour > 0.9 > white_neighbour


def test_the_bias_field_has_the_requested_inhomogeneity_and_unit_mean():
    rng = np.random.default_rng(1)
    field = bias_field(rng, (32, 32, 32), inhomogeneity=0.25, correlation_voxels=20.0)
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-4)
    assert float(field.max() / field.min()) == pytest.approx(1.25, rel=0.02)
    assert float(field.min()) > 0.0


def test_a_disabled_bias_field_is_exactly_one():
    field = bias_field(np.random.default_rng(1), (8, 8, 8), inhomogeneity=0.0, correlation_voxels=4.0)
    assert np.array_equal(field, np.ones((8, 8, 8), dtype=np.float32))


def test_a_full_kspace_window_with_no_noise_is_the_identity():
    rng = np.random.default_rng(2)
    volume = rng.random((16, 16, 16))
    out = simulate_acquisition(rng, volume, kspace_fraction=1.0, noise_sigma=0.0)
    assert np.allclose(out, volume, atol=1e-10)


def test_truncating_kspace_blurs_the_image():
    volume = np.zeros((32, 32, 32))
    volume[12:20, 12:20, 12:20] = 1.0
    rng = np.random.default_rng(3)
    out = simulate_acquisition(rng, volume, kspace_fraction=0.6, noise_sigma=0.0)
    edge = np.abs(np.diff(volume, axis=0)).max()
    blurred_edge = np.abs(np.diff(out, axis=0)).max()
    assert blurred_edge < edge
    # Ringing: values outside the cube stop being exactly zero.
    assert float(np.abs(out[np.asarray(volume) == 0]).max()) > 1e-3


def test_the_kspace_window_keeps_the_dc_term_and_shrinks_with_the_fraction():
    full = kspace_window((16, 16, 16), 1.0)
    half = kspace_window((16, 16, 16), 0.5)
    assert full.all()
    assert half[8, 8, 8]  # fftshifted DC
    assert 0.0 < half.mean() < 1.0


def test_the_noise_in_air_is_rayleigh_at_the_requested_level():
    """Magnitude reconstruction of complex Gaussian noise is Rician; in air, Rayleigh.

    Checked on a pure-noise volume, where there is no ringing from a bright rim
    to contaminate the estimate.
    """
    rng = np.random.default_rng(4)
    sigma = 0.02
    out = simulate_acquisition(
        rng, np.zeros((48, 48, 48)), kspace_fraction=1.0, noise_sigma=sigma
    )
    assert float(out.min()) >= 0.0  # a magnitude image is non-negative
    assert float(out.mean()) == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.05)
    assert float(np.median(out)) == pytest.approx(sigma * np.sqrt(2 * np.log(2)), rel=0.05)


# ---------------------------------------------------------------------------
# Intensities carry no class information
# ---------------------------------------------------------------------------
def test_structure_intensities_are_not_conditioned_on_the_shape_class(settings):
    """The leak guard.

    If the ten classes had ten intensity bands, the model could name a structure
    from its grey level and the three relations would stop being the only route
    to the target. Over many scenes the per-class means must therefore be within
    sampling noise of each other: the spread *between* class means must be far
    smaller than the spread *within* a class.
    """
    assert settings.class_conditioned is False
    draws = 400
    values = {spec.name: [] for spec in SHAPE_VOCABULARY}
    for seed in range(draws):
        rng = np.random.default_rng(seed)
        intensities = structure_intensities(rng, settings, parenchyma=0.6)
        for spec in SHAPE_VOCABULARY:
            values[spec.name].append(intensities[spec.id])

    class_means = np.array([np.mean(values[name]) for name in values])
    within = float(np.mean([np.std(values[name]) for name in values]))
    between = float(class_means.std())
    # If the classes were identical, `between` would be pure sampling noise,
    # `within / sqrt(draws)`. Three times that leaves room for the draw while
    # still failing loudly on a real per-class band.
    assert between < 3.0 * within / np.sqrt(draws)


def test_class_conditioned_intensities_are_available_but_opt_in():
    """The ablation that deliberately leaks class, so the leak can be measured."""
    config = copy.deepcopy(load_config("appearance"))
    config["intensities"] = {**config["intensities"], "class_conditioned": True}
    leaky = AppearanceSettings.from_config(config)
    first = structure_intensities(np.random.default_rng(0), leaky, parenchyma=0.6)
    second = structure_intensities(np.random.default_rng(99), leaky, parenchyma=0.6)
    assert first == second  # deterministic per class, not drawn
    assert len(set(first.values())) == len(SHAPE_VOCABULARY)


def test_a_darker_polarity_puts_every_structure_below_the_parenchyma(settings):
    intensities = structure_intensities(np.random.default_rng(7), settings, parenchyma=0.6)
    assert settings.structure_polarity == "darker"
    assert all(value < 0.6 for value in intensities.values())


# ---------------------------------------------------------------------------
# The simulated scene
# ---------------------------------------------------------------------------
def test_a_generated_scene_carries_an_image_that_is_not_the_occupancy(scene):
    appearance = scene.appearance
    assert appearance is not None
    image = appearance.image
    assert image.shape == scene.volume_shape and image.dtype == np.float32
    assert len(np.unique(image)) > 10_000
    assert not np.array_equal(image > 0.5, scene.scene_volume.astype(bool))
    # Structures sit in tissue, so the surrounding voxels are not zero.
    surrounding = image[(scene.instance_labels == 0) & (appearance.body_fraction > 0.99)]
    assert float(surrounding.mean()) > 0.3
    assert float(surrounding.std()) > 0.0


def test_the_structures_are_visible_but_low_contrast(scene):
    """Segmentable, but not by a threshold - the point of the new corpus."""
    measured = scene.appearance.parameters["measured"]
    assert measured["contrast_to_noise"] > 2.0
    # Against the variation of the tissue around them the margin is small, which
    # is what makes this a stand-in for deep grey nuclei on T1.
    assert 0.5 < measured["contrast_to_background_variation"] < 8.0


def test_no_single_threshold_recovers_the_structures(scene):
    """A sweep over every threshold must fail, or the image is still binary.

    On the binary corpus this reached Dice 1.0 by construction. At the current
    defaults the best global threshold reaches about 0.07, so the bound below
    leaves a wide margin and only fires if a configuration change made the
    foreground separable again.
    """
    image = scene.appearance.image
    truth = scene.instance_labels != 0
    best = 0.0
    for threshold in np.linspace(float(image.min()), float(image.max()), 128):
        for predicted in (image < threshold, image > threshold):
            overlap = float((predicted & truth).sum())
            dice = 2.0 * overlap / float(predicted.sum() + truth.sum() + 1e-9)
            best = max(best, dice)
    assert best < 0.5, f"a global threshold reaches Dice {best:.2f}; the image is too easy"


def test_the_appearance_is_reproducible_from_the_scene_seed(scene):
    again = generate_scene(1000000)
    assert np.array_equal(scene.appearance.image, again.appearance.image)
    assert scene.appearance.parameters == again.appearance.parameters


def test_the_appearance_stream_is_disjoint_from_the_packing_stream():
    """The packer's stream is ``[seed, stage, attempt]``; appearance must not collide."""
    stream = appearance_rng(1234, 0, 1).bit_generator.state
    packing = np.random.default_rng([1234, 0, 1]).bit_generator.state
    assert stream != packing
    assert APPEARANCE_STREAM > 50  # far above any escalation stage index


def test_the_appearance_does_not_change_the_geometry():
    """Same seeds, same layouts, with and without an image.

    Only ``body.constrain_placement`` may move an object, and it is turned off
    here; everything else about the appearance draws from its own stream.
    """
    free = settings_with(constrain_placement=False)
    config = copy.deepcopy(load_config("appearance"))
    config["enabled"] = False
    off = AppearanceSettings.from_config(config)

    with_image = GeneratorSettings.from_config(appearance=free)
    without_image = GeneratorSettings.from_config(appearance=off)
    for seed in (1000000, 1000001, 2000000):
        a = generate_scene(seed, settings=with_image)
        b = generate_scene(seed, settings=without_image)
        assert np.array_equal(a.instance_labels, b.instance_labels)
        assert a.shape_params == b.shape_params
        assert b.appearance is None
        assert np.array_equal(b.image, b.scene_volume)


def test_the_body_constraint_keeps_every_structure_inside_the_head(scene):
    body = scene.body
    assert body is not None
    for spec in SHAPE_VOCABULARY:
        center = scene.shape_centers[spec.name]
        half = half_extent_world(spec.name, scene.shape_params[spec.name])
        assert body.contains_box(center, half)
    # And the head does not fill the field of view, so there is air around it.
    assert float((scene.appearance.body_fraction < 0.01).mean()) > 0.05


def test_a_body_contains_a_box_only_when_all_eight_corners_are_inside(settings):
    body = draw_body(1000000, 0, settings, VOLUME_SHAPE, SPACING)
    centre = body.center
    assert body.contains_box(centre, (1.0, 1.0, 1.0))
    assert not body.contains_box(centre, tuple(axis * 1.2 for axis in body.semi_axes))


def test_measure_image_reports_per_structure_statistics(scene):
    measured = measure_image(
        scene.appearance.image, scene.instance_labels, scene.appearance.body_fraction
    )
    assert set(measured["per_structure"]) == set(SHAPE_VOCABULARY.names)
    for name, stats in measured["per_structure"].items():
        assert stats["voxels"] > 0, name
        assert stats["std"] > 0.0, name
    assert measured["noise_sigma_estimate"] > 0.0


# ---------------------------------------------------------------------------
# Load-time normalisation
# ---------------------------------------------------------------------------
def test_percentile_normalisation_maps_the_body_of_the_histogram_into_zero_one(scene):
    out = normalize_image(scene.appearance.image, mode="percentile", percentiles=(0.5, 99.5))
    assert out.dtype == np.float32
    assert float(np.percentile(out, 0.5)) == pytest.approx(0.0, abs=1e-5)
    assert float(np.percentile(out, 99.5)) == pytest.approx(1.0, abs=1e-5)


def test_zscore_normalisation_standardises_the_volume(scene):
    out = normalize_image(scene.appearance.image, mode="zscore", clip=None)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-4)


def test_normalisation_commutes_with_an_axis_rotation(scene):
    """It has to, or an augmented item would be normalised differently.

    Every mode here is a function of the volume's own intensity distribution,
    which a 90 degree rotation does not change.
    """
    from src.training.augmentations import AxisRotation

    rotation = AxisRotation.from_angles(90, 0, 90)
    image = scene.appearance.image
    rotated_then = normalize_image(rotation.apply(image))
    then_rotated = rotation.apply(normalize_image(image))
    assert np.allclose(rotated_then, then_rotated, atol=1e-6)


def test_an_unknown_normalisation_mode_is_rejected():
    with pytest.raises(AppearanceError):
        normalize_image(np.zeros((4, 4, 4), dtype=np.float32), mode="nope")


def test_an_invalid_polarity_is_rejected():
    config = copy.deepcopy(load_config("appearance"))
    config["intensities"] = {**config["intensities"], "structure_polarity": "sideways"}
    with pytest.raises(AppearanceError):
        AppearanceSettings.from_config(config)


def test_simulate_scene_appearance_accepts_a_scene_without_a_body(scene, settings):
    appearance = simulate_scene_appearance(
        scene.seed,
        0,
        instance_labels=scene.instance_labels,
        shape_params=scene.shape_params,
        shape_centers=scene.shape_centers,
        body=None,
        settings=settings,
        volume_shape=scene.volume_shape,
        spacing=scene.spacing,
    )
    assert np.array_equal(appearance.body_fraction, np.ones(scene.volume_shape, dtype=np.float32))
    assert appearance.parameters["body"] is None
    assert float(appearance.image.min()) >= 0.0
