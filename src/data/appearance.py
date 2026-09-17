"""MRI-like appearance model for a generated scene.

The previous milestone stored one foreground intensity on a zero background.
That is a segmentation problem a threshold solves, and nothing about it
resembles the setting this project is aimed at - segmenting dense subcortical
structures (thalamus, caudate, putamen...) in a brain MRI, where a structure
differs from the tissue around it by a few percent, its boundary is smeared
across a voxel, the whole volume is tilted by a smooth receive-coil gain, and
everything sits under Rician noise.

This module turns a packed scene into such a volume. The geometry is untouched:
which shapes, where and how big is decided in :mod:`src.data.scene_generator`
and drawn from a different random stream, so the same seed gives the same
layout with and without an appearance. The labels are untouched too - they stay
the crisp centre-in-solid masks, exactly as a manual segmentation of a real
acquisition stays crisp while the image under it does not.

Pipeline
--------
1. **Partial volume.** Each solid is supersampled inside its own bounding box,
   so a boundary voxel holds the fraction of its volume inside the solid rather
   than a hard 0/1. This is the dominant edge effect at 1 mm.
2. **Tissue map.** Air, parenchyma, a scalp-like rim and the ten structures each
   get an intensity; the fractions mix them. Structures displace whatever
   background was at that voxel.
3. **Texture.** Two zero-mean Gaussian random fields (one slow, one fine) plus
   an independent field per structure, all multiplicative, so no compartment is
   flat.
4. **Bias field.** One very smooth multiplicative field - the RF receive
   inhomogeneity.
5. **Acquisition.** ``fft -> truncate k-space -> add complex Gaussian noise ->
   ifft -> magnitude``. Truncation blurs and rings (Gibbs); the magnitude of a
   complex Gaussian is what makes the noise Rician, and Rayleigh in air.

Why structure intensities carry no class information
----------------------------------------------------
``intensities.class_conditioned`` is false by default and should stay false.
Giving each shape class its own grey level would let the model recognise a
class from its intensity, and the point of the benchmark is that the target is
identified *only* through three spatial relations. It is also the faithful
choice: real deep grey nuclei are close to isointense with each other and with
the white matter around them, which is precisely why spatial priors are needed
to separate them at all.

Determinism
-----------
Everything here is drawn from ``default_rng([seed, APPEARANCE_STREAM, stage,
substream])``. The packing retries use ``default_rng([seed, stage, attempt])``,
so the two never share state: a scene's appearance does not shift when packing
happens to need one more attempt, and geometry is bit-identical to an
appearance-free corpus generated from the same seed (unless
``body.constrain_placement`` is on, which is the one appearance setting that
deliberately restricts placement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.config import load_config
from src.data.primitives import SHAPE_VOCABULARY
from src.data.voxelization import partial_volume, supersampled_fraction

#: Second element of every appearance seed vector. Any value distinct from the
#: escalation-stage indices used by the packer keeps the two streams apart.
APPEARANCE_STREAM = 7919

#: Sub-stream indices, so the body can be drawn before packing and the rest
#: after, without either depending on the other's consumption of the generator.
_BODY_SUBSTREAM = 0
_IMAGE_SUBSTREAM = 1

_POLARITIES = ("darker", "brighter", "mixed", "random")
_NORMALIZATIONS = ("percentile", "zscore", "none")


class AppearanceError(ValueError):
    """Raised when the appearance configuration is inconsistent."""


def appearance_rng(seed: int, stage: int, substream: int) -> np.random.Generator:
    """The generator for one appearance sub-stream of one scene."""
    return np.random.default_rng([int(seed), APPEARANCE_STREAM, int(stage), int(substream)])


def _range(value: Any, name: str) -> tuple[float, float]:
    """Accept either a scalar or a ``[low, high]`` pair and return a pair."""
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    values = tuple(float(v) for v in value)
    if len(values) != 2 or values[0] > values[1]:
        raise AppearanceError(f"{name} must be a scalar or an ordered [low, high], got {value!r}")
    return values  # type: ignore[return-value]


def _draw(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low) if low == high else float(rng.uniform(low, high))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppearanceSettings:
    """Everything ``configs/appearance.yaml`` says, resolved and checked."""

    enabled: bool
    version: str
    subdivisions: int
    body_enabled: bool
    constrain_placement: bool
    body_semi_axis_fraction: tuple[float, float]
    body_exponent: tuple[float, float]
    body_center_jitter: float
    body_subdivisions: int
    rim_enabled: bool
    rim_thickness: tuple[float, float]
    air_intensity: float
    parenchyma_intensity: tuple[float, float]
    rim_intensity: tuple[float, float]
    structure_contrast: float
    structure_contrast_jitter: float
    structure_polarity: str
    class_conditioned: bool
    texture_coarse_correlation: float
    texture_coarse_amplitude: float
    texture_fine_correlation: float
    texture_fine_amplitude: float
    texture_per_structure_amplitude: float
    bias_enabled: bool
    bias_inhomogeneity: tuple[float, float]
    bias_correlation: float
    kspace_fraction: tuple[float, float]
    noise_sigma: tuple[float, float]
    normalization_mode: str
    normalization_percentiles: tuple[float, float]
    normalization_clip: tuple[float, float] | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "AppearanceSettings":
        config = dict(config or load_config("appearance"))
        body = dict(config.get("body", {}))
        rim = dict(body.get("rim", {}))
        intensities = dict(config.get("intensities", {}))
        texture = dict(config.get("texture", {}))
        coarse = dict(texture.get("coarse", {}))
        fine = dict(texture.get("fine", {}))
        bias = dict(config.get("bias_field", {}))
        acquisition = dict(config.get("acquisition", {}))
        normalization = dict(config.get("normalization", {}))

        polarity = str(intensities.get("structure_polarity", "darker"))
        if polarity not in _POLARITIES:
            raise AppearanceError(
                f"intensities.structure_polarity must be one of {_POLARITIES}, got {polarity!r}"
            )
        mode = str(normalization.get("mode", "percentile"))
        if mode not in _NORMALIZATIONS:
            raise AppearanceError(
                f"normalization.mode must be one of {_NORMALIZATIONS}, got {mode!r}"
            )
        percentiles = tuple(float(v) for v in normalization.get("percentiles", (0.5, 99.5)))
        if len(percentiles) != 2 or not 0.0 <= percentiles[0] < percentiles[1] <= 100.0:
            raise AppearanceError(
                f"normalization.percentiles must be an ordered pair in [0, 100], got {percentiles}"
            )
        clip = normalization.get("clip")
        clip_pair = None if clip is None else _range(clip, "normalization.clip")

        return cls(
            enabled=bool(config.get("enabled", True)),
            version=str(config.get("appearance_version", "0.0.0")),
            subdivisions=int(dict(config.get("partial_volume", {})).get("subdivisions", 4)),
            body_enabled=bool(body.get("enabled", True)),
            constrain_placement=bool(body.get("constrain_placement", True)),
            body_semi_axis_fraction=_range(
                body.get("semi_axis_fraction", [0.44, 0.48]), "body.semi_axis_fraction"
            ),
            body_exponent=_range(body.get("exponent", [3.0, 4.5]), "body.exponent"),
            body_center_jitter=float(body.get("center_jitter_voxels", 0.0)),
            body_subdivisions=int(body.get("subdivisions", 2)),
            rim_enabled=bool(rim.get("enabled", True)),
            rim_thickness=_range(rim.get("thickness_voxels", [1.5, 3.0]), "body.rim.thickness_voxels"),
            air_intensity=float(intensities.get("air", 0.0)),
            parenchyma_intensity=_range(
                intensities.get("parenchyma", [0.55, 0.70]), "intensities.parenchyma"
            ),
            rim_intensity=_range(intensities.get("rim", [0.80, 1.00]), "intensities.rim"),
            structure_contrast=float(intensities.get("structure_contrast", 0.15)),
            structure_contrast_jitter=float(intensities.get("structure_contrast_jitter", 0.04)),
            structure_polarity=polarity,
            class_conditioned=bool(intensities.get("class_conditioned", False)),
            texture_coarse_correlation=float(coarse.get("correlation_voxels", 14.0)),
            texture_coarse_amplitude=float(coarse.get("amplitude", 0.055)),
            texture_fine_correlation=float(fine.get("correlation_voxels", 3.0)),
            texture_fine_amplitude=float(fine.get("amplitude", 0.022)),
            texture_per_structure_amplitude=float(texture.get("per_structure_amplitude", 0.02)),
            bias_enabled=bool(bias.get("enabled", True)),
            bias_inhomogeneity=_range(
                bias.get("inhomogeneity", [0.15, 0.30]), "bias_field.inhomogeneity"
            ),
            bias_correlation=float(bias.get("correlation_voxels", 40.0)),
            kspace_fraction=_range(
                acquisition.get("kspace_fraction", [0.82, 0.94]), "acquisition.kspace_fraction"
            ),
            noise_sigma=_range(
                acquisition.get("noise_sigma", [0.015, 0.03]), "acquisition.noise_sigma"
            ),
            normalization_mode=mode,
            normalization_percentiles=percentiles,  # type: ignore[arg-type]
            normalization_clip=clip_pair,
        )


# ---------------------------------------------------------------------------
# The body: a superellipsoid "head" the structures live inside
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BodyGeometry:
    """The head outline of one scene, in world units ``(x, y, z)``."""

    center: tuple[float, float, float]
    semi_axes: tuple[float, float, float]
    exponent: float
    rim_thickness: float

    def _radial(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, shrink: float) -> np.ndarray:
        """Superellipsoid radius; ``<= 1`` is inside the shrunk body."""
        axes = [max(axis - shrink, 1e-6) for axis in self.semi_axes]
        return (
            np.abs((x - self.center[0]) / axes[0]) ** self.exponent
            + np.abs((y - self.center[1]) / axes[1]) ** self.exponent
            + np.abs((z - self.center[2]) / axes[2]) ** self.exponent
        )

    def contains_box(
        self, center_world: Sequence[float], half_extent: Sequence[float], *, margin: float = 0.0
    ) -> bool:
        """Is an axis-aligned box entirely inside the body?

        The body is convex for ``exponent >= 1``, so testing the eight corners
        settles it for the whole box.
        """
        corners = np.array(
            [
                [
                    float(center_world[axis]) + sign * float(half_extent[axis])
                    for axis, sign in zip(range(3), signs)
                ]
                for signs in (
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                )
            ],
            dtype=np.float64,
        )
        radial = self._radial(corners[:, 0], corners[:, 1], corners[:, 2], shrink=margin)
        return bool(np.all(radial <= 1.0))

    def fractions(
        self,
        volume_shape: Sequence[int],
        spacing: Sequence[float],
        *,
        shrink: float = 0.0,
        subdivisions: int = 2,
    ) -> np.ndarray:
        """Per-voxel fraction inside the body, optionally shrunk by ``shrink``."""

        def indicator(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            return self._radial(x, y, z, shrink=shrink) <= 1.0

        return supersampled_fraction(
            indicator, volume_shape, spacing, bounds_world=None, subdivisions=subdivisions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_world": list(self.center),
            "semi_axes_world": list(self.semi_axes),
            "exponent": self.exponent,
            "rim_thickness_world": self.rim_thickness,
        }


def draw_body(
    seed: int,
    stage: int,
    settings: AppearanceSettings,
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> BodyGeometry | None:
    """Draw the head outline of one scene, or ``None`` when bodies are off.

    Drawn before packing so that placement can be constrained to it, from its
    own sub-stream so that the draw does not depend on how packing goes.
    """
    if not (settings.enabled and settings.body_enabled):
        return None
    rng = appearance_rng(seed, stage, _BODY_SUBSTREAM)
    depth, height, width = (int(v) for v in volume_shape)
    sizes = (width, height, depth)  # world-axis order (x, y, z)
    steps = tuple(float(v) for v in spacing)
    spans = tuple((size - 1) * step for size, step in zip(sizes, steps))
    center = tuple(
        span / 2.0 + float(rng.normal(0.0, settings.body_center_jitter)) * step
        for span, step in zip(spans, steps)
    )
    semi_axes = tuple(
        _draw(rng, settings.body_semi_axis_fraction) * size * step
        for size, step in zip(sizes, steps)
    )
    exponent = _draw(rng, settings.body_exponent)
    thickness = _draw(rng, settings.rim_thickness) * float(np.mean(steps))
    return BodyGeometry(
        center=center,  # type: ignore[arg-type]
        semi_axes=semi_axes,  # type: ignore[arg-type]
        exponent=exponent,
        rim_thickness=thickness if settings.rim_enabled else 0.0,
    )


# ---------------------------------------------------------------------------
# Random fields
# ---------------------------------------------------------------------------
def smooth_random_field(
    rng: np.random.Generator,
    volume_shape: Sequence[int],
    correlation_voxels: float,
) -> np.ndarray:
    """A zero-mean, unit-variance Gaussian random field with a smooth spectrum.

    White noise filtered by a Gaussian of width ``correlation_voxels``, done in
    the Fourier domain so the cost is one FFT pair and the result is
    periodic-free of edge effects at the scales that matter here. Rescaled to
    unit variance afterwards, so ``correlation_voxels`` changes the *shape* of
    the field and the caller's amplitude changes its size.
    """
    shape = tuple(int(v) for v in volume_shape)
    field = rng.normal(0.0, 1.0, size=shape)
    if correlation_voxels <= 0:
        return field.astype(np.float32)
    frequencies = [np.fft.fftfreq(size) for size in shape]
    squared = sum(
        np.reshape(freq, [-1 if axis == i else 1 for i in range(3)]) ** 2
        for axis, freq in enumerate(frequencies)
    )
    transfer = np.exp(-2.0 * (np.pi * correlation_voxels) ** 2 * squared)
    smoothed = np.real(np.fft.ifftn(np.fft.fftn(field) * transfer))
    smoothed -= smoothed.mean()
    deviation = float(smoothed.std())
    if deviation > 0:
        smoothed /= deviation
    return smoothed.astype(np.float32)


def bias_field(
    rng: np.random.Generator,
    volume_shape: Sequence[int],
    *,
    inhomogeneity: float,
    correlation_voxels: float,
) -> np.ndarray:
    """A smooth, strictly positive multiplicative gain field.

    Normalised so that ``max / min - 1`` equals ``inhomogeneity`` and the mean
    gain is 1, which makes the configured number mean what it says instead of
    depending on how the draw happened to land.
    """
    if inhomogeneity <= 0:
        return np.ones(tuple(int(v) for v in volume_shape), dtype=np.float32)
    field = smooth_random_field(rng, volume_shape, correlation_voxels)
    span = float(field.max() - field.min())
    if span <= 0:  # pragma: no cover - a constant field cannot be inhomogeneous
        return np.ones_like(field)
    # exp(a*field) has ratio exp(a*span); solve for the requested ratio.
    gain = np.log1p(inhomogeneity) / span
    out = np.exp(gain * field)
    out /= float(out.mean())
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def kspace_window(volume_shape: Sequence[int], fraction: float) -> np.ndarray:
    """Centred rectangular sampling window over a fftshifted k-space."""
    shape = tuple(int(v) for v in volume_shape)
    fraction = float(np.clip(fraction, 1e-3, 1.0))
    window = np.ones(shape, dtype=bool)
    for axis, size in enumerate(shape):
        # fftshift puts the DC term at index size // 2.
        centre = size // 2
        half = max(fraction * size / 2.0, 0.5)
        offsets = np.abs(np.arange(size) - centre)
        keep = offsets <= half
        window &= np.reshape(keep, [-1 if axis == i else 1 for i in range(3)])
    return window


def simulate_acquisition(
    rng: np.random.Generator,
    volume: np.ndarray,
    *,
    kspace_fraction: float,
    noise_sigma: float,
) -> np.ndarray:
    """Turn a noise-free tissue map into a magnitude image.

    Sampling a finite k-space band blurs the image and rings at sharp edges
    (Gibbs). Complex Gaussian noise added in k-space becomes complex Gaussian
    noise in the image, and the magnitude of that is Rician - Rayleigh where
    there is no signal, which is why the air around the head is not black.

    ``noise_sigma`` is the standard deviation of each image-domain component,
    not of the k-space samples: with ``ifftn`` scaling by ``1 / N`` and a
    fraction ``rho`` of samples retained, k-space noise of standard deviation
    ``sigma * sqrt(N / rho)`` lands at ``sigma`` in the image.
    """
    data = np.asarray(volume, dtype=np.float64)
    spectrum = np.fft.fftshift(np.fft.fftn(data))
    window = kspace_window(data.shape, kspace_fraction)
    if noise_sigma > 0:
        retained = float(window.mean())
        scale = float(noise_sigma) * np.sqrt(data.size / max(retained, 1e-12))
        spectrum = spectrum + (
            rng.normal(0.0, scale, size=data.shape) + 1j * rng.normal(0.0, scale, size=data.shape)
        )
    spectrum = spectrum * window
    return np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))


# ---------------------------------------------------------------------------
# The simulated scene
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneAppearance:
    """One scene's simulated image and everything that went into it."""

    image: np.ndarray  # (D, H, W) float32, the MRI-like volume
    tissue_image: np.ndarray  # (D, H, W) float32, noise-free and bias-free
    bias: np.ndarray  # (D, H, W) float32, the multiplicative gain
    structure_fraction: np.ndarray  # (D, H, W) float32, total partial volume
    body_fraction: np.ndarray  # (D, H, W) float32, fraction inside the head
    parameters: dict[str, Any]  # JSON-serialisable record of every draw


def structure_intensities(
    rng: np.random.Generator,
    settings: AppearanceSettings,
    parenchyma: float,
) -> dict[int, float]:
    """One mean intensity per instance ID.

    With ``class_conditioned`` false - the default - the draw is independent of
    the shape class, so grey level carries no information about *what* a
    structure is and the target can only be found through the relations.
    """
    intensities: dict[int, float] = {}
    scene_sign = 1.0 if rng.random() < 0.5 else -1.0
    for spec in SHAPE_VOCABULARY:
        if settings.structure_polarity == "darker":
            sign = -1.0
        elif settings.structure_polarity == "brighter":
            sign = 1.0
        elif settings.structure_polarity == "mixed":
            sign = 1.0 if rng.random() < 0.5 else -1.0
        else:  # "random": one sign per scene
            sign = scene_sign
        if settings.class_conditioned:
            # Deterministic per-class offset across the configured band. Only
            # for ablations that deliberately want appearance to leak class.
            index = SHAPE_VOCABULARY.index_of(spec.name)
            spread = (index / max(len(SHAPE_VOCABULARY) - 1, 1)) - 0.5
            contrast = settings.structure_contrast * (1.0 + spread)
        else:
            contrast = float(
                rng.normal(settings.structure_contrast, settings.structure_contrast_jitter)
            )
        contrast = abs(contrast)
        intensities[spec.id] = float(max(parenchyma + sign * contrast, 0.0))
    return intensities


def simulate_scene_appearance(
    seed: int,
    stage: int,
    *,
    instance_labels: np.ndarray,
    shape_params: Mapping[str, Mapping[str, float]],
    shape_centers: Mapping[str, Sequence[float]],
    body: BodyGeometry | None,
    settings: AppearanceSettings,
    volume_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> SceneAppearance:
    """Simulate the MRI-like volume of one packed scene.

    Args:
        seed: the scene seed; the appearance stream derives from it.
        stage: the escalation stage the scene was packed at.
        instance_labels: ``(D, H, W)`` labels, 0 background and 1..10 instances.
        shape_params: per shape name, the size parameters that were placed.
        shape_centers: per shape name, the placed centre in world ``(x, y, z)``.
        body: the head outline, or ``None`` for no body.
        settings: the resolved appearance configuration.
        volume_shape: ``(D, H, W)``.
        spacing: world units per voxel, ordered ``(x, y, z)``.
    """
    shape = tuple(int(v) for v in volume_shape)
    rng = appearance_rng(seed, stage, _IMAGE_SUBSTREAM)

    # 1. Partial-volume fractions, one per structure.
    fractions: dict[int, np.ndarray] = {}
    for spec in SHAPE_VOCABULARY:
        fractions[spec.id] = partial_volume(
            spec.name,
            shape_params[spec.name],
            shape_centers[spec.name],
            shape,
            spacing,
            subdivisions=settings.subdivisions,
        )
    total = np.zeros(shape, dtype=np.float32)
    for fraction in fractions.values():
        total += fraction
    # Two solids never share a voxel *centre*, but they can share a boundary
    # voxel; scale those back so the compartments still partition the voxel.
    overflow = total > 1.0
    if overflow.any():
        scale = np.ones(shape, dtype=np.float32)
        scale[overflow] = 1.0 / total[overflow]
        for instance_id in fractions:
            fractions[instance_id] = fractions[instance_id] * scale
        total = np.minimum(total, 1.0)

    # 2. Compartment intensities.
    parenchyma = _draw(rng, settings.parenchyma_intensity)
    rim_intensity = _draw(rng, settings.rim_intensity)
    structure_intensity = structure_intensities(rng, settings, parenchyma)

    if body is None:
        body_fraction = np.ones(shape, dtype=np.float32)
        core_fraction = body_fraction
        rim_fraction = np.zeros(shape, dtype=np.float32)
    else:
        body_fraction = body.fractions(
            shape, spacing, shrink=0.0, subdivisions=settings.body_subdivisions
        )
        if body.rim_thickness > 0:
            core_fraction = body.fractions(
                shape,
                spacing,
                shrink=body.rim_thickness,
                subdivisions=settings.body_subdivisions,
            )
            rim_fraction = np.clip(body_fraction - core_fraction, 0.0, 1.0)
        else:
            core_fraction = body_fraction
            rim_fraction = np.zeros(shape, dtype=np.float32)
    air_fraction = np.clip(1.0 - body_fraction, 0.0, 1.0)

    background = (
        air_fraction * settings.air_intensity
        + rim_fraction * rim_intensity
        + core_fraction * parenchyma
    ).astype(np.float32)

    # 3. Texture. One shared pair of fields plus an independent field per
    #    structure, so neighbouring structures do not vary in lockstep.
    texture = np.zeros(shape, dtype=np.float32)
    if settings.texture_coarse_amplitude > 0:
        texture += settings.texture_coarse_amplitude * smooth_random_field(
            rng, shape, settings.texture_coarse_correlation
        )
    if settings.texture_fine_amplitude > 0:
        texture += settings.texture_fine_amplitude * smooth_random_field(
            rng, shape, settings.texture_fine_correlation
        )

    tissue = (1.0 - total) * background
    for spec in SHAPE_VOCABULARY:
        fraction = fractions[spec.id]
        local = np.full(shape, structure_intensity[spec.id], dtype=np.float32)
        if settings.texture_per_structure_amplitude > 0:
            local = local * (
                1.0
                + settings.texture_per_structure_amplitude
                * smooth_random_field(rng, shape, settings.texture_fine_correlation)
            )
        tissue += fraction * local
    tissue = np.clip(tissue * (1.0 + texture), 0.0, None).astype(np.float32)

    # 4. Receive-coil inhomogeneity.
    inhomogeneity = _draw(rng, settings.bias_inhomogeneity) if settings.bias_enabled else 0.0
    gain = bias_field(
        rng, shape, inhomogeneity=inhomogeneity, correlation_voxels=settings.bias_correlation
    )

    # 5. Acquisition.
    fraction_kspace = _draw(rng, settings.kspace_fraction)
    noise_sigma = _draw(rng, settings.noise_sigma)
    image = simulate_acquisition(
        rng,
        tissue * gain,
        kspace_fraction=fraction_kspace,
        noise_sigma=noise_sigma,
    ).astype(np.float32)

    labels = np.asarray(instance_labels)
    requested_contrast = abs(
        float(np.mean([structure_intensity[spec.id] for spec in SHAPE_VOCABULARY])) - parenchyma
    )
    parameters: dict[str, Any] = {
        "appearance_version": settings.version,
        "seed": int(seed),
        "stage": int(stage),
        "stream": [int(seed), APPEARANCE_STREAM, int(stage), _IMAGE_SUBSTREAM],
        "parenchyma_intensity": parenchyma,
        "rim_intensity": rim_intensity if (body is not None and body.rim_thickness > 0) else None,
        "air_intensity": settings.air_intensity,
        "structure_polarity": settings.structure_polarity,
        "class_conditioned_intensities": settings.class_conditioned,
        "structure_intensities": {
            SHAPE_VOCABULARY.id_to_name(instance_id): value
            for instance_id, value in sorted(structure_intensity.items())
        },
        "bias_inhomogeneity": inhomogeneity,
        "kspace_fraction": fraction_kspace,
        "noise_sigma": noise_sigma,
        "requested_contrast": requested_contrast,
        "body": None if body is None else body.to_dict(),
        "measured": measure_image(image, labels, body_fraction),
    }
    return SceneAppearance(
        image=image,
        tissue_image=tissue,
        bias=gain,
        structure_fraction=total,
        body_fraction=body_fraction,
        parameters=parameters,
    )


#: Median of a Rayleigh variable with scale 1. The air region of a magnitude
#: image is Rayleigh, so ``median(air) / this`` recovers the noise level of a
#: finished volume. The median rather than the standard deviation, because
#: Gibbs ringing from the bright rim spills into the air and inflates the tail.
_RAYLEIGH_MEDIAN = float(np.sqrt(2.0 * np.log(2.0)))


def measure_image(
    image: np.ndarray,
    instance_labels: np.ndarray,
    body_fraction: np.ndarray | None = None,
) -> dict[str, Any]:
    """Intensity statistics of a simulated scene, for the run report.

    Reports what the volume actually looks like rather than what was asked for.
    Two difficulty numbers, because they answer different questions:

    ``contrast_to_noise``
        structure-to-surroundings contrast over the noise standard deviation,
        recovered from the air region (a Rayleigh distribution, so
        ``sigma = median / sqrt(2 ln 2)``). This is the classical CNR.
    ``contrast_to_background_variation``
        the same contrast over the standard deviation of the surrounding
        tissue, which also contains the texture and the bias field. This is the
        honest difficulty of the segmentation problem, and it is the smaller of
        the two.
    """
    values = np.asarray(image, dtype=np.float64)
    labels = np.asarray(instance_labels)
    foreground = labels != 0
    fractions = None if body_fraction is None else np.asarray(body_fraction)
    if fractions is None:
        background = ~foreground
        air = np.zeros_like(background)
    else:
        background = (~foreground) & (fractions > 0.99)
        air = fractions < 0.01
    background_values = values[background]
    background_mean = float(background_values.mean()) if background_values.size else 0.0
    background_std = float(background_values.std()) if background_values.size else 0.0
    air_values = values[air]
    air_std = float(air_values.std()) if air_values.size else 0.0
    noise_sigma = (
        float(np.median(air_values)) / _RAYLEIGH_MEDIAN if air_values.size else 0.0
    )

    per_structure: dict[str, dict[str, float]] = {}
    for spec in SHAPE_VOCABULARY:
        mask = labels == spec.id
        if not mask.any():
            continue
        selected = values[mask]
        per_structure[spec.name] = {
            "mean": float(selected.mean()),
            "std": float(selected.std()),
            "voxels": int(mask.sum()),
        }
    contrasts = [abs(stats["mean"] - background_mean) for stats in per_structure.values()]
    mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "tissue_background_mean": background_mean,
        "tissue_background_std": background_std,
        "air_std": air_std,
        "noise_sigma_estimate": noise_sigma,
        "per_structure": per_structure,
        "mean_contrast": mean_contrast,
        "contrast_to_noise": (mean_contrast / noise_sigma) if noise_sigma > 0 else None,
        "contrast_to_background_variation": (
            (mean_contrast / background_std) if background_std > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Load-time normalisation
# ---------------------------------------------------------------------------
def normalize_image(
    image: np.ndarray,
    *,
    mode: str = "percentile",
    percentiles: Sequence[float] = (0.5, 99.5),
    clip: Sequence[float] | None = (-0.5, 1.5),
) -> np.ndarray:
    """Per-volume intensity normalisation, applied at load time.

    The stored image keeps the simulated units; a model never sees them raw,
    because a real acquisition's units are arbitrary too. ``percentile`` maps
    the two given percentiles to ``[0, 1]``, which is robust to the bright rim
    and to the noise floor in air; ``zscore`` standardises the whole volume.

    This is deterministic and depends only on the volume itself, so it commutes
    with the axis-aligned rotation augmentation.
    """
    values = np.asarray(image, dtype=np.float32)
    if mode == "none":
        return values
    if mode == "zscore":
        mean = float(values.mean())
        deviation = float(values.std())
        out = (values - mean) / deviation if deviation > 0 else values - mean
    elif mode == "percentile":
        low, high = (float(v) for v in percentiles)
        lo, hi = np.percentile(values, [low, high])
        span = float(hi - lo)
        out = (values - float(lo)) / span if span > 0 else values - float(lo)
    else:
        raise AppearanceError(f"unknown normalization mode {mode!r}")
    if clip is not None:
        out = np.clip(out, float(clip[0]), float(clip[1]))
    return out.astype(np.float32)
