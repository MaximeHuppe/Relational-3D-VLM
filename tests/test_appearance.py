"""Seeded MRI-like appearance: noisy background, overlapping structure means."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.appearance import (
    AppearanceSettings,
    class_intensity_bias,
    gaussian_blur_3d,
    render_appearance,
)


def _labels(shape=(16, 16, 16)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    for index in range(10):
        z = 1 + (index // 4) * 4
        y = 1 + (index % 4) * 3
        x = 1 + (index % 3) * 4
        labels[z : z + 2, y : y + 2, x : x + 2] = index + 1
    return labels


def test_disabled_appearance_is_binary_occupancy():
    labels = _labels()
    settings = AppearanceSettings.from_config({"enabled": False})
    volume = render_appearance(labels, np.random.default_rng(0), settings)
    np.testing.assert_array_equal(volume, (labels != 0).astype(np.float32))


def test_background_is_not_zero_and_structures_are_not_one():
    labels = _labels()
    settings = AppearanceSettings.from_config({})
    volume = render_appearance(labels, np.random.default_rng(1), settings)
    background = volume[labels == 0]
    foreground = volume[labels != 0]
    assert background.std() > 0
    assert not np.allclose(background, 0.0, atol=1e-6)
    assert not np.allclose(foreground, 1.0, atol=1e-6)
    assert float(foreground.mean()) > float(background.mean())
    assert volume.min() >= settings.clip[0]
    assert volume.max() <= settings.clip[1]


def test_appearance_is_seed_stable_and_does_not_mutate_labels():
    labels = _labels()
    original = labels.copy()
    settings = AppearanceSettings.from_config({})
    a = render_appearance(labels, np.random.default_rng(42), settings)
    b = render_appearance(labels, np.random.default_rng(42), settings)
    c = render_appearance(labels, np.random.default_rng(43), settings)
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_class_bias_overlaps_rather_than_separating_classes():
    low, high = 0.45, 0.75
    means = [0.5 * (low + high) + class_intensity_bias(i, 0.04) for i in range(1, 11)]
    assert min(means) >= low
    assert max(means) <= high
    assert max(means) - min(means) < (high - low)


def test_blur_mixes_an_edge():
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[3, 3, 3] = 1.0
    blurred = gaussian_blur_3d(volume, sigma=0.6)
    assert blurred[3, 3, 3] < 1.0
    assert blurred[3, 3, 4] > 0.0


def test_from_config_rejects_inverted_ranges():
    with pytest.raises(ValueError):
        AppearanceSettings.from_config({"structure_mean_range": [0.8, 0.2]})
