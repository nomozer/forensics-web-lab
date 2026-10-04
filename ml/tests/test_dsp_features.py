"""Unit tests for DSP feature extraction module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ml.training.dsp_features import (
    DSP_FEATURE_DIM,
    DSP_FEATURE_NAMES,
    compute_2d_fft_analysis,
    compute_8x8_dct_analysis,
    compute_jpeg_block_grid_analysis,
    compute_laplacian_variance,
    compute_noise_residual_analysis,
    extract_dsp_features,
    to_grayscale_float,
)


def test_feature_names_and_dimensions():
    assert len(DSP_FEATURE_NAMES) == 16
    assert DSP_FEATURE_DIM == 16
    assert len(set(DSP_FEATURE_NAMES)) == 16  # All unique


def test_to_grayscale_float():
    # 3-channel RGB image
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Red
    gray = to_grayscale_float(arr)
    assert gray.shape == (32, 32)
    assert gray.dtype == np.float32
    assert np.allclose(gray, 0.299, atol=1e-3)

    # PIL Image
    pil_img = Image.fromarray(arr)
    gray_pil = to_grayscale_float(pil_img)
    assert gray_pil.shape == (32, 32)
    assert np.allclose(gray_pil, gray, atol=1e-5)


def test_extract_dsp_features_shape_and_finite():
    rng = np.random.default_rng(42)
    noise_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    feats = extract_dsp_features(noise_img)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (16,)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_extract_dsp_features_is_deterministic():
    rng = np.random.default_rng(1337)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    f1 = extract_dsp_features(img)
    f2 = extract_dsp_features(img)
    assert np.array_equal(f1, f2)


def test_synthetic_patterns_produce_expected_contrasts():
    # Flat image
    flat = np.ones((128, 128), dtype=np.float32) * 0.5
    flat_fft = compute_2d_fft_analysis(flat)
    flat_dct = compute_8x8_dct_analysis(flat)
    flat_noise = compute_noise_residual_analysis(flat)
    flat_jpeg = compute_jpeg_block_grid_analysis(flat)
    flat_lap = compute_laplacian_variance(flat)

    assert flat_fft["fft_high_freq_energy_ratio"] == pytest.approx(0.0, abs=1e-5)
    assert flat_dct["dct_mean_high_freq_energy"] == pytest.approx(0.0, abs=1e-5)
    assert flat_noise["noise_global_level"] == pytest.approx(0.0, abs=1e-5)
    assert flat_lap == pytest.approx(0.0, abs=1e-5)

    # High frequency checkerboard pattern
    y, x = np.ogrid[:128, :128]
    checker = ((x + y) % 2).astype(np.float32)
    checker_fft = compute_2d_fft_analysis(checker)
    checker_dct = compute_8x8_dct_analysis(checker)

    assert checker_fft["fft_high_freq_energy_ratio"] > 0.1
    assert checker_dct["dct_mean_high_freq_energy"] > 0.0


def test_edge_case_image_sizes():
    for h, w in [(16, 16), (24, 40), (128, 64), (224, 224)]:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        feats = extract_dsp_features(arr)
        assert feats.shape == (16,)
        assert np.all(np.isfinite(feats))
