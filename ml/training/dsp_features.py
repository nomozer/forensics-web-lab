"""Digital Signal Processing (DSP) Forensic Feature Extraction Module.

Extracts 16 deterministic, physics- and signal-inspired forensic features from a single
image, mirroring the browser-side implementation in packages/forensics:
  - 2D-FFT Radial & High-Frequency Power Analysis (4 features)
  - 2D-DCT 8x8 Block AC Energy Distribution (3 features)
  - Spatial Noise Residual Consistency via 3x3 Laplacian (5 features)
  - JPEG 8x8 Block Artifact Grid (BAG) Gradient Ratio (3 features)
  - Global Texture / Sharpness Variance (1 feature)

Prediction contract:
  - Functions operate purely on individual image pixels.
  - Zero access to peer images, source IDs, file paths, or ground-truth labels.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


DSP_FEATURE_NAMES: tuple[str, ...] = (
    "fft_high_freq_energy_ratio",
    "fft_spectral_peak_count",
    "fft_radial_anomaly_score",
    "fft_spectral_decay_slope",
    "dct_mean_high_freq_energy",
    "dct_ac_energy_variance",
    "dct_anomaly_score",
    "noise_global_level",
    "noise_max_to_min_ratio",
    "noise_std_of_variances",
    "noise_coef_variation",
    "noise_inconsistency_score",
    "jpeg_boundary_difference_ratio",
    "jpeg_periodic_grid_strength",
    "jpeg_artifact_score",
    "laplacian_variance",
)

DSP_FEATURE_DIM = len(DSP_FEATURE_NAMES)
assert DSP_FEATURE_DIM == 16


def to_grayscale_float(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert PIL image or numpy array to 2D float32 array in [0.0, 1.0]."""
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            elif arr.shape[0] in (1, 3):  # Channel-first
                arr = np.moveaxis(arr, 0, -1)[:, :, :3]
    else:
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float32)

    # Standard ITU-R BT.601 luminance weights
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return np.ascontiguousarray(gray, dtype=np.float32)


def compute_2d_fft_analysis(
    gray: np.ndarray, target_size: int = 128
) -> dict[str, float]:
    """Compute 2D FFT radial spectrum and high-frequency anomaly metrics matching fft2d.ts."""
    h, w = gray.shape
    if h < 8 or w < 8:
        return {
            "fft_high_freq_energy_ratio": 0.0,
            "fft_spectral_peak_count": 0.0,
            "fft_radial_anomaly_score": 0.0,
            "fft_spectral_decay_slope": -2.0,
        }

    # Resample to target power-of-two square if not already target size
    if h == target_size and w == target_size:
        res_arr = gray.astype(np.float32)
    else:
        pil_gray = Image.fromarray((np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8))
        resampled = pil_gray.resize((target_size, target_size), Image.Resampling.BILINEAR)
        res_arr = np.asarray(resampled, dtype=np.float32) / 255.0

    # 2D FFT and centered magnitude spectrum
    fft2 = np.fft.fft2(res_arr)
    fft_shifted = np.fft.fftshift(fft2)
    mag = np.abs(fft_shifted)

    half = target_size // 2
    num_bins = half
    radial_bins = np.zeros(num_bins, dtype=np.float64)
    radial_counts = np.zeros(num_bins, dtype=np.int32)

    cy, cx = half, half
    y, x = np.ogrid[:target_size, :target_size]
    dx = x - cx
    dy = y - cy
    dist = np.floor(np.sqrt(dx * dx + dy * dy)).astype(np.int32)

    total_energy = float(np.sum(mag))
    high_freq_cutoff = int(half * 0.6)
    high_freq_mask = dist >= high_freq_cutoff
    high_freq_energy = float(np.sum(mag[high_freq_mask]))

    for b in range(num_bins):
        mask = dist == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            radial_bins[b] = float(np.sum(mag[mask])) / cnt
            radial_counts[b] = cnt

    # Detect anomalous high-frequency spikes (higher than 1.4x surrounding neighborhood)
    peak_count = 0.0
    for b in range(3, num_bins - 2):
        prev_val = radial_bins[b - 1]
        curr_val = radial_bins[b]
        next_val = radial_bins[b + 1]
        if curr_val > 1.4 * prev_val and curr_val > 1.4 * next_val:
            peak_count += 1.0

    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0.0
    radial_anomaly_score = min(
        1.0, peak_count * 0.3 + (0.4 if high_freq_ratio > 0.35 else 0.0)
    )

    # Power-law decay slope: log(P(r)) ~ alpha * log(r)
    valid_r = np.arange(2, num_bins - 2)
    valid_p = radial_bins[valid_r]
    nonzero = valid_p > 1e-12
    if np.sum(nonzero) >= 5:
        log_r = np.log(valid_r[nonzero])
        log_p = np.log(valid_p[nonzero])
        slope, _ = np.polyfit(log_r, log_p, 1)
    else:
        slope = -2.0

    return {
        "fft_high_freq_energy_ratio": float(high_freq_ratio),
        "fft_spectral_peak_count": float(min(peak_count, 10.0) / 10.0),
        "fft_radial_anomaly_score": float(radial_anomaly_score),
        "fft_spectral_decay_slope": float(np.clip(slope, -10.0, 2.0)),
    }


def compute_8x8_dct_analysis(gray: np.ndarray) -> dict[str, float]:
    """Compute 8x8 block-wise 2D-DCT AC high-frequency energy distribution."""
    h, w = gray.shape
    blocks_y = h // 8
    blocks_x = w // 8
    total_blocks = blocks_y * blocks_x
    if total_blocks == 0:
        return {
            "dct_mean_high_freq_energy": 0.0,
            "dct_ac_energy_variance": 0.0,
            "dct_anomaly_score": 0.0,
        }

    # Precompute 8x8 2D-DCT basis matrix
    basis = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for u in range(8):
        cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
        for v in range(8):
            cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
            scale = 0.25 * cu * cv
            for y in range(8):
                cos_y = math.cos((2 * y + 1) * v * math.pi / 16.0)
                for x in range(8):
                    cos_x = math.cos((2 * x + 1) * u * math.pi / 16.0)
                    basis[u, v, y, x] = scale * cos_x * cos_y

    high_freq_mask = np.zeros((8, 8), dtype=bool)
    for u in range(8):
        for v in range(8):
            if u + v >= 6:
                high_freq_mask[u, v] = True

    cropped = gray[: blocks_y * 8, : blocks_x * 8]
    # Reshape into blocks of shape (blocks_y, blocks_x, 8, 8)
    blocks = cropped.reshape(blocks_y, 8, blocks_x, 8).transpose(0, 2, 1, 3)

    # Compute high frequency energy per block
    # blocks: (i, j, y, x), basis: (u, v, y, x) -> coeffs: (i, j, u, v)
    dct_coeffs = np.einsum("ijyx,uvyx->ijuv", blocks, basis)
    high_freq_energies = np.sum(np.abs(dct_coeffs[:, :, high_freq_mask]), axis=-1)

    mean_high = float(np.mean(high_freq_energies))
    variance = float(np.var(high_freq_energies))

    normalized_var = math.sqrt(variance) / (mean_high + 1e-4)
    anomaly_score = float(np.clip(normalized_var * 0.8, 0.0, 1.0))

    return {
        "dct_mean_high_freq_energy": mean_high,
        "dct_ac_energy_variance": variance,
        "dct_anomaly_score": anomaly_score,
    }


def compute_noise_residual_analysis(
    gray: np.ndarray, grid_cols: int = 6, grid_rows: int = 6
) -> dict[str, float]:
    """Extract noise residual via 3x3 Laplacian and measure spatial inconsistency."""
    h, w = gray.shape
    if h < grid_rows * 2 or w < grid_cols * 2:
        return {
            "noise_global_level": 0.0,
            "noise_max_to_min_ratio": 1.0,
            "noise_std_of_variances": 0.0,
            "noise_coef_variation": 0.0,
            "noise_inconsistency_score": 0.0,
        }

    # 3x3 discrete Laplacian filter
    laplacian_kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    residual = np.abs(convolve2d(gray, laplacian_kernel, mode="same", boundary="symm"))

    patch_h = h // grid_rows
    patch_w = w // grid_cols
    patch_variances: list[float] = []

    for gy in range(grid_rows):
        y0 = gy * patch_h
        y1 = (gy + 1) * patch_h if gy < grid_rows - 1 else h
        for gx in range(grid_cols):
            x0 = gx * patch_w
            x1 = (gx + 1) * patch_w if gx < grid_cols - 1 else w
            patch = residual[y0:y1, x0:x1]
            patch_variances.append(float(np.mean(patch)))

    var_arr = np.asarray(patch_variances, dtype=np.float64)
    avg_var = float(np.mean(var_arr))
    min_var = float(np.min(var_arr))
    max_var = float(np.max(var_arr))

    max_to_min_ratio = max_var / max(min_var, 1e-4) if max_var > 1e-3 else 1.0
    std_var = float(np.std(var_arr))
    coef_var = std_var / (avg_var + 1e-4)
    inconsistency_score = float(np.clip((coef_var - 0.25) / 0.75, 0.0, 1.0))

    return {
        "noise_global_level": avg_var,
        "noise_max_to_min_ratio": float(min(max_to_min_ratio, 50.0) / 50.0),  # normalized
        "noise_std_of_variances": std_var,
        "noise_coef_variation": float(min(coef_var, 5.0)),
        "noise_inconsistency_score": inconsistency_score,
    }


def compute_jpeg_block_grid_analysis(gray: np.ndarray) -> dict[str, float]:
    """Measure 8x8 block boundary gradient vs interior gradient ratio."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return {
            "jpeg_boundary_difference_ratio": 1.0,
            "jpeg_periodic_grid_strength": 0.0,
            "jpeg_artifact_score": 0.0,
        }

    # Horizontal gradient differences
    h_diff = np.abs(gray[:, 1:] - gray[:, :-1])  # (h, w-1)
    x_indices = np.arange(1, w)
    boundary_x = (x_indices % 8) == 0
    boundary_h = h_diff[:, boundary_x]
    interior_h = h_diff[:, ~boundary_x]

    # Vertical gradient differences
    v_diff = np.abs(gray[1:, :] - gray[:-1, :])  # (h-1, w)
    y_indices = np.arange(1, h)
    boundary_y = (y_indices % 8) == 0
    boundary_v = v_diff[boundary_y, :]
    interior_v = v_diff[~boundary_y, :]

    total_boundary_sum = float(np.sum(boundary_h)) + float(np.sum(boundary_v))
    total_boundary_count = boundary_h.size + boundary_v.size

    total_interior_sum = float(np.sum(interior_h)) + float(np.sum(interior_v))
    total_interior_count = interior_h.size + interior_v.size

    avg_boundary = total_boundary_sum / max(total_boundary_count, 1)
    avg_interior = total_interior_sum / max(total_interior_count, 1)

    ratio = avg_boundary / max(avg_interior, 1e-3)
    periodic_strength = max(0.0, ratio - 1.0)
    jpeg_artifact_score = float(np.clip(periodic_strength * 2.0, 0.0, 1.0))

    return {
        "jpeg_boundary_difference_ratio": float(min(ratio, 5.0)),
        "jpeg_periodic_grid_strength": float(min(periodic_strength, 4.0)),
        "jpeg_artifact_score": jpeg_artifact_score,
    }


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute overall variance of the 3x3 Laplacian operator."""
    kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    lap = convolve2d(gray, kernel, mode="valid")
    return float(np.var(lap))


def extract_dsp_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract all 16 DSP features into a single 1D numpy array of shape (16,)."""
    gray = to_grayscale_float(image)

    fft_res = compute_2d_fft_analysis(gray)
    dct_res = compute_8x8_dct_analysis(gray)
    noise_res = compute_noise_residual_analysis(gray)
    jpeg_res = compute_jpeg_block_grid_analysis(gray)
    lap_var = compute_laplacian_variance(gray)

    features = [
        fft_res["fft_high_freq_energy_ratio"],
        fft_res["fft_spectral_peak_count"],
        fft_res["fft_radial_anomaly_score"],
        fft_res["fft_spectral_decay_slope"],
        dct_res["dct_mean_high_freq_energy"],
        dct_res["dct_ac_energy_variance"],
        dct_res["dct_anomaly_score"],
        noise_res["noise_global_level"],
        noise_res["noise_max_to_min_ratio"],
        noise_res["noise_std_of_variances"],
        noise_res["noise_coef_variation"],
        noise_res["noise_inconsistency_score"],
        jpeg_res["jpeg_boundary_difference_ratio"],
        jpeg_res["jpeg_periodic_grid_strength"],
        jpeg_res["jpeg_artifact_score"],
        lap_var,
    ]

    arr = np.asarray(features, dtype=np.float32)
    # Ensure all values are finite and valid
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
