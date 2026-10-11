"""Local Quadrant DSP Feature Extraction Module.

Implements non-overlapping quadrant extraction from 512x512 canvas and 48-D statistical
summaries (mean, population std ddof=0, max) per Protocol Amendment v2.2.0.

Feature vector contract:
  - Input: 512x512 RGB canvas.
  - Quadrants: Four 256x256 non-overlapping tiles.
  - Quadrant features: 16-D canonical DSP per tile -> (4, 16) matrix.
  - Summaries: Mean (16-d), Population Std ddof=0 (16-d), Max (16-d) -> 48-D float32.
  - Concatenation order: Visual (576-d) + Global DSP (16-d) + Local DSP (48-d) = 640-D.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from PIL import Image

from ml.training.dsp_features import extract_dsp_features, DSP_FEATURE_DIM

QUADRANT_BOXES: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 0, 256, 256),      # Q0: Top-Left
    (256, 0, 512, 256),    # Q1: Top-Right
    (0, 256, 256, 512),    # Q2: Bottom-Left
    (256, 256, 512, 512),  # Q3: Bottom-Right
)

LOCAL_DSP_SUMMARY_DIM = 48  # 16 mean + 16 std + 16 max
COMBINED_LOCAL_FEATURE_DIM = 640  # 576 visual + 16 global DSP + 48 local DSP


def extract_local_dsp_summaries(image_512: Image.Image | np.ndarray) -> np.ndarray:
    """Extracts 48-D local DSP statistical summaries across 4 non-overlapping quadrants.

    Args:
        image_512: PIL Image or numpy array representing the 512x512 RGB canvas.

    Returns:
        np.ndarray of shape (48,), float32:
            - [0:16]   : Mean of each DSP feature across the 4 quadrants.
            - [16:32]  : Population standard deviation (ddof=0) of each feature.
            - [32:48]  : Maximum value of each feature across the 4 quadrants.
    """
    if isinstance(image_512, np.ndarray):
        if image_512.dtype != np.uint8 and image_512.max() <= 1.0:
            image_512 = (image_512 * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(image_512).convert("RGB")
    elif isinstance(image_512, Image.Image):
        pil_img = image_512.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_512)}")

    if pil_img.size != (512, 512):
        pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)

    quad_features: List[np.ndarray] = []
    for box in QUADRANT_BOXES:
        quad_tile = pil_img.crop(box)
        dsp_16 = extract_dsp_features(quad_tile)
        quad_features.append(dsp_16)

    d_mat = np.stack(quad_features, axis=0)  # Shape (4, 16)
    assert d_mat.shape == (4, DSP_FEATURE_DIM), f"Unexpected quadrant matrix shape: {d_mat.shape}"

    mean_vec = np.mean(d_mat, axis=0)
    std_vec = np.std(d_mat, axis=0, ddof=0)  # Population std as specified in protocol
    max_vec = np.max(d_mat, axis=0)

    summary_48 = np.concatenate([mean_vec, std_vec, max_vec], axis=0).astype(np.float32)
    assert summary_48.shape == (LOCAL_DSP_SUMMARY_DIM,), f"Unexpected summary shape: {summary_48.shape}"
    return summary_48
