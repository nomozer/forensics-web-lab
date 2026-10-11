"""
ml/tests/test_document_observation_pilot.py

Unit and contract tests for the Document AI-Tampering Observation Geometry Pilot (P0 vs P1 vs P2).
Tests transform mathematics, mask coverage retention, 5-view probability aggregation,
and protocol contracts.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


def test_protocol_document_pilot_invariants():
    """Verify that PROTOCOL_DOCUMENT_PILOT_V1.0.0 exists and defines mandatory sections."""
    proto_path = Path("research/evidence/document_pilot/PROTOCOL_DOCUMENT_PILOT_V1.0.0.md")
    assert proto_path.exists(), "Protocol document must exist"
    content = proto_path.read_text(encoding="utf-8")
    assert "PROTOCOL_DOCUMENT_PILOT_V1.0.0" in content
    assert "P0: Canonical Square Center-Crop" in content
    assert "P1: Orientation-Correct Letterbox Canvas" in content
    assert "P2: Global + 4 Fixed Quadrants" in content
    assert "s_{\\text{AI}}" in content
    assert "locked_models_registry.json" in content


def test_observation_transforms_geometry():
    """Verify geometric dimensions and padding values for P0, P1, and P2."""
    from scripts.research.run_document_observation_pilot import (
        transform_p0_center_crop,
        transform_p1_letterbox,
        transform_p2_multitile,
    )

    # Synthetic document: vertical orientation (e.g. A4 / invoice: 1000 x 600)
    h, w = 1000, 600
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    mask = Image.new("L", (w, h), color=0)
    # Put a small tampered region near the top: [y: 100..150, x: 200..300]
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[100:150, 200:300] = 255
    mask = Image.fromarray(mask_np)

    # 1. P0 (Center Crop: square min(w, h) = 600, y_range: [200, 800])
    p0_img_512, p0_img_224, p0_mask_512, p0_retention = transform_p0_center_crop(img, mask)
    assert p0_img_512.size == (512, 512)
    assert p0_img_224.size == (224, 224)
    assert p0_mask_512.size == (512, 512)
    # The mask at y=100..150 is outside the center crop [200:800], so retention should be 0.0!
    assert p0_retention == 0.0, "Tampered text at y=100 should be completely cut off by P0 center crop"

    # 2. P1 (Letterbox: scale = 512 / 1000 = 0.512, new w = 307, new h = 512, pad on left/right)
    p1_img_512, p1_img_224, p1_mask_512, p1_retention = transform_p1_letterbox(img, mask)
    assert p1_img_512.size == (512, 512)
    assert p1_img_224.size == (224, 224)
    assert p1_mask_512.size == (512, 512)
    # P1 preserves 100% of the document content
    assert p1_retention == 1.0, "P1 letterbox must retain 100% of the tampered region"

    # Check neutral padding color is 128
    p1_np = np.array(p1_img_512)
    # At top-left outside the document canvas (x=10, y=10) it should be padded
    assert p1_np[10, 10, 0] == 128
    assert p1_np[10, 10, 1] == 128
    assert p1_np[10, 10, 2] == 128

    # 3. P2 (Multi-Tile: 5 views)
    views = transform_p2_multitile(img, mask)
    assert len(views) == 5, "P2 must yield exactly 5 views (global + 4 quadrants)"
    for v_idx, v in enumerate(views):
        assert v["img_512"].size == (512, 512)
        assert v["img_224"].size == (224, 224)
        assert "mask_retention" in v


def test_five_view_probability_aggregation():
    """Verify arithmetic mean aggregation across 5 views and binary decision threshold."""
    from scripts.research.run_document_observation_pilot import aggregate_multiview_probabilities

    # 5 views of probabilities [p_auth, p_edit, p_gen]
    view_probs = [
        np.array([0.70, 0.20, 0.10]),  # Global view: mostly sees authentic background
        np.array([0.10, 0.80, 0.10]),  # Quadrant with tampered text: strong edited signal!
        np.array([0.80, 0.10, 0.10]),  # Quadrant 2: authentic
        np.array([0.75, 0.15, 0.10]),  # Quadrant 3: authentic
        np.array([0.65, 0.25, 0.10]),  # Quadrant 4: authentic
    ]

    mean_probs, binary_score, decision = aggregate_multiview_probabilities(view_probs, threshold=0.5)

    # Expected mean:
    # auth: (0.70 + 0.10 + 0.80 + 0.75 + 0.65) / 5 = 3.00 / 5 = 0.60
    # edit: (0.20 + 0.80 + 0.10 + 0.15 + 0.25) / 5 = 1.50 / 5 = 0.30
    # gen:  (0.10 + 0.10 + 0.10 + 0.10 + 0.10) / 5 = 0.50 / 5 = 0.10
    np.testing.assert_allclose(mean_probs, [0.60, 0.30, 0.10], atol=1e-5)
    # binary score: p_edit + p_gen = 0.30 + 0.10 = 0.40
    assert abs(binary_score - 0.40) < 1e-5
    assert decision == "authentic"

    # Test when tampered quadrant has even stronger signal
    view_probs_strong = [
        np.array([0.40, 0.45, 0.15]),
        np.array([0.05, 0.90, 0.05]),
        np.array([0.50, 0.35, 0.15]),
        np.array([0.55, 0.35, 0.10]),
        np.array([0.50, 0.40, 0.10]),
    ]
    m_probs2, b_score2, dec2 = aggregate_multiview_probabilities(view_probs_strong, threshold=0.5)
    # mean_edit = (0.45 + 0.90 + 0.35 + 0.35 + 0.40)/5 = 2.45 / 5 = 0.49
    # mean_gen = (0.15 + 0.05 + 0.15 + 0.10 + 0.10)/5 = 0.55 / 5 = 0.11
    # binary score = 0.49 + 0.11 = 0.60 >= 0.50
    assert b_score2 >= 0.50
    assert dec2 == "ai_edited"
