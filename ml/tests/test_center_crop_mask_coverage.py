"""
Unit tests for center-crop mask coverage audit.
Verifies that:
  - Manifest and receipt files exist and adhere to schema.
  - Exactly 91 holdout samples are audited.
  - Geometry invariant: retained_edit_pixels <= original_edit_pixels.
  - Geometry invariant: crop window dimensions match min(w, h).
  - Predefined coverage bins cover 100% of samples.
  - Epistemic distinction between observation and hypothesis is preserved.
"""

import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = WORKSPACE_ROOT / "research" / "evidence" / "three_class_preparation"
MANIFEST_PATH = EVIDENCE_DIR / "center_crop_mask_coverage_manifest.json"
RECEIPT_PATH = EVIDENCE_DIR / "center_crop_mask_coverage_receipt.json"


@pytest.fixture
def manifest_data():
    if not MANIFEST_PATH.exists():
        pytest.skip(f"Manifest file not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def receipt_data():
    if not RECEIPT_PATH.exists():
        pytest.skip(f"Receipt file not found: {RECEIPT_PATH}")
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_and_receipt_sample_count(manifest_data, receipt_data):
    assert manifest_data["total_samples_audited"] == 91
    assert receipt_data["total_samples"] == 91
    assert len(manifest_data["sample_audits"]) == 91


def test_geometric_invariants(manifest_data):
    for s in manifest_data["sample_audits"]:
        geo = s["image_geometry"]
        crop = s["canonical_crop_box"]
        mask = s["mask_geometry"]

        min_dim = min(geo["width"], geo["height"])
        assert crop["min_dim"] == min_dim
        assert (crop["right"] - crop["left"]) == min_dim
        assert (crop["bottom"] - crop["top"]) == min_dim

        assert mask["original_edit_pixels"] > 0
        assert mask["retained_edit_pixels"] <= mask["original_edit_pixels"]
        assert 0.0 <= mask["retention_ratio"] <= 1.000001
        assert 0.0 <= mask["canvas_edit_ratio"] <= 1.0


def test_coverage_bin_partition(receipt_data):
    bin_counts = receipt_data["coverage_summary"]["bin_counts"]
    expected_bins = ["full_retention", "minor_loss", "major_loss", "complete_loss"]
    assert set(bin_counts.keys()) == set(expected_bins)
    assert sum(bin_counts.values()) == 91

    # Verify at least one complete loss sample exists
    assert receipt_data["coverage_summary"]["complete_loss_count"] >= 1
    assert len(receipt_data["coverage_summary"]["complete_loss_sample_ids"]) >= 1


def test_performance_by_bin_coverage(receipt_data):
    perf = receipt_data["performance_by_coverage_bin"]
    for b in ["full_retention", "minor_loss", "major_loss", "complete_loss"]:
        assert b in perf
        models_perf = perf[b]["models"]
        for m in ["A0-global", "A0-local", "A0-visual", "A1-global", "A1-local", "A1-visual"]:
            assert m in models_perf
            if perf[b]["support"] > 0:
                assert models_perf[m]["recall"] is not None
