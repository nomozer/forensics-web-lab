"""Unit tests for the 2x2 Factorial Source x Compression Diagnosis.

Validates:
1. Manifest structure, provenance, balanced N=182, and disk file existence.
2. Predictions validity (5184 records, probabilities sum to 1, no NaNs).
3. Receipt structure, 6 models presence, endpoints correctness.
4. Interaction math (difference-in-differences) and bootstrap consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np


MANIFEST_PATH = Path("research/evidence/three_class_preparation/factorial_2x2_cohort_manifest.json")
RECEIPT_PATH = Path("research/evidence/three_class_preparation/factorial_source_compression_receipt.json")
PREDICTIONS_PATH = Path("research/evidence/three_class_preparation/factorial_source_compression_predictions.json")


def test_factorial_manifest_validity():
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("schema_version") == "1.0.0"
    assert manifest.get("manifest_type") == "FACTORIAL_SOURCE_COMPRESSION_COHORT"
    balanced = manifest.get("balanced_cohort", {})
    coco_samples = balanced.get("coco_samples", [])
    imagenet_samples = balanced.get("imagenet_samples", [])
    assert len(coco_samples) == 91
    assert len(imagenet_samples) == 91
    total_balanced = coco_samples + imagenet_samples
    assert len(total_balanced) == 182

    for s in total_balanced:
        assert s["ground_truth_label"] == "authentic"
        assert Path(s["relative_path"]).exists(), f"File missing on disk: {s['relative_path']}"

    full_pool = manifest.get("full_sensitivity_cohort", {})
    assert len(full_pool.get("coco_samples", [])) == 91
    assert len(full_pool.get("imagenet_samples", [])) == 341


def test_factorial_receipt_structure_and_models():
    assert RECEIPT_PATH.exists(), f"Receipt file missing: {RECEIPT_PATH}"
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt.get("schema_version") == "1.0.0"
    assert receipt.get("status") == "FACTORIAL_DIAGNOSIS_SUCCESS"
    models_evaluated = receipt.get("models_evaluated", [])
    expected_models = ["A0-global", "A0-local", "A1-global", "A1-local", "A0-visual", "A1-visual"]
    assert set(models_evaluated) == set(expected_models)

    results_by_model = receipt.get("results_by_model", {})
    assert len(results_by_model) == 6

    for model_key in expected_models:
        assert model_key in results_by_model
        m_data = results_by_model[model_key]
        balanced_design = m_data.get("balanced_2x2_design", {})
        cells = balanced_design.get("cells", {})
        for cond_key in ["coco_original", "coco_jpeg_q75", "imagenet_original", "imagenet_jpeg_q75"]:
            assert cond_key in cells
            assert "false_positive_ai_rate" in cells[cond_key]
            assert 0.0 <= cells[cond_key]["false_positive_ai_rate"] <= 1.0
            assert "mean_p_authentic" in cells[cond_key]

        # Verify interaction calculations
        inter = balanced_design.get("interaction_effect", {})
        b_res = inter.get("bootstrap_result", {})
        observed_interaction = b_res.get("observed_interaction")

        delta_coco = balanced_design["within_source_jpeg_effect"]["coco_delta_fp"]["observed_delta"]
        delta_imagenet = balanced_design["within_source_jpeg_effect"]["imagenet_delta_fp"]["observed_delta"]
        expected_diff = delta_imagenet - delta_coco
        # Allow tolerance for 4-decimal rounding
        assert abs(observed_interaction - expected_diff) < 1e-3, "Interaction difference-in-differences must match within rounding precision"

        ci = b_res.get("ci_95", [])
        assert len(ci) == 2
        assert ci[0] <= ci[1]


def test_factorial_predictions_validity():
    assert PREDICTIONS_PATH.exists(), f"Predictions file missing: {PREDICTIONS_PATH}"
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    assert isinstance(preds, list)
    assert len(preds) == 5184  # 432 images * 6 models * 2 conditions

    for record in preds[:200]:  # sample check
        assert "sample_id" in record
        assert "model_name" in record
        assert "condition" in record
        assert "probabilities" in record
        p = record["probabilities"]
        assert len(p) == 3
        assert abs(sum(p) - 1.0) < 1e-4
        assert not any(np.isnan(p))
