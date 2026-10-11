"""
Unit test suite for Three-Class Pilot artifacts, contracts, and evaluation receipts.
Validates:
1. Split locking contracts (750 train, 273 holdout, 3-class balance 1:1:1).
2. Zero leakage with Locked-Test (343 sources) and TGIF N=400 (400 sources).
3. Cached feature array shapes and numerical integrity.
4. Pilot evaluation receipt schema, metric boundaries, and bootstrap CI.
5. Diagnostic source control evaluation receipt consistency.
"""

import json
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"

SPLITS_JSON_PATH = EVIDENCE_DIR / "three_class_pilot_splits.json"
FEATURE_CACHE_PATH = REPO_ROOT / "data" / "research" / "cache" / "three_class_features.npz"
EVAL_RECEIPT_PATH = EVIDENCE_DIR / "three_class_pilot_evaluation_receipt.json"
CONTROL_RECEIPT_PATH = EVIDENCE_DIR / "source_control_evaluation_receipt.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_pilot_predictions.json"


def test_three_class_splits_locking_and_balance():
    assert SPLITS_JSON_PATH.exists(), f"Missing {SPLITS_JSON_PATH}"
    with open(SPLITS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V2"
    assert data["total_samples"] == 1023

    # Development train
    trn = data["development_train"]
    assert trn["total_samples"] == 750
    assert trn["authentic_samples"] == 250
    assert trn["ai_edited_samples"] == 250
    assert trn["fully_generated_samples"] == 250

    # Development holdout
    hld = data["development_holdout"]
    assert hld["total_samples"] == 273
    assert hld["authentic_samples"] == 91
    assert hld["ai_edited_samples"] == 91
    assert hld["fully_generated_samples"] == 91

    # Isolation audit
    iso = data["isolation_audit"]
    assert iso["locked_test_sources_checked"] == 343
    assert iso["locked_test_hashes_checked"] == 2970
    assert iso["locked_test_overlap"] == 0
    assert iso["tgif_n400_sources_checked"] == 400
    assert iso["tgif_n400_hashes_checked"] == 800
    assert iso["tgif_n400_overlap"] == 0
    assert iso["status"] == "LEAKAGE_AUDIT_PASS"


def test_three_class_cached_features_contract():
    if not FEATURE_CACHE_PATH.exists():
        pytest.skip(f"Feature cache not found: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH)
    assert data["X_vis_train"].shape == (750, 576)
    assert data["X_dsp_train"].shape == (750, 16)
    assert data["y_train"].shape == (750,)

    assert data["X_vis_holdout"].shape == (273, 576)
    assert data["X_dsp_holdout"].shape == (273, 16)
    assert data["y_holdout"].shape == (273,)

    assert data["X_vis_nature"].shape == (341, 576)
    assert data["X_dsp_nature"].shape == (341, 16)
    assert data["y_nature"].shape == (341,)

    # Check finite numbers (0 NaNs, 0 Infs)
    for k in ["X_vis_train", "X_dsp_train", "X_vis_holdout", "X_dsp_holdout", "X_vis_nature", "X_dsp_nature"]:
        assert np.isfinite(data[k]).all(), f"Non-finite values in {k}"


def test_three_class_pilot_evaluation_receipt():
    assert EVAL_RECEIPT_PATH.exists(), f"Missing {EVAL_RECEIPT_PATH}"
    with open(EVAL_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V2"
    assert receipt["amendment"] == "PROTOCOL_AMENDMENT_V2.0.0.md"

    baselines = receipt["baselines_evaluated"]
    b1 = baselines["baseline_1_visual_only"]
    b2 = baselines["baseline_2_visual_dsp"]

    assert b1["feature_dim"] == 576
    assert b2["feature_dim"] == 592

    # Check metric value ranges
    assert 0.0 <= b1["metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= b2["metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= b1["metrics"]["multiclass_brier_score"] <= 2.0
    assert 0.0 <= b2["metrics"]["multiclass_brier_score"] <= 2.0
    assert 0.0 <= b1["metrics"]["multiclass_top_class_ece"] <= 1.0
    assert 0.0 <= b2["metrics"]["multiclass_top_class_ece"] <= 1.0

    paired = receipt["paired_comparison"]
    delta = paired["delta_macro_f1_point_estimate"]
    ci = paired["ci_95_percentile"]
    assert ci[0] <= delta <= ci[1] or abs(delta - np.mean(ci)) < 0.1
    assert 0.0 <= paired["probability_positive_delta"] <= 1.0


def test_three_class_source_control_receipt():
    assert CONTROL_RECEIPT_PATH.exists(), f"Missing {CONTROL_RECEIPT_PATH}"
    with open(CONTROL_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["cohort"]["sample_count"] == 341
    res = receipt["diagnostic_results"]
    vis = res["visual_only"]
    fus = res["visual_dsp"]

    assert vis["total_samples"] == 341
    assert fus["total_samples"] == 341
    assert vis["correct_authentic"] + vis["false_positive_ai_total"] == 341
    assert fus["correct_authentic"] + fus["false_positive_ai_total"] == 341
    assert 0.0 <= vis["false_positive_ai_rate"] <= 1.0
    assert 0.0 <= fus["false_positive_ai_rate"] <= 1.0


def test_three_class_predictions_integrity():
    assert PREDICTIONS_PATH.exists(), f"Missing {PREDICTIONS_PATH}"
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    hld_preds = preds["holdout_predictions"]
    nat_preds = preds["nature_control_predictions"]

    assert len(hld_preds) == 273
    assert len(nat_preds) == 341

    for row in hld_preds:
        assert len(row["visual_only"]["probs"]) == 3
        assert len(row["visual_dsp"]["probs"]) == 3
        assert abs(sum(row["visual_only"]["probs"]) - 1.0) < 1e-4
        assert abs(sum(row["visual_dsp"]["probs"]) - 1.0) < 1e-4
