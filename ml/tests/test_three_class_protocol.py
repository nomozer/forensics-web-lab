"""
Tests for Three-Class Evaluation Protocol & Preflight Validator
===============================================================
Verifies:
1. YAML protocol configuration schema and invariants.
2. Dimensions and parity between baseline 1 (visual-only) and baseline 2 (visual + DSP).
3. Correctness of multiclass metrics: Macro-F1, Balanced Accuracy, Multiclass Brier, ECE.
4. Preflight receipt execution and blocking status verification.
"""

import json
from pathlib import Path
import pytest
import yaml

from scripts.research.preflight_three_class_protocol import (
    compute_confusion_matrix_and_f1,
    compute_multiclass_brier,
    compute_multiclass_ece,
    compute_shannon_entropy,
    run_preflight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROTOCOL_PATH = REPO_ROOT / "ml" / "configs" / "three_class_evaluation_protocol.yaml"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "protocol_preflight_receipt.json"


def test_protocol_yaml_schema():
    assert PROTOCOL_PATH.exists(), f"Missing protocol config at {PROTOCOL_PATH}"
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    assert proto["protocol_metadata"]["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V1"
    assert proto["protocol_metadata"]["status"] == "DRAFT_PENDING_ACQUISITION"
    assert proto["protocol_metadata"]["training_readiness"] == "DATA_BLOCKED_FOR_3CLASS"

    assert proto["class_mapping"]["num_classes"] == 3
    class_names = [c["name"] for c in proto["class_mapping"]["classes"]]
    assert class_names == ["authentic", "ai_edited", "fully_generated"]


def test_baselines_parity_and_dimensions():
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    b1 = proto["baselines"]["baseline_1_visual_only"]
    b2 = proto["baselines"]["baseline_2_visual_dsp"]

    assert b1["backbone"]["feature_dim"] == 1280
    assert b1["classification_head"]["input_dim"] == 1280
    assert b1["classification_head"]["output_dim"] == 3

    assert b2["backbone"]["feature_dim"] == 1280
    assert b2["dsp_extractor"]["feature_dim"] == 16
    assert b2["classification_head"]["input_dim"] == 1296
    assert b2["classification_head"]["output_dim"] == 3


def test_multiclass_metrics_math():
    # Synthetic perfect predictions
    labels = [0, 1, 2]
    probs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    brier = compute_multiclass_brier(probs, labels)
    assert brier == pytest.approx(0.0)

    ece = compute_multiclass_ece(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0)

    cm_res = compute_confusion_matrix_and_f1(probs, labels, num_classes=3)
    assert cm_res["macro_f1"] == pytest.approx(1.0)
    assert cm_res["balanced_accuracy"] == pytest.approx(1.0)
    assert cm_res["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    # Test entropy
    assert compute_shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert compute_shannon_entropy([1/3, 1/3, 1/3]) > 1.0


def test_preflight_execution_and_receipt():
    receipt = run_preflight()
    assert RECEIPT_PATH.exists()
    assert receipt["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V1"
    assert receipt["protocol_status"] == "DRAFT_PENDING_ACQUISITION"
    assert receipt["dataset_readiness_status"] == "DATA_BLOCKED_MISSING_FULLY_GENERATED"
    assert receipt["dry_run_metrics_simulation"]["metrics_computation_status"] == "VERIFIED_PASS"
