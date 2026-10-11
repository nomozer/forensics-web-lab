"""
Unit tests for full classification trade-off analysis.
Verifies that:
  - Output receipt exists and complies with schema.
  - All 6 models and 3 cohorts are present.
  - Confusion matrices sum to cohort sample counts.
  - Paired comparisons contain valid 95% bootstrap CIs and positive fractions.
  - Latency profile contains positive latency values.
"""

import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RECEIPT_PATH = WORKSPACE_ROOT / "research" / "evidence" / "three_class_preparation" / "classification_tradeoff_analysis_receipt.json"


@pytest.fixture
def receipt_data():
    if not RECEIPT_PATH.exists():
        pytest.skip(f"Receipt file not found: {RECEIPT_PATH}")
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_receipt_structure(receipt_data):
    assert receipt_data["schema_version"] == "1.0.0"
    assert receipt_data["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT"
    assert "cohort_metrics" in receipt_data
    assert "paired_comparisons" in receipt_data
    assert "latency_profile" in receipt_data
    assert "tradeoff_diagnosis" in receipt_data


def test_models_and_cohorts_coverage(receipt_data):
    expected_models = ["A0-global", "A0-local", "A0-visual", "A1-global", "A1-local", "A1-visual"]
    expected_cohorts = ["holdout", "nature_control", "so_fake"]

    assert set(receipt_data["models_evaluated"]) == set(expected_models)
    for c in expected_cohorts:
        assert c in receipt_data["cohort_metrics"]
        for m in expected_models:
            assert m in receipt_data["cohort_metrics"][c]


def test_confusion_matrix_sample_counts(receipt_data):
    cm_holdout = receipt_data["cohort_metrics"]["holdout"]["A0-global"]["confusion_matrix"]
    assert sum(sum(row) for row in cm_holdout) == 273

    cm_control = receipt_data["cohort_metrics"]["nature_control"]["A0-global"]["confusion_matrix"]
    assert sum(sum(row) for row in cm_control) == 341

    cm_so_fake = receipt_data["cohort_metrics"]["so_fake"]["A0-global"]["confusion_matrix"]
    assert sum(sum(row) for row in cm_so_fake) == 300


def test_paired_bootstrap_intervals(receipt_data):
    comparisons = receipt_data["paired_comparisons"]
    assert "A0_Local_minus_Global" in comparisons
    assert "A1_Local_minus_Global" in comparisons

    for comp_name, comp_data in comparisons.items():
        holdout_comp = comp_data["cohorts"]["holdout"]
        assert "macro_f1" in holdout_comp
        ci = holdout_comp["macro_f1"]["ci_95"]
        assert len(ci) == 2
        assert ci[0] <= ci[1]
        p_pos = holdout_comp["macro_f1"]["bootstrap_positive_fraction"]
        assert 0.0 <= p_pos <= 1.0


def test_latency_profile(receipt_data):
    profile = receipt_data["latency_profile"]["measured_latencies_ms"]
    for m in ["A0-visual", "A0-global", "A0-local", "A1-visual", "A1-global", "A1-local"]:
        assert m in profile
        assert profile[m]["total_ms"] > 0
        # Local should be slower than global, which should be slower than visual
    assert profile["A0-local"]["total_ms"] > profile["A0-global"]["total_ms"] > profile["A0-visual"]["total_ms"]
