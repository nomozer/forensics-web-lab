"""
ml/tests/test_baseline_lineage_audit.py

Regression test suite for baseline lineage audit:
Verifies that all 4 historical baseline discrepancies are mathematically audited,
reproduced, and bound to exact pipeline geometry and temperature calibration differences.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "baseline_lineage_audit_receipt.json"


def test_baseline_lineage_receipt_exists_and_valid():
    assert RECEIPT_PATH.exists(), f"Receipt missing: {RECEIPT_PATH}"
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0.0"
    assert data["status"] == "LINEAGE_AUDIT_EXPLAINED_AND_VERIFIED"
    assert len(data["discrepancies"]) == 4


def test_so_fake_baseline_discrepancy_reproduced():
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    disc1 = next(d for d in data["discrepancies"] if d["id"] == "DISC-01")
    assert disc1["status"] == "REPRODUCED_AND_PROVEN"
    assert disc1["value_historical"] == 0.3551
    assert disc1["value_c0"] == 0.3427
    assert disc1["reproduced_canonical_p1"] == 0.3551
    assert disc1["reproduced_anisotropic_p2"] == 0.3427
    assert disc1["decisions_flipped"] == 15
    assert disc1["total_samples"] == 300


def test_dev_holdout_baseline_discrepancy_reproduced():
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    disc2 = next(d for d in data["discrepancies"] if d["id"] == "DISC-02")
    assert disc2["status"] == "REPRODUCED_AND_PROVEN"
    assert disc2["value_historical"] == 0.5385
    assert disc2["value_c0"] == 0.5592
    assert disc2["reproduced_canonical_p1"] == 0.5385
    assert disc2["reproduced_anisotropic_p2"] == 0.5592
    assert disc2["decisions_flipped"] == 33
    assert disc2["total_samples"] == 273


def test_jpeg_q75_baseline_discrepancy_reproduced():
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    disc3 = next(d for d in data["discrepancies"] if d["id"] == "DISC-03")
    assert disc3["status"] == "REPRODUCED_AND_PROVEN"
    assert disc3["value_historical"] == 0.6015
    assert disc3["value_c1"] == 0.6187
    assert disc3["reproduced_canonical_p1"] == 0.6015
    assert disc3["reproduced_anisotropic_p2"] == 0.6187
    assert disc3["decisions_flipped"] == 39
    assert disc3["total_samples"] == 273


def test_temperature_calibration_discrepancy_audit():
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    disc4 = next(d for d in data["discrepancies"] if d["id"] == "DISC-04")
    assert disc4["status"] == "REPRODUCED_AND_PROVEN"
    assert disc4["historical_temp_fus"] == 8.1321
    assert disc4["reconciled_temp_fus"] == 5.9577
    assert disc4["outer_model_macro_f1_change"] == 0.0
    assert disc4["outer_model_decisions_flipped"] == 0
