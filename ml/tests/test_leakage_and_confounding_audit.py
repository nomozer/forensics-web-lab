"""
Tests for Leakage & Confounding Audit
====================================
Verifies that:
1. Leakage report exists and has required sections.
2. Source groups preserve pairing between authentic and edited variants.
3. Zero cross-cohort leakage with sealed cohorts (TGIF N=400 and locked-test 343 sources).
4. Confounding factors (resolution, format, compression) are audited.
5. Overall summary verdict is LEAKAGE_AUDIT_PASS.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "leakage_and_confounding_report.json"


def test_leakage_report_exists_and_valid():
    assert REPORT_FILE.exists(), f"Report file missing: {REPORT_FILE}"
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "source_group_audit" in data
    assert "exact_duplicates" in data
    assert "cross_cohort_leakage" in data
    assert "confounding_factor_analysis" in data
    assert "summary_verdict" in data


def test_source_group_integrity():
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sga = data["source_group_audit"]
    assert sga["status"] == "PASS"
    assert sga["total_distinct_sources"] == 341
    assert sga["complete_authentic_edited_pairs"] == 341
    assert len(sga["unpaired_sources"]) == 0


def test_zero_cross_cohort_leakage():
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    ccl = data["cross_cohort_leakage"]
    assert ccl["status"] == "PASS"
    assert ccl["overlapping_exact_sha256_count"] == 0
    assert ccl["overlapping_source_ids_count"] == 0


def test_confounding_factors_audited():
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfa = data["confounding_factor_analysis"]
    assert "authentic" in cfa
    assert "ai_edited" in cfa
    for lbl in ["authentic", "ai_edited"]:
        info = cfa[lbl]
        assert info["sample_count"] == 341
        assert len(info["resolutions"]) > 0
        assert "file_size_bytes" in info

    cra = data["confounding_risk_assessment"]
    assert "risk_explanation" in cra


def test_summary_verdict_pass():
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary_verdict"] == "LEAKAGE_AUDIT_PASS"
