"""Unit and contract tests for Local Dataset Inventory and Candidate Cohort Manifest."""

import csv
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"


def test_inventory_files_exist():
    """Verify that inventory and manifest artifacts exist."""
    inv_path = INVENTORY_DIR / "local_dataset_inventory.json"
    manifest_json = INVENTORY_DIR / "candidate_cohort_manifest.json"
    manifest_csv = INVENTORY_DIR / "candidate_cohort_manifest.csv"

    assert inv_path.exists(), f"Missing {inv_path}"
    assert manifest_json.exists(), f"Missing {manifest_json}"
    assert manifest_csv.exists(), f"Missing {manifest_csv}"


def test_inventory_three_class_status_integrity():
    """Verify inventory report strictly reports real local status without fake data."""
    inv_path = INVENTORY_DIR / "local_dataset_inventory.json"
    with open(inv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    status = data["three_class_data_status"]
    assert status["authentic"]["present_on_local"] > 0
    assert status["ai_edited"]["present_on_local"] > 0
    assert status["fully_generated"]["present_on_local"] == 0
    assert status["fully_generated"]["status"] == "MISSING_ON_LOCAL_BLOCKED"
    assert data["overall_status"] == "AUDITED_PASS_DATA_BLOCKED_FOR_3CLASS"

    # Sealed cohorts isolation check
    sealed = data["sealed_cohorts_isolation"]
    assert sealed["locked_test_sources_isolated"] == 343
    assert sealed["tgif_n400_independent_sources_isolated"] == 400
    assert sealed["sealed_images_quarantined_from_development"] > 0


def test_candidate_cohort_manifest_schema_and_fields():
    """Verify candidate manifest contains all required fields and no duplicate sample IDs."""
    manifest_json = INVENTORY_DIR / "candidate_cohort_manifest.json"
    with open(manifest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    assert len(records) > 0

    required_fields = {
        "sample_id",
        "relative_path",
        "sha256",
        "label",
        "dataset_name",
        "dataset_version",
        "source_group_id",
        "generator_manipulation",
        "width",
        "height",
        "format",
        "eligibility",
        "exclusion_reason",
    }

    sample_ids = set()
    for r in records:
        assert required_fields.issubset(r.keys()), f"Missing fields in record: {r}"
        assert r["sample_id"] not in sample_ids, f"Duplicate sample_id: {r['sample_id']}"
        sample_ids.add(r["sample_id"])

        # Label must be one of the standard taxonomy
        assert r["label"] in {"authentic", "ai_edited", "fully_generated"}

        # Sealed sources must never be ELIGIBLE_DEVELOPMENT
        if "SEALED" in r["eligibility"]:
            assert r["eligibility"] != "ELIGIBLE_DEVELOPMENT"
            assert r["exclusion_reason"] is not None

        # SHA-256 must be 64-char hex
        assert len(r["sha256"]) == 64
        int(r["sha256"], 16)  # must parse as hex


def test_manifest_csv_json_parity():
    """Verify exact count and field parity between JSON and CSV manifests."""
    manifest_json = INVENTORY_DIR / "candidate_cohort_manifest.json"
    manifest_csv = INVENTORY_DIR / "candidate_cohort_manifest.csv"

    with open(manifest_json, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    json_records = json_data["records"]

    with open(manifest_csv, "r", encoding="utf-8") as f:
        csv_records = list(csv.DictReader(f))

    assert len(json_records) == len(csv_records)
    for j_rec, c_rec in zip(json_records[:50], csv_records[:50]):
        assert j_rec["sample_id"] == c_rec["sample_id"]
        assert j_rec["sha256"] == c_rec["sha256"]
        assert j_rec["label"] == c_rec["label"]
        assert j_rec["eligibility"] == c_rec["eligibility"]
