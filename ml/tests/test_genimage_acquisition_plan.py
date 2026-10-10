"""
Tests for GenImage Acquisition Plan & Adapter
=============================================
Verifies:
1. Plan complies with JSON schema requirements.
2. Scientific constraints (approval status pending, license track research-only).
3. Adapter inspection dry-run behavior and receipt generation.
"""

import json
from pathlib import Path
import pytest

from scripts.research.acquire_genimage_candidate_sample import run_acquisition_inspection

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLAN_PATH = REPO_ROOT / "datasets" / "acquisition-plans" / "pilot-c-genimage-fully-generated.v1.json"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "genimage_acquisition_dryrun_receipt.json"


def test_genimage_plan_structure_and_constraints():
    assert PLAN_PATH.exists()
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    assert plan["planId"] == "pilot-c-genimage-fully-generated"
    assert plan["datasetId"] == "genimage"
    assert plan["approvalStatus"] == "pending-user-approval"
    assert plan["license"]["track"] == "research-only"
    assert "quarantined-to-research-track" in plan["license"]["derivativeWeightsPolicy"]

    comps = plan["components"]
    assert len(comps) == 1
    c = comps[0]
    assert c["assignedLabel"] == "fully_generated"
    assert c["semanticRole"] == "fully_generated_diffusion_text2image"


def test_acquisition_inspection_dryrun_receipt():
    receipt = run_acquisition_inspection(dry_run=True)
    assert RECEIPT_PATH.exists()
    assert receipt["plan_id"] == "pilot-c-genimage-fully-generated"
    assert receipt["approval_status"] == "pending-user-approval"
    assert receipt["is_approved_by_user"] is False
    assert receipt["status"] == "PASS_INSPECTION_BLOCKED_PENDING_APPROVAL"
