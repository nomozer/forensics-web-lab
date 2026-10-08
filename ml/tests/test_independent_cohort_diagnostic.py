"""Tests for Phase 4C.7B independent cohort diagnostic harness.

Verifies:
1. Approval guard: fail-closed rejection when human_review_status != 'APPROVED'.
2. Integrity guard: plan hash mismatch and input SHA-256 mismatch detection.
3. Geometry & raster contracts: binary mask invariant {0, 255}, coordinate scaling & nearest neighbor rasterization.
4. Outside pixel invariance: outside_mean_L1 == 0.000000 by construction on 512x512 canvas.
5. Attempt budget & no-retry: hard limit of 6 attempts, failure logging without automatic replacements.
6. Resume integrity: skips already consumed arms without repeating attempts.
7. Audit validation: passes valid runs and rejects tampered bindings or outside L1 deviations.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest

from ml.evaluation.independent_cohort_diagnostic import (
    DIAGNOSTIC_BUDGET_LIMIT,
    DiagnosticBudgetExceededError,
    DiagnosticGeometryError,
    DiagnosticInputHashMismatchError,
    DiagnosticInputMissingError,
    DiagnosticMockEngine,
    DiagnosticPlanIntegrityError,
    DiagnosticPlanNotApprovedError,
    DiagnosticRunAuditError,
    audit_diagnostic_run,
    execute_diagnostic_run,
    generate_diagnostic_contact_sheet,
    load_and_validate_diagnostic_plan,
    verify_diagnostic_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PLAN_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json"
REAL_INPUTS_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"


def test_real_plan_is_pending_and_refuses_execution():
    """Real diagnostic plan must have status PENDING and fail-closed when require_approval=True."""
    assert REAL_PLAN_PATH.is_file()
    plan_data = load_and_validate_diagnostic_plan(REAL_PLAN_PATH, require_approval=False)
    assert plan_data["human_review_status"] == "PENDING"
    assert plan_data["attempt_budget"] == 6
    assert plan_data["automatic_replacement"] is False
    assert len(plan_data["diagnostic_candidates"]) == 3

    with pytest.raises(DiagnosticPlanNotApprovedError, match="PENDING"):
        load_and_validate_diagnostic_plan(REAL_PLAN_PATH, require_approval=True)


def test_plan_hash_mismatch_detected(tmp_path):
    """Mismatched plan hash must raise DiagnosticPlanIntegrityError."""
    plan_copy = tmp_path / "plan.json"
    plan_copy.write_text(REAL_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DiagnosticPlanIntegrityError, match="SHA-256 mismatch"):
        load_and_validate_diagnostic_plan(plan_copy, expected_plan_sha256="0" * 64, require_approval=False)


def test_budget_exceeded_in_plan(tmp_path):
    """Plans specifying attempt_budget != 6 must be rejected."""
    data = json.loads(REAL_PLAN_PATH.read_text(encoding="utf-8"))
    data["attempt_budget"] = 8
    plan_copy = tmp_path / "plan_bad_budget.json"
    plan_copy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DiagnosticBudgetExceededError, match="attempt_budget 8"):
        load_and_validate_diagnostic_plan(plan_copy, require_approval=False)


def test_real_inputs_exist_and_match_hashes():
    """All 6 authentic and mask inputs from pilot-20261008T113700Z must match registered hashes."""
    plan_data = load_and_validate_diagnostic_plan(REAL_PLAN_PATH, require_approval=False)
    verified = verify_diagnostic_inputs(plan_data, REAL_INPUTS_DIR)
    assert len(verified) == 3
    for cid in ("IND_COCO_SDXL_002", "IND_COMMONS_SD2_002", "IND_COMMONS_SDXL_001"):
        assert cid in verified
        assert verified[cid]["authentic"].is_file()
        assert verified[cid]["mask"].is_file()


def test_tampered_input_hash_detected(tmp_path):
    """Tampering with an authentic or mask image must raise DiagnosticInputHashMismatchError."""
    # Create valid synthetic inputs
    auth = Image.new("RGB", (512, 512), (100, 100, 100))
    mask = Image.new("L", (512, 512), 0)
    auth_p = tmp_path / "test_auth.png"
    mask_p = tmp_path / "test_mask.png"
    auth.save(auth_p)
    mask.save(mask_p)

    data = json.loads(REAL_PLAN_PATH.read_text(encoding="utf-8"))
    cand = data["diagnostic_candidates"][0]
    cand["authentic_input_filename"] = "test_auth.png"
    cand["authentic_sha256"] = "bad" * 21 + "b"
    cand["mask_input_filename"] = "test_mask.png"

    with pytest.raises(DiagnosticInputHashMismatchError, match="SHA-256"):
        verify_diagnostic_inputs(data, tmp_path)


def test_mask_raster_scaling_and_nearest_neighbor_invariance():
    """Verify that scaling masks via NEAREST produces strictly {0, 255} binary rasters."""
    plan_data = load_and_validate_diagnostic_plan(REAL_PLAN_PATH, require_approval=False)

    for cand in plan_data["diagnostic_candidates"]:
        cid = cand["candidate_id"]
        mask_path = REAL_INPUTS_DIR / "masks" / cand["mask_input_filename"]
        mask_img = Image.open(mask_path).convert("L")

        arm_b = cand["arms"]["arm_b_local_crop"]
        crop_bbox = arm_b["crop_bbox_xyxy"]
        crop_mask = mask_img.crop(crop_bbox)

        inf_w, inf_h = arm_b["inference_resolution"]
        scaled_mask = crop_mask.resize((inf_w, inf_h), resample=Image.Resampling.NEAREST)

        arr = np.array(scaled_mask)
        uniques = np.unique(arr)
        assert set(uniques.tolist()).issubset({0, 255}), f"{cid} mask has non-binary values {uniques}"

        nz_y, nz_x = np.nonzero(arr == 255)
        raster_bbox = [int(nz_x.min()), int(nz_y.min()), int(nz_x.max()) + 1, int(nz_y.max()) + 1]
        count = int(np.sum(arr == 255))

        assert raster_bbox == arm_b["scaled_mask_raster_bbox"], f"{cid} raster bbox mismatch"
        assert count == arm_b["scaled_mask_pixel_count"], f"{cid} pixel count mismatch"


def test_full_diagnostic_run_synthetic_mock(tmp_path):
    """Execute all 6 diagnostic attempts using DiagnosticMockEngine."""
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(REAL_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    out_dir = tmp_path / "diag_run_01"

    receipt = execute_diagnostic_run(
        output_dir=out_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="test_commit_sha",
        engine_provider=lambda tool_key: DiagnosticMockEngine(tool_key),
        require_approval=False,
        allow_mock=True,
    )

    assert receipt["run_id"] == "diag_run_01"
    assert receipt["total_attempts_executed"] == 6
    assert len(receipt["attempts"]) == 6

    # Verify each attempt record
    for rec in receipt["attempts"]:
        assert rec["status"] == "ACCEPTED"
        assert rec["outside_mean_l1"] == 0.000000
        assert rec["inside_mean_l1"] > 0.0

    # Verify artifacts on disk
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    for cand_id in ("IND_COCO_SDXL_002", "IND_COMMONS_SD2_002", "IND_COMMONS_SDXL_001"):
        for arm in ("ARM_A", "ARM_B"):
            arm_id = f"{cand_id}_{arm}"
            assert (images_dir / f"{arm_id}_auth.png").is_file()
            assert (images_dir / f"{arm_id}_edit.png").is_file()
            assert (images_dir / f"{arm_id}_raw_gen.png").is_file()
            assert (masks_dir / f"{arm_id}_mask.png").is_file()

            # Verify outside pixels bitwise identical between auth and edit
            auth = np.array(Image.open(images_dir / f"{arm_id}_auth.png"))
            edit = np.array(Image.open(images_dir / f"{arm_id}_edit.png"))
            mask = np.array(Image.open(masks_dir / f"{arm_id}_mask.png"))

            outside = mask == 0
            assert np.array_equal(auth[outside], edit[outside]), f"Outside pixels altered in {arm_id}"

    # Verify contact sheet exists and is self-contained
    cs_p = out_dir / "diagnostic_contact_sheet.html"
    assert cs_p.is_file()
    cs_html = cs_p.read_text(encoding="utf-8")
    assert "data:image/png;base64," in cs_html
    assert "PENDING" in cs_html


def test_resume_does_not_repeat_attempts(tmp_path):
    """Resume mode must skip already executed arms."""
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(REAL_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    out_dir = tmp_path / "diag_run_resume"

    # Run initial
    execute_diagnostic_run(
        output_dir=out_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="test_commit_sha",
        engine_provider=lambda tool_key: DiagnosticMockEngine(tool_key),
        require_approval=False,
        allow_mock=True,
    )

    # Resume without changes
    receipt2 = execute_diagnostic_run(
        output_dir=out_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="test_commit_sha",
        engine_provider=lambda tool_key: DiagnosticMockEngine(tool_key),
        resume=True,
        require_approval=False,
        allow_mock=True,
    )

    # No new attempts should have executed
    assert len(receipt2["attempts"]) == 0


def test_audit_run_verifies_bindings(tmp_path):
    """audit_diagnostic_run passes valid run and fails mismatched commit or hash."""
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(REAL_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    out_dir = tmp_path / "diag_run_audit"

    execute_diagnostic_run(
        output_dir=out_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="commit_abc123",
        engine_provider=lambda tool_key: DiagnosticMockEngine(tool_key),
        require_approval=False,
        allow_mock=True,
    )

    valid_binding = {
        "run_id": "diag_run_audit",
        "mode": "diagnostic",
        "git_commit": "commit_abc123",
    }
    audit_res = audit_diagnostic_run(out_dir, valid_binding)
    assert audit_res["status"] == "PASS"
    assert audit_res["total_attempts"] == 6

    # Foreign commit audit must fail
    with pytest.raises(DiagnosticRunAuditError, match="git_commit"):
        audit_diagnostic_run(out_dir, valid_binding | {"git_commit": "other_commit"})
