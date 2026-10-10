"""Unit tests for Phase 4C.7B independent cohort calibration harness.

Verifies:
1. Approval guard: fail-closed rejection when human_review_status != 'APPROVED'.
2. Integrity guards: plan SHA-256 mismatch, invariant parameter drift, and input hash mismatch detection.
3. Pipeline transmission: prompt, negative_prompt, guidance_scale, seed, steps transmitted correctly to engine.
4. Budget & failure accounting: 2 attempts budget cap, failure counts towards budget, no retry.
5. Resume integrity: skips already consumed attempts without re-generating.
6. Mock safeguard: mock engine prohibited in production mode unless allow_mock=True.
7. Outside pixel invariance: outside_mean_l1 == 0.000000 strictly enforced and audited.
8. Telemetry & audit: comprehensive telemetry captured and run binding audited.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ml.evaluation.independent_cohort_calibration import (
    APPROVED_CALIBRATION_PLAN_SHA256,
    CALIBRATION_BUDGET_LIMIT,
    CalibrationBudgetExceededError,
    CalibrationEngineError,
    CalibrationGeometryError,
    CalibrationInputHashMismatchError,
    CalibrationInputMissingError,
    CalibrationMockEngine,
    CalibrationPlanIntegrityError,
    CalibrationPlanNotApprovedError,
    CalibrationRunAuditError,
    audit_calibration_run,
    collect_runtime_telemetry,
    execute_calibration_run,
    generate_calibration_contact_sheet,
    load_and_validate_calibration_plan,
    verify_calibration_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CALIB_PLAN_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_calibration_proposal.json"
REAL_INPUTS_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"


def test_real_calibration_plan_approved_and_matches_sha256():
    """Real calibration proposal must have status APPROVED and match registered SHA-256."""
    assert REAL_CALIB_PLAN_PATH.is_file()
    plan_data = load_and_validate_calibration_plan(
        REAL_CALIB_PLAN_PATH,
        expected_plan_sha256=APPROVED_CALIBRATION_PLAN_SHA256,
        require_approval=True,
    )
    assert plan_data["human_review_status"] == "APPROVED"
    assert plan_data["human_reviewer"] == "Dũng Phạm <valdung04@gmail.com>"
    assert plan_data["human_reviewed_at_utc"] == "2026-10-08T19:34:30Z"
    assert plan_data["attempt_budget"] == 2
    assert plan_data["automatic_replacement"] is False
    assert len(plan_data["calibration_attempts"]) == 2

    # Check the two approved guidance scales
    attempts = plan_data["calibration_attempts"]
    assert attempts[0]["guidance_scale"] == 7.5
    assert attempts[1]["guidance_scale"] == 9.5
    assert attempts[0]["generation_seed"] == 20272319
    assert attempts[1]["generation_seed"] == 20272319
    assert attempts[0]["prompt"] == attempts[1]["prompt"]
    assert attempts[0]["negative_prompt"] == attempts[1]["negative_prompt"]
    assert attempts[0]["crop_applied"] is False
    assert attempts[1]["crop_applied"] is False


def test_unapproved_calibration_plan_fails_closed(tmp_path):
    """Plans with human_review_status != APPROVED must be refused when require_approval=True."""
    plan_data = json.loads(REAL_CALIB_PLAN_PATH.read_text(encoding="utf-8"))
    plan_data["human_review_status"] = "PENDING"
    plan_data["human_reviewer"] = None
    plan_data["human_reviewed_at_utc"] = None
    p_path = tmp_path / "pending_calib_plan.json"
    p_path.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(CalibrationPlanNotApprovedError, match="PENDING"):
        load_and_validate_calibration_plan(p_path, require_approval=True)


def test_calibration_plan_hash_mismatch_detected(tmp_path):
    """Mismatched plan hash must raise CalibrationPlanIntegrityError."""
    p_copy = tmp_path / "calib_plan.json"
    p_copy.write_text(REAL_CALIB_PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(CalibrationPlanIntegrityError, match="SHA-256 mismatch"):
        load_and_validate_calibration_plan(
            p_copy, expected_plan_sha256="0" * 64, require_approval=False
        )


def test_calibration_budget_mismatch_rejected(tmp_path):
    """Plans specifying attempt_budget != 2 must be rejected."""
    data = json.loads(REAL_CALIB_PLAN_PATH.read_text(encoding="utf-8"))
    data["attempt_budget"] = 4
    p_copy = tmp_path / "bad_budget.json"
    p_copy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationBudgetExceededError, match="attempt_budget 4"):
        load_and_validate_calibration_plan(p_copy, require_approval=False)


def test_calibration_invariant_drift_rejected(tmp_path):
    """Modifying controlled invariants must raise CalibrationPlanIntegrityError."""
    data = json.loads(REAL_CALIB_PLAN_PATH.read_text(encoding="utf-8"))
    data["controlled_invariants"]["generation_seed"] = 99999
    p_copy = tmp_path / "drifted.json"
    p_copy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationPlanIntegrityError, match="generation_seed"):
        load_and_validate_calibration_plan(p_copy, require_approval=False)


def test_real_calibration_inputs_verified():
    """Inputs from pilot-20261008T113700Z must match registered authentic and mask hashes."""
    plan_data = load_and_validate_calibration_plan(REAL_CALIB_PLAN_PATH, require_approval=False)
    verified = verify_calibration_inputs(plan_data, REAL_INPUTS_DIR)
    assert len(verified) == 2
    for att_id in ("CALIB_COCO_SDXL_002_PROMPT_G75", "CALIB_COCO_SDXL_002_PROMPT_G95"):
        assert att_id in verified
        assert verified[att_id]["authentic"].is_file()
        assert verified[att_id]["mask"].is_file()


def test_tampered_calibration_input_detected(tmp_path):
    """Tampering with an authentic or mask image must raise CalibrationInputHashMismatchError."""
    auth = Image.new("RGB", (512, 512), (120, 120, 120))
    mask = Image.new("L", (512, 512), 0)
    auth_p = tmp_path / "test_auth.png"
    mask_p = tmp_path / "test_mask.png"
    auth.save(auth_p)
    mask.save(mask_p)

    data = json.loads(REAL_CALIB_PLAN_PATH.read_text(encoding="utf-8"))
    att = data["calibration_attempts"][0]
    att["authentic_input_filename"] = "test_auth.png"
    att["authentic_sha256"] = "bad" * 21 + "b"
    att["mask_input_filename"] = "test_mask.png"

    with pytest.raises(CalibrationInputHashMismatchError, match="SHA-256"):
        verify_calibration_inputs(data, tmp_path)


def test_pipeline_transmission_prompt_negative_prompt_guidance(tmp_path):
    """Verify that prompt, negative_prompt, and guidance_scale are accurately passed to engine."""
    plan_data = load_and_validate_calibration_plan(REAL_CALIB_PLAN_PATH, require_approval=False)
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(json.dumps(plan_data), encoding="utf-8")

    captured_calls: list[dict] = []

    class InspectingMockEngine(CalibrationMockEngine):
        def inpaint(self, **kwargs):
            captured_calls.append(dict(kwargs))
            return super().inpaint(**kwargs)

    execute_calibration_run(
        output_dir=tmp_path / "run",
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="c" * 40,
        engine_provider=lambda _: InspectingMockEngine(),
        allow_mock=True,
        require_approval=False,
    )

    assert len(captured_calls) == 2
    # Check attempt 1
    call1 = captured_calls[0]
    assert call1["guidance_scale"] == 7.5
    assert call1["seed"] == 20272319
    assert call1["prompt"] == plan_data["calibration_attempts"][0]["prompt"]
    assert call1["negative_prompt"] == plan_data["calibration_attempts"][0]["negative_prompt"]
    assert call1["num_inference_steps"] == 30
    assert call1["height"] == 512 and call1["width"] == 512

    # Check attempt 2
    call2 = captured_calls[1]
    assert call2["guidance_scale"] == 9.5
    assert call2["seed"] == 20272319
    assert call2["prompt"] == plan_data["calibration_attempts"][1]["prompt"]
    assert call2["negative_prompt"] == plan_data["calibration_attempts"][1]["negative_prompt"]
    assert call2["num_inference_steps"] == 30
    assert call2["height"] == 512 and call2["width"] == 512


def test_mock_safeguard_refuses_mock_in_production(tmp_path):
    """Using a mock engine with allow_mock=False must raise CalibrationEngineError."""
    plan_data = load_and_validate_calibration_plan(REAL_CALIB_PLAN_PATH, require_approval=False)
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(CalibrationEngineError, match="Mock engine is prohibited"):
        execute_calibration_run(
            output_dir=tmp_path / "run_prod",
            plan_path=plan_p,
            inputs_dir=REAL_INPUTS_DIR,
            expected_commit="d" * 40,
            engine_provider=lambda _: CalibrationMockEngine(),
            allow_mock=False,
            require_approval=False,
        )


def test_failure_accounts_towards_budget_and_resume_skips_consumed(tmp_path):
    """An attempt that starts and fails counts towards the 2-attempt budget; resume skips consumed."""
    plan_data = load_and_validate_calibration_plan(REAL_CALIB_PLAN_PATH, require_approval=False)
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(json.dumps(plan_data), encoding="utf-8")

    run_dir = tmp_path / "calib_run_fail"

    attempt_counter = 0

    class FailingOnFirstEngine(CalibrationMockEngine):
        def inpaint(self, **kwargs):
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter == 1:
                raise RuntimeError("Simulated OOM or CUDA error on attempt 1")
            return super().inpaint(**kwargs)

    # First run fails on attempt 1
    with pytest.raises(RuntimeError, match="Simulated OOM"):
        execute_calibration_run(
            output_dir=run_dir,
            plan_path=plan_p,
            inputs_dir=REAL_INPUTS_DIR,
            expected_commit="e" * 40,
            engine_provider=lambda _: FailingOnFirstEngine(),
            allow_mock=True,
            require_approval=False,
        )

    # Check attempt_ledger.jsonl: attempt 1 must have STARTED and FAILED records
    ledger_path = run_dir / "attempt_ledger.jsonl"
    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["event"] == "ATTEMPT_STARTED" and lines[0]["status"] == "STARTED"
    assert lines[1]["event"] == "ATTEMPT_FINISHED" and lines[1]["status"] == "FAILED"
    assert "Simulated OOM" in lines[1]["error_message"]

    # Resume run: attempt 1 was consumed and must be skipped; attempt 2 runs
    class WorkingEngine(CalibrationMockEngine):
        pass

    execute_calibration_run(
        output_dir=run_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit="e" * 40,
        engine_provider=lambda _: WorkingEngine(),
        resume=True,
        allow_mock=True,
        require_approval=False,
    )

    lines_resumed = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    # Now should have: attempt 1 STARTED, attempt 1 FAILED, attempt 2 STARTED, attempt 2 ACCEPTED
    assert len(lines_resumed) == 4
    assert lines_resumed[2]["attempt_id"] == "CALIB_COCO_SDXL_002_PROMPT_G95"
    assert lines_resumed[2]["status"] == "STARTED"
    assert lines_resumed[3]["status"] == "ACCEPTED"


def test_outside_pixel_invariance_and_audit(tmp_path):
    """Generated composites must have outside_mean_l1 == 0.000000; audit verifies all checks."""
    plan_data = load_and_validate_calibration_plan(REAL_CALIB_PLAN_PATH, require_approval=False)
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(json.dumps(plan_data), encoding="utf-8")

    run_dir = tmp_path / "calib_audit_run"
    commit = "f" * 40

    receipt = execute_calibration_run(
        output_dir=run_dir,
        plan_path=plan_p,
        inputs_dir=REAL_INPUTS_DIR,
        expected_commit=commit,
        engine_provider=lambda _: CalibrationMockEngine(),
        allow_mock=True,
        require_approval=False,
    )

    assert receipt["total_attempts_executed"] == 2
    for att in receipt["attempts"]:
        assert att["outside_mean_l1"] == 0.0
        assert att["outside_max_delta"] == 0.0
        assert att["human_content_qc_status"] == "PENDING"

    # Contact sheet must exist and be self-contained
    cs_path = run_dir / "calibration_contact_sheet.html"
    assert cs_path.is_file()
    assert "data:image/png;base64," in cs_path.read_text(encoding="utf-8")

    # Audit run
    expected_binding = {
        "run_id": run_dir.name,
        "mode": "calibration",
        "git_commit": commit,
        "calibration_plan_sha256": receipt["calibration_plan_sha256"],
    }
    audit_res = audit_calibration_run(run_dir, expected_binding)
    assert audit_res["status"] == "PASS"


def test_runtime_telemetry_fields():
    """Verify all required telemetry fields are present without speculation."""
    telemetry = collect_runtime_telemetry()
    for field in ("python", "platform", "torch", "diffusers", "transformers", "numpy", "pillow", "cuda", "gpu"):
        assert field in telemetry
        assert isinstance(telemetry[field], str)
