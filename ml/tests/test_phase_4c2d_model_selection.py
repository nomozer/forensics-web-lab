"""Unit and regression test suite for Phase 4C.2D Final Model-Selection Gate.

Verifies:
  1. 15/15 Stage 1 lineage parity (receipt == metrics == predictions == checkpoint == history).
  2. 15/15 Stage 2 lineage parity.
  3. Cross-artifact numerical parity within 1e-6 tolerance.
  4. Calibration wording prohibits absolute claims (overconfidence/underconfidence).
  5. Strictly exactly one selected protocol (Stage 1) with post-hoc label.
  6. Locked-test accesses strictly zero and sealed.
  7. Zero absolute Windows paths in Git evidence.
  8. Markdown numbers strictly match machine-readable CSV/JSON.
  9. Fail-closed behavior on tolerance violations.
 10. Runner contract trace for Stage 1 checkpoint reload.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVIDENCE_DIR_4C2C = REPO_ROOT / "research" / "evidence" / "phase-4c.2c"
EVIDENCE_DIR_4C2C1 = REPO_ROOT / "research" / "evidence" / "phase-4c.2c.1"
EVIDENCE_DIR_4C2D = REPO_ROOT / "research" / "evidence" / "phase-4c.2d"


# -----------------------------------------------------------------------------
# Test 1: 15/15 Stage 1 Lineage Parity
# -----------------------------------------------------------------------------
def test_stage1_lineage_15_runs_parity():
    """Confirms 100% internal parity across all 15 raw Stage 1 runs."""
    lineage_p = EVIDENCE_DIR_4C2D / "stage1_metric_lineage.json"
    assert lineage_p.is_file(), f"Missing {lineage_p}"

    data = json.loads(lineage_p.read_text(encoding="utf-8"))
    assert len(data) == 15, f"Expected 15 Stage 1 runs, got {len(data)}"

    for r in data:
        assert r["overall_parity_verdict"] == "PARITY_VERIFIED"
        assert r["receipt_best_vs_final_parity"] == "MATCH"
        assert r["receipt_final_vs_metrics_parity"] == "MATCH"
        assert r["metrics_vs_predictions_parity"] == "MATCH"
        assert r["metrics_vs_checkpoint_parity"] == "MATCH"
        assert r["metrics_vs_epoch_history_parity"] == "MATCH"
        assert r["receipt_epoch_vs_checkpoint_parity"] == "MATCH"
        assert r["receipt_epoch_vs_history_parity"] == "MATCH"


# -----------------------------------------------------------------------------
# Test 2: 15/15 Stage 2 Lineage Parity
# -----------------------------------------------------------------------------
def test_stage2_lineage_15_runs_parity():
    """Confirms 100% internal parity across all 15 raw Stage 2 runs."""
    lineage_p = EVIDENCE_DIR_4C2C1 / "stage2_metric_lineage.json"
    assert lineage_p.is_file(), f"Missing {lineage_p}"

    data = json.loads(lineage_p.read_text(encoding="utf-8"))
    assert len(data) == 15, f"Expected 15 Stage 2 runs, got {len(data)}"

    for r in data:
        assert r["overall_parity_verdict"] == "PARITY_VERIFIED"
        assert r["receipt_best_vs_final_parity"] == "MATCH"
        assert r["receipt_final_vs_metrics_parity"] == "MATCH"
        assert r["metrics_vs_predictions_parity"] == "MATCH"
        assert r["metrics_vs_checkpoint_parity"] == "MATCH"
        assert r["metrics_vs_epoch_history_parity"] == "MATCH"


# -----------------------------------------------------------------------------
# Test 3: Cross-Artifact Numerical Parity (Tolerance 1e-6)
# -----------------------------------------------------------------------------
def test_final_metric_parity_cross_artifacts():
    """Verifies that final_metric_parity.json asserts 100% parity within 1e-6 tolerance."""
    parity_p = EVIDENCE_DIR_4C2D / "final_metric_parity.json"
    assert parity_p.is_file(), f"Missing {parity_p}"

    parity_data = json.loads(parity_p.read_text(encoding="utf-8"))
    assert parity_data["overall_parity_verdict"] == "FINAL_PARITY_VERIFIED"
    assert parity_data["tolerance"] <= 1e-6
    assert parity_data["stage1_lineage_parity"] == "15/15 MATCH"
    assert parity_data["stage2_lineage_parity"] == "15/15 MATCH"
    assert parity_data["csv_vs_raw_parity"] == "15/15 MATCH"
    assert parity_data["tests_vs_summary_parity"] == "3/3 COHORTS MATCH"
    assert len(parity_data["runs"]) == 15


# -----------------------------------------------------------------------------
# Test 4: Calibration Wording Prohibits Absolute Claims
# -----------------------------------------------------------------------------
def test_calibration_wording_prohibits_absolutes():
    """Ensures absolute claims like 'bác bỏ hoàn toàn overconfidence' are strictly prohibited."""
    prohibited_patterns = [
        r"bác bỏ hoàn toàn.*overconfidence",
        r"khẳng định mô hình.*underconfident",
        r"bác bỏ giả thuyết overconfidence mang tính hệ thống",
        r"definitively overconfident",
        r"definitively underconfident",
    ]

    reports_to_check = [
        EVIDENCE_DIR_4C2C / "PHASE_REPORT.md",
        EVIDENCE_DIR_4C2C1 / "PHASE_REPORT.md",
        EVIDENCE_DIR_4C2D / "PHASE_REPORT.md",
    ]

    required_formulation = (
        "Không quan sát thấy xu hướng overconfidence trung bình trên inner-validation; "
        "signed confidence calibration gap âm gợi ý xu hướng underconfidence nhẹ. "
        "Đây là bằng chứng phát triển mô hình, không phải kết luận confirmatory."
    )

    for rep in reports_to_check:
        assert rep.is_file(), f"Missing report: {rep}"
        content = rep.read_text(encoding="utf-8")
        for pat in prohibited_patterns:
            assert not re.search(pat, content, re.IGNORECASE), f"Prohibited pattern '{pat}' found in {rep.name}"
        assert required_formulation in content, f"Required calibration wording missing from {rep.name}"


# -----------------------------------------------------------------------------
# Test 5: Single Selected Protocol & Post-Hoc Labeling
# -----------------------------------------------------------------------------
def test_single_selected_protocol():
    """Asserts that exactly one protocol is chosen (Stage 1) and correctly marked as post-hoc."""
    decision_p = EVIDENCE_DIR_4C2D / "FINAL_MODEL_SELECTION.json"
    assert decision_p.is_file(), f"Missing {decision_p}"

    data = json.loads(decision_p.read_text(encoding="utf-8"))
    assert data["selected_protocol"] == "stage1_frozen_backbone_linear_probe"
    assert data["rejected_or_secondary_protocol"] == "stage2_preregistered_partial_finetuning"
    assert data["preregistered_rule_present"] is False
    assert data["post_hoc_decision"] is True
    assert data["decision_verdict"] == "SELECT_STAGE1_AS_PRIMARY_CANDIDATE"
    assert "decision_basis" in data
    assert "locked_test_implications" in data


# -----------------------------------------------------------------------------
# Test 6: Locked-Test Accesses Zero
# -----------------------------------------------------------------------------
def test_locked_test_accesses_zero():
    """Asserts locked_test_accesses == 0 across all evidence artifacts and status is SEALED."""
    files_to_check = [
        EVIDENCE_DIR_4C2D / "FINAL_MODEL_SELECTION.json",
        EVIDENCE_DIR_4C2D / "locked_test_readiness.json",
        EVIDENCE_DIR_4C2D / "provenance_bindings.json",
    ]

    for p in files_to_check:
        assert p.is_file(), f"Missing {p}"
        data = json.loads(p.read_text(encoding="utf-8"))
        if "locked_test_accesses" in data:
            assert data["locked_test_accesses"] == 0
        if "locked_test_accesses_to_date" in data:
            assert data["locked_test_accesses_to_date"] == 0

    readiness = json.loads((EVIDENCE_DIR_4C2D / "locked_test_readiness.json").read_text(encoding="utf-8"))
    assert readiness["locked_test_partition_status"] == "SEALED"
    assert readiness["unsealing_authorization"] == "FORBIDDEN_IN_PHASE_4C2D"


# -----------------------------------------------------------------------------
# Test 7: No Absolute Windows Paths in Git Evidence
# -----------------------------------------------------------------------------
def test_no_absolute_windows_paths_in_phase_4c2d_evidence():
    """Verifies no drive letters or absolute local paths exist in Phase 4C.2D evidence."""
    drive_pat = re.compile(r"[a-zA-Z]:[\\/]")
    backslash_pat = re.compile(r"\\(?:Users|Documents|forensics|content)")

    for p in EVIDENCE_DIR_4C2D.rglob("*"):
        if p.is_file() and p.suffix in [".json", ".csv", ".md"]:
            content = p.read_text(encoding="utf-8")
            assert not drive_pat.search(content), f"Absolute drive path found in {p.name}"
            assert not backslash_pat.search(content), f"Backslash path found in {p.name}"


# -----------------------------------------------------------------------------
# Test 8: Markdown Numbers Match Machine-Readable JSON
# -----------------------------------------------------------------------------
def test_markdown_numbers_match_machine_readable_json():
    """Asserts that numbers in Phase 4C.2D PHASE_REPORT.md match stage1_metric_lineage.json."""
    lineage_p = EVIDENCE_DIR_4C2D / "stage1_metric_lineage.json"
    report_p = EVIDENCE_DIR_4C2D / "PHASE_REPORT.md"
    assert lineage_p.is_file() and report_p.is_file()

    lineage = json.loads(lineage_p.read_text(encoding="utf-8"))
    report = report_p.read_text(encoding="utf-8")

    for r in lineage:
        f1_str = f"{r['metrics_json_macro_f1']:.6f}"
        assert f1_str in report, f"Report missing Stage 1 F1 {f1_str} for N={r['sample_size']} seed={r['seed']}"


# -----------------------------------------------------------------------------
# Test 9: Fail-Closed on Tolerance Violation
# -----------------------------------------------------------------------------
def test_fail_closed_on_tolerance_violation():
    """Verifies that an intentional numerical discrepancy > 1e-6 triggers fail-closed error."""
    from scripts.research.finalize_phase_4c2d_model_selection import verify_cross_artifact_parity

    s1_lineage_path = EVIDENCE_DIR_4C2D / "stage1_metric_lineage.json"
    s2_lineage_path = EVIDENCE_DIR_4C2C1 / "stage2_metric_lineage.json"
    paired_csv = EVIDENCE_DIR_4C2C / "paired_run_metrics.csv"
    paired_summary_json = EVIDENCE_DIR_4C2C / "paired_summary.json"
    primary_tests_json = EVIDENCE_DIR_4C2C / "primary_statistical_tests.json"

    s1_lin = json.loads(s1_lineage_path.read_text(encoding="utf-8"))
    s2_lin = json.loads(s2_lineage_path.read_text(encoding="utf-8"))

    # Tamper with one entry to exceed 1e-6 tolerance
    s1_tampered = [dict(r) for r in s1_lin]
    s1_tampered[0]["metrics_json_macro_f1"] += 0.001

    with pytest.raises(ValueError, match="Stage 1 F1 mismatch"):
        verify_cross_artifact_parity(
            s1_tampered,
            s2_lin,
            paired_csv,
            paired_summary_json,
            primary_tests_json,
        )


# -----------------------------------------------------------------------------
# Test 10: Runner Contract Trace for Stage 1 Checkpoint Reload
# -----------------------------------------------------------------------------
def test_runner_contract_trace_stage1_checkpoint_reload():
    """Verifies the exact lines in run_phase_4c1.py where checkpoint is reloaded before final eval."""
    runner_p = REPO_ROOT / "ml" / "training" / "run_phase_4c1.py"
    assert runner_p.is_file(), f"Missing runner: {runner_p}"

    code = runner_p.read_text(encoding="utf-8")
    assert "checkpoint = torch.load(checkpoint_path, map_location=device)" in code
    assert "model.load_state_dict(checkpoint[\"model_state_dict\"])" in code
    assert "final_metrics = evaluate(model, val_loader, device, criterion)" in code
    assert "predictions_data = collect_predictions(model, val_loader, device)" in code
