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
 11. Cohort N=100 seed variability exact sample SD parity.
 12. Cohort N=100 variability fault injections & fail-closed detection.
 13. Variability narrative prohibits confirmatory claims for n=5 seeds.
 14. Non-circular evidence seal provenance semantics and no pending placeholders.
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


# -----------------------------------------------------------------------------
# Test 11: Cohort N=100 Seed Variability Exact Sample SD Parity
# -----------------------------------------------------------------------------
def test_cohort_n100_variability_exact_sample_sd_parity():
    """Computes Stage 1 SD, Stage 2 SD, and paired-delta SD directly from raw metrics.

    Verifies:
      - Stage 1 Macro-F1 sample SD == 0.004802230617102873 (tolerance < 1e-6)
      - Stage 2 Macro-F1 sample SD == 0.01915920444456796 (tolerance < 1e-6)
      - Paired-delta sample SD == 0.019371286120698345 (tolerance < 1e-6)
      - Stage 2 / Stage 1 SD ratio == 3.989647... (approx 3.99)
      - FINAL_MODEL_SELECTION.json and PHASE_REPORT.md accurately reflect these exact values.
    """
    paired_csv = EVIDENCE_DIR_4C2C / "paired_run_metrics.csv"
    assert paired_csv.is_file(), f"Missing {paired_csv}"

    with open(paired_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n100_rows = [r for r in rows if int(r["sample_size"]) == 100]
    assert len(n100_rows) == 5, f"Expected 5 runs for N=100, got {len(n100_rows)}"

    s1_vals = np.array([float(r["stage1_macro_f1"]) for r in n100_rows])
    s2_vals = np.array([float(r["stage2_macro_f1"]) for r in n100_rows])
    delta_vals = np.array([float(r["delta_macro_f1"]) for r in n100_rows])

    s1_sd = float(np.std(s1_vals, ddof=1))
    s2_sd = float(np.std(s2_vals, ddof=1))
    delta_sd = float(np.std(delta_vals, ddof=1))
    ratio = s2_sd / s1_sd

    EXPECTED_S1_SD = 0.004802230617102873
    EXPECTED_S2_SD = 0.01915920444456796
    EXPECTED_DELTA_SD = 0.019371286120698345
    EXPECTED_RATIO = 3.9896468895795083

    assert abs(s1_sd - EXPECTED_S1_SD) < 1e-6, f"S1 SD {s1_sd} != {EXPECTED_S1_SD}"
    assert abs(s2_sd - EXPECTED_S2_SD) < 1e-6, f"S2 SD {s2_sd} != {EXPECTED_S2_SD}"
    assert abs(delta_sd - EXPECTED_DELTA_SD) < 1e-6, f"Delta SD {delta_sd} != {EXPECTED_DELTA_SD}"
    assert abs(ratio - EXPECTED_RATIO) < 1e-3, f"Ratio {ratio} != {EXPECTED_RATIO}"

    # Verify appearance in FINAL_MODEL_SELECTION.json
    decision_p = EVIDENCE_DIR_4C2D / "FINAL_MODEL_SELECTION.json"
    assert decision_p.is_file()
    decision_json = json.loads(decision_p.read_text(encoding="utf-8"))
    variability_narrative_json = decision_json["decision_basis"]["statistical_uncertainty_and_seed_variability"]

    assert "0.019159" in variability_narrative_json, "Stage 2 SD 0.019159 missing from JSON"
    assert "0.004802" in variability_narrative_json, "Stage 1 SD 0.004802 missing from JSON"
    assert "0.019371" in variability_narrative_json, "Paired-delta SD 0.019371 missing from JSON"
    assert "3.99" in variability_narrative_json or "3.989" in variability_narrative_json, "Ratio 3.99 missing from JSON"

    # Verify appearance in PHASE_REPORT.md
    report_p = EVIDENCE_DIR_4C2D / "PHASE_REPORT.md"
    assert report_p.is_file()
    report_text = report_p.read_text(encoding="utf-8")

    assert "0.019159" in report_text, "Stage 2 SD 0.019159 missing from PHASE_REPORT.md"
    assert "0.004802" in report_text, "Stage 1 SD 0.004802 missing from PHASE_REPORT.md"
    assert "0.019371" in report_text, "Paired-delta SD 0.019371 missing from PHASE_REPORT.md"
    assert "3.99" in report_text or "3.989" in report_text, "Ratio 3.99 missing from PHASE_REPORT.md"


# -----------------------------------------------------------------------------
# Test 12: Cohort N=100 Variability Fault Injections & Fail-Closed Detection
# -----------------------------------------------------------------------------
def test_cohort_n100_variability_fault_injection_detects_swapped_sds():
    """Ensures that confusing paired-delta SD for stage SD or N=50 SD for N=100 fails closed."""
    paired_csv = EVIDENCE_DIR_4C2C / "paired_run_metrics.csv"
    with open(paired_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n100_rows = [r for r in rows if int(r["sample_size"]) == 100]
    n50_rows = [r for r in rows if int(r["sample_size"]) == 50]

    s1_100_sd = float(np.std([float(r["stage1_macro_f1"]) for r in n100_rows], ddof=1))
    s2_100_sd = float(np.std([float(r["stage2_macro_f1"]) for r in n100_rows], ddof=1))
    delta_100_sd = float(np.std([float(r["delta_macro_f1"]) for r in n100_rows], ddof=1))

    s1_50_sd = float(np.std([float(r["stage1_macro_f1"]) for r in n50_rows], ddof=1))
    delta_50_sd = float(np.std([float(r["delta_macro_f1"]) for r in n50_rows], ddof=1))

    TOLERANCE = 1e-6

    def audit_variability_sds(s1_tested, s2_tested, delta_tested):
        errors = []
        if abs(s1_tested - s1_100_sd) > TOLERANCE:
            errors.append(f"Stage 1 SD mismatch: tested={s1_tested}, actual={s1_100_sd}")
        if abs(s2_tested - s2_100_sd) > TOLERANCE:
            errors.append(f"Stage 2 SD mismatch: tested={s2_tested}, actual={s2_100_sd}")
        if abs(delta_tested - delta_100_sd) > TOLERANCE:
            errors.append(f"Paired-delta SD mismatch: tested={delta_tested}, actual={delta_100_sd}")
        if errors:
            raise ValueError("; ".join(errors))

    # Base case: true numbers must pass
    audit_variability_sds(s1_100_sd, s2_100_sd, delta_100_sd)

    # Fault 1: Using paired-delta SD in place of Stage 2 SD (diff = ~0.000212 > 1e-6)
    with pytest.raises(ValueError, match="Stage 2 SD mismatch"):
        audit_variability_sds(s1_100_sd, delta_100_sd, delta_100_sd)

    # Fault 2: Using paired-delta SD in place of Stage 1 SD (diff = ~0.014569 > 1e-6)
    with pytest.raises(ValueError, match="Stage 1 SD mismatch"):
        audit_variability_sds(delta_100_sd, s2_100_sd, delta_100_sd)

    # Fault 3: Using N=50 paired-delta SD (0.008220) in place of N=100 Stage 1 SD (0.004802)
    with pytest.raises(ValueError, match="Stage 1 SD mismatch"):
        audit_variability_sds(delta_50_sd, s2_100_sd, delta_100_sd)

    # Fault 4: Using N=50 Stage 1 SD (0.0148) in place of N=100 Stage 1 SD
    with pytest.raises(ValueError, match="Stage 1 SD mismatch"):
        audit_variability_sds(s1_50_sd, s2_100_sd, delta_100_sd)

    # Fault 5: Old drafting errors: s2 = 0.0194, s1 = 0.0082 (both must fail closed)
    with pytest.raises(ValueError, match="Stage 1 SD mismatch.*Stage 2 SD mismatch"):
        audit_variability_sds(0.0082, 0.0194, delta_100_sd)


# -----------------------------------------------------------------------------
# Test 13: Variability Narrative Prohibits Confirmatory Claims for n=5 Seeds
# -----------------------------------------------------------------------------
def test_variability_narrative_prohibits_confirmatory_claims():
    """Ensures variance comparisons on n=5 seeds are qualified as exploratory and not confirmatory."""
    decision_p = EVIDENCE_DIR_4C2D / "FINAL_MODEL_SELECTION.json"
    report_p = EVIDENCE_DIR_4C2D / "PHASE_REPORT.md"

    decision_data = json.loads(decision_p.read_text(encoding="utf-8"))
    report_text = report_p.read_text(encoding="utf-8")

    # Extract the specific variance text in JSON
    json_var_text = decision_data["decision_basis"]["statistical_uncertainty_and_seed_variability"]

    # Extract the specific variance line in Markdown
    md_var_line = ""
    for line in report_text.splitlines():
        if "Độ ổn định phương sai" in line:
            md_var_line = line
            break
    assert md_var_line, "Missing 'Độ ổn định phương sai' section in PHASE_REPORT.md"

    # Affirmative confirmatory claims are strictly prohibited
    affirmative_patterns = [
        r"bằng chứng confirmatory về (?:phương sai|variance)",
        r"(?<!không phải )kết luận confirmatory về (?:phương sai|variance)",
        r"(?<!rather than a )(?<!not a )confirmatory conclusion on variance",
        r"(?<!not )(?<!no )confirmatory evidence (?:of|for) variance",
    ]

    for pat in affirmative_patterns:
        assert not re.search(pat, json_var_text, re.IGNORECASE), f"Prohibited affirmative pattern '{pat}' in JSON"
        assert not re.search(pat, md_var_line, re.IGNORECASE), f"Prohibited affirmative pattern '{pat}' in Markdown"

    # Must contain explicit negation and exploratory qualification
    assert "rather than a confirmatory conclusion on variance" in json_var_text
    assert "exploratory" in json_var_text
    assert "n=5 seeds" in json_var_text

    assert "không phải kết luận confirmatory về variance" in md_var_line
    assert "mang tính khám phá" in md_var_line
    assert "5 seeds" in md_var_line


# -----------------------------------------------------------------------------
# Test 14: Non-Circular Evidence Seal Provenance
# -----------------------------------------------------------------------------
def test_non_circular_evidence_seal_provenance():
    """Verifies that Phase 4C.2D evidence seal avoids circular hash references.

    Asserts:
      1. No 'PENDING_HOTFIX_SEAL' string exists in provenance_bindings.json or any evidence.
      2. No self-referential 'evidence_seal_commit' field in provenance_bindings.json.
      3. 'evidence_seal_semantics' contains valid 40-char hex SHA-1 for:
         - audited_through_commit
         - hotfix_content_commit
      4. selected_protocol remains 'stage1_frozen_backbone_linear_probe'.
      5. locked_test_accesses == 0.
    """
    prov_p = EVIDENCE_DIR_4C2D / "provenance_bindings.json"
    assert prov_p.is_file(), f"Missing {prov_p}"

    raw_text = prov_p.read_text(encoding="utf-8")
    assert "PENDING_HOTFIX_SEAL" not in raw_text, "Found unsealed PENDING_HOTFIX_SEAL placeholder in provenance_bindings.json"

    # Also check no other evidence file in Phase 4C.2D contains PENDING_HOTFIX_SEAL
    for ep in EVIDENCE_DIR_4C2D.rglob("*"):
        if ep.is_file() and ep.suffix in [".json", ".csv", ".md"]:
            content = ep.read_text(encoding="utf-8")
            assert "PENDING_HOTFIX_SEAL" not in content, f"Found PENDING_HOTFIX_SEAL in {ep.name}"

    prov_data = json.loads(raw_text)
    assert "evidence_seal_commit" not in prov_data, "Self-referential 'evidence_seal_commit' field must be removed"

    assert "evidence_seal_semantics" in prov_data, "Missing 'evidence_seal_semantics' object"
    semantics = prov_data["evidence_seal_semantics"]

    sha1_pat = re.compile(r"^[0-9a-f]{40}$")
    assert sha1_pat.match(semantics.get("audited_through_commit", "")), f"Invalid audited_through_commit SHA: {semantics.get('audited_through_commit')}"
    assert sha1_pat.match(semantics.get("hotfix_content_commit", "")), f"Invalid hotfix_content_commit SHA: {semantics.get('hotfix_content_commit')}"

    assert semantics["audited_through_commit"] == "e24d0ec65d97fe139c3a4efd0ac03c836b3e9aa9"
    assert semantics["hotfix_content_commit"] == "c8bcc0be16155a8da004a43c2617b048035a01af"
    assert "circular hash dependency" in semantics.get("seal_resolution", "").lower()

    assert prov_data["selected_protocol"] == "stage1_frozen_backbone_linear_probe"
    assert prov_data["locked_test_accesses"] == 0
