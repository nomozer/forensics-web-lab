"""
ml/tests/test_document_pilot_reporting_reconciliation.py

Unit and contract tests for reporting reconciliation, mathematical self-consistency,
discrepancy ledger verification, public candidate dataset accessibility audit,
and independent document generation fallback design.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RECEIPT_V110_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_reporting_correction_receipt.json"
REPORT_V110_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_REPORTING_CORRECTION_V1.1.0.md"

RECEIPT_V120_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_reporting_correction_receipt_v1.2.0.json"
REPORT_V120_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_REPORTING_CORRECTION_V1.2.0.md"

RECON_REPORT_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_RECONCILIATION_REPORT.md"
AIFORGE_PROTO_PATH = REPO_ROOT / "research/evidence/document_pilot/PROTOCOL_AIFORGE_DOC_INTAKE_V1.0.0.md"
AIFORGE_AUDIT_PATH = REPO_ROOT / "research/evidence/document_pilot/aiforge_doc_access_blocker_audit.json"

PUBLIC_CANDIDATES_AUDIT_PATH = REPO_ROOT / "research/evidence/document_pilot/public_document_candidates_audit.json"
FALLBACK_DESIGN_PATH = REPO_ROOT / "research/evidence/document_pilot/INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md"


def compute_exact_macro_f1(tp: int, tn: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    f1_pos = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f1_neg = (2.0 * tnr * npv) / (tnr + npv) if (tnr + npv) > 0 else 0.0
    return (f1_pos + f1_neg) / 2.0


def test_reporting_correction_receipt_v110_and_v120_integrity():
    assert RECEIPT_V110_PATH.exists(), f"Missing {RECEIPT_V110_PATH}"
    assert RECEIPT_V120_PATH.exists(), f"Missing {RECEIPT_V120_PATH}"

    data_v11 = json.loads(RECEIPT_V110_PATH.read_text(encoding="utf-8"))
    assert data_v11["status"] == "REPORTING_RECONCILIATION_SUCCESS"

    data_v12 = json.loads(RECEIPT_V120_PATH.read_text(encoding="utf-8"))
    assert data_v12["schema_version"] == "1.2.0"
    assert data_v12["status"] == "REPORTING_RECONCILIATION_SUCCESS"
    assert "discrepancy_ledger" in data_v12
    assert len(data_v12["discrepancy_ledger"]) >= 4


def test_discrepancy_ledger_mathematical_derivation_and_no_hand_transcription():
    """
    Guarantees that every implied Macro-F1 in the discrepancy ledger is derived
    strictly from its implied confusion matrix counts, preventing rough hand-typed approximations.
    """
    assert RECEIPT_V120_PATH.exists()
    data = json.loads(RECEIPT_V120_PATH.read_text(encoding="utf-8"))

    expected_rational_fractions = {
        "DISC-TXT-01": 293 / 493,   # 0.5943204868...
        "DISC-TXT-02": 43 / 75,     # 0.5733333333...
        "DISC-TXT-03": 719 / 1479,  # 0.4861392832...
        "DISC-TXT-04": 839 / 1599,  # 0.5247029393...
    }

    for item in data["discrepancy_ledger"]:
        item_id = item["item_id"]
        imp_cm = item["chat_text_reported"]["implied_confusion_matrix"]
        tp, tn, fp, fn = imp_cm["tp"], imp_cm["tn"], imp_cm["fp"], imp_cm["fn"]

        # Calculate exact value from confusion matrix
        calculated_f1 = compute_exact_macro_f1(tp, tn, fp, fn)
        stored_f1 = item["chat_text_reported"]["implied_macro_f1"]

        # Assert calculation matches stored value within high precision
        assert abs(calculated_f1 - stored_f1) < 1e-6, (
            f"Ledger {item_id} stored implied F1 {stored_f1} does not match calculated {calculated_f1}"
        )

        # Assert match with theoretical rational fraction
        expected_rf = expected_rational_fractions[item_id]
        assert abs(calculated_f1 - expected_rf) < 1e-6, (
            f"Ledger {item_id} calculated F1 {calculated_f1} diverges from rational expectation {expected_rf}"
        )

        # Confirm verified SSOT is also self-consistent
        ssot_cm = item["ssot_raw_verified"]["confusion_matrix"]
        ssot_calc_f1 = compute_exact_macro_f1(ssot_cm["tp"], ssot_cm["tn"], ssot_cm["fp"], ssot_cm["fn"])
        assert abs(ssot_calc_f1 - item["ssot_raw_verified"]["macro_f1_exact"]) < 1e-6


def test_epistemic_scope_rectifications_present():
    assert RECEIPT_V120_PATH.exists()
    assert REPORT_V120_PATH.exists()

    receipt_data = json.loads(RECEIPT_V120_PATH.read_text(encoding="utf-8"))
    assert "epistemic_scope_rectifications" in receipt_data

    rects = receipt_data["epistemic_scope_rectifications"]
    assert "confusion_counts_scope" in rects
    assert "rounding_precision_scope" in rects
    assert "bootstrap_scope" in rects
    assert "multiclass_brier_scope" in rects

    report_text = REPORT_V120_PATH.read_text(encoding="utf-8")
    assert "## 5. Giới Hạn & Ranh Giới Khẳng Định Khoa Học (Epistemic Rectification)" in report_text
    assert "Ranh giới tính tự nhất quán" in report_text
    assert "Quy chuẩn độ chính xác số làm tròn" in report_text
    assert "Ranh giới Bootstrap" in report_text
    assert "Không nhầm lẫn Bootstrap Fraction với p-value" in report_text
    assert "Giới hạn Multiclass Brier" in report_text


def test_public_document_candidates_audit_receipt():
    assert PUBLIC_CANDIDATES_AUDIT_PATH.exists()
    audit_data = json.loads(PUBLIC_CANDIDATES_AUDIT_PATH.read_text(encoding="utf-8"))

    assert audit_data["total_candidates_audited"] == 5
    assert audit_data["eligible_candidates_count"] == 0
    assert audit_data["gated_candidates_count"] == 2
    assert audit_data["out_of_scope_non_ai_candidates_count"] == 3

    cand_statuses = {c["repository"]: c["epistemic_status"] for c in audit_data["candidates"]}
    assert cand_statuses["Scam-AI/AIForge-Doc-v1"] == "YEU_CAU_DANG_NHAP"
    assert cand_statuses["Scam-AI/gpt4o-receipt"] == "YEU_CAU_DANG_NHAP"
    assert cand_statuses["MohamedDhouib1/TDoc-2.8M"] == "KHONG_DUNG_PHAM_VI"
    assert cand_statuses["Jason37437/RealText-V2-Syn25k"] == "KHONG_DUNG_PHAM_VI"
    assert cand_statuses["Zegkim/FSTS"] == "KHONG_DUNG_PHAM_VI"


def test_independent_document_generation_pilot_design():
    assert FALLBACK_DESIGN_PATH.exists()
    design_text = FALLBACK_DESIGN_PATH.read_text(encoding="utf-8")

    assert "CORD v2" in design_text
    assert "stable-diffusion-xl-1.0-inpainting" in design_text or "stable-diffusion-2-inpainting" in design_text
    assert "Google Colab GPU" in design_text
    assert "40 cặp nguồn" in design_text
    assert "outside_mean_l1 = 0.000000" in design_text
    assert "PNG lossless" in design_text
    assert "survivorship bias" in design_text


def test_mathematical_consistency_across_all_configurations_v120():
    data = json.loads(RECEIPT_V120_PATH.read_text(encoding="utf-8"))
    table = data["ssot_metrics_table"]

    for cond in ["replay_c0", "replay_c1"]:
        for model, geoms in table[cond].items():
            for geom, met in geoms.items():
                tp, tn, fp, fn = met["tp"], met["tn"], met["fp"], met["fn"]
                assert tp + tn + fp + fn == 40, f"Sample count sum mismatch in {cond} {model} {geom}"

                # Test rates
                assert abs(met["tpr_recall"] - (tp / 20.0)) < 1e-4
                assert abs(met["fpr"] - (fp / 20.0)) < 1e-4
                assert abs(met["tnr"] - (tn / 20.0)) < 1e-4

                # Test balanced accuracy
                expected_bal_acc = round((met["tpr_recall"] + met["tnr"]) / 2.0, 4)
                assert abs(met["balanced_accuracy"] - expected_bal_acc) < 1e-4

                # Test binary Macro-F1 formula
                prec = met["precision"]
                tpr = met["tpr_recall"]
                tnr = met["tnr"]
                npv = met["npv"]
                f1_pos = (2.0 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0
                f1_neg = (2.0 * tnr * npv) / (tnr + npv) if (tnr + npv) > 0 else 0.0
                expected_macro_f1 = round((f1_pos + f1_neg) / 2.0, 4)
                assert abs(met["macro_f1"] - expected_macro_f1) < 1e-4


if __name__ == "__main__":
    pytest.main([str(Path(__file__)), "-v"])
