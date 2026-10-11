"""
ml/tests/test_document_pilot_reporting_reconciliation.py

Unit and contract tests for reporting reconciliation, mathematical self-consistency,
and AIForge-Doc access blocker verification.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RECEIPT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_reporting_correction_receipt.json"
REPORT_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_REPORTING_CORRECTION_V1.1.0.md"
RECON_REPORT_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_RECONCILIATION_REPORT.md"
AIFORGE_PROTO_PATH = REPO_ROOT / "research/evidence/document_pilot/PROTOCOL_AIFORGE_DOC_INTAKE_V1.0.0.md"
AIFORGE_AUDIT_PATH = REPO_ROOT / "research/evidence/document_pilot/aiforge_doc_access_blocker_audit.json"


def test_reporting_correction_receipt_integrity():
    assert RECEIPT_PATH.exists(), f"Missing {RECEIPT_PATH}"
    data = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "REPORTING_RECONCILIATION_SUCCESS"
    assert "discrepancy_ledger" in data
    assert len(data["discrepancy_ledger"]) >= 4

    # Check DISC-TXT-01 (the A0-visual P0 contradiction identified by user)
    d1 = next(item for item in data["discrepancy_ledger"] if item["item_id"] == "DISC-TXT-01")
    assert d1["configuration"] == "A0-visual / C0 / P0"
    assert d1["chat_text_reported"]["macro_f1"] == 0.6011
    assert d1["chat_text_reported"]["tpr_recall"] == 0.3500
    assert d1["chat_text_reported"]["fpr"] == 0.1000
    assert d1["chat_text_reported"]["implied_macro_f1"] == 0.5943

    # Check verified SSOT counts
    ssot = d1["ssot_raw_verified"]
    assert ssot["confusion_matrix"]["tp"] == 6
    assert ssot["confusion_matrix"]["tn"] == 20
    assert ssot["confusion_matrix"]["fp"] == 0
    assert ssot["confusion_matrix"]["fn"] == 14
    assert ssot["tpr_recall"] == 0.3000
    assert ssot["fpr"] == 0.0000
    assert ssot["balanced_accuracy"] == 0.6500
    assert ssot["macro_f1"] == 0.6011


def test_mathematical_consistency_across_all_configurations():
    data = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
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


def test_correction_report_and_table_parity():
    assert REPORT_PATH.exists()
    assert RECON_REPORT_PATH.exists()

    report_text = REPORT_PATH.read_text(encoding="utf-8")
    recon_text = RECON_REPORT_PATH.read_text(encoding="utf-8")

    # Confirm required rectification sections exist in report
    assert "DISC-TXT-01" in report_text
    assert "TP=6, TN=20, FP=0, FN=14" in report_text
    assert "F1=0.6011" in report_text

    # Confirm RECON_REPORT_PATH has explicit confusion matrix counts matching SSOT
    assert "| **A0-visual** | C0 | P0 |  6 | 20 |  0 | 14 |" in recon_text
    assert "| **A0-visual** | C0 | **P1** | 11 | 17 |  3 |  9 |" in recon_text


def test_aiforge_doc_intake_protocol_and_blocker_audit():
    assert AIFORGE_PROTO_PATH.exists()
    proto_text = AIFORGE_PROTO_PATH.read_text(encoding="utf-8")
    assert "Scam-AI/AIForge-Doc-v1" in proto_text
    assert "40 cặp nguồn" in proto_text
    assert "TestingSet" in proto_text
    assert "FAIL-CLOSED" in proto_text

    assert AIFORGE_AUDIT_PATH.exists()
    audit_data = json.loads(AIFORGE_AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit_data["target_dataset"] == "Scam-AI/AIForge-Doc-v1"
    assert audit_data["file_access_result"]["http_status_code"] == 401
    assert audit_data["epistemic_determination"] == "AUTHENTICATION_REQUIRED_GATED_DATASET"


if __name__ == "__main__":
    test_reporting_correction_receipt_integrity()
    test_mathematical_consistency_across_all_configurations()
    test_correction_report_and_table_parity()
    test_aiforge_doc_intake_protocol_and_blocker_audit()
    print("ALL 4 REPORTING RECONCILIATION & AIFORGE AUDIT TESTS PASSED.")
