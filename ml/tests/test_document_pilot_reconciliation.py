"""
ml/tests/test_document_pilot_reconciliation.py

Unit tests for document pilot reconciliation, taxonomy/compression audit,
and paired replay artifacts.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TAXONOMY_PATH = REPO_ROOT / "research/evidence/document_pilot/document_sample_taxonomy_audit.json"
FORMAT_AUDIT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_format_compression_audit.json"
REPLAY_PRED_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_replay_predictions.json"
REPLAY_RECEIPT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_replay_receipt.json"
RECON_REPORT_PATH = REPO_ROOT / "research/evidence/document_pilot/DOCUMENT_PILOT_RECONCILIATION_REPORT.md"
AMENDMENT_PATH = REPO_ROOT / "research/evidence/three_class_preparation/PROTOCOL_LOCALIZATION_PREPARATION_AMENDMENT_V1.1.0.md"


def test_taxonomy_audit_integrity():
    assert TAXONOMY_PATH.exists(), f"Missing {TAXONOMY_PATH}"
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert data["total_samples"] == 40
    counts = data["category_counts"]
    assert counts["verified_ai_manipulation"] == 0
    assert counts["verified_non_ai_manipulation"] == 0
    assert counts["manipulation_provenance_unverified"] == 20
    assert counts["authentic_pristine"] == 20

    # Verify each sample has required fields
    for s in data["samples"]:
        assert "sample_id" in s
        assert "provenance_category" in s
        assert "document_domain_type" in s
        assert "manipulation_type" in s
        assert "ai_manipulation_evidence" in s
        assert "source_relation_status" in s
        assert "unpaired" in s["source_relation_status"]


def test_format_and_compression_audit():
    assert FORMAT_AUDIT_PATH.exists(), f"Missing {FORMAT_AUDIT_PATH}"
    data = json.loads(FORMAT_AUDIT_PATH.read_text(encoding="utf-8"))
    assert data["total_samples"] == 40
    findings = data["confounding_findings"]
    assert "100%" in findings["format_label_correlation"]
    assert "isolation_audit_rectification" in findings

    samples = data["samples"]
    forged_samples = [s for s in samples if s["source_label"] == "black"]
    auth_samples = [s for s in samples if s["source_label"] == "white"]
    assert len(forged_samples) == 20
    assert len(auth_samples) == 20

    for fs in forged_samples:
        assert fs["magic_format"] == "JPEG"
        assert fs["jpeg_subsampling"] == "4:2:0"
        assert fs["jpeg_q0_mean"] == 5.77

    for as_sample in auth_samples:
        assert as_sample["magic_format"] == "PNG"
        assert as_sample["mode"] == "RGB"


def test_replay_predictions_and_receipt():
    assert REPLAY_PRED_PATH.exists(), f"Missing {REPLAY_PRED_PATH}"
    assert REPLAY_RECEIPT_PATH.exists(), f"Missing {REPLAY_RECEIPT_PATH}"

    receipt = json.loads(REPLAY_RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["total_samples"] == 40
    assert "C0" in receipt["conditions"]
    assert "C1" in receipt["conditions"]

    # Verify recalculated historical baseline matches exactly
    recalc = receipt["recalculated_historical_baseline"]
    assert "A0-visual" in recalc
    assert recalc["A0-visual"]["P1"]["macro_f1"] == 0.6931
    assert recalc["A0-visual"]["P1"]["balanced_accuracy"] == 0.7000
    assert recalc["A0-visual"]["P1"]["tpr_recall"] == 0.5500
    assert recalc["A0-visual"]["P1"]["tnr"] == 0.8500
    assert recalc["A0-visual"]["P1"]["fpr"] == 0.1500

    # Verify C1 vs C0 label flips documented
    flips = receipt["c1_vs_c0_label_flips"]
    assert "A0-visual" in flips
    assert flips["A0-visual"]["P1"]["total_flips"] == 0
    assert flips["A0-visual"]["P1"]["flip_rate"] == 0.0

    # Verify latency breakdown separates shared feature extraction and per-model scoring
    lat = receipt["isolated_latency_breakdown"]
    for geom in ["P0", "P1", "P2"]:
        assert "preprocessing" in lat[geom]
        assert "shared_feature_extraction" in lat[geom]
        assert "per_model_scoring" in lat[geom]
        assert "end_to_end_total_pipeline" in lat[geom]
        # Per model scoring must be under 2 ms
        for m, m_lat in lat[geom]["per_model_scoring"].items():
            assert m_lat["median_ms"] < 2.0


def test_reconciliation_report_and_amendment():
    assert RECON_REPORT_PATH.exists()
    recon_text = RECON_REPORT_PATH.read_text(encoding="utf-8")
    assert "P1 tối ưu vượt trội" in recon_text
    assert "P1 bảo toàn 100% dấu vết forensic" in recon_text
    assert "spectral leakage" in recon_text
    assert "A1 loại bỏ 100% báo động giả" in recon_text

    assert AMENDMENT_PATH.exists()
    amend_text = AMENDMENT_PATH.read_text(encoding="utf-8")
    assert "Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI" in amend_text
    assert "250 ảnh TGIF SD2 inpainting edited" in amend_text
    assert "91 ảnh TGIF SD2 edited" in amend_text
    assert "Xóa bỏ hoàn toàn cam kết tự động huấn luyện U-Net" in amend_text


if __name__ == "__main__":
    test_taxonomy_audit_integrity()
    test_format_and_compression_audit()
    test_replay_predictions_and_receipt()
    test_reconciliation_report_and_amendment()
    print("ALL 4 DOCUMENT PILOT RECONCILIATION TESTS PASSED.")
