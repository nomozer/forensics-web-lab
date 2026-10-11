# Document Pilot Reporting Reconciliation & AIForge-Doc Protocol Implementation Plan

> **Mã kế hoạch**: `PLAN-2026-10-12-DOCUMENT-PILOT-REPORTING-RECONCILIATION`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Trạng thái**: PRE-REGISTERED & READY FOR EXECUTION

---

## 1. Mục Tiêu & Ranh Giới Nghiên Cứu

1. **Khắc phục triệt để mâu thuẫn báo cáo (Reporting Discrepancy Reconciliation)**:
   - Truy nguyên nguồn gốc mâu thuẫn giữa báo cáo TXT (chat text/stdout), báo cáo Markdown (`DOCUMENT_PILOT_RECONCILIATION_REPORT.md`), và raw outputs (`document_pilot_predictions.json`, `document_pilot_replay_predictions.json`).
   - Xây dựng một pipeline chuẩn hóa duy nhất tái tính toán 100% metrics từ raw predictions trên đĩa:
     * Ma trận nhầm lẫn (TP, TN, FP, FN);
     * Precision, Recall (TPR), TNR, FPR, Balanced Accuracy;
     * Binary Macro-F1 ($F_{1,\text{macro}} = (F_{1,\text{pos}} + F_{1,\text{neg}})/2$);
     * AUROC, AP (Average Precision);
     * Binary Brier Score (so với mốc không có thông tin 0.25);
     * ECE 10-bin;
     * Paired bootstrap 10,000 replicate CIs.
   - Tự động sinh bảng số liệu và báo cáo sửa đổi từ cùng một nguồn dữ liệu duy nhất (zero hand-transcription).
   - Xuất bản báo cáo đính chính version 1.1.0 (`DOCUMENT_PILOT_REPORTING_CORRECTION_V1.1.0.md`) và biên nhận sửa đổi (`document_pilot_reporting_correction_receipt.json`), bảo toàn nguyên vẹn artifacts cũ.

2. **Khảo sát Nguồn Sơ Cấp Tài Liệu AI (AIForge-Doc) & Hồ Sơ Điểm Chặn**:
   - Khảo sát bộ dữ liệu `Scam-AI/AIForge-Doc-v1` và `v2` (arXiv:2602.20569, CC-BY-NC-SA 4.0):
     * Xác minh nguồn gốc AI (inpainting bằng Gemini 2.5 Flash Image, Ideogram v2, GPT-Image-2 trên hóa đơn CORD, SROIE, WildReceipt, XFUND).
     * Xác minh cấu trúc ghép cặp authentic $\to$ tampered có ground-truth mask.
     * Xác định rào cản truy cập: Gated dataset (`gated: auto`, HTTP 401 Unauthorized khi tải ảnh không có token xác thực).
   - Xây dựng protocol tiếp nhận đã khóa (`PROTOCOL_AIFORGE_DOC_INTAKE_V1.0.0.md`), pipeline tiền xử lý, script tải mẫu hợp lệ khi có token, và hồ sơ nguồn có thể kiểm chứng.
   - Báo cáo trung thực tình trạng authentication wall và không bịa đặt số liệu hay tải dữ liệu trái phép.

3. **Đính chính các phát biểu vượt bằng chứng theo Amendment v1.1.0**:
   - Localization là hướng ứng viên khám phá, không bắt buộc;
   - Không đồng nhất `black`/forged với AI-edited khi chưa xác minh provenance;
   - Không đồng nhất 0 byte collisions với độc lập nguồn;
   - Bóc tách latency: thời gian classifier head (< 0.2 ms) vs end-to-end pipeline (226 - 826 ms).

---

## 2. Danh Mục File & Trách Nhiệm

| File | Trách nhiệm |
| :--- | :--- |
| `scripts/research/reconcile_document_pilot_reporting.py` | Script chuẩn tắc đọc raw predictions, tính toán toán học thống nhất toàn bộ metrics, kiểm tra tính tự nhất quán, và xuất bảng kết quả tự động. |
| `research/evidence/document_pilot/document_pilot_reporting_correction_receipt.json` | Biên nhận máy chính thức ghi nhận run_id, sha256 raw predictions, bảng metrics đã sửa, và đối soát chi tiết từng dòng. |
| `research/evidence/document_pilot/DOCUMENT_PILOT_REPORTING_CORRECTION_V1.1.0.md` | Báo cáo đính chính chuyên sâu, giải thích chi tiết nguồn gốc mâu thuẫn, thay thế bảng số liệu cũ bằng bảng tự động sinh. |
| `research/evidence/document_pilot/PROTOCOL_AIFORGE_DOC_INTAKE_V1.0.0.md` | Protocol chuẩn hóa quy định điều kiện chọn mẫu 40 cặp (80 ảnh), sampling tất định, leakage guards, và runner pipeline cho AIForge-Doc. |
| `scripts/research/acquire_aiforge_doc_panel.py` | Script tiếp nhận có kiểm soát cho AIForge-Doc hỗ trợ `--token` hợp lệ và chế độ fail-closed khi thiếu quyền truy cập. |
| `research/evidence/document_pilot/aiforge_doc_access_blocker_audit.json` | Hồ sơ kiểm toán rào cản truy cập cho AIForge-Doc ghi nhận URL, license, gated header, HTTP 401 status. |
| `ml/tests/test_document_pilot_reporting_reconciliation.py` | Bộ test tự động kiểm chứng tính nhất quán toán học (TP+TN+FP+FN=40, F1 khớp formula, zero table divergence). |
| Continuity files (`CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`) | Đồng bộ trạng thái nghiên cứu theo đúng continuity contract. |

---

## 3. Các Bước Thực Hiện Chi Tiết

1. **Bước 1**: Lập trình `scripts/research/reconcile_document_pilot_reporting.py` để đọc `document_pilot_predictions.json` và `document_pilot_replay_predictions.json`, tính toán toàn bộ chỉ số theo công thức chuẩn tắc.
2. **Bước 2**: Thực thi script, xuất biên nhận `document_pilot_reporting_correction_receipt.json` và bảng Markdown chuẩn.
3. **Bước 3**: Soạn thảo báo cáo đính chính `DOCUMENT_PILOT_REPORTING_CORRECTION_V1.1.0.md`.
4. **Bước 4**: Xây dựng protocol và script tiếp nhận AIForge-Doc (`PROTOCOL_AIFORGE_DOC_INTAKE_V1.0.0.md`, `acquire_aiforge_doc_panel.py`, `aiforge_doc_access_blocker_audit.json`).
5. **Bước 5**: Viết bộ test `ml/tests/test_document_pilot_reporting_reconciliation.py` và chạy xác minh.
6. **Bước 6**: Chạy full test suites (`pnpm test`, `pytest ml/tests`, `pnpm continuity:check -- --staged`).
7. **Bước 7**: Cập nhật continuity, commit và push lên `research/independent-cohort-acquisition`.
