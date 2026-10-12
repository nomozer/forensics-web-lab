# Document Reporting Reconciliation v1.2.0 & Public Candidate Accessibility Audit Implementation Plan

> **Mã kế hoạch**: `PLAN-2026-10-12-DOCUMENT-REPORTING-V1.2.0-AND-DATA-AUDIT`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Trạng thái**: PRE-REGISTERED & READY FOR EXECUTION

---

## 1. Mục Tiêu & Ranh Giới Nghiên Cứu

1. **Khắc phục triệt để sai lệch tính toán trong Discrepancy Ledger (Reporting Reconciliation v1.2.0)**:
   - Thay thế việc ghi nhận thủ công xấp xỉ bằng thuật toán tính toán thống nhất: hàm `compute_confusion_metrics(tp, tn, fp, fn)` tính tự động $100\%$ các chỉ số tỷ lệ và Macro-F1 từ số đếm ô nhầm lẫn.
   - Sửa chính xác các giá trị ngụ ý trong `discrepancy_ledger`:
     * DISC-TXT-02: TP=7, TN=17, FP=3, FN=13 $\implies$ binary Macro-F1 = $0.573333\dots$ (thay vì làm tròn thô $0.5900$).
     * DISC-TXT-03: TP=5, TN=16, FP=4, FN=15 $\implies$ binary Macro-F1 = $0.486139\dots$ (thay vì $0.4900$).
     * DISC-TXT-04: TP=10, TN=11, FP=9, FN=10 $\implies$ binary Macro-F1 = $0.524703\dots$ (thay vì $0.5100$).
   - Bổ sung unit test bắt buộc để phát hiện bất kỳ sự không khớp nào giữa số đếm ma trận nhầm lẫn và Macro-F1 trong ledger.
   - Chuẩn hóa phạm vi khẳng định nhận thức:
     * Tính nhất quán của số đếm ma trận nhầm lẫn chỉ chứng minh tính tự nhất quán số học trong bảng, không chứng minh toàn bộ pipeline và probabilities đều đúng.
     * Số đã làm tròn không gọi là “bit-exact” với giá trị đầy đủ.
     * Không gọi paired image bootstrap là source-cluster bootstrap khi quan hệ nguồn chưa xác minh.
     * Bootstrap positive fraction là tỷ lệ mẫu bootstrap có delta dương, không phải p-value.
     * Bỏ hoặc ghi rõ: Multiclass Brier trên nhãn forged nhị phân là phép chiếu giả định tham khảo, không phải đánh giá 3 lớp AI hợp lệ.
   - Xuất bản `document_pilot_reporting_correction_receipt_v1.2.0.json` và `DOCUMENT_PILOT_REPORTING_CORRECTION_V1.2.0.md`, bảo toàn nguyên vẹn phiên bản v1.1.0 cũ.

2. **Kiểm toán khả năng tiếp cận và phân loại nguồn gốc 5 ứng viên dữ liệu tài liệu**:
   - Rà soát 5 ứng viên chính thức:
     1. `Scam-AI/AIForge-Doc-v1`
     2. `Scam-AI/gpt4o-receipt`
     3. `MohamedDhouib1/TDoc-2.8M`
     4. `Jason37437/RealText-V2-Syn25k`
     5. `Zegkim/FSTS` (và `qcf-568/DocTamper`)
   - Thử nghiệm tải thực tế không dùng token, kiểm tra URL, license, generator/provenance, pairing, và định dạng.
   - Xuất biên nhận kiểm toán `research/evidence/document_pilot/public_document_candidates_audit.json`.

3. **Bàn giao thiết kế kỹ thuật cho cohort tự xây dựng đối chứng (Fallback Design)**:
   - Do 100% ứng viên GenAI tài liệu đều có gated access/token wall, và 100% ứng viên mở ungated đều là chỉnh sửa truyền thống (non-AI), hệ thống thực hiện fail-closed và bàn giao bản thiết kế kỹ thuật hoàn chỉnh `INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md` phục vụ việc tạo dữ liệu đối chứng có kiểm soát trong tương lai.
   - Tuyệt đối không chạy tạo ảnh hay tải checkpoint trong lượt này.

---

## 2. Danh Mục File & Trách Nhiệm

| File | Trách nhiệm |
| :--- | :--- |
| `scripts/research/reconcile_document_pilot_reporting.py` | Nâng cấp v1.2.0: tính tự động toàn bộ metrics trong ledger và bảng SSOT, chuẩn hóa epistemic scopes, xuất đồng thời Markdown và JSON v1.2.0. |
| `research/evidence/document_pilot/document_pilot_reporting_correction_receipt_v1.2.0.json` | Receipt v1.2.0 ghi nhận bảng metrics và ledger đã sửa chính xác toán học. |
| `research/evidence/document_pilot/DOCUMENT_PILOT_REPORTING_CORRECTION_V1.2.0.md` | Báo cáo đính chính v1.2.0 chuẩn hóa số liệu và giới hạn khẳng định. |
| `scripts/research/audit_public_document_candidates.py` | Script kiểm toán tự động thăm dò metadata và probe download 5 ứng viên dữ liệu. |
| `research/evidence/document_pilot/public_document_candidates_audit.json` | Biên nhận kiểm toán khả năng tiếp cận và provenance của 5 ứng viên. |
| `research/evidence/document_pilot/INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md` | Bản thiết kế hoàn chỉnh cho pilot tự xây dựng dữ liệu đối chứng GenAI. |
| `ml/tests/test_document_pilot_reporting_reconciliation.py` | Mở rộng kiểm thử: kiểm chứng F1 công thức toán học trong ledger, kiểm tra receipt v1.2.0 và audit receipt. |
| Continuity files (`CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`) | Đồng bộ trạng thái nghiên cứu theo Continuity Protocol. |

---

## 3. Các Bước Thực Hiện Chi Tiết

- [ ] **Bước 1**: Nâng cấp `scripts/research/reconcile_document_pilot_reporting.py` để tính toán tự động 100% số liệu trong `discrepancy_ledger` và cập nhật các cảnh báo nhận thức; sinh artifacts v1.2.0.
- [ ] **Bước 2**: Viết script kiểm toán `scripts/research/audit_public_document_candidates.py` và xuất `public_document_candidates_audit.json`.
- [ ] **Bước 3**: Soạn thảo bản thiết kế `INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md`.
- [ ] **Bước 4**: Mở rộng và chạy test suite `ml/tests/test_document_pilot_reporting_reconciliation.py`.
- [ ] **Bước 5**: Chạy toàn bộ quality gates (`pytest`, `pnpm test`, `continuity-check`).
- [ ] **Bước 6**: Cập nhật continuity docs, commit và push lên branch.
