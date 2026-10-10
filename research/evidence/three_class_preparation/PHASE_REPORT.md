# Phase Report: Three-Class RQ1 & RQ2 Experiment Preparation

> **Phase**: Three-Class RQ1 & RQ2 Experiment Preparation (Phase 4C.8 trace)<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Execution Date**: 2026-10-11<br>
> **Status**: `PREPARATION_COMPLETED_DATA_BLOCKED`

---

## 1. Mục tiêu & Bối cảnh
Chuẩn bị toàn diện cơ sở thực nghiệm cho hai câu hỏi nghiên cứu cốt lõi:
- **RQ1 (Three-Class Generalization)**: Khả năng phân biệt đồng thời 3 lớp `authentic`, `ai_edited`, và `fully_generated` trên kiến trúc thị giác nhẹ (MobileNetV3).
- **RQ2 (Multimodal Evidence Fusion)**: Đóng góp của đặc trưng DSP canonical 16-D khi kết hợp với nhánh thị giác so với visual-only baseline trong không gian 3 lớp dưới các phép biến đổi nén và suy thoái ảnh.

---

## 2. Kết quả Kiểm kê Dữ liệu Thực tế trên Local (Task 1)
- **Tổng số ảnh quét và giải mã thực tế (Pillow decode + SHA-256 byte stream)**: 6,156 file ảnh (5,890,096,035 bytes) từ `data/research/tgif/` và `research/reference_samples/`.
- **Phân bổ theo lớp và tính hợp lệ**:
  - `authentic`: 341 ảnh eligible development (từ MS-COCO val2017).
  - `ai_edited`: 341 ảnh eligible development (từ TGIF SD2 inpainting spliced `sd2-sp/validation`).
  - `fully_generated`: **0 ảnh trên LOCAL** (trạng thái chính thức: `DATA_BLOCKED: MISSING_FULLY_GENERATED_COHORT`).
- **Niêm phong bảo vệ**:
  - 343 locked-test sources (3,087 ảnh) giữ nguyên seal SHA-256 `519e7a0e...`.
  - TGIF N=400 clean subset (800 ảnh) giữ nguyên cách ly phục vụ kiểm định độc lập.
- **Hiện vật xuất bản**:
  - `research/evidence/three_class_preparation/local_dataset_inventory.json`
  - `research/evidence/three_class_preparation/candidate_cohort_manifest.json` (6,156 bản ghi, 13 trường chuẩn tắc)
  - `research/evidence/three_class_preparation/candidate_cohort_manifest.csv`

---

## 3. Kết quả Kiểm toán Rò rỉ & Yếu tố Gây nhiễu (Task 2)
Thực thi kiểm toán độc lập qua `scripts/research/audit_leakage_and_confounding.py`:
- **Source Group Integrity**: 341/341 nhóm nguồn hợp lệ có đủ cặp authentic ↔ edited (PASS).
- **Exact Hash Collisions**: 0 va chạm SHA-256 ngoài các biến thể cùng nguồn (PASS).
- **Perceptual Near-Duplicates**: 0 va chạm dHash 64-bit Hamming distance $\le 3$ giữa các nguồn khác nhau (PASS).
- **Cross-Cohort Isolation**: 0 rò rỉ giữa development cohort với locked-test hoặc TGIF N=400 (PASS).
- **Phân tích Confounding Factors**:
  - Định dạng: 100% PNG cho cả hai lớp (không có format bias).
  - Kích thước & Tỷ lệ khung hình: Phân bố dị thể (heterogeneous), cần quy chuẩn xử lý ảnh trong protocol để tránh shortcut learning.
- **Biên nhận**: `research/evidence/three_class_preparation/leakage_and_confounding_report.json` (`verdict: LEAKAGE_AUDIT_PASS`).

---

## 4. Đặc tả Protocol Thực nghiệm 3 Lớp & Preflight (Task 3)
- **Cấu hình protocol**: `ml/configs/three_class_evaluation_protocol.yaml`
  - Class taxonomy: `0: authentic`, `1: ai_edited`, `2: fully_generated`.
  - Hai nhánh baseline đối chứng:
    1. Baseline A: `visual_only_multiclass` (576-d MobileNetV3-small backbone).
    2. Baseline B: `visual_dsp_multiclass` (576-d visual + 16-d canonical DSP).
  - Metric bắt buộc: Macro-F1 (unweighted), per-class P/R/F1, Balanced Accuracy, Multiclass Brier Score, ECE (10 bins), Confusion Matrix (3x3).
  - Trạng thái split: `DRAFT_PENDING_ACQUISITION` do thiếu lớp `fully_generated`.
- **Preflight dry-run**: Chạy thử mô phỏng trên fixture tổng hợp deterministic qua `scripts/research/preflight_three_class_protocol.py`:
  - 100% metric functions kiểm chứng thành công.
  - Zero training, zero model refit.
  - Biên nhận: `research/evidence/three_class_preparation/protocol_preflight_receipt.json`.

---

## 5. Kế hoạch Tiếp nhận GenImage & Adapter (Task 4)
- **Acquisition Plan**: `datasets/acquisition-plans/pilot-c-genimage-fully-generated.v1.json`
  - Nguồn chuẩn tắc: GenImage SD v1.4 (Zhu et al., NeurIPS 2023, CC BY-NC-SA 4.0 research-only).
  - Trạng thái phê duyệt: `pending-user-approval`.
  - Dung lượng ổ đĩa: 65 GB khả dụng (> 24 GB yêu cầu) -> DISK PASS.
  - Tuân thủ Rule 3.1: Tuyệt đối không tự động tải dữ liệu khi chưa có sự xác nhận của người dùng.
- **Adapter preflight**: `scripts/research/acquire_genimage_candidate_sample.py`
  - Biên nhận dry-run: `research/evidence/three_class_preparation/genimage_acquisition_dryrun_receipt.json`.

---

## 6. Mở rộng Y Văn & Khoảng Trống Nghiên Cứu (Task 5)
Cập nhật `docs/LITERATURE_RESEARCH_GAP.md`:
- Bổ sung 2 ứng viên khoảng trống nghiên cứu gắn trực tiếp với RQ1 và RQ2:
  1. *Ứng viên 1 (RQ1)*: Phân loại đồng thời 3 lớp trên kiến trúc tích chập gọn nhẹ (<5M params).
  2. *Ứng viên 2 (RQ2)*: Đóng góp của đặc trưng DSP canonical 16-D trong không gian 3 lớp khi đối mặt đồng thời với inpainting và text-to-image toàn ảnh.
- Nâng tổng số ứng viên khoảng trống cần xác minh lên 5.

---

## 7. Tổng kết Kiểm thử & Tính Toàn Vẹn
- **Pytest**: 15/15 ML unit tests PASS (`test_three_class_protocol.py`, `test_dataset_inventory_audit.py`, `test_leakage_and_confounding_audit.py`, `test_genimage_acquisition_plan.py`).
- **Vitest & Continuity**: 80/80 tests PASS (67 JS tests qua 6 packages/apps + 13 continuity tests).
- **Continuity Check**: `node scripts/continuity-check.mjs` -> `CONTINUITY_CHECK: PASS`.
- **Git diff cleanliness**: `git diff --check` sạch hoàn toàn, không có whitespace lỗi hay conflict markers.
