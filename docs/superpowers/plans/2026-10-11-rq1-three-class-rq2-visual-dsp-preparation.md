# RQ1 Three-Class & RQ2 Visual/DSP Experimental Preparation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuẩn bị toàn diện cơ sở thực nghiệm cho RQ1 (phân loại ba lớp authentic / ai_edited / fully_generated) và RQ2 (đối chứng visual-only vs visual + DSP): kiểm kê dữ liệu thực trên LOCAL, xác minh provenance nhãn, kiểm tra rò rỉ (leakage) và yếu tố gây nhiễu (confounding), lập manifest cohort dự kiến, thiết lập protocol thực nghiệm 3 lớp có preflight/dry-run runnable, chuẩn bị adapter/acquisition plan cho lớp fully_generated đang thiếu, cập nhật ma trận khoảng trống y văn, và hoàn thành thủ tục continuity.

**Architecture:** 
1. **Kiểm kê & Manifest**: Viết script kiểm toán dữ liệu thực trên LOCAL qua Pillow decode thật và SHA-256 byte stream; trích xuất toàn bộ nguồn ảnh hợp lệ, phân loại nhãn chuẩn tắc theo provenance (MS-COCO -> `authentic`, TGIF SD2-sp -> `ai_edited`, TGIF fr -> `quarantined_conditional_regeneration`, `fully_generated` -> ghi nhận hiện trạng thiếu); lập file manifest JSON/CSV đầy đủ 12 trường bắt buộc.
2. **Kiểm tra Rò rỉ & Nhiễu**: Xây dựng script kiểm tra trùng lặp exact (SHA-256) và tri giác (64-bit dHash Hamming <= 3); kiểm tra cách ly hoàn toàn với TGIF N=400 và locked-test 343 sources đã niêm phong; phân tích phân bố độ phân giải, tỷ lệ nén, kênh màu và metadata giữa các lớp để cảnh báo rủi ro confounding.
3. **Protocol Thực nghiệm 3 Lớp & Preflight**: Xây dựng protocol YAML/Python đóng băng class mapping (0: authentic, 1: ai_edited, 2: fully_generated), metrics đánh giá (Macro-F1, per-class P/R/F1, Balanced Accuracy, Multiclass Brier, ECE 10 bins, Confusion Matrix), định nghĩa 2 nhánh baseline (visual-only vs visual+DSP); tích hợp script preflight/dry-run có khả năng xác thực dữ liệu và chạy thử trên synthetic fixtures mà không vi phạm quy tắc cấm train.
4. **Acquisition Plan & Adapter cho `fully_generated`**: Xây dựng acquisition plan chuẩn schema cho tập ảnh sinh toàn phần từ tài liệu chính thức (GenImage SDv1.4 / SDXL), chuẩn bị script adapter kiểm tra tính khả thi khi tải mẫu nhỏ có kiểm soát.
5. **Cập nhật Y Văn & Khoảng Trống**: Mở rộng `docs/LITERATURE_RESEARCH_GAP.md` tập trung vào bài toán 3 lớp, mô hình nhẹ, và generalization giữa các generator.
6. **Kiểm chứng & Continuity**: Viết unit test cho các script mới, chạy `pnpm test`, `pnpm continuity:check`, `git diff --check`, cập nhật `CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`, commit và push lên branch `research/independent-cohort-acquisition`.

**Tech Stack:** Python 3.12, Pillow, NumPy, SciPy, PyTorch (inference/extractor), Vitest, TypeScript, Git.

**Global Constraints:**
- Tuyệt đối không tự ý huấn luyện, refit, QAFT, generation hoặc tải hàng loạt dataset mới.
- Không sửa/push main hoặc deploy trong phiên làm việc này.
- Phải giữ nguyên 100% weights, bindings, thresholds và receipts của các phase lịch sử.
- TGIF N=400 và locked-test 343 sources là read-only, cấm đưa vào development.
- Nếu thiếu `fully_generated`, báo cáo trung thực trạng thái `DATA_BLOCKED`, không tạo split giả.

---

### Task 1: Audit & Inventory Local Real Data (Section C)
**Files:**
- Create: `scripts/research/audit_local_dataset_inventory.py`
- Test: `ml/tests/test_dataset_inventory_audit.py`
- Output: `research/evidence/three_class_preparation/local_dataset_inventory.json`
- Output: `research/evidence/three_class_preparation/candidate_cohort_manifest.json`
- Output: `research/evidence/three_class_preparation/candidate_cohort_manifest.csv`

- [x] **Step 1: Viết script `audit_local_dataset_inventory.py`**
  - Quét toàn bộ file ảnh trong `data/research/tgif/` và `research/reference_samples/`.
  - Kiểm tra decode thực tế bằng Pillow, lấy kích thước (width, height), format (PNG/JPEG), mode (RGB/RGBA).
  - Tính toán SHA-256 chính xác từ byte stream.
  - Xác minh provenance theo bảng đối chiếu:
    * `orig/validation` -> `authentic` (MS-COCO 2014 validation)
    * `orig/testing` -> `authentic` (MS-COCO 2014 test/val, locked-test cohort)
    * `sd2-sp/validation` -> `ai_edited` (TGIF SD2 inpainting spliced)
    * `sd2-sp/testing` -> `ai_edited` (TGIF SD2 inpainting spliced, locked-test cohort)
    * `fr` (nếu có) -> `excluded: conditional_regeneration_quarantined`
  - Đánh dấu eligibility (`ELIGIBLE_DEVELOPMENT`, `SEALED_LOCKED_TEST`, `SEALED_INDEPENDENT_EVALUATION`, `EXCLUDED_CORRUPTED`, `EXCLUDED_UNVERIFIED_PROVENANCE`).
  - Ghi nhận `fully_generated` = 0 mẫu trên LOCAL.
  - Xuất file inventory và manifest đầy đủ 12 trường bắt buộc.

- [x] **Step 2: Viết unit test `test_dataset_inventory_audit.py`**
  - Kiểm tra khả năng bắt lỗi ảnh hỏng, nhãn sai, trùng hash, và loại trừ `fr`.
  - Chạy pytest xác nhận PASS.

- [x] **Step 3: Thực thi audit và xuất artifacts**
  - Chạy `ml/.venv/Scripts/python scripts/research/audit_local_dataset_inventory.py`.
  - Xác nhận artifacts được ghi đúng cấu trúc schema.

---

### Task 2: Leakage & Confounding Audit (Section D)
**Files:**
- Create: `scripts/research/audit_leakage_and_confounding.py`
- Test: `ml/tests/test_leakage_and_confounding_audit.py`
- Output: `research/evidence/three_class_preparation/leakage_and_confounding_report.json`

- [x] **Step 1: Viết script `audit_leakage_and_confounding.py`**
  - Nhóm ảnh theo `source_group_id` (đảm bảo authentic và toàn bộ biến thể edited của cùng ảnh gốc có chung group ID).
  - Kiểm tra exact duplicates (SHA-256 collisions).
  - Kiểm tra perceptual duplicates qua 64-bit dHash (Hamming distance <= 3).
  - Kiểm tra cách ly với TGIF N=400 (`research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json`) và locked-test (`research/evidence/phase-4b.2/split-lock.json`).
  - Thống kê các yếu tố gây nhiễu (confounding factors): phân bố chiều rộng/cao, tỷ lệ khung hình, định dạng tệp, profile nén JPEG, và kênh siêu dữ liệu giữa các lớp.
  - Phân tích rủi ro shortcut learning (mô hình học đặc trưng định dạng/kích thước thay vì dấu vết AI).

- [x] **Step 2: Viết unit test và thực thi**
  - Kiểm tra phát hiện rò rỉ nguồn chéo và trùng lặp.
  - Thực thi và xuất báo cáo `leakage_and_confounding_report.json`.

---

### Task 3: 3-Class & Visual/DSP Protocol Specification & Preflight (Section E)
**Files:**
- Create: `ml/configs/three_class_evaluation_protocol.yaml`
- Create: `scripts/research/preflight_three_class_protocol.py`
- Test: `ml/tests/test_three_class_protocol.py`
- Output: `research/evidence/three_class_preparation/protocol_preflight_receipt.json`

- [x] **Step 1: Định nghĩa Protocol `three_class_evaluation_protocol.yaml`**
  - Khóa class mapping: `0: authentic`, `1: ai_edited`, `2: fully_generated`.
  - Phân tách rõ ràng:
    * Frozen: MobileNetV3-small backbone FP32 ImageNet weights (`047dcff4...`), trích xuất 576-d feature vector; 16 đặc trưng DSP chuẩn tắc (`extractCanonicalDspFeatures16`).
    * Trainable: Multiclass Softmax / Logistic Regression probe (cho visual-only) và Multiclass Stacker / Combined probe (cho visual + DSP).
  - Định nghĩa 2 baseline đối chứng:
    1. Baseline A: `visual_only_multiclass` (576-d visual features).
    2. Baseline B: `visual_dsp_multiclass` (576-d visual + 16-d DSP features).
  - Định nghĩa metrics 3 lớp: Macro-F1 (không trọng số), Per-class Precision/Recall/F1, Balanced Accuracy (trung bình Recall từng lớp), Multiclass Brier score, Multiclass Expected Calibration Error (ECE với 10 bins phân bố xác suất cao nhất), và Confusion Matrix (3x3).
  - Ghi nhận trạng thái dữ liệu: `DATA_BLOCKED: MISSING_FULLY_GENERATED_COHORT`. Protocol được khóa phần kiến trúc/metrics, và giữ phần split ở trạng thái `DRAFT_PENDING_ACQUISITION`.

- [x] **Step 2: Viết script `preflight_three_class_protocol.py`**
  - Nạp protocol và manifest.
  - Xác nhận class taxonomy.
  - Chạy mô phỏng dry-run trên fixture giả lập (`ml/tests/fixtures/smoke_generator.py`) để kiểm chứng toàn bộ luồng tính toán metrics 3 lớp và confusion matrix mà không cần dữ liệu train thật.
  - Xuất biên nhận `protocol_preflight_receipt.json` với status `PREFLIGHT_PASS_DATA_BLOCKED`.

- [x] **Step 3: Viết unit test và chạy thử**
  - Viết `test_three_class_protocol.py`.
  - Chạy preflight và xác nhận PASS.

---

### Task 4: Acquisition Plan & Adapter for `fully_generated` (Section E)
**Files:**
- Create: `datasets/acquisition-plans/pilot-c-genimage-fully-generated.v1.json`
- Create: `scripts/research/acquire_genimage_candidate_sample.py`
- Test: `ml/tests/test_genimage_acquisition_plan.py`

- [x] **Step 1: Soạn thảo Acquisition Plan `pilot-c-genimage-fully-generated.v1.json`**
  - Tuân thủ JSON schema `docs/schemas/acquisition-plan.v1.schema.json`.
  - Xác định nguồn tài liệu chính thức: GenImage (Zhu et al., NeurIPS 2023, arXiv:2306.09008, https://github.com/GenImage-Dataset/GenImage).
  - Chọn generator ứng viên: Stable Diffusion v1.4 hoặc SDXL text-to-image để có cùng họ Diffusion với SD2 inpainting, giảm thiểu confounding về kiến trúc tạo sinh.
  - Thiết lập dung lượng trần (budget), mã hash sha256 dự kiến, điều kiện cấp phép (CC BY-NC-SA 4.0), và cơ chế safe extraction.

- [x] **Step 2: Viết script adapter và test**
  - Viết adapter mock/dry-run kiểm tra tính tương thích cấu trúc và schema validation.
  - Đảm bảo tuân thủ nguyên tắc không tải dữ liệu lớn (>50MB) khi chưa có lệnh người dùng.

---

### Task 5: Literature & Research Gap Analysis Update (Section F)
**Files:**
- Modify: `docs/LITERATURE_RESEARCH_GAP.md`
- Output: Cập nhật bảng đối chiếu công trình sơ cấp và phân định khoảng trống 3 lớp.

- [x] **Step 1: Khảo sát nguồn sơ cấp**
  - Xác minh các công trình liên quan đến 3 lớp (authentic / fully_generated / ai_edited): GenImage (Zhu et al. 2023), TGIF/TGIF2 (Mareen et al. 2024, 2026), DIRE (Wang et al. 2023), LAID (Chivaran & Ni 2025), Universal Fake Detectors (Ojha et al. 2023).
  - Xác minh tác giả, năm, venue, số lớp và giới hạn thực tế.
- [x] **Step 2: Cập nhật `docs/LITERATURE_RESEARCH_GAP.md`**
  - Bổ sung bảng ma trận đối chiếu 3 lớp.
  - Phân định rạch ròi 3 nhóm: Well-studied, Not observed, Candidate gaps to verify.
  - Gắn trực tiếp ứng viên khoảng trống với RQ1 và RQ2 trong protocol.

---

### Task 6: Verification, Continuity & Handover (Section G)
**Files:**
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [x] **Step 1: Chạy toàn bộ test suites và kiểm tra chất lượng**
  - `pnpm test` (vitest + continuity suites).
  - `pytest ml/tests/`.
  - `node scripts/continuity-check.mjs`.
  - `git diff --check`.
- [x] **Step 2: Cập nhật tài liệu continuity**
  - Ghi nhận trạng thái Phase 4C.9: Preparation for RQ1 (3-Class) & RQ2 (Visual/DSP).
  - Cập nhật CODE_INDEX với các script, config và receipt mới.
  - Thêm mục tóm tắt (<20 dòng) vào STATUS_LEDGER.
- [ ] **Step 3: Commit & Push lên branch `research/independent-cohort-acquisition`**
  - Commit thông điệp chuẩn mực.
  - Push lên `origin/research/independent-cohort-acquisition`.
- [ ] **Step 4: Lập báo cáo bàn giao chi tiết trả lời đầy đủ 7 câu hỏi của Section G**.
