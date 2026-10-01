# Phase 4C.2C.0 — Safely Ingest and Audit Completed Stage 2 Colab Results

## 1. Mục tiêu và Giới hạn (Phase Objectives & Bounds)

Giai đoạn **Phase 4C.2C.0** thực hiện nhập khẩu an toàn bản lưu trữ kết quả hoàn chỉnh Stage 2 từ Google Drive vào kho lưu trữ artifacts cục bộ, kiểm tra toàn vẹn mật mã học, kiểm toán an toàn tệp lưu trữ TAR, giải nén nguyên tử và kiểm toán độc lập toàn diện 15/15 runs trước khi tiến hành phân tích đối chứng Stage 1–Stage 2.

### Các ranh giới bất biến (Invariant Boundaries)
* **Zero Training**: Tuyệt đối không chạy thêm lượt huấn luyện nào (0 training runs).
* **Zero GPU Calls**: Không gọi GPU máy trạm local.
* **Zero Locked-Test Access**: Tuyệt đối không truy cập hoặc giải phóng tập dữ liệu niêm phong `locked_test` (0 evaluations).
* **Zero Stage 1 Modifications**: Bảo toàn nguyên vẹn 100% kết quả và baseline Stage 1.
* **No Git Bloat**: Tuyệt đối không commit archive TAR, checkpoint nhị phân `.pt`, predictions raw hoặc dataset vào Git repository.
* **Pure Ingest/Audit**: Giới hạn nghiêm ngặt ở bước nhập khẩu và kiểm toán; không thực hiện phân tích thống kê đối chuẩn Stage 1–Stage 2 trong phase này.

---

## 2. Kiểm tra Đầu vào Trước Giải nén (Section A: Input Verification)

Bản lưu trữ kết quả hoàn chỉnh Stage 2 được tải từ Google Colab T4 qua Google Drive đã được kiểm tra streaming hash và kích thước nhị phân fail-closed:

* **Tên archive**: `execution_9ee7fdb_complete_results.tar.gz`
* **Kích thước thực tế**: `83,796,910 bytes` (khớp chính xác 100% với `expected_bytes = 83796910`)
* **SHA-256 thực tế**: `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2` (khớp bitwise với `expected_sha256`)
* **Tệp sidecar**: `execution_9ee7fdb_complete_results.tar.gz.sha256`
* **Nội dung sidecar**: `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2  execution_9ee7fdb_complete_results.tar.gz`
* **Kết quả đối soát sidecar**: Khớp 100% giữa sidecar digest, archive digest và canonical expected digest.

---

## 3. Kiểm toán An toàn Archive (Section B: Safe TAR Audit)

Trước khi thực hiện bất kỳ thao tác giải nén nào, toàn bộ 173 mục trong archive đã được duyệt và kiểm toán theo quy chuẩn bảo mật nghiêm ngặt:

| Tiêu chuẩn an toàn | Kiểm tra thực tế | Trạng thái |
| :--- | :--- | :--- |
| **Absolute Paths** | 0 mục bắt đầu bằng `/` hoặc `\` | PASS |
| **Windows Drive Paths** | 0 mục chứa tiền tố ổ đĩa (vd: `C:`, `D:`) | PASS |
| **Path Traversal (`..`)** | 0 mục chứa `..` hoặc ký tự điều hướng | PASS |
| **Symlinks & Hardlinks** | 0 symlinks (`member.issym()`), 0 hardlinks (`member.islnk()`) | PASS |
| **Special Devices** | 0 FIFOs, 0 char devices, 0 block devices, 0 sockets | PASS |
| **Duplicate Entries** | 0 mục trùng lặp đường dẫn | PASS |
| **Null Bytes & Control Chars** | 0 ký tự null `\0` hoặc ký tự điều khiển trong tên | PASS |
| **Supported Types** | 156 regular files (`0o600`), 17 directories (`0o700`) | PASS |

**Phán quyết kiểm toán TAR**: **PASS** (173/173 mục an toàn tuyệt đối).

---

## 4. Giải nén An toàn và Nguyên tử (Section C: Safe Extraction)

* **Thư mục tạm thời (Sibling)**: `execution_9ee7fdb.extracting`
* **Quy trình giải nén**:
  1. Toàn bộ 173 tệp được giải nén vào thư mục tạm `execution_9ee7fdb.extracting`.
  2. Thực hiện kiểm toán toàn diện 15 runs và provenance trên thư mục tạm.
  3. Sau khi toàn bộ các kiểm toán PASS, thực hiện phép đổi tên nguyên tử (`atomic rename` qua `os.replace`) thành `execution_9ee7fdb`.
* **Cơ chế Idempotent Reuse**: Nếu thư mục đích `execution_9ee7fdb` đã tồn tại, công cụ kiểm toán toàn bộ nội dung hiện có; nếu hợp lệ 100%, tái sử dụng mà không giải nén đè hoặc xóa dữ liệu (`reused_existing_target: true`). Nếu phát hiện sai khác, dừng ngay lập tức ở trạng thái fail-closed.

---

## 5. Kiểm toán Chi tiết 15 Runs Stage 2 (Section D: 15-Run Audit)

Toàn bộ ma trận thực nghiệm 15 runs ($N \in \{50, 100, 250\} \times \text{Seeds} \in \{42, 1337, 2025, 3407, 9001\}$) đã được thẩm định độc lập:

### 5.1. Kiểm kê 10 Artifacts Bắt buộc mỗi Run
Mỗi thư mục run chứa đúng 10 artifacts chuẩn tắc, kích thước > 0 bytes:
1. `best_checkpoint.pt`
2. `run_receipt.json`
3. `epoch_history.json`
4. `predictions.json`
5. `training_history.csv`
6. `metrics.json`
7. `environment.json`
8. `environment-binding.json`
9. `trainable-parameter-inventory.json`
10. `checksums.json`

* Tổng số artifacts đã kiểm tra: **150 artifacts** across 15 runs.
* Tệp thiếu: **0**.
* Tệp 0 bytes: **0**.

### 5.2. Toàn vẹn Checksums & Checkpoint Binding
* `checksums.json` không tự băm chính nó.
* `checksums.json` bao phủ chính xác 9 artifacts còn lại trong run.
* 100% kích thước byte và mã băm SHA-256 thực tế trên đĩa khớp hoàn toàn với `checksums.json`.
* `receipt.checkpoint_sha256` khớp bitwise với mã băm thực tế của `best_checkpoint.pt`.
* Lỗi checksum: **0**.

### 5.3. Trạng thái Receipt & Invariants
* `receipt.status`: 15/15 runs ghi nhận `"completed"`.
* `receipt.stage`: 15/15 runs ghi nhận `"partial_finetune"`.
* `receipt.treatment_designation`: `"pre-registered partial fine-tuning protocol"`.
* `receipt.sample_size` và `receipt.seed`: 100% khớp với định danh thư mục.
* `receipt.locked_test_access`: **0** across all 15 runs.
* `receipt.stage1_output_writes`: **0** across all 15 runs.
* `receipt.validation_source_count`: **91** across all 15 runs.
* `receipt.validation_sample_count`: **182** across all 15 runs.

### 5.4. Kiểm toán Cohort Dự đoán & Chống Rò rỉ (Prediction Cohort Audit)
* Mỗi run chứa đúng 182 dòng dự đoán trong `predictions.json`.
* Tập hợp `source_id` duy nhất có đúng **91 sources**, mỗi source xuất hiện chính xác 2 lần.
* Mỗi `source_id` có đúng cặp nhãn $\{0, 1\}$ (1 authentic, 1 ai_edited).
* Đối chiếu với phân vùng chuẩn tắc từ manifest:
  - Khớp 100% với 91 `inner_validation` sources.
  - Rò rỉ sang `development_train`: **0 source** (0 sample).
  - Rò rỉ sang `locked_test`: **0 source** (0 sample).
* 15/15 runs đánh giá trên cùng một tập 91 sources đồng nhất, không phân kỳ.

### 5.5. Kiểm kê Tham số Huấn luyện (Trainable Parameter Inventory)
* `trainable-parameter-inventory.json` chứa đúng **7 tensors** thuộc allowlist đã đăng ký trước:
  1. `features.12.0.weight`: `[576, 96, 1, 1]`, numel: `55,296`
  2. `features.12.1.weight`: `[576]`, numel: `576`
  3. `features.12.1.bias`: `[576]`, numel: `576`
  4. `classifier.0.weight`: `[256, 576]`, numel: `147,456`
  5. `classifier.0.bias`: `[256]`, numel: `256`
  6. `classifier.3.weight`: `[2, 256]`, numel: `512`
  7. `classifier.3.bias`: `[2]`, numel: `2`
* Tổng số tham số huấn luyện (Trainable parameters): **204,674** (19.04%).
* Tổng số tham số đóng băng (Frozen parameters): **870,560** (80.96%).
* Tổng tham số mô hình: **1,075,234**.

---

## 6. Kiểm toán Nguồn gốc Thực thi (Section E: Provenance Audit)

* **`OPERATOR_STATUS.json`**:
  - `status`: `"completed"`
  - `mode`: `"execute"`
  - `stage2_invocations`: `1`
  - `training_runs_completed`: `15`
  - `execution_short_sha`: `"9ee7fdb"`
  - `timestamp_utc`: `"2026-10-01T10:48:24Z"`
  - Ghi chú về nhãn hiển thị cũ: Trường `verdict = "READY_FOR_USER_COLAB_PREFLIGHT"` được xác định là chuỗi hiển thị mẫu chưa chỉnh sửa; bảo toàn nguyên trạng không sửa sau thực thi và không làm mất hiệu lực trạng thái hoàn thành canonical.
* **Environment Lock (`phase4c2_environment_lock.json`) & Sidecar (`.sha256`)**:
  - Sidecar hash khớp bitwise với SHA-256 của file lock.
  - `full_execution_commit_sha`: `9ee7fdbb88fad16167f5790b5105867747801372`
  - `operator.sha256`: `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`
  - `code_archive.normalization_manifest_sha256`: `818566420d925595cf4875c09939985bd73a260c93e9c8a317a6ed37221d59d9`
  - `dataset.archive_sha256`: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`
  - `dataset.content_sha256`: `c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`
  - `dataset.manifest_sha256`: `411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d`
* **`normalized_execution_manifest.json`**: Tồn tại nguyên vẹn, SHA-256 khớp khóa môi trường.

---

## 7. Bảng Tóm tắt 15 Runs Hoàn thành (Completed Runs Summary)

| Run ID | Cohort N | Seed | Best Epoch | Epochs Completed | Val Macro-F1 | Locked Test Access | Stage 1 Writes | Artifacts | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `n50_seed_42` | 50 | 42 | 16 | 20 | 0.5401 | 0 | 0 | 10/10 | COMPLETED |
| `n50_seed_1337` | 50 | 1337 | 8 | 13 | 0.5057 | 0 | 0 | 10/10 | COMPLETED |
| `n50_seed_2025` | 50 | 2025 | 1 | 6 | 0.5445 | 0 | 0 | 10/10 | COMPLETED |
| `n50_seed_3407` | 50 | 3407 | 1 | 6 | 0.5366 | 0 | 0 | 10/10 | COMPLETED |
| `n50_seed_9001` | 50 | 9001 | 2 | 7 | 0.5387 | 0 | 0 | 10/10 | COMPLETED |
| `n100_seed_42` | 100 | 42 | 2 | 7 | 0.5739 | 0 | 0 | 10/10 | COMPLETED |
| `n100_seed_1337` | 100 | 1337 | 15 | 20 | 0.5694 | 0 | 0 | 10/10 | COMPLETED |
| `n100_seed_2025` | 100 | 2025 | 4 | 9 | 0.5518 | 0 | 0 | 10/10 | COMPLETED |
| `n100_seed_3407` | 100 | 3407 | 3 | 8 | 0.5471 | 0 | 0 | 10/10 | COMPLETED |
| `n100_seed_9001` | 100 | 9001 | 4 | 9 | 0.5692 | 0 | 0 | 10/10 | COMPLETED |
| `n250_seed_42` | 250 | 42 | 4 | 9 | 0.5627 | 0 | 0 | 10/10 | COMPLETED |
| `n250_seed_1337` | 250 | 1337 | 3 | 8 | 0.5543 | 0 | 0 | 10/10 | COMPLETED |
| `n250_seed_2025` | 250 | 2025 | 7 | 12 | 0.5619 | 0 | 0 | 10/10 | COMPLETED |
| `n250_seed_3407` | 250 | 3407 | 2 | 7 | 0.5684 | 0 | 0 | 10/10 | COMPLETED |
| `n250_seed_9001` | 250 | 9001 | 3 | 8 | 0.5539 | 0 | 0 | 10/10 | COMPLETED |

*(Ghi chú: Các giá trị metric thô trên chỉ được ghi nhận từ run receipts; phân tích đối chứng chi tiết Stage 1–Stage 2, paired deltas và significance tests sẽ được thực hiện độc lập trong Phase 4C.2C).*

---

## 8. Bằng chứng Thực thi và Biên bản (Artifacts & Receipts)

### 8.1. Local Receipt ngoài Git
* Đường dẫn: `phase_4c2/receipts/stage2_colab_import_receipt.json` (trong kho lưu trữ artifacts cục bộ)
* Trạng thái: Đã tạo và xác thực đầy đủ.

### 8.2. Evidence trong Git
Thư mục: `research/evidence/phase-4c.2c.0/`
1. `PHASE_REPORT.md` (tài liệu này)
2. `import_audit_summary.json` (tóm tắt kết quả kiểm toán máy đọc, không chứa đường dẫn Windows tuyệt đối)
3. `archive_inventory.json` (kiểm kê 173 tệp TAR thành phần)

### 8.3. Bộ công cụ Tái lập và Kiểm thử
* Importer: `scripts/research/ingest_phase_4c2_results.py`
* Tests: `ml/tests/test_phase_4c2_results_ingest.py` (21 unit và fault-injection tests PASS)

---

## 9. Kết luận (Final Verdict)

Toàn bộ 15/15 runs của Stage 2 Partial Fine-Tuning đã được nhập khẩu và thẩm định toàn vẹn 100%, bảo đảm tính bất biến, không rò rỉ và tuân thủ tuyệt đối các cam kết khoa học.

Phán quyết chính thức:
**`READY_FOR_PHASE_4C2C_PAIRED_ANALYSIS`**
