# Kế Hoạch Triển Khai: Đánh Giá Đánh Đổi Phân Loại Đầy Đủ, Kiểm Tra Vùng Chỉnh Sửa Bị Mất Sau Center-Crop & Chuẩn Bị Bước Localization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chốt phần chẩn đoán phân loại bằng đánh giá đánh đổi đầy đủ trên 6 locked models và 3 cohorts; kiểm tra định lượng tỷ lệ vùng chỉnh sửa bị loại bỏ do center-crop trên 91 mẫu holdout edited; xây dựng công cụ đánh giá localization tối thiểu và chuẩn bị protocol localization sẵn sàng triển khai cho lượt sau.

**Architecture:**
1. *Classification Trade-off*: Tổng hợp các dự đoán chuẩn mực (Canonical Preprocessing: Center-crop square -> 512×512 Lanczos -> 224×224 visual / 512×512 DSP) của 6 locked models (`A0-global`, `A0-local`, `A0-visual`, `A1-global`, `A1-local`, `A1-visual`) trên 3 cohorts (Dev Holdout N=273, ImageNet Control N=341, So-Fake Probe N=300). Tính toán Macro-F1, Balanced Accuracy, Recall/Precision/F1 từng lớp, FPR authentic phân rã sang edited/generated, Brier, ECE, CPU latency, paired deltas và 95% bootstrap CIs (10,000 replicates).
2. *Center-Crop Mask Coverage Audit*: Đo giao cắt hình học giữa bounding box / binary mask của 91 ảnh edited trong development holdout với canonical crop box $\min(w,h)$. Phân loại theo 4 nhóm retention (100%, 75-99%, 1-74%, 0%) và đối chiếu với độ chính xác dự đoán của 6 models.
3. *Localization Preparation*: Triển khai `ml/evaluation/localization_evaluator.py` tính pixel-level IoU, Dice, xử lý empty masks và false alarm area; kiểm thử bằng synthetic fixtures; lập protocol localization tối thiểu v1.0.0 (đánh dấu mô hình `NOT_RUN / NOT_MEASURED`).

**Tech Stack:** Python 3.12, PyTorch (CPU), Torchvision, Pillow, NumPy, SciPy, Pytest, Node.js / pnpm.

**Spec:** Yêu cầu người dùng tại phiên 2026-10-11T21:45:47+07:00, tuân thủ `AGENTS.md`, ADRs, và Continuity Protocol.

## Global Constraints

- Chạy trên LOCAL CPU, không dùng Colab/GPU.
- Không tải dataset/checkpoint mới.
- Không retrain, refit, recalibrate, tuning ngưỡng hoặc thêm kiến trúc trong lượt này.
- Không tạo PR, không tạo/xóa branch, không merge main, không sửa demo production.
- Không mở lại sealed Locked-Test / TGIF N=400; không can thiệp ground-truth label của ảnh bị crop mất vùng sửa.
- Không tự động suy diễn nhân quả đơn thuần từ tương quan hình học.
- Đánh dấu trạng thái localization model là `NOT_RUN / NOT_MEASURED`.

---

### Task 1: Đánh Giá Đánh Đổi Phân Loại Đầy Đủ (Section D)

**Files:**
- Create: `scripts/research/analyze_full_classification_tradeoffs.py`
- Create: `research/evidence/three_class_preparation/classification_tradeoff_analysis_receipt.json`
- Create: `ml/tests/test_classification_tradeoff_analysis.py`

**Interfaces:**
- Consumes: `locked_models_registry.json`, `three_class_pilot_predictions_reconciled.json`, `three_class_ablation_a1_predictions.json`, `three_class_local_dsp_predictions.json`, `so_fake_probe_predictions.json`, `feature_extraction_receipt.json`, `local_dsp_feature_extraction_receipt.json`.
- Produces: `classification_tradeoff_analysis_receipt.json` chứa đầy đủ bảng số liệu 6 models × 3 cohorts, paired deltas và 95% bootstrap CIs.

- [ ] **Step 1: Viết script phân tích đánh đổi phân loại đầy đủ**
  - Tập hợp dự đoán của 6 locked models (`A0-global`, `A0-local`, `A0-visual`, `A1-global`, `A1-local`, `A1-visual`) trên 3 cohorts:
    * Development Holdout $N=273$ (3 lớp cân bằng 91 authentic, 91 edited, 91 generated).
    * ImageNet Authentic Control $N=341$ (100% authentic nature).
    * So-Fake Probe $N=300$ (3 lớp cân bằng 100 authentic, 100 edited, 100 generated).
  - Tính toán cho từng model trên từng cohort:
    * Macro-F1 và Balanced Accuracy (cho cohorts 3 lớp: Holdout và So-Fake).
    * Per-class Precision, Recall, F1 và Confusion Matrix $3 \times 3$.
    * Authentic False-Positive AI rate ($FP / N$), tách rõ nhầm sang `ai_edited` vs `fully_generated`.
    * Multiclass Brier Score và Top-class ECE (10 bins đều).
    * CPU inference latency per image (trích xuất từ receipts gốc đã đo).
  - Tính paired deltas và 95% bootstrap CIs (10,000 reps, seed 20261011):
    * Local − Global trong A0 và A1.
    * Local − Visual trong A0 và A1.
    * A1 − A0 cho cùng loại đặc trưng (global, local, visual).
    * Holdout: Cluster bootstrap theo 91 COCO source groups.
    * Control & So-Fake: Image-level paired bootstrap. Báo cáo Bootstrap Positive Fraction $P(\Delta^* > 0)$ rõ ràng, không gọi là p-value.
  - Trả lời câu hỏi trọng tâm: "Local DSP giảm báo động giả bằng cách cải thiện phân biệt, hay đổi lại bằng việc bỏ sót thêm ảnh edited/generated?"
  - Xuất `classification_tradeoff_analysis_receipt.json`.

- [ ] **Step 2: Viết unit test kiểm chứng trade-off analysis**
  - Tạo `ml/tests/test_classification_tradeoff_analysis.py` xác thực tính toàn vẹn của receipt, cấu trúc bảng 6 models × 3 cohorts, và tính đúng đắn của phép tính bootstrap delta.

- [ ] **Step 3: Chạy test và xác nhận PASS**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/test_classification_tradeoff_analysis.py -v`.

---

### Task 2: Kiểm Tra Vùng Chỉnh Sửa Bị Mất Sau Center-Crop (Section E)

**Files:**
- Create: `scripts/research/audit_center_crop_mask_coverage.py`
- Create: `research/evidence/three_class_preparation/center_crop_mask_coverage_manifest.json`
- Create: `research/evidence/three_class_preparation/center_crop_mask_coverage_receipt.json`
- Create: `ml/tests/test_center_crop_mask_coverage.py`

**Interfaces:**
- Consumes: `three_class_pilot_splits.json` (91 edited samples), thư mục masks `data/research/tgif/masks/validation/`, và dự đoán của 6 models từ Task 1.
- Produces: `center_crop_mask_coverage_manifest.json` và `center_crop_mask_coverage_receipt.json`.

- [ ] **Step 1: Viết script audit hình học center-crop và mask coverage**
  - Nạp 91 mẫu `ai_edited` từ `development_holdout` và tìm file ground-truth mask tương ứng trong `data/research/tgif/masks/validation/`.
  - Xác minh pairing: metadata ID, kích thước ảnh $w \times h$ và kích thước mask.
  - Tái lập tọa độ canonical crop box chính xác:
    `min_dim = min(w, h)`, `left = (w - min_dim) // 2`, `top = (h - min_dim) // 2`, `right = left + min_dim`, `bottom = top + min_dim`.
  - Đo lường pixel-level:
    * Diện tích vùng sửa gốc $A_{\text{orig}}$ (tổng pixel mask $> 0$).
    * Bounding box vùng sửa gốc `(min_x, min_y, max_x, max_y)`.
    * Diện tích vùng sửa nằm trong crop box $A_{\text{retained}}$.
    * Tỷ lệ vùng sửa giữ lại $\text{retention\_ratio} = A_{\text{retained}} / A_{\text{orig}}$.
    * Tỷ lệ diện tích sửa trên canvas vuông $\text{canvas\_ratio} = A_{\text{retained}} / (min\_dim \times min\_dim)$.
  - Phân nhóm coverage theo các ngưỡng định trước (predefined bins):
    * `full_retention`: $\text{retention\_ratio} = 1.0$ (100% giữ nguyên).
    * `minor_loss`: $0.75 \le \text{retention\_ratio} < 1.0$ (mất nhẹ 1–25%).
    * `major_loss`: $0 < \text{retention\_ratio} < 0.75$ (mất nhiều > 25%).
    * `complete_loss`: $\text{retention\_ratio} == 0.0$ (vùng sửa bị cắt bỏ 100%, không còn pixel nào trong canvas).
  - Đối chiếu với kết quả dự đoán của 6 locked models:
    * Tỷ lệ Recall `ai_edited` theo từng nhóm coverage.
    * Tỷ lệ lỗi `ai_edited -> authentic` theo từng nhóm coverage.
    * Tỷ lệ lỗi `ai_edited -> fully_generated` theo từng nhóm coverage.
  - Xuất `center_crop_mask_coverage_manifest.json` (per-image details) và `center_crop_mask_coverage_receipt.json` (summary statistics).

- [ ] **Step 2: Viết unit test kiểm chứng mask coverage audit**
  - Tạo `ml/tests/test_center_crop_mask_coverage.py` kiểm tra tính toán hình học crop, bảo toàn số lượng 91 mẫu, và tính hợp lệ của manifest.

- [ ] **Step 3: Chạy test và xác nhận PASS**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/test_center_crop_mask_coverage.py -v`.

---

### Task 3: Chuẩn Bị Bước Localization & Triển Khai Evaluator Tối Thiểu (Section F)

**Files:**
- Create: `ml/evaluation/localization_evaluator.py`
- Create: `ml/tests/test_localization_evaluator.py`
- Create: `research/evidence/three_class_preparation/PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md`

**Interfaces:**
- Consumes: Ground-truth masks contract và heatmap predictions contract.
- Produces: Thư viện `localization_evaluator.py` có hàm kiểm tra alignment, pixel IoU, Dice, xử lý empty masks và false alarm area; protocol chuẩn bị cho phase sau.

- [ ] **Step 1: Triển khai module `localization_evaluator.py`**
  - Định nghĩa hàm `check_mask_alignment(image_shape, mask_shape)`.
  - Định nghĩa hàm `compute_pixel_iou(pred_mask, gt_mask, threshold=0.5)`.
  - Định nghĩa hàm `compute_pixel_dice(pred_mask, gt_mask, threshold=0.5)`.
  - Quy ước xử lý empty mask (ảnh authentic hoặc fully_generated không có vùng inpainting):
    * Nếu ground-truth rỗng và prediction rỗng: IoU = 1.0, Dice = 1.0, False Positive Area = 0.
    * Nếu ground-truth rỗng và prediction có pixel dương: IoU = 0.0, Dice = 0.0, False Positive Area = $\sum \text{pred} / \text{total\_pixels}$.
  - Hỗ trợ đánh giá batch và xuất cấu trúc kết quả chuẩn machine-readable.

- [ ] **Step 2: Viết unit test kiểm chứng evaluator với synthetic fixtures**
  - Tạo `ml/tests/test_localization_evaluator.py` kiểm tra toàn bộ các ca: perfect match, partial overlap, zero overlap, empty GT + empty pred, empty GT + false alarm, shape mismatch exception. Sử dụng mảng NumPy tổng hợp độc lập với dữ liệu nghiên cứu.

- [ ] **Step 3: Chạy test và xác nhận PASS**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/test_localization_evaluator.py -v`.

- [ ] **Step 4: Lập Protocol Chuẩn Bị Localization v1.0.0**
  - Tạo `research/evidence/three_class_preparation/PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md` quy định:
    * Kiến trúc đề xuất: Lightweight inpainting segmentor (<10MB on disk, <50ms CPU inference).
    * Phân chia tập dữ liệu: Giữ nguyên split TGIF masks (train/val/test), cấm rò rỉ vào sealed cohorts.
    * Geometry contract: Quy định chuyển đổi tọa độ giữa input crop canvas và ảnh gốc.
    * Model evaluation status: Ghi rõ `MODEL_STATUS: NOT_RUN / NOT_MEASURED`. Không tự huấn luyện hoặc tải mô hình ở lượt này.

---

### Task 4: Khảo Sát Y Văn Có Mục Tiêu, Cập Nhật Tài Liệu & Verification Toàn Diện (Section G & H)

**Files:**
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Khảo sát y văn có mục tiêu về lightweight inpainting localization và source/compression bias**
  - Tra cứu các tài liệu sơ cấp (TruFor, Mantranet, PSCC-Net, Noiseprint, FOCAL, v.v.).
  - Bổ sung phân tích đối sánh y văn vào Section 17 của `PHASE_REPORT.md`.

- [ ] **Step 2: Cập nhật tài liệu Continuity Protocol**
  - Cập nhật `CURRENT_STATE.md` phản ánh kết quả trade-off, crop coverage audit, và trạng thái localization preparation.
  - Cập nhật `CODE_INDEX.md` với các script và module mới.
  - Cập nhật `STATUS_LEDGER.md` thêm tóm tắt ngắn (< 20 dòng).

- [ ] **Step 3: Chạy full verification suite**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/ -v`.
  - Run: `pnpm test`.
  - Run: `pnpm continuity:check -- --staged`.
  - Run: `git diff --check`.

- [ ] **Step 4: Commit và push lên nhánh nghiên cứu**
  - Commit với thông điệp chuẩn mực.
  - Push lên `origin/research/independent-cohort-acquisition`.
