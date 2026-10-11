# Kế Hoạch Triển Khai: Đối Soát Lineage Baseline & Chẩn Đoán Đối Chứng Nguồn Ảnh × Nén JPEG (2×2 Factorial Design)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đối soát toàn diện nguồn gốc các chênh lệch baseline lịch sử so với C0/C1, đính chính các kết luận vượt quá bằng chứng (epistemic overclaiming), và thực thi chẩn đoán đối chứng nhân tố 2×2 (Nguồn: COCO vs ImageNet × Nén: Original vs JPEG Q75) trên toàn bộ 6 locked models bằng dữ liệu có sẵn trên LOCAL CPU.

**Architecture:** Sử dụng pipeline trích xuất chuẩn mực (Canonical Preprocessing: Center-crop square -> 512×512 Lanczos canvas -> 16-D Global DSP & 48-D Local DSP -> 224×224 Bicubic Visual Tensor) để phân lập ảnh hưởng của hình học tiền xử lý (anisotropic squash vs aspect-ratio preserving crop), đối soát chênh lệch baseline với các biên nhận máy đọc; sau đó thực hiện thiết kế nhân tố 2×2 trên các ảnh authentic chưa từng qua huấn luyện của mô hình và không thuộc sealed cohorts.

**Tech Stack:** Python 3.12, PyTorch (CPU), Torchvision (MobileNetV3-small frozen backbone `IMAGENET1K_V1`), Pillow (JPEG encoder quality=75, subsampling=2), Scipy / NumPy (2D FFT / DCT), Pytest, Node.js / pnpm.

**Spec:** Yêu cầu người dùng tại phiên 2026-10-11T20:47:28+07:00, tuân thủ `AGENTS.md`, ADRs, và Continuity Protocol.

## Global Constraints

- Chạy trên LOCAL CPU, không dùng Colab/GPU.
- Không tải dataset/checkpoint mới.
- Không retrain, refit, recalibrate, tuning ngưỡng hoặc thêm kiến trúc.
- Không tạo PR, không tạo/xóa branch, không merge main, không sửa demo production.
- Không mở lại sealed Locked-Test / TGIF N=400; không dùng training samples để lấp thiếu dữ liệu đánh giá.
- Không ghi đè artifacts lịch sử; mỗi đợt chẩn đoán có receipt và predictions riêng.
- Tỷ lệ bootstrap positive fraction ($P(\Delta^* > 0)$) không được trình bày như p-value.
- Mọi chênh lệch baseline phải được chứng minh bằng lineage máy đọc (code, parameters, pre-processing, cache).

---

### Task 1: Đối Soát Toàn Diện Lineage Baseline & Truy Nguyên 4 Mâu Thuẫn

**Files:**
- Create: `scripts/research/audit_baseline_lineage.py`
- Create: `research/evidence/three_class_preparation/baseline_lineage_audit_receipt.json`
- Test: `ml/tests/test_baseline_lineage_audit.py`

**Interfaces:**
- Consumes: `locked_models_registry.json`, `three_class_pilot_splits.json`, `so_fake_probe_manifest_v2.json`, `three_class_pilot_evaluation_receipt_reconciled.json`, `compression_sensitivity_receipt.json`
- Produces: `baseline_lineage_audit_receipt.json` chứa bảng đối chiếu bit-exact giữa Canonical Pipeline (Pipe 1) và Anisotropic Sensitivity Pipeline (Pipe 2).

- [ ] **Step 1: Viết script truy nguyên và lập bảng lineage cho 4 chênh lệch baseline**
  - Chênh lệch 1: A0-global So-Fake probe 0.3551 (Canonical Pipe 1: Center crop square -> 512 Lanczos -> 224 Bicubic) vs 0.3427 (Anisotropic Pipe 2: Direct raw resize to 512 Lanczos & direct raw resize to 224 Bicubic).
  - Chênh lệch 2: A0-global Dev Holdout 0.5385 (Canonical Pipe 1) vs 0.5592 (Anisotropic Pipe 2).
  - Chênh lệch 3: A0-global JPEG Q75 0.6015 (Canonical Pipe 1 sau JPEG Q75) vs 0.6187 (Anisotropic Pipe 2 sau JPEG Q75).
  - Chênh lệch 4: Brier/ECE thay đổi giữa historical temperature ($T_{\text{fus}}=8.1321$) và reconciled temperature ($T_{\text{fus}}=5.9577$) do math của softmax calibration, trong khi argmax decisions và Macro-F1 giữ nguyên 100% bit-identical.
  - Xuất `baseline_lineage_audit_receipt.json`.

- [ ] **Step 2: Viết unit test kiểm chứng lineage audit**
  - Tạo `ml/tests/test_baseline_lineage_audit.py` xác thực tính chính xác của bảng lineage và các khẳng định tái lập.

- [ ] **Step 3: Chạy test và xác nhận PASS**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/test_baseline_lineage_audit.py -v`.

---

### Task 2: Khóa Protocol Thực Nghiệm & Xây Dựng Manifest 2×2 Factorial Design

**Files:**
- Create: `scripts/research/build_factorial_2x2_manifest.py`
- Create: `research/evidence/three_class_preparation/factorial_2x2_cohort_manifest.json`
- Test: `ml/tests/test_factorial_source_compression.py`

**Interfaces:**
- Consumes: `three_class_pilot_splits.json` (91 COCO authentic từ holdout), `source_control_cohort_manifest.json` (341 ImageNet nature authentic từ control pilot), `locked_models_registry.json`.
- Produces: `factorial_2x2_cohort_manifest.json` chứa danh sách 182 mẫu cân bằng (91 COCO + 91 ImageNet lấy tất định theo seed 20261011) và cờ `full_sensitivity_pool` cho toàn bộ 341 ImageNet.

- [ ] **Step 1: Viết script xây dựng manifest 2×2**
  - Rút 91 mẫu COCO authentic từ `development_holdout` (đảm bảo disjoint khỏi training của A0/A1).
  - Rút 91 mẫu ImageNet authentic tất định từ `source_control_cohort_manifest.json` theo seed 20261011 đã khóa.
  - Lưu giữ metadata nguồn: source ID, SHA-256 hash, kích thước gốc, nhãn authentic.
  - Kiểm tra 0 collision với tập huấn luyện và sealed sets.
  - Xuất `factorial_2x2_cohort_manifest.json`.

---

### Task 3: Thực Thi Thí Nghiệm Đối Chứng Nguồn Ảnh × Nén JPEG (2×2 Factorial Design)

**Files:**
- Create: `scripts/research/run_factorial_source_compression_diagnosis.py`
- Create: `research/evidence/three_class_preparation/factorial_source_compression_predictions.json`
- Create: `research/evidence/three_class_preparation/factorial_source_compression_receipt.json`

**Interfaces:**
- Consumes: `factorial_2x2_cohort_manifest.json`, `locked_models_registry.json`.
- Produces: `factorial_source_compression_predictions.json` (raw outputs) và `factorial_source_compression_receipt.json`.

- [ ] **Step 1: Viết harness thực thi 2×2 factorial evaluation**
  - Khóa 2 điều kiện nén trên cùng ảnh:
    * `original`: Không nén thêm, đưa vào Canonical Preprocessing chuẩn mực.
    * `jpeg_q75`: Nén JPEG độ phân giải gốc một lần (`quality=75, subsampling=2 (4:2:0), optimize=False`), sau đó đưa vào Canonical Preprocessing chuẩn mực.
  - Đánh giá trên toàn bộ 6 locked models: `A0-global`, `A0-local`, `A1-global`, `A1-local`, `A0-visual`, `A1-visual`.
  - Tính các chỉ số chính cho cohort authentic:
    * Primary: False-Positive AI rate ($FP / N$).
    * Secondary: Phân rã nhầm sang edited vs generated; $P(\text{authentic})$; decision flip rate (%) vs original.
    * Tác động JPEG trong từng nguồn: $\Delta_{\text{JPEG}}(\text{COCO}) = FP(\text{COCO, Q75}) - FP(\text{COCO, orig})$, $\Delta_{\text{JPEG}}(\text{ImageNet}) = FP(\text{ImageNet, Q75}) - FP(\text{ImageNet, orig})$.
    * Hiệu ứng tương tác nhân tố 2×2: $\Delta_{\text{interaction}} = \Delta_{\text{JPEG}}(\text{ImageNet}) - \Delta_{\text{JPEG}}(\text{COCO})$.
  - Paired sample bootstrap (10,000 replicates, seed 20261011) cho từng nguồn và cho hiệu ứng tương tác.
  - Đánh giá bổ trợ trên toàn bộ $N=341$ ImageNet để kiểm tra tính ổn định.

- [ ] **Step 2: Thực thi thí nghiệm và lưu raw predictions / receipt**
  - Run: `ml/.venv/Scripts/python.exe scripts/research/run_factorial_source_compression_diagnosis.py`.

---

### Task 4: Rà Soát Y Văn Có Mục Tiêu (Targeted Literature Gap Review)

**Files:**
- Create/Update: `research/evidence/three_class_preparation/TARGETED_LITERATURE_GAP_REVIEW.md`

**Interfaces:**
- Consumes: Web search qua `search_web` hoặc tài liệu sơ cấp có sẵn.
- Produces: Bản phân tích y văn đối chiếu về source/compression confounding, preprocessing order, và frequency fusion.

- [ ] **Step 1: Tìm kiếm tài liệu sơ cấp về các chủ đề trọng tâm**
  - Query: "source dataset confounding deepfake detection frequency artifacts"
  - Query: "JPEG compression resize order artifacts CNN frequency domain"
  - Trích xuất tác giả, năm, hội nghị, kết luận đối chiếu với bài toán lightweight in-browser forensics của đề tài.
  - Trả lời câu hỏi trọng tâm: “Lợi ích của DSP còn tồn tại khi kiểm soát nguồn ảnh và quy trình nén/resize hay không?”

---

### Task 5: Đính Chính Các Kết Luận Vượt Quá Bằng Chứng & Cập Nhật Tài Liệu

**Files:**
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md` (Cập nhật Section 15 và bổ sung Section 16).
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Đính chính các phát biểu vượt quá bằng chứng trong PHASE_REPORT.md**
  - Panel 16 ảnh bit-exact: Ghi rõ chỉ xác nhận những bước và panel 16 ảnh được kiểm tra; không khái quát hóa thành model-quality parity.
  - Visual robustness: Đổi "visual miễn nhiễm" thành "ít nhạy hơn trong các điều kiện đã thử".
  - C2 false alarm: Ghi rõ đây là quan sát thực nghiệm; cơ chế tương tác khối DCT 8×8 với lưới FFT là giả thuyết giải thích, chưa phải quan hệ nhân quả cô lập 100%.
  - Multiclass Brier: Ghi rõ Brier tệ hơn xác suất đều ngẫu nhiên phản ánh chất lượng dự đoán xác suất tổng thể kém ngoài miền; riêng Brier chưa đủ để quy toàn bộ sai số cho miscalibration.
  - Random baseline: Phân biệt rõ Macro-F1 ngẫu nhiên của bộ dự đoán đều ($1/3$) với balanced accuracy.
  - Trùng byte SHA-256: Ghi rõ chỉ chứng minh không phát hiện trùng byte trong phạm vi audit.
  - Holdout: Khẳng định đây là development/exploratory holdout, không phải kiểm định độc lập mới.
  - Local quadrant DSP: Xác nhận đây là đặc trưng phân loại bổ trợ, chưa phải bộ định vị (localization).

- [ ] **Step 2: Cập nhật kết quả thí nghiệm 2×2 Factorial vào PHASE_REPORT.md (Section 16)**
  - Trình bày bảng kết quả 2×2 đầy đủ.
  - Phân tích tương tác nguồn × nén và false positive rates.

- [ ] **Step 3: Cập nhật Continuity Files**
  - Cập nhật `CURRENT_STATE.md`, `CODE_INDEX.md`, và `STATUS_LEDGER.md` (prepended entry < 20 lines).

---

### Task 6: Kiểm Thử Tự Động, Verification & Bàn Giao

**Files:**
- Create: `ml/tests/test_factorial_source_compression.py`

- [ ] **Step 1: Viết và chạy bộ test tự động đầy đủ**
  - Run: `ml/.venv/Scripts/python.exe -m pytest ml/tests/test_baseline_lineage_audit.py ml/tests/test_factorial_source_compression.py ml/tests/test_compression_sensitivity.py ml/tests/test_so_fake_probe.py -v`.
- [ ] **Step 2: Chạy workspace tests & continuity check**
  - Run: `pnpm test`.
  - Run: `pnpm continuity:check`.
  - Run: `git diff --check`.
- [ ] **Step 3: Commit và push lên branch nghiên cứu**
  - Commit message tuân thủ semantic conventions.
  - Push lên `origin/research/independent-cohort-acquisition`.
