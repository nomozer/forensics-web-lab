# Document Generation Feasibility Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai thử nghiệm khả thi tạo dữ liệu tài liệu AI trên đúng bốn tài liệu CORD v2 (2 replacement, 2 removal) bằng `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` trên Google Colab GPU và kiểm chứng kỹ thuật trên LOCAL CPU, không dùng token/tài khoản Hugging Face.

**Architecture:** Thiết lập quy trình 2 tầng tách biệt: (1) LOCAL chuẩn bị plan máy tất định, module hình học đảo ngược (inverse geometry composite), harness kiểm chứng artifacts và notebook Colab; (2) COLAB GPU thực thi tải weights FP16 (6.46 GiB), nạp 4 mẫu CORD v2 và chạy đúng 4 attempts khóa trước, đóng gói archive kết quả về LOCAL để kiểm toán QC độc lập.

**Tech Stack:** Python 3.12/3.10+, PyTorch, Diffusers 0.21+, Hugging Face Safetensors (FP16), PyArrow Parquet, Pillow, NumPy, Jupyter Notebook.

**Spec:** Yêu cầu người dùng (Mục tiêu được phê duyệt A-K, ngày 2026-10-12).

## Global Constraints
- **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”
- **Giới hạn phạm vi**: Thử nghiệm khả thi tạo dữ liệu (generation feasibility study); KHÔNG huấn luyện detector; KHÔNG đánh giá 6 locked models; KHÔNG huấn luyện localization; KHÔNG mở rộng sang 40 nguồn trong lượt này.
- **Ràng buộc xác thực**: KHÔNG yêu cầu đăng nhập Hugging Face, KHÔNG cung cấp `HF_TOKEN`, KHÔNG dùng authenticated fallback.
- **Bảo toàn**: Giữ nguyên 6 locked models trong `locked_models_registry.json`, `main`, web demo, và sealed cohorts (TGIF N=400, Locked-Test 343 sources).
- **Ngân sách lượt này**: Checkpoint + dependencies trên Colab $\le 12\text{ GB}$; dữ liệu gốc $\le 200\text{ MB}$; artifacts về Local $\le 100\text{ MB}$; GPU execution $\le 60\text{ phút}$.
- **Geometry Invariant**: Bảo toàn 100% pixel ngoài mask bằng so sánh mảng (`np.array_equal`), không chỉ mean L1 làm tròn.

---

### Task 1: Protocol Amendment Specification v1.1.0 (Design Rectification)

**Files:**
- Create: `research/evidence/document_pilot/INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.1.0.md`
- Test: `ml/tests/test_document_feasibility_geometry.py`

**Interfaces:**
- Consumes: `INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md` (v1.0.0, sealed for lineage)
- Produces: Bản tu chính v1.1.0 đính chính toàn diện 7 phát biểu vượt bằng chứng, khóa ngân sách 6 thành phần, và định danh feasibility study 4 attempts.

- [ ] **Step 1: Soạn thảo văn bản tu chính thiết kế v1.1.0**
  * Đính chính “Không tồn tại dataset mở” thành “Chưa tìm được nguồn đáp ứng điều kiện trong các ứng viên đã kiểm tra.”
  * Không gọi tự tạo dữ liệu là phương án duy nhất.
  * Không cam kết 5 phút hay GPU T4/A100 chắc chắn được cấp.
  * Không gọi PNG parity là loại bỏ toàn bộ confounding.
  * Không gọi zero hash collisions là độc lập tuyệt đối.
  * Không gọi biến đổi pixel là thành công thay nội dung.
  * Ghi rõ điều kiện giấy phép: CORD v2 (CC-BY 4.0), SDXL Inpainting (OpenRAIL++).
  * Tách ngân sách 6 thành phần: Checkpoint Colab (6.46 GiB), Dữ liệu gốc ($\le 200$ MB), Dependencies ($\le 2$ GB), Artifacts về Local ($\le 100$ MB), Đĩa/VRAM (15 GB / 13 GB), Thời gian ($\le 60$ phút).

- [ ] **Step 2: Viết test kiểm chứng tính toàn vẹn của Amendment v1.1.0**
  * Kiểm tra sự tồn tại của file, các mệnh đề tu chính, tên đề tài, và bảng ngân sách.

- [ ] **Step 3: Chạy test và xác nhận PASS**
  * `& .\ml\.venv\Scripts\python.exe -m pytest ml/tests/test_document_feasibility_geometry.py::test_pilot_design_amendment_v110 -v`

---

### Task 2: Geometry & Composite Module (`document_feasibility_geometry.py`)

**Files:**
- Create: `ml/evaluation/document_feasibility_geometry.py`
- Test: `ml/tests/test_document_feasibility_geometry.py`

**Interfaces:**
- Consumes: NumPy arrays, Pillow Image, Bounding box `[ymin, xmin, ymax, xmax]`
- Produces: `extract_roi_and_build_canvas()`, `composite_inpainted_patch()`, `verify_outside_equality()`

- [ ] **Step 1: Viết test failing cho module hình học**
  * Test bảo toàn pixel ngoài mask (`np.array_equal == True`).
  * Test tính nhị phân của mask $\{0, 255\}$ (nearest neighbor).
  * Test ánh xạ ngược tọa độ từ canvas $1024 \times 1024$ về kích thước ảnh gốc.

- [ ] **Step 2: Chạy test để xác nhận FAIL**
  * `& .\ml\.venv\Scripts\python.exe -m pytest ml/tests/test_document_feasibility_geometry.py::test_geometry_roundtrip -v`

- [ ] **Step 3: Cài đặt module `document_feasibility_geometry.py`**
  * Lưu trữ ảnh decoded RGB gốc.
  * Crop vùng can thiệp có đệm ngữ cảnh (context crop).
  * Resize giữ nguyên tỷ lệ lên canvas $1024 \times 1024$ (bicubic cho ảnh, nearest cho mask).
  * Ánh xạ patch ngược về tọa độ gốc và dán qua binary composite.
  * Hàm `verify_outside_equality()` kiểm tra `np.array_equal(edited[final_mask == 0], authentic[final_mask == 0])`.

- [ ] **Step 4: Chạy test để xác nhận PASS**
  * `& .\ml\.venv\Scripts\python.exe -m pytest ml/tests/test_document_feasibility_geometry.py -v`

---

### Task 3: Deterministic Plan Builder for 4 CORD-v2 Documents

**Files:**
- Create: `scripts/research/build_document_feasibility_plan.py`
- Output: `research/evidence/document_pilot/document_feasibility_plan.json`
- Test: `ml/tests/test_document_feasibility_geometry.py`

**Interfaces:**
- Consumes: `naver-clova-ix/cord-v2` validation split metadata via HTTP Range reading
- Produces: `document_feasibility_plan.json` chứa 4 attempts đã khóa cứng: source ID, bounding box, original text, target text, prompt, negative prompt, seed, model revision.

- [ ] **Step 1: Viết script `build_document_feasibility_plan.py`**
  * Đọc metadata của CORD-v2 validation split qua HTTP Range.
  * Dùng PRNG seed `20261012` chọn đúng 4 receipt khác nhau có đủ annotation.
  * Phân công: Attempt 1 (replacement số tiền), Attempt 2 (replacement tên món), Attempt 3 (removal dòng chiết khấu/món), Attempt 4 (removal dòng thuế/phí).
  * Khóa cứng: prompt chi tiết cho SDXL inpainting, negative prompt, seed, steps=30, guidance_scale=7.5, revision `115134f363124c53c7d878647567d04daf26e41e`.
  * Đánh dấu cách ly: `quarantine_tag: "DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION"`.

- [ ] **Step 2: Thực thi script và sinh `document_feasibility_plan.json`**
  * `& .\ml\.venv\Scripts\python.exe scripts/research/build_document_feasibility_plan.py`

- [ ] **Step 3: Viết test xác thực plan JSON**
  * Kiểm tra đúng 4 attempts (2 replacement, 2 removal), không trùng source ID, bbox hợp lệ, schema đầy đủ.

---

### Task 4: Colab Feasibility Runner & Notebook Launcher

**Files:**
- Create: `scripts/research/run_document_feasibility_colab.py`
- Create: `notebooks/independent_document_feasibility_colab.ipynb`
- Test: `ml/tests/test_document_feasibility_geometry.py`

**Interfaces:**
- Consumes: `document_feasibility_plan.json`
- Produces: Notebook Colab tự động hóa chạy 4 attempts, xuất `feasibility_results.zip` chứa ảnh authentic, edited, mask, context, receipts và `human_review.csv` (PENDING).

- [ ] **Step 1: Cài đặt `run_document_feasibility_colab.py`**
  * Preflight kiểm tra GPU và đĩa.
  * Download checkpoint FP16 variant từ Hugging Face (không dùng token).
  * Download 4 ảnh CORD-v2 gốc theo URL/parquet.
  * Chạy inference qua `StableDiffusionXLInpaintPipeline` cho từng attempt.
  * Thực hiện composite và kiểm tra `outside_equality`.
  * Ghi receipt chi tiết cho từng attempt (thời gian, VRAM, hashes).
  * Đóng gói toàn bộ kết quả thành zip archive.

- [ ] **Step 2: Xây dựng canonical notebook 5 cells `independent_document_feasibility_colab.ipynb`**
  * Cell 1: Tiêu đề, hướng dẫn và kiểm tra GPU.
  * Cell 2: Cài đặt dependencies (diffusers, transformers, accelerate, pyarrow).
  * Cell 3: Staging plan và kịch bản thực thi.
  * Cell 4: Chạy runner `--execute`.
  * Cell 5: Tải file archive `document_feasibility_results.zip` về máy local.

- [ ] **Step 3: Viết test kiểm tra cấu trúc notebook**
  * Kiểm tra JSON hợp lệ, không chứa hardcoded token, đúng commit/revision và budget guards.

---

### Task 5: Local Artifact Verification Harness & Contact Sheet Generator

**Files:**
- Create: `scripts/research/verify_document_feasibility_artifacts.py`
- Test: `ml/tests/test_document_feasibility_geometry.py`

**Interfaces:**
- Consumes: Thư mục kết quả hoặc archive `document_feasibility_results.zip`
- Produces: Báo cáo kiểm chứng kỹ thuật, contact sheet so sánh trực quan, và receipt `document_feasibility_verification_receipt.json`.

- [ ] **Step 1: Cài đặt harness kiểm chứng local**
  * Kiểm tra tính toàn vẹn 4 attempts (authentic, edited, mask, raw output).
  * Kiểm tra mảng điểm ảnh ngoài mask bằng `np.array_equal`.
  * Đo đạc pixel change statistics trong mask.
  * Tạo contact sheet hình ảnh trực quan (Original vs ROI zoom vs Mask vs Edited vs Diff).
  * Khởi tạo `human_review.csv` với trạng thái `PENDING_HUMAN_REVIEW`.

- [ ] **Step 2: Viết test tự động cho harness với synthetic mock**
  * Test xác nhận harness phát hiện lỗi nếu pixel ngoài mask bị thay đổi hoặc thiếu file.

---

### Task 6: Execution, Verification, Continuity & Handoff

**Files:**
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`
- Create: `research/evidence/document_pilot/FEASIBILITY_STUDY_PREPARATION_REPORT.md`

- [ ] **Step 1: Chạy toàn bộ test suites**
  * `pytest ml/tests`
  * `pnpm test`
  * `node scripts/continuity-check.mjs --staged`

- [ ] **Step 2: Cập nhật các tài liệu continuity**
  * `CURRENT_STATE.md`: Cập nhật Phase 4C.8 feasibility study preparation.
  * `CODE_INDEX.md`: Thêm các module mới vào Mục 21.
  * `STATUS_LEDGER.md`: Thêm tóm tắt mới ở đầu file.

- [ ] **Step 3: Git diff check, commit và push**
  * `git diff --check`
  * Commit lên `research/independent-cohort-acquisition`.
  * Push lên remote origin.

- [ ] **Step 4: Handoff trực tiếp trong chat**
  * Trả lời đầy đủ 8 câu hỏi cốt lõi, đưa ra đúng một đề xuất tiếp theo và hướng dẫn chạy Colab cụ thể.
