# Phase Report: Three-Class RQ1 & RQ2 Experiment Preparation

> **Phase**: Three-Class RQ1 & RQ2 Experiment Preparation (Phase 4C.8 trace)<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Execution Date**: 2026-10-11<br>
> **Status**: `PREPARATION_COMPLETED_DATA_BLOCKED`

---

## 1. Mục tiêu & Bối cảnh

Chuẩn bị toàn diện cơ sở thực nghiệm cho hai câu hỏi nghiên cứu cốt lõi:
- **RQ1 (Three-Class Generalization)**: Khả năng phân biệt đồng thời 3 lớp `authentic`, `ai_edited`, và `fully_generated` trên kiến trúc thị giác tích chập siêu nhẹ (MobileNetV3-small).
- **RQ2 (Multimodal Evidence Fusion)**: Đóng góp của đặc trưng DSP canonical 16-D khi kết hợp với nhánh thị giác (Late Fusion 592-D) so với visual-only baseline (576-D) trong không gian 3 lớp dưới các phép biến đổi nén và suy thoái ảnh.

Tuân thủ nghiêm ngặt **Quy tắc trung thực kỹ thuật và nghiên cứu** (Rule 3 AGENTS.md):
- Không tải archive vượt ngân sách (ngân sách: tải $\le 2\text{ GiB}$, đĩa + tạm $\le 5\text{ GiB}$).
- Không tự ý huấn luyện khi chưa có dữ liệu/GPU; không giả lập huấn luyện.
- Không tạo số liệu hoặc splits giả.
- Trạng thái khi thiếu dữ liệu phải ghi nhận trung thực là `DATA_BLOCKED`.

---

## 2. Kiểm kê Dữ liệu Thực tế trên Local (Task 1)

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

## 3. Khắc phục Protocol & Backbone Mismatch (Task 2 - Section C)

- **Truy nguyên nguyên nhân gốc rễ ($systematic-debugging)**:
  - Bản thảo cấu hình cũ `ml/configs/three_class_evaluation_protocol.yaml` do lấy template nháp MobileNetV4 ghi nhầm backbone `mobilenet_v4_small` với số chiều visual 1280-d (fusion 1296-d).
  - Toàn bộ codebase thực tế đã được kiểm chứng (Phase 4C.2, 4C.7, 4C.8) sử dụng kiến trúc **MobileNetV3-small** (`in_features = 576`), kết hợp 16 đặc trưng DSP chuẩn tắc cho ra vector đầu vào phân loại **592 chiều** ($576 + 16$).
- **Điều chỉnh & Đồng bộ hóa Contract**:
  - Cập nhật `ml/configs/three_class_evaluation_protocol.yaml`:
    - `backbone: mobilenet_v3_small`
    - `visual_feature_dim: 576`
    - `dsp_feature_dim: 16`
    - `fusion_feature_dim: 592`
    - `provenance: MS-COCO val2017` (sửa triệt để bất nhất COCO 2014)
  - Cập nhật `scripts/research/preflight_three_class_protocol.py` (v2.0):
    - Kiểm chứng forward pass thực tế bằng PyTorch `MobileNetV3Forensics` và hàm `extract_dsp_features` trên panel tensor $[4, 3, 224, 224]$:
      * Tensor visual output: $[4, 576]$
      * Tensor DSP output: $[4, 16]$
      * Tensor fusion concatenation: $[4, 592]$
      * Tensor logits output: $[4, 3]$
      * Trạng thái panel: `panel_verification: VERIFIED_PASS`.
    - Phân tách dứt khoát khu vực mô phỏng số học fixture metrics (`dry_run_macro_f1 = 1.0`) sang trường `fixture_metrics_simulation_dry_run_only` với disclaimer:
      `is_research_evidence: false` và `disclaimer: SYNTHETIC_FIXTURE_DRY_RUN_NOT_RESEARCH_EVIDENCE`.
    - Biên nhận cập nhật: `research/evidence/three_class_preparation/protocol_preflight_receipt.json`.

---

## 4. Kiểm toán Cách ly với Sealed Cohorts Thật (Task 3 - Section D)

- **Truy nguyên nguyên nhân `sealed_n400_sources_checked = 0` ($systematic-debugging)**:
  - Script audit ban đầu chỉ quét các thư mục ảnh trên đĩa (`orig/` và `sd2-sp/`), không nạp gói dữ liệu kiểm định độc lập TGIF N=400 nằm trong package zip `data/research/tgif/tgif_train_clean_subset_package.zip`.
- **Nâng cấp và Kiểm toán Cách ly Triệt để (`audit_leakage_and_confounding.py` v2.0)**:
  - **Locked-Test Cohort (Phase 4B/4C.2G)**:
    - Nạp `split-lock.json`: **343 unique source IDs**.
    - Quét trực tiếp trên đĩa (`orig/testing` & `sd2-sp/testing`): **3,087 file ảnh**.
    - Tính mã băm byte stream: **2,970 distinct SHA-256 hashes**.
  - **TGIF N=400 Cohort (Phase 4C.7B)**:
    - Nạp `tgif_train_clean_subset_manifest_locked_n400.json`: **400 unique source IDs**.
    - Xác thực tính toàn vẹn gói zip `tgif_train_clean_subset_package.zip`: SHA-256 `27046ec2c10b92942cd1ef0acd10e3fdac37d55c7f2fa919ede0242d96b5ff66` khớp 100% với historic receipt.
    - Đọc trực tiếp từ zip stream và tính mã băm: **800 file ảnh**, **800 distinct SHA-256 hashes**.
  - **Đối chiếu với Development Cohort (341 pairs / 682 ảnh)**:
    - Overlap sources với Locked-Test: **0**.
    - Overlap SHA-256 hashes với Locked-Test: **0**.
    - Overlap sources với TGIF N=400: **0**.
    - Overlap SHA-256 hashes với TGIF N=400: **0**.
    - Cross-source near-duplicate collisions (dHash 64-bit $\le 3$): **0**.
    - Trạng thái kiểm toán: **`LEAKAGE_AUDIT_PASS`** (ghi nhận rõ các trường số lượng riêng biệt trong `research/evidence/three_class_preparation/leakage_and_confounding_report.json`).

---

## 5. Tiếp nhận GenImage trong Ngân sách & Điểm chặn SD1.4 (Task 4 - Sections B, E, F)

- **Tiếp nhận trong Ngân sách Cho phép**:
  - Tải về 2 shards parquet từ mirror cộng đồng `TheKernel01/Tiny-GenImage` (CC BY-NC-SA 4.0):
    * `train-00000-of-00008-0118eb3009b0b146.parquet`: 442,866,357 bytes (SHA-256 `d8795c47...`)
    * `train-00001-of-00008-0143aeb29ff43ee5.parquet`: 432,298,255 bytes (SHA-256 `34e6b216...`)
    * Tổng dung lượng tải: **875,164,612 bytes (~834.6 MB)** $\le 2\text{ GiB}$ ngân sách mạng cho phép.
  - Trích xuất thành công **341 ảnh authentic Nature control** từ ImageNet (43,514,795 bytes = 41.5 MB trên đĩa), 100% decode Pillow thành công, tính toán SHA-256 đầy đủ, lưu tại `data/research/genimage/nature_control_pilot/`.
  - Xuất danh mục: `research/evidence/three_class_preparation/source_control_cohort_manifest.json` (341 bản ghi).
- **Phân định Vai trò Hai Tập Dữ liệu**:
  - **Primary Cohort**: Mixed-source pilot gồm 341 ảnh authentic (MS-COCO val2017), 341 ảnh ai_edited (TGIF SD2 inpainting) và mẫu fully_generated tương lai.
  - **Source Control Cohort**: 341 ảnh authentic từ ImageNet (nature) được giữ hoàn toàn tách biệt, dùng làm công cụ chẩn đoán ngoại vi kiểm tra xem mô hình có bị shortcut học nhận diện phân phối nền (COCO domain vs ImageNet domain) hay thực sự học được dấu vết tạo sinh/chỉnh sửa của AI.
- **Phát hiện & Audit Điểm chặn Khách quan của SD1.4**:
  - Phân tích toàn bộ nhãn generator trong mirror Tiny-GenImage:
    Mirror chỉ chứa 7 generators (ADM, BigGAN, GLIDE, Midjourney, SD1.5, VQDM, Wukong). Generator 5 (SD1.4) có đúng **0 ảnh**.
  - Kiểm tra kho phân phối chính thức của tác giả GenImage (Zhu et al., NeurIPS 2023):
    Dữ liệu SD1.4 được đóng gói dạng multipart zip gồm 30 tệp (`imagenet_ai_0419_sdv4.zip`, `.z01` đến `.z29`), tổng dung lượng **~96.5 GiB**.
    Không thể tải một phần nhỏ độc lập mà không tải toàn bộ archive. Vượt quá ngân sách tải 2 GiB và dung lượng đĩa 5 GiB.
  - **Quy tắc Trung thực Khoa học**:
    - Không tự ý thay thế SD1.5 khi chưa có sự đồng ý của người dùng (giữ nguyên cam kết generator SD1.4).
    - Không tạo splits giả hoặc fixture giả.
    - Xuất biên nhận `research/evidence/three_class_preparation/genimage_acquisition_receipt.json` xác lập chính thức trạng thái:
      **`DATA_BLOCKED: GENIMAGE_SD14_DISTRIBUTION_EXCEEDS_BUDGET_AND_MIRROR_OMITS_SD14`**.
- **Công cụ Đóng băng Splits**:
  - Đã xây dựng sẵn script `scripts/research/lock_three_class_pilot_splits.py` tích hợp sẵn logic phân chia 250 train / 91 val cho cả 3 lớp và kiểm tra rò rỉ khi điểm chặn dữ liệu được giải quyết.

---

## 6. Mở rộng Y Văn & Định vị Khoảng Trống Nghiên Cứu (Task 5 - Section G)

- **Cập nhật Y Văn theo So-Fake (arXiv:2505.18660)**:
  - Bổ sung So-Fake (hzlsaber et al., tháng 5/2025) vào bảng đối chiếu tài liệu `docs/LITERATURE_RESEARCH_GAP.md`.
  - **Thừa nhận thực tế khoa học**: So-Fake (với mô hình nền tảng lớn So-Fake-R1) đã chính thức hóa bài toán phân loại 3 lớp (REAL / FULL_SYNTHETIC / TAMPERED). Do đó, **bản thân bài toán 3 lớp không phải là tính mới tự thân của đề tài**.
- **Định vị Khoảng Trống Kỹ thuật của Đề tài**:
  - Khoảng trống nằm ở việc giải quyết bài toán 3 lớp bằng **mô hình tích chập siêu nhẹ (< 5M parameters, MobileNetV3-small 576-d)** kết hợp **16-D Canonical DSP** có khả năng chạy **Zero-Egress client-side hoàn toàn trong trình duyệt web** qua WASM/SIMD.
  - Vector đặc trưng DSP 16-D là một **ứng viên khoảng trống cần xác minh thực nghiệm** (candidate hypothesis), không tự nhận là bằng chứng tính mới trước khi có số liệu đối chứng.

---

## 7. Tổng kết Kiểm thử & Tính Toàn Vẹn

- **Pytest**: 15/15 ML tests chuyên biệt PASS (`test_three_class_protocol.py`, `test_dataset_inventory_audit.py`, `test_leakage_and_confounding_audit.py`, `test_genimage_acquisition_plan.py`). Full suite 1,132 tests đang được giám sát xác nhận.
- **Vitest & Continuity**: 80/80 tests PASS.
- **Git Hygiene**: Sạch hoàn toàn, không có tệp nhị phân dữ liệu lớn nào bị đưa vào Git tracking.

---

## 8. Kết luận & Đề xuất Bước Thực Nghiệm Tiếp theo

1. **Kết luận Trạng thái**:
   - Khắc phục triệt để mismatch protocol và backbone PyTorch: Hoàn tất (`VERIFIED_PASS`).
   - Kiểm toán cách ly với cả hai sealed cohorts thật: Hoàn tất (`LEAKAGE_AUDIT_PASS`, 0 collisions).
   - Tiếp nhận mẫu đối chứng nguồn ImageNet (Nature control): Hoàn tất (341 ảnh, 43.5 MB).
   - Khóa splits 3 lớp: Dừng trung thực ở trạng thái **`DATA_BLOCKED`** do cơ chế phân phối của archive SD1.4 gốc (96.5 GiB) vượt ngân sách và mirror cộng đồng thiếu SD1.4.
2. **Đề xuất Bước Thực nghiệm Tiếp theo**:
   - **Phương án A (Khuyến nghị)**: Mở rộng protocol cho phép sử dụng **GenImage SD v1.5** (đã có sẵn trong shards parquet của mirror Tiny-GenImage trên đĩa), thực hiện trích xuất ngay 341 ảnh `fully_generated` từ SD1.5 mà không cần tải thêm byte nào qua mạng.
   - **Phương án B**: Tiếp nhận tập SD1.4 thông qua worker trung gian (Colab) tương tự quy trình tiếp nhận TGIF N=400 (tải và trích xuất 341 ảnh trên cloud, đóng gói zip ~45 MB bàn giao về local).
