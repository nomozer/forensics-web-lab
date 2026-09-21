# Báo cáo Nghiệm thu Giai đoạn Phase 4A.4 (Phase 4A.4 Completion Report)

> **Giai đoạn**: Phase 4A.4 — Evidence Correction and Acquisition Safety Gate  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting commit**: `9ab0e67`  
> **Implementation snapshot commit**: Xác định sau commit báo cáo qua `git log -1 --format=%H -- research/evidence/phase-4a.4/PHASE_REPORT.md`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không commit trực tiếp)  
> **Evidence generated at UTC**: `2026-09-21T13:25:00Z`  
> **Working tree**: clean  
> **Tuyên bố trung thực khoa học**: `No scientific model-performance claim is currently supported.`  
> **Phán quyết giai đoạn (Verdict)**: **`PASS`**

---

## 1. Mục tiêu Giai đoạn

1. **Sửa minh chứng và commit references Phase 4A.3**: Cập nhật chính xác starting commit `0c42f3c`, ending commit `9ab0e67`, main `460f6d5`, working tree clean; loại bỏ triệt để các placeholder `See repository HEAD` và commit trung gian `863a491`.
2. **Hiệu chỉnh các tuyên bố khoa học**: Loại bỏ hoàn toàn các khẳng định cực đoan ("triệt tiêu 100% shortcut", "loại bỏ hoàn toàn shortcut", "matched pairs hoàn hảo", "xác minh tuyệt đối 100%"); thay bằng cách diễn đạt định lượng chính xác về việc matched-pair giảm đáng kể nguy cơ học đặc trưng nguồn ảnh nhưng vẫn cần đo lường bằng baseline và source-held-out evaluation.
3. **Chuẩn hóa Metadata-Only Baseline Guard**: Thay thế ngưỡng tùy tiện "+15%" bằng giao thức thống kê Paired Stratified Bootstrap 95% Confidence Interval ($\Delta\text{Macro-F1} > 0.0$) trên test set; Pilot C tiếp tục giữ trạng thái `exploratory`.
4. **Kiểm toán Cardinality và Quan hệ TGIF**: Thẩm định nguồn chính thức TGIF/TGIF2 (arXiv:2407.11566) và metadata Nextcloud, phân loại rõ ràng 10 trường thành `verified`, `estimated`, và `unverified` trong `tgif-cardinality-audit.json`.
5. **Ban hành Acquisition Plan máy đọc có mã băm**: Tạo `datasets/acquisition-plans/pilot-a-tgif.v1.json` tuân thủ JSON Schema `docs/schemas/acquisition-plan.v1.schema.json`, tính SHA-256 xác thực.
6. **Hoàn thiện Acquisition Safety Downloader**: Bổ sung 15 chốt an toàn trong `ml/datasets/acquire.py` (free disk, `.part`, resume, checksum, zip slip, symlink, staging, receipt, hash approval gate).
7. **Kiểm thử Offline Toàn diện**: Viết 15 bài test unit trong `ml/tests/test_acquisition_safety.py` sử dụng mock/fixtures, bảo đảm mạng ngoại vi đạt tuyệt đối `0 bytes`.

---

## 2. Kết quả Thực hiện Chi tiết

### 2.1. Sửa Commit References Phase 4A.3
- Đã sửa toàn bộ tài liệu continuity và báo cáo nghiệm thu Phase 4A.3:
  - `docs/continuity/CURRENT_STATE.md`: `Documented through substantive commit: 9ab0e67`.
  - `docs/continuity/STATUS_LEDGER.md`: `Ending commit: 9ab0e67`.
  - `research/evidence/phase-4a.3/PHASE_REPORT.md`: `Ending commit: 9ab0e67`, `Working tree khi nghiệm thu: clean`.
  - `research/evidence/phase-4a.3/evidence-manifest.json`: `commitAudited: 9ab0e67`.
- Không còn bất kỳ placeholder `See repository HEAD` nào tồn tại trong repository.

### 2.2. Hiệu chỉnh Tuyên bố Khoa học (Calibrated Claims)
- Đã rà soát và thay thế toàn bộ các phát biểu tuyệt đối hóa bằng công thức khoa học chuẩn tắc:
  > *Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec, quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh vẫn phải được đo bằng baseline và source-held-out evaluation.*
- Giữ vững các kết luận phương pháp luận:
  - `sp` phù hợp với nhãn `ai_edited` (`project-taxonomy-decision`).
  - `fr` là conditional regeneration từ ảnh thật (`verified-from-primary-source`).
  - `fr` không thuộc `fully_generated` theo taxonomy hiện tại (`project-taxonomy-decision`).
  - `fr` tiếp tục ở trạng thái quarantined trong pilot đầu tiên (`pipeline-enforced`).

### 2.3. Chuẩn hóa Metadata-Only Baseline Guard (Thống kê Paired Bootstrap)
- Đã xây dựng module `ml/evaluation/bootstrap_guard.py`:
  - Tính $\Delta\text{Macro-F1} = \text{Macro-F1}_{\text{visual}} - \text{Macro-F1}_{\text{metadata}}$.
  - Ước lượng Paired Stratified Bootstrap 95% CI ($1,000$ resamples) trên tập test set.
  - Báo cáo đầy đủ: visual Macro-F1, metadata Macro-F1, $\Delta\text{Macro-F1}$, bootstrap 95% CI, balanced accuracy, confusion matrices, sample counts từng lớp.
  - Điều kiện chấp thuận: Cận dưới 95% CI của $\Delta\text{Macro-F1} > 0.0$.
  - Cập nhật quy tắc vào `docs/PILOT_PROTOCOL.md`, `docs/EVALUATION.md`, và `docs/DATASETS.md`.

### 2.4. Kiểm toán Cardinality TGIF (`tgif-cardinality-audit.json`)
- Đã phân loại minh bạch 10 trường cardinality:
  1. `authentic_images_orig`: 3,124 ảnh MS-COCO (`verified`).
  2. `mask_count`: Ước tính ~6,248 masks (2 per source image: segm & bbox) (`estimated`).
  3. `sd2_sp_images`: 18,744 ảnh inpainting SD2 (`verified`).
  4. `split_counts`: 78.1% train (2,440/14,640), 10.9% val (341/2,046), 11.0% test (343/2,058) (`verified`).
  5. `edit_variants_per_source_id`: 6 variants per source_id trong `sd2-sp` (2 masks * 3 batch) (`verified`).
  6. `relationship_orig_to_manipulated`: 1 authentic : 6 edited trong `sd2-sp`, 1 : 24 trên toàn bộ 4 sub-datasets TGIF (74,976 manipulated) (`verified`).
  7. `mask_sharing_relation`: Một mask dùng chung cho 3 biến thể batch cùng loại (`verified`).
  8. `filename_convention`: Tiền tố 12 chữ số COCO ID (`estimated`).
  9. `source_id_extraction_method`: Trích xuất số định danh COCO ID (`estimated`).
  10. `download_granularity`: Nextcloud dynamic ZIP theo subfolder (`verified`).

### 2.5. Kế hoạch Thu nạp Dữ liệu Máy đọc (`pilot-a-tgif.v1.json`)
- Đã ban hành JSON Schema: `docs/schemas/acquisition-plan.v1.schema.json`.
- Đã ban hành Acquisition Plan: `datasets/acquisition-plans/pilot-a-tgif.v1.json`.
- Mã băm xác thực SHA-256: `7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`.
- Plan quy định:
  - Destination: `data/research/tgif`.
  - Required Free Disk: `59,055,800,320` bytes (~55 GB).
  - Group Split Key: `source_id` (Zero-Leakage Invariant).
  - Resume capability: `resumeSupported: false` (Nextcloud dynamic zip).
  - Checksum policy: SHA-256 local calculation and zip test.
  - Staging extraction with path traversal and symlink guards.

### 2.6. Nâng cấp Downloader Safety (`ml/datasets/acquire.py`)
- Trang bị đầy đủ 15 chốt an toàn:
  1. Preflight free disk check (`check_free_disk_space`).
  2. Ghi file tạm `.part`.
  3. Xử lý resume (nối tiếp khi hỗ trợ, ghi đè khi không hỗ trợ).
  4. Content-Length mismatch check (tự động xóa `.part`).
  5. Checksum SHA-256 check (tự động xóa `.part`).
  6. Atomic rename `.part` $\rightarrow$ final path.
  7. Idempotent rerun check (bỏ qua nếu file đã tồn tại và đúng hash).
  8. Safe zip extract chống Zip Slip (chặn `..` và absolute paths).
  9. Safe zip extract chống symlink độc hại.
  10. Staging directory isolation (`.staging`).
  11. Chuyển staging sang destination nguyên tử.
  12. Sinh `acquisition-receipt.json` lưu vết nguồn gốc (lineage).
  13. Hỗ trợ CLI `--plan` và `--approved-plan-sha256`.
  14. Khóa cứng việc tải mạng ngoài trong Phase 4A.4.
  15. Tách biệt rõ ràng request metadata và request content.

---

## 3. Tổng hợp Kiểm thử & Đo đạc Kỹ thuật

```text
============================= TEST EXECUTION SUMMARY =============================
Vitest (Workspace Packages & Web App):
  * Total Test Files:        6 passed (6)
  * Total Tests:             57 passed (57)
  * Failed Tests:            0
  * Test Duration:           ~8.8s

Pytest (Python ML Suite):
  * Total Test Files:        4 passed (4)
  * Total Tests:             38 passed (38)
    - ml/tests/test_acquisition_safety.py:  15 passed (Phase 4A.4)
    - ml/tests/test_contamination_guard.py: 10 passed
    - ml/tests/test_model_pipeline.py:       5 passed
    - ml/tests/test_pilot_protocol.py:       8 passed
  * Failed Tests:            0
  * Test Duration:           ~7.5s

Config Validator:
  * Pilot Configs Validated: 2 (pilot_tgif_edit.yaml, pilot_genimage_generated.yaml)
  * Status:                  All valid, label semantics gate passed.

Web Application Build:
  * Exit Code:               0
  * Build Duration:          4.57s
  * Dist Artifacts:
    - dist/index.html:                                 1.08 kB
    - dist/assets/forensics.worker.js:               437.93 kB
    - dist/assets/ort-wasm-simd-threaded.jsep.wasm:  28.31 MB
    - dist/assets/index.js:                          683.69 kB
    - dist/assets/index.css:                           3.97 kB

Clean Link Invariance:
  * Machine-local links (file:///, C:\, D:\):        0
==================================================================================
```

### Hạch toán Mạng (Network Accounting):
* Metadata requests: `0`
* Metadata response bytes: `0`
* Dataset content requests: `0`
* Dataset content bytes: `0`
* Model weights requests: `0`
* Model weights bytes: `0`
* Training runs executed: `0`

---

## 4. Đánh giá Tiêu chí Nghiệm thu (Pass/Fail Verdict)

| Tiêu chí | Yêu cầu | Thực tế đạt được | Phán quyết |
| :--- | :--- | :--- | :--- |
| **Phase 4A.3 Commit Reference** | Ending commit ghi đúng `9ab0e67` | `9ab0e67` đã cập nhật đầy đủ | **PASS** |
| **Eliminate Placeholders** | Không còn placeholder `See repository HEAD` | 0 placeholder trong báo cáo Phase 4A.3 | **PASS** |
| **Calibrated Shortcut Claims** | Loại bỏ phát biểu "triệt tiêu 100% shortcut" | Đã thay thế bằng phát biểu khoa học định lượng | **PASS** |
| **Statistical Metadata Guard** | Thay ngưỡng 15% bằng Paired Bootstrap 95% CI | Đã triển khai và unit-test $CI_{\text{lower}} > 0.0$ | **PASS** |
| **TGIF Cardinality Audit** | Phân loại verified, estimated, unverified | 10 trường đã được kiểm toán minh bạch | **PASS** |
| **Machine-Readable Plan** | Schema v1, plan JSON có SHA-256 | Plan và Schema hợp lệ, SHA-256 xác thực | **PASS** |
| **Downloader Safety** | 15 chốt an toàn được kiểm thử offline | 15 unit tests offline đạt 100% | **PASS** |
| **External Dataset Bytes** | Bắt buộc `0 bytes` | `0 bytes` | **PASS** |
| **Model Weights Bytes** | Bắt buộc `0 bytes` | `0 bytes` | **PASS** |
| **Training Runs** | Bắt buộc `0` | `0` | **PASS** |
| **Full Test Suite** | Toàn bộ vitest và pytest pass | 57 vitest + 38 pytest passing (100%) | **PASS** |
| **Web Build** | Exit code 0, không lỗi compile | Exit code 0, build hoàn tất trong 4.57s | **PASS** |
| **Continuity Documents** | Cập nhật CURRENT_STATE, CODE_INDEX, STATUS_LEDGER | Đã cập nhật đầy đủ, max 20 dòng trên ledger | **PASS** |
| **Git Safety** | Main `460f6d5` giữ nguyên, không push/deploy | Main bảo toàn nguyên vẹn, working tree clean | **PASS** |

**KẾT LUẬN GIAI ĐOẠN**: **`PASS`**

---

## 5. Yêu cầu Phê duyệt Tải Dữ liệu (NEXT APPROVAL REQUEST)

Dự án đệ trình kế hoạch thu nạp dữ liệu máy đọc đã được kiểm thử an toàn toàn diện để người dùng xem xét và phê duyệt trước khi tải thật:

* **Plan ID**: `pilot-a-tgif`
* **Plan Version**: `1.0.0`
* **Plan File Path**: `datasets/acquisition-plans/pilot-a-tgif.v1.json`
* **Approved Plan SHA-256**: `7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`
* **Dataset / Component**:
  1. `tgif-orig`: 7,301,444,403 bytes (~6.80 GB) [verified] | 3,124 authentic images [verified]
  2. `tgif-masks`: 42,362,470 bytes (~0.04 GB) [verified] | ~6,248 binary masks [estimated]
  3. `tgif-sd2-sp`: 18,575,836,774 bytes (~17.30 GB) [verified] | 18,744 inpainting images [verified]
* **Tổng dung lượng tải nén**: `25,919,643,647` bytes (~24.14 GB)
* **Dung lượng giải nén ước tính**: ~27.0 GB
* **Dung lượng ổ đĩa trống yêu cầu**: $\ge 55\text{ GB}$ (`59,055,800,320` bytes)
* **Destination**: `data/research/tgif`
* **License**: CC BY-SA 4.0 (TGIF) / CC BY 4.0 (MS-COCO 2017)
* **License Track**: `research-only` (tuyệt đối cấm đưa vào sản phẩm)
* **Download Method**: Nextcloud dynamic ZIP per subfolder
* **Resume Capability**: `resumeSupported: false` (Nextcloud endpoint sinh ZIP động trên máy chủ)
* **Checksum Policy**: Tính SHA-256 cục bộ và kiểm tra tính toàn vẹn ZIP ngay sau khi tải
* **Mục đích khoa học**: Thử nghiệm Pilot A (Authentic vs AI-Edited classification và inpainting localization trên cặp matched pairs MS-COCO với phân chia Zero-Leakage theo `source_id`).
* **Lệnh thực thi chính xác (sau khi người dùng phê duyệt)**:
  ```bash
  python -m ml.datasets.acquire \
    --plan datasets/acquisition-plans/pilot-a-tgif.v1.json \
    --execute \
    --approved-plan-sha256 7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e
  ```
