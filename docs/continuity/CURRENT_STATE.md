# Trạng thái Hiện tại: Forensics Web Lab (Current State)

> **Tài liệu đọc đầu tiên bắt buộc cho mọi phiên làm việc AI mới.**  
> **Documented through substantive commit**: `8de2151`  
> **Ending commit Phase 4A.5**: `8de2151`  
> **Phase hoàn thành gần nhất**: Phase 4B.0 — TGIF Masks Live Acquisition Smoke  
> **Branch**: `feat/production-ai-image-forensics`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không commit trực tiếp)  
> **Working tree**: clean  
> **Tuyên bố khoa học tối thượng**:  
> **`No scientific model-performance claim is currently supported.`**

---

## 1. Định vị Đề tài & Mục tiêu Nghiên cứu

* **Tên đề tài**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Tên ứng dụng web minh chứng**: `Forensics Web Lab`
* **Mục tiêu nghiên cứu**:
  * Phát triển công cụ nhẹ chạy trực tiếp trên trình duyệt (Zero-Egress), phát hiện ảnh AI tạo sinh toàn phần (`fully_generated`), ảnh chỉnh sửa cục bộ (`ai_edited`), và ảnh thông thường (`authentic`).
  * Ứng dụng mô hình nơ-ron tích chập nhẹ (backbone MobileNetV3), hiệu chuẩn xác suất (Temperature Scaling), và kết hợp tín hiệu pháp chứng số (DSP 2D FFT/DCT, noise residual, JPEG grid).
  * Web app đóng vai trò là sản phẩm minh chứng thực nghiệm (*proof-of-concept demonstration artifact*) cho khóa luận và bài báo nghiên cứu.

---

## 2. Câu hỏi Nghiên cứu Đã Đóng băng (Research Questions)

* **RQ1 (Three-Class Generalization)**: Mô hình nhẹ có phân biệt được `authentic`, `fully_generated` và `ai_edited` trên dữ liệu chưa thấy với độ chính xác vượt trội random baseline hay không?
* **RQ2 (Multimodal Evidence Fusion)**: Việc kết hợp mô hình thị giác với siêu dữ liệu C2PA/EXIF và tín hiệu DSP có cải thiện Macro-F1 hoặc độ hiệu chuẩn xác suất (ECE) hay không?
* **RQ3 (ONNX INT8 Quantization Trade-Offs)**: Quá trình lượng tử hóa INT8 ảnh hưởng thế nào đến chất lượng dự đoán, kích thước mô hình và tốc độ suy luận?
* **RQ4 (Browser CPU/WASM Runtime)**: Mô hình có thể vận hành ổn định trên trình duyệt web thông qua CPU/WASM với độ trễ và bộ nhớ cho phép tương tác hay không?
* **Auxiliary RQ5 (Inpainting Localization)**: Phương pháp sliding window patch heatmap có định vị chấp nhận được vùng chỉnh sửa khi đối chiếu ground-truth mask hay không?

---

## 3. Phạm vi Nghiên cứu & Bốn Trạng thái Đầu ra

### 3.1. Phạm vi hỗ trợ
* **Định dạng ảnh**: JPEG, PNG, WebP (kiểm tra bằng magic bytes nhị phân, guard chống decompression bomb).
* **Không gian phân loại 3 lớp**: `authentic`, `fully_generated`, `ai_edited`.
* **Trạng thái quyết định sau kiểm chuẩn**: `uncertain` (áp dụng khi độ tin cậy thấp hoặc tín hiệu mâu thuẫn; **không phải nhãn huấn luyện thứ tư**).
* **Ngoài phạm vi**: Không hỗ trợ video deepfake; không gửi dữ liệu về máy chủ; không tuyên bố tính nguyên bản tuyệt đối của ảnh.

### 3.2. Bốn trạng thái đầu ra của Web App
1. `no_ai_evidence`: Không phát hiện dấu vết AI; nhất quán với ảnh thông thường (không gọi là "ảnh thật 100%").
2. `fully_generated`: Dấu vết tạo sinh AI toàn phần.
3. `ai_edited`: Dấu vết chỉnh sửa, thay thế nội dung cục bộ bằng AI.
4. `uncertain`: Mức độ tin cậy thấp, tín hiệu mâu thuẫn, hoặc **chưa cài đặt mô hình**.

---

## 4. Chiến lược Tách biệt Hai Luồng (Dual-Track Isolation - ADR-0006)

* **Research Track (`data/research/`)**: Dành riêng cho nghiên cứu học thuật, khóa luận và viết bài báo. Cho phép sử dụng các dataset phi thương mại (GenImage CC BY-NC-SA 4.0, TGIF CC BY-SA 4.0).
* **Product Track (`data/product/`)**: Dành cho bản web thương mại/sản phẩm. Tuyệt đối cấm sử dụng trọng số mô hình hoặc dữ liệu từ nguồn phi thương mại.
* **Fixture Track (`ml/tests/fixtures/`)**: Dành cho kiểm thử tự động nội bộ (dữ liệu tổng hợp bằng code deterministic, 0 byte external network).

---

## 5. Hiện trạng Dữ liệu và Mô hình

* **Dữ liệu ngoại vi đã tải**: `42,327,429 bytes` (~40.37 MiB nhận qua mạng; tổng giải nén 141,559,934 bytes across 31,238 mask PNG files). **External dataset content không còn 0 bytes**.
* **Phạm vi tải**: **Duy nhất component `tgif-masks` đã được tải**. Các component `tgif-orig` (7,301,444,403 bytes) và `tgif-sd2-sp` (18,575,836,774 bytes) vẫn **chưa tải** và tiếp tục bị khóa (locked).
* **Archive Checksum**: SHA-256 archive tải về là `62c89a65441a35e6bd10d49abec8f2cad9ab028e91edf3d72b344ec29bfa0fe9`.
* **Biên nhận thu nạp**: Lưu tại `data/research/tgif/acquisition-receipt.json`.
* **Manifest mặt nạ cục bộ**: Lưu tại `data/research/tgif/manifests/masks-manifest.jsonl` (31,238 bản ghi).
* **Trọng số mô hình đã tải / huấn luyện**: `0 bytes` (chưa tải checkpoint hay trọng số nào).
* **Số lượt huấn luyện (Training runs)**: `0`.
* **Chỉ số khoa học**: `not evaluated` (chưa đo lường thực nghiệm; mask acquisition chỉ chứng minh pipeline và inventory, chưa chứng minh chất lượng phát hiện).
* **Acquisition Plan**: Plan máy đọc `datasets/acquisition-plans/pilot-a-tgif.v1.json` (SHA-256: `7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`) được giữ nguyên toàn vẹn.

---

## 6. Kết quả Kiểm thử & Bản dựng Gần nhất (Latest Verification)

* **TypeScript & Continuity Test Suite (`pnpm test`)**: 70/70 tests passing (57 vitest tests trên 6 packages + 13 continuity checker unit tests).
* **Python Test Suite (`pytest ml/tests -v`)**: 46/46 tests passing (bao gồm 23 bài test an toàn thu nạp dữ liệu, zip slip guard, hard network ceiling, hostname redirect restriction, và component scoping).
* **Pilot Config Validation**: 2/2 pilot configs valid theo `ml/configs/validator.py`.
* **Continuity Enforcement Gate (`pnpm continuity:check`)**: `CONTINUITY_CHECK: PASS`.
* **Production Web Build (`pnpm build`)**: Exit code 0, bundle tối ưu hợp lệ (3.11s).
* **Clean Link Invariance**: 0 machine-local links (`file:///`, `C:\`, `D:\`) trong toàn bộ markdown và evidence repository.

---

## 7. Giới hạn Kỹ thuật và Nguy cơ Ảnh hưởng Độ tin cậy

1. **Nguy cơ Shortcut Nguồn Dữ liệu**: Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec, quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh vẫn phải được đo bằng baseline và source-held-out evaluation.
2. **Không có Model AI Cài Đặt**: Hiện tại toàn bộ kết quả phân tích AI trên UI hiển thị trung thực là `uncertain` với banner "Model not installed".
3. **Chưa có Đo đạc Trực tiếp Trình duyệt Đa Thiết bị**: Runtime latency và peak memory trên mobile/low-end devices cần được kiểm chứng khi có checkpoint thật.

---

## 8. Danh mục Minh chứng Then chốt (Key Evidence Register)

* `EV-CONTINUITY-001`: Bộ 3 file continuity thống nhất tại `docs/continuity/`.
* `EV-RESEARCH-SCOPE-001`: Đóng băng phạm vi nghiên cứu 3 lớp và trạng thái `uncertain`.
* `EV-RQ-METRIC-MAP-001`: Khóa 4 câu hỏi nghiên cứu và hệ thống chỉ số toàn diện.
* `EV-TGIF-METADATA-001`: Kiểm kê metadata TGIF/TGIF2 qua Nextcloud (65.4 GB, hỗ trợ tải lẻ).
* `EV-TGIF-LICENSE-001`: Thẩm định giấy phép TGIF CC BY-SA 4.0 và MS-COCO CC BY 4.0.
* `EV-GENIMAGE-REMOTE-METADATA-001`: Khảo sát Google Drive GenImage (BigGAN ~24 GB multi-part).
* `EV-LABEL-GATE-001`: Đóng băng 3 nhãn chuẩn tắc và loại bỏ nhãn giả tạo.
* `EV-TGIF-SEMANTICS-001`: Thẩm định `sp` (ai_edited) và `fr` (quarantined/conditional regeneration).
* `EV-PILOT-DESIGN-001`: Kiến trúc thí nghiệm pilot hai nhánh độc lập A/B/C.
* `EV-SHORTCUT-PROTOCOL-001`: Giao thức chống rò rỉ và kiểm soát shortcut (dedup, group-split, metadata guard).
* `EV-PILOT-CONFIGS-001`: Cấu hình pilot YAML máy đọc và validator tự động.
* `EV-ACQUISITION-DRYRUN-001`: Dry-run thu nạp dữ liệu Pilot A/B đạt 0 bytes ngoại vi.
* `EV-PHASE4A3-CORRECTION-001`: Sửa toàn bộ commit references của Phase 4A.3 về `9ab0e67`.
* `EV-TGIF-CARDINALITY-001`: Kiểm toán cardinality TGIF phân biệt verified, estimated và unverified.
* `EV-ACQUISITION-PLAN-001`: Kế hoạch thu nạp dữ liệu máy đọc và schema v1 có mã băm SHA-256.
* `EV-DOWNLOADER-SAFETY-001`: Bộ lọc an toàn tải file (.part, resume, checksum, safe-extract, staging).
* `EV-BOOTSTRAP-GUARD-001`: Cổng thống kê Paired Stratified Bootstrap 95% CI cho metadata baseline guard.
* `EV-PHASE4A4-CLOSURE-001`: Đóng Phase 4A.4 với ending commit a480776, chuẩn hóa group-isolation invariant.
* `EV-CONTINUITY-CONTRACT-001`: Quy chế Continuity Contract trong AGENTS.md cho mọi coding agent.
* `EV-CONTINUITY-CHECKER-001`: Công cụ kiểm tra continuity scripts/continuity-check.mjs đa nền tảng.
* `EV-CONTINUITY-TESTS-001`: Bộ kiểm thử 13 unit tests cho continuity checker trong scripts/__tests__/.
* `EV-CI-CONTINUITY-001`: Tích hợp continuity gate vào quy trình CI GitHub Actions.
* `EV-PHASE4B0-SMOKE-001`: Live acquisition smoke tải thành công `tgif-masks` (42,327,429 bytes, trần 64 MiB).
* `EV-PHASE4B0-INVENTORY-001`: Kiểm toán toàn diện 31,238 file mask PNG trên 2,242 `source_id`; xác minh cardinality và mapping.

---

## 9. Công việc Đang thực hiện & Công việc Tiếp theo

* **Đã hoàn thành (Phase 4B.0)**: Mở rộng component-scoped acquisition và trần mạng trong `acquire.py`; vượt qua toàn bộ 46 tests Python và 70 tests TS; hoàn thành live acquisition smoke cho `tgif-masks` (42,327,429 bytes nhận, archive SHA-256 `62c89a65...`); trích xuất an toàn và kiểm toán 31,238 file mask PNG (141,559,934 bytes uncompressed); sinh local manifest và receipt; xác minh `mask_count`, `filename_convention`, `source_id_extraction_method` chuyển sang `verified`.
* **Hiện trạng nghiên cứu**: External dataset content đạt 42,327,429 bytes; `orig` và `sd2-sp` tiếp tục locked; model content vẫn `0 bytes`; training runs bằng `0`; scientific detection metrics giữ trạng thái `not evaluated`.
* **Công việc tiếp theo (Phase 4B.1 / Live Data Acquisition)**: Chờ người dùng xem xét và phê duyệt `NEXT APPROVAL REQUEST` để mở khóa tải hai component còn lại của TGIF (`tgif-orig`: ~7.30 GB, `tgif-sd2-sp`: ~18.58 GB) nhằm hoàn thiện bộ dữ liệu cho Pilot A.

