# Trạng thái Hiện tại: Forensics Web Lab (Current State)

> **Tài liệu đọc đầu tiên bắt buộc cho mọi phiên làm việc AI mới.**  
> **Documented through substantive commit**: `a480776`  
> **Implementation snapshot commit**: `311cfd3`  
> **Phase hoàn thành gần nhất**: Phase 4A.5 — Model-Agnostic Continuity Enforcement  
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

## 4. Bảng Phân loại Trạng thái Triển khai (Implementation Status Ledger)

Hệ thống phân biệt rõ ràng 5 mức độ sẵn sàng:
1. `implemented`: Đã viết mã nguồn.
2. `technically verified`: Đã có unit test / integration test tự động kiểm chứng hoạt động kỹ thuật.
3. `scientifically evaluated`: Đã đo lường trên dataset khoa học độc lập với số liệu thật (hiện tại: **chưa có tính năng nào đạt mức này**).
4. `planned`: Đang trong kế hoạch lộ trình, chưa cài đặt.
5. `blocked`: Bị chặn bởi giấy phép bản quyền, tài nguyên hoặc phụ thuộc bên ngoài.

| Hạng mục / Chức năng | Trạng thái kỹ thuật | Trạng thái khoa học | Ghi chú & Giới hạn |
| :--- | :--- | :--- | :--- |
| **Defensive Image Validation** | `technically verified` | `not-applicable` | Magic byte audit JPEG/PNG/WebP, bomb guard |
| **EXIF / XMP / IPTC Extraction** | `technically verified` | `not-applicable` | Trích xuất thẻ phần mềm AI, camera metadata |
| **C2PA Provenance Adapter** | `technically verified` | `not-applicable` | Graceful fallback sang `unsupported` |
| **2D FFT / DCT Spectral DSP** | `technically verified` | `exploratory` | Heuristic tín hiệu khám phá, không phải kết luận AI |
| **Noise Residual & JPEG ELA** | `technically verified` | `exploratory` | Heuristic phát hiện bất thường cục bộ |
| **Multimodal Fusion Engine** | `technically verified` | `unverified` | Logic tổng hợp phán quyết và abstention |
| **Honest No-Model State** | `technically verified` | `verified` | Trả về `uncertain`, `confidence: null` khi thiếu model |
| **Web Worker Async Architecture** | `technically verified` | `not-applicable` | Xử lý đa luồng ngầm không đơ UI |
| **Canvas Heatmap Viewer** | `technically verified` | `exploratory` | Bản đồ nhiệt patch score; chưa đo mIoU trên mask thật |
| **JSON & Printable PDF Report** | `technically verified` | `not-applicable` | Xuất báo cáo chuẩn schema v1 |
| **Dataset Manifest & Adapters** | `technically verified` | `pipeline-only` | Kiểm thử trên fixture hình học nội bộ |
| **Group-Split Anti-Leakage** | `technically verified` | `pipeline-only` | Ngăn chặn rò rỉ nhóm ảnh gốc `source_id` |
| **PyTorch MobileNetV3 Architecture**| `technically verified` | `architecture-only` | Kiến trúc khởi tạo hợp lệ; **chưa huấn luyện** |
| **Focal Loss & Calibration Math** | `technically verified` | `pipeline-only` | Đã test trên synthetic tensors |
| **ONNX Export & Parity Harness** | `technically verified` | `pipeline-only` | Parity test $L_\infty < 10^{-4}$ trên un-trained model |
| **Dual-Track Contamination Guards**| `technically verified` | `verified` | Cấm model sản phẩm dùng dataset phi thương mại |
| **Scientific Pilot Protocol**     | `technically verified` | `exploratory` | Hai nhánh độc lập Pilot A/B, Pilot C có baseline guard |
| **Label-Semantics Gate**          | `technically verified` | `verified` | sp là ai_edited, fr loại trừ khỏi fully_generated |
| **Continuity Checker & CI Gate**  | `technically verified` | `not-applicable` | scripts/continuity-check.mjs, 13 unit tests, CI enforced |
| **In-Browser ONNX Inference**     | `implemented` | `blocked` | Chờ checkpoint huấn luyện thật từ Phase 4 |
| **Scientific Detection Accuracy** | `planned` | `unverified` | **Not evaluated**; không có số liệu F1/ECE thật |

* **RQ1 (3-Class Generalization)**: Một mô hình nhẹ (MobileNetV3) khi huấn luyện trên dữ liệu đa nguồn có đạt Macro-F1 $\ge 0.82$ trên tập kiểm thử in-domain và không suy giảm quá $15\%$ Macro-F1 khi kiểm thử trên generator chưa từng thấy (*unseen generator*)?
* **RQ2 (Lightweight Localization)**: Đầu ra localization nhẹ (patch-level / convolutional head) có đạt mIoU $\ge 0.55$ trên ảnh chỉnh sửa cục bộ (`ai_edited`) trong khi duy trì False Positive Rate $\le 0.08$ trên ảnh thật (`authentic`)?
* **RQ3 (Zero-Egress Browser Efficiency)**: Pipeline pháp chứng kết hợp (DSP + ONNX inference) có chạy hoàn toàn trên trình duyệt client với độ trễ $\le 1500\text{ ms}$ (ảnh $512\times 512$ trên CPU WASM) và đỉnh bộ nhớ $\le 250\text{ MB}$?
* **RQ4 (Calibration & Honest Uncertainty)**: Temperature scaling có giảm Expected Calibration Error (ECE) xuống $\le 0.06$ và cơ chế ngưỡng tự động có gắn cờ trung thực trạng thái `uncertain` cho các mẫu ngoài phân phối (*out-of-distribution*)?

---

## 3. Hệ thống Nhãn Chuẩn tắc (Classification Taxonomy)

Dự án đóng băng cấu trúc 3 nhãn phân loại chính và 1 trạng thái phụ trợ:

1. **`authentic`**: Ảnh chụp từ cảm biến thực tế, không chứa pixel tạo sinh.
2. **`fully_generated`**: Toàn bộ nội dung ảnh sinh từ mô hình AI từ nhiễu hoặc văn bản, không bắt đầu từ ảnh thật cần bảo tồn danh tính.
3. **`ai_edited`**: Ảnh bắt đầu từ ảnh thật, sau đó một phần hoặc toàn bộ canvas bị biến đổi bằng generative inpainting/editing có điều kiện từ ảnh nguồn.
4. **`uncertain`** (Diagnostic State): Trạng thái phụ trợ khi độ tin cậy thấp hoặc chưa cài đặt mô hình AI.

> **Quyết định ngữ nghĩa TGIF**: Thành phần `sp` (spliced) được gán là `ai_edited`; thành phần `fr` (fully regenerated) là conditional regeneration từ ảnh MS-COCO thật nên **bị cách ly khỏi `fully_generated` và không đưa vào pilot ban đầu**.

---

## 4. Chiến lược Tách biệt Hai Luồng (Dual-Track Isolation - ADR-0006)

* **Research Track (`data/research/`)**: Dành riêng cho nghiên cứu học thuật, khóa luận và viết bài báo. Cho phép sử dụng các dataset phi thương mại (GenImage CC BY-NC-SA 4.0, TGIF CC BY-SA 4.0).
* **Product Track (`data/product/`)**: Dành cho bản web thương mại/sản phẩm. Tuyệt đối cấm sử dụng trọng số mô hình hoặc dữ liệu từ nguồn phi thương mại.
* **Fixture Track (`ml/tests/fixtures/`)**: Dành cho kiểm thử tự động nội bộ (dữ liệu tổng hợp bằng code deterministic, 0 byte external network).

---

## 5. Hiện trạng Dữ liệu và Mô hình

* **Dữ liệu ngoại vi đã tải**: `0 bytes` (chưa tải bất kỳ dataset thật nào).
* **Trọng số mô hình đã huấn luyện**: `0 bytes` (chưa có checkpoint nào).
* **Chỉ số khoa học**: `not evaluated` (chưa đo lường thực nghiệm).
* **Acquisition Plan**: Đã ban hành plan máy đọc có chữ ký SHA-256: `datasets/acquisition-plans/pilot-a-tgif.v1.json` (`7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`).

---

## 6. Kết quả Kiểm thử & Bản dựng Gần nhất (Latest Verification)

* **TypeScript & Continuity Test Suite (`pnpm test`)**: 70/70 tests passing (57 vitest tests trên 6 packages + 13 continuity checker unit tests).
* **Python Test Suite (`pytest ml/tests -v`)**: 38/38 tests passing (bao gồm 15 bài test an toàn thu nạp dữ liệu, zip slip guard, free disk check, và paired bootstrap guard).
* **Pilot Config Validation**: 2/2 pilot configs valid theo `ml/configs/validator.py`.
* **Continuity Enforcement Gate (`pnpm continuity:check`)**: `CONTINUITY_CHECK: PASS` (kiểm tra toàn bộ 3 file continuity, duplicate files, local links, placeholders, và ledger line limits).
* **Production Web Build (`pnpm build`)**: Exit code 0, bundle tối ưu hợp lệ (3.08s).
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

---

## 9. Công việc Đang thực hiện & Công việc Tiếp theo

* **Đã hoàn thành (Phase 4A.5)**: Thiết lập cơ chế kiểm tra continuity tự động, đóng chính xác Phase 4A.4 với ending commit `a480776`, thay thế thuật ngữ sang group-isolation invariant, ban hành Continuity Contract trong `AGENTS.md`, triển khai `scripts/continuity-check.mjs`, tích hợp CI và hoàn thành 13 bài test unit.
* **Hiện trạng nghiên cứu**: Dataset content vẫn `0 bytes`, model content vẫn `0 bytes`, training runs bằng `0`, scientific metrics tiếp tục giữ trạng thái `not evaluated`.
* **Công việc tiếp theo (Phase 4B / Live Data Acquisition)**: Chờ người dùng xem xét và phê duyệt `NEXT APPROVAL REQUEST` để mở khóa tải dữ liệu thật cho Pilot A (`pilot-a-tgif.v1.json`).

