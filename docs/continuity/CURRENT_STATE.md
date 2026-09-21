# Trạng thái Hiện tại: Forensics Web Lab (Current State)

> **Tài liệu đọc đầu tiên bắt buộc cho mọi phiên làm việc AI mới.**  
> **Documented through substantive commit**: `863a491`  
> **Continuity update commit**: see repository HEAD  
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
| **In-Browser ONNX Inference** | `implemented` | `blocked` | Chờ checkpoint huấn luyện thật từ Phase 4 |
| **Scientific Detection Accuracy** | `planned` | `unverified` | **Not evaluated**; không có số liệu F1/ECE thật |

---

## 5. Trạng thái Hiện tại của Model và Dataset

* **Model Checkpoint**: `none` (chưa có checkpoint; `models/registry.json` ghi nhận `status: not-trained`, `sizeBytes: 0`, `path: ""`).
* **Kích thước mô hình**: `estimated` (~10.2 MB FP32, ~2.6 MB INT8 theo tham số lý thuyết; chưa có file thật).
* **Runtime Web Engine**: `measured build artifact` (File WASM engine `ort-wasm-simd-threaded.jsep.wasm` đo được 28.3 MB; đây là runtime binary của ONNX Web, **không phải model weights**).
* **Dataset bên ngoài**: `0 bytes` (chưa tải bất kỳ byte dataset bên ngoài nào; mạng bị khóa an toàn).
* **Synthetic Smoke Fixture**: Track `fixture-only`, purpose `fixture`, `commercialUse: internal-testing-only` (sinh cục bộ bằng code, dùng cho test kỹ thuật).
* **GenImage**: Track `research-only`, `derivativeWeights: unclear`, `productionPromotion: prohibited-by-project-policy`. BigGAN archive ~24 GB nén đã được kiểm kê toàn diện.
* **TGIF / TGIF2**: Track `research-only`, `datasetLicense: CC BY-SA 4.0`, `original: CC BY 4.0 MS-COCO`. Khảo sát Nextcloud: 65.4 GB TGIF, 110 GB TGIF2 FLUX; hỗ trợ tải lẻ từng thư mục.
* **Datasets bị khóa**: `realhd`, `sagi-d`, `raid` (trạng thái `blocked`, `licenseStatus: unverified`).

---

## 6. Kết quả Kiểm thử & Bản dựng Gần nhất (Latest Verification)

* **TypeScript Test Suite (`pnpm test`)**: 56/56 tests passing trên 6 package (@forensics/shared: 34, @forensics/provenance: 3, @forensics/report: 4, @forensics/forensics: 5, @forensics/inference: 8, web: 3).
* **Python Test Suite (`pytest ml/tests -v`)**: 15/15 tests passing (kiểm soát ô nhiễm, download lock, manifest, split, focal loss, calibration).
* **Production Web Build (`pnpm build`)**: Exit code 0, 3.03s, bundle hợp lệ.
* **Clean Link Invariance**: 0 machine-local links (`file:///`, `C:\`, `D:\`) trong markdown links repository.

---

## 7. Giới hạn Kỹ thuật và Nguy cơ Ảnh hưởng Độ tin cậy

1. **Nguy cơ Shortcut Nguồn Dữ liệu**: Nếu các lớp lấy từ các nguồn ảnh khác nhau, mô hình có thể học đặc trưng camera/compression thay vì dấu vết AI. Biện pháp: group split theo `source_id`, dùng matched pairs của TGIF, chuẩn hóa tiền xử lý.
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
* `EV-NOMODEL-001`: Trung thực trạng thái không có model (`uncertain`, `confidence: null`).
* `EV-EXTERNAL-DOWNLOAD-LOCK-001`: Khóa tải ngoài bằng CLI, yêu cầu người dùng phê duyệt cụ thể.

---

## 9. Công việc Đang thực hiện & Công việc Tiếp theo

* **Đang thực hiện (Phase 4A.2)**: Đóng băng phạm vi nghiên cứu, thiết lập bộ tài liệu continuity 3 file, hoàn thành kiểm kê TGIF/TGIF2.
* **Công việc tiếp theo (Phase 4A)**: Trình đề xuất Pilot B (~3,000 ảnh từ TGIF) để người dùng phê duyệt trước khi tải bất kỳ byte dữ liệu nào.
