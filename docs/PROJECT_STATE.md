# Project State: Forensics Web Lab

> **Ngày cập nhật**: 2026-09-21  
> **Commit phản ánh**: `6555b02` (và các cập nhật của Phase 3.5)  
> **Branch**: `feat/production-ai-image-forensics`

---

## 1. Thông tin chung & Đề tài nghiên cứu

* **Tên sản phẩm (UI)**: `Forensics Web Lab`
* **Tên đề tài nghiên cứu**: `Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web`
* **Mục tiêu cốt lõi**:
  * Phát hiện, phân loại và định vị dấu vết ảnh do AI tạo sinh hoặc can thiệp chỉnh sửa.
  * Chạy **100% trên thiết bị người dùng (Zero-Egress)**: không gửi ảnh lên máy chủ, không yêu cầu GPU rời phía server, bảo vệ quyền riêng tư tuyệt đối.
  * Runtime tối ưu cho Web: ONNX Runtime Web với **WASM/CPU là mặc định**, **WebGPU là tùy chọn tăng tốc**. Tự động fallback về WASM nếu WebGPU không khả dụng.
  * Kích thước mô hình mục tiêu: backbone 5–15 triệu tham số, tổng tải xuống $\le 35\text{ MB}$ sau lượng tử hóa INT8.

---

## 2. Phạm vi phiên bản (Scope Boundaries)

### 2.1 Trong phạm vi hỗ trợ (In-Scope)
* **Định dạng file**: JPEG/JPG, PNG, WebP (kiểm tra bằng magic bytes nhị phân, không chỉ extension).
* **Loại nội dung**:
  * Ảnh tạo hoàn toàn bởi AI (Fully generated: GAN, Diffusion).
  * Ảnh được AI chỉnh sửa cục bộ (AI-edited: inpainting, object removal/replacement).
  * Ảnh thông thường qua Photoshop hoặc chỉnh sửa truyền thống (dùng làm hard negatives).
  * Ảnh bị suy giảm thực tế: resize, crop, blur nhẹ, screenshot, nén lại nhiều lần.

### 2.2 Ngoài phạm vi hỗ trợ (Out-of-Scope)
* Tài liệu văn phòng: PDF, Word, Excel, PowerPoint.
* Đa phương tiện khác: Video, Audio, Deepfake video khuôn mặt theo thời gian thực.
* Nhận diện văn bản do AI viết (LLM text detection).

---

## 3. Bốn trạng thái kết luận chuẩn mực

Hệ thống tuân thủ nguyên tắc khoa học: **Không sử dụng từ "ảnh thật" như một kết luận tuyệt đối**.

1. `no_ai_evidence`: Chưa tìm thấy bằng chứng rõ ràng cho thấy ảnh được tạo hoặc sửa bằng AI trong phạm vi của bộ phân tích.
2. `fully_generated`: Có dấu hiệu và bằng chứng nhất quán cho thấy toàn bộ ảnh được tạo hoàn toàn bằng AI.
3. `ai_edited`: Có dấu hiệu một phần ảnh được chỉnh sửa, inpainting, xóa hoặc thay thế bằng AI.
4. `uncertain`: Bằng chứng chưa đủ chắc chắn, các bộ phân tích mâu thuẫn nhau, hoặc ảnh suy giảm quá nặng khiến không thể kết luận an toàn.

---

## 4. Kiến trúc kỹ thuật Client-Side

* **Frontend**: React 19 + TypeScript strict mode + Vite.
* **Xử lý bất đồng bộ**: Web Worker (`forensics.worker.ts`) cách ly hoàn toàn việc giải mã ảnh, DSP và suy luận ONNX khỏi main UI thread, giữ giao diện luôn 60 FPS mượt mà.
* **Xử lý tín hiệu số (DSP Forensics)**: Thuần TypeScript chạy trong worker (2D-FFT radial spectrum, 2D-DCT block energy, Laplacian noise residual consistency, JPEG block artifact grid, ELA).
* **Metadata & Provenance**: Trích xuất EXIF, XMP, phát hiện chữ ký phần mềm AI, adapter C2PA Content Credentials (trạng thái an toàn `unsupported` nếu runtime browser chưa có WASM validator an toàn).
* **Định vị bản đồ nhiệt (Heatmap)**: Tích lũy điểm nghi vấn từ các patch $224 \times 224$ (stride 112, 50% overlap) bằng kernel Gauss 2D, render colormap Turbo lên Canvas, có thanh trượt opacity và bật/tắt khung viền vùng nghi vấn.
* **Báo cáo**: Xuất JSON tuân thủ schema v1.0.0 và giao diện in ấn / Save-as-PDF qua print stylesheet.

---

## 5. Lịch sử Phase & Commit

| Phase | Trạng thái | Commit | Nội dung chính |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Hoàn thành | `625e584` | Audit repo, tạo branch `feat/production-ai-image-forensics`, viết toàn bộ tài liệu kiến trúc, research plan, data licenses, model card, registry.json, ADRs 0001-0005, schemas. |
| **Phase 1** | Hoàn thành | `f2a7fb4` | Thiết lập monorepo pnpm, React/Vite/TS, Python ML package skeleton, Docker, CI workflow, UI components, `@forensics/shared`, `@forensics/report`. |
| **Phase 2** | Hoàn thành | `a882eee` | Pipeline giám định trình duyệt: kiểm tra magic bytes, parser EXIF/XMP/C2PA, thuật toán DSP 2D-FFT/DCT/Noise/JPEG BAG/ELA, Web Worker với tiến trình và hủy bỏ, unit tests. |
| **Phase 3** | Hoàn thành | `6555b02` | Dataset manifest generator, adapters (GenImage, SAGI-D, RealHD, RAID), perceptual hash dedup, group split chống rò rỉ dữ liệu, augmentation suy giảm thực tế, PyTorch MobileNetV3 backbone, Focal loss, calibration, test pipeline. |
| **Phase 3.5** | Đang thực hiện | (Pending) | Thiết lập trí nhớ dự án bền vững (`AGENTS.md`, `PROJECT_STATE.md`, `CODE_MAP.md`, `SESSION_HANDOFF.md`), audit trung thực code, bảo đảm trạng thái không có model (No-Model Honest State), khảo sát checkpoint pretrained, xây dựng Model Acquisition Gate. |
| **Phase 4** | Chưa bắt đầu | *Chờ dữ liệu* | Huấn luyện mô hình nhẹ trên dữ liệu thật, đo đạc chỉ số thật, lập reliability diagram, calibration. |
| **Phase 5** | Chưa bắt đầu | *Chờ Phase 4* | Xuất ONNX, kiểm thử contract PyTorch/ONNX, lượng tử hóa INT8, tích hợp vào Web Worker, đo latency WASM/WebGPU. |
| **Phase 6** | Chưa bắt đầu | — | Hoàn thiện UX forensic, kết nối model thật vào dashboard, tinh chỉnh tương tác heatmap và báo cáo. |
| **Phase 7** | Chưa bắt đầu | — | Đóng gói sản phẩm, kiểm thử stress-test, security audit XSS/decompression bomb, chuẩn bị phát hành. |

---

## 6. Trạng thái hiện tại của Model và Dataset

* **Model Checkpoint**: `not trained` (chưa có checkpoint huấn luyện hoặc tích hợp; `models/registry.json` ghi nhận `status: not-trained`, `sizeBytes: 0`, `path: ""`).
* **Dataset**: `none` (chưa tải bất kỳ dataset nào theo đúng nguyên tắc không tự ý tải dữ liệu lớn khi chưa được phê duyệt).
* **Ứng xử khi không có model**: Giao diện và worker phải hiển thị rõ `Model not installed`, kết luận bắt buộc là `uncertain`, xác suất ba lớp là `null`, không tạo heatmap từ model giả.

---

## 7. Các Blocker hiện tại

1. **Cần phê duyệt tải dữ liệu hoặc checkpoint mẫu**: Để triển khai Phase 4 và Phase 5 cần có dữ liệu huấn luyện hoặc pretrained checkpoint phù hợp được cấp phép rõ ràng.
2. **Cổng kiểm định Model Acquisition Gate**: Bắt buộc phải hoàn thiện thủ tục kiểm toán nguồn gốc, SHA-256, license và format trước khi bất kỳ file binary nào được tải về máy.

---

## 8. Nhiệm vụ tiếp theo

1. Hoàn tất Phase 3.5: Cập nhật code để đảm bảo trạng thái No-Model hoàn toàn trung thực (không sinh xác suất giả, verdict `uncertain`, UI thông báo rõ `Model not installed`).
2. Trình duyệt ứng viên pretrained model phù hợp (ví dụ LAID MobileNetV3 / ShuffleNet cho Group A) và quy trình phê duyệt trong `docs/MODEL_ACQUISITION_GATE.md`.
3. Chờ người dùng cấp phép trước khi tải bất kỳ checkpoint nào.
