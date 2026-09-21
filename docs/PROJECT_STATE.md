# Project State: Forensics Web Lab

> **Ngày cập nhật**: 2026-09-21  
> **Commit phản ánh**: `cd59136` (và các cập nhật của Phase 3.6)  
> **Branch**: `feat/production-ai-image-forensics`

---

## 1. Thông tin chung & Đề tài nghiên cứu

* **Tên sản phẩm (UI)**: `Forensics Web Lab`
* **Tên đề tài nghiên cứu**: `Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web`
* **Mục tiêu cốt lõi**:
  * Phát hiện, phân loại và định vị dấu vết ảnh do AI tạo sinh hoặc can thiệp chỉnh sửa.
  * Chạy **100% trên thiết bị người dùng (Zero-Egress)**: không gửi ảnh lên máy chủ, không yêu cầu GPU rời phía server, bảo vệ quyền riêng tư tuyệt đối.
  * Runtime tối ưu cho Web: ONNX Runtime Web với **WASM/CPU là mặc định**, **WebGPU là tùy chọn tăng tốc**. Tự động fallback về WASM nếu WebGPU không khả dụng.
  * Kích thước mô hình mục tiêu: backbone 5–15 triệu tham số, tổng tải xuống $\le 35\text{ MB}$ sau lượng tử hóa INT8 (`target budget`, chưa đo trên checkpoint thật).

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
4. `uncertain`: Bằng chứng chưa đủ chắc chắn, các bộ phân tích mâu thuẫn nhau, ảnh suy giảm quá nặng, hoặc **khi chưa cài đặt mô hình học sâu hợp lệ**.

---

## 4. Kiến trúc kỹ thuật Client-Side

* **Frontend**: React 19 + TypeScript strict mode + Vite.
* **Xử lý bất đồng bộ**: Web Worker (`forensics.worker.ts`) cách ly hoàn toàn việc giải mã ảnh, DSP và suy luận ONNX khỏi main UI thread, giữ giao diện mượt mà.
* **Xử lý tín hiệu số (DSP Forensics)**: Thuần TypeScript chạy trong worker (2D-FFT radial spectrum, 2D-DCT block energy, Laplacian noise residual consistency, JPEG block artifact grid, ELA). Các tín hiệu này được phân loại là **tín hiệu khám phá sơ bộ (exploratory heuristics)**, không cấu thành kết luận mô hình.
* **Metadata & Provenance**: Trích xuất EXIF, XMP, phát hiện chữ ký phần mềm AI, adapter C2PA Content Credentials (trạng thái an toàn `unsupported` nếu runtime browser chưa có WASM validator an toàn).
* **Định vị bản đồ nhiệt (Heatmap)**: Kiến trúc tích lũy điểm nghi vấn từ các patch $224 \times 224$ (stride 112, 50% overlap) bằng kernel Gauss 2D (`architecture-only`), render colormap Turbo lên Canvas. Khi không có mô hình học sâu, tính năng này bị vô hiệu hóa hoàn toàn, không tạo dữ liệu nhiệt giả.
* **Báo cáo**: Xuất JSON tuân thủ schema v1.0.0 và giao diện in ấn / Save-as-PDF qua print stylesheet. Khi không có mô hình, báo cáo ghi rõ `modelAvailable: false`, `modelStatus: "not-installed"`, `confidence: null`, `probabilities: null`.

---

## 5. Lịch sử Phase & Commit

| Phase | Trạng thái | Commit | Nội dung chính |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Hoàn thành | `625e584` | Audit repo, tạo branch `feat/production-ai-image-forensics`, viết tài liệu kiến trúc, research plan, data licenses, model card, registry.json, ADRs 0001-0005, schemas. |
| **Phase 1** | Hoàn thành | `f2a7fb4` | Thiết lập monorepo pnpm, React/Vite/TS, Python ML package skeleton, Docker, CI workflow, UI components, `@forensics/shared`, `@forensics/report`. |
| **Phase 2** | Hoàn thành | `a882eee` | Pipeline giám định trình duyệt: kiểm tra magic bytes, parser EXIF/XMP/C2PA, thuật toán DSP 2D-FFT/DCT/Noise/JPEG BAG/ELA, Web Worker với tiến trình và hủy bỏ, unit tests. |
| **Phase 3** | Hoàn thành | `6555b02` | Dataset manifest generator, adapters (GenImage, SAGI-D, RealHD, RAID), perceptual hash dedup, group split chống rò rỉ dữ liệu, augmentation suy giảm thực tế, PyTorch MobileNetV3 backbone, Focal loss, calibration, test pipeline. |
| **Phase 3.5** | Hoàn thành | `c5dc20e`<br/>`cd59136` | Thiết lập trí nhớ dự án bền vững (`AGENTS.md`, `PROJECT_STATE.md`, `CODE_MAP.md`, `SESSION_HANDOFF.md`), audit trung thực code, áp đặt trạng thái No-Model Honest State (`uncertain`, `null` probabilities), khảo sát ứng viên pretrained, xây dựng Model Acquisition Gate v1. |
| **Phase 3.6** | Đang thực hiện | (Pending) | Evidence Hardening & Claim Correction: Khảo sát và cải chính các tuyên bố vượt quá bằng chứng, xây dựng Evidence Register máy đọc được, schema validation, cổng kiểm duyệt dữ liệu Phase 4A. |
| **Phase 4** | Chưa bắt đầu | *Chờ phê duyệt* | Huấn luyện mô hình trên dữ liệu thật được cấp phép, đo đạc chỉ số thật, lập reliability diagram, calibration. |
| **Phase 5** | Chưa bắt đầu | *Chờ Phase 4* | Xuất ONNX từ checkpoint thật, parity test, lượng tử hóa INT8 thật, đo latency WASM/WebGPU thực tế trên trình duyệt. |
| **Phase 6** | Chưa bắt đầu | — | Hoàn thiện UX forensic, kết nối model thật vào dashboard, tinh chỉnh tương tác heatmap và báo cáo. |
| **Phase 7** | Chưa bắt đầu | — | Đóng gói sản phẩm, kiểm thử stress-test, security audit XSS/decompression bomb, chuẩn bị phát hành. |

---

## 6. Trạng thái Hiện tại của Model và Dataset (Tình trạng Minh chứng)

* **Model Checkpoint**: `none` (chưa có checkpoint huấn luyện hoặc tích hợp; `models/registry.json` ghi nhận `status: not-trained`, `sizeBytes: 0`, `path: ""`).
* **Kiến trúc mô hình**: `architecture candidate` (`ARCH-C2-INHOUSE-MNV3` định nghĩa trong `ml/training/mobilenetv3_forensics.py`).
* **Dung lượng mô hình**: `estimated` (~10.2 MB FP32, ~2.6 MB INT8; tính toán dựa trên số lượng tham số lý thuyết, **chưa đo trên artifact file thật**).
* **Khả năng phân loại 3 lớp & Localization**: `architecture-only` (mã xử lý đã có, nhưng chưa có trọng số mô hình để kiểm chứng khả năng phát hiện hay định vị thực tế; khả năng dự đoán thực tế là `unverified`).
* **ONNX Parity**: `pipeline-only` (chỉ chứng minh pipeline kiểm thử chạy thành công trên mô hình PyTorch chưa huấn luyện trong unit test; chưa thực hiện trên checkpoint thật).
* **WASM Runtime**: `pipeline-only` (file asset `.wasm` có trong bundle Vite, nhưng khả năng chạy ổn định trên các trình duyệt thực tế Chrome/Edge/Firefox/Safari là `unverified`).
* **Chỉ số khoa học**: `not evaluated` (tuyệt đối không bịa đặt chỉ số accuracy, F1, ECE, AUROC, mIoU).
* **Dataset**: `none` (chưa tải bất kỳ dataset nào).
* **Tập dữ liệu nhỏ (< 50 MB)**: Chỉ dùng cho `pipeline smoke test`, không đủ đa dạng và không được dùng để đưa ra bất kỳ kết luận khoa học nào.
* **Giấy phép trọng số**: `not-applicable` (chưa có trọng số tồn tại; quyền phân phối sau này phụ thuộc vào mã nguồn, pretrained backbone ImageNet và giấy phép các dataset huấn luyện).

---

## 7. Các Blocker hiện tại

1. **Cần phê duyệt danh mục và điều khoản sử dụng Dataset trước Phase 4A**: Bắt buộc phải hoàn thiện thủ tục kiểm toán nguồn gốc, kích thước và giấy phép trong `docs/PHASE_4A_DATA_FEASIBILITY.md` trước khi tải bất kỳ dữ liệu nào.
2. **Cổng kiểm định Model Acquisition Gate v2**: Bắt buộc tuân thủ 17 tiêu chuẩn kỹ thuật và pháp lý trước khi một model được chuyển sang trạng thái `ready`.

---

## 8. Nhiệm vụ tiếp theo

1. Hoàn thành Phase 3.6:
   - Xây dựng `docs/EVIDENCE_REGISTER.md` và `research/evidence/phase-3.6/evidence-manifest.json` theo schema chuẩn.
   - Thêm automated test cho schema validation và registry integrity.
   - Hoàn thiện `docs/PHASE_4A_DATA_FEASIBILITY.md` kèm approval gate.
2. Chờ người dùng xem xét và cấp phép lựa chọn dataset trước khi bước vào Phase 4A.
