# Trạng thái Dự án: Forensics Web Lab (Project State)

> **Cập nhật lúc**: Phase 4A.1  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting commit**: `a0e71d5`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn)  
> **Nguyên tắc cốt lõi**: Trung thực khoa học (*Scientific Honesty*), không dùng số liệu giả, không gọi kiến trúc là checkpoint, cách ly tuyệt đối hai luồng Research Track và Product Track.

---

## 1. Mục tiêu và Định vị Sản phẩm

* **Tên hiển thị giao diện**: `Forensics Web Lab`
* **Tên đề tài nghiên cứu**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Định vị kỹ thuật**:
  * Chạy **100% trên thiết bị người dùng (Zero-Egress)**: `architecture-supported` (không gửi ảnh lên server; `runtime-network-verification: unverified` - đang có backlog kiểm thử E2E chặn network).
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
* **Đầu ra hệ thống**: 4 trạng thái chuẩn tắc (`no_ai_evidence`, `fully_generated`, `ai_edited`, `uncertain`). Tuyệt đối không dùng từ "ảnh thật" như một kết luận tuyệt đối.

### 2.2 Ngoài phạm vi (Out-of-Scope)
* Không khẳng định tính nguyên bản tuyệt đối của ảnh.
* Không hỗ trợ định dạng video/deepfake video trong phạm vi web nhẹ này.
* Không gửi ảnh về server hay xử lý đám mây (Cloud API).

---

## 3. Hệ thống Phân loại Bằng chứng Chuẩn mực

Mọi tuyên bố kỹ thuật trong dự án bắt buộc phải thuộc một trong 8 trạng thái minh chứng:
* `verified`: Đã có minh chứng trực tiếp, đo lường thật và có thể tái tạo 100%.
* `reported`: Được báo cáo trong y văn/tài liệu tham khảo nhưng chưa kiểm chứng độc lập.
* `estimated`: Giá trị ước lượng lý thuyết (tính toán toán học), chưa đo trên artifact thật.
* `architecture-only`: Mới tồn tại ở tầng thiết kế kiến trúc hoặc mã nguồn PyTorch/TS, chưa huấn luyện.
* `pipeline-only`: Chỉ chứng minh pipeline kỹ thuật/dòng chảy dữ liệu hoạt động với dữ liệu giả định.
* `unverified`: Chưa có minh chứng hoặc chưa thực hiện đo lường.
* `blocked`: Bị chặn do thiếu dữ liệu, trọng số, giấy phép bản quyền hoặc tài nguyên phần cứng.
* `rejected`: Đã đánh giá toàn diện và kết luận không phù hợp với tiêu chuẩn dự án.

---

## 4. Chính sách Hai Luồng Dữ liệu & Trọng số (Dual-Track Policy)

Dự án áp dụng quy định cách ly nghiêm ngặt giữa hai luồng:
1. **Research Track**:
   - Chấp nhận các dataset học thuật mang điều khoản phi thương mại / Share-Alike (ví dụ: `GenImage` theo `CC BY-NC-SA 4.0 with additional dataset terms`).
   - Lưu trữ dữ liệu tại `data/research/` và trọng số tại `models/research/`.
   - Phục vụ đánh giá benchmark, so sánh học thuật; **nghiêm cấm** đưa vào production bundle của web client.
2. **Product Track**:
   - Chỉ sử dụng dữ liệu tự sở hữu hoặc dữ liệu có giấy phép thương mại và phân phối trọng số rõ ràng (`product-eligible`).
   - Lưu trữ tại `data/product/` và trọng số tại `models/product/`.
   - Phải vượt qua 17 tiêu chuẩn kiểm định nghiêm ngặt tại `docs/MODEL_ACQUISITION_GATE.md` trước khi nạp vào web client.

---

## 5. Tiến độ Các Giai đoạn (Phase Status)

| Giai đoạn | Trạng thái | Commit | Nội dung cốt lõi |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Hoàn thành | `625e584` | Khảo sát và đặc tả kiến trúc, kế hoạch nghiên cứu, threat model, license, ADRs 0001-0005. |
| **Phase 1** | Hoàn thành | `f2a7fb4` | Khởi tạo Monorepo (`@forensics/*`, `apps/web`, `ml/`), CI/CD GitHub Actions, Docker. |
| **Phase 2** | Hoàn thành | `a882eee` | Pipeline giám định trình duyệt: magic byte audit, trích xuất EXIF/XMP, DSP (FFT/DCT/Noise/ELA), Web Worker. |
| **Phase 3** | Hoàn thành | `6555b02` | Dataset manifest generator, adapters, perceptual hash dedup, group split chống rò rỉ, augmentation, PyTorch MobileNetV3 backbone, Focal loss, calibration, test pipeline. |
| **Phase 3.5** | Hoàn thành | `cd59136` | Trí nhớ dự án bền vững trong Git (`AGENTS.md`, `PROJECT_STATE.md`, `CODE_MAP.md`, `SESSION_HANDOFF.md`), Truth Audit, Model Acquisition Gate v1, trung thực trạng thái không có model (`uncertain`, `confidence: null`). |
| **Phase 3.6** | Hoàn thành | `e36cdb6` | Evidence Hardening: loại bỏ toàn bộ đường dẫn máy cá nhân, sửa các tuyên bố quá mức, thiết lập Evidence Register và schema manifest máy đọc, củng cố kiểm định Model Registry, phân định smoke test vs scientific training. |
| **Phase 4A.0**| Hoàn thành | `a0e71d5` | Thẩm định và đính chính giấy phép dataset (GenImage, RealHD), thiết lập chính sách Dual-Track (ADR-0006), Dataset Registry máy đọc, Acquisition CLI dry-run, Synthetic smoke fixture và Contamination guards. |
| **Phase 4A.1**| Hoàn thành | *Current* | Đính chính mức độ minh chứng & giấy phép tồn dư (GenImage weights unclear, synthetic-smoke fixture-only), kiểm tra Google Drive GenImage remote inventory, kết luận feasibility B (BigGAN multi-part ~24 GB), đề xuất 3 cấp A/B/C (`docs/GENIMAGE_ACQUISITION_PROPOSAL.md`), `--metadata-only` acquisition CLI, external download lock. |
| **Phase 4A** | Chưa bắt đầu | *Chờ phê duyệt* | Chuẩn bị dữ liệu (Lựa chọn tải BigGAN archive chính thức cho Research Track hoặc giữ nguyên fixture-only). |
| **Phase 4** | Chưa bắt đầu | *Chờ Phase 4A* | Huấn luyện mô hình thật, đo đạc chỉ số F1/ECE thật, cân chỉnh xác suất (Temperature Scaling). |
| **Phase 5** | Chưa bắt đầu | *Chờ Phase 4* | Xuất ONNX từ checkpoint thật, parity test, lượng tử hóa INT8 thật, đo latency WASM/WebGPU thực tế trên trình duyệt. |

---

## 6. Trạng thái Hiện tại của Model và Dataset (Tình trạng Minh chứng)

* **Model Checkpoint**: `none` (chưa có checkpoint huấn luyện hoặc tích hợp; `models/registry.json` ghi nhận `status: not-trained`, `sizeBytes: 0`, `path: ""`).
* **Kiến trúc mô hình**: `architecture candidate` (`ARCH-C2-INHOUSE-MNV3` định nghĩa trong `ml/training/mobilenetv3_forensics.py`).
* **Dung lượng mô hình**: `estimated` (~10.2 MB FP32, ~2.6 MB INT8; tính toán dựa trên số lượng tham số lý thuyết, **chưa đo trên artifact file thật**).
* **Khả năng phân loại 3 lớp & Localization**: `architecture-only` (mã xử lý đã có, nhưng chưa có trọng số mô hình để kiểm chứng khả năng phát hiện hay định vị thực tế; khả năng dự đoán thực tế là `unverified`).
* **ONNX Parity**: `pipeline-only` (chỉ chứng minh pipeline kiểm thử chạy thành công trên mô hình PyTorch chưa huấn luyện trong unit test; chưa thực hiện trên checkpoint thật).
* **Web Runtime & WASM Asset Budget**: `measured build artifact` (tài nguyên engine ONNX Runtime Web WASM `ort-wasm-simd-threaded.jsep.wasm` trong bundle Vite đo được 28.3 MB; đây là engine WebAssembly của runtime, KHÔNG PHẢI kích thước mô hình model weights).
* **Chỉ số khoa học**: `not evaluated` (tuyệt đối không bịa đặt chỉ số accuracy, F1, ECE, AUROC, mIoU).
* **Dataset**: `none` (chưa tải bất kỳ dataset bên ngoài nào; `external dataset bytes downloaded: 0`, `model bytes downloaded: 0`, `training runs: 0`).
* **Synthetic Smoke Fixture**: Track `fixture-only`, purpose `fixture`, license `pending-project-license-decision`, `commercialUse: internal-testing-only`. Dữ liệu giả lập hình học/gradient do code tự sinh trong `ml/tests/fixtures/generated-smoke/`, dùng riêng cho kiểm thử loader/contract, không tham gia huấn luyện sản phẩm hay đo accuracy.
* **GenImage Remote Inventory & Feasibility**: Remote metadata đã kiểm chứng tại Google Drive (`research/evidence/phase-4a.1/genimage-remote-inventory.json`). Kết luận khả thi: **Conclusion B** (phải tải multi-part archive chính thức, không có endpoint tải lẻ từng ảnh; archive nhỏ nhất là BigGAN gồm 8 file `.z01`–`.z07` + `.zip` với tổng dung lượng nén ~24 GB, giải nén ~26 GB, cần tối thiểu 55 GB đĩa trống).
* **Quy chế trọng số GenImage**: `derivativeWeights: unclear` và `productionPromotion: prohibited-by-project-policy`. Giữ toàn bộ checkpoint học từ GenImage trong research track.
* **SAGI-D, RAID, RealHD**: Track `blocked`, `licenseStatus: unverified`, `acquisitionEnabled: false`.

---

## 7. Các Blocker hiện tại

1. **Cần người dùng phê duyệt phương án tải dữ liệu cụ thể trước Phase 4A**: Proposal chi tiết đã lập tại `docs/GENIMAGE_ACQUISITION_PROPOSAL.md`. Người dùng cần xem xét phê duyệt hoặc từ chối tải archive BigGAN (~24 GB) cho Research Track, hoặc tiếp tục dùng `fixture-only`.
2. **Khóa tải ngoài (External Download Lock)**: Lệnh `python -m ml.datasets.acquire --execute` đang bị khóa an toàn với thông báo `Acquisition is prepared. User approval with exact archive and byte size is required.`

---

## 8. Nhiệm vụ tiếp theo

1. Trình báo cáo nghiệm thu Phase 4A.1 cho người dùng kèm proposal 3 cấp (`docs/GENIMAGE_ACQUISITION_PROPOSAL.md`).
2. Dừng lại chờ chỉ thị phê duyệt chính thức của người dùng về dataset, archive, dung lượng byte trước khi kích hoạt bất kỳ lệnh tải mạng nào.
