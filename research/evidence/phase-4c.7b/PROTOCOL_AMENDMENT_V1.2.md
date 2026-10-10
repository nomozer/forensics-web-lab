# Protocol Amendment Version 1.2: Phase 4C.7B

> **Document Class**: Protocol Amendment<br>
> **Predecessor Version**: Protocol Amendment v1.1.0 (`research/evidence/phase-4c.7a/PROTOCOL_AMENDMENT_V1.1.md`)<br>
> **Amendment Version**: v1.2.0<br>
> **Date (UTC)**: 2026-10-07T01:45:00Z<br>
> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Status**: `AMENDMENT_LOCKED_PRE_ACQUISITION`<br>
> **Author**: Research Team & Independent Evaluator Gatekeeper

---

## 1. Cơ Sở Điều Chỉnh và Thỏa Thuận Nghiên Cứu

Giai đoạn chuẩn bị Phase 4C.7A đã xác lập các điều kiện tiên quyết cần thiết cho việc thu thập cohort độc lập, nhưng kết luận bằng trạng thái `BLOCKED_WITH_EXACT_ACQUISITION_REQUIREMENTS` do hai nút thắt nguồn lực khách quan:
1. Thiếu 160 ảnh chụp thực địa độc quyền từ người dùng (Research Field Collection).
2. Thiếu tài khoản bản quyền, phiên bản ổn định và API thực thi khả dụng cho Adobe Firefly / Photoshop Generative Fill.

Người dùng và nhóm nghiên cứu đã chính thức đồng ý điều chỉnh kế hoạch thu thập theo nguồn lực thực tế:
- Thay thế 160 ảnh tự chụp bằng ảnh chụp đời thực công khai có provenance và giấy phép rõ ràng.
- Thay thế Adobe Firefly bằng công cụ inpainting mã nguồn mở (open-weights) khả dụng, minh bạch và có thể tái lập hoàn toàn bằng mã nguồn.

Phụ lục này thiết lập Giao thức Sửa đổi Version 1.2.0 (Protocol Amendment v1.2) trước khi tiến hành bất kỳ hoạt động thu thập hay tạo sinh nào của Phase 4C.7B.

---

## 2. Các Ràng Buộc Bất Biến Giữ Nguyên Tuyệt Đối (Immutable Scientific Invariants)

Mọi nguyên tắc cốt lõi của bài toán kiểm định độc lập được bảo toàn tuyệt đối, không thay đổi:
1. **Bài toán phân loại nhị phân**: `authentic` = 0 vs. `ai_edited` = 1.
2. **Cỡ mẫu mục tiêu khóa chặt**: $\mathbf{N_{\text{target}} = 400 \text{ paired sources}}$ (800 ảnh hợp lệ: 400 authentic, 400 ai_edited).
3. **Hai candidate recipes đối đầu**:
   - `visual_calibrated`: MobileNetV3-Small linear probe + Temperature Scaling (đối chứng visual).
   - `late_fusion_dsp_augmented`: Multi-variant DSP augmentation + Temperature Scaling + Stacker (Phase 4C.6B).
4. **Năm candidate models đóng băng**: Cả 5 outer-fold models giữ nguyên 100% trọng số, siêu tham số và mã băm SHA-256 đã kiểm toán trong `model_bindings_audit.json` (sizes: 58418, 58030, 58465, 58455, 58390 bytes).
5. **Sáu điều kiện kiểm thử chuẩn tắc**: `original`, `jpeg_q95`, `jpeg_q75`, `jpeg_q50`, `resize_0.5`, `resize_0.5_jpeg_q75`.
6. **Primary Estimand**: $\Delta \overline{\text{Macro-F1}} = \overline{\text{Macro-F1}}_{\text{augmented}} - \overline{\text{Macro-F1}}_{\text{visual}}$ tại `jpeg_q75`, tính bằng **trung bình số học của 5 models riêng biệt**; nghiêm cấm việc lấy trung bình xác suất trước khi phân ngưỡng (no probability averaging).
7. **Paired source-cluster bootstrap**: 10,000 resamples, RNG seed PCG64 `20261007`, 95% Percentile CI; tiêu chí thành công là CI lower bound $> 0$.
8. **Nguyên tắc cô lập mô hình (Detector Isolation Invariant)**: Tuyệt đối không gọi mô hình detector, không dự đoán xác suất, và không sử dụng điểm số detector trong suốt quá trình tuyển mẫu, tải ảnh, tạo sinh, hoặc kiểm định chất lượng (QC).

---

## 3. Nội Dung Điều Chỉnh Kế Hoạch Thu Thập (Amended Acquisition Plan)

### 3.1. Nguồn Ảnh Authentic Công Khai Có Provenance
Bỏ yêu cầu ảnh người dùng tự chụp. Lựa chọn 2 nguồn ảnh chụp công cộng có giấy phép cho phép sử dụng và tạo tác phẩm phái sinh trong nghiên cứu:

1. **COCO 2017 Dataset (Phân vùng sạch ngoài Option P) — 50% (200 pairs)**:
   - *Phạm vi tuyển mẫu*: Các ảnh từ `val2017` và `test2017` hoàn toàn không nằm trong 684 source IDs của Option P lịch sử.
   - *Thẩm tra giấy phép (License Audit)*: Không mặc định toàn bộ ảnh COCO đều có giấy phép CC-BY 4.0. Theo tài liệu chính thức của COCO Consortium, các chú thích thuộc bản quyền CC-BY 4.0, nhưng bản thân ảnh gốc được thu thập từ Flickr và chịu sự điều chỉnh của giấy phép Flickr do từng tác giả gốc thiết lập (thường là CC-BY 2.0, CC-BY-SA 2.0, CC0 hoặc Flickr Commons). Quy trình thu thập bắt buộc ghi nhận: `flickr_id`, `author_id`, `flickr_url`, và `license_name` cho từng ảnh.
2. **Unsplash Verified Open Collection (Chụp/Công bố trước năm 2022) — 50% (200 pairs)**:
   - *Phạm vi tuyển mẫu*: Ảnh chụp phong cảnh, đời sống, đồ vật, con người tuyển chọn từ Unsplash có thời điểm công bố trước năm 2022 (trước thời kỳ bùng nổ của các mô hình text-to-image thương mại).
   - *Thẩm tra giấy phép*: Unsplash License cho phép tải về, chỉnh sửa và sử dụng miễn phí cho mục đích phi thương mại và nghiên cứu khoa học. Bắt buộc lưu trữ: `unsplash_photo_id`, `author_name`, `author_url`, `published_at`.
   - *Lưu ý khoa học quan trọng*: **Không xem thời điểm công bố trước 2022 hoặc metadata EXIF là bằng chứng tuyệt đối rằng ảnh chưa từng qua chỉnh sửa kỹ thuật số**. Đây chỉ là tiêu chí sàng lọc hợp lý để giảm thiểu rủi ro can thiệp của AI tạo sinh hiện đại.

### 3.2. Công Cụ Inpainting Open-Weights Khả Dụng
Bỏ công cụ thương mại đóng mã nguồn Adobe Firefly. Lựa chọn 2 kiến trúc inpainting diffusion mã nguồn mở tiêu biểu:

1. **Stable Diffusion 2 Inpainting — 50% (200 pairs)**:
   - Checkpoint: `stabilityai/stable-diffusion-2-inpainting` (512-base-ema).
   - Kiến trúc: Latent Diffusion Model (UNet 865M tham số + OpenCLIP ViT-H/14 text encoder).
   - Giấy phép: CreativeML OpenRAIL-M (cho phép sử dụng nghiên cứu khoa học, quy định các điều kiện đạo đức sử dụng AI).
   - Thiết lập tham số: Resolution $512 \times 512$, scheduler DDIM, 50 inference steps, guidance scale 7.5, fixed PRNG seed sequence.
2. **SDXL Inpainting 1.0 — 50% (200 pairs)**:
   - Checkpoint: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (hoặc `stabilityai/stable-diffusion-xl-base-1.0` inpainting pipeline).
   - Kiến trúc: 2.6B tham số dual text encoder (OpenCLIP ViT-bigG + CLIP ViT-L), cross-attention conditioning cải tiến.
   - Giấy phép: CreativeML OpenRAIL++ (cho phép nghiên cứu học thuật).
   - Thiết lập tham số: Scheduler EulerDiscreteScheduler, 30 inference steps, guidance scale 7.5, fixed PRNG seed sequence, resolution native sau đó chuẩn hóa về $512 \times 512$.

### 3.3. Ma Trận Phân Bổ Cân Bằng Trực Giao $2 \times 2$ (Orthogonal Balanced Allocation)
Việc ghép cặp giữa nguồn ảnh authentic và công cụ inpainting được khóa chặt theo ma trận cân bằng đối xứng:

| Nguồn Ảnh Authentic | SD 2.0 Inpainting (50%) | SDXL Inpainting (50%) | Tổng Cặp Theo Nguồn |
| :--- | :---: | :---: | :---: |
| **COCO 2017 Clean (50%)** | 100 pairs | 100 pairs | **200 pairs (50%)** |
| **Unsplash Verified (50%)** | 100 pairs | 100 pairs | **200 pairs (50%)** |
| **Tổng Cặp Theo Công Cụ** | **200 pairs (50%)** | **200 pairs (50%)** | **400 pairs (100%)** |

- **Buffer Pool**: 440 pairs total = 110 pairs cho mỗi ô quota (ô 1: COCO-SD2, ô 2: COCO-SDXL, ô 3: Unsplash-SD2, ô 4: Unsplash-SDXL).
- **Quy tắc thay thế mẫu lỗi theo Strata (Stratum-Preserving Replacement Rule)**: Nếu một candidate trong ô $(S, T)$ bị loại do vi phạm tiêu chí kỹ thuật hoặc QC, nó chỉ được phép thay thế bằng candidate tiếp theo trong **cùng ô quota $(S, T)$** theo danh sách ứng viên đã khóa ngẫu nhiên từ trước. Tuyệt đối không chuyển dịch quota giữa các ô để bù đắp mẫu lỗi.
- **Làm rõ về mặt khoa học**: Bỏ cách diễn đạt "triệt tiêu mọi confounding". Ma trận phân bổ trực giao chỉ **kiểm soát các yếu tố đã thiết kế** (giữa 2 nguồn ảnh và 2 công cụ inpainting), không thể triệt tiêu hoàn toàn các biến số nhiễu tiềm ẩn (unobserved confounding) vốn có trong phân bố dữ liệu tự nhiên.

### 3.4. Giữ Nguyên Quota Thao Tác Chỉnh Sửa và Diện Tích Mask
- **Loại thao tác chỉnh sửa**:
  * Thay thế vật thể (Object Replacement): 40% (160 pairs = 40 pairs/ô).
  * Xóa bỏ vật thể và tái tạo nền (Object Removal & Infill): 30% (120 pairs = 30 pairs/ô).
  * Chèn thêm vật thể mới (Object Insertion): 30% (120 pairs = 30 pairs/ô).
- **Diện tích vùng mask**:
  * Nhỏ ($< 10\%$ diện tích canvas): 30% (120 pairs).
  * Vừa ($10\% - 30\%$ diện tích canvas): 40% (160 pairs).
  * Lớn ($> 30\%$ diện tích canvas): 30% (120 pairs).
- **Ngữ nghĩa của Mask**: Binary mask PNG $512 \times 512$ (pixel 0 = vùng nền giữ nguyên, pixel 255 = vùng yêu cầu inpainting). Cần làm rõ: Đây là **mask yêu cầu chỉnh sửa (inpainting request mask)** cung cấp cho pipeline diffusion, không mặc nhiên đồng nhất tuyệt đối với việc mọi pixel bên ngoài mask đều bất biến 100% ở mức sai số lượng tử hóa latent/VAE.

---

## 4. Kiểm Soát Quy Trình QC và Trạng Thái Khóa

1. **Technical QC (Kiểm tra Kỹ thuật)**:
   - Decode PIL không lỗi, đúng $512 \times 512$ RGB 8-bit.
   - Đúng binary mask PNG $512 \times 512$ với pixel $\in \{0, 255\}$.
   - Vượt qua kiểm tra decompression bomb ($\le 50,000,000$ pixels).
   - Không phải ảnh đơn sắc (solid black/white/blank) hoặc NaN/Inf values.
   - Vượt qua 4 tầng Source-Disjoint Guard đối chiếu với 684 historical Option P sources (source_id, origin_id, raw_sha256, master_sha256).
2. **Content QC (Kiểm tra Nội dung Chỉnh sửa)**:
   - Đánh giá tính hợp lý và sự hiện diện của thao tác chỉnh sửa trên ảnh.
   - Nếu chưa có cơ chế thẩm định nội dung con người hoàn tất, trạng thái cohort được đánh dấu là `PENDING_CONTENT_QC`; tuyệt đối không công bố cohort hoàn tất chỉ dựa trên việc ảnh giải mã được về mặt kỹ thuật.
3. **Quy Tắc Dừng và Đóng Băng**:
   - Dừng ngay khi đạt đủ 400 valid pairs (100 pairs cho mỗi ô quota).
   - Tính SHA-256 toàn bộ các file ảnh, mask và manifest; khóa niêm phong trước khi chuyển giao cho khâu đánh giá.

---

## 5. Giới Hạn Kết Luận Nghiên Cứu

Giao thức Amendment v1.2 ghi nhận rành mạch các giới hạn khoa học phát sinh từ việc điều chỉnh:
- Kết quả kiểm định độc lập của Phase 4C.7B sẽ phản ánh năng lực phân loại trên các công cụ inpainting mã nguồn mở hiện đại (Stable Diffusion 2 và SDXL).
- Kết quả **không tự động suy rộng ra các công cụ inpainting thương mại độc quyền (như Adobe Firefly / Midjourney)** do các công cụ này đã được đưa ra khỏi phạm vi kiểm định thực tế của cohort.
