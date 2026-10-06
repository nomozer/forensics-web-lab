# Cohort Acquisition Plan: Independent Validation Cohort (Finalized)

> **Phase**: 4C.7A — Independent Validation Preparation<br>
> **Status**: `COHORT_DESIGN_FINALIZED_PRE_ACQUISITION`<br>
> **Task Scope**: Binary classification (`authentic` = 0 vs. `ai_edited` = 1)<br>
> **Prespecified Target Sample Size**: $\mathbf{N_{\text{target}} = 400 \text{ paired sources}}$ ($800$ valid images; acquisition buffer $= 440$ pairs)<br>
> **Three-Class Note**: Evaluation of `fully_generated` (3-class classification, RQ1) remains incomplete within the broader project and is excluded due to absent 3-class data.

---

## 1. Mục Tiêu, Quần Thể Đích (Target Population) và Phạm Vi Kết Luận

### 1.1. Mục Tiêu
Cung cấp một tập dữ liệu kiểm định độc lập, chưa từng tham gia bất kỳ giai đoạn phát triển nào (không nằm trong 341 development sources) và hoàn toàn tách biệt khỏi tập locked-test lịch sử (`SEALED_AND_RETIRED`), nhằm thực hiện phép so sánh khách quan giữa hai candidate recipes:
1. `visual_calibrated`: MobileNetV3-Small linear probe + Temperature Scaling (đối chứng visual).
2. `late_fusion_dsp_augmented`: Multi-variant DSP augmentation + Temperature Scaling + Stacker (Phase 4C.6B).

### 1.2. Quần Thể Đích (Target Population)
- **Đặc điểm ảnh thực tế**: Ảnh chụp phong cảnh, đời sống, đồ vật, kiến trúc và con người trong điều kiện ánh sáng tự nhiên/nhân tạo thông thường.
- **Phạm vi kết luận**: Khẳng định hoặc bác bỏ ưu thế của việc bổ sung đặc trưng DSP (được tăng cường biến thể) so với mô hình thị giác đơn thuần trên ảnh chỉnh sửa inpainting cục bộ ở bài toán nhị phân (`authentic` vs `ai_edited`). Kết quả không tự động suy rộng ra video deepfake, ảnh generative tạo sinh toàn phần (`fully_generated`), hay các tác vụ ngoài bài toán.

---

## 2. Nguồn Ảnh Authentic và Thẩm Tra Giấy Phép (License & Terms Audit)

Tuyệt đối không giả định toàn bộ ảnh trong một dataset đều có cùng giấy phép. Việc thu thập yêu cầu xác minh provenance cho từng ảnh nguồn:

1. **Bộ Ảnh Chụp Thực Nghiệm Độc Quyền (Research Field Collection - 40%, 160 pairs)**:
   - Ảnh do nhóm nghiên cứu tự chụp trực tiếp bằng thiết bị số đa dạng (smartphone iOS/Android, camera DSLR Sony/Canon).
   - *Bản quyền*: Nhóm nghiên cứu nắm toàn quyền sở hữu trí tuệ; cấp phép vĩnh viễn cho mục đích nghiên cứu khoa học mở.
   - *Ưu điểm*: Kiểm soát 100% provenance, cảm biến gốc, và loại trừ hoàn toàn rủi ro AI can thiệp trước đó.
2. **COCO 2017 Dataset (CC-BY 4.0 - 35%, 140 pairs)**:
   - Lấy mẫu từ COCO 2017 `test2017` hoặc các partitions sạch không thuộc Option P.
   - *Giấy phép*: CC-BY 4.0 (Creative Commons Attribution 4.0 International). Yêu cầu trích dẫn tác giả ảnh gốc theo metadata Flickr.
3. **Unsplash Verified Open Collection (25%, 100 pairs)**:
   - Tuyển chọn từ các tác giả uy tín trên Unsplash với ảnh chụp trước năm 2022 (trước kỷ nguyên bùng nổ text-to-image commercial).
   - *Giấy phép*: Unsplash License (miễn phí cho mục đích phi thương mại và nghiên cứu).

---

## 3. Công Cụ Inpainting, Quotas và Phân Bố Thao Tác

Để kiểm định tính bền vững tổng quát, ảnh chỉnh sửa được tạo sinh từ 3 công cụ đại diện theo tỷ lệ quota định sẵn:

| Công cụ Inpainting | Phiên bản / Checkpoint | Môi trường | Quota (% / Số lượng) | Ghi chú Provenance |
| :--- | :--- | :--- | :---: | :--- |
| **Stable Diffusion 2 Inpainting** | `stabilityai/stable-diffusion-2-inpainting` (512-base-ema) | Local GPU / Diffusers | **40% (160 pairs)** | Open-weights, seed cố định, DDIM scheduler, 50 steps |
| **SDXL Inpainting** | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | Local GPU / Diffusers | **40% (160 pairs)** | Open-weights, seed cố định, EulerDiscreteScheduler |
| **Adobe Firefly / Photoshop Generative Fill** | Adobe Firefly Image 3 Model (2024 Release) | Desktop Application | **20% (80 pairs)** | Commercial closed-source, cloud diffusion backend |

### 3.1. Phân Bố Loại Thao Tác Chỉnh Sửa
- **Thay thế vật thể (Object Replacement)**: 40% (160 pairs).
- **Xóa bỏ vật thể và tái tạo nền (Object Removal)**: 30% (120 pairs).
- **Chèn thêm vật thể mới (Object Insertion)**: 30% (120 pairs).

### 3.2. Phân Bố Diện Tích Vùng Mask (Mask Area Coverage)
- **Mask nhỏ ($< 10\%$ diện tích canvas)**: 30% (120 pairs).
- **Mask vừa ($10\% - 30\%$ diện tích canvas)**: 40% (160 pairs).
- **Mask lớn ($> 30\%$ diện tích canvas)**: 30% (120 pairs).
Mỗi ảnh `ai_edited` bắt buộc đi kèm 1 file binary mask PNG tương ứng ghi nhận pixel 0 (giữ nguyên) và 255 (vùng chỉnh sửa).

---

## 4. Chuẩn Hóa Canvas và Chống Shortcut Nhãn (Artifact Parity)

Để ngăn chặn mô hình học các shortcut về nén file, kích thước hoặc codec:
1. **Canvas Resolution Đồng Nhất**:
   - Mọi cặp ảnh (authentic và edited) đều được crop/scale về cùng kích thước canvas cố định: **$512 \times 512$ pixels** (RGB 8-bit).
   - Kích thước pixel giữa authentic và ai_edited của cùng một source phải khớp tuyệt đối ($W_{\text{auth}} = W_{\text{edit}} = 512$, $H_{\text{auth}} = H_{\text{edit}} = 512$).
2. **Định Dạng Master File**:
   - Cả ảnh authentic và edited đều được lưu dưới định dạng **PNG không nén lossy (RGB)** trước khi đưa vào pipeline kiểm định.
   - *Làm rõ*: Điều kiện `original` trong 6 conditions kiểm định đại diện cho ảnh master input đầu vào theo quy trình chuẩn hóa này, không mặc nhiên giả định là ảnh raw chưa nén từ cảm biến.
3. **Quy Trình Xử Lý Độc Lập**:
   - Khi mô hình dự đoán, mỗi ảnh được đưa vào độc lập hoàn toàn. Không bao giờ đưa thông tin cặp, mask, source ID hay nhãn vào vector đặc trưng.

---

## 5. Cơ Chế Source-Disjoint Guard Đa Tầng

Validator [`ml/evaluation/independent_cohort.py`](ml/evaluation/independent_cohort.py) kiểm tra nghiêm ngặt 4 lớp chống ô nhiễm dữ liệu lịch sử:
1. **Lớp 1: Source ID Guard**: Không trùng bất kỳ `source_id` nào trong 684 historical Option P sources.
2. **Lớp 2: Image SHA-256 Guard**: Không trùng mã băm SHA-256 của bất kỳ file ảnh lịch sử nào (chống việc đổi tên file hoặc copy từ tập cũ).
3. **Lớp 3: Origin / Instance ID Guard**: Không trùng URL hoặc instance ID Flickr/COCO lịch sử.
4. **Lớp 4: Resolution & Pairing Integrity**: Mỗi source bắt buộc có đúng 1 authentic và 1 ai_edited, resolution khớp tuyệt đối.

---

## 6. Quy Tắc Dừng và Xử Lý Mẫu Lỗi (Strict Stopping Rule)

1. **Ngân sách Thu Thập**: Thu thập tối đa **440 pairs** (dự phòng 10% hao hụt do lỗi corrupt decode, bomb guard hoặc mask lỗi).
2. **Quy Tắc Dừng Tuyệt Đối**:
   - Quá trình duyệt mẫu theo thứ tự ngẫu nhiên prespecified.
   - **Dừng thu thập ngay khi đạt đủ $N_{\text{target}} = 400$ pairs hợp lệ**.
   - Tuyệt đối cấm đánh giá mô hình trong lúc thu thập; không kéo dài thu thập hoặc dừng sớm dựa trên kết quả trung gian.
   - Khi đã khóa 400 pairs, manifest được tính SHA-256 và niêm phong trước khi chạy bất kỳ phép scoring nào.
