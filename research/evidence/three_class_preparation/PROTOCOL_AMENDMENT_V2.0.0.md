# Protocol Amendment v2.0.0: Three-Class Pilot Generator Specification (SD1.4 → SD1.5)

> **Protocol ID**: `PROTO-RQ1-RQ2-3CLASS-V2`  
> **Supersedes**: `PROTO-RQ1-RQ2-3CLASS-V1` (Draft v1.0.0)  
> **Date**: 2026-10-11  
> **Branch**: `research/independent-cohort-acquisition`  
> **Approval**: Approved for empirical pilot execution on LOCAL under Resource Budget Envelope (2 GiB network / 5 GiB disk)

---

## 1. Bối cảnh & Lý do Sửa đổi (Motivation & Justification)

1. **Điểm chặn dữ liệu của Protocol v1.0.0**:
   - Protocol v1.0.0 đặt mục tiêu tiếp nhận mẫu `fully_generated` từ tập con Stable Diffusion v1.4 của GenImage.
   - Kiểm toán thực địa (`genimage_acquisition_receipt.json`) xác nhận hai rào cản khách quan:
     * Archive phân phối chính thức của GenImage SD1.4 (`imagenet_ai_0419_sdv4.zip`) là tập tin nén đa phần (multipart zip 30 files) có tổng dung lượng **~96.5 GiB**, vượt quá xa ngân sách cho phép ($2\text{ GiB}$ mạng / $5\text{ GiB}$ đĩa).
     * Mirror cộng đồng có cấu trúc bảng Parquet `TheKernel01/Tiny-GenImage` (CC BY-NC-SA 4.0) gồm 7 generators (ADM, BigGAN, GLIDE, Midjourney, SD1.5, VQDM, Wukong), trong đó **thiếu hoàn toàn Generator 5 (SD1.4, 0 mẫu)**.
   - Theo Quy tắc trung thực kỹ thuật (Rule 3 AGENTS.md), hệ thống đã dừng lại và kích hoạt trạng thái `DATA_BLOCKED`.

2. **Cơ sở khoa học của việc sửa đổi sang Stable Diffusion v1.5**:
   - SD1.5 và SD1.4 cùng thuộc họ mô hình Latent Diffusion Models (LDM) của CompVis / RunwayML / Stability AI: cùng sử dụng VQ-f8 autoencoder, cùng cấu trúc UNet với cross-attention text conditioning, và cùng bộ giải mã không gian tiềm ẩn (latent space decoding).
   - Trong mirror `TheKernel01/Tiny-GenImage`, **Generator 6 là Stable Diffusion v1.5** (`sdv5`), có sẵn 125 mẫu/shard. Bằng việc tải thêm shard 2 (435.5 MB), tổng dung lượng tải mạng cộng dồn đạt **1,310,738,385 bytes (~1.22 GiB)**, hoàn toàn nằm trong ngân sách $2\text{ GiB}$ cho phép.
   - Việc chuyển đổi sang SD1.5 giải quyết dứt điểm điểm chặn kỹ thuật mà không vi phạm cam kết tài nguyên và không tạo dữ liệu giả.

---

## 2. Các Điều khoản Sửa đổi Cụ thể (Specific Amendment Terms)

### 2.1. Phân bổ và Định danh Lớp
- **Lớp 0 (`authentic`)**: 341 ảnh phát triển từ MS-COCO val2017 (nguyên bản từ TGIF Option P).
- **Lớp 1 (`ai_edited`)**: 341 ảnh phát triển từ TGIF Stable Diffusion 2 Inpainting (`sd2-sp/validation`).
- **Lớp 2 (`fully_generated`)**: **341 ảnh từ GenImage Stable Diffusion v1.5** (`TheKernel01/Tiny-GenImage`, generator index 6, label 1).
- Tổng kích thước tập phát triển 3 lớp: **1,023 ảnh**.

### 2.2. Kiến trúc & Vector Đặc trưng Giữ nguyên Không Đổi
- Backbone thị giác: **MobileNetV3-small** (đóng băng hoàn toàn ImageNet weights SHA-256 `047dcff4...`).
  - Tiền xử lý: Resize/Crop về 512×512 RGB Lanczos $\to$ Bicubic Keys spline ($a = -0.5$) $\to$ Tensor [3, 224, 224] chuẩn hóa ImageNet.
  - Vector thị giác áp chót (penultimate features): **576 chiều** (`visual_feature_dim = 576`).
- Xử lý tín hiệu số (DSP): **Canonical 16-D DSP** (`dsp_feature_dim = 16`) trích xuất trực tiếp từ ảnh 512×512 (FFT radial energy & decay, DCT 8×8 frequency band energy, spatial Laplacian noise residual, JPEG 8×8 grid inconsistency, Laplacian variance).
- Ghép đặc trưng (Feature Concatenation): **592 chiều** ($576 + 16$).
  - **Lưu ý thuật ngữ quan trọng**: Vector 592-d này là **ghép nối tầng đặc trưng (feature concatenation probe)**, không được gọi nhầm là late fusion. Late fusion chỉ dành cho việc kết hợp ở tầng logits/xác suất.

### 2.3. Hai Mô hình Đối chứng (Baselines)
1. **Baseline 1 (`visual_only_multiclass`)**: Linear probe / Logistic Regression đa lớp trên vector thị giác 576-d.
2. **Baseline 2 (`visual_dsp_multiclass`)**: Linear probe / Logistic Regression đa lớp trên vector ghép nối 592-d.
- Cả hai mô hình đều huấn luyện strictly trên `development_train` (750 mẫu, tỉ lệ 1:1:1 = 250:250:250), fit scaler trên training set.
- Đánh giá trên `development_holdout` (273 mẫu, tỉ lệ 1:1:1 = 91:91:91).
- Nghiêm cấm sử dụng `development_holdout` hoặc `source_control_cohort` để chọn siêu tham số (hyperparameter selection).

### 2.4. Tập Đối chứng Chẩn Đoán Nguồn (Source Control Cohort)
- 341 ảnh authentic ImageNet (`nature`) giữ hoàn toàn độc lập với nhãn `partition: source_control_diagnostic`.
- Tuyệt đối không tham gia fitting scaler, classifier hay calibration.
- Dùng chuyên biệt để chẩn đoán độ chệch phân phối nguồn (COCO vs ImageNet) và đo False Positive AI Rate trên ảnh thông thường từ miền dữ liệu khác.

---

## 3. Giới hạn Diễn giải Khoa học (Scientific Scope & Limitations)

1. Kết quả thực nghiệm của pilot này phản ánh năng lực phân loại trên tập dữ liệu cụ thể (COCO authentic, TGIF SD2 inpainting, Tiny-GenImage SD1.5 fully generated).
2. Không suy diễn thành kết luận tổng quát cho mọi trình tạo ảnh AI hoặc mọi phiên bản Stable Diffusion.
3. Không tuyên bố sẵn sàng đưa mô hình 3 lớp lên bản web demo thương mại chỉ vì pilot hoàn thành; bản web tiếp tục duy trì pipeline nhị phân FP32 đã kiểm chứng parity.
