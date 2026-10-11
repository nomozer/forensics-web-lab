# Protocol Specification v1.0.0: Lightweight Tampering Localization Preparation

> **Protocol ID**: `PROTO-LOCALIZATION-PREPARATION-V1.0.0`<br>
> **Date**: 2026-10-11<br>
> **Status**: `LOCKED_LOCALIZATION_SPECIFICATION_PREPARATION`<br>
> **Model Status**: `NOT_RUN / NOT_MEASURED` (Không tải hoặc huấn luyện mô hình trong lượt chuẩn bị này)<br>
> **Pre-requisites**: `PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT`

---

## 1. Động Lực Khoa Học & Chuyển Dịch Trọng Tâm Đề Tài

Đề tài nghiên cứu được xác định bất biến:
> *"Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web".*

Qua hai giai đoạn chẩn đoán phân loại 3 lớp (A0/A1 visual, global DSP, local quadrant DSP) và kiểm toán hình học center-crop:
1. **Giới hạn nội tại của phân loại toàn ảnh (image-level classification)**:
   - Vùng chỉnh sửa AI (TGIF SD2 inpainting) có diện tích rất nhỏ (trung vị **6.58%** diện tích canvas; 75% số ảnh có diện tích $< 15\%$).
   - Các biểu diễn toàn thể (Global Average Pooling 576-D và Global DSP 16-D) làm loãng tín hiệu pháp chứng cục bộ vào bối cảnh thật của ảnh COCO.
   - Thử nghiệm đặc trưng ô phần tư (Local Quadrant DSP 48-D) làm tăng chi phí CPU lên gấp ~5.4x (183ms vs 34ms) nhưng không khái quát hóa được sang generator ngoại lai (So-Fake probe Macro-F1 chỉ đạt ~0.35, tương đương ngẫu nhiên).
2. **Hiện tượng mất tín hiệu do hình học tiền xử lý (Center-Crop Issue)**:
   - Kiểm toán 91 ảnh holdout edited (`center_crop_mask_coverage_receipt.json`) phát hiện có tới **36.26%** ảnh bị mất một phần hoặc toàn bộ vùng sửa do phép cắt vuông tâm ($\min(w, h)$).
   - Đặc biệt, mẫu `tgif_sd2_edit_000000415990` bị cắt mất 100% vùng sửa, dẫn đến việc mô hình chỉ quan sát pixel authentic và dự đoán nhãn `authentic` là hoàn toàn tự nhiên theo nội dung quan sát thực tế.
3. **Chuyển dịch trọng tâm sang Định vị Điểm ảnh (Pixel-Level Localization)**:
   - Không thể giải quyết bài toán "định vị" bằng cách phân loại ô phần tư (quadrant classification).
   - Cần xây dựng mô hình phân đoạn điểm ảnh (segmentation / localization) trực tiếp xuất ra bản đồ nhiệt (tampering heatmap / binary mask), cho phép khoanh vùng chính xác ranh giới can thiệp inpainting mà không bị lệ thuộc vào nhãn toàn ảnh.

---

## 2. Kiến Trúc Baseline Đề Xuất & Ngân Sách Tính Toán

Để đảm bảo khả năng chạy trực tiếp trên trình duyệt web (client-side web platform qua ONNX Runtime Web WebGL/WASM) mà không cần máy chủ GPU:

### 2.1. Đề Xuất Kiến Trúc
- **Mô hình**: Lightweight Encoder-Decoder (U-Net mini hoặc MobileNetV3-small Encoder kết hợp với Lightweight Depthwise Separable Decoder).
- **Encoder**: MobileNetV3-small (`IMAGENET1K_V1`), tận dụng các feature maps đa tỉ lệ ($1/4, 1/8, 1/16, 1/32$).
- **Decoder**: Các khối giải chập / nội suy song tuyến (bilinear upsampling) kết hợp tích chập tách biệt theo chiều sâu (depthwise separable convolution) và skip connections để phục hồi chi tiết biên giới inpainting.
- **Output Head**: 1 kênh kích hoạt Sigmoid xuất ra xác suất giả mạo điểm ảnh $p(x, y) \in [0, 1]$.

### 2.2. Ngân Sách Tài Nguyên Khóa Cứng
- **Kích thước file mô hình trên đĩa**: $< 10\text{ MB}$ (định dạng ONNX FP32 hoặc INT8 quantized $< 3\text{ MB}$).
- **Số lượng tham số**: $< 2.5\text{M}$ tham số (parameters).
- **Thời gian xử lý trên Local CPU (Single Thread)**: $< 50\text{ ms}$ cho một ảnh kích thước $256 \times 256$ hoặc $< 120\text{ ms}$ cho $512 \times 512$.
- **Thời gian inference trên trình duyệt (WebGL/WebGPU)**: Mục tiêu $< 35\text{ ms}$ (đáp ứng tương tác thời gian thực).

---

## 3. Khế Ước Hình Học (Geometry Contract) & Tránh Cắt Bỏ

Để khắc phục triệt để hiện tượng mất vùng sửa do center-crop:

### 3.1. Hợp Đồng Biến Đổi Hình Học (Geometric Preprocessing Contract)
1. **Phép biến đổi không cắt bỏ (Non-destructive Aspect-preserving Letterbox)**:
   - Ảnh gốc kích thước tùy ý $(w, h)$ được thu phóng giữ nguyên tỷ lệ (aspect ratio) sao cho cạnh lớn nhất bằng $S = 512$ (hoặc $256$).
   - Đệm viền (padding) đối xứng bằng giá trị trung tính (neutral constant padding = 0 hoặc edge reflection) để tạo canvas vuông $S \times S$.
   - **Cam kết**: 100% diện tích ảnh gốc và 100% vùng can thiệp inpainting được bảo toàn bên trong canvas quan sát, loại bỏ hoàn toàn rủi ro mất vùng sửa như center-crop.
2. **Khế ước ánh xạ ngược tọa độ (Coordinate Remapping Contract)**:
   - Bản đồ nhiệt đầu ra $H \in \mathbb{R}^{S \times S}$ được cắt bỏ phần padding và nội suy ngược về hệ tọa độ gốc $(w, h)$ của ảnh người dùng thông qua ma trận affine tỷ lệ:
     $$x_{\text{orig}} = \frac{x_{\text{canvas}} - x_{\text{pad}}}{s}, \quad y_{\text{orig}} = \frac{y_{\text{canvas}} - y_{\text{pad}}}{s}$$
     với tỷ lệ thu phóng $s = \frac{S}{\max(w, h)}$.

---

## 4. Quản Lý Dữ Liệu, Phân Chia Tập & Rào Chắn Rò Rỉ (Leakage Guards)

### 4.1. Dữ Liệu Huấn Luyện & Đánh Giá
- **Tập dữ liệu**: TGIF-Inpainting dataset gồm các bộ ba `(authentic_image, edited_image, ground_truth_mask)`.
- **Nguồn gốc Authentic**: MS-COCO val2017.
- **Nguồn gốc Edited**: Stable Diffusion 2 Inpainting trên mặt nạ vật thể COCO.
- **Ground-truth mask**: Binary mask PNG độ phân giải gốc, pixel $=255$ đại diện cho vùng inpainting, pixel $=0$ đại diện cho nền thực tế.

### 4.2. Khóa Cứng Phân Chia Tập Dữ Liệu (Split Isolation Contract)
- **Development Train**: $N = 750$ cặp (giữ nguyên phân vùng `development_train` trong `three_class_pilot_splits.json`). Toàn bộ việc huấn luyện encoder-decoder và tuning siêu tham số chỉ được thực hiện trong phạm vi này.
- **Development Validation (Internal)**: Tách 20% từ `development_train` theo source group (đảm bảo source clustering) để tối ưu ngưỡng binarization $\tau$.
- **Development Holdout**: $N = 91$ cặp edited + 91 authentic + 91 generated (giữ nguyên phân vùng `development_holdout`). Chỉ dùng để đánh giá kiểm chứng cuối cùng.
- **Rào chắn tuyệt đối (Sealed Cohort Guard)**:
  - CẤM MỞ hoặc sử dụng `TGIF Sealed Test N=400` và `Locked-Test` cohorts cho bất kỳ bước training, validation, hay threshold selection nào.
  - Vi phạm rào chắn này sẽ hủy bỏ hoàn toàn giá trị khoa học của đề tài.

---

## 5. Kế Hoạch Huấn Luyện & Tối Ưu Hóa Ngưỡng (Training & Threshold Protocol)

### 5.1. Hàm Mất Mát (Loss Formulation)
Do diện tích vùng sửa chiếm tỷ lệ rất nhỏ trên ảnh, áp dụng hàm mất mát kết hợp chống mất cân bằng điểm ảnh:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{SoftDice}}$$
với $\lambda = 1.0$, trong đó:
$$\mathcal{L}_{\text{SoftDice}} = 1 - \frac{2 \sum_{i} p_i y_i + \epsilon}{\sum_i p_i^2 + \sum_i y_i^2 + \epsilon}$$

### 5.2. Lựa Chọn Ngưỡng Binarization ($\tau$)
- Ngưỡng $\tau \in [0.1, 0.9]$ được quét trên tập **Development Validation nội bộ** để tối đa hóa chỉ số Pixel Dice Score trên ảnh edited, đồng thời khống chế False Alarm Area trên ảnh authentic $< 1.0\%$.
- TUYỆT ĐỐI KHÔNG tối ưu hóa ngưỡng trên tập Holdout hoặc Test.

### 5.3. Môi Trường Huấn Luyện Đề Xuất
- **Môi trường**: Do huấn luyện phân đoạn ảnh (segmentation) đòi hỏi lặp lại nhiều epochs với batch size $\ge 8$, quá trình huấn luyện bắt buộc phải thực hiện trên môi trường **COLAB GPU** (hoặc GPU tương đương).
- **Môi trường LOCAL CPU**: Dành riêng cho code verification, inference kiểm chứng, đo đạc latency, và chạy unit tests.

---

## 6. Chỉ Số Đánh Giá Chuẩn Mực (Standardized Localization Metrics)

Sử dụng thư viện chuẩn mực đã được kiểm chứng [`ml/evaluation/localization_evaluator.py`](../../ml/evaluation/localization_evaluator.py):

| Chỉ số | Phạm vi đánh giá | Công thức / Định nghĩa | Mục tiêu tối thiểu |
| :--- | :--- | :--- | :--- |
| **Pixel IoU (Jaccard)** | Ảnh `ai_edited` | $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ | $\ge 0.40$ |
| **Pixel Dice (F1)** | Ảnh `ai_edited` | $\frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | $\ge 0.50$ |
| **Pixel Recall** | Ảnh `ai_edited` | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | $\ge 0.60$ |
| **False Alarm Area ($FPR_{\text{pixel}}$)** | Ảnh `authentic` | $\frac{\text{FP}}{\text{Total Pixels}}$ | $\le 1.5\%$ |
| **Image-level Clean Rate** | Ảnh `authentic` | Tỷ lệ ảnh authentic có $0$ pixel báo động | $\ge 85\%$ |

---

## 7. Trạng Thái Hiện Tại & Cam Kết Khoa Học (Scientific Honesty)

- **Trạng thái mô hình localization**: `MODEL_STATUS: NOT_RUN / NOT_MEASURED`.
- **Hiện trạng kho mã**: Hiện tại kho mã CHƯA có mô hình localization nào được huấn luyện hoặc đóng gói.
- **Không tự ý giả lập**: Tuyệt đối không sinh heatmap giả mạo hay gán nhãn tùy tiện cho dữ liệu nghiên cứu thật.
- **Cam kết lượt tiếp theo**: Triển khai huấn luyện kiến trúc U-Net mini đã đặc tả trên Colab GPU theo đúng các ranh giới và khế ước dữ liệu đã khóa tại văn bản này.
