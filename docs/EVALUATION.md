# Evaluation & Benchmark Protocol: Forensics Web Lab

> **Phiên bản**: Phase 4A.2 (Metric Hierarchy Freeze)  
> **Trạng thái tài liệu**: Đã đóng băng hệ thống chỉ số (*Evaluation Protocol Frozen*)  
> **Nguyên tắc**: Trung thực khoa học (*Scientific Honesty*), toàn bộ chỉ số thực nghiệm tiếp tục mang trạng thái `not evaluated`.

---

## 1. Nguyên tắc Đánh giá và Tính Toàn vẹn Khoa học

1. **Zero Synthetic Metrics**: Tuyệt đối không sinh số liệu giả, không điền số liệu phỏng đoán hay ước lượng chủ quan vào bảng kết quả.
2. **Minh bạch Trạng thái Chưa Đo Lường**: Mọi chỉ số chưa chạy thực tế trên dữ liệu và mô hình thật bắt buộc phải ghi rõ `not evaluated` hoặc `not measured`.
3. **Phân cấp Chỉ số Rõ ràng**: Tách biệt rõ chỉ số chính (primary), chỉ số phụ (secondary), chỉ số hiệu chuẩn, chỉ số định vị và chỉ số hiệu năng trình duyệt.

---

## 2. Hệ thống Chỉ số Toàn diện (Comprehensive Metric Framework)

### 2.1. Nhóm Chỉ số Phân loại (Classification Metrics)

* **Primary Metrics (Chỉ số Quyết định Chính)**:
  1. **Macro-F1**: Trung bình cộng không trọng số F1-score của cả 3 lớp (`authentic`, `fully_generated`, `ai_edited`):
     $$\text{Macro-F1} = \frac{1}{3} \sum_{c \in \{\text{auth}, \text{gen}, \text{edit}\}} \text{F1}_c$$
     *Đây là chỉ số đánh giá tổng thể quan trọng nhất để chống lại hiện tượng lệch lớp (class imbalance).*
  2. **Balanced Accuracy**: Trung bình cộng Recall (Sensitivity) của từng lớp:
     $$\text{Balanced Accuracy} = \frac{1}{3} \sum_{c=1}^3 \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

* **Secondary Metrics (Chỉ số Bổ trợ Chẩn đoán)**:
  1. **Per-Class Precision**: Tỷ lệ mẫu thực sự thuộc lớp $c$ trên tổng số mẫu được dự đoán là lớp $c$.
  2. **Per-Class Recall**: Tỷ lệ phát hiện thành công các mẫu thuộc lớp $c$.
  3. **Per-Class F1**: Điều hòa giữa Precision và Recall của từng lớp riêng biệt.
  4. **AUROC One-vs-Rest**: Diện tích dưới đường cong ROC cho từng lớp đối chiếu với 2 lớp còn lại.
  5. **Confusion Matrix (Ma trận Nhầm lẫn 3x3)**: Minh bạch các trường hợp nhầm lẫn chéo, đặc biệt giữa `authentic` và `ai_edited`.

---

### 2.2. Nhóm Chỉ số Hiệu chuẩn & Độ Bất định (Calibration & Uncertainty Metrics)

* **Expected Calibration Error (ECE)**: Đo lường độ lệch giữa xác suất tin cậy dự báo ($\text{conf}$) và tần suất đúng thực tế ($\text{acc}$) trên $M=10$ khoảng chia:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* **Brier Score**: Trung bình bình phương sai số xác suất dự đoán đa lớp:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^3 (p_{ic} - y_{ic})^2$$
* **Negative Log-Likelihood (NLL)**: Độ mất mát cross-entropy trên tập kiểm thử sau hiệu chuẩn.
* **Coverage**: Tỷ lệ phần trăm các mẫu được hệ thống đưa ra phán quyết tin cậy (không rơi vào trạng thái `uncertain`).
* **Selective Risk (Selective Error Rate)**: Tỷ lệ lỗi tính riêng trên các mẫu được chấp nhận đưa ra kết luận (loại bỏ các mẫu `uncertain`).

---

### 2.3. Nhóm Chỉ số Định vị Vùng Chỉnh sửa (Localization Metrics)
> **Phạm vi áp dụng**: Chỉ áp dụng khi có ground-truth mask nhị phân (như TGIF/TGIF2). Bản đồ nhiệt từ patch scores khi chưa đối chiếu mask được phân loại là *exploratory*.

1. **Mean Intersection over Union (mIoU)**: Tỷ lệ trùng khớp giữa vùng dự đoán nhị phân (theo ngưỡng tối ưu $\tau^*$) và mask thật:
   $$\text{mIoU} = \frac{|\hat{M} \cap M^*|}{|\hat{M} \cup M^*|}$$
2. **Dice Score / F1 Mask**: Đo độ tương đồng không gian giữa mask dự đoán và mask thật ($2|\hat{M} \cap M^*| / (|\hat{M}| + |M^*|)$).
3. **Pixel AUROC**: Đánh giá khả năng xếp hạng xác suất điểm ảnh nghi vấn trên toàn bộ tọa độ pixel của ảnh.

---

### 2.4. Nhóm Chỉ số Độ Bền vững Thực tế (Robustness Metrics)

Đo lường độ sụt giảm hiệu năng ($\Delta \text{Macro-F1}$) trước các tác vụ làm suy giảm chất lượng:
1. **JPEG Compression**: Thử nghiệm tại các mức chất lượng $Q \in \{90, 70, 50\}$.
2. **Spatial Resizing**: Giảm kích thước ảnh $0.75\times, 0.50\times, 0.25\times$ sau đó phóng to lại.
3. **Screenshot**: Mô phỏng tái raster hóa và nén màn hình.
4. **Gaussian Blur**: Làm mờ nhẹ $\sigma \in \{1.0, 2.0\}$.
5. **WebP Compression**: Nén lại bằng định dạng WebP hiện đại.
6. **Unseen Generator**: Đánh giá trên họ mô hình tạo sinh hoàn toàn không có trong tập huấn luyện.
7. **Cross-Dataset Generalization**: Đánh giá trên dataset nguồn khác để phát hiện shortcut learning.

---

### 2.5. Nhóm Chỉ số Hiệu năng Trình duyệt (Browser Runtime Metrics)

1. **Model Size**: Kích thước file nhị phân checkpoint PyTorch `.pt` và ONNX unquantized `.onnx` (bytes).
2. **WASM Transfer Size**: Kích thước file lượng tử hóa `model.quant.onnx` truyền qua mạng (gzip/brotli transfer bytes).
3. **Initialization Time**: Thời gian khởi tạo `ort.InferenceSession` trong Web Worker (ms).
4. **Median Latency**: Thời gian suy luận trung vị trên ảnh chuẩn (ms).
5. **P95 Latency**: Phân vị thứ 95 của độ trễ suy luận trên 50 lần chạy liên tiếp (ms).
6. **Peak Memory**: Bộ nhớ heap tối đa của Web Worker trong chu kỳ phân tích (MB).
7. **FP32 vs INT8 Parity ($L_\infty$)**: Độ lệch kết quả số học tối đa giữa suy luận FP32 PyTorch và INT8 ONNX Runtime ($L_\infty < 10^{-2}$).

---

## 3. Bảng Theo dõi Chỉ số Khoa học (Evaluation Status Table)

| Hạng mục | Chỉ số cụ thể | Ngưỡng mục tiêu nghiên cứu | Trạng thái thực tế |
| :--- | :--- | :--- | :--- |
| **Phân loại 3 lớp** | Clean Macro-F1 | $\ge 0.80$ | `not evaluated` |
| **Phân loại 3 lớp** | Balanced Accuracy | $\ge 0.80$ | `not evaluated` |
| **Khái quát hóa** | Unseen-Generator Macro-F1 | $\ge 0.70$ | `not evaluated` |
| **Bền vững nén** | JPEG ($Q=50$) Macro-F1 | $\ge 0.65$ | `not evaluated` |
| **Hiệu chuẩn** | Expected Calibration Error (ECE) | $\le 0.10$ | `not evaluated` |
| **Hiệu chuẩn** | Brier Score | Tối thiểu hóa | `not evaluated` |
| **Định vị (mục tiêu phụ)**| Inpainting Pixel AUROC | $\ge 0.75$ | `not evaluated` |
| **Định vị (mục tiêu phụ)**| Inpainting mIoU | $\ge 0.45$ | `not evaluated` |
| **Kích thước mô hình** | Lượng tử hóa INT8 ONNX size | $\le 10\text{ MB}$ (backbone) | `not evaluated` |
| **Độ trễ suy luận** | Single-image global inference (WASM CPU)| $\le 250\text{ ms}$ | `not evaluated` |
| **Đỉnh bộ nhớ Web** | Web Worker Peak RAM | $\le 200\text{ MB}$ | `not evaluated` |
