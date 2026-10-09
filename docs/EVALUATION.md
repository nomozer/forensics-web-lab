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

### 2.6. Tiêu chuẩn Đánh giá Giao thức Pilot và Cổng Shortcut (docs/PILOT_PROTOCOL.md)

1. **Pilot A (Authentic vs AI-Edited + Localization)**:
   - Đo đạc trên cặp matched pairs MS-COCO của TGIF.
   - Chỉ số bắt buộc: Macro-F1, Balanced Accuracy, AUROC, mIoU, Dice, Pixel AUROC.
   - Điều kiện: Không rò rỉ `source_id` giữa Train, Val và Test.
2. **Pilot B (Authentic vs Fully-Generated)**:
   - Đo đạc trên ImageNet val vs BigGAN trong GenImage.
   - Chỉ số bắt buộc: Macro-F1, Balanced Accuracy, Cross-Generator F1 Drop.
3. **Pilot C (Three-Class Exploratory) & Metadata-Only Baseline Guard**:
   - Khi hợp nhất 3 lớp từ các nguồn khác nhau, bắt buộc huấn luyện thêm một **Metadata-Only Baseline** (chỉ dùng resolution, aspect ratio, file size, codec để đoán nhãn).
   - Metadata-only baseline là diagnostic baseline, không phải bằng chứng duy nhất để loại trừ shortcut.
   - Tính: $\Delta\text{Macro-F1} = \text{Macro-F1}_{\text{visual}} - \text{Macro-F1}_{\text{metadata}}$.
   - Ước lượng paired stratified bootstrap 95% confidence interval ($1,000$ resamples) trên test set.
   - Báo cáo: Macro-F1 của visual model, Macro-F1 của metadata-only baseline, $\Delta\text{Macro-F1}$, bootstrap 95% CI, balanced accuracy, confusion matrix, sample count từng lớp.
   - **Quy tắc nghiệm thu**: Kết quả chỉ được xem là có bằng chứng visual model vượt metadata baseline khi **cận dưới 95% CI của $\Delta\text{Macro-F1}$ lớn hơn 0** ($CI_{\text{lower}} > 0.0$).
   - Kết luận vẫn phải ghi nhận khả năng tồn tại shortcut khác; Pilot C tiếp tục mang trạng thái `exploratory`.

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

---

## 4. Tiêu chuẩn Nghiệm thu Khoa học với Dữ liệu Nhỏ (Few-Data Scientific Acceptance Protocol)

Chi tiết quy định tại [SMALL_DATA_PROTOCOL.md](docs/SMALL_DATA_PROTOCOL.md). Để kết luận mô hình học máy đạt hiệu quả với dữ liệu nhỏ (*sample-efficient*), bắt buộc phải thỏa mãn:

1. **Khóa đơn vị độc lập là `source_id`**: Mọi bảng kết quả cỡ mẫu phải báo cáo đồng thời `file_count`, `variant_count`, `unique_source_id_count` và `paired_source_id_count`. Tuyệt đối không dùng 31,238 mask files làm cỡ mẫu độc lập.
2. **So sánh bắt buộc với 6 Baseline**:
   * Stratified Dummy Classifier.
   * Metadata-Only Classifier (EXIF/C2PA).
   * DSP-Only Classifier (2D FFT / DCT / Noise).
   * Frozen Visual Backbone (Stage 1).
   * Fine-Tuned Visual Model (Stage 2).
   * Multimodal Fusion Calibrator.
3. **Ý nghĩa Thống kê Cận dưới (Statistically Significant Lower Bound)**:
   * $\text{CI}_{95\%}[\Delta\text{Macro-F1}_{\text{visual} - \text{dummy}}] > 0$.
   * $\text{CI}_{95\%}[\Delta\text{Macro-F1}_{\text{visual} - \text{metadata}}] > 0$.
   * Khoảng tin cậy được tính bằng Paired Stratified Bootstrap với tối thiểu $1,000$ lần tái lấy mẫu trên tập Test cố định.
4. **Độ Ổn định qua các Hạt giống**: Độ lệch chuẩn $\sigma_{\text{Macro-F1}} < 0.03$ trên tối thiểu 3 seeds (exploratory) và 5 seeds (confirmatory).
5. **Độ Đơn điệu của Đường cong Học tập**: Đường cong học tập trên các mức $N \in \{50, 100, 250\}$ thể hiện xu hướng tăng trưởng nhất quán.
6. **Định nghĩa Trạng thái Quyết định**: Nhãn `no_ai_evidence` chỉ biểu thị không phát hiện đủ bằng chứng AI trong phạm vi mô hình và dữ liệu đánh giá; không đồng nghĩa với chứng minh ảnh thật tuyệt đối. Mọi trường hợp độ tin cậy thấp hoặc xung đột tín hiệu bắt buộc chuyển sang `uncertain`.

---

## 5. Kết Quả Kiểm Định Độc Lập Thực Tế (Empirical Independent Evaluation Results — TGIF N=400 Cohort)

> **Căn cứ thực nghiệm**: Thực thi kiểm định độc lập Phase 4C.7B trên tập dữ liệu `TGIF-Train-Clean-Subset` ($N=400$ cặp nguồn MS-COCO chưa từng thấy, 800 ảnh) trên LOCAL CPU theo phê duyệt chính thức ngày `2026-10-09T18:08:30Z`. Kiểm toán số học 100% PASS từ predictions đã lưu (không chạy lại detector).

### 5.1. Bảng Kết Quả Tổng Hợp Qua 6 Điều Kiện

| Điều Kiện (Condition) | Visual Calibrated (Mean ± Std) | Late Fusion DSP Augmented (Mean ± Std) | $\Delta \text{Macro-F1}$ (Aug - Vis) | 95% Bootstrap CI | Kết Luận Thống Kê |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `original` | 0.5611 ± 0.0084 | 0.5681 ± 0.0076 | +0.0070 | [-0.0021, +0.0163] | CI cắt 0.0 |
| `jpeg_q95` | 0.5598 ± 0.0090 | 0.5664 ± 0.0081 | +0.0066 | [-0.0027, +0.0160] | CI cắt 0.0 |
| **`jpeg_q75` (Primary)** | **0.5624 ± 0.0060** | **0.5596 ± 0.0126** | **-0.0027** | **[-0.0126, +0.0072]** | **`INCONCLUSIVE` (CI cắt 0.0)** |
| `jpeg_q50` | 0.5580 ± 0.0033 | 0.5485 ± 0.0098 | -0.0095 | [-0.0199, +0.0010] | CI cắt 0.0 |
| `resize_0.5` | 0.5586 ± 0.0065 | 0.5657 ± 0.0049 | +0.0071 | [-0.0015, +0.0159] | CI cắt 0.0 |
| `resize_0.5_jpeg_q75` | 0.5510 ± 0.0059 | 0.5375 ± 0.0124 | -0.0135 | [-0.0250, -0.0022] | CI âm hoàn toàn |

### 5.2. Kết Luận Khoa Học Cho Endpoint Sơ Cấp tại `jpeg_q75`
- **Chỉ số Sơ cấp**: $\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{augmented}} - \text{Macro-F1}_{\text{visual}} = -0.0027$.
- **Stratified Paired Source Cluster Bootstrap** (10.000 replicates, PCG64 seed `20261007`, phân tầng 14 Large / 221 Medium / 165 Small):
  - 95% Percentile CI: **$[-0.0126, +0.0072]$** (bao trùm giá trị 0.0).
  - Tỷ lệ số lượt bootstrap dương: **$30.29\%$** (không phải xác suất giả thuyết nghiên cứu đúng).
- **Phán Quyết Khoa Học**: **`INDEPENDENT_JPEG75_INCONCLUSIVE`**.
- **Diễn giải Trung thực Khoa học**:
  1. **Chưa chứng minh được sự cải thiện (unproven improvement)** của phương pháp Late Fusion DSP Augmented so với Visual Calibrated trên tập kiểm định độc lập tại điều kiện nén JPEG Q=75. Tuyệt đối không kết luận là "bác bỏ sự cải thiện" hay tuyên bố hai phương pháp tương đương nhau.
  2. **Phân biệt kết quả phát triển vs kiểm định độc lập**: Hiệu quả cải thiện $\Delta = +0.1363$ quan sát được trên tập phát triển (Phase 4C.6B) không tái lập được trên tập kiểm định nguồn mới (Phase 4C.7B).
  3. **Phạm vi kiểm định**: Đây là kiểm định trên các ảnh nguồn mới (unseen sources) trong cùng họ benchmark TGIF (\cite{mareen2024tgif}) và cùng generator SD2, **không phải bằng chứng khái quát ngoài phân phối (OOD)**.


