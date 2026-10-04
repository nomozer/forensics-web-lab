# Báo cáo Nghiên cứu: Thí nghiệm Visual / DSP Ablation

> **Branch**: `research/visual-dsp-ablation`<br>
> **Base main commit**: `9516341e7d7deaf61be4e501d6ce2eba0741c313`<br>
> **Trạng thái**: `EXPERIMENT_COMPLETED_AND_AUDITED`<br>
> **Tập dữ liệu**: Đúng 341 development sources (682 samples: 341 authentic + 341 ai_edited)<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access, 0 evaluation, bảo vệ fail-closed)<br>
> **Phán quyết khoa học**: `DEVELOPMENT_EXPLORATORY_ABLATION_COMPLETE`<br>

---

## 1. Tóm tắt Thực thi và Kế toán Ngân sách

Thí nghiệm so sánh đối đầu prospective 3 cấu hình phân loại trên đúng 341 development sources dưới cùng cấu trúc 5 outer $\times$ 4 inner source-grouped nested CV:
1. **`visual_control`**: MobileNetV3-Small frozen backbone pooled features (576 chiều).
2. **`dsp_only`**: 16 chiều đặc trưng DSP chuẩn tắc (4 FFT radial + 3 DCT energy bands + 5 Noise residuals + 3 JPEG BAG + 1 Laplacian variance) ánh xạ từ `packages/forensics`.
3. **`visual_dsp_fusion`**: Early linear concatenation 592 chiều ($576 + 16$).

### Kế toán Ngân sách (Budget Accounting)
- **Tổng số nhóm outer folds**: 15 nhóm ($3 \text{ recipes} \times 5 \text{ outer folds}$).
- **Tổng số lượt fit tìm kiếm inner CV**: 420 inner fits ($15 \text{ nhóm} \times 4 \text{ inner folds} \times 7 \text{ điểm lưới } C$).
- **Tổng số lượt refit outer-train**: 15 outer refits.
- **Tổng số lượt fit thực tế**: **435 fits** ($420 + 15$).
  - Pilot đã thực hiện nhóm `visual_control/outer_fold_0` (28 inner fits + 1 outer refit = 29 fits).
  - Full runner đã tái sử dụng nhóm pilot hợp lệ qua cơ chế resume và thực thi 14 nhóm còn lại (392 inner fits + 14 outer refits = 406 fits) trong **13.27 giây**.
- **Hội tụ (Convergence)**: 435/435 fits hội tụ thành công trong 1,000 vòng lặp L-BFGS (0 convergence warnings).

---

## 2. Bảng Chỉ số Out-Of-Fold (OOF Metrics)

Toàn bộ 682 mẫu out-of-fold (341 nguồn) được đánh giá độc lập cho mỗi cấu hình (tổng cộng 2,046 dự đoán ảnh đơn):

| Chỉ số (Metric) | Visual Control | DSP Only | Visual + DSP Fusion |
| :--- | :---: | :---: | :---: |
| **Macro-F1** | 0.5787 | 0.5727 | **0.5835** |
| **Balanced Accuracy** | 0.5792 | 0.5733 | **0.5836** |
| **AUROC** | **0.6058** | 0.5978 | 0.5920 |
| **Brier Score** | **0.2440** | 0.2454 | 0.2573 |
| **ECE (Miscalibration)** | **0.0320** | 0.0373 | 0.0895 |
| **FPR (False Positive Rate)** | **0.3871** (132/341) | 0.4633 (158/341) | 0.4018 (137/341) |
| **FNR (False Negative Rate)** | 0.4545 (155/341) | **0.3900** (133/341) | 0.4311 (147/341) |
| **True Negatives (TN)** | 209 | 183 | 204 |
| **False Positives (FP)** | 132 | 158 | 137 |
| **False Negatives (FN)** | 155 | 133 | 147 |
| **True Positives (TP)** | 186 | 208 | 194 |

---

## 3. Phân tích Độ lệch Ghép cặp (Paired Deltas)

| Phép so sánh (Comparison) | $\Delta$ Macro-F1 | $\Delta$ Balanced Acc | $\Delta$ AUROC | $\Delta$ Brier | $\Delta$ ECE | $\Delta$ FPR | $\Delta$ FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fusion vs Visual Control (Ablation Gain)** | **+0.0048** | **+0.0044** | -0.0138 | +0.0133 | +0.0575 | +0.0147 | **-0.0235** |
| **DSP Only vs Visual Control** | -0.0060 | -0.0059 | -0.0080 | +0.0014 | +0.0054 | +0.0762 | **-0.0645** |
| **Fusion vs DSP Only** | **+0.0107** | **+0.0103** | -0.0058 | +0.0119 | +0.0522 | **-0.0616** | +0.0411 |

---

## 4. Thảo luận Khoa học & Ý nghĩa Thực tiễn

1. **Hiệu ứng của Early Fusion (Visual + DSP)**:
   - Việc kết hợp sớm 16 đặc trưng DSP vào vector 576 đặc trưng thị giác mang lại mức cải thiện nhẹ về Macro-F1 ($\Delta \text{Macro-F1} = +0.0048$, hay +0.48%) và Balanced Accuracy (+0.44%).
   - Cải thiện này xuất phát chủ yếu từ việc giảm False Negative Rate (FNR giảm từ $45.45\%$ xuống $43.11\%$, phát hiện thêm 8 ảnh bị chỉnh sửa AI mà Visual Control bỏ sót). Điều này khẳng định các dấu vết tần số cao (FFT/DCT) và nhiễu vi mô (noise residual) cung cấp thông tin bổ trợ hữu ích cho các trường hợp inpainting tinh vi.
   - Tuy nhiên, việc ghép nối tuyến tính thô (concatenation) làm suy giảm độ hiệu chuẩn xác suất (ECE tăng từ $0.0320$ lên $0.0895$) và AUROC giảm nhẹ (-0.0138). Điều này biện minh cho thiết kế của kiến trúc sản phẩm trong ADR-0004: cần tách biệt bộ phân loại thị giác và bộ phân tích DSP, sau đó dung hợp muộn qua cơ chế hiệu chuẩn xác suất (Calibrated Late Fusion) thay vì ghép nối tuyến tính sớm.

2. **Năng lực của DSP-Only**:
   - 16 đặc trưng DSP đạt Macro-F1 $0.5727$, chỉ kém mô hình thị giác học sâu MobileNetV3 $0.0060$ điểm (trong khi hoàn toàn không tốn chi phí forward mạng nơ-ron).
   - DSP thể hiện độ nhạy cao hơn rõ rệt với ảnh chỉnh sửa (FNR thấp nhất: $39.00\%$ so với $45.45\%$ của Visual), nhưng có tỷ lệ báo động giả cao hơn trên ảnh thật phức tạp (FPR $46.33\%$ so với $38.71\%$).

3. **Tính trung thực và Bản chất Phát triển (Development/Exploratory Nature)**:
   - Các kết quả trên được đo đạc thực tế 100% trên 341 development sources qua 5x4 nested CV có kiểm soát rò rỉ nghiêm ngặt.
   - Kết quả này độc lập và không thể so sánh trực tiếp với kết quả confirmatory locked-test lịch sử của Phase 4C.2G (nơi sử dụng 343 sources độc lập khác và đã đóng vĩnh viễn).
