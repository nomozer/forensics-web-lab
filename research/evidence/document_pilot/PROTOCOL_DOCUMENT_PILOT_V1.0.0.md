# Giao Thức Đánh Giá Thực Nghiệm Tài Liệu Giả Mạo AI và Hình Học Quan Sát (Protocol Document Pilot v1.0.0)

> **Mã định danh**: `PROTOCOL_DOCUMENT_PILOT_V1.0.0`
> **Trạng thái**: `LOCKED_PREREGISTERED`
> **Phiên bản**: `1.0.0`
> **Ngày ban hành**: 2026-10-11
> **Môi trường**: LOCAL CPU, branch `research/independent-cohort-acquisition`
> **Liên kết Registry**: `research/evidence/three_class_preparation/locked_models_registry.json`

---

## 1. Định Vị Đề Tài & Phạm Vi Làm Việc (Scope & Working Definitions)

- **Tên đề tài chính thức**: *“Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”*
- **Định nghĩa làm việc về "Tài liệu"**: Ảnh tài liệu hoặc trang PDF dạng hình ảnh có nội dung bị chỉnh sửa, làm giả bằng AI (hóa đơn, biên lai, biểu mẫu, sao kê, văn bản hành chính, hợp đồng...).
- **Ranh giới nhận thức bắt buộc**:
  - Phạm vi này là **định nghĩa làm việc (working definition)** phục vụ pilot thực nghiệm và cần đối soát lại với đề cương khóa luận; **không tự ý khẳng định đã được hội đồng hoặc giảng viên hướng dẫn phê duyệt**.
  - Không mở rộng sang bài toán phát hiện văn bản do LLM sinh (LLM text detection).

---

## 2. Mô Hình Đánh Giá Đã Khóa (Six Locked Models)

Đánh giá đúng 6 mô hình đã niêm phong mã băm trong `locked_models_registry.json`, **tuyệt đối không huấn luyện lại (zero retraining), không refit, không hiệu chuẩn lại (zero recalibration), và không tinh chỉnh ngưỡng**:

1. `A0-visual` (576-D): Backbone MobileNetV3-small đóng băng, huấn luyện trên A0 ($N=750$, COCO/TGIF/SD1.5).
2. `A0-global` (592-D): Visual 576-D + Global DSP 16-D, huấn luyện trên A0.
3. `A0-local` (640-D): Visual 576-D + Global DSP 16-D + Local Quadrant DSP 48-D, huấn luyện trên A0.
4. `A1-visual` (576-D): Backbone MobileNetV3-small đóng băng, huấn luyện trên A1 ($N=1,000$, bổ sung 250 ImageNet authentic).
5. `A1-global` (592-D): Visual 576-D + Global DSP 16-D, huấn luyện trên A1.
6. `A1-local` (640-D): Visual 576-D + Global DSP 16-D + Local Quadrant DSP 48-D, huấn luyện trên A1.

> **Đính chính kiến trúc**: Toàn bộ 6 mô hình sử dụng chung backbone thị giác `MobileNetV3-small` (576-D GAP). Điểm khác biệt giữa A0 và A1 là phân phối dữ liệu huấn luyện, không phải thay đổi backbone sang EfficientNet-B0 (đính chính dứt điểm sai sót sao chép văn bản lịch sử).

---

## 3. Ba Điều Kiện Hình Học Quan Sát (Observation Geometries P0, P1, P2)

### P0: Canonical Square Center-Crop (Baseline)
- Cắt hình học vuông từ tâm ảnh: $s = \min(W, H)$, tọa độ cắt $x_0 = (W - s)//2$, $y_0 = (H - s)//2$.
- Resize Lanczos về $512 \times 512$ để trích xuất DSP (Global 16-D, Local 48-D).
- Resize bicubic về $224 \times 224$ (chuẩn hóa ImageNet) để trích xuất đặc trưng thị giác 576-D.
- *Hạn chế*: Đối với tài liệu khổ dọc (A4, hóa đơn dài), center-crop loại bỏ đỉnh và đáy tài liệu, có nguy cơ cắt mất hoàn toàn vùng chữ bị làm giả.

### P1: Orientation-Correct Letterbox Canvas (Bảo toàn tỷ lệ)
- Resize bảo toàn tỷ lệ khung hình với cạnh dài nhất bằng 512 pixel: $f = 512 / \max(W, H)$, kích thước mới $w' = \text{round}(W \times f)$, $h' = \text{round}(H \times f)$.
- Dán vào giữa khung canvas trung tính $512 \times 512$ với giá trị đệm cố định 128 (neutral gray RGB): lề $dx = (512 - w')//2$, $dy = (512 - h')//2$.
- Trích xuất 16-D Global DSP và 48-D Local DSP trực tiếp từ canvas $512 \times 512$.
- Resize canvas bicubic về $224 \times 224$ để nạp vào backbone thị giác 576-D.
- *Đặc tính*: Bảo toàn 100% diện tích và vùng sửa của tài liệu, không làm biến dạng hình học chữ viết.

### P2: Global + 4 Fixed Quadrants (Multi-Tile 5 Views)
- Gồm đúng 5 góc nhìn (views) độc lập:
  - **View 0**: Toàn ảnh tài liệu dưới dạng Letterbox Canvas P1.
  - **Views 1–4**: Bốn góc phần tư (quadrants) không chồng lấn cắt trực tiếp từ ảnh gốc:
    - View 1 (Top-Left): $[0:H//2, 0:W//2]$
    - View 2 (Top-Right): $[0:H//2, W//2:W]$
    - View 3 (Bottom-Left): $[H//2:H, 0:W//2]$
    - View 4 (Bottom-Right): $[H//2:H, W//2:W]$
    - Mỗi quadrant được letterbox vào canvas $512 \times 512$ (giá trị đệm 128) và resize $224 \times 224$.
- Trích xuất đặc trưng và tính xác suất đã hiệu chuẩn $[p_A^{(v)}, p_E^{(v)}, p_G^{(v)}]$ cho từng view $v \in \{0, 1, 2, 3, 4\}$.
- **Tổng hợp xác suất**: Lấy trung bình số học không trọng số qua 5 views:
  $$\bar{p}_c = \frac{1}{5} \sum_{v=0}^4 p_c^{(v)} \quad \text{với } c \in \{\text{authentic}, \text{ai\_edited}, \text{fully\_generated}\}$$

---

## 4. Quy Tắc Quyết Định Nhị Phân Tiền Đăng Ký (Decision Rule)

Để đánh giá phát hiện tài liệu có can thiệp AI, chỉ số điểm tin cậy AI nhị phân $s_{\text{AI}}$ được định nghĩa cố định:
$$s_{\text{AI}} = \bar{p}(\text{ai\_edited}) + \bar{p}(\text{fully\_generated})$$
- Quyết định: **`AI_DETECTED`** nếu $s_{\text{AI}} \ge 0.5$, ngược lại là **`AUTHENTIC`**.
- Ngưỡng cố định $\tau = 0.5$, không điều chỉnh ngưỡng post-hoc trên tập pilot.

---

## 5. Chỉ Số Đánh Giá & Thử Nghiệm Thống Kê

1. **Chỉ số phân loại nhị phân**:
   - AUROC (Area under ROC curve dựa trên $s_{\text{AI}}$).
   - AP (Average Precision / PR-AUC).
   - Binary Macro-F1 (tính tại ngưỡng 0.5).
   - Balanced Accuracy.
   - Sensitivity / Recall trên tài liệu edited.
   - Specificity / True Negative Rate trên tài liệu authentic.
   - False Positive AI Rate trên tài liệu authentic ($1 - \text{TNR}$).
2. **Chất lượng xác suất**:
   - Multiclass Brier score.
   - Top-class Expected Calibration Error (ECE, 10 uniform bins).
3. **Độ giữ diện tích vùng can thiệp hình học (Mask Coverage Retention)**:
   - Tỷ lệ diện tích mask gốc nằm trong crop box của P0.
   - Phân loại: Full retention (100%), Minor loss (80-99.9%), Major loss (0.1-79.9%), Complete loss (0.0%).
4. **Kiểm định so sánh ghép cặp (Paired Bootstrap Tests)**:
   - Ước lượng paired delta vs P0: $\Delta(P1 - P0)$ và $\Delta(P2 - P0)$ cho từng model.
   - Resampling: Paired sample bootstrap (10,000 replicates, seed 20261011).
   - Báo cáo 95% Percentile Confidence Interval và Bootstrap Positive Fraction $P(\Delta^* > 0)$.
5. **Đo lường chi phí CPU**:
   - Thời gian suy luận trung bình (ms/ảnh) trên LOCAL CPU cho P0, P1, P2.
