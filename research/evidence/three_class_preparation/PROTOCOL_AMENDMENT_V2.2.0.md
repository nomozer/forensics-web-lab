# Protocol Amendment v2.2.0: Local Quadrant DSP Feature Ablation

> **Protocol ID**: `PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT`<br>
> **Date**: 2026-10-11<br>
> **Status**: `LOCKED_LOCAL_DSP_ABLATION_AMENDMENT`<br>
> **Supercedes**: Extends `PROTOCOL_AMENDMENT_V2.1.0.md` without modifying historical baseline artifacts.

---

## 1. Cơ Sở Khoa Học & Động Lực Nghiên Cứu

Trong thực nghiệm Pilot và Ablation A0/A1, ma trận nhầm lẫn cho thấy điểm nghẽn chính của mô hình 3 lớp là sự phân biệt giữa `authentic` và `ai_edited`:
- Trên 91 ảnh holdout `ai_edited` (TGIF SD2 inpainting), diện tích can thiệp cục bộ thực tế có trung vị là **6.58%** diện tích canvas (75% số ảnh có diện tích inpainting $< 15\%$, tối thiểu chỉ $0.62\%$).
- Việc sử dụng Global Average Pooling (GAP) cho visual branch (576-D) kết hợp với đặc trưng DSP toàn ảnh (16-D trên 512×512 canvas) có nguy cơ làm loãng tín hiệu pháp chứng cục bộ vào bối cảnh nền thực tế của MS-COCO.

Để kiểm tra giả thuyết: **"Biểu diễn DSP cục bộ tổng hợp có giúp cải thiện khả năng phân biệt authentic / ai_edited mà không làm tăng báo động giả trên control hoặc làm suy giảm nghiêm trọng khả năng phát hiện fully_generated hay không?"**, văn bản này khóa một ablation cục bộ tối giản, không thay đổi backbone hoặc tăng chi phí inference vượt mức cho phép.

---

## 2. Giải Quyết & Làm Rõ Mâu Thuẫn Số Liệu Q75

Trước khi tiến hành ablation mới, kiểm toán hệ thống ($systematic-debugging) đã đối chiếu toàn diện hiện tượng mâu thuẫn giữa hai con số Macro-F1 Q75 ($0.5235$ và $0.6015$):
1. **Truy nguyên nguồn gốc số liệu**:
   - Tệp predictions thô [`three_class_predictions_jpeg_q75.json`](./robustness/three_class_predictions_jpeg_q75.json) và biên nhận máy [`three_class_robustness_audit_receipt.json`](./robustness/three_class_robustness_audit_receipt.json) (được tạo ở commit `0f1b22d`) ghi nhận bit-exact Macro-F1 của Fusion là **$0.6015$** (Visual: **$0.5163$**).
   - Tệp biên nhận ablation [`three_class_ablation_evaluation_receipt.json`](./three_class_ablation_evaluation_receipt.json) (tạo ở commit `344f647`) cũng ghi nhận A0-Fusion Q75 là **$0.6015$** (Visual: **$0.5163$**).
   - Con số $0.5235$ trong Bảng 9.1 của `PHASE_REPORT.md` là lỗi sao chép văn bản thủ công (human transcription error) từ một bản nháp phân tích trước đó, không khớp với biên nhận máy đã ký duyệt.
2. **Kiểm tra tính nhất quán của Pipeline Q75**:
   - Cả hai harness đều áp dụng nén JPEG Q75 (`quality=75, subsampling=2, optimize=False`) trên ảnh gốc trước khi resize về 512×512 Lanczos (`JPEG_Q75_CANONICAL_PRE_RESIZE`). Cả visual và DSP đều nhận cùng canvas pixel.
   - Thử nghiệm đối chứng cho thấy nếu nén JPEG sau khi resize 512×512 (`JPEG_Q75_POST_RESIZE`), Macro-F1 suy giảm xuống $0.4588$, chứng minh độ nhạy cao của các đặc trưng tần số với thứ tự nén.
3. **Quy ước chuẩn mực báo cáo**:
   - Định danh cấu hình nén chính thức là `JPEG_Q75_CANONICAL_PRE_RESIZE` với số liệu đã kiểm chứng là Macro-F1 = **$0.6015$** (A0-Fusion).
   - Endpoint Q75 được đánh dấu là **chỉ số phụ khám phá (exploratory secondary endpoint)**; toàn bộ kết luận khoa học chính của ablation Local DSP được neo trên **ảnh gốc (original canvas)**.

---

## 3. Đặc Tả Kỹ Thuật Đặc Trưng DSP Cục Bộ (Local Quadrant DSP)

### 3.1. Hình Học Quadrant Bất Biến
Từ canvas RGB 512×512 chuẩn hóa, trích xuất 4 ô phần tư (quadrants) 256×256 cố định, không chồng lấn:
- **Q0 (Top-Left)**: Box `(0, 0, 256, 256)`
- **Q1 (Top-Right)**: Box `(256, 0, 512, 256)`
- **Q2 (Bottom-Left)**: Box `(0, 256, 256, 512)`
- **Q3 (Bottom-Right)**: Box `(256, 256, 512, 512)`

### 3.2. Trích Xuất & Tổng Hợp Vector 48-D
1. Áp dụng cùng hàm `extract_dsp_features` canonical 16-D cho từng quadrant mà không sửa đổi thuật toán:
   $$D = [\mathbf{d}_0, \mathbf{d}_1, \mathbf{d}_2, \mathbf{d}_3]^T \in \mathbb{R}^{4 \times 16}$$
2. Tổng hợp theo từng chiều đặc trưng $j \in \{0, \dots, 15\}$:
   - **Mean**: $\mu_j = \frac{1}{4} \sum_{i=0}^3 D_{i, j} \in \mathbb{R}^{16}$
   - **Population Std** ($\text{ddof}=0$): $\sigma_j = \sqrt{\frac{1}{4} \sum_{i=0}^3 (D_{i, j} - \mu_j)^2} \in \mathbb{R}^{16}$
   - **Max**: $M_j = \max_{i=0}^3 D_{i, j} \in \mathbb{R}^{16}$
3. Vector tổng hợp cục bộ:
   $$\mathbf{s}_{\text{local}} = [\mu, \sigma, M] \in \mathbb{R}^{48}$$

### 3.3. Ghép Nối Vector Không Gian Đặc Trưng 640-D
Thứ tự ghép nối vector được khóa chặt chẽ:
$$\mathbf{x}_{640} = [\mathbf{v}_{576}, \mathbf{d}_{\text{global}, 16}, \mathbf{s}_{\text{local}, 48}]$$
- Chiều 0 đến 575: Visual embedding (MobileNetV3 avgpool 576-D)
- Chiều 576 đến 591: Global canonical DSP từ canvas 512×512 (16-D)
- Chiều 592 đến 639: Local quadrant DSP summaries (48-D)

*Lưu ý: Không dùng ground-truth mask, ảnh authentic ghép cặp hoặc nhãn để chọn patch trong quá trình inference. Đây là phân loại đặc trưng cục bộ, chưa phải bài toán localization.*

---

## 4. Thiết Kế Bốn Cấu Hình Đối Sánh

| Mã Cấu Hình | Điều Kiện Dữ Liệu | Không Gian Đặc Trưng | Kích Thước | Trạng Thái Model |
| :--- | :--- | :--- | :---: | :--- |
| **A0-Global** | A0 ($N=750$, weight 1.0) | Visual (576-D) + Global DSP (16-D) | 592-D | Tái sử dụng model reconciled |
| **A0-Local** | A0 ($N=750$, weight 1.0) | Visual (576-D) + Global DSP (16-D) + Local DSP (48-D) | 640-D | Huấn luyện model mới |
| **A1-Global** | A1 ($N=1,000$, weight balanced) | Visual (576-D) + Global DSP (16-D) | 592-D | Tái sử dụng model A1 reconciled |
| **A1-Local** | A1 ($N=1,000$, weight balanced) | Visual (576-D) + Global DSP (16-D) + Local DSP (48-D) | 640-D | Huấn luyện model mới |

### Các Ranh Giới Bất Biến:
1. Giữ nguyên toàn bộ dữ liệu, splits (750 train, 273 holdout, 341 control), sample weights (auth COCO 0.5, auth ImageNet 0.5, edit TGIF 1.0, gen SD1.5 1.0).
2. Không tinh chỉnh backbone MobileNetV3; không sinh thêm ảnh; không tải mạng mới.
3. Khóa $C=1.0$, solver `'lbfgs'`, max_iter=1000.
4. Grouped inner CV 5-fold theo `source_group_id`; `StandardScaler` cô lập fold; hiệu chuẩn nhiệt độ chỉ trên true OOF logits.

---

## 5. Tiêu Trí Đánh Giá & Các Endpoint Nghiên Cứu

1. **Endpoint Chính (Original Canvas)**:
   - Trên Development Holdout ($N=273$): Macro-F1, Balanced Accuracy, per-class P/R/F1, Multiclass Brier, ECE, Confusion Matrix.
   - Trên ImageNet Authentic Control ($N=341$): Tỷ lệ báo động giả False-Positive AI (tách biệt nhầm sang `ai_edited` vs `fully_generated`).
   - Trên 91 cặp COCO holdout: Độ dịch chuyển xác suất theo cặp $\Delta P(\text{ai\_edited})$.
2. **Phân Tích Thống Kê**:
   - Paired source-cluster bootstrap (10,000 replicates, seed `20261011`) cho chênh lệch $\Delta = \text{Local} - \text{Global}$.
3. **Phân Tích Bổ Trợ Sau Dự Đoán (Post-Hoc Diagnosis)**:
   - Phân tầng kết quả `ai_edited` theo diện tích inpainting từ 91 ground-truth masks TGIF (100% coverage đã xác minh) thành 4 nhóm phân vị: $<2.85\%$, $2.85\%-6.58\%$, $6.58\%-14.88\%$, $>14.88\%$.
4. **Đo Lường Chi Phí Tính Toán**:
   - Đo thời gian CPU trích xuất Local DSP trên panel phát triển chuẩn 16 mẫu để xác định latency overhead.
