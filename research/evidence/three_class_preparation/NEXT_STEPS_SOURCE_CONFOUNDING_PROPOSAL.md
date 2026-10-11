# Đề Xuất Thực Nghiệm Tiếp Theo: Kiểm Soát & Giảm Thiểu Source Confounding (RQ1–RQ2)

> **Mã văn bản**: `PROPOSAL-EXP-SOURCE-CONFOUNDING-V1`<br>
> **Tác giả / Module**: Forensics Web Lab Research Track (`research/independent-cohort-acquisition`)<br>
> **Ngày ban hành**: 2026-10-11<br>
> **Mục tiêu**: Xây dựng ma trận thực nghiệm có kiểm soát nhằm phân rã (disentangle) và giảm thiểu sai số chệch nguồn dữ liệu (*source domain confounding*) và bất đối xứng lịch sử nén (*compression provenance confounding*) đã được vạch trần trong pilot ba lớp.

---

## 1. Bối Cảnh & Vấn Đề Khoa Học Đã Vạch Trần

Kết quả thực nghiệm pilot ba lớp và phân tích lỗi post-hoc trên mô hình đóng băng (commit 13896df và đợt tái lập kiểm chứng) ghi nhận:
1. **Tỷ lệ báo động giả AI cực cao trên tập đối chứng nguồn độc lập**:
   - Trên 341 ảnh ImageNet tự nhiên (`nature`), mô hình Visual-only nhầm AI là **67.45%** (230/341), Visual+DSP nhầm **68.04%** (232/341).
   - Tỷ lệ này duy trì bền bỉ trên cả 4 điều kiện biến đổi: `original` (67.45%), `jpeg_q95` (66.86%), `jpeg_q75` (67.16%), và `resize_0.5` (62.17%).
2. **Nguyên nhân cốt lõi**:
   - **Domain Shift Confounding (H1)**: Tập huấn luyện sử dụng ảnh thật từ MS-COCO (phân phối bối cảnh đa vật thể, góc rộng, ảnh đường phố/nội thất), trong khi đối chứng nguồn là ImageNet Nature (đối tượng sinh vật/thiên nhiên đặc thù đơn lẻ). Backbone MobileNetV3 đông lạnh nhạy cảm cao với phân phối ngữ nghĩa/bối cảnh này.
   - **Compression History Confounding (H2)**: Ảnh MS-COCO/TGIF trong pipeline được nạp từ PNG (mặc dù COCO gốc tải từ camera), trong khi ảnh ImageNet mang cấu trúc quantization table JPEG ban đầu của máy ảnh số. Việc re-encode JPEG Q95/Q75 sau đó không xóa được lịch sử nén kép hoặc dấu vết cảm biến nguyên bản.
   - **Localized Inpainting Invisibility (H3)**: Với ảnh `ai_edited` (TGIF SD2 inpainting), diện tích chỉnh sửa chỉ chiếm một phần (1%–50%), phần nền còn lại là 100% COCO authentic. Do mô hình trích xuất đặc trưng toàn cục (global pooling 576-D), đặc trưng nền COCO lấn át đặc trưng inpainting, dẫn đến sự nhầm lẫn lớn giữa `authentic` và `ai_edited` (38 vs 46 trên holdout).

---

## 2. Thiết Kế Ma Trận Thực Nghiệm 2×2 Balanced Cross-Domain

Để phân rã dứt khoát giữa **khả năng nhận diện dấu vết AI** và **học vẹt bối cảnh nguồn (domain shortcut)**, đề xuất triển khai thiết kế nhân tố cân bằng $2 \times 2$:

### Ma Trận Dữ Liệu Đối Xứng

| Nhóm Nguồn | Lớp 0: `authentic` | Lớp 1: `ai_edited` | Lớp 2: `fully_generated` | Vai Trò |
| :--- | :--- | :--- | :--- | :--- |
| **Miền A: MS-COCO** | MS-COCO val2017 ($N=250$) | TGIF SD2 Inpainting trên nền COCO ($N=250$) | SD1.5 sinh từ COCO text prompts ($N=250$) | Cohort COCO-centric |
| **Miền B: ImageNet** | ImageNet val2012 ($N=250$) | SD2 Inpainting trên nền ImageNet ($N=250$) | GenImage SD1.5 trên nhãn ImageNet ($N=250$) | Cohort ImageNet-centric |

### Nguyên Tắc Ghép Cặp & Kiểm Soát Biến

1. **Kiểm soát bối cảnh nội miền (Within-Domain Matching)**:
   - Trong mỗi miền, cả 3 lớp đều có cùng phân phối bối cảnh và từ vựng ngữ nghĩa (ví dụ: Miền B gồm toàn bộ động vật/thực vật cho cả 3 lớp).
   - Mô hình được huấn luyện trên Miền A và kiểm định chéo trên Miền B (và ngược lại).
2. **Đo đạc khoảng cách chuyển miền (Cross-Domain Generalization Gap)**:
   $$\Delta_{\text{domain}} = \text{Macro-F1}_{\text{in-domain}} - \text{Macro-F1}_{\text{cross-domain}}$$
   Nếu một kiến trúc thực sự phát hiện dấu vết AI chứ không dựa vào shortcut nguồn, $\Delta_{\text{domain}}$ phải tiệm cận 0 và False Positive Rate trên ảnh authentic miền lạ phải $\le 15\%$.

---

## 3. Giao Thức Chuẩn Hóa Lịch Sử Nén (Standardized Provenance Pipeline)

Để triệt tiêu giả thuyết H2 (asymmetry về JPEG compression):
1. **Synthetic Uniform Re-Compression Baseline**:
   - Tất cả ảnh nguồn (cả COCO, TGIF, GenImage và ImageNet) sau khi giải mã pixel RGB phải được đưa qua một bước nén chuẩn hóa thống nhất:
     * Cấu hình chuẩn hóa: JPEG Quality $Q \in \{85, 95\}$, fixed chroma subsampling 4:2:0.
   - Thao tác này thiết lập một lưới nén DCT đồng nhất lên mọi ảnh trước khi bước vào pipeline trích xuất DSP và thị giác.
2. **Noise Residual Decoupling**:
   - Trong nhánh DSP 16-D, tách riêng thành phần nhiễu cảm biến cao tần (high-frequency noise residual) và lưới khối JPEG 8x8 thành hai nhóm kiểm soát riêng biệt để đo mức độ nhạy cảm của từng thành phần với nén ảnh.

---

## 4. Tách Biệt Đặc Trưng Cục Bộ vs Toàn Cục (Local vs Global Modeling)

Để giải quyết vấn đề H3 (authentic ↔ ai_edited confusion do nền chi phối):
1. **Dual-Branch Patch Localization**:
   - Bổ sung một nhánh trích xuất đặc trưng theo sliding window patches $64 \times 64$ hoặc $128 \times 128$.
   - Tính toán độ lệch bất đối xứng giữa các patch nội tại của bức ảnh (*intra-image patch inconsistency*).
   - Ảnh `fully_generated` có độ đồng nhất tạo sinh trên toàn bộ canvas; ảnh `authentic` có độ đồng nhất quang học tự nhiên; chỉ có ảnh `ai_edited` mới có sự nhảy vọt thống kê (statistical boundary jump) tại ranh giới inpainting.
2. **Định lượng thay đổi xác suất theo cặp (Pairwise Margin Constraint)**:
   - Áp dụng hàm mất mát xếp hạng theo cặp (Pairwise Ranking Loss) trong huấn luyện probe:
     $$\mathcal{L}_{\text{pair}} = \max\left(0, \gamma - \left[ P(\text{ai\_edited} \mid x_{\text{edit}}) - P(\text{ai\_edited} \mid x_{\text{auth}}) \right]\right)$$
   - Ép mô hình phải tăng xác suất `ai_edited` khi vùng inpainting xuất hiện trên cùng một nền ảnh, thay vì dựa vào nhãn lớp độc lập.

---

## 5. Tiêu Khóa Kiểm Chứng & Ngưỡng Nghiệm Thu (Stopping Rules)

Đề xuất thực nghiệm này chỉ được coi là thành công khi thỏa mãn đồng thời 3 tiêu chí khách quan:
1. **False Positive AI Rate trên tập đối chứng ngoài miền**: Giảm từ mức hiện tại $\sim 67.5\%$ xuống dưới **$15.0\%$**.
2. **Holdout Macro-F1 trong miền**: Duy trì $\ge 0.6500$ (vượt trội có ý nghĩa so với random baseline $0.3333$).
3. **Holdout Paired Shift Positive Fraction**: Tỷ lệ cặp nguồn có $\Delta P(\text{ai\_edited}) > 0$ tăng từ mức hiện tại $59.3\%$ lên $\ge \mathbf{85.0\%}$.
