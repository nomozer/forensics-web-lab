# Sample Size Justification: Independent Validation Cohort

> **Phase**: 4C.7A — Independent Validation Preparation<br>
> **Methodological Class**: `SYNTHETIC_PLANNING_SIMULATION`<br>
> **Status**: `PRESPECIFIED_SAMPLE_SIZE_RATIONALE`<br>
> **Primary Endpoint**: $\Delta \overline{\text{Macro-F1}}_{\text{jpeg\_q75}} = \overline{\text{Macro-F1}}_{\text{augmented}} - \overline{\text{Macro-F1}}_{\text{visual}}$ trên paired source-cluster bootstrap 95% CI.

---

> [!IMPORTANT]
> Toàn bộ các ước lượng phương sai, hiệu ứng kỳ vọng và phân tích power dưới đây là **MÔ PHỎNG LẬP KẾ HOẠCH (SYNTHETIC PLANNING)** dựa trên các thuộc tính quan sát được ở giai đoạn phát triển trước đó. Chúng KHÔNG đại diện cho hiệu năng thực tế của mô hình trên dữ liệu kiểm định độc lập chưa thu thập.

---

## 1. Bản chất Cấu trúc Dữ liệu và Phương pháp Đánh giá

Phép kiểm định độc lập sử dụng cấu trúc **ghép cặp theo nguồn (paired source clusters)**:
- Mỗi đơn vị lấy mẫu ngẫu nhiên (cluster) là một nguồn ảnh $S_i$ cung cấp 2 ảnh: authentic ($A_i$) và ai_edited ($E_i$).
- Khi đánh giá dưới bất kỳ điều kiện transform nào (ví dụ `jpeg_q75`), cả hai mô hình (`visual_calibrated` và `late_fusion_dsp_augmented`) cùng được chấm điểm trên chính xác cùng một tập ảnh.
- Do đó, sai số của hai mô hình trên cùng một nguồn có tương quan dương mạnh ($\rho > 0$), giúp phương sai của hiệu số ghép cặp $\Delta_i$ nhỏ hơn đáng kể so với hai mẫu độc lập:
  $$\sigma^2_{\Delta} = \sigma^2_{\text{aug}} + \sigma^2_{\text{vis}} - 2 \rho \, \sigma_{\text{aug}} \, \sigma_{\text{vis}}$$

---

## 2. Công thức Xác định Cỡ mẫu Thống kê

### 2.1. Kiểm định Giả thuyết Một phía cho Endpoint Chính
Mục tiêu là chứng minh mô hình augmented có hiệu năng cao hơn mô hình visual đối chứng tại điều kiện nén `jpeg_q75`, tức kiểm định:
$$H_0: \delta \le 0 \quad \text{vs.} \quad H_1: \delta > 0$$
Tiêu chí thành công prespecified là **cận dưới của khoảng tin cậy 95% hai phía строго lớn hơn 0** (tương đương kiểm định mức ý nghĩa $\alpha = 0.025$ một phía, $z_{1 - \alpha/2} = 1.96$).

Với statistical power $1 - \beta$ (sử dụng $80\%$ với $z_{1-\beta} = 0.8416$, hoặc $90\%$ với $z_{1-\beta} = 1.2816$), cỡ mẫu cụm nguồn $N_{\text{pairs}}$ cần thiết là:
$$N_{\text{pairs}} \ge \left( \frac{z_{1 - \alpha/2} + z_{1 - \beta}}{\delta / \sigma_{\Delta}} \right)^2$$
trong đó:
- $\delta$: Độ lệch thực sự kỳ vọng giữa hai phương pháp (true effect size $\Delta \overline{\text{Macro-F1}}$).
- $\sigma_{\Delta}$: Độ lệch chuẩn của hiệu số ghép cặp giữa hai mô hình trên từng cụm nguồn.

### 2.2. Kiểm soát Độ rộng Khoảng Tin cậy (Margin of Error)
Ngoài khả năng bác bỏ $H_0$, khoảng tin cậy 95% cần đủ hẹp để cung cấp giá trị thông tin khoa học cao. Nửa độ rộng khoảng tin cậy (Margin of Error - $ME$) xấp xỉ:
$$ME = z_{1 - \alpha/2} \cdot \frac{\sigma_{\Delta}}{\sqrt{N_{\text{pairs}}}} = 1.96 \cdot \frac{\sigma_{\Delta}}{\sqrt{N_{\text{pairs}}}}$$
Để đạt được $ME \le ME_{\text{target}}$, cỡ mẫu cần thỏa mãn:
$$N_{\text{pairs}} \ge \left( \frac{1.96 \cdot \sigma_{\Delta}}{ME_{\text{target}}} \right)^2$$

---

## 3. Kịch bản Phân tích Độ nhạy (Sensitivity Scenarios)

Dựa trên dữ liệu mô phỏng và variance quan sát từ Phase 4C.6B, độ lệch chuẩn ghép cặp $\sigma_{\Delta}$ dao động trong khoảng $[0.12, 0.16]$. Chúng tôi khảo sát các kịch bản với $\sigma_{\Delta} = 0.15$:

| Kịch bản | Hiệu ứng Kỳ vọng ($\delta$) | Tỷ số Tín hiệu / Nhiễu ($\delta / \sigma_{\Delta}$) | $N_{\text{pairs}}$ (Power 80%) | $N_{\text{pairs}}$ (Power 90%) | Margin of Error ($ME$) tại $N=400$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Lạc quan (Optimistic)** | $+0.100$ | $0.667$ | **18** | **24** | $\pm 0.0147$ |
| **B. Trung bình (Moderate)** | $+0.050$ | $0.333$ | **71** | **94** | $\pm 0.0147$ |
| **C. Bảo thủ (Conservative)** | $+0.035$ | $0.233$ | **145** | **193** | $\pm 0.0147$ |
| **D. Tối thiểu (Marginal)** | $+0.025$ | $0.167$ | **282** | **378** | $\pm 0.0147$ |
| **E. Rất nhỏ (Subtle)** | $+0.015$ | $0.100$ | **784** | **1050** | $\pm 0.0147$ |

### 3.1. Phân tích Độ rộng CI theo Cỡ mẫu ($N_{\text{pairs}}$)
Giả định $\sigma_{\Delta} = 0.15$:
- Tại $N_{\text{pairs}} = 100$ (200 ảnh): $ME \approx 1.96 \times 0.15 / 10 = \pm 0.0294$ (Độ rộng CI $\approx 0.059$).
- Tại $N_{\text{pairs}} = 250$ (500 ảnh): $ME \approx 1.96 \times 0.15 / 15.81 = \pm 0.0186$ (Độ rộng CI $\approx 0.037$).
- Tại $N_{\text{pairs}} = 350$ (700 ảnh): $ME \approx 1.96 \times 0.15 / 18.71 = \pm 0.0157$ (Độ rộng CI $\approx 0.031$).
- Tại $N_{\text{pairs}} = 500$ (1000 ảnh): $ME \approx 1.96 \times 0.15 / 22.36 = \pm 0.0131$ (Độ rộng CI $\approx 0.026$).

---

## 4. Kết luận và Khuyến nghị Cỡ mẫu

Chúng tôi kiên quyết bác bỏ việc ấn định cỡ mẫu một cách chủ quan (chẳng hạn tự ý cho rằng "500 pairs là con số mặc định"). Từ các tính toán trên:

1. **Ngưỡng sàn Tối thiểu Tuyệt đối**:
   - $N_{\text{pairs}} \ge 250$ cụm nguồn (500 ảnh) để đảm bảo power $> 80\%$ ngay cả khi hiệu ứng thực tế rơi xuống mức bảo thủ $\delta = +0.035$.
2. **Cỡ mẫu Khuyến nghị Tiêu chuẩn**:
   - **$N_{\text{pairs}} = 350 - 400$ cụm nguồn** (tương đương **700 - 800 ảnh**).
   - Tại mức này:
     * Đạt power $\approx 90\%$ cho hiệu ứng $\delta = +0.025 - +0.035$.
     * Độ rộng CI hẹp $\le \pm 0.015$, cho phép kết luận thống kê dứt khoát về mức độ cải thiện.
     * Có biên dự phòng cho khoảng $5\% - 10\%$ mẫu có thể bị loại do lỗi kỹ thuật (corrupt decode, bomb guard).
3. **Cỡ mẫu Mở rộng Lý tưởng**:
   - **$N_{\text{pairs}} = 500$ cụm nguồn** (1000 ảnh) nếu tài nguyên thu thập cho phép, giúp phát hiện cả những cải thiện vi mô ($\delta \approx +0.020$) với power cao và sai số ước lượng cực nhỏ ($ME \approx \pm 0.013$).

*Ghi chú*: Cỡ mẫu này được phê duyệt trước khi thu thập dữ liệu và bị khóa trong giao thức thực nghiệm.
