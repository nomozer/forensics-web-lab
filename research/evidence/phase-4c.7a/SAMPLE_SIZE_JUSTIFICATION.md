# Sample Size Justification & Planning Simulation: Phase 4C.7A

> **Planning Class**: `SYNTHETIC PLANNING — NOT REAL PERFORMANCE`<br>
> **Simulation Engine**: PCG64 Monte Carlo Engine (Seed `20261007`)<br>
> **Replicates**: 50 synthetic cohorts per setting × 500 source-cluster bootstrap resamples<br>
> **Significance Level**: Two-sided $\alpha = 0.05$ (Percentile 95% CI; success when CI lower bound $> 0$)<br>
> **Primary Estimand**: $\Delta \overline{\text{Macro-F1}} = \overline{\text{Macro-F1}}_{\text{aug}} - \overline{\text{Macro-F1}}_{\text{vis}}$ (arithmetic mean across 5 outer-fold models, no probability averaging)

---

## 1. Bản Chất Thống Kê & Cơ Sở Lập Kế Hoạch

Báo cáo này thay thế hoàn toàn các công thức z-test xấp xỉ trước đây. Macro-F1 là một hàm phi tuyến tính trên toàn bộ cohort; do đó **không thể chia nhỏ thành F1 riêng của từng source cluster** để tính phương sai độc lập.

Kế hoạch cỡ mẫu này sử dụng **mô phỏng Monte Carlo ghép cặp đầy đủ**:
1. Mỗi nguồn $S_i$ cung cấp 2 ảnh (authentic và ai_edited).
2. Giữ nguyên tương quan trong nguồn ($u_s$), tương quan giữa các mô hình candidate ($k=0..4$), và tương quan giữa hai recipes.
3. Tái lập chính xác quy trình đánh giá: tính Macro-F1 riêng cho 5 models, lấy trung bình số học, và tính 95% CI qua paired source-cluster bootstrap (500 resamples per cohort).

---

## 2. Kết Quả Mô Phỏng Theo Các Kịch Bản Hiệu Ứng

### Kịch bản 1: Giả thuyết Null (Hiệu ứng thực sự $\delta = 0.000$)

| Cỡ mẫu ($N_{\text{pairs}}$) | Tổng số ảnh | Hiệu ứng đo được (Mean $\Delta$) | Tỷ lệ kết luận Improvement (Power) | Số cohort thành công | Wilson 95% CI (Tỷ lệ) | MC Std Error | Mean 95% CI $[L, U]$ | Độ rộng CI ($U - L$) | Margin of Error ($ME$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 200 | +0.0013 | **4.0%** | 2/50 | [1.1%, 13.5%] | ±2.8% | [-0.0174, +0.0202] | 0.0376 | ±0.0188 |
| 200 | 400 | -0.0013 | **2.0%** | 1/50 | [0.4%, 10.5%] | ±2.0% | [-0.0147, +0.0120] | 0.0266 | ±0.0133 |
| 300 | 600 | +0.0016 | **4.0%** | 2/50 | [1.1%, 13.5%] | ±2.8% | [-0.0095, +0.0124] | 0.0219 | ±0.0110 |
| 400 | 800 | -0.0007 | **2.0%** | 1/50 | [0.4%, 10.5%] | ±2.0% | [-0.0101, +0.0088] | 0.0189 | ±0.0095 |
| 500 | 1000 | +0.0005 | **2.0%** | 1/50 | [0.4%, 10.5%] | ±2.0% | [-0.0081, +0.0091] | 0.0172 | ±0.0086 |

### Kịch bản 2: Hiệu ứng Vi mô / Cận biên (Hiệu ứng thực sự $\delta \approx +0.025$)

| Cỡ mẫu ($N_{\text{pairs}}$) | Tổng số ảnh | Hiệu ứng đo được (Mean $\Delta$) | Tỷ lệ kết luận Improvement (Power) | Số cohort thành công | Wilson 95% CI (Tỷ lệ) | MC Std Error | Mean 95% CI $[L, U]$ | Độ rộng CI ($U - L$) | Margin of Error ($ME$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 200 | +0.0279 | **82.0%** | 41/50 | [69.2%, 90.2%] | ±5.4% | [+0.0089, +0.0476] | 0.0386 | ±0.0193 |
| 200 | 400 | +0.0276 | **96.0%** | 48/50 | [86.5%, 98.9%] | ±2.8% | [+0.0140, +0.0418] | 0.0278 | ±0.0139 |
| 300 | 600 | +0.0273 | **98.0%** | 49/50 | [89.5%, 99.6%] | ±2.0% | [+0.0160, +0.0387] | 0.0227 | ±0.0114 |
| 400 | 800 | +0.0282 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0186, +0.0382] | 0.0196 | ±0.0098 |
| 500 | 1000 | +0.0282 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0194, +0.0371] | 0.0177 | ±0.0088 |

### Kịch bản 3: Hiệu ứng Cải thiện Nhỏ (Hiệu ứng thực sự $\delta \approx +0.050$)

| Cỡ mẫu ($N_{\text{pairs}}$) | Tổng số ảnh | Hiệu ứng đo được (Mean $\Delta$) | Tỷ lệ kết luận Improvement (Power) | Số cohort thành công | Wilson 95% CI (Tỷ lệ) | MC Std Error | Mean 95% CI $[L, U]$ | Độ rộng CI ($U - L$) | Margin of Error ($ME$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 200 | +0.0548 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0339, +0.0776] | 0.0437 | ±0.0218 |
| 200 | 400 | +0.0563 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0409, +0.0724] | 0.0315 | ±0.0158 |
| 300 | 600 | +0.0551 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0425, +0.0682] | 0.0257 | ±0.0128 |
| 400 | 800 | +0.0545 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0438, +0.0657] | 0.0218 | ±0.0109 |
| 500 | 1000 | +0.0553 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0454, +0.0654] | 0.0199 | ±0.0100 |

### Kịch bản 4: Hiệu ứng Cải thiện Vừa (Hiệu ứng thực sự $\delta \approx +0.100$)

| Cỡ mẫu ($N_{\text{pairs}}$) | Tổng số ảnh | Hiệu ứng đo được (Mean $\Delta$) | Tỷ lệ kết luận Improvement (Power) | Số cohort thành công | Wilson 95% CI (Tỷ lệ) | MC Std Error | Mean 95% CI $[L, U]$ | Độ rộng CI ($U - L$) | Margin of Error ($ME$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 200 | +0.1072 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0797, +0.1372] | 0.0576 | ±0.0288 |
| 200 | 400 | +0.1084 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0886, +0.1293] | 0.0407 | ±0.0204 |
| 300 | 600 | +0.1084 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0919, +0.1256] | 0.0337 | ±0.0169 |
| 400 | 800 | +0.1073 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0931, +0.1223] | 0.0292 | ±0.0146 |
| 500 | 1000 | +0.1073 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.0945, +0.1205] | 0.0260 | ±0.0130 |

### Kịch bản 5: Hiệu ứng Lớn (Hiệu ứng thực sự $\delta \approx +0.140$)

| Cỡ mẫu ($N_{\text{pairs}}$) | Tổng số ảnh | Hiệu ứng đo được (Mean $\Delta$) | Tỷ lệ kết luận Improvement (Power) | Số cohort thành công | Wilson 95% CI (Tỷ lệ) | MC Std Error | Mean 95% CI $[L, U]$ | Độ rộng CI ($U - L$) | Margin of Error ($ME$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 200 | +0.1552 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.1209, +0.1921] | 0.0713 | ±0.0356 |
| 200 | 400 | +0.1567 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.1317, +0.1826] | 0.0509 | ±0.0254 |
| 300 | 600 | +0.1578 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.1374, +0.1791] | 0.0418 | ±0.0209 |
| 400 | 800 | +0.1546 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.1369, +0.1728] | 0.0359 | ±0.0180 |
| 500 | 1000 | +0.1571 | **100.0%** | 50/50 | [92.9%, 100.0%] | ±0.0% | [+0.1411, +0.1734] | 0.0323 | ±0.0162 |

---

## 3. Nhận Xét & Phân Tích Thống Kê (Từ Kết Quả Máy Sinh)

1. **Kiểm soát Tỷ lệ Báo động Giả (False Positive Rate under Null)**:
   - Dưới kịch bản Null ($\delta = 0.000$), tỷ lệ các cohort có 95% CI lower $> 0$ dao động trong khoảng $2.0\% - 4.0\%$:
     * Tại $N=100$: 4.0% (2/50 cohorts, Wilson 95% CI: [1.1%, 13.5%]).
     * Tại $N=200$: 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]).
     * Tại $N=300$: 4.0% (2/50 cohorts, Wilson 95% CI: [1.1%, 13.5%]).
     * Tại $N=400$: 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]).
     * Tại $N=500$: 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]).
   - *Đánh giá trung thực*: Với quy mô mô phỏng $M=50$ cohorts, tỷ lệ thực tế $2.0\% - 4.0\%$ tương thích với dao động ngẫu nhiên xung quanh mức danh định $\alpha/2 = 2.5\%$. Không tuyên bố sai lệch rằng tỷ lệ null rejection luôn luôn $\le 2.0\%$ tại mọi cỡ mẫu.
2. **Độ Rộng Khoảng Tin Cậy & Margin of Error (Precision)**:
   - Tại $N=100$: Độ rộng CI trung bình 0.0376 (Null) / 0.0386 (Subtle) $\implies ME \approx \pm 0.0193$.
   - Tại $N=200$: Độ rộng CI trung bình 0.0266 (Null) / 0.0278 (Subtle) $\implies ME \approx \pm 0.0139$.
   - Tại $N=300$: Độ rộng CI trung bình 0.0219 (Null) / 0.0227 (Subtle) $\implies ME \approx \pm 0.0114$.
   - Tại $N=400$: Độ rộng CI trung bình 0.0189 (Null) / 0.0196 (Subtle) $\implies ME \approx \pm 0.0098$ ($ME < 0.010$).
   - Tại $N=500$: Độ rộng CI trung bình 0.0172 (Null) / 0.0177 (Subtle) $\implies ME \approx \pm 0.0088$.
   - *Kết luận độ chính xác*: Cỡ mẫu $N_{\text{pairs}} \ge 400$ là ngưỡng tối thiểu để margin of error co hẹp dưới mức $\pm 0.010$ điểm Macro-F1.
3. **Công Suất Thống Kê & Giới Hạn Mô Phỏng Monte Carlo**:
   - Dưới hiệu ứng cận biên / vi mô ($\delta \approx +0.028$):
     * $N=100$: 41/50 cohorts đạt ý nghĩa (82.0%, Wilson 95% CI: [69.2%, 90.2%]).
     * $N=200$: 48/50 cohorts (96.0%, Wilson 95% CI: [86.5%, 98.9%]).
     * $N=300$: 49/50 cohorts (98.0%, Wilson 95% CI: [89.5%, 99.6%]).
     * $N=400$: 50/50 cohorts đạt ý nghĩa (ước lượng điểm 100.0%, Wilson 95% CI: [92.9%, 100.0%]).
   - *Giới hạn khoa học*: Kết quả 50/50 cohorts thành công tại $N=400$ là ước lượng điểm trong mô phỏng Monte Carlo $M=50$, KHÔNG tương đương với cam kết công suất 100% trong đời thực. Khoảng tin cậy Wilson $[92.9\%, 100.0\%]$ phản ánh độ không chắc chắn thực tế của phép thử mô phỏng.

---

## 4. Quyết Định Prespecified: Cỡ Mẫu Mục Tiêu ($N_{\text{target}}$) & Quy Tắc Dừng

Dựa trên kết quả mô phỏng Monte Carlo và khả năng thu thập thực tế:

1. **Cỡ Mẫu Mục Tiêu Khóa Chặt (Prespecified Target)**:
   $$\mathbf{N_{\text{target}} = 400 \text{ paired sources}} \quad (\text{tương đương } 800 \text{ ảnh hợp lệ})$$
2. **Hạn Mức Thu Thập & Dự Phòng Sự Cố (Acquisition Buffer)**:
   - Kế hoạch thu thập sẽ lấy **440 nguồn ảnh** (dự phòng $10\%$ hao hụt cho các trường hợp ảnh lỗi không decode được, vi phạm magic bytes, hoặc bị chặn bởi bomb guard).
   - Quá trình sàng lọc hợp lệ tuân thủ schema sẽ lấy đúng **400 sources hợp lệ đầu tiên** theo thứ tự ID ngẫu nhiên đã định sẵn trước khi niêm phong.
3. **Quy Tắc Dừng Thu Thập Tuyệt Đối (Strict Stopping Rule)**:
   - Việc thu thập dừng lại ngay khi đạt đủ $N_{\text{target}} = 400$ cặp ảnh hợp lệ đã khóa.
   - **Tuyệt đối không chạy đánh giá mô hình trong lúc thu thập**; không có quyết định 'thu thập thêm' hay 'dừng sớm' dựa trên điểm số hoặc kết quả sơ bộ.
   - Nếu do giới hạn khách quan chỉ thu thập được ít hơn (ví dụ $N=300$), báo cáo sẽ ghi nhận trung thực phạm vi ước lượng và suy giảm công suất, tuyệt đối không bịa đặt số liệu.
