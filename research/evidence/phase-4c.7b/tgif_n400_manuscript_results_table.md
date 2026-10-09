| Điều Kiện (Condition) | Mô Hình (Recipe) | Macro-F1 (Mean ± Std) | Balanced Acc (Mean ± Std) | AUROC (Mean ± Std) | Brier Score (Mean ± Std) | ECE (Mean ± Std) | $\Delta \text{Macro-F1}$ | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `original` | Visual Calibrated | 0.5611 ± 0.0084 | 0.5613 ± 0.0085 | 0.5858 ± 0.0025 | 0.2456 ± 0.0007 | 0.0246 ± 0.0104 | Ref | — |
| `original` | Late Fusion DSP Aug | 0.5681 ± 0.0076 | 0.5685 ± 0.0076 | 0.5973 ± 0.0027 | 0.2439 ± 0.0007 | 0.0204 ± 0.0063 | +0.0070 | *(không đăng ký)* |
| `jpeg_q95` | Visual Calibrated | 0.5598 ± 0.0090 | 0.5603 ± 0.0092 | 0.5856 ± 0.0025 | 0.2456 ± 0.0007 | 0.0236 ± 0.0076 | Ref | — |
| `jpeg_q95` | Late Fusion DSP Aug | 0.5664 ± 0.0081 | 0.5675 ± 0.0078 | 0.5978 ± 0.0028 | 0.2440 ± 0.0007 | 0.0233 ± 0.0067 | +0.0066 | *(không đăng ký)* |
| **`jpeg_q75` (Primary)** | **Visual Calibrated** | **0.5624 ± 0.0060** | **0.5628 ± 0.0064** | **0.5851 ± 0.0025** | **0.2456 ± 0.0007** | **0.0206 ± 0.0099** | **Ref** | **—** |
| **`jpeg_q75` (Primary)** | **Late Fusion DSP Aug** | **0.5596 ± 0.0126** | **0.5647 ± 0.0098** | **0.5983 ± 0.0029** | **0.2443 ± 0.0006** | **0.0354 ± 0.0053** | **-0.0027** | **[-0.0126, +0.0072]** |
| `jpeg_q50` | Visual Calibrated | 0.5580 ± 0.0033 | 0.5585 ± 0.0035 | 0.5841 ± 0.0028 | 0.2458 ± 0.0007 | 0.0221 ± 0.0117 | Ref | — |
| `jpeg_q50` | Late Fusion DSP Aug | 0.5485 ± 0.0098 | 0.5613 ± 0.0042 | 0.5971 ± 0.0030 | 0.2452 ± 0.0005 | 0.0477 ± 0.0093 | -0.0095 | *(không đăng ký)* |
| `resize_0.5` | Visual Calibrated | 0.5586 ± 0.0065 | 0.5588 ± 0.0066 | 0.5864 ± 0.0028 | 0.2453 ± 0.0007 | 0.0242 ± 0.0096 | Ref | — |
| `resize_0.5` | Late Fusion DSP Aug | 0.5657 ± 0.0049 | 0.5690 ± 0.0046 | 0.5976 ± 0.0026 | 0.2441 ± 0.0005 | 0.0263 ± 0.0052 | +0.0071 | *(không đăng ký)* |
| `resize_0.5_jpeg_q75` | Visual Calibrated | 0.5510 ± 0.0059 | 0.5523 ± 0.0058 | 0.5814 ± 0.0030 | 0.2458 ± 0.0006 | 0.0275 ± 0.0146 | Ref | — |
| `resize_0.5_jpeg_q75` | Late Fusion DSP Aug | 0.5375 ± 0.0124 | 0.5517 ± 0.0073 | 0.5935 ± 0.0039 | 0.2456 ± 0.0006 | 0.0484 ± 0.0095 | -0.0135 | *(không đăng ký)* |

*Ghi chú kỹ thuật & quy ước thống kê*:
1. **Độ lệch chuẩn qua 5 fold**: Áp dụng nhất quán độ lệch chuẩn mẫu ($N-1=4$, `ddof=1`) cho toàn bộ các cột Mean ± Std.
2. **Expected Calibration Error (ECE)**: Tính toán phân loại nhị phân trên 10 khoảng đều (uniform bins $[0.0, 1.0]$) theo định nghĩa tiêu chuẩn: $\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
3. **Phân định Endpoint sơ cấp và Phân tích phụ**: Khoảng tin cậy 95% Percentile Bootstrap chỉ áp dụng và tiền đăng ký duy nhất cho endpoint sơ cấp tại `jpeg_q75` (Stratified Paired Source Cluster Bootstrap, 10.000 replicates, PCG64 seed `20261007`, phân tầng chính xác 14 Large / 221 Medium / 165 Small). 5 điều kiện còn lại là phân tích phụ khám phá, chỉ báo cáo độ chênh lệch điểm ước lượng $\Delta$, không gán khoảng tin cậy chưa được tiền đăng ký.
4. **Dung sai số học**: Toàn bộ chỉ số tái tính toán từ dữ liệu dự đoán chi tiết (`predictions.json`) khớp với biên nhận thực thi (`receipt.json`) trong dung sai số học dấu phẩy động $\le 1.11 \times 10^{-16}$.
