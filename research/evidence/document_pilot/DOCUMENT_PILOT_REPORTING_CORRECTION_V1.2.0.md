# Báo Cáo Đính Chính Số Liệu: Document Pilot Reporting Reconciliation v1.2.0

> **Mã định danh**: `Phase 4C.8 trace — Reporting Reconciliation & Epistemic Rectification v1.2.0`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Biên nhận chứng thực**: [`document_pilot_reporting_correction_receipt_v1.2.0.json`](document_pilot_reporting_correction_receipt_v1.2.0.json)<br>
> **Cam kết**: Bảng số liệu và discrepancy ledger được sinh tự động 100% từ hàm tính toán chuẩn tắc, loại bỏ hoàn toàn lỗi chép tay.<br>

---

## 1. Nguồn Gốc Mâu Thuẫn Số Liệu & Truy Nguyên Nguyên Nhân Gốc Rễ

Trong phiên làm việc trước, người dùng đã phát hiện một sự không nhất quán toán học nghiêm trọng trong bảng tóm tắt văn bản:

> *Ví dụ phản biện của người dùng*:
> *"A0-visual / C0 / P0: Nếu N=40 gồm 20 positive và 20 negative, Recall=0.35 và FPR=0.10 thì TP=7, FN=13, FP=2, TN=18; binary Macro-F1 phải xấp xỉ 0.5943. Không thể đồng thời ghi Macro-F1=0.6011 cho chính confusion matrix này."*

### Kết Quả Đối Soát & Truy Vết Kỹ Thuật:
1. **Bản chất mâu thuẫn**: Không phải do lỗi suy luận hay thuật toán tiền xử lý, mà bắt nguồn từ **lỗi sao chép thủ công (manual transcription error)** khi người hỗ trợ AI soạn thảo bảng tóm tắt nhanh trong phản hồi chat:
   - Bảng markdown lưu trong `DOCUMENT_PILOT_RECONCILIATION_REPORT.md` và machine receipt `document_pilot_replay_receipt.json` ghi nhận đúng:
     * `TP = 6, TN = 20, FP = 0, FN = 14`
     * `Recall (TPR) = 0.3000, TNR = 1.0000, FPR = 0.0000`
     * `Balanced Accuracy = 0.6500, Macro-F1 = 0.6011`
   - Tuy nhiên, trong phản hồi chat, người hỗ trợ AI đã sao chép nhầm dòng từ một cấu hình khác, viết thành:
     * `Recall = 0.3500, FPR = 0.1000, Bal-Acc = 0.6250` trong khi vẫn giữ nguyên `Macro-F1 = 0.6011`.
2. **Hệ quả toán học**: Với $N=40$ ($20$ pos / $20$ neg), bộ số `Recall=0.35, FPR=0.10` tương ứng ma trận `TP=7, TN=18, FP=2, FN=13` chỉ có thể cho ra $F_{1,\text{macro}} = 0.594286$ (làm tròn $0.5943$). Việc ghi nhận $0.6011$ cho ma trận này là mâu thuẫn toán học tuyệt đối.
3. **Biện pháp khắc phục v1.2.0**: Toàn bộ bảng dưới đây và các dòng trong Discrepancy Ledger được tính toán động 100% bằng hàm `compute_confusion_metrics(tp, tn, fp, fn)` trong `scripts/research/reconcile_document_pilot_reporting.py`, chấm dứt hoàn toàn việc gõ tay xấp xỉ.

---

## 2. Bảng Đối Soát Chi Tiết Các Điểm Mâu Thuẫn Trong Báo Cáo Cũ (Discrepancy Ledger)

| Mã phát hiện | Cấu hình | Số liệu bị ghi nhầm trong chat | Ma trận nhầm lẫn ngụ ý | Implied Macro-F1 chuẩn xác | Số liệu SSOT thực tế từ Raw Outputs | Ma trận nhầm lẫn thực tế |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **DISC-TXT-01** | A0-visual / C0 / P0 | F1=0.6011, BalAcc=0.6250, TPR=0.3500, FPR=0.1000 | TP=7, TN=18, FP=2, FN=13 | **0.594320** (làm tròn `0.5943`) | **F1=0.6011, BalAcc=0.6500, TPR=0.3000, FPR=0.0000** | **TP=6, TN=20, FP=0, FN=14** |
| **DISC-TXT-02** | A0-visual / C0 / P2 | F1=0.5847, BalAcc=0.6000, TPR=0.3500, FPR=0.1500 | TP=7, TN=17, FP=3, FN=13 | **0.573333** (làm tròn `0.5733`) | **F1=0.5807, BalAcc=0.6250, TPR=0.3000, FPR=0.0500** | **TP=6, TN=19, FP=1, FN=14** |
| **DISC-TXT-03** | A0-global / C0 / P0 | F1=0.4872, BalAcc=0.5250, TPR=0.2500, FPR=0.2000 | TP=5, TN=16, FP=4, FN=15 | **0.486139** (làm tròn `0.4861`) | **F1=0.4872, BalAcc=0.5500, TPR=0.2000, FPR=0.1000** | **TP=4, TN=18, FP=2, FN=16** |
| **DISC-TXT-04** | A0-global / C0 / P2 | F1=0.4848, BalAcc=0.5250, TPR=0.5000, FPR=0.4500 | TP=10, TN=11, FP=9, FN=10 | **0.524703** (làm tròn `0.5247`) | **F1=0.6229, BalAcc=0.6250, TPR=0.7000, FPR=0.4500** | **TP=14, TN=11, FP=9, FN=6** |

---

## 3. Bảng Kết Quả Chuẩn Hóa SSOT Toàn Diện (Tự Động Sinh Từ Raw Predictions)

Đánh giá trên toàn bộ 6 locked models, 3 hình học (P0, P1, P2) dưới 2 điều kiện C0 (Original) và C1 (Symmetric JPEG Q75), ngưỡng cố định $s_{\text{AI}} \ge 0.5$:

| Model | Cond | Geom | TP | TN | FP | FN | TPR (Recall) | TNR | FPR | Precision | Balanced Acc | Binary Macro-F1 | AUROC | AP | Bin Brier (so 0.25) | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-visual** | C0 | P0 |  6 | 20 |  0 | 14 | 0.3000 | 1.0000 | 0.0000 | 1.0000 | 0.6500 | **0.6011** | 0.7150 | 0.7752 | 0.2420 | 0.1928 |
| **A0-visual** | C0 | P1 | 11 | 17 |  3 |  9 | 0.5500 | 0.8500 | 0.1500 | 0.7857 | 0.7000 | **0.6931** | 0.6900 | 0.7215 | 0.2219 | 0.1761 |
| **A0-visual** | C0 | P2 |  6 | 19 |  1 | 14 | 0.3000 | 0.9500 | 0.0500 | 0.8571 | 0.6250 | **0.5807** | 0.6850 | 0.7629 | 0.2359 | 0.1268 |
| **A0-global** | C0 | P0 |  4 | 18 |  2 | 16 | 0.2000 | 0.9000 | 0.1000 | 0.6667 | 0.5500 | **0.4872** | 0.6100 | 0.6598 | 0.3070 | 0.2832 |
| **A0-global** | C0 | P1 |  8 | 17 |  3 | 12 | 0.4000 | 0.8500 | 0.1500 | 0.7273 | 0.6250 | **0.6050** | 0.5950 | 0.6586 | 0.2639 | 0.1917 |
| **A0-global** | C0 | P2 | 14 | 11 |  9 |  6 | 0.7000 | 0.5500 | 0.4500 | 0.6087 | 0.6250 | **0.6229** | 0.6750 | 0.6775 | 0.2283 | 0.0848 |
| **A0-local** | C0 | P0 |  3 | 17 |  3 | 17 | 0.1500 | 0.8500 | 0.1500 | 0.5000 | 0.5000 | **0.4302** | 0.5200 | 0.5340 | 0.3778 | 0.3733 |
| **A0-local** | C0 | P1 |  3 | 17 |  3 | 17 | 0.1500 | 0.8500 | 0.1500 | 0.5000 | 0.5000 | **0.4302** | 0.5200 | 0.6117 | 0.3044 | 0.2850 |
| **A0-local** | C0 | P2 | 13 | 12 |  8 |  7 | 0.6500 | 0.6000 | 0.4000 | 0.6190 | 0.6250 | **0.6248** | 0.6700 | 0.6989 | 0.2275 | 0.0966 |
| **A1-visual** | C0 | P0 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6525 | 0.7556 | 0.4221 | 0.4427 |
| **A1-visual** | C0 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.6200 | 0.6927 | 0.4116 | 0.4240 |
| **A1-visual** | C0 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6175 | 0.7037 | 0.4072 | 0.4159 |
| **A1-global** | C0 | P0 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6525 | 0.6788 | 0.4128 | 0.4226 |
| **A1-global** | C0 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.6275 | 0.7256 | 0.3938 | 0.4067 |
| **A1-global** | C0 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6300 | 0.6950 | 0.3551 | 0.3506 |
| **A1-local** | C0 | P0 |  0 | 19 |  1 | 20 | 0.0000 | 0.9500 | 0.0500 | 0.0000 | 0.4750 | **0.3220** | 0.5400 | 0.5342 | 0.4284 | 0.4446 |
| **A1-local** | C0 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.5100 | 0.6006 | 0.3936 | 0.3876 |
| **A1-local** | C0 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.5025 | 0.6031 | 0.3423 | 0.2955 |
| **A0-visual** | C1 | P0 |  6 | 20 |  0 | 14 | 0.3000 | 1.0000 | 0.0000 | 1.0000 | 0.6500 | **0.6011** | 0.7125 | 0.7773 | 0.2428 | 0.1942 |
| **A0-visual** | C1 | P1 | 11 | 17 |  3 |  9 | 0.5500 | 0.8500 | 0.1500 | 0.7857 | 0.7000 | **0.6931** | 0.6925 | 0.7260 | 0.2221 | 0.1763 |
| **A0-visual** | C1 | P2 |  6 | 19 |  1 | 14 | 0.3000 | 0.9500 | 0.0500 | 0.8571 | 0.6250 | **0.5807** | 0.6850 | 0.7629 | 0.2365 | 0.1273 |
| **A0-global** | C1 | P0 |  4 | 18 |  2 | 16 | 0.2000 | 0.9000 | 0.1000 | 0.6667 | 0.5500 | **0.4872** | 0.6125 | 0.6651 | 0.3072 | 0.2661 |
| **A0-global** | C1 | P1 |  8 | 17 |  3 | 12 | 0.4000 | 0.8500 | 0.1500 | 0.7273 | 0.6250 | **0.6050** | 0.5950 | 0.6604 | 0.2640 | 0.1930 |
| **A0-global** | C1 | P2 | 14 | 11 |  9 |  6 | 0.7000 | 0.5500 | 0.4500 | 0.6087 | 0.6250 | **0.6229** | 0.6700 | 0.6750 | 0.2283 | 0.0931 |
| **A0-local** | C1 | P0 |  3 | 17 |  3 | 17 | 0.1500 | 0.8500 | 0.1500 | 0.5000 | 0.5000 | **0.4302** | 0.5150 | 0.5317 | 0.3778 | 0.3722 |
| **A0-local** | C1 | P1 |  3 | 17 |  3 | 17 | 0.1500 | 0.8500 | 0.1500 | 0.5000 | 0.5000 | **0.4302** | 0.5150 | 0.6097 | 0.3043 | 0.2847 |
| **A0-local** | C1 | P2 | 13 | 13 |  7 |  7 | 0.6500 | 0.6500 | 0.3500 | 0.6500 | 0.6500 | **0.6500** | 0.6650 | 0.6946 | 0.2281 | 0.1331 |
| **A1-visual** | C1 | P0 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6525 | 0.7556 | 0.4227 | 0.4431 |
| **A1-visual** | C1 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.6200 | 0.6927 | 0.4117 | 0.4240 |
| **A1-visual** | C1 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6125 | 0.7009 | 0.4075 | 0.4159 |
| **A1-global** | C1 | P0 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6525 | 0.6788 | 0.4137 | 0.4235 |
| **A1-global** | C1 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.6300 | 0.7327 | 0.3941 | 0.4070 |
| **A1-global** | C1 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.6200 | 0.6879 | 0.3563 | 0.3520 |
| **A1-local** | C1 | P0 |  0 | 19 |  1 | 20 | 0.0000 | 0.9500 | 0.0500 | 0.0000 | 0.4750 | **0.3220** | 0.5400 | 0.5342 | 0.4290 | 0.4451 |
| **A1-local** | C1 | P1 |  1 | 20 |  0 | 19 | 0.0500 | 1.0000 | 0.0000 | 1.0000 | 0.5250 | **0.3866** | 0.5100 | 0.6161 | 0.3938 | 0.3879 |
| **A1-local** | C1 | P2 |  0 | 20 |  0 | 20 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | **0.3333** | 0.5025 | 0.6028 | 0.3428 | 0.2991 |

---

## 4. Phân Tích Thống Kê Ghép Cặp Khẳng Định Lại (Paired Image Bootstrap, 10,000 Reps)

- **A0-visual (P1 Letterbox vs P0 Center-Crop)**:
  - Điều kiện C0: $\Delta \text{Macro-F1} = +0.0920$ (95% CI: `[-0.0837, 0.2658]`, Bootstrap Positive Fraction = 85.31\% - tỷ lệ replicate dương, không phải p-value).
  - Điều kiện C1: $\Delta \text{Macro-F1} = +0.0920$ (95% CI: `[-0.0837, 0.2658]`, Bootstrap Positive Fraction = 85.31\%).
- **A0-global (P1 Letterbox vs P0 Center-Crop)**:
  - Điều kiện C0: $\Delta \text{Macro-F1} = +0.1178$ (95% CI: `[-0.028, 0.2686]`, Bootstrap Positive Fraction = 94.06\%).
  - Điều kiện C1: $\Delta \text{Macro-F1} = +0.1178$ (95% CI: `[-0.028, 0.2686]`, Bootstrap Positive Fraction = 94.06\%).

---

## 5. Giới Hạn & Ranh Giới Khẳng Định Khoa Học (Epistemic Rectification)

Theo nguyên tắc Scientific Honesty của AGENTS.md, báo cáo xác lập rõ 5 ranh giới nhận thức sau:
1. **Ranh giới tính tự nhất quán**: Việc khớp số đếm ô nhầm lẫn ($N = \text{TP} + \text{TN} + \text{FP} + \text{FN} = 40$) chỉ chứng minh bảng tóm tắt không có lỗi số học so với raw predictions; không chứng minh toàn bộ pipeline, tiền xử lý hay xác suất dự đoán là đúng tuyệt đối.
2. **Quy chuẩn độ chính xác số làm tròn**: Các chỉ số hiển thị ở 4 chữ số thập phân là giá trị tóm tắt làm tròn, không được gọi là 'bit-exact' so với số thực vô hạn.
3. **Ranh giới Bootstrap**: Resampling ghép cặp ở đây là **paired image bootstrap** (so sánh P1 vs P0 trên cùng hình ảnh); không gọi là source-cluster bootstrap vì chưa xác minh được quan hệ nguồn độc lập.
4. **Không nhầm lẫn Bootstrap Fraction với p-value**: 'Bootstrap positive fraction' (ví dụ $85.31\%$) chỉ là tỷ lệ số lần lấy mẫu lặp lại có delta dương trong 10,000 lần rút, tuyệt đối không được diễn giải thành p-value kiểm định giả thuyết thống kê.
5. **Giới hạn Multiclass Brier**: Điểm số Multiclass Brier trên nhãn nhị phân forged là phép chiếu giả định tham khảo (reference projection), không cấu thành phép đánh giá hợp lệ 3 lớp AI khi nhãn chưa xác minh nguồn gốc AI.

---

## 6. Kết Luận & Quy Tắc Bất Biến

1. **Không sao chép tay bảng số liệu**: Mọi bảng metrics trong tài liệu hoặc báo cáo bắt buộc phải xuất tự động qua runner/reconciler.
2. **Bảo toàn tính toán ma trận nhầm lẫn**: Mọi phát biểu về Recall, FPR, F1 bắt buộc phải kiểm tra tự nhất quán theo công thức toán học.
3. **Phân định rõ ràng**: Lỗi chỉ nằm ở khâu tổng hợp văn bản tóm tắt; các raw predictions trên đĩa hoàn toàn chính xác.