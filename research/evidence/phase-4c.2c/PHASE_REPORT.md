# Phase 4C.2C — Paired Stage 1 vs Stage 2 Analysis Report

> **Protocol**: Pre-registered paired learning curve analysis comparing Stage 1 frozen backbone linear probe vs Stage 2 pre-registered partial fine-tuning across identical sample sizes and random seeds.
> **Primary Endpoint**: $\Delta$ Macro-F1 paired $= \text{Macro-F1}_{\text{Stage 2}} - \text{Macro-F1}_{\text{Stage 1}}$ per cohort $N \in \{50, 100, 250\}$.
> **Evaluation Target**: Inner-validation partition ($91$ unique sources, $182$ balanced samples, $0$ development leaks, $0$ locked-test evaluations).
> **Scientific Status**: Model-development evidence. Locked-test remains sealed.
> **Verdict**: `PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE`

---

## 1. Mục tiêu và Giới hạn Khoa học (Scientific Bounds)

Giai đoạn **Phase 4C.2C** thực hiện phân tích thống kê đối chứng ghép cặp 1:1 giữa 15 runs Stage 1 (frozen linear probe) và 15 runs Stage 2 (pre-registered partial fine-tuning) trên cùng cỡ mẫu ($N \in \{50, 100, 250\}$) và cùng hạt giống ngẫu nhiên (seeds $42, 1337, 2025, 3407, 9001$).

### Ranh giới giải thích khoa học bắt buộc:
1. **Định danh can thiệp**: Stage 2 được định danh chuẩn tắc là **pre-registered partial fine-tuning protocol**.
2. **Không quy kết nhân quả đơn biến**: Kết quả phản ánh sự so sánh giữa hai quy trình huấn luyện đã đăng ký trước, không phải tác động thuần túy duy nhất của việc unfreeze, vì hai quy trình còn khác biệt về ngân sách epoch (Stage 1 max 25 epochs vs Stage 2 max 20 epochs), scheduler (`CosineAnnealingLR` $T_{\text{max}}=25$ vs $T_{\text{max}}=20$, $\eta_{\text{min}}=10^{-6}$), và optimizer parameter groups (tốc độ học phân hóa $5 \times 10^{-5}$ cho backbone và $5 \times 10^{-4}$ cho classifier head).
3. **Bản chất bằng chứng**: Tập `inner_validation` đã được dùng trong cả hai giai đoạn để dừng sớm và chọn checkpoint tốt nhất (`best_checkpoint.pt`). Do đó, kết quả là **bằng chứng phát triển mô hình (model-development evidence)**, không phải bằng chứng kiểm chuẩn xác nhận cuối cùng (confirmatory locked-test evidence).
4. **Locked-test niêm phong tuyệt đối**: Tập `locked_test` hoàn toàn không bị truy cập (0 evaluations, 0 byte reads).
5. **Không có training hoặc inference mới**: Toàn bộ phân tích được thực hiện trên các artifact đã hoàn thành và niêm phong trong kho lưu trữ local artifacts (0 new training runs, 0 GPU calls).

---

## 2. Kết quả Điểm cuối Chính (Primary Endpoint: Paired $\Delta$ Macro-F1)

Primary endpoint là mức chênh lệch Macro-F1 ghép cặp theo từng cỡ mẫu $N$, với $n = 5$ random seeds:

$$\Delta \text{ Macro-F1} = \text{Macro-F1}_{\text{Stage 2}} - \text{Macro-F1}_{\text{Stage 1}}$$

| Sample Size ($N$) | Seeds ($n$) | Stage 1 Mean $\pm$ SD | Stage 2 Mean $\pm$ SD | Paired $\Delta$ Mean $\pm$ SD | 95% CI (df=4) | Paired $t$ ($p_{\text{raw}}$) | Sign-Flip ($p_{\text{raw}}$) | Holm-Adjusted $p$ ($t$) | Holm-Adjusted $p$ (Perm) | Seed Count (Better / Equal / Worse) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 50** | 5 | 0.5322 ± 0.0148 | 0.5416 ± 0.0141 | +0.0094 ± 0.0082 | [-0.0008, +0.0196] | 0.0628 (t=2.56) | 0.1250 | 0.1884 | 0.3750 | 4 / 0 / 1 |
| **N = 100** | 5 | 0.5728 ± 0.0048 | 0.5766 ± 0.0192 | +0.0038 ± 0.0194 | [-0.0202, +0.0279] | 0.6833 (t=0.44) | 0.8750 | 0.8925 | 0.8750 | 2 / 0 / 3 |
| **N = 250** | 5 | 0.5627 ± 0.0172 | 0.5691 ± 0.0108 | +0.0064 ± 0.0170 | [-0.0147, +0.0275] | 0.4463 (t=0.84) | 0.3125 | 0.8925 | 0.6250 | 4 / 0 / 1 |

> [!IMPORTANT]
> **Nhận định Thống kê Nghiêm ngặt**:
> 1. **Cỡ mẫu $n=5$**: Với 5 seeds, paired t-test chỉ mang tính chất thăm dò (*exploratory*). Phép kiểm hoán vị dấu chính xác (*exact two-sided sign-flip permutation test*) trên $2^5 = 32$ tổ hợp dấu có độ phân giải tối thiểu là $2 / 32 = 0.0625$. Do đó, kiểm định hoán vị chính xác **về mặt toán học không thể đạt được $p < 0.05$** khi $n=5$.
> 2. **Không có ý nghĩa thống kê ở $\alpha = 0.05$**: Tại tất cả các cỡ mẫu ($N=50, 100, 250$), khoảng tin cậy 95% của $\Delta$ Macro-F1 đều bao hàm giá trị 0. Cả giá trị $p$ thô và giá trị $p$ sau hiệu chỉnh Holm-Bonferroni (cho cả paired t-test và permutation test) đều lớn hơn 0.05. Không có bằng chứng thống kê nào cho thấy Stage 2 vượt trội hơn Stage 1 một cách có ý nghĩa thống kê trên tập inner-validation.
> 3. **Quy mô hiệu ứng quan sát được**: Mức chênh lệch trung bình giữa hai giao thức là rất nhỏ: $+0.0094$ ($N=50$), $+0.0038$ ($N=100$), và $+0.0064$ ($N=250$).

---

## 3. Bảng Chi tiết Cấp Run (Run-Level Paired Metrics)

| $N$ | Seed | S1 Macro-F1 | S2 Macro-F1 | $\Delta$ Macro-F1 | S1 BalAcc | S2 BalAcc | $\Delta$ BalAcc | S1 AUROC | S2 AUROC | $\Delta$ AUROC | S1 Brier | S2 Brier | $\Delta$ Brier | S1 ECE | S2 ECE | $\Delta$ ECE | S1 Loss | S2 Loss | $\Delta$ Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 42 | 0.5415 | 0.5549 | +0.0135 | 0.5495 | 0.5549 | +0.0055 | 0.5607 | 0.5492 | -0.0115 | 0.2478 | 0.2483 | +0.0005 | 0.0265 | 0.0326 | +0.0060 | 0.1728 | 0.1731 | +0.0003 |
| 50 | 1337 | 0.5371 | 0.5546 | +0.0175 | 0.5385 | 0.5549 | +0.0165 | 0.5437 | 0.5654 | +0.0217 | 0.2486 | 0.2471 | -0.0015 | 0.0165 | 0.0197 | +0.0032 | 0.1734 | 0.1745 | +0.0011 |
| 50 | 2025 | 0.5382 | 0.5380 | -0.0003 | 0.5385 | 0.5385 | -0.0000 | 0.5381 | 0.5523 | +0.0142 | 0.2488 | 0.2478 | -0.0010 | 0.0146 | 0.0128 | -0.0018 | 0.1738 | 0.1735 | -0.0002 |
| 50 | 3407 | 0.5059 | 0.5208 | +0.0149 | 0.5275 | 0.5220 | -0.0055 | 0.5421 | 0.5248 | -0.0173 | 0.2489 | 0.2495 | +0.0006 | 0.0179 | 0.0019 | -0.0160 | 0.1741 | 0.1747 | +0.0006 |
| 50 | 9001 | 0.5384 | 0.5398 | +0.0014 | 0.5385 | 0.5549 | +0.0165 | 0.5261 | 0.5661 | +0.0400 | 0.2496 | 0.2476 | -0.0020 | 0.0100 | 0.0254 | +0.0154 | 0.1755 | 0.1734 | -0.0020 |
| 100 | 42 | 0.5773 | 0.5659 | -0.0114 | 0.5824 | 0.5659 | -0.0165 | 0.5920 | 0.5736 | -0.0184 | 0.2456 | 0.2467 | +0.0011 | 0.0543 | 0.0332 | -0.0211 | 0.1709 | 0.1735 | +0.0026 |
| 100 | 1337 | 0.5658 | 0.5750 | +0.0092 | 0.5659 | 0.5824 | +0.0165 | 0.5734 | 0.5679 | -0.0054 | 0.2461 | 0.2479 | +0.0018 | 0.0316 | 0.0494 | +0.0178 | 0.1728 | 0.1811 | +0.0083 |
| 100 | 2025 | 0.5740 | 0.6093 | +0.0353 | 0.5769 | 0.6099 | +0.0330 | 0.5865 | 0.5874 | +0.0008 | 0.2455 | 0.2462 | +0.0007 | 0.0492 | 0.0770 | +0.0278 | 0.1709 | 0.1729 | +0.0020 |
| 100 | 3407 | 0.5701 | 0.5604 | -0.0097 | 0.5714 | 0.5604 | -0.0110 | 0.5965 | 0.5869 | -0.0097 | 0.2452 | 0.2451 | -0.0001 | 0.0437 | 0.0223 | -0.0214 | 0.1705 | 0.1727 | +0.0022 |
| 100 | 9001 | 0.5766 | 0.5723 | -0.0043 | 0.5769 | 0.5769 | +0.0000 | 0.5823 | 0.5740 | -0.0083 | 0.2455 | 0.2461 | +0.0006 | 0.0463 | 0.0470 | +0.0007 | 0.1713 | 0.1766 | +0.0052 |
| 250 | 42 | 0.5769 | 0.5578 | -0.0191 | 0.5769 | 0.5604 | -0.0165 | 0.5807 | 0.5842 | +0.0035 | 0.2460 | 0.2453 | -0.0007 | 0.0527 | 0.0322 | -0.0206 | 0.1709 | 0.1760 | +0.0051 |
| 250 | 1337 | 0.5473 | 0.5672 | +0.0199 | 0.5714 | 0.5714 | +0.0000 | 0.5850 | 0.5706 | -0.0144 | 0.2463 | 0.2468 | +0.0005 | 0.0404 | 0.0398 | -0.0006 | 0.1725 | 0.1737 | +0.0013 |
| 250 | 2025 | 0.5449 | 0.5688 | +0.0239 | 0.5549 | 0.5769 | +0.0220 | 0.5765 | 0.5845 | +0.0080 | 0.2464 | 0.2460 | -0.0004 | 0.0269 | 0.0381 | +0.0112 | 0.1721 | 0.1785 | +0.0064 |
| 250 | 3407 | 0.5834 | 0.5869 | +0.0035 | 0.5879 | 0.5879 | -0.0000 | 0.5939 | 0.5999 | +0.0060 | 0.2468 | 0.2459 | -0.0008 | 0.0696 | 0.0634 | -0.0062 | 0.1710 | 0.1709 | -0.0001 |
| 250 | 9001 | 0.5610 | 0.5649 | +0.0038 | 0.5714 | 0.5659 | -0.0055 | 0.5939 | 0.5923 | -0.0016 | 0.2465 | 0.2447 | -0.0018 | 0.0503 | 0.0372 | -0.0131 | 0.1710 | 0.1720 | +0.0011 |

*(Quy ước dấu: $\Delta = \text{Stage 2} - \text{Stage 1}$. Đối với Macro-F1, Balanced Accuracy, AUROC: $\Delta > 0$ là tốt hơn. Đối với Brier, ECE, và loss: $\Delta < 0$ thường là tốt hơn).*

---

## 4. Phân tích Các Chỉ số Thứ cấp (Secondary Endpoints)

### 4.1. Balanced Accuracy và AUROC
- **Balanced Accuracy**:
  - $N=50$: S1 0.5385 vs S2 0.5451 ($\Delta = +0.0066 \pm 0.0098$).
  - $N=100$: S1 0.5747 vs S2 0.5791 ($\Delta = +0.0044 \pm 0.0203$).
  - $N=250$: S1 0.5725 vs S2 0.5725 ($\Delta = -0.0000 \pm 0.0140$).
- **AUROC**:
  - $N=50$: S1 0.5421 vs S2 0.5516 ($\Delta = +0.0094 \pm 0.0238$).
  - $N=100$: S1 0.5861 vs S2 0.5779 ($\Delta = -0.0082 \pm 0.0070$).
  - $N=250$: S1 0.5860 vs S2 0.5863 ($\Delta = +0.0003 \pm 0.0090$).

### 4.2. Chi phí Vận hành và Tài nguyên (Operational Metrics)
- **Training Time**: Thời gian huấn luyện trung bình Stage 1 vs Stage 2:
  - $N=50$: 109.8s vs 106.2s ($\Delta = -3.5s$).
  - $N=100$: 178.3s vs 184.4s ($\Delta = +6.1s$).
  - $N=250$: 248.2s vs 268.6s ($\Delta = +20.4s$).
- **Peak VRAM**: Stage 1 duy trì $109.0\text{ MB}$; Stage 2 tiêu thụ $110.1\text{ MB}$ (mức tăng chỉ $+1.1\text{ MB}$ cho việc kích hoạt gradient trên `features.12` và hai nhóm optimizer riêng biệt).
- **Best Epoch & Convergence**: Cả hai giao thức đều đạt điểm dừng sớm qua patience=5 sau khoảng 10-14 epochs (best epoch trung bình rơi vào khoảng epoch 5-9).

---

## 5. Phân tích Hiệu chuẩn (Calibration & Reliability Analysis)

Tuân thủ nghiêm ngặt chuẩn ngữ nghĩa thống kê, phân tích hiệu chuẩn phân tách rõ ràng hai khái niệm toán học độc lập:

1. **Calibration-in-the-large**: $\text{mean}(p_{\text{positive}} - y)$, đo lường mức độ sai lệch biên xác suất dương so với tỷ lệ mẫu thực tế (prevalence).
2. **Signed confidence calibration gap**: $\text{mean}(\text{confidence} - \text{correctness})$, trong đó $\text{confidence} = \max(p, 1-p)$ và $\text{correctness} = \mathbb{I}(\hat{y} = y)$. Chỉ chỉ số này mới phản ánh xu hướng tự tin quá mức (overconfidence $> 0$) hoặc bảo thủ/thiếu tự tin (underconfidence $< 0$).

| Sample Size ($N$) | Stage 1 CITL | Stage 2 CITL | $\Delta$ CITL | Stage 1 Conf Gap | Stage 2 Conf Gap | $\Delta$ Conf Gap | Stage 1 ECE | Stage 2 ECE | $\Delta$ ECE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 50** | -0.0052 ± 0.0096 | +0.0034 ± 0.0080 | +0.0086 | -0.0138 ± 0.0091 | -0.0167 ± 0.0134 | -0.0029 | 0.0171 ± 0.0061 | 0.0185 ± 0.0118 | +0.0014 |
| **N = 100** | -0.0016 ± 0.0066 | +0.0059 ± 0.0123 | +0.0075 | -0.0450 ± 0.0085 | -0.0375 ± 0.0224 | +0.0075 | 0.0450 ± 0.0085 | 0.0458 ± 0.0206 | +0.0008 |
| **N = 250** | +0.0013 ± 0.0129 | +0.0012 ± 0.0142 | -0.0001 | -0.0480 ± 0.0158 | -0.0332 ± 0.0194 | +0.0148 | 0.0480 ± 0.0158 | 0.0421 ± 0.0122 | -0.0059 |

> [!NOTE]
> **Quan sát Định lượng về Hiệu chuẩn**:
> 1. **Calibration-in-the-large gần 0**: Sai số trung bình $\text{mean}(p_{\text{positive}} - y)$ dao động trong khoảng $\pm 0.001$ đến $\pm 0.006$, chứng minh xác suất dự đoán trung bình không bị lệch khỏi tỷ lệ cân bằng 50% của nhãn.
> 2. **Signed confidence calibration gap mang giá trị âm**: Không quan sát thấy xu hướng overconfidence trung bình trên inner-validation; signed confidence calibration gap âm gợi ý xu hướng underconfidence nhẹ. Đây là bằng chứng phát triển mô hình, không phải kết luận confirmatory.
> 3. **Phân bố xác suất**: Xác suất dự đoán của cả hai giai đoạn chủ yếu tập trung hẹp trong khoảng $[0.35, 0.65]$, không rơi vào các vùng cực đoan $[0.0, 0.1]$ hay $[0.9, 1.0]$.
> 4. **Không fit Temperature Scaling trong phase này**: Để đảm bảo tính trung thực khoa học, Temperature Scaling không được fit trên tập inner-validation vì tập này đã được dùng để lựa chọn checkpoint.

---

## 6. Danh mục Biểu đồ Minh chứng (Figures Register)

Tất cả các biểu đồ đều được xuất thành cặp định dạng PNG (raster độ nét cao) và SVG (vector chuẩn):

1. **`figures/paired_macro_f1_by_n.svg` / `.png`**: Biểu diễn Macro-F1 ghép cặp theo cỡ mẫu $N$, bao gồm giá trị trung bình từng stage kèm thanh sai số 95% CI (Student's t, df=4) và các đường nối từng seed ghép cặp.
2. **`figures/delta_macro_f1_by_seed.svg` / `.png`**: Biểu diễn chi tiết mức chênh lệch $\Delta$ Macro-F1 theo từng hạt giống ngẫu nhiên, đường tham chiếu $\Delta=0$, và giá trị trung bình kèm 95% CI của từng cohort.
3. **`figures/calibration_comparison.svg` / `.png`**: Biểu đồ độ tin cậy (*reliability diagram*) 3 bảng cho $N=50, 100, 250$, so sánh đường cong hiệu chuẩn Stage 1 và Stage 2 đối chiếu với đường chéo hiệu chuẩn hoàn hảo, bao gồm thanh sai số $\pm 1$ SD thể hiện biến thiên giữa 5 seeds và không nối qua các bin rỗng.

---

## 7. Bằng chứng Toàn vẹn và Ràng buộc Môi trường (Provenance & Integrity)

- **Stage 1 Artifacts Base**: `research/evidence/phase-4c.1d/` (15 completed runs, frozen stage, commit `79bb115`).
- **Stage 2 Ingestion Base**: `execution_9ee7fdb` (15 completed runs, partial_finetune stage, commit `9ee7fdb`, archive SHA-256 `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`).
- **Toàn vẹn Checksums**: 100% tệp trong `checksums.json` của cả 30 runs đều khớp mã băm SHA-256 và kích thước byte.
- **Bảo toàn Partition**: $91$ validation sources được đánh giá giống hệt nhau ở cả 30 runs ($182$ mẫu, tỷ lệ $1:1$ authentic/edited). $0$ trường hợp rò rỉ development train hoặc locked test.
- **Không ghi đè dữ liệu**: `stage1_output_writes == 0` tuyệt đối.
- **Số lượt huấn luyện mới**: $0$.
- **Số lần truy cập locked-test**: $0$.

---

## 8. Kết luận và Quyết định Tiếp theo (Conclusion & Next Approved Action)

Phân tích đối chứng ghép cặp giữa Stage 1 frozen linear probe và Stage 2 pre-registered partial fine-tuning protocol trên 15 cặp thực nghiệm độc lập cho thấy:

1. **Không quan sát thấy sự vượt trội có ý nghĩa thống kê của Stage 2 so với Stage 1 trên tập inner-validation** ($p > 0.05$ trên mọi cỡ mẫu $N$, cả qua t-test thăm dò và exact sign-flip permutation test, trước và sau hiệu chỉnh Holm-Bonferroni).
2. Mức chênh lệch Macro-F1 ghép cặp trung bình là nhỏ ($+0.0038$ đến $+0.0094$), và khoảng tin cậy 95% đều bao hàm giá trị 0.
3. Cả hai giao thức đều đạt mức Macro-F1 khoảng $0.56 - 0.58$ ở $N=100$ và $N=250$, vượt qua Stratified Dummy baseline ($0.4749 \pm 0.0325$ qua 5 seeds trên inner-validation) và uninformative metadata placeholder baseline ($0.5000$). Giá trị 0.5000 là baseline giữ chỗ phi thông tin, không đại diện cho mô hình siêu dữ liệu hoàn chỉnh và không dùng để kết luận siêu dữ liệu vô ích.
4. Kết quả này phản ánh rằng việc mở khóa tầng `features.12` kết hợp differential learning rate trong khuôn khổ protocol đã đăng ký chưa tạo ra bước nhảy vọt đáng kể về năng lực phân loại trên tập inner-validation so với linear probe đóng băng.

Phán quyết chính thức:
**`PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE`**

**Hành động tiếp theo được phê duyệt**: Báo cáo kết quả Phase 4C.2C lên người dùng, cập nhật continuity documentation, và chờ quyết định chiến lược tiếp theo (không tự ý mở locked-test và không tự ý merge vào main).