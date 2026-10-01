# Báo cáo Phân tích Thực nghiệm: Đường cong Học tập Stage 1 (Phase 4C.1D Learning Curve Report)

> **Repository**: `forensics-web-lab`<br>
> **Phase**: `Phase 4C.1D — Ingest, Verify and Analyze 15 Stage-1 Runs`<br>
> **Môi trường huấn luyện từ xa**: NVIDIA Tesla T4 GPU (14.56 GB VRAM, Google Colab runtime, CUDA 12.8, PyTorch 2.11+cu128)<br>
> **Môi trường phân tích đối soát**: Local Analysis Workstation (Windows 11, NVIDIA GeForce GTX 1650 4.0 GB VRAM, Python 3.12.10, PyTorch 2.5.1+cu121)<br>
> **Thời điểm thẩm tra**: 2026-10-01 UTC<br>
> **Dataset**: Reusable N=250 Canonical Dataset Bundle (`c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`)<br>
> **Tập đánh giá**: Canonical Inner Validation Cohort (91 source pairs = 182 mẫu cân bằng 1:1, 0% dev/locked leakage)

---

## 1. Tóm tắt Thực thi & Bối cảnh Nghiên cứu

Giai đoạn Stage 1 của Phase 4C.1 khảo sát tính hiệu quả của mô hình MobileNetV3-Small ở cấu hình **Frozen Backbone** (Linear Probing với đầu phân loại huấn luyện lại). Toàn bộ 15 runs hoàn chỉnh ($N \in \{50, 100, 250\} \times 5$ random seeds $\{42, 1337, 2025, 3407, 9001\}$) đã được thực thi trên Colab T4 thông qua canonical autonomous operator `phase_4c1_t4_execute_all_stage1.sh` (Phase 4C.1C.15).

Toàn bộ 15 runs đạt chuẩn an toàn nghiên cứu tuyệt đối:
- Trạng thái: `status: completed` và `stage: frozen`.
- Bảo vệ tập kiểm tra: `locked_test_access == 0` (0 vi phạm).
- Không kích hoạt Stage 2: `stage2_invocations == 0` (0 vi phạm).
- Đánh giá trên cùng một tập 91 validation sources (`inner_validation`, 182 mẫu cân bằng 91 authentic : 91 ai_edited).
- Khớp 100% mã băm code snapshot (`79bb115...`), dataset content hash (`c365c812...`), và cấu hình huấn luyện (`e03c07da...`).

---

## 2. Bảng Dữ liệu Chi tiết Từng Run (15 Runs)

| Run ID | Cỡ mẫu (N) | Seed | Best Epoch / Tổng | Macro-F1 | Balanced Acc | AUROC | Brier Score | ECE | Runner Val Loss | Thời gian (s) | Checkpoint SHA-256 (8 ký tự đầu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phase4c1-stage1-n50-seed42` | 50 | 42 | 18 / 23 | 0.5415 | 0.5495 | 0.5607 | 0.2478 | 0.0265 | 0.1728 | 231.3 | `b5fa53bf...` |
| `phase4c1-stage1-n50-seed1337` | 50 | 1337 | 5 / 10 | 0.5371 | 0.5385 | 0.5437 | 0.2486 | 0.0165 | 0.1734 | 99.7 | `6a709a64...` |
| `phase4c1-stage1-n50-seed2025` | 50 | 2025 | 2 / 7 | 0.5382 | 0.5385 | 0.5381 | 0.2488 | 0.0146 | 0.1738 | 69.1 | `2827a79f...` |
| `phase4c1-stage1-n50-seed3407` | 50 | 3407 | 3 / 8 | 0.5059 | 0.5275 | 0.5421 | 0.2489 | 0.0179 | 0.1741 | 79.4 | `575240ac...` |
| `phase4c1-stage1-n50-seed9001` | 50 | 9001 | 2 / 7 | 0.5384 | 0.5385 | 0.5261 | 0.2496 | 0.0100 | 0.1755 | 69.4 | `1f74b9df...` |
| `phase4c1-stage1-n100-seed42` | 100 | 42 | 7 / 12 | 0.5773 | 0.5824 | 0.5920 | 0.2456 | 0.0543 | 0.1709 | 162.2 | `65617b42...` |
| `phase4c1-stage1-n100-seed1337` | 100 | 1337 | 14 / 19 | 0.5658 | 0.5659 | 0.5734 | 0.2461 | 0.0316 | 0.1728 | 260.1 | `5a9f04bc...` |
| `phase4c1-stage1-n100-seed2025` | 100 | 2025 | 6 / 11 | 0.5740 | 0.5769 | 0.5865 | 0.2455 | 0.0492 | 0.1709 | 152.0 | `ce3b5e0b...` |
| `phase4c1-stage1-n100-seed3407` | 100 | 3407 | 7 / 12 | 0.5701 | 0.5714 | 0.5965 | 0.2452 | 0.0437 | 0.1705 | 166.5 | `78dd4949...` |
| `phase4c1-stage1-n100-seed9001` | 100 | 9001 | 6 / 11 | 0.5766 | 0.5769 | 0.5823 | 0.2455 | 0.0463 | 0.1713 | 150.8 | `24da001f...` |
| `phase4c1-stage1-n250-seed42` | 250 | 42 | 6 / 11 | 0.5769 | 0.5769 | 0.5807 | 0.2460 | 0.0527 | 0.1709 | 267.9 | `c941f42e...` |
| `phase4c1-stage1-n250-seed1337` | 250 | 1337 | 5 / 10 | 0.5473 | 0.5714 | 0.5850 | 0.2463 | 0.0404 | 0.1725 | 243.6 | `69e706f9...` |
| `phase4c1-stage1-n250-seed2025` | 250 | 2025 | 5 / 10 | 0.5449 | 0.5549 | 0.5765 | 0.2464 | 0.0269 | 0.1721 | 243.8 | `92ce5ee9...` |
| `phase4c1-stage1-n250-seed3407` | 250 | 3407 | 4 / 9 | 0.5834 | 0.5879 | 0.5939 | 0.2468 | 0.0696 | 0.1710 | 218.3 | `4897821e...` |
| `phase4c1-stage1-n250-seed9001` | 250 | 9001 | 6 / 11 | 0.5610 | 0.5714 | 0.5939 | 0.2465 | 0.0503 | 0.1710 | 267.3 | `5f0f8803...` |

---

## 3. Tổng hợp Thống kê theo Cỡ Mẫu ($N$)

*Khoảng tin cậy 95% được tính toán theo phân phối Student's $t$ với bậc tự do $df = 4$ ($t_{0.975, 4} \approx 2.7764$), độ lệch chuẩn mẫu hiệu chỉnh $ddof = 1$. Thử nghiệm mang tính chất thăm dò (exploratory, $n=5$ seeds).*

### Bảng 1: Chỉ số Đánh giá Mô hình theo $N$ (Mean ± Std, 95% CI)

| Cỡ mẫu ($N$) | Macro-F1 (Mean ± Std) | Macro-F1 [95% CI] | Balanced Acc (Mean ± Std) | AUROC (Mean ± Std) | AUROC [95% CI] | Brier Score (Thấp hơn là tốt) | ECE (Thấp hơn là tốt) | Runner Val Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 50$** | 0.5322 ± 0.0148 | [0.5139, 0.5506] | 0.5385 ± 0.0078 | 0.5421 ± 0.0124 | [0.5267, 0.5576] | 0.2487 ± 0.0007 | 0.0171 ± 0.0061 | 0.1739 ± 0.0010 |
| **$N = 100$** | 0.5728 ± 0.0048 | [0.5668, 0.5787] | 0.5747 ± 0.0063 | 0.5861 ± 0.0089 | [0.5750, 0.5972] | 0.2456 ± 0.0003 | 0.0450 ± 0.0085 | 0.1713 ± 0.0009 |
| **$N = 250$** | 0.5627 ± 0.0172 | [0.5413, 0.5841] | 0.5725 ± 0.0119 | 0.5860 ± 0.0078 | [0.5763, 0.5957] | 0.2464 ± 0.0003 | 0.0480 ± 0.0158 | 0.1715 ± 0.0007 |

### Bảng 2: Thống kê Mô tả Vận hành & Tài nguyên theo $N$

| Cỡ mẫu ($N$) | Best Epoch (Mean, Median, Min-Max) | Epochs Completed (Mean, Min-Max) | Thời gian Huấn luyện (s) (Mean ± Std) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 50$** | 6.0 (median 3, 2 - 18) | 11.0 (7 - 23) | 109.8 ± 69.0 (69.1 - 231.3) | 109.0 |
| **$N = 100$** | 8.0 (median 7, 6 - 14) | 13.0 (11 - 19) | 178.3 ± 46.2 (150.8 - 260.1) | 109.0 |
| **$N = 250$** | 5.2 (median 5, 4 - 6) | 10.2 (9 - 11) | 248.2 ± 20.6 (218.3 - 267.9) | 109.0 |

### 3.1. Truy vết Mã nguồn và Ngữ nghĩa Chỉ số Validation Loss

Truy vết mã nguồn trong execution snapshot `79bb115` (`ml/training/run_phase_4c1.py` dòng 46, 517 và `ml/training/loss.py` dòng 6-51):
- **Hàm mất mát cấu hình**: `criterion = FocalLoss(gamma=2.0, label_smoothing=0.05, reduction='mean')`.
- **Đầu vào hàm loss**: Logits thô (`outputs = model(inputs)`, shape `[B, 2]`) và nhãn integer (`targets`, shape `[B]`).
- **Cơ chế tính toán nội bộ**: `FocalLoss` áp dụng `F.log_softmax(logits, dim=1)`, làm mịn nhãn với `label_smoothing=0.05` trên 2 lớp, nhân trọng số focal `(1 - p)^2`, và tính `loss.mean()` trên từng mini-batch.
- **Tích lũy & Thu gọn (Reduction)**: Trong hàm `evaluate()`, loss được tích lũy theo số mẫu `running_loss += loss.item() * targets.size(0)`, và epoch loss được chuẩn hóa bằng tổng số mẫu `running_loss / max(1, total)` (182 mẫu `inner_validation`).
- **Kết luận ngữ nghĩa**: Giá trị được ghi nhận trong `metrics.json` là trung bình có trọng số theo mẫu của Multi-class Focal Loss (gamma=2.0, label_smoothing=0.05) trên logits thô, không phải Binary Cross Entropy và không qua sigmoid độc lập. Do đó, chỉ số này được định danh chính xác là **runner-reported validation loss** và không suy diễn thang đo ngoài định nghĩa toán học của hàm.

---

## 4. Phân tích Paired Deltas (Kiểm định Cặp trên cùng 5 Seeds)

Do mỗi hạt ngẫu nhiên trong $\{42, 1337, 2025, 3407, 9001\}$ được huấn luyện nhất quán trên cả 3 cỡ mẫu, kiểm định cặp (Paired Samples) được thực hiện để loại trừ phương sai khởi tạo:
- **Paired $t$-test**: Mang tính chất thăm dò (exploratory) với quy mô mẫu nhỏ $n = 5$ random seeds.
- **Exact Paired Sign-Flip Permutation Test**: Với $n = 5$, không gian hoán vị gồm $2^5 = 32$ hoán vị đối xứng, độ phân giải tối thiểu hai phía là $2 / 32 = 0.0625$. Do đó về mặt toán học không thể đạt mức ý nghĩa $\alpha = 0.05$ dù 5/5 seed đều ghi nhận độ lệch cùng chiều dương; nghiên cứu không khẳng định ý nghĩa thống kê xác quyết khi permutation test chưa đạt ngưỡng 0.05.
- **Hiệu chỉnh Multiple Comparisons**: Duy trì quy trình hiệu chỉnh Holm-Bonferroni cho họ 3 so sánh giả thuyết chính của Macro-F1 đã đăng ký trước ($N_{100}-N_{50}$, $N_{250}-N_{100}$, $N_{250}-N_{50}$).

### Bảng 3: Chi tiết Paired Deltas theo Seed

| Cặp so sánh | Seed 42 | Seed 1337 | Seed 2025 | Seed 3407 | Seed 9001 | $\Delta$ Mean | Std ($ddof=1$) | 95% CI | $t$-statistic | $p$-value ($t$-test) | $p$-value (Holm) | $p$-value (Perm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\Delta \text{Macro-F1}_{(100 - 50)}$** | +0.0358 | +0.0288 | +0.0358 | +0.0642 | +0.0382 | **+0.0406** | 0.0137 | [+0.0236, +0.0575] | +6.628 | 0.0027 | 0.0081 | 0.0625 |
| **$\Delta \text{Macro-F1}_{(250 - 100)}$** | -0.0004 | -0.0185 | -0.0291 | +0.0132 | -0.0156 | **-0.0101** | 0.0166 | [-0.0307, +0.0105] | -1.357 | 0.2463 | 0.2463 | 0.1875 |
| **$\Delta \text{Macro-F1}_{(250 - 50)}$** | +0.0354 | +0.0103 | +0.0067 | +0.0774 | +0.0226 | **+0.0305** | 0.0286 | [-0.0050, +0.0660] | +2.386 | 0.0755 | 0.1510 | 0.0625 |
| **$\Delta \text{AUROC}_{(100 - 50)}$** | +0.0313 | +0.0297 | +0.0484 | +0.0545 | +0.0562 | **+0.0440** | 0.0127 | [+0.0283, +0.0597] | +7.761 | 0.0015 | N/A | 0.0625 |
| **$\Delta \text{AUROC}_{(250 - 100)}$** | -0.0112 | +0.0116 | -0.0100 | -0.0027 | +0.0116 | **-0.0001** | 0.0112 | [-0.0141, +0.0138] | -0.029 | 0.9783 | N/A | 1.0000 |
| **$\Delta \text{AUROC}_{(250 - 50)}$** | +0.0200 | +0.0413 | +0.0384 | +0.0518 | +0.0677 | **+0.0439** | 0.0176 | [+0.0220, +0.0657] | +5.577 | 0.0051 | N/A | 0.0625 |

---

## 5. Đối chiếu với các Baselines Không Học (Non-Learning Baselines)

### 5.1. Định nghĩa & Kiểm toán Mã nguồn của Baselines
1. **Stratified Bernoulli Dummy Baseline**:
   - Được triển khai qua `DummyClassifier(strategy='stratified', random_state=seed)` trong `ml/training/run_phase_4c1.py`.
   - Sinh dự đoán độc lập dựa trên tần suất tiên nghiệm của tập kiểm tra (50:50).
   - Do phụ thuộc vào hạt ngẫu nhiên `seed`, giá trị Macro-F1 thay đổi theo từng seed:
     - Seed 42: `0.4940`
     - Seed 1337: `0.4173`
     - Seed 2025: `0.4820`
     - Seed 3407: `0.4930`
     - Seed 9001: `0.4883`
   - **Tổng hợp 5 seeds**: Mean = `0.4749 ± 0.0325`, 95% CI `[0.4345, 0.5153]`.
2. **Uninformative Metadata Placeholder Baseline**:
   - Triển khai qua `run_metadata_baseline` trả về hợp đồng hằng số `macro_f1 = 0.5000` (Mean = `0.5000 ± 0.0000`), mô phỏng trường hợp không có bộ phân loại metadata được huấn luyện.
   - Giá trị hằng số 0.5000 này là một placeholder tham chiếu chưa qua huấn luyện, không phải kết quả đánh giá của một mô hình metadata hoàn chỉnh, và tuyệt đối không được dùng làm căn cứ để kết luận siêu dữ liệu (EXIF/C2PA) không có giá trị phân biệt pháp chứng.

### 5.2. Bảng So sánh Cặp giữa Mô hình và Baselines

| Cấu hình Mô hình | Macro-F1 (Mean ± Std) | $\Delta$ vs Dummy (Mean ± Std) | 95% CI vs Dummy | $p$-value vs Dummy ($t$ / Perm) | $\Delta$ vs Metadata (Mean ± Std) | 95% CI vs Metadata | $p$-value vs Metadata ($t$ / Perm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 ($N = 50$)** | 0.5322 ± 0.0148 | +0.0573 ± 0.0388 | [+0.0092, +0.1055] | 0.0298 / 0.0625 | +0.0322 ± 0.0148 | [+0.0139, +0.0506] | 0.0082 / 0.0625 |
| **Stage 1 ($N = 100$)** | 0.5728 ± 0.0048 | +0.0979 ± 0.0288 | [+0.0621, +0.1337] | 0.0016 / 0.0625 | +0.0728 ± 0.0048 | [+0.0668, +0.0787] | 0.0000 / 0.0625 |
| **Stage 1 ($N = 250$)** | 0.5627 ± 0.0172 | +0.0878 ± 0.0258 | [+0.0558, +0.1198] | 0.0016 / 0.0625 | +0.0627 ± 0.0172 | [+0.0413, +0.0841] | 0.0012 / 0.0625 |

---

## 6. Biểu đồ Trực quan hóa

Toàn bộ biểu đồ định dạng SVG vector và PNG raster chất lượng cao được lưu trữ tại `research/evidence/phase-4c.1d/figures/`:
1. **Learning Curve Macro-F1 (Mean & 95% CI)**: [SVG](figures/learning_curve_macro_f1.svg) | [PNG](figures/learning_curve_macro_f1.png)
2. **Per-Seed Trajectories**: [SVG](figures/per_seed_trajectories.svg) | [PNG](figures/per_seed_trajectories.png)
3. **Balanced Accuracy and AUROC vs N**: [SVG](figures/balanced_accuracy_auroc_vs_n.svg) | [PNG](figures/balanced_accuracy_auroc_vs_n.png)
4. **Calibration Quality (Brier Score & ECE) vs N**: [SVG](figures/brier_ece_vs_n.svg) | [PNG](figures/brier_ece_vs_n.png)
5. **Model vs Baselines**: [SVG](figures/model_vs_baselines.svg) | [PNG](figures/model_vs_baselines.png)

---

## 7. Diễn giải Khoa học Thận trọng & Giới hạn Nghiên cứu

1. **Xu hướng Học tập Quan sát được**:
   - Từ $N = 50$ lên $N = 100$: Hiệu năng phân loại Macro-F1 tăng $+0.0406$ (Holm-corrected $p = 0.0081$, exact permutation $p = 0.0625$), với 5/5 seeds ghi nhận cải thiện.
   - Từ $N = 100$ lên $N = 250$: **Không quan sát thấy cải thiện** ($\Delta \text{Macro-F1} = -0.0101$, Holm-corrected $p = 0.2463$, 95% CI $[-0.0307, +0.0105]$ bao hàm giá trị 0; $\Delta \text{AUROC} = -0.0001$).
2. **Căn nguyên Kỹ thuật (Hypothesis)**:
   - Hiện tượng hiệu năng đi ngang khi tăng dữ liệu từ 100 lên 250 cặp ảnh **nhất quán với, nhưng không chứng minh, một điểm nghẽn biểu diễn (representation bottleneck)** của đầu phân loại tuyến tính trên backbone ImageNet đóng băng.
   - Các đặc trưng cấp cao của ImageNet có thể chưa đủ nhạy với ranh giới chỉnh sửa cục bộ nếu không được tinh chỉnh trọng số.
   - **Stage 2 (Backbone Fine-tuning)** được đề xuất như một **thực nghiệm tiếp theo để kiểm tra giả thuyết này**, không phải một khẳng định đã được chứng minh trước.
3. **Độ Hiệu chuẩn Xác suất (Calibration)**:
   - Sai số hiệu chuẩn kỳ vọng (ECE) tăng từ $0.0171$ ($N=50$) lên $0.0450$ ($N=100$) và $0.0480$ ($N=250$). ECE tăng cho thấy độ lệch hiệu chuẩn lớn hơn; hướng lệch overconfidence hay underconfidence cần được xác định bằng reliability diagram hoặc signed calibration error.
   - Temperature Scaling là phương án calibration cần được đánh giá trên tập calibration độc lập hoặc bằng quy trình nested/cross-fitted phù hợp, tuyệt đối không fit temperature trên chính dữ liệu dùng để lựa chọn mô hình rồi báo cáo trên cùng tập đó.
4. **Giới hạn của Thí nghiệm**:
   - Thí nghiệm mang tính thăm dò với quy mô $n = 5$ random seeds.
   - Dữ liệu đánh giá hiện tại là nhị phân (`authentic` vs `ai_edited`) trên tập TGIF Option P; kết quả này chưa khái quát hóa ra ngoài phân phối hoặc sang không gian 3 lớp đầy đủ.
   - Chưa khẳng định tính khả dụng cho môi trường sản phẩm.
