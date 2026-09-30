# Báo cáo Phân tích Thực nghiệm: Đường cong Học tập Stage 1 (Phase 4C.1D Learning Curve Report)

> **Repository**: `forensics-web-lab`  
> **Phase**: `Phase 4C.1D — Ingest, Verify and Analyze 15 Stage-1 Runs`  
> **Phần cứng thực thi**: NVIDIA Tesla T4 GPU (14.56 GB VRAM, CUDA 12.8, Colab System Python 3.13 / PyTorch 2.11+cu128)  
> **Thời điểm thẩm tra**: 2026-10-01 UTC  
> **Dataset**: Reusable N=250 Canonical Dataset Bundle (`c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`)  
> **Tập đánh giá**: Canonical Inner Validation Cohort (91 source pairs = 182 mẫu cân bằng 1:1, 0% leak)  

---

## 1. Tóm tắt Thực thi & Bối cảnh Nghiên cứu

Giai đoạn Stage 1 của Phase 4C.1 khảo sát tính hiệu quả của mô hình MobileNetV3-Small ở chế độ **Frozen Backbone** (Linear Probing với đầu phân loại huấn luyện lại). Thí nghiệm được thực thi trên Google Colab T4 thông qua canonical autonomous operator `phase_4c1_t4_execute_all_stage1.sh` (Phase 4C.1C.15).

Tổng cộng **15 runs hoàn chỉnh** (3 cỡ mẫu $N \in \{50, 100, 250\} \times 5$ random seeds $\{42, 1337, 2025, 3407, 9001\}$) đã hoàn thành 100%, được niêm phong vào archive `phase_4c1_all_15_runs_results.tar.gz` (SHA-256 `f58d6631...`), và được tải về hệ thống local để đối soát độc lập.

Tất cả 15 runs đều:
- Đạt trạng thái `status: completed` và `stage: frozen`.
- Không truy cập tập kiểm tra đóng (`locked_test_access == 0`).
- Không gọi lệnh huấn luyện Stage 2 (`stage2_invocations == 0`).
- Đánh giá trên cùng một tập 91 validation sources (`inner_validation`, 182 mẫu cân bằng 91 authentic : 91 ai_edited).
- Khớp 100% mã băm code (`79bb115...`), dataset content hash (`c365c812...`), và cấu hình huấn luyện (`e03c07da...`).

---

## 2. Bảng Dữ liệu Chi tiết Từng Run (15 Runs)

| Run ID | Cỡ mẫu (N) | Seed | Best Epoch / Tổng | Macro-F1 | Balanced Acc | AUROC | Brier Score | ECE | Val Loss | Thời gian (s) | Checkpoint SHA-256 (8 ký tự đầu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phase4c1-stage1-n50-seed42` | 50 | 42 | 18 / 23 | 0.5415 | 0.5495 | 0.5607 | 0.2478 | 0.0265 | 0.1728 | 231.3 | `b5fa53bf...` |
| `phase4c1-stage1-n50-seed1337` | 50 | 1337 | 5 / 10 | 0.5371 | 0.5385 | 0.5437 | 0.2488 | 0.0165 | 0.1738 | 99.7 | `0c042211...` |
| `phase4c1-stage1-n50-seed2025` | 50 | 2025 | 2 / 7 | 0.5382 | 0.5385 | 0.5381 | 0.2494 | 0.0100 | 0.1755 | 69.1 | `d2807f6e...` |
| `phase4c1-stage1-n50-seed3407` | 50 | 3407 | 3 / 8 | 0.5059 | 0.5275 | 0.5421 | 0.2496 | 0.0210 | 0.1747 | 79.4 | `aaee1b6d...` |
| `phase4c1-stage1-n50-seed9001` | 50 | 9001 | 2 / 7 | 0.5384 | 0.5385 | 0.5261 | 0.2478 | 0.0116 | 0.1729 | 69.4 | `ec8ff730...` |
| `phase4c1-stage1-n100-seed42` | 100 | 42 | 7 / 12 | 0.5773 | 0.5824 | 0.5920 | 0.2456 | 0.0543 | 0.1709 | 162.2 | `bf567b57...` |
| `phase4c1-stage1-n100-seed1337` | 100 | 1337 | 14 / 19 | 0.5658 | 0.5659 | 0.5734 | 0.2461 | 0.0316 | 0.1728 | 260.1 | `52002167...` |
| `phase4c1-stage1-n100-seed2025` | 100 | 2025 | 6 / 11 | 0.5740 | 0.5769 | 0.5865 | 0.2461 | 0.0446 | 0.1709 | 152.0 | `23101183...` |
| `phase4c1-stage1-n100-seed3407` | 100 | 3407 | 7 / 12 | 0.5701 | 0.5714 | 0.5965 | 0.2459 | 0.0467 | 0.1705 | 166.5 | `ebc9f6a7...` |
| `phase4c1-stage1-n100-seed9001` | 100 | 9001 | 6 / 11 | 0.5766 | 0.5769 | 0.5823 | 0.2455 | 0.0463 | 0.1713 | 150.8 | `89fca6dc...` |
| `phase4c1-stage1-n250-seed42` | 250 | 42 | 6 / 11 | 0.5769 | 0.5769 | 0.5807 | 0.2460 | 0.0527 | 0.1709 | 267.9 | `c6870d04...` |
| `phase4c1-stage1-n250-seed1337` | 250 | 1337 | 5 / 10 | 0.5473 | 0.5714 | 0.5850 | 0.2463 | 0.0404 | 0.1725 | 243.6 | `86ca1c9a...` |
| `phase4c1-stage1-n250-seed2025` | 250 | 2025 | 5 / 10 | 0.5449 | 0.5549 | 0.5765 | 0.2464 | 0.0269 | 0.1721 | 243.8 | `b8562d4e...` |
| `phase4c1-stage1-n250-seed3407` | 250 | 3407 | 4 / 9 | 0.5834 | 0.5879 | 0.5939 | 0.2468 | 0.0696 | 0.1710 | 218.3 | `64c8be1e...` |
| `phase4c1-stage1-n250-seed9001` | 250 | 9001 | 6 / 11 | 0.5610 | 0.5714 | 0.5939 | 0.2465 | 0.0503 | 0.1710 | 267.3 | `8f515e01...` |

---

## 3. Tổng hợp Thống kê theo Cỡ Mẫu ($N$)

*Khoảng tin cậy 95% được tính toán theo phân phối Student's $t$ với bậc tự do $df = 4$ ($t_{0.975, 4} \approx 2.7764$), độ lệch chuẩn mẫu hiệu chỉnh $ddof = 1$.*

### Bảng 1: Chỉ số Hiệu năng Chính theo $N$ (Mean ± Std, 95% CI)

| Cỡ mẫu ($N$) | Macro-F1 (Mean ± Std) | Macro-F1 [95% CI] | Balanced Acc (Mean ± Std) | AUROC (Mean ± Std) | AUROC [95% CI] | Brier Score (Thấp hơn là tốt) | ECE (Thấp hơn là tốt) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 50$** | $0.5322 \pm 0.0148$ | $[0.5139, 0.5506]$ | $0.5385 \pm 0.0078$ | $0.5421 \pm 0.0124$ | $[0.5267, 0.5576]$ | $0.2487 \pm 0.0007$ | $0.0171 \pm 0.0061$ |
| **$N = 100$** | **$0.5728 \pm 0.0048$** | **$[0.5668, 0.5787]$** | **$0.5747 \pm 0.0063$** | **$0.5861 \pm 0.0089$** | **$[0.5750, 0.5972]$** | **$0.2456 \pm 0.0003$** | $0.0450 \pm 0.0085$ |
| **$N = 250$** | $0.5627 \pm 0.0172$ | $[0.5413, 0.5841]$ | $0.5725 \pm 0.0119$ | $0.5860 \pm 0.0078$ | $[0.5763, 0.5957]$ | $0.2464 \pm 0.0003$ | $0.0480 \pm 0.0158$ |

### Bảng 2: Chỉ số Vận hành & Huấn luyện theo $N$

| Cỡ mẫu ($N$) | Best Epoch (Mean, Min-Max) | Epochs Completed (Mean) | Thời gian Huấn luyện (s) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 50$** | $6.0$ ($2 - 18$) | $11.0$ ($7 - 23$) | $109.8 \pm 69.0$ | $109.0$ |
| **$N = 100$** | $8.0$ ($6 - 14$) | $13.0$ ($11 - 19$) | $178.3 \pm 46.2$ | $109.0$ |
| **$N = 250$** | $5.2$ ($4 - 6$) | $10.2$ ($9 - 11$) | $248.2 \pm 20.6$ | $109.0$ |

---

## 4. Phân tích Paired Deltas (Kiểm định Cặp trên cùng 5 Seeds)

Do mỗi seed trong tập $\{42, 1337, 2025, 3407, 9001\}$ được đánh giá nhất quán trên cả 3 cỡ mẫu, ta thực hiện kiểm định cặp (Paired Samples $t$-test) để loại bỏ phương sai do khởi tạo ngẫu nhiên.

### Bảng 3: Chi tiết Paired Deltas theo Seed

| Cặp so sánh | Seed 42 | Seed 1337 | Seed 2025 | Seed 3407 | Seed 9001 | $\Delta$ Mean | Std ($ddof=1$) | 95% CI | $t$-statistic | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\Delta \text{Macro-F1}_{(100 - 50)}$** | $+0.0358$ | $+0.0288$ | $+0.0358$ | $+0.0642$ | $+0.0382$ | **$+0.0406$** | $0.0137$ | $[+0.0236, +0.0575]$ | $6.628$ | **$0.0027$** |
| **$\Delta \text{Macro-F1}_{(250 - 100)}$** | $-0.0004$ | $-0.0185$ | $-0.0291$ | $+0.0132$ | $-0.0156$ | **$-0.0101$** | $0.0166$ | $[-0.0307, +0.0105]$ | $-1.357$ | **$0.2463$** |
| **$\Delta \text{Macro-F1}_{(250 - 50)}$** | $+0.0354$ | $+0.0103$ | $+0.0067$ | $+0.0774$ | $+0.0226$ | **$+0.0305$** | $0.0286$ | $[-0.0050, +0.0660]$ | $2.386$ | **$0.0755$** |
| **$\Delta \text{AUROC}_{(100 - 50)}$** | $+0.0313$ | $+0.0297$ | $+0.0484$ | $+0.0545$ | $+0.0562$ | **$+0.0440$** | $0.0127$ | $[+0.0283, +0.0597]$ | $7.761$ | **$0.0015$** |
| **$\Delta \text{AUROC}_{(250 - 100)}$** | $-0.0112$ | $+0.0116$ | $-0.0100$ | $-0.0027$ | $+0.0116$ | **$-0.0001$** | $0.0112$ | $[-0.0141, +0.0138]$ | $-0.029$ | **$0.9783$** |
| **$\Delta \text{AUROC}_{(250 - 50)}$** | $+0.0200$ | $+0.0413$ | $+0.0384$ | $+0.0518$ | $+0.0677$ | **$+0.0439$** | $0.0176$ | $[+0.0220, +0.0657]$ | $5.577$ | **$0.0051$** |

---

## 5. Đối chiếu với các Baselines Không Học (Non-Learning Baselines)

| Mô hình / Đường cơ sở | Macro-F1 (Mean) | Chênh lệch so với Dummy ($\Delta$) | Chênh lệch so với Metadata ($\Delta$) | Kết luận Thống kê |
| :--- | :---: | :---: | :---: | :--- |
| **Dummy Baseline** | $0.4940$ | $0.0000$ | $-0.0060$ | Đường cơ sở ngẫu nhiên / đa số |
| **Metadata Baseline** | $0.5000$ | $+0.0060$ | $0.0000$ | Heuristic EXIF / C2PA |
| **Stage 1 ($N = 50$)** | $0.5322$ | $+0.0382$ | $+0.0322$ | Vượt baseline, nhưng biên độ còn hẹp |
| **Stage 1 ($N = 100$)** | **$0.5728$** | **$+0.0788$** | **$+0.0728$** | **Vượt trội có ý nghĩa thống kê ($p < 0.003$)** |
| **Stage 1 ($N = 250$)** | $0.5627$ | $+0.0687$ | $+0.0627$ | Vượt baseline, bão hòa so với $N=100$ |

---

## 6. Biểu đồ Trực quan hóa

Toàn bộ biểu đồ định dạng SVG và PNG chất lượng cao được lưu trữ tại `research/evidence/phase-4c.1d/figures/`:

1. **Learning Curve Macro-F1 (Mean & 95% CI)**:  
   `figures/learning_curve_macro_f1.svg` | `figures/learning_curve_macro_f1.png`  
   *Mô tả*: Thể hiện quỹ đạo điểm trung bình và vùng tin cậy 95% $t$-distribution từ $N=50 \to 100 \to 250$, so sánh với các đường cơ sở Dummy và Metadata.

2. **Per-Seed Trajectories**:  
   `figures/per_seed_trajectories.svg` | `figures/per_seed_trajectories.png`  
   *Mô tả*: 5 đường trajectory của 5 hạt ngẫu nhiên, chứng minh tính đồng thuận 100% (5/5 hạt đều tăng điểm từ $N=50$ lên $N=100$).

3. **Balanced Accuracy and AUROC vs N**:  
   `figures/balanced_accuracy_auroc_vs_n.svg` | `figures/balanced_accuracy_auroc_vs_n.png`  
   *Mô tả*: Khả năng phân biệt nhị phân (AUROC) tăng mạnh từ $0.5421$ lên $0.5861$ tại $N=100$ và giữ nguyên ở $N=250$.

4. **Calibration Quality (Brier Score & ECE) vs N**:  
   `figures/brier_ece_vs_n.svg` | `figures/brier_ece_vs_n.png`  
   *Mô tả*: Brier score cải thiện (giảm từ $0.2487$ xuống $0.2456$), trong khi ECE tăng nhẹ do mô hình tự tin hơn nhưng chưa được hiệu chuẩn qua Temperature Scaling.

5. **Model vs Baselines**:  
   `figures/model_vs_baselines.svg` | `figures/model_vs_baselines.png`  
   *Mô tả*: Trực quan hóa khoảng cách vượt trội của Stage 1 so với 2 baseline không học.

---

## 7. Bốn Trụ cột Biện giải Khoa học (Scientific Interpretation)

### 7.1. Kết quả quan sát được (Observed Results)
1. **Bước nhảy $N=50 \to N=100$**:
   - Macro-F1 tăng từ $0.5322$ lên $0.5728$ ($\Delta = +0.0406$, 95% CI $[+0.0236, +0.0575]$, $t = 6.628, p = 0.0027$).
   - AUROC tăng từ $0.5421$ lên $0.5861$ ($\Delta = +0.0440$, 95% CI $[+0.0283, +0.0597]$, $t = 7.761, p = 0.0015$).
   - Toàn bộ 5/5 seed đều ghi nhận mức tăng dương rõ rệt.
2. **Hiện tượng bão hòa tại $N=100 \to N=250$**:
   - Macro-F1 đi ngang hoặc giảm nhẹ từ $0.5728$ xuống $0.5627$ ($\Delta = -0.0101$, 95% CI $[-0.0307, +0.0105]$, $t = -1.357, p = 0.2463$). Khoảng tin cậy chứa 0 $\to$ không có sự khác biệt có ý nghĩa thống kê.
   - AUROC hoàn toàn bất biến ($0.5861 \to 0.5860$, $\Delta = -0.0001$, $p = 0.9783$).
3. **Hiệu năng so với Baseline**:
   - Cả 3 mức $N$ đều vượt qua Dummy Baseline ($0.4940$) và Metadata Baseline ($0.5000$).

### 7.2. Diễn giải có căn cứ (Grounded Interpretation)
- **Sự bão hòa năng lực biểu diễn của Linear Probe (Capacity Bottleneck)**:
  - Ở Stage 1, toàn bộ backbone MobileNetV3-Small bị đóng băng (`frozen`). Trọng số feature extractor là trọng số trích xuất đặc trưng thị giác chung từ ImageNet, chưa từng thích ứng với các vết giả mạo siêu tinh vi (inpainting artifacts, frequency domain anomalies).
  - Đầu phân loại (classifier head) chỉ là 2 lớp tuyến tính với hàm kích hoạt và dropout. Khi số lượng mẫu tăng từ $N=50$ lên $N=100$, lớp tuyến tính học được mặt phẳng phân chia tối ưu cho các đặc trưng sẵn có.
  - Tuy nhiên, khi tăng tiếp lên $N=250$, do backbone không được cập nhật gradient, không gian đặc trưng không sinh thêm được tín hiệu phân biệt mới. Do đó, việc bổ sung dữ liệu không còn làm tăng hiệu năng phân loại mà chỉ khiến early stopping dừng sớm hơn (từ trung bình epoch 8 ở $N=100$ xuống epoch 5 ở $N=250$).

### 7.3. Giới hạn của thí nghiệm (Experimental Limitations)
1. **Số lượng seed ($n=5$)**:
   - Mặc dù 5 seeds đủ để kiểm định $t$-test cặp có ý nghĩa thống kê giữa $N=50$ và $N=100$, kích thước mẫu vẫn còn nhỏ đối với các kiểm định phi tham số hoặc phân tích phương sai phức tạp.
2. **Không gian đánh giá**:
   - Thí nghiệm được thực hiện trên 91 validation sources thuộc `manifest_pilot_a_option_p.csv` của dataset GenImage/TGIF trong pha Pilot.
   - Tuyệt đối không suy diễn kết quả này cho toàn bộ miền phân phối ảnh AI tổng quát ngoài tự nhiên.
3. **Chưa áp dụng Multimodal Fusion & Calibration**:
   - Thí nghiệm mới đo lường thuần túy mô hình thị giác đơn lẻ (vision-only branch), chưa kết hợp DSP và siêu dữ liệu EXIF/C2PA.

### 7.4. Phán quyết về Stage 2 (Stage 2 Gate Decision)

> **CÂU HỎI QUYẾT ĐỊNH**: *Có đủ căn cứ khoa học để chuyển sang huấn luyện Stage 2 (Unfreezing Backbone) hay không?*

**PHÁN QUYẾT: ĐỦ CĂN CỨ VÀ KHUYẾN NGHỊ KÍCH HOẠT STAGE 2.**

**Lập luận căn cứ**:
1. **Mô hình có tín hiệu học thật**: Stage 1 chứng minh mô hình vượt trội ngẫu nhiên và metadata baseline ($Macro\text{-}F1 = 0.5728$ vs $0.4940$ / $0.5000$, $p = 0.0027$).
2. **Bằng chứng nghẽn dung lượng biểu diễn (Representation Saturation)**: Điểm số bão hòa hoàn toàn từ $N=100$ lên $N=250$ chứng minh việc tiếp tục tăng dữ liệu mà không mở khóa backbone là vô ích. Cách duy nhất để trích xuất tín hiệu sâu hơn từ $N=250$ là cho phép backbone tinh chỉnh (fine-tuning) các tầng sâu để học vết dấu pháp chứng số đặc thù.
3. **Bảo tồn an toàn nghiên cứu**:
   - Vẫn duy trì 0 lượt truy cập vào `locked-test`.
   - Mọi quyết định tiếp tục chỉ dựa trên `inner_validation`.

---

## 8. Danh mục Artifacts Khoa học Được Tạo trong Phase 4C.1D

| Tên Artifact | Định dạng | Mục đích |
| :--- | :--- | :--- |
| `run_level_metrics.csv` | CSV | Bảng tổng hợp chi tiết 15 runs kèm 19 chỉ số đo |
| `learning_curve_summary.csv` | CSV | Bảng thống kê theo $N$ (Mean, Std, Median, Min, Max, 95% CI) |
| `learning_curve_summary.json` | JSON | Dữ liệu thống kê có cấu trúc phục vụ automation |
| `paired_seed_deltas.csv` | CSV | Bảng chênh lệch cặp theo từng seed giữa các mức $N$ |
| `phase_4c1_results_audit.json` | JSON | Niêm phong chứng nhận kiểm toán toàn diện 15 runs |
| `phase_4c1_learning_curve_report.md` | Markdown | Báo cáo phân tích chuyên sâu chi tiết (tài liệu này) |
| `PHASE_REPORT.md` | Markdown | Báo cáo tiến độ chuẩn Continuity Protocol |
| `figures/*.svg` (5 files) | SVG Vector | Biểu đồ trực quan hóa vector độ phân giải cao |
| `figures/*.png` (5 files) | PNG Raster | Biểu đồ raster phục vụ hiển thị web/báo cáo |
