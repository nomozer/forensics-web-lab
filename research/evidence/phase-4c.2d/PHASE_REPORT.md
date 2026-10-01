# Phase 4C.2D — Final Model-Selection Gate and PR Closure Report

> **Giai đoạn**: Phase 4C.2D<br>
> **Mục tiêu**: Hoàn tất quyết định lựa chọn mô hình Stage 1 vs Stage 2 một cách trung thực khoa học; kiểm toán lineage và parity nhị phân 30 runs; chuẩn hóa ngữ nghĩa hiệu chuẩn; đóng băng protocol chính cho locked-test; chuẩn bị PR closure vào main.<br>
> **Branch**: `research/phase-4c2-finetuning`<br>
> **Evaluation Partition**: `inner_validation` ($91$ unique sources, $182$ balanced samples).<br>
> **Niêm phong Locked-Test**: $0$ truy cập, $0$ evaluation, giữ nguyên trạng thái niêm phong tuyệt đối.<br>
> **Huấn luyện mới**: $0$ lượt chạy (0 GPU calls).<br>
> **Sửa đổi raw artifacts**: $0$ tệp thô bị sửa đổi (raw Colab/local outputs 100% byte-identical).<br>
> **Quyết định Mô hình**: **`stage1_frozen_backbone_linear_probe`** (Protocol được chọn)<br>
> **Phán quyết cuối cùng**: **`READY_FOR_PR_REVIEW`**

---

## 1. Tóm tắt Quyết định Lựa chọn Mô hình (Model-Selection Decision)

Căn cứ trên toàn bộ bằng chứng thực nghiệm đối chứng ghép cặp 15 runs Stage 1 (Frozen Backbone Linear Probe) và 15 runs Stage 2 (Pre-Registered Partial Fine-Tuning) trên phân vùng `inner_validation`:

1. **Giao thức được chọn (Selected Primary Protocol)**: **`stage1_frozen_backbone_linear_probe`**
2. **Giao thức thứ cấp / không chọn (Secondary Protocol)**: **`stage2_preregistered_partial_finetuning`**
3. **Tính chất của Quyết định**: **Post-analysis Model-Selection Decision** (`preregistered_rule_present = false`, `post_hoc_decision = true`).

### Cơ sở Khoa học:
- **Không đạt ý nghĩa thống kê**: Mọi khoảng tin cậy 95% Student's t đều cắt 0 (N50: $[-0.0008, +0.0196]$, N100: $[-0.0202, +0.0279]$, N250: $[-0.0146, +0.0275]$); mọi giá trị $p$ (thô và hiệu chỉnh Holm-Bonferroni, parametric paired t lẫn exact permutation) đều $> 0.05$.
- **Tính kinh tế tham số (Parameter Efficiency)**: Stage 1 chỉ cập nhật $148,226$ tham số ở classification head, bảo tồn toàn vẹn đặc trưng MobileNetV3. Stage 2 mở thêm block `features.12` cập nhật $204,674$ tham số (+38.1% tham số) nhưng chỉ mang lại mức tăng trung bình không đáng kể $+0.0066$ Macro-F1.
- **Độ ổn định phương sai**: Tại $N=100$, Stage 2 kém hơn Stage 1 ở 3/5 seeds. Độ lệch chuẩn Macro-F1 giữa các seed của Stage 2 là 0.019159, cao hơn Stage 1 là 0.004802 (xấp xỉ 3.99 lần), cho thấy độ biến thiên theo seed lớn hơn trong thí nghiệm này. Paired-delta SD là 0.019371. Đây là bằng chứng phát triển mô hình mang tính khám phá (exploratory), không phải kết luận confirmatory về variance do chỉ có 5 seeds.
- **Nguyên tắc khoa học thận trọng (Ockham's Razor)**: Khi một can thiệp tinh chỉnh phức tạp hơn không chứng minh được sự vượt trội rõ rệt và có ý nghĩa thống kê so với baseline đơn giản hơn, mô hình đơn giản và ít tham số hơn (Stage 1) được giữ làm mô hình chính thức.

---

## 2. Kiểm toán Lineage Stage 1 (Stage 1 Lineage Gate)

Tương tự như kiểm toán Phase 4C.2C.1 cho Stage 2, toàn bộ 15 runs thô của Stage 1 tại `extracted_15_runs` đã được kiểm toán đối soát qua 7 nguồn dữ liệu nhị phân:

| Run ID | Cohort $N$ | Seed | Receipt Best Epoch | Checkpoint Epoch | Receipt Best Val F1 | Metrics.json Macro-F1 | Recomputed Pred F1 | Checkpoint SHA-256 (8 ký tự) | Parity Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phase4c1-stage1-n50-seed42` | 50 | 42 | 18 | 18 | 0.541477 | 0.541477 | 0.541477 | `b5fa53bf` | PARITY_VERIFIED |
| `phase4c1-stage1-n50-seed1337` | 50 | 1337 | 5 | 5 | 0.537064 | 0.537064 | 0.537064 | `6a709a64` | PARITY_VERIFIED |
| `phase4c1-stage1-n50-seed2025` | 50 | 2025 | 2 | 2 | 0.538238 | 0.538238 | 0.538238 | `2827a79f` | PARITY_VERIFIED |
| `phase4c1-stage1-n50-seed3407` | 50 | 3407 | 3 | 3 | 0.505934 | 0.505934 | 0.505934 | `575240ac` | PARITY_VERIFIED |
| `phase4c1-stage1-n50-seed9001` | 50 | 9001 | 2 | 2 | 0.538406 | 0.538406 | 0.538406 | `1f74b9df` | PARITY_VERIFIED |
| `phase4c1-stage1-n100-seed42` | 100 | 42 | 7 | 7 | 0.577313 | 0.577313 | 0.577313 | `65617b42` | PARITY_VERIFIED |
| `phase4c1-stage1-n100-seed1337` | 100 | 1337 | 14 | 14 | 0.565816 | 0.565816 | 0.565816 | `5a9f04bc` | PARITY_VERIFIED |
| `phase4c1-stage1-n100-seed2025` | 100 | 2025 | 6 | 6 | 0.574030 | 0.574030 | 0.574030 | `ce3b5e0b` | PARITY_VERIFIED |
| `phase4c1-stage1-n100-seed3407` | 100 | 3407 | 7 | 7 | 0.570131 | 0.570131 | 0.570131 | `78dd4949` | PARITY_VERIFIED |
| `phase4c1-stage1-n100-seed9001` | 100 | 9001 | 6 | 6 | 0.576604 | 0.576604 | 0.576604 | `24da001f` | PARITY_VERIFIED |
| `phase4c1-stage1-n250-seed42` | 250 | 42 | 6 | 6 | 0.576910 | 0.576910 | 0.576910 | `c941f42e` | PARITY_VERIFIED |
| `phase4c1-stage1-n250-seed1337` | 250 | 1337 | 5 | 5 | 0.547321 | 0.547321 | 0.547321 | `69e706f9` | PARITY_VERIFIED |
| `phase4c1-stage1-n250-seed2025` | 250 | 2025 | 5 | 5 | 0.544930 | 0.544930 | 0.544930 | `92ce5ee9` | PARITY_VERIFIED |
| `phase4c1-stage1-n250-seed3407` | 250 | 3407 | 4 | 4 | 0.583371 | 0.583371 | 0.583371 | `4897821e` | PARITY_VERIFIED |
| `phase4c1-stage1-n250-seed9001` | 250 | 9001 | 6 | 6 | 0.561039 | 0.561039 | 0.561039 | `5f0f8803` | PARITY_VERIFIED |

### Xác nhận Hợp đồng Runner Stage 1:
- Bằng chứng dòng mã tại `ml/training/run_phase_4c1.py` (dòng 586–594) xác nhận rằng sau khi kết thúc vòng lặp huấn luyện, runner Stage 1 **luôn nạp lại `best_checkpoint.pt`** vào mô hình trước khi đánh giá `final_metrics` và sinh `predictions.json`.
- Toàn bộ 15/15 runs Stage 1 đạt **100% PARITY_VERIFIED** giữa receipt, metrics.json, predictions.json, epoch_history.json và best_checkpoint metadata.
- Số liệu Stage 1 trong `paired_run_metrics.csv` khớp 100% với các artifact thô Stage 1 từ Phase 4C.1D và không có bất kỳ thay đổi ngầm nào.

---

## 3. Xác minh Parity Chéo Số học (Cross-Artifact Numerical Parity)

- **Dung sai kiểm tra (Tolerance)**: $10^{-6}$ (chế độ fail-closed).
- **Kết quả đối chiếu 30 runs** (15 Stage 1 + 15 Stage 2):
  - Stage 1 Raw vs `paired_run_metrics.csv`: **15/15 MATCH** ($< 10^{-6}$)
  - Stage 2 Raw vs `paired_run_metrics.csv`: **15/15 MATCH** ($< 10^{-6}$)
  - `paired_run_metrics.csv` vs `paired_summary.json`: **MATCH** ($< 10^{-6}$)
  - `primary_statistical_tests.json` vs `paired_summary.json`: **MATCH** ($< 10^{-6}$)
  - Mọi số liệu trong các bảng báo cáo Markdown đều được sinh từ các tệp JSON/CSV chuẩn tắc.

---

## 4. Chuẩn hóa Ngữ nghĩa Hiệu chuẩn (Calibration Semantics)

Trong toàn bộ tài liệu nghiên cứu của repository, cách diễn đạt tuyệt đối về hiệu chuẩn đã được chuẩn hóa lại theo nguyên tắc khiêm tốn khoa học:

> *"Không quan sát thấy xu hướng overconfidence trung bình trên inner-validation; signed confidence calibration gap âm gợi ý xu hướng underconfidence nhẹ. Đây là bằng chứng phát triển mô hình, không phải kết luận confirmatory."*

- Phân tách toán học triệt để giữa:
  1. **Calibration-in-the-large**: $\text{mean}(p_{\text{positive}} - y) \approx \pm 0.001 \to \pm 0.006$ (độ lệch tỷ lệ lớp dương xấp xỉ 0).
  2. **Signed confidence calibration gap**: $\text{mean}(\text{confidence} - \text{correctness}) \approx -0.01 \to -0.05$ (độ tự tin trung bình thấp hơn độ chính xác thực tế, biểu hiện tính thận trọng/dè dặt).

---

## 5. Tuyên bố Sẵn sàng cho Locked-Test (Locked-Test Readiness)

- **Trạng thái niêm phong**: Tập kiểm thử khóa (`locked_test`) **tiếp tục được niêm phong 100%** ($0$ lượt truy cập, $0$ mẫu rò rỉ).
- **Quy tắc tiếp theo**: Quá trình mở niêm phong đánh giá locked-test chỉ được thực hiện trong một phase chuyên biệt độc lập sau khi PR này được duyệt và hòa vào nhánh chính.

---

## 6. Bảng Kê Ràng buộc Bất biến (Invariants Accounting)

| Chỉ số / Ràng buộc Kỹ thuật | Giá trị Ghi nhận | Đánh giá Tuân thủ |
| :--- | :---: | :---: |
| **Raw artifacts modified** | **0** | Đạt chuẩn bất biến |
| **New training runs** | **0** | Đạt chuẩn bất biến |
| **GPU inference calls** | **0** | Đạt chuẩn bất biến |
| **Locked-test evaluations** | **0** | Niêm phong tuyệt đối |
| **Stage 1 raw runs parity** | **15/15 (100%)** | Parity verified |
| **Stage 2 raw runs parity** | **15/15 (100%)** | Parity verified |
| **Cross-artifact numerical parity** | **100% (< 1e-6)** | Parity verified |
| **Absolute Windows paths in Git evidence** | **0** | Đạt chuẩn di động |
| **Final Model Selected** | `stage1_frozen_backbone_linear_probe` | Nhất quán khoa học |
| **Final Verdict** | **`READY_FOR_PR_REVIEW`** | **HOÀN THÀNH** |
