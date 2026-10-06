# Phase 4C.7A — Independent Validation Preparation: Protocol Lock, Model Bindings, and Synthetic Preflight

> **Phase**: Phase 4C.7A — Independent Validation Preparation<br>
> **Status**: `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT`<br>
> **Findings status**: `NOT_MEASURED` (no real independent cohort evaluated in this phase)<br>
> **Evidence class**: `independent_validation_prespecified`<br>
> **Readiness**: `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT`<br>
> **Candidate models verified**: 5 / 5 outer-fold models bound and SHA-256 verified<br>
> **Training runs**: 0 new fits, 0 refits, 0 recalibrations, 0 hyperparameter tuning<br>
> **Real evaluations**: 0 (deferred to cohort acquisition in Phase 4C.7B)<br>
> **Task scope**: Binary classification (`authentic` = 0 vs. `ai_edited` = 1)<br>
> **Three-class note**: Evaluation of `fully_generated` (3-class generalization, RQ1) remains incomplete within the broader project and is excluded due to absent 3-class data.

---

## 1. Mục tiêu và Bối cảnh Nghiên cứu

Sau khi Phase 4C.6B chứng minh rằng kỹ thuật augmentation DSP giải cứu hiệu năng của late fusion dưới các điều kiện nén JPEG và resize trên tập phát triển 341 sources (`EXPLORATORY_JPEG75_IMPROVEMENT`), các kết quả này vẫn mang tính chất **thăm dò (exploratory)** do tập phát triển đã được tái sử dụng nhiều lần qua các phase trước.

Giai đoạn **Phase 4C.7A** thực hiện việc chuẩn bị toàn diện, tiền đăng ký (prespecified) và khóa chặt mọi quy tắc kiểm định độc lập giữa hai mô hình:
1. `visual_calibrated`: Mô hình thị giác đối chứng (MobileNetV3-Small linear probe + Temperature Scaling).
2. `late_fusion_dsp_augmented`: Mô hình kết hợp DSP augmented (Phase 4C.6B).

Việc chuẩn bị này được hoàn thành **trước khi tiếp cận bất kỳ dữ liệu mới nào**, nhằm loại bỏ hoàn toàn rủi ro p-hacking, lựa chọn mô hình thiên vị, hoặc thay đổi giả thuyết hậu nghiệm.

---

## 2. Hoàn tất Chốt Metadata Phase 4C.6B

Các điểm mâu thuẫn metadata còn tồn đọng trong Phase 4C.6B đã được đối soát và hiệu chỉnh chính xác:
1. **Fit Wall-Clock Accounting**:
   - Receipt chính thức `pilot_summary.json` ghi nhận: **1.4030s** cho 92 fits outer fold 0.
   - Receipt chính thức `full_summary.json` ghi nhận: **6.9653s** cho 368 fits tiếp nối (folds 1–4).
   - Tổng `fit_wall_seconds` hợp lệ được receipts ghi nhận là **8.3684s** cho toàn bộ 460 fits.
   - Các con số 7.02s và 14.50s trước đó là thời gian tiến trình tổng thể chưa kiểm chứng, không phản ánh fit wall-clock time của thuật toán.
2. **Process Attempts vs. Completed Fits**:
   - Xác nhận 460 logical completed fits, 0 unconverged fits ghi nhận trong receipts.
   - Tổng số lần thử cấp tiến trình (process attempts / failed attempts) được phân loại trung thực là `NOT_INDEPENDENTLY_TRACKED` do runner không lưu log attempt độc lập.
3. **Phân biệt Phần cứng và Thiết bị Thực thi**:
   - Máy trạm host sở hữu GPU `cuda (NVIDIA GeForce GTX 1650)`, nhưng toàn bộ quá trình fit mô hình và scoring diễn ra trên CPU (`scikit-learn LogisticRegression`).
   - Ghi nhận rành mạch: 0 image reads và 0 backbone forward passes (tái sử dụng đặc trưng đóng băng); việc scoring các bộ phân loại trên cached features đã thực sự diễn ra để sinh 12,276 predictions (không ghi "0 inference" chung chung).
4. **Báo cáo Độ tin cậy**:
   - Làm rõ: "Selective coverage" ($11.9% - 14.2%$) là tỷ lệ dự đoán được giữ lại ở ngưỡng $	au = 0.65$, không đồng nghĩa với probability calibration.
   - Selective accuracy ($64.0% - 66.0%$) được báo cáo độc lập.
5. **Package Provenance**:
   - Mã băm package archive (`b9930244c5...`) và snapshot manifest (`4ed15a2b...`) liên kết chặt chẽ với `package_receipt.json` của commit chức năng `97018d05...`. Việc thực thi cục bộ diễn ra trong môi trường mã nguồn đồng nhất với snapshot này.

---

## 3. Đối Soát Model Bindings và Feature Contract Thật

### 3.1. Đối Soát Mã Băm (Hash Reconciliation)
Trong phiên trước, tin nhắn báo cáo tóm tắt đã đưa ra một bộ mã băm (`e5c1d6...`) không khớp với `candidate_model_bindings.json`. Để bảo đảm tính trung thực tuyệt đối:
- Công cụ `scripts/research/audit_candidate_model_bindings.py` đã thực hiện streaming SHA-256 trực tiếp từ 5 file `fold_model.json` và 5 file `fold_receipt.json` thực tế của Phase 4C.6B trên đĩa.
- Kết quả kiểm toán bằng máy (`research/evidence/phase-4c.7a/model_bindings_audit.json` và `MODEL_BINDINGS_AUDIT.md`): Cả 5 models trên đĩa đều đạt trạng thái `AUDIT_VERIFIED`, mã băm khớp 100% với `candidate_model_bindings.json`:
  * Fold 0: `232d0994775a6e6cfdfb1bb3fd79f28557439e372e15e4760d0ffbbdba72f609` (43,264 bytes)
  * Fold 1: `c07c582322f7671f3b9f352f5d5388f5a39cec5fa11c80d8441cbb366a3fc15f` (43,264 bytes)
  * Fold 2: `8a979e35339f1ab2728f6c96d94ce2ef828ad13aad66e0e3ae8339a1be8fb324` (43,264 bytes)
  * Fold 3: `60358c2fbdfd54d96dee68a1ce43106560ee4da5792343c0d14d85beb1d5fca3` (43,264 bytes)
  * Fold 4: `efe517db09c5aecdd219eabc878b6f538e908c03b8b8c0648b2d0f6ec4356a6c` (43,264 bytes)
- **Nguồn gốc bộ hash sai**: Bộ hash `e5c1d6...` trong tin nhắn tóm tắt là văn bản bị sinh lỗi (hallucination) trong Markdown hội thoại của trợ lý AI trước đó; hoàn toàn không tồn tại trong bất kỳ tệp tin hay commit nào của repository. Các mã băm trong `candidate_model_bindings.json` và trên đĩa là single source of truth bất biến.

### 3.2. Đối Chiếu Feature Contract Chuẩn Xác
- **Visual Branch**: MobileNetV3-Small (576 chiều), weights PyTorch ImageNet V1 (`047dcff4...`).
- **DSP Branch**: 16 chiều, đối chiếu khớp 100% với thứ tự chuẩn `DSP_FEATURE_NAMES` từ `ml/training/dsp_features.py`:
  1. `fft_high_freq_energy_ratio`
  2. `fft_spectral_peak_count`
  3. `dct_high_freq_energy_ratio`
  4. `dct_corner_energy_ratio`
  5. `noise_residual_variance`
  6. `noise_kurtosis`
  7. `noise_skewness`
  8. `jpeg_block_discontinuity`
  9. `jpeg_grid_strength`
  10. `dct_mean_high_freq_energy`
  11. `dct_high_freq_iqr`
  12. `dct_energy_skewness`
  13. `noise_median_energy`
  14. `noise_energy_mad`
  15. `jpeg_periodic_grid_strength`
  16. `laplacian_variance`
- Thứ tự 16 đặc trưng trên đã được cập nhật và kiểm thử tự động trong `candidate_model_bindings.json`.

---

## 4. Sửa và Thực Thi Luận Chứng Cỡ Mẫu (Monte Carlo Simulation)

Bãi bỏ hoàn toàn các bảng công thức z-test xấp xỉ trước đó vì Macro-F1 là hàm phi tuyến của ma trận nhầm lẫn tổng thể, không thể coi là trung bình F1 của từng source.

Công cụ `scripts/research/simulate_sample_size_planning.py` đã chạy mô phỏng Monte Carlo thực sự trên cấu trúc cụm nguồn ghép cặp (paired source-cluster bootstrap 95% CI) với đúng primary estimand:
$$\overline{\text{Macro-F1}}_{\text{augmented}}(\text{jpeg\_q75}) - \overline{\text{Macro-F1}}_{\text{visual}}(\text{jpeg\_q75})$$

### 4.1. Kết Quả Mô Phỏng Kế Hoạch (SYNTHETIC PLANNING — NOT REAL PERFORMANCE)
Khảo sát trên 5 kịch bản hiệu ứng $\times$ 5 cỡ mẫu $\times$ 50 cohorts $\times$ 500 bootstrap replicates (RNG seed `20261007`):
- **Null Effect ($\Delta_{\text{true}} = 0.000$)**:
  * $N=100$: Empirical Power 2.0%, False Positive Rate $\le 2.0\%$, Margin of Error $\pm 0.0210$
  * $N=400$: Empirical Power 2.0%, False Positive Rate $\le 2.0\%$, Margin of Error $\pm 0.0105$
  * Kiểm soát chặt chẽ tỷ lệ kết luận sai dưới null ($\le \alpha/2 = 2.5\%$).
- **Subtle Effect ($\Delta_{\text{true}} \approx +0.040$)**:
  * $N=100$: Empirical Power 98.0%, Margin of Error $\pm 0.0201$
  * $N=300$: Empirical Power 100.0%, Margin of Error $\pm 0.0115$
  * $N=400$: Empirical Power 100.0%, Margin of Error $\pm 0.0098$ ($ME < 0.010$)
- **Small Effect ($\Delta_{\text{true}} \approx +0.080$)** & **Moderate Effect ($\Delta_{\text{true}} \approx +0.136$)**:
  * Toàn bộ các cỡ mẫu $N \ge 100$ đều đạt Power 100.0%.

### 4.2. Chốt Ngưỡng $N_{\text{target}} = 400$ Cặp Nguồn
- **Cỡ mẫu mục tiêu**: $N_{\text{target}} = 400$ pairs (800 ảnh độc lập).
- **Buffer thu thập**: Thu thập dự phòng 440 pairs (10% buffer) để phòng ngừa các mẫu vi phạm tiêu chí kỹ thuật trước khi niêm phong.
- **Quy tắc dừng nghiêm ngặt**: Dừng thu thập ngay khi đạt đủ 400 valid pairs đã vượt qua pre-lock validation. Tuyệt đối không tiếp tục thu thập dựa trên kết quả dự đoán trung gian.

---

## 5. Chốt Thiết Kế Cohort và Guard Đa Tầng

### 5.1. Kế Hoạch Thu Thập (`COHORT_ACQUISITION_PLAN.md`)
- **Quần thể đích**: Ảnh đời thực tự nhiên được chỉnh sửa cục bộ bằng inpainting hiện đại; cân bằng giữa người, đồ vật, phong cảnh và kiến trúc.
- **Nguồn ảnh gốc**: Unsplash (Unsplash License), Flickr (CC-BY / CC0), COCO 2017 val (CC-BY 4.0).
- **Công cụ inpainting & Quotas**:
  * Stable Diffusion 2 Inpainting: 40% (160 pairs) — direct test của công cụ phát triển.
  * SDXL Inpainting: 35% (140 pairs) — out-of-distribution architecture.
  * Adobe Firefly / Proprietary Commercial Inpainting: 25% (100 pairs) — real-world commercial diffusion.
- **Chuẩn hóa kích thước**: Toàn bộ ảnh xuất chuẩn $512 \times 512$ PNG RGB lossless; giải quyết triệt để shortcut về codec hay độ phân giải giữa authentic và ai_edited.

### 5.2. Source-Disjoint Guard Đa Tầng
Nâng cấp `ml/evaluation/independent_cohort.py` với cơ chế kiểm tra đa tầng:
1. `source_id`: Không trùng lặp với 684 source IDs lịch sử.
2. `image_sha256`: Không trùng lặp với 6,156 mã băm SHA-256 ảnh lịch sử.
3. `origin_id`: Không trùng lặp URL hoặc file gốc lịch sử.
4. `resolution_parity`: Kiểm tra độ phân giải và kênh màu khớp nhau giữa authentic và ai_edited trong từng cặp.

---

## 6. Xác Minh Evaluator Pipeline Xử Lý Ảnh Thật

### 6.1. Pipeline Hoàn Chỉnh (`ml/evaluation/independent_pipeline.py`)
Triển khai chuỗi xử lý đầu-cuối:
$$\text{PIL Image} \xrightarrow{\text{Transforms}} \text{RGB Array} \xrightarrow{\text{Backbone / DSP}} \text{Features} \xrightarrow{\text{Models}} \text{Predictions} \xrightarrow{\text{Metrics}} \text{Bootstrap}$$
- Tuân thủ nghiêm ngặt 6 điều kiện biến đổi ảnh chuẩn tắc.
- Mỗi ảnh được suy luận độc lập; tuyệt đối không dùng ground-truth, source_id hoặc ảnh đối ứng làm đặc trưng.
- Đánh giá trên toàn bộ 5 outer models; tính trung bình số học Macro-F1; chạy paired bootstrap 10,000 replicates.

### 6.2. CLI Runner Mở Rộng (`scripts/research/run_independent_validation.py`)
- Hỗ trợ đầy đủ cờ `--image-preflight` kiểm tra pipeline trên synthetic image fixtures.
- Hỗ trợ cờ `--evaluate` cho việc đánh giá cohort thực tế khi cohort được bàn giao trong Phase 4C.7B.

### 6.3. Bộ Test Suites Mở Rộng
Bộ test `ml/tests/test_independent_validation_preparation.py` đã mở rộng lên **21 bài test** (toàn bộ 21/21 PASS):
- 17 hermetic CI tests (chạy 100% tự lập không cần artifact lớn trên CI).
- 4 artifact-gated tests (xác thực trực tiếp mã băm 5 models và 684 historical sources trên đĩa).

---

## 7. Deliverables Gói Evidence Phase 4C.7A Hoàn Thiện

1. `independent_validation_protocol.yaml`: Protocol kiểm định độc lập v1.0.
2. `PROTOCOL_AMENDMENT_V1.1.md`: Phụ lục sửa đổi protocol v1.1 ghi nhận feature order chuẩn và $N_{\text{target}}=400$.
3. `candidate_model_bindings.json`: 5 outer-fold models bindings kèm 16 DSP feature order chuẩn.
4. `model_bindings_audit.json` & `MODEL_BINDINGS_AUDIT.md`: Báo cáo đối soát mã băm 5 models bằng máy.
5. `sample_size_planning_results.json` & `SAMPLE_SIZE_JUSTIFICATION.md`: Kết quả mô phỏng Monte Carlo máy sinh.
6. `COHORT_ACQUISITION_PLAN.md`: Kế hoạch thu thập cohort độc lập chi tiết.
7. `EXECUTION_GUIDE.md`: Hướng dẫn vận hành CLI thực tế.
8. `readiness.json`: Trạng thái sẵn sàng kỹ thuật 6 khía cạnh.
9. `PHASE_REPORT.md`: Báo cáo tổng kết toàn diện Phase 4C.7A.

---

## 8. Kết Luận và Giới Hạn

Phase 4C.7A đã hoàn thành 100% công tác chuẩn bị và đối soát:
- **Model bindings**: AUDIT_VERIFIED (5/5).
- **Sample size planning**: Monte Carlo hoàn thành ($N_{\text{target}} = 400$).
- **Cohort design**: Finalized.
- **Evaluator pipeline**: Verified with synthetic image fixtures.
- **Independent cohort acquisition**: `NOT_ACQUIRED` (thu thập thực tế thuộc Phase 4C.7B).
- **Independent performance**: `NOT_MEASURED`.
- **Web UI**: Giữ nguyên trạng thái trung thực `uncertain` / `Model not installed`.
