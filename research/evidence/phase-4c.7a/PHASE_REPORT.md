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
> **Evidence classification**: cited architecture/calibration/dataset claims are `external-source`; model-binding, simulation and test results are `internal-empirical`; bibliography keys resolve through `docs/references.bib`.

---

## 1. Mục tiêu và Bối cảnh Nghiên cứu

Sau khi Phase 4C.6B chứng minh rằng kỹ thuật augmentation DSP giải cứu hiệu năng của late fusion dưới các điều kiện nén JPEG và resize trên tập phát triển 341 sources (`EXPLORATORY_JPEG75_IMPROVEMENT`), các kết quả này vẫn mang tính chất **thăm dò (exploratory)** do tập phát triển đã được tái sử dụng nhiều lần qua các phase trước.

Giai đoạn **Phase 4C.7A** thực hiện việc chuẩn bị toàn diện, tiền đăng ký (prespecified) và khóa chặt mọi quy tắc kiểm định độc lập giữa hai mô hình:
1. `visual_calibrated`: Mô hình thị giác đối chứng (MobileNetV3-Small `[@howard2019mobilenetv3; @torchvisionMobilenetV3Small]` linear probe + Temperature Scaling `[@guo2017calibration]`). Các nguồn ngoài hỗ trợ kiến trúc/phương pháp, không chứng minh kết quả nội bộ.
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

### 3.1. Đối Soát Kích Thước và Mã Băm Mô Hình (Model Sizes & Hash Reconciliation)
Để đảm bảo tính trung thực kỹ thuật và cơ sở bằng chứng khách quan:
- Công cụ `scripts/research/audit_candidate_model_bindings.py` đã thực hiện streaming SHA-256 và đo kích thước byte thực tế từ 5 file `fold_model.json` của Phase 4C.6B trên đĩa.
- Kết quả kiểm toán bằng máy (`research/evidence/phase-4c.7a/model_bindings_audit.json` và `MODEL_BINDINGS_AUDIT.md`): Cả 5 models trên đĩa đều đạt trạng thái `AUDIT_VERIFIED`, mã băm và kích thước thực tế khớp chính xác với `candidate_model_bindings.json`:
  * Fold 0: `232d0994775a6e6cfdfb1bb3fd79f28557439e372e15e4760d0ffbbdba72f609` (58,418 bytes / 58418 bytes)
  * Fold 1: `c07c582322f7671f3b9f352f5d5388f5a39cec5fa11c80d8441cbb366a3fc15f` (58,030 bytes / 58030 bytes)
  * Fold 2: `8a979e35339f1ab2728f6c96d94ce2ef828ad13aad66e0e3ae8339a1be8fb324` (58,465 bytes / 58465 bytes)
  * Fold 3: `60358c2fbdfd54d96dee68a1ce43106560ee4da5792343c0d14d85beb1d5fca3` (58,455 bytes / 58455 bytes)
  * Fold 4: `efe517db09c5aecdd219eabc878b6f538e908c03b8b8c0648b2d0f6ec4356a6c` (58,390 bytes / 58390 bytes)
- **Đối soát mâu thuẫn báo cáo**: Bản thảo trước đó của báo cáo ghi sai kích thước cả 5 mô hình là 43.264 bytes placeholder (sao chép số liệu ước tính tạm thời). Kích thước file thực tế dao động từ 58,030 đến 58,465 bytes do sự khác biệt nhỏ về số chữ số thập phân trong vector trọng số được serialize JSON giữa các fold.

### 3.2. Đối Chiếu Feature Contract Chuẩn Xác (DSP Feature Contract)
- **Visual Branch**: MobileNetV3-Small (576 chiều), weights PyTorch ImageNet V1 (`047dcff4...`). Architecture/enum metadata: `[@howard2019mobilenetv3; @torchvisionMobilenetV3Small]`; hash và feature dimension là binding nội bộ từ artifact.
- **DSP Branch**: 16 chiều, đối chiếu khớp 100% với định nghĩa và thứ tự chuẩn `DSP_FEATURE_NAMES` từ `ml/training/dsp_features.py`:
  1. `fft_high_freq_energy_ratio`
  2. `fft_spectral_peak_count`
  3. `fft_radial_anomaly_score`
  4. `fft_spectral_decay_slope`
  5. `dct_mean_high_freq_energy`
  6. `dct_ac_energy_variance`
  7. `dct_anomaly_score`
  8. `noise_global_level`
  9. `noise_max_to_min_ratio`
  10. `noise_std_of_variances`
  11. `noise_coef_variation`
  12. `noise_inconsistency_score`
  13. `jpeg_boundary_difference_ratio`
  14. `jpeg_periodic_grid_strength`
  15. `jpeg_artifact_score`
  16. `laplacian_variance`
- **Đối soát mâu thuẫn báo cáo**: Bản thảo báo cáo trước đó đã liệt kê nhầm 16 tên đặc trưng từ bản phác thảo thử nghiệm ablation cũ. Tuy nhiên, mã nguồn trích xuất thực tế (`dsp_features.py`), cache receipts và `candidate_model_bindings.json` luôn thực thi đúng 16 đặc trưng chuẩn tắc nêu trên theo đúng thứ tự. Cơ chế kiểm tra thứ tự đặc trưng chặt chẽ đã được bổ sung vào model loader (`load_candidate_models`) và bộ regression test.

---

## 4. Đối Soát Luận Chứng Cỡ Mẫu (Monte Carlo Simulation)

Bãi bỏ hoàn toàn các bảng công thức z-test xấp xỉ trước đó vì Macro-F1 là hàm phi tuyến của ma trận nhầm lẫn tổng thể, không thể coi là trung bình F1 của từng source.

Công cụ `scripts/research/simulate_sample_size_planning.py` đã chạy mô phỏng Monte Carlo trên cấu trúc cụm nguồn ghép cặp (paired source-cluster bootstrap 95% CI) với đúng primary estimand:
$$\Delta \overline{\text{Macro-F1}} = \overline{\text{Macro-F1}}_{\text{augmented}}(\text{jpeg\_q75}) - \overline{\text{Macro-F1}}_{\text{visual}}(\text{jpeg\_q75})$$

### 4.1. Kết Quả Mô Phỏng Kế Hoạch (SYNTHETIC PLANNING — NOT REAL PERFORMANCE)
Khảo sát trên 5 kịch bản hiệu ứng $\times$ 5 cỡ mẫu $\times$ 50 cohorts $\times$ 500 bootstrap replicates (RNG seed `20261007`):
- **Null Effect ($\delta = 0.000$)**:
  * $N=100$: Tỷ lệ rejection 4.0% (2/50 cohorts, Wilson 95% CI: [1.1%, 13.5%]), Mean 95% CI: [-0.0174, +0.0202], Margin of Error $\pm 0.0188$
  * $N=200$: Tỷ lệ rejection 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]), Mean 95% CI: [-0.0147, +0.0120], Margin of Error $\pm 0.0133$
  * $N=300$: Tỷ lệ rejection 4.0% (2/50 cohorts, Wilson 95% CI: [1.1%, 13.5%]), Mean 95% CI: [-0.0095, +0.0124], Margin of Error $\pm 0.0110$
  * $N=400$: Tỷ lệ rejection 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]), Mean 95% CI: [-0.0101, +0.0088], Margin of Error $\pm 0.0095$
  * $N=500$: Tỷ lệ rejection 2.0% (1/50 cohorts, Wilson 95% CI: [0.4%, 10.5%]), Mean 95% CI: [-0.0081, +0.0091], Margin of Error $\pm 0.0086$
  * *Nhận định*: Với $M=50$ cohorts, tỷ lệ kết luận sai dưới null dao động $2.0\% - 4.0\%$, phù hợp với kỳ vọng danh định $\alpha/2 = 2.5\%$. Không đưa ra nhận định sai lệch rằng null rejection luôn $\le 2.0\%$.
- **Subtle Effect ($\delta \approx +0.025$, đo được Mean $\Delta \approx +0.028$)**:
  * $N=100$: Power 82.0% (41/50 cohorts, Wilson 95% CI: [69.2%, 90.2%]), Margin of Error $\pm 0.0193$
  * $N=200$: Power 96.0% (48/50 cohorts, Wilson 95% CI: [86.5%, 98.9%]), Margin of Error $\pm 0.0139$
  * $N=300$: Power 98.0% (49/50 cohorts, Wilson 95% CI: [89.5%, 99.6%]), Margin of Error $\pm 0.0114$
  * $N=400$: Power 100.0% (50/50 cohorts, Wilson 95% CI: [92.9%, 100.0%]), Margin of Error $\pm 0.0098$ ($ME < 0.010$)
  * $N=500$: Power 100.0% (50/50 cohorts, Wilson 95% CI: [92.9%, 100.0%]), Margin of Error $\pm 0.0088$
  * *Giới hạn thống kê*: Kết quả 50/50 cohorts đạt ý nghĩa tại $N=400$ là ước lượng điểm Monte Carlo trên $M=50$, không đảm bảo 100% power trong thực tế; khoảng tin cậy Wilson $[92.9\%, 100.0\%]$ thể hiện độ biến động của mô phỏng.
- **Small Effect ($\delta \approx +0.050$), Moderate Effect ($\delta \approx +0.100$) và Large Effect ($\delta \approx +0.140$)**:
  * Đạt 50/50 cohorts (100.0%, Wilson 95% CI: [92.9%, 100.0%]) tại mọi $N \ge 100$.

### 4.2. Căn Cứ Lựa Chọn Cỡ Mẫu $N_{\text{target}} = 400$ Cặp Nguồn
- **Cỡ mẫu mục tiêu khóa chặt**: $N_{\text{target}} = 400$ pairs (800 ảnh độc lập).
- **Căn cứ thiết kế**: Ngưỡng $N=400$ là điểm tối thiểu mà Margin of Error co hẹp xuống dưới $\pm 0.010$ (đạt $\pm 0.0095$ dưới null và $\pm 0.0098$ dưới subtle effect), đảm bảo độ phân giải thống kê đủ để phân biệt hiệu ứng nhỏ mà không lãng phí chi phí thu thập.
- **Buffer thu thập**: Thu thập 440 pairs (10% buffer dự phòng lỗi decode, file hỏng, vi phạm guard).
- **Quy tắc dừng nghiêm ngặt**: Dừng thu thập ngay khi đạt đủ 400 valid pairs đã khóa; không đánh giá kết quả giữa chừng để quyết định dừng hay tiếp tục.

---

## 5. Chốt Thiết Kế Cohort Duy Nhất và Guard Đa Tầng

Tất cả các tài liệu protocol, amendment, cohort plan và execution guide được thống nhất theo cấu hình chuẩn `research/evidence/phase-4c.7a/cohort_specification.json`:

### 5.1. Kế Hoạch Thu Thập (`COHORT_ACQUISITION_PLAN.md`)
- **Quần thể đích**: Ảnh đời thực tự nhiên được chỉnh sửa cục bộ bằng inpainting hiện đại; cân bằng giữa người, đồ vật, phong cảnh và kiến trúc.
- **Nguồn ảnh authentic & Quotas**:
  * Research Field Collection (ảnh tự chụp độc quyền): **40% (160 pairs)**.
  * COCO 2017 Dataset (ngoài Option P): **35% (140 pairs)**. Ghi chú license correction: không dùng blanket `CC BY 4.0` thay cho provenance; mỗi origin phải giữ author/license record riêng như intake Phase 4C.7B.
  * Unsplash Verified (ảnh trước 2022): **25% (100 pairs)**.
- **Công cụ inpainting & Quotas**:
  * Stable Diffusion 2 Inpainting (`stabilityai/stable-diffusion-2-inpainting`): **40% (160 pairs)**.
  * SDXL Inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`): **40% (160 pairs)**.
  * Adobe Firefly / Commercial Generative Fill: **20% (80 pairs)**.
- **Ma trận phân bổ trực giao (Orthogonal Allocation)**:
  * Research Field: 64 SD2, 64 SDXL, 32 Firefly (= 160 pairs)
  * COCO 2017: 56 SD2, 56 SDXL, 28 Firefly (= 140 pairs)
  * Unsplash: 40 SD2, 40 SDXL, 20 Firefly (= 100 pairs)
  * Tổng cộng: 160 SD2, 160 SDXL, 80 Firefly (= 400 pairs)
- **Chuẩn hóa kích thước & Giới hạn kiểm soát**:
  * Toàn bộ ảnh xuất chuẩn $512 \times 512$ PNG RGB lossless để bảo đảm độ phân giải và định dạng đồng nhất giữa authentic và ai_edited.
  * *Làm rõ*: Chuẩn hóa PNG $512 \times 512$ là kiểm soát container đầu ra, không tuyên bố loại bỏ toàn bộ dấu vết nén JPEG từ cảm biến gốc trong quá khứ hoặc mọi shortcut tiềm ẩn.

### 5.2. Source-Disjoint Guard Đa Tầng
Validator `ml/evaluation/independent_cohort.py` kiểm tra nghiêm ngặt 4 tầng bảo vệ:
1. `source_id`: Không trùng lặp với 684 source IDs Option P lịch sử.
2. `image_sha256`: Không trùng lặp với 6,156 mã băm SHA-256 ảnh lịch sử.
3. `origin_id`: Không trùng lặp URL hoặc instance ID Flickr/COCO lịch sử.
4. `resolution_parity`: Kiểm tra độ phân giải và kênh màu khớp nhau tuyệt đối giữa authentic và ai_edited trong từng cặp.

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
Bộ test `ml/tests/test_independent_validation_preparation.py` đã mở rộng lên **25 bài test** (toàn bộ 25/25 PASS):
- 21 hermetic CI tests (chạy 100% tự lập không cần artifact lớn trên CI, bao gồm các regression tests chặn lệch kích thước model, lệch feature names/order, lệch cohort quotas).
- 4 artifact-gated tests (xác thực trực tiếp mã băm 5 models và 684 historical sources trên đĩa).

---

## 7. Deliverables Gói Evidence Phase 4C.7A Hoàn Thiện

1. `independent_validation_protocol.yaml`: Protocol kiểm định độc lập v1.0.
2. `PROTOCOL_AMENDMENT_V1.1.md`: Phụ lục sửa đổi protocol v1.1 ghi nhận feature order chuẩn và $N_{\text{target}}=400$.
3. `candidate_model_bindings.json`: 5 outer-fold models bindings kèm 16 DSP feature order chuẩn.
4. `model_bindings_audit.json` & `MODEL_BINDINGS_AUDIT.md`: Báo cáo đối soát mã băm và kích thước 5 models bằng máy.
5. `sample_size_planning_results.json` & `SAMPLE_SIZE_JUSTIFICATION.md`: Kết quả mô phỏng Monte Carlo máy sinh (kèm Wilson 95% CIs).
6. `cohort_specification.json`: Cấu hình máy canonical chuẩn tắc cho cohort độc lập (N=400, quotas, orthogonal matrix).
7. `COHORT_ACQUISITION_PLAN.md`: Kế hoạch thu thập cohort độc lập chi tiết kèm ma trận phân bổ trực giao.
8. `EXECUTION_GUIDE.md`: Hướng dẫn vận hành CLI thực tế.
9. `readiness.json`: Trạng thái sẵn sàng kỹ thuật 6 khía cạnh.
10. `PHASE_REPORT.md`: Báo cáo tổng kết toàn diện Phase 4C.7A.

---

## 8. Kết Luận và Giới Hạn

Phase 4C.7A đã hoàn thành 100% công tác chuẩn bị và đối soát:
- **Model bindings**: AUDIT_VERIFIED (5/5 mô hình, SHA-256 và kích thước khớp đĩa).
- **DSP Feature Contract**: Khóa chặt 16 đặc trưng chuẩn tắc từ `dsp_features.py` với guard kiểm tra thứ tự.
- **Sample size planning**: Monte Carlo hoàn thành ($N_{\text{target}} = 400$, đạt Margin of Error $< \pm 0.010$).
- **Cohort design**: Finalized với ma trận phân bổ trực giao triệt tiêu confounding.
- **Evaluator pipeline**: Verified with synthetic image fixtures.
- **Independent cohort acquisition**: `NOT_ACQUIRED` (thu thập thực tế thuộc Phase 4C.7B).
- **Independent performance**: `NOT_MEASURED`.
- **Web UI**: Giữ nguyên trạng thái trung thực `uncertain` / `Model not installed`.

---

## 9. Bảng Đối Soát Mâu Thuẫn (Discrepancy Reconciliation Table)

Bảng đối soát chi tiết giữa các nguồn tài liệu, giá trị máy sinh và biện pháp xử lý:

| Hạng mục mâu thuẫn | Giá trị báo cáo cũ | Giá trị máy chuẩn tắc | Bằng chứng quyết định | Phân loại lỗi | Biện pháp xử lý |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Kích thước mô hình (Model sizes)** | Ghi đồng nhất 43.264 bytes placeholder cho cả 5 fold models | Fold 0: 58,418 B<br>Fold 1: 58,030 B<br>Fold 2: 58,465 B<br>Fold 3: 58,455 B<br>Fold 4: 58,390 B | File `fold_model.json` thực tế trên đĩa và `model_bindings_audit.json` | Lỗi tài liệu báo cáo (sao chép số liệu ước tính tạm thời) | Cập nhật bảng kích thước chính xác từ đĩa vào `PHASE_REPORT.md`. Thêm regression test `test_report_model_sizes_and_hashes_match_audit`. |
| **B. Hợp đồng đặc trưng DSP (DSP feature contract)** | Danh sách 16 đặc trưng từ bản nháp thử nghiệm cũ (chứa `dct_corner_energy_ratio`, `noise_residual_variance`, `jpeg_block_discontinuity`) | 16 đặc trưng chuẩn tắc từ `DSP_FEATURE_NAMES` trong `dsp_features.py` (bắt đầu bằng `fft_high_freq_energy_ratio`, kết thúc bằng `laplacian_variance`) | `DSP_FEATURE_NAMES` trong `dsp_features.py`, cache receipts và `candidate_model_bindings.json` | Lỗi tài liệu báo cáo (code runtime luôn đúng) | Cập nhật danh sách 16 đặc trưng chuẩn vào báo cáo. Bổ sung guard kiểm tra thứ tự trong `load_candidate_models()` và regression test. |
| **C. Luận chứng cỡ mẫu (Sample size planning)** | Báo cáo ghi null rejection $\le 2\%$, power subtle effect $\approx 68\%$, ME và CI width lệch JSON | Null rejection: 2.0% - 4.0% (Wilson CIs tương ứng); Power subtle ($N=400$): 100.0% (50/50, Wilson [92.9%, 100.0%]); ME: $\pm 0.0098$ | File máy sinh `sample_size_planning_results.json` (Seed 20261007) | Lỗi diễn giải và template văn bản trong script sinh báo cáo | Bổ sung hàm tính khoảng tin cậy Wilson; render tự động bảng và nhận xét từ JSON; diễn giải rõ 50/50 là ước lượng điểm mô phỏng không phải cam kết đời thực. |
| **D. Thiết kế Cohort (Cohort Quotas & Sources)** | Quotas inpainting ghi 40/35/25%; nguồn authentic chỉ ghi Unsplash, Flickr, COCO | Quotas inpainting: 40% SD2, 40% SDXL, 20% Firefly; Nguồn authentic: 40% Research Field (160), 35% COCO (140), 25% Unsplash (100) | `COHORT_ACQUISITION_PLAN.md` và `cohort_specification.json` | Lỗi tài liệu báo cáo (nhầm lẫn giữa tỷ lệ nguồn ảnh và tỷ lệ công cụ) | Ban hành cấu hình máy `cohort_specification.json` làm nguồn chuẩn duy nhất; thiết lập ma trận phân bổ trực giao; bổ sung regression test kiểm tra quotas. |

---

## 10. Bảng Kiểm Tra Tính Khả Thi Thu Thập và Yêu Cầu Handoff

Bảng đánh giá điều kiện tiên quyết trước khi tiến hành thu thập cohort độc lập:

| Tài nguyên / Hạng mục | Nhu cầu định mức | Trạng thái thực tế | Đánh giá khả thi & Blocker |
| :--- | :--- | :---: | :--- |
| **Ảnh Research Field tự chụp** | 160 ảnh authentic độc quyền | `NOT_ACQUIRED` | **BLOCKER**: Cần người dùng/nhóm nghiên cứu chụp và cung cấp 160 ảnh gốc từ thiết bị thực tế kèm metadata cảm biến. |
| **Ảnh COCO 2017** | 140 ảnh authentic (CC-BY 4.0) | `READY_FOR_AUTOMATION` | Sẵn sàng: Có thể tự động tải qua COCO/Flickr API kèm kiểm tra provenance và loại trừ 684 sources Option P lịch sử. |
| **Ảnh Unsplash Verified** | 100 ảnh authentic (trước 2022) | `READY_FOR_CURATION` | Sẵn sàng: Cần lọc danh sách URL/ID ảnh trước năm 2022 để bảo đảm không bị AI tạo sinh can thiệp. |
| **SD 2.0 Inpainting** | 160 cặp ảnh inpainting | `READY_FOR_EXECUTION` | Sẵn sàng: Checkpoint `stabilityai/stable-diffusion-2-inpainting` có sẵn trên HuggingFace/Diffusers, chạy được trên GPU local/cloud. |
| **SDXL Inpainting** | 160 cặp ảnh inpainting | `READY_FOR_EXECUTION` | Sẵn sàng: Checkpoint `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` có sẵn, chạy được trên GPU local/cloud. |
| **Adobe Firefly Generative Fill** | 80 cặp ảnh inpainting | `BLOCKED_SUBSCRIPTION` | **BLOCKER**: Cần tài khoản Adobe Creative Cloud bản quyền và can thiệp giao diện desktop / API token của người dùng. |
| **Phần cứng GPU & Lưu trữ** | GPU $\ge 4\text{GB}$ VRAM, $\approx 2\text{GB}$ disk | `FEASIBLE` | Sẵn sàng: Máy trạm sở hữu GPU NVIDIA GTX 1650 (hoặc Google Colab T4) đủ đáp ứng nhu cầu inpainting và pipeline kiểm định. |

> **BÀN GIAO TIẾP THEO (NEXT HANDOFF ACTION)**:
> Trạng thái bàn giao chính thức: **`BLOCKED_WITH_EXACT_ACQUISITION_REQUIREMENTS`**
> Quá trình chuẩn bị kỹ thuật, protocol và đường chạy evaluator đã sẵn sàng 100%. Tuy nhiên, việc thu thập cohort thực tế bị chặn bởi 2 yêu cầu cụ thể từ người dùng:
> 1. Cung cấp **160 ảnh tự chụp thực địa** (Research Field Collection).
> 2. Cung cấp quyền truy cập **Adobe Firefly / Photoshop Generative Fill** (hoặc phê duyệt Protocol Amendment nếu muốn thay đổi công cụ closed-source).
> Tuyệt đối không tự ý tiến hành thu thập cohort khi chưa giải quyết các yêu cầu này.
