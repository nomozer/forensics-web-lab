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
   - Làm rõ: "Selective coverage" ($11.9\% - 14.2\%$) là tỷ lệ dự đoán được giữ lại ở ngưỡng $\tau = 0.65$, không đồng nghĩa với probability calibration.
   - Selective accuracy ($64.0\% - 66.0\%$) được báo cáo độc lập.
5. **Package Provenance**:
   - Mã băm package archive (`b9930244c5...`) và snapshot manifest (`4ed15a2b...`) liên kết chặt chẽ với `package_receipt.json` của commit chức năng `97018d05...`. Việc thực thi cục bộ diễn ra trong môi trường mã nguồn đồng nhất với snapshot này.

---

## 3. Đóng Băng Hai Candidate Recipes và Model Bindings

Toàn bộ 5 mô hình outer fold ($k \in \{0, 1, 2, 3, 4\}$) từ Phase 4C.6B/4C.3B đã được liên kết và đóng băng trong `candidate_model_bindings.json`:

| Outer Fold | Đường dẫn Model | SHA-256 Model | Trạng thái Xác minh |
| :---: | :--- | :--- | :---: |
| Fold 0 | `data/.../dsp_augmentation/fits/outer_0/fold_model.json` | `232d0994775a6e6cfdfb1bb3fd79f28557439e372e15e4760d0ffbbdba72f609` | **VERIFIED** |
| Fold 1 | `data/.../dsp_augmentation/fits/outer_1/fold_model.json` | `c07c582322f7671f3b9f352f5d5388f5a39cec5fa11c80d8441cbb366a3fc15f` | **VERIFIED** |
| Fold 2 | `data/.../dsp_augmentation/fits/outer_2/fold_model.json` | `8a979e35339f1ab2728f6c96d94ce2ef828ad13aad66e0e3ae8339a1be8fb324` | **VERIFIED** |
| Fold 3 | `data/.../dsp_augmentation/fits/outer_3/fold_model.json` | `60358c2fbdfd54d96dee68a1ce43106560ee4da5792343c0d14d85beb1d5fca3` | **VERIFIED** |
| Fold 4 | `data/.../dsp_augmentation/fits/outer_4/fold_model.json` | `efe517db09c5aecdd219eabc878b6f538e908c03b8b8c0648b2d0f6ec4356a6c` | **VERIFIED** |

### Nguyên tắc Đánh giá Cố định:
- **Giữ nguyên cả 5 models**: Không lựa chọn best fold, không loại bỏ model kém, không refit model mới.
- **Ước lượng Primary Recipe**:
  $$\overline{\text{Macro-F1}}_{\text{recipe}} = \frac{1}{5} \sum_{k=0}^{4} \text{Macro-F1}(M_k)$$
- **Nghiêm cấm trung bình xác suất trước**: Không tính $\frac{1}{5}\sum p_k$ rồi mới phân loại; không tạo ensemble.
- **5 models không phải 5 quan sát độc lập**: Không so sánh trực tiếp estimand này với pooled OOF lịch sử như thể cùng một bản chất thống kê.

---

## 4. Đặc Tả Cohort Độc Lập và Lập Luận Cỡ Mẫu

### 4.1. Tiêu chí Cohort (`COHORT_ACQUISITION_PLAN.md`)
- **Source-Disjoint Tuyệt đối**: Tuyệt đối không trùng lặp với 684 nguồn lịch sử trong `manifest_pilot_a_option_p.csv` (250 dev train + 91 dev val + 343 retired locked-test).
- **Cấu trúc Ghép cặp Nguồn**: Mỗi nguồn ảnh cung cấp chính xác 1 ảnh authentic và 1 ảnh ai_edited tương ứng từ cùng cảnh chụp gốc.
- **Provenance Minh bạch**: Cấp phép hợp pháp (CC0, CC-BY 4.0 hoặc tự chụp), có mask ground-truth và thông tin pipeline chỉnh sửa (SD2 Inpainting, SDXL, Adobe Firefly).
- **Bảo vệ Niêm phong**: Tuyệt đối không truy cập hoặc đổi tên tập `locked_test` lịch sử.

### 4.2. Luận chứng Cỡ mẫu (`SAMPLE_SIZE_JUSTIFICATION.md`)
- Áp dụng phân tích power ghép cặp cụm nguồn và kiểm soát độ rộng khoảng tin cậy 95% ($ME \le 0.017$).
- Khuyến nghị cỡ mẫu prespecified: **$N_{\text{pairs}} = 300 - 500$ cụm nguồn** (tương đương 600 - 1,000 ảnh).
- Bác bỏ việc ấn định cỡ mẫu chủ quan không có căn cứ. Mọi mô phỏng được gắn nhãn minh bạch: `SYNTHETIC PLANNING`.

---

## 5. Khóa Giao Thức Kiểm Định (`independent_validation_protocol.yaml`)

- **Sáu Điều kiện Thử nghiệm**: `original`, `jpeg_q95`, `jpeg_q75`, `jpeg_q50`, `resize_0.5`, `resize_0.5_jpeg_q75`.
- **Endpoint Chính Prespecified**:
  $$\Delta_{\text{primary}} = \overline{\text{Macro-F1}}_{\text{augmented}}(\text{jpeg\_q75}) - \overline{\text{Macro-F1}}_{\text{visual}}(\text{jpeg\_q75})$$
- **Phương pháp Bootstrap**:
  - Paired source-cluster bootstrap: 10,000 replicates.
  - Bộ sinh số ngẫu nhiên: PCG64 với seed `20261007`.
  - Cùng các mẫu bootstrap dùng chung cho cả 2 recipes và cả 5 models.
  - 95% Percentile Confidence Interval $[2.5\%, 97.5\%]$.
- **Tiêu chí Thành công**: Cận dưới khoảng tin cậy 95% $> 0$ (`INDEPENDENT_JPEG75_IMPROVEMENT`).
  *Ghi chú*: Kết quả này chỉ khẳng định ưu thế theo endpoint đã prespecify; không tự động chứng minh model đạt chất lượng sản phẩm.
- **Từ vựng Kết luận**:
  - `INDEPENDENT_JPEG75_IMPROVEMENT`: CI lower > 0
  - `INDEPENDENT_JPEG75_DEGRADATION`: CI upper < 0
  - `INDEPENDENT_JPEG75_INCONCLUSIVE`: CI chứa 0
  - `SYNTHETIC_ONLY_NOT_MEASURED`: Chạy trên dữ liệu mô phỏng

---

## 6. Triển Khai Công Cụ và Synthetic Preflight Verification

Đã triển khai và kiểm thử thành công:
1. `ml/evaluation/independent_cohort.py`: Schema validator, label checker, và source-overlap guard.
2. `ml/evaluation/independent_model_bindings.py`: Trình nạp model bindings kiểm tra SHA-256 mã băm, scorer đại số xác thực.
3. `ml/evaluation/independent_evaluator.py`: Bộ tính toán mean-per-model Macro-F1, secondary metrics, và paired bootstrap engine.
4. `scripts/research/run_independent_validation.py`: CLI runner kiểm tra readiness và synthetic preflight.

### Kết quả Synthetic Preflight:
```
Readiness Status: PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT
Model Bindings Verified: 5 / 5
Real Cohort Acquired: False
Preflight Status: SYNTHETIC_PREFLIGHT_PASS
Synthetic Verdict: SYNTHETIC_ONLY_NOT_MEASURED
```

---

## 7. Deliverables Gói Evidence Phase 4C.7A

Tất cả deliverables đã được tạo tại `research/evidence/phase-4c.7a/`:
- `independent_validation_protocol.yaml`
- `candidate_model_bindings.json`
- `COHORT_ACQUISITION_PLAN.md`
- `SAMPLE_SIZE_JUSTIFICATION.md`
- `EXECUTION_GUIDE.md`
- `readiness.json`
- `PHASE_REPORT.md`

---

## 8. Kết luận và Trạng Thái Dừng

Phase 4C.7A đã hoàn thành toàn bộ công tác chuẩn bị khoa học, kỹ thuật và kiểm thử cho phép kiểm định độc lập.
- **Huấn luyện mới**: 0 fits.
- **Đánh giá thực tế**: 0 runs (kết quả cohort độc lập là `NOT_MEASURED`).
- **Web UI**: Duy trì trung thực trạng thái `uncertain` / `Model not installed` theo ADR-0004.
- **Trạng thái chính thức**: `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT`.
