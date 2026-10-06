# Phase 4C.7A / 4C.7B Independent Validation Execution Guide (Finalized)

> **Phase**: 4C.7A — Independent Validation Preparation<br>
> **Scope**: Hướng dẫn quy trình thực thi kiểm định độc lập giữa `visual_calibrated` và `late_fusion_dsp_augmented`.<br>
> **Target Script**: `scripts/research/run_independent_validation.py`<br>
> **Audit Script**: `scripts/research/audit_candidate_model_bindings.py`<br>
> **Planning Simulation Script**: `scripts/research/simulate_sample_size_planning.py`

---

## 1. Điều Kiện Tiên Quyết (Prerequisites)

1. **Candidate Models Đầy Đủ và Bất Biến**:
   - Toàn bộ 5 file `fold_model.json` ($k=0..4$) tại `data/research/local-artifacts/dsp_augmentation/fits/outer_k/` phải khớp chính xác mã băm SHA-256 đã ghi nhận trong `candidate_model_bindings.json` và `model_bindings_audit.json`.
   - Tuyệt đối không refit, không fine-tune, và không thay đổi bất kỳ hệ số nào.
2. **Cohort Độc Lập Hợp Lệ (Independent Cohort)**:
   - File manifest `independent_cohort_manifest.csv` phải tuân thủ schema quy định trong `COHORT_ACQUISITION_PLAN.md`.
   - Mỗi `source_id` bắt buộc có đúng 1 ảnh `authentic` (nhãn 0) và 1 ảnh `ai_edited` (nhãn 1), cùng resolution.
   - Kiểm tra **Source-Disjoint Guard Đa Tầng**: Không trùng `source_id`, không trùng mã băm `image_sha256`, không trùng `origin_id` với 684 nguồn lịch sử trong `manifest_pilot_a_option_p.csv`.
   - Tuyệt đối không truy cập hoặc đổi tên dữ liệu `locked_test` đã niêm phong (`SEALED_AND_RETIRED`).

---

## 2. Các Lệnh Đã Triển Khai và Kiểm Thử Thực Tế

Tất cả các lệnh dưới đây đều đã được cài đặt đầy đủ và kiểm thử trực tiếp trong CLI `run_independent_validation.py`:

### Lệnh 1: Kiểm Tra Tính Sẵn Sàng (Readiness Check)
Kiểm tra tính toàn vẹn của protocol, 5 model bindings và trạng thái cohort:
```powershell
python scripts/research/run_independent_validation.py --check-readiness
```
- **Hành vi**: Xác minh 5 models trên đĩa, kiểm tra manifest lịch sử, xuất `research/evidence/phase-4c.7a/readiness.json`.

### Lệnh 2: Kiểm Thử Đặc Trưng Giả Lập (Synthetic Feature Preflight)
Kiểm tra luồng xử lý mảng đặc trưng giả lập và phép tính bootstrap:
```powershell
python scripts/research/run_independent_validation.py --synthetic-preflight --bootstrap-replicates 500 --seed 20261007
```
- **Hành vi**: Sinh cohort giả lập, kiểm tra luồng tính điểm của 5 models, chạy paired source bootstrap, xuất kết quả mang nhãn `SYNTHETIC_ONLY_NOT_MEASURED`.

### Lệnh 3: Kiểm Thử Đường Chạy Ảnh Thực Thụ (End-to-End PIL Image Pipeline Preflight)
Kiểm tra toàn bộ chuỗi xử lý từ file ảnh đến kết luận thống kê:
```powershell
python scripts/research/run_independent_validation.py --image-preflight --bootstrap-replicates 100 --seed 20261007
```
- **Hành vi**: Khởi tạo ảnh PIL RGB, áp dụng 6 biến đổi canonical (JPEG encode/decode subsampling 2, bicubic downscale), trích xuất 16-d DSP và 576-d Visual, chấm điểm qua 5 candidate models, tính toán arithmetic mean-per-model Macro-F1, và chạy paired source bootstrap.

### Lệnh 4: Đánh Giá Cohort Độc Lập Thực Tế (Real Cohort Evaluation — Dành Cho Phase 4C.7B)
Khi tập dữ liệu độc lập mới đã được thu thập và niêm phong manifest:
```powershell
python scripts/research/run_independent_validation.py --evaluate --manifest path/to/independent_cohort_manifest.csv --data-root path/to/images --bootstrap-replicates 10000 --seed 20261007
```
- **Hành vi**: Kích hoạt bộ kiểm tra source-disjoint đa tầng, chạy qua `IndependentEvaluationPipeline`, và xuất verdict chính thức (`INDEPENDENT_JPEG75_IMPROVEMENT`, `DEGRADATION`, hoặc `INCONCLUSIVE`).

---

## 3. Quy Chuẩn Tính Toán Bắt Buộc

1. **Ước Lượng Hiệu Năng Chính (Primary Recipe Estimate)**:
   - Với mỗi recipe và mỗi điều kiện, tính Macro-F1 độc lập cho từng model trong 5 outer-fold models.
   - Primary estimate là **trung bình số học (arithmetic mean)** của 5 chỉ số Macro-F1:
     $$\overline{\text{Macro-F1}}_{\text{recipe}} = \frac{1}{5} \sum_{k=0}^4 \text{Macro-F1}(M_k)$$
   - **CẤM**: Không lấy trung bình xác suất trước khi tính Macro-F1; không tạo ensemble.
2. **Phép So Sánh Chính (Primary Endpoint)**:
   - $\Delta \overline{\text{Macro-F1}}$ tại điều kiện `jpeg_q75` giữa `late_fusion_dsp_augmented` và `visual_calibrated`.
   - Paired source-cluster bootstrap: 10,000 replicates, seed PCG64 `20261007`, 95% Percentile Confidence Interval.
3. **Tiêu Chí Thành Công**:
   - Thành công khi cận dưới khoảng tin cậy 95% $> 0$ (`INDEPENDENT_JPEG75_IMPROVEMENT`).
