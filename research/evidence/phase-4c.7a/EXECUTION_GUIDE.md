# Phase 4C.7A / 4C.7B Independent Validation Execution Guide

> **Phase**: 4C.7A — Independent Validation Preparation<br>
> **Scope**: Hướng dẫn quy trình thực thi kiểm định độc lập giữa `visual_calibrated` và `late_fusion_dsp_augmented` khi đã có cohort mới.<br>
> **Target Script**: `scripts/research/run_independent_validation.py`

---

## 1. Điều kiện Tiên quyết (Prerequisites)

Trước khi tiến hành đánh giá trên dữ liệu thực tế, các điều kiện sau bắt buộc phải được thỏa mãn:

1. **Candidate Models Đầy đủ và Bất biến**:
   - Toàn bộ 5 file `fold_model.json` ($k=0..4$) tại `data/research/local-artifacts/dsp_augmentation/fits/outer_k/` phải khớp chính xác mã băm SHA-256 đã khóa trong `candidate_model_bindings.json`.
   - Tuyệt đối không refit, không fine-tune, và không thay đổi bất kỳ hệ số nào.
2. **Cohort Độc lập Hợp lệ (Independent Cohort)**:
   - File manifest `independent_cohort_manifest.csv` phải tuân thủ schema quy định trong `COHORT_ACQUISITION_PLAN.md`.
   - Mỗi `source_id` bắt buộc có đúng 1 ảnh `authentic` (nhãn 0) và 1 ảnh `ai_edited` (nhãn 1).
   - Kiểm tra **Source-Disjoint Guard**: Không có bất kỳ `source_id` nào trùng lặp với 684 nguồn lịch sử trong `manifest_pilot_a_option_p.csv`.
   - Tuyệt đối không truy cập hoặc đổi tên dữ liệu `locked_test` đã niêm phong (`SEALED_AND_RETIRED`).

---

## 2. Quy trình Thực thi Chuẩn (Standard Operating Procedure)

### Bước 1: Kiểm tra tính sẵn sàng (Readiness Check)
Chạy lệnh kiểm tra tính toàn vẹn của protocol, bindings và môi trường:
```powershell
python scripts/research/run_independent_validation.py --check-readiness
```
- Kết quả mong đợi: `Model Bindings Verified: 5 / 5`. File `research/evidence/phase-4c.7a/readiness.json` được cập nhật.

### Bước 2: Kiểm thử mô phỏng (Synthetic Preflight)
Chạy kiểm thử giả lập end-to-end trên dữ liệu tổng hợp để đảm bảo runtime và thuật toán bootstrap hoạt động chính xác:
```powershell
python scripts/research/run_independent_validation.py --synthetic-preflight --bootstrap-replicates 1000
```
- Kết quả mong đợi: `Preflight Status: SYNTHETIC_PREFLIGHT_PASS`, `Synthetic Verdict: SYNTHETIC_ONLY_NOT_MEASURED`.

### Bước 3: Đánh giá Cohort Thực tế (Real Cohort Execution - Khi Cohort Đã Sẵn Sàng)
Khi dữ liệu độc lập đã được thu thập và thẩm định:
```powershell
python scripts/research/run_independent_validation.py --evaluate --manifest path/to/independent_cohort_manifest.csv --bootstrap-replicates 10000 --seed 20261007
```

---

## 3. Quy chuẩn Tính toán và Báo cáo Kết quả

1. **Ước lượng Hiệu năng Chính (Primary Recipe Estimate)**:
   - Với mỗi recipe và mỗi condition, tính Macro-F1 độc lập cho từng model trong 5 outer-fold models.
   - Primary estimate là **trung bình số học (arithmetic mean)** của 5 chỉ số Macro-F1 này.
   - **CẤM**: Không lấy trung bình xác suất (probabilities) của 5 models trước khi tính Macro-F1; không tạo ensemble.
2. **Phép So sánh Chính (Primary Endpoint)**:
   - $\Delta \overline{\text{Macro-F1}}$ tại điều kiện `jpeg_q75` giữa `late_fusion_dsp_augmented` và `visual_calibrated`.
   - Paired source-cluster bootstrap: 10,000 replicates, seed PCG64 `20261007`, khoảng tin cậy 95% percentile $[2.5\%, 97.5\%]$.
3. **Từ vựng Kết luận Cố định (Fixed Verdict Vocabulary)**:
   - `INDEPENDENT_JPEG75_IMPROVEMENT`: Cận dưới của 95% CI $> 0$.
   - `INDEPENDENT_JPEG75_DEGRADATION`: Cận trên của 95% CI $< 0$.
   - `INDEPENDENT_JPEG75_INCONCLUSIVE`: Khoảng 95% CI chứa 0 (không chứng minh được ưu thế, cũng không chứng minh tính tương đương).

---

## 4. Giới hạn Khoa học và Bảo vệ Trạng thái Web

- Kết quả kiểm định độc lập (kể cả khi đạt `IMPROVEMENT`) chỉ khẳng định ưu thế theo đúng endpoint đã khóa và trên tập dữ liệu được đánh giá; **không tự động chứng minh mô hình đạt chất lượng để tích hợp sản phẩm thương mại**.
- Trạng thái web UI vẫn duy trì trạng thái trung thực: **`uncertain` / `Model not installed`**.
