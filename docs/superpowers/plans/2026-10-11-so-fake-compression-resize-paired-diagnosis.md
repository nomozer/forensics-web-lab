# So-Fake Probe Corrections & Paired Compression/Resize Sensitivity Diagnosis Implementation Plan

> **Protocol Context**: RQ1 & RQ2 Lightweight Visual-only vs Visual+DSP Forensics  
> **Branch**: `research/independent-cohort-acquisition`  
> **Commit Base**: `0ea68d03154b75ff511dc219a8c5ce548632baec`  
> **Status**: `LOCKED_PLAN_BEFORE_EXECUTION`  

---

## 1. Executive Summary & Goals

Mục tiêu của đợt thực nghiệm này gồm hai phần gắn kết chặt chẽ:
1. **Sửa các điểm ảnh hưởng đến bằng chứng và diễn giải của So-Fake exploratory probe**:
   - Sửa lỗi gộp cụm giả tạo trong manifest do cắt ngắn chuỗi `Path(stem)[:16]`. Xác lập rõ ràng: 300 mẫu có 300 `sample_id` và filename duy nhất, không có metadata ảnh nguồn ban đầu trong So-Fake parquet; do đó quan hệ nguồn là `unverified`.
   - Tính lại paired bootstrap trên từng ảnh (paired sample bootstrap) kèm giả định và giới hạn rõ ràng; không gọi là "source-cluster bootstrap đã kiểm chứng". Không diễn giải bootstrap replicate fraction như p-value.
   - Sửa phân tích theo bộ sinh (generator breakdown): tính theo cặp `(generator, ground_truth_label)`. Không gộp nhãn dòng đầu tiên cho toàn bộ generator (đặc biệt FLUX.1 có 44 edit + 15 gen; latent_diffusion có 12 edit + 20 gen).
   - Phân tích taxonomy cho `OpenForensics_fake` ($N=9$): giữ nguyên $N=300$ nhãn nguồn, bổ sung phân tích sensitivity riêng cho ranh giới phân loại khuôn mặt ghép vs ảnh tổng hợp toàn phần.
   - Đính chính tài liệu: xác nhận A1 train là $N=1,000$ (A0 pilot 750 + 250 ImageNet train); đính chính giấy phép `saberzl/So-Fake-Set` là `unspecified` trong card metadata (không tự gán CC-BY-NC-4.0 từ `So-Fake-OOD`).
2. **Kiểm tra pipeline và thực hiện thí nghiệm ghép cặp về nén JPEG và thứ tự resize**:
   - Kiểm tra pipeline kỹ thuật trên panel tất định: xác nhận decode, orientation, feature extraction và scoring khớp bit-exact.
   - Đánh giá độ nhạy trên 2 cohorts: So-Fake probe ($N=300$) và development holdout ($N=273$).
   - Khóa ba điều kiện xử lý ảnh:
     * **C0**: Baseline pipeline nguyên trạng.
     * **C1**: Nén JPEG full-resolution ($W, H$) tại quality=75, subsampling=2 (4:2:0), optimize=False; sau đó resize canvas 512×512 (Lanczos) cho DSP và 224×224 (bicubic) cho Visual.
     * **C2**: Resize canvas 512×512 (Lanczos) trước; sau đó nén JPEG 512×512 cùng tham số quality=75, subsampling=2, optimize=False; decode và đưa trực tiếp vào trích xuất DSP (không resize lặp lại); resize 224×224 (bicubic) cho Visual.
   - Đánh giá cả 6 mô hình đã khóa trên cả hai cohorts dưới 3 điều kiện (36 lượt đo).
   - Đo lường chi tiết: Macro-F1, Balanced Accuracy, per-class P/R/F1, Brier (so sánh với baseline đều $[1/3, 1/3, 1/3] = 0.6667$), ECE, tỷ lệ đổi nhãn, độ dịch chuyển xác suất theo cặp, và paired bootstrap $\Delta$.

---

## 2. File Structure & Responsibilities

| File Path | Role & Technical Responsibility |
| :--- | :--- |
| `scripts/research/fix_so_fake_manifest_and_audit.py` | Tạo manifest v2 (`so_fake_probe_manifest_v2.json`) sửa `source_group_id` (trả về unverified/unique sample), kiểm tra lại tính toàn vẹn 300 mẫu. |
| `scripts/research/re_evaluate_so_fake_baseline.py` | Tính lại baseline So-Fake C0 với paired sample bootstrap, generator breakdown theo `(generator, ground_truth)`, sensitivity analysis cho `OpenForensics_fake`; xuất `so_fake_probe_evaluation_receipt_v2.json`. |
| `scripts/research/run_paired_compression_sensitivity.py` | Thực thi thí nghiệm ghép cặp C0 vs C1 vs C2 cho cả 6 models trên So-Fake ($N=300$) và Dev Holdout ($N=273$); đo tỷ lệ đổi nhãn, độ dịch xác suất, Brier/ECE, Macro-F1; xuất raw predictions và receipt. |
| `ml/tests/test_compression_sensitivity.py` | Bộ test tự động kiểm tra: C1 vs C2 pipeline invariants, tính toàn vẹn outputs, không resize lặp, schema receipts, Brier baseline benchmark. |
| `research/evidence/three_class_preparation/compression_sensitivity_predictions.json` | Bảng dự đoán thô 3 điều kiện × 2 cohorts × 6 models. |
| `research/evidence/three_class_preparation/compression_sensitivity_receipt.json` | Biên nhận đánh giá chi tiết độ nhạy nén và thứ tự resize. |
| `research/evidence/three_class_preparation/PHASE_REPORT.md` | Bổ sung Mục 15 báo cáo toàn diện các đính chính So-Fake và kết quả thí nghiệm nén/resize ghép cặp. |
| `docs/continuity/CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md` | Cập nhật hồ sơ liên tục dự án. |

---

## 3. Detailed Step-by-Step Execution Plan

### Task 1: Fix So-Fake Manifest Source Grouping & Provenance
- Sửa lỗi cắt chuỗi `Path(fname).stem[:16]`.
- Thiết lập:
  * `sample_id`: giữ nguyên định danh ổn định (`so_fake_aut_xxx`, `so_fake_edi_xxx`, `so_fake_ful_xxx`).
  * `source_group_id`: gán rõ `unverified_external_sample_{sample_id}` hoặc bảo toàn filename gốc đầy đủ, đánh dấu `source_provenance_status: "UNVERIFIED_INDEPENDENT_SOURCE"`.
  * Xuất `research/evidence/three_class_preparation/so_fake_probe_manifest_v2.json`.

### Task 2: Re-evaluate So-Fake C0 with Corrected Breakdown & Taxonomy Sensitivity
- Nhóm generator theo `(generator, ground_truth_label)`:
  * `flux.1-dev` (ai_edited, N=44) vs `flux.1-dev` (fully_generated, N=15).
  * `latent_diffusion` (ai_edited, N=12) vs `latent_diffusion` (fully_generated, N=20).
  * `sdxl` (ai_edited, N=40).
  * `OpenForensics_fake` (fully_generated, N=9).
- Chạy Paired Sample Bootstrap (10,000 reps) cho 6 models:
  * Resample theo 300 paired image indices.
  * Tính 95% Percentile CI cho các paired deltas.
  * Báo cáo fraction $P(\Delta^* > 0)$ là tỷ lệ replicate dương, không phải p-value.
- Phân tích độ nhạy Taxonomy cho `OpenForensics_fake`:
  * Đánh giá bổ sung trên $N=291$ (loại trừ 9 mẫu OpenForensics) để đo tác động của việc gán nhãn face manipulation vào fully_generated.
- Xuất receipt v2: `so_fake_probe_evaluation_receipt_v2.json`.

### Task 3: Pipeline Technical Verification on Deterministic Panel
- Kiểm tra panel 16 mẫu:
  * Xác nhận decode RGB nguyên bản từ parquet bytes vs PNG.
  * Xác nhận EXIF orientation không làm xoay ảnh.
  * Xác nhận C0, C1, C2 tạo ra đúng mảng ảnh khác biệt:
    - C1: JPEG encode full-res -> decode -> resize 512x512.
    - C2: Resize 512x512 -> JPEG encode 512x512 -> decode (không resize lại).
    - Đo độ lệch pixel giữa C1 và C2 trên panel để chứng minh hai điều kiện là hai đường xử lý khác biệt.

### Task 4: Execute Paired Sensitivity Experiment (C0, C1, C2)
- Cohort 1: So-Fake probe ($N=300$).
- Cohort 2: Development holdout ($N=273$) từ `three_class_pilot_splits.json`.
- Chạy cả 6 models (`A0-global`, `A0-local`, `A1-global`, `A1-local`, `A0-visual`, `A1-visual`).
- Tính toán:
  * Metrics chuẩn: Macro-F1, Balanced Accuracy, per-class F1, Multiclass Brier, Top-class ECE, authentic FP rate.
  * Độ dịch chuyển: Mean $|\Delta P(c)|$, tỷ lệ đổi nhãn (decision flip rate) so với C0.
  * Paired delta: $\Delta(\text{C1} - \text{C0})$ và $\Delta(\text{C2} - \text{C0})$ kèm 95% bootstrap CI.
  * So sánh Brier với baseline phân phối đều: $B_{\text{uniform}} = 2/3 \approx 0.6667$.

### Task 5: Unit Tests & Verification
- Tạo `ml/tests/test_compression_sensitivity.py`:
  * Test 1: C1 vs C2 process divergence (khác biệt pixel trên canvas 512x512).
  * Test 2: Invariance của model weights và config hashes.
  * Test 3: Predictions và receipt schema.
  * Test 4: Manifest v2 integrity (300 distinct sample IDs, không cắt cụm giả).
  * Test 5: Brier so sánh chuẩn xác với 0.6667.
- Chạy full test suite: pytest, pnpm test, pnpm continuity:check, git diff --check.

### Task 6: Documentation & Evidence Delivery
- Cập nhật `PHASE_REPORT.md` (Mục 15).
- Cập nhật `CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`.
- Trả lời đầy đủ 4 câu hỏi nghiên cứu của người dùng.
- Git commit và push lên `origin/research/independent-cohort-acquisition`.
