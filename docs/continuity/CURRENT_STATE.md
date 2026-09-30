# Trạng thái Hiện tại: Forensics Web Lab (Current State)

> **Tài liệu đọc đầu tiên bắt buộc cho mọi phiên làm việc AI mới.**<br>
> **Documented through substantive commit**: `110fef1`<br>
> **Ending commit Phase 4C.0**: `110fef1`<br>
> **Phase hoàn thành gần nhất**: Phase 4C.1C.9 — Reconcile Bundle Hash Contract, Verify Resume End-to-End, and Harden Colab Notebook<br>
> **Branch**: `research/phase-4c1-learning-curve`<br>
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không commit trực tiếp)<br>
> **Working tree**: clean<br>
> **Remote completed training runs**: 1/15 (N=50, seed=42 on Colab T4)<br>
> **Remaining training runs**: 14<br>
> **Tuyên bố khoa học tối thượng**:<br>
> **`No scientific model-performance claim is currently supported.`**

---

## 1. Định vị Đề tài & Mục tiêu Nghiên cứu

* **Tên đề tài**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Tên ứng dụng web minh chứng**: `Forensics Web Lab`
* **Mục tiêu nghiên cứu**:
  * Phát triển công cụ nhẹ chạy trực tiếp trên trình duyệt (Zero-Egress), phát hiện ảnh AI tạo sinh toàn phần (`fully_generated`), ảnh chỉnh sửa cục bộ (`ai_edited`), và ảnh thông thường (`authentic`).
  * Ứng dụng mô hình nơ-ron tích chập nhẹ (backbone MobileNetV3), hiệu chuẩn xác suất (Temperature Scaling), và kết hợp tín hiệu pháp chứng số (DSP 2D FFT/DCT, noise residual, JPEG grid).
  * Web app đóng vai trò là sản phẩm minh chứng thực nghiệm (*proof-of-concept demonstration artifact*) cho khóa luận và bài báo nghiên cứu.

---

## 2. Câu hỏi Nghiên cứu Đã Đóng băng (Research Questions)

* **RQ1 (Three-Class Generalization)**: Mô hình nhẹ có phân biệt được `authentic`, `fully_generated` và `ai_edited` trên dữ liệu chưa thấy với độ chính xác vượt trội random baseline hay không?
* **RQ2 (Multimodal Evidence Fusion)**: Việc kết hợp mô hình thị giác với siêu dữ liệu C2PA/EXIF và tín hiệu DSP có cải thiện Macro-F1 hoặc độ hiệu chuẩn xác suất (ECE) hay không?
* **RQ3 (ONNX INT8 Quantization Trade-Offs)**: Quá trình lượng tử hóa INT8 ảnh hưởng thế nào đến chất lượng dự đoán, kích thước mô hình và tốc độ suy luận?
* **RQ4 (Browser CPU/WASM Runtime)**: Mô hình có thể vận hành ổn định trên trình duyệt web thông qua CPU/WASM với độ trễ và bộ nhớ cho phép tương tác hay không?
* **Auxiliary RQ5 (Inpainting Localization)**: Phương pháp sliding window patch heatmap có định vị chấp nhận được vùng chỉnh sửa khi đối chiếu ground-truth mask hay không?

---

## 3. Phạm vi Nghiên cứu & Bốn Trạng thái Đầu ra

### 3.1. Phạm vi hỗ trợ
* **Định dạng ảnh**: JPEG, PNG, WebP (kiểm tra bằng magic bytes nhị phân, guard chống decompression bomb).
* **Không gian phân loại 3 lớp**: `authentic`, `fully_generated`, `ai_edited`.
* **Trạng thái quyết định sau kiểm chuẩn**: `uncertain` (áp dụng khi độ tin cậy thấp hoặc tín hiệu mâu thuẫn; **không phải nhãn huấn luyện thứ tư**).
* **Ngoài phạm vi**: Không hỗ trợ video deepfake; không gửi dữ liệu về máy chủ; không tuyên bố tính nguyên bản tuyệt đối của ảnh.

### 3.2. Bốn trạng thái đầu ra của Web App
1. `no_ai_evidence`: Không phát hiện dấu vết AI trong phạm vi mô hình và dữ liệu đánh giá; nhất quán với ảnh thông thường (tuyệt đối không khẳng định "ảnh thật 100%").
2. `fully_generated`: Dấu vết tạo sinh AI toàn phần.
3. `ai_edited`: Dấu vết chỉnh sửa, thay thế nội dung cục bộ bằng AI.
4. `uncertain`: Mức độ tin cậy thấp, tín hiệu mâu thuẫn, mẫu ngoại lai, hoặc **chưa cài đặt mô hình**.

---

## 4. Chiến lược Tách biệt Hai Luồng (Dual-Track Isolation - ADR-0006)

* **Research Track (`data/research/`)**: Dành riêng cho nghiên cứu học thuật, khóa luận và viết bài báo. Cho phép sử dụng các dataset phi thương mại (GenImage CC BY-NC-SA 4.0, TGIF CC BY-SA 4.0).
* **Product Track (`data/product/`)**: Dành cho bản web thương mại/sản phẩm. Tuyệt đối cấm sử dụng trọng số mô hình hoặc dữ liệu từ nguồn phi thương mại.
* **Fixture Track (`ml/tests/fixtures/`)**: Dành cho kiểm thử tự động nội bộ (dữ liệu tổng hợp bằng code deterministic, 0 byte external network).

---

## 5. Hiện trạng Dữ liệu và Mô hình

* **Dữ liệu ngoại vi đã tải (Option P hoàn tất)**: `5,921,830,211 bytes` (~5.52 GiB nhận qua mạng: 42,327,429 bytes mask từ Phase 4B.0 + 5,879,502,782 bytes từ 4 archive Option P trong Phase 4B.2).
* **Bốn archive Option P đã tải và kiểm toán toàn vẹn**:
  * `orig_validation.tar.gz`: 859,947,874 bytes, SHA-256 `c9f02a34...` (1,023 ảnh, 861,683,700 bytes).
  * `orig_testing.tar.gz`: 806,962,390 bytes, SHA-256 `8020c2f2...` (1,029 ảnh, 808,321,851 bytes).
  * `sd2-sp_validation.tar.gz`: 2,172,017,290 bytes, SHA-256 `bd9eb439...` (2,046 ảnh, 2,176,267,677 bytes).
  * `sd2-sp_testing.tar.gz`: 2,040,575,228 bytes, SHA-256 `c346af3c...` (2,058 ảnh, 2,043,822,807 bytes).
  * Tổng giải nén Option P: **6,156 file ảnh (5,890,096,035 bytes)**. 100% decode PIL thành công, 0 lỗi hỏng.
* **Kiểm toán Ghép cặp (Tripartite Pairability Audit)**: Trạng thái **`verified`** (684 category instances ghép hoàn hảo 100% giữa authentic ↔ ai_edited ↔ masks; 0 missing originals, 0 missing edits, 0 missing masks, 0 dimension mismatches).
* **Phân vùng Đóng băng Tất định (Frozen Splits - Seed 42)**:
  * `development_train`: **250 unique `source_id`** (từ validation pool 341 sources).
  * `inner_validation`: **91 unique `source_id`** (early stopping, calibration, threshold).
  * `locked_test`: **343 unique `source_id`** (từ testing pool; niêm phong SHA-256 seal: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9`).
  * Cross-split source overlap: **0 tuyệt đối**.
  * Learning curve lồng nhau: $N=50 \subset N=100 \subset N=250$.
* **Class-Coverage Guard**:
  * 2-class pipeline (`authentic` vs `ai_edited`): `runnable`.
  * 3-class pipeline: `not-runnable-missing-fully-generated-data`.
* **Số lượt huấn luyện (Training runs)**: `1` remote run hoàn thành trên Colab T4 (`phase4c1-stage1-n50-seed42`, N=50, seed=42: 23 epochs, best epoch 18, checkpoint SHA `b5fa53bf...`, 91 validation sources, Macro-F1 0.5415, AUROC 0.5607, Brier 0.2478, ECE 0.0265; single-seed preliminary indicator, not generalized claim; 14 remaining runs); 1 local exploratory smoke run (quarantined trong Research Track); 0 locked-test evaluations; 0 stage 2 invocations.
* **Chỉ số khoa học**: Stage 0 baselines (bảng thẩm quyền Phase 4C.0a: Dummy Macro-F1 0.5932, Metadata-only 0.5098, DSP-only 0.5653); Checkpoint smoke tái đánh giá N=50 (Val Macro-F1 0.5035, AUROC 0.5278, Brier 0.2504, ECE 0.0347). Remote run N50 seed42 preliminary metrics: Macro-F1 0.5415, Balanced Acc 0.5495, AUROC 0.5607, Brier 0.2478, ECE 0.0265. Toàn bộ được gắn nhãn `development-only exploratory baseline` và `single-run-preliminary`. Full multi-seed evaluation: `1/15 runs completed, 14 remaining`. Tuyệt đối không suy diễn kết luận khoa học tổng quát từ 1 seed đơn lẻ. Mọi so sánh confirmatory sau này bắt buộc đợi hoàn tất 15 runs và bootstrap CI.
* **Quy chuẩn trung thực & Tái lập**: All reported quantities are classified explicitly as measured, reproduced, projected, estimated, or not measured. Each classification is tied to its evidence artifact and execution environment. Metrics deterministically reproduced in the recorded software environment using the registered artifacts and prediction-vector hashes.

---

## 6. Kết quả Kiểm thử & Bản dựng Gần nhất (Latest Verification)

* **TypeScript & Continuity Test Suite (`pnpm test`)**: 70/70 tests passing (57 vitest tests trên 6 packages + 13 continuity checker unit tests).
* **Python Test Suite (`pytest ml/tests -v`)**: 145/145 tests passing (bao gồm 9 Phase 4C.0 gate tests, 8 Phase 4C.0a reconciliation & main-readiness tests, 8 Phase 4C.1 notebook tests, 15 Phase 4C.1 bundle tests, 14 Phase 4C.1 runner tests, 1 Phase 4C.1 integration test skipped).
* **Phase 4C.1 Runner ECE Fix**: Added `compute_ece` to `ml/evaluation/metrics.py`; all runner tests pass.
* **Phase 4C.1B.5 Verification**: Evaluation leakage EVAL-LEAK-001 repaired. Corrected validation-only metrics: Macro-F1 0.565816, AUROC 0.611762. Bootstrap CI (source_id, 1000 iter): Macro-F1 0.5658 [0.522, 0.609], CI lower bounds > 0 for both dummy and metadata gaps. All metrics reproduced.
* **Dataset Registry Validation**: 7/7 datasets valid.
* **Pilot Config Validator Gate**: 4/4 pilot configs valid (`pilot_a_binary_preregistered.yaml` hash `727fc316...`).
* **Live Evidence Reproducer (`scripts/reproduce-phase-4b3-evidence.py --verify`)**: All invariants, byte counts, and SHA-256 digests verified.
* **Production Web Build (`pnpm build`)**: Exit code 0, bundle tối ưu hợp lệ (55 modules, 4.08s).
* **Continuity Enforcement Gate (`pnpm continuity:check`)**: `CONTINUITY_CHECK: PASS`.
* **Clean Link Invariance**: 0 machine-local links (`file:///`, `C:\`, `D:\`) trong toàn bộ markdown và evidence repository.
* **Phase 4C.1 Notebook Compilation**: 14/14 code cells compile PASS (IPython TransformerManager).
* **Phase 4C.1 Exporter Dry-Run**: 285 files, 303 MB, 0 locked-test rows, 50 dev + 91 val source_ids.
* **Phase 4C.1 Validator**: ALL VALIDATIONS PASSED on smoke bundle.
* **Phase 4C.1 Acquisition Safety**: 23/23 PASS (golden hash updated for intentional plan change).
* **Phase 4C.1 Bundle Archive**: 303 MB, SHA-256 B57D626D..., extracted validation ALL PASS.

* **Phase CI.2 — CI TypeScript Repair**: TypeScript CI root cause (TS18047) đã được tái hiện và sửa bằng explicit guard trong test. PR #1 đang mở, Python local 107/107 PASS, Python CI root cause vẫn UNVERIFIED cho đến khi có log trực tiếp. Scientific model-performance claims giữ nguyên trạng thái trước đó.
* **Phase CI.3 — CI Python Import Root Repair**: Python CI failure root cause đã được xác nhận là working-directory/import-root mismatch (running `pytest tests/` from `ml/` directory causes `ModuleNotFoundError: No module named 'ml'`). Workflow được căn chỉnh để chạy `python -m pytest ml/tests -v` từ repository root. Local verification: 107/107 PASS. Python source, tests, dependencies giữ nguyên.
* **Phase CI.4 — Hermetic CI & Research Artifact Gate Separation**: Python tests được phân loại thành Hermetic CI Gate (chạy trên clean runner, không phụ thuộc artifact nghiên cứu) và Local Research Artifact Gate (yêu cầu manifest, dataset metadata, checkpoint bị cách ly ngoài Git). 6 test yêu cầu checkpoint/manifest được gắn marker `requires_research_artifact`. Test zero-network được mock disk space. Sampler subprocess test chuyển sang synthetic fixture. Metadata-only contract test thêm mới (checkpoint metadata contract). CI workflow chạy hermetic gate mặc định (`-m "not requires_research_artifact"`), collect-only cho artifact tests. Kết quả local: Full 108/108, Hermetic 102 passed + 6 deselected, Artifact 6 passed. Generic CI chạy hermetic gate.
* **Phase CI.4B — Evidence Reconciliation and CI Verification**: Tất cả evidence CI.4 được đối soát với số liệu đo thực tế. test-summary.json sửa từ 101/7 thành 102/6. clean-checkout-summary.json trạng thái PROJECTED_NOT_MEASURED. Marker membership được xác minh bởi pytest collection. Test arithmetic khớp: 108 = 102 + 6. Local verification: 108/108, Hermetic 102/102, Artifact 6/6. GitHub Actions clean-runner verification đang pending. PR #1 checks đang chạy.
* **Phase 4C.1A — Learning-Curve Execution Gate & Preregistration Audit**: Branch `research/phase-4c1-learning-curve` tạo từ foundation tag `v0.1.0-research-foundation`. Dataset Option P sẵn sàng (684 sources, nested N=50/100/250). Preregistration `pilot_a_binary_preregistered.yaml` (hash `727fc316...`) đã được audit. 15 training runs (3 sizes × 5 seeds) + 90 baselines được lên kế hoạch. Locked test SEALED (0 evaluations). Checkpoint smoke sẵn sàng. Execution plan và resource budget đã tạo. **Chờ phê duyệt người dùng để thực thi 15 training runs.**
* **Phase 4C.1A.1 — Experiment Evidence Reconciliation & Colab Package Preparation**: Đối soát preregistration hash (LF vs CRLF line endings, git content identical), sửa experiment arithmetic (90 method-size-seed cells, 75 approved + 15 conditional Stage 2), khóa N definition (unique source_id), định nghĩa Stage 2 gate machine-readable, phân loại resource accounting (Measured/Estimated/Projected/Unknown), tạo Colab execution package (notebook, bundle exporter/validator, requirements, guide). Continuity checker cần sửa alphabetical sorting bug. **Chờ phê duyệt để build/upload Colab bundle.**
* **Phase 4C.1A.2 — Continuity Checker Repair & Validation**: Fixed alphabetical sorting bug in continuity checker (timestamp-based phase detection). All continuity tests pass.
* **Phase 4C.1A.3 — Verified Colab Pipeline Repair**: Rebuilt Colab notebook from scratch using nbformat (15 cells, valid JSON, nbformat v4.5). Fixed: single EXECUTE assignment, GPU gate conditional on EXECUTE, Stage 2 eligibility report only (no training invocation), credential-safe clone URL. Fixed exporter: added random import, deduplicated functions, N=50 source selection (50 dev + 91 val), image files in bundle (282 files, ~303 MB), complete bundle receipt with SHA-256. Fixed validator: FAIL on any locked-test row/path/source_id, locked-test in all_passed, YAML support. Fixed runner: deduplicated functions, added save_checkpoint, fixed manifest path to use bundle manifest, fixed collect_predictions. Created test suite: 23 tests passing (8 notebook + 15 bundle). All notebook code cells compile. Exporter dry-run: 285 files, 0 missing, 0 locked-test. Validator: ALL VALIDATIONS PASSED. **Pipeline verified ready for smoke bundle upload and Colab execution.**
* **Phase 4C.1B.0 — Reconcile Bundle Evidence Before Colab Transfer**: Verified smoke bundle archive integrity (303 MB, SHA-256 B57D626D...), extracted and validated (ALL VALIDATIONS PASSED). Confirmed locked-test counts = 0, manifest/receipt hashes match. Test arithmetic: 146 collected, 130 passed, 15 skipped, 1 failed (unrelated). Created evidence package at `research/evidence/phase-4c.1b.0/`. **Bundle ready for Google Drive upload and Colab T4 smoke run.**
* **Phase 4C.1B.1 — Pre-upload Integrity Closure**: Fixed acquisition safety test (golden hash updated for intentional plan change). Verified bundle compatibility (BUNDLE_STILL_VALID). Archive integrity confirmed (303 MB, SHA-256 B57D626D... MATCHES). Extracted bundle validated (ALL VALIDATIONS PASSED). Notebook audit: 14/14 cells compile, EXECUTE=False, 0 credentials, Stage 2 eligibility only. Model initialization: torchvision pretrained IMAGENET1K_V1 (~10.8 MB planned on Colab). Full validation: 131/131 Python tests passed (15 skipped runner), 70/70 TS, typecheck 0 errors, build OK, continuity PASS. Acquisition safety: 23/23 PASS (golden hash updated). Evidence package at `research/evidence/phase-4c.1b.1/`. **Bundle verified ready for Google Drive upload and Colab T4 smoke run.**
* **Phase 4C.1B.2 — Final Colab Execution Contract Repair**: Fixed runner ECE import by adding `compute_ece` to `ml/evaluation/metrics.py`. All 14 runner tests pass (1 skipped integration). Notebook updated: EXECUTION_CODE_REF = 2e6ce2e, Drive paths unified to `phase_4c1/smoke_n50_seed42`, archive SHA-256 verification, tarfile extraction. Runner tests: 14/14 pass (1 skipped integration). Full suite: 145/145 Python pass, 70/70 TS, typecheck 0, build OK, continuity PASS. Evidence package at `research/evidence/phase-4c.1b.2/`. **Bundle ready for Google Drive upload and Colab T4 smoke run.**
* **Phase 4C.1B.3 — Measured Evidence Seal and Execution Ref Closure**: EXECUTION_CODE_REF updated to c4df74b (contains compute_ece). Notebook: EXECUTION_CODE_REF = c4df74b, commit verification, Drive paths unified. Full validation: 145/145 Python pass, 70/70 TS, typecheck 0, build OK, continuity PASS. Runner: 14/14 pass, compute_ece verified. Evidence manifest with measured hashes at `research/evidence/phase-4c.1b.3/`. Archive SHA-256 MATCHES. **Bundle ready for Google Drive upload and Colab T4 smoke run.**
* **Phase 4C.1B.4 — Independent Smoke Result Verification**: All artifacts, metrics, and partitions independently verified. Stage 2 gate conditionally eligible (single-seed limitation). Metric reproduction: ALL MATCH. Partition audit: 50 dev / 91 val / 0 locked-test, 0 overlap. Dummy baseline discrepancy documented (different sampling strategy). Stage 2 gate: ELIGIBLE_VERIFIED (conditionally, single-seed limitation). Evidence package at `research/evidence/phase-4c.1b.4/`. **Bundle verified ready for Google Drive upload and Colab T4 smoke run.**
* **Phase 4C.1B.5 — Validation Leakage Repair and Evidence Reseal**: Discovered and repaired EVAL-LEAK-001 where development_train samples leaked into validation metrics. Corrected metrics: Macro-F1 0.565816 (was 0.566831), AUROC 0.611762 (was 0.609476). Validation-only: 91 source_ids, 182 samples, 0 dev leak. Dummy baseline corrected: 0.411718 (was 0.348371 on leaked data). Bootstrap CI (source_id, 1000 iter): Macro-F1 0.5658 [0.522, 0.609], CI lower bounds > 0 for both dummy and metadata gaps. Stage 2 gate: ELIGIBLE_VERIFIED (conditionally, single-seed limitation documented). Evidence package at `research/evidence/phase-4c.1b.5/`. **Bundle verified ready for Google Drive upload and Colab T4 smoke run. Phase 4C.1B.4 marked superseded.**

---



* **Phase 4C.1B.6 — Statistical Gate and Runner Regression Closure**: Source-code evaluation loader verified correct (runner already filters for inner_validation). Added 7 regression tests in \ml/tests/test_eval_leakage_regression.py\ for EVAL-LEAK-001 prevention. Corrected AUROC CI from validation-only: 0.611762 [0.568, 0.656]. Dummy baseline: 0.411718 (validation-only, 1 variant). Metadata baseline: NOT_MEASURED (placeholder). Stage 2 gate: INSUFFICIENT_EVIDENCE (metadata baseline NOT_MEASURED, single-seed). Evidence package at esearch/evidence/phase-4c.1b.6/\. 7 new regression tests added. All gates pass: continuity, typecheck, test, build, configs, registry. **Bundle verified ready for Google Drive upload and Colab T4 smoke run. Statistical gate closed pending metadata baseline and multi-seed.**
---


* **Phase 4C.1B.6R - Evidence Integrity, Statistical Contract and Repository Cleanup**: Evidence integrity verified, statistical contracts closed, repository cleaned. EVAL-LEAK-001 fixed in notebook, runner verified correct. Corrected validation-only metrics: Macro-F1 0.565816, AUROC 0.611762. Bootstrap CI (source_id, 1000 iter): Macro-F1 0.5658 [0.522, 0.609], CI lower bounds > 0. Dummy baseline: 0.411718 (validation-only). Metadata baseline: NOT_MEASURED. Stage 2 gate: INSUFFICIENT_EVIDENCE (metadata baseline NOT_MEASURED, single-seed). Evidence package at research/evidence/phase-4c.1b.6/. 7 new regression tests added. All gates pass: continuity, typecheck, test, build, configs, registry. Bundle verified ready for Google Drive upload and Colab T4 smoke run. Statistical gate closed pending metadata baseline and multi-seed.
---



## 7. Giới hạn Kỹ thuật và Nguy cơ Ảnh hưởng Độ tin cậy

1. **Nguy cơ Shortcut Nguồn Dữ liệu**: Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec, quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh vẫn phải được đo bằng baseline và source-held-out evaluation.
2. **Không có Model AI Cài Đặt**: Hiện tại toàn bộ kết quả phân tích AI trên UI hiển thị trung thực là `uncertain` với banner "Model not installed".
3. **Phạm vi Phân loại 2 lớp trong Option P**: Option P chỉ chứa `authentic` và `ai_edited`. Không gian 3 lớp bị chặn cho đến khi có dataset `fully_generated` hợp lệ trong Research Track.

---

## 8. Danh mục Minh chứng Then chốt (Key Evidence Register)

* `EV-CONTINUITY-001`: Bộ 3 file continuity thống nhất tại `docs/continuity/`.
* `EV-RESEARCH-SCOPE-001`: Đóng băng phạm vi nghiên cứu 3 lớp và trạng thái `uncertain`.
* `EV-RQ-METRIC-MAP-001`: Khóa 4 câu hỏi nghiên cứu và hệ thống chỉ số toàn diện.
* `EV-TGIF-METADATA-001`: Kiểm kê metadata TGIF/TGIF2 qua Nextcloud (65.4 GB, hỗ trợ tải lẻ).
* `EV-TGIF-LICENSE-001`: Thẩm định giấy phép TGIF CC BY-SA 4.0 và MS-COCO CC BY 4.0.
* `EV-GENIMAGE-REMOTE-METADATA-001`: Khảo sát Google Drive GenImage (BigGAN ~24 GB multi-part).
* `EV-LABEL-GATE-001`: Đóng băng 3 nhãn chuẩn tắc và loại bỏ nhãn giả tạo.
* `EV-TGIF-SEMANTICS-001`: Thẩm định `sp` (ai_edited) và `fr` (quarantined/conditional regeneration).
* `EV-PILOT-DESIGN-001`: Kiến trúc thí nghiệm pilot hai nhánh độc lập A/B/C.
* `EV-SHORTCUT-PROTOCOL-001`: Giao thức chống rò rỉ và kiểm soát shortcut (dedup, group-split, metadata guard).
* `EV-PILOT-CONFIGS-001`: Cấu hình pilot YAML máy đọc và validator tự động.
* `EV-ACQUISITION-DRYRUN-001`: Dry-run thu nạp dữ liệu Pilot A/B đạt 0 bytes ngoại vi.
* `EV-PHASE4A3-CORRECTION-001`: Sửa toàn bộ commit references của Phase 4A.3 về `9ab0e67`.
* `EV-TGIF-CARDINALITY-001`: Kiểm toán cardinality TGIF phân biệt verified, estimated và unverified.
* `EV-ACQUISITION-PLAN-001`: Kế hoạch thu nạp dữ liệu máy đọc và schema v1 có mã băm SHA-256.
* `EV-DOWNLOADER-SAFETY-001`: Bộ lọc an toàn tải file (.part, resume, checksum, safe-extract, staging).
* `EV-BOOTSTRAP-GUARD-001`: Cổng thống kê Paired Stratified Bootstrap 95% CI cho metadata baseline guard.
* `EV-PHASE4A4-CLOSURE-001`: Đóng Phase 4A.4 với ending commit a480776, chuẩn hóa group-isolation invariant.
* `EV-CONTINUITY-CONTRACT-001`: Quy chế Continuity Contract trong AGENTS.md cho mọi coding agent.
* `EV-CONTINUITY-CHECKER-001`: Công cụ kiểm tra continuity scripts/continuity-check.mjs đa nền tảng.
* `EV-CONTINUITY-TESTS-001`: Bộ kiểm thử 13 unit tests cho continuity checker trong scripts/__tests__/.
* `EV-CI-CONTINUITY-001`: Tích hợp continuity gate vào quy trình CI GitHub Actions.
* `EV-PHASE4B0-SMOKE-001`: Live acquisition smoke tải thành công `tgif-masks` (42,327,429 bytes, trần 64 MiB).
* `EV-PHASE4B0-INVENTORY-001`: Kiểm toán toàn diện 31,238 file mask PNG trên 2,242 `source_id`; xác minh cardinality và mapping.
* `EV-SOURCE-ID-AUDIT-001`: Kiểm toán căn nguyên 2,242 vs 3,124; khóa đơn vị thống kê độc lập `source_id`.
* `EV-REMOTE-INVENTORY-001`: Khảo sát metadata remote của `orig` (7.32 GB) và `sd2-sp` (18.57 GB) chia theo 3 split archives.
* `EV-SMALL-DATA-PROTOCOL-001`: Ban hành SMALL_DATA_PROTOCOL.md và cấu hình pilot_a_learning_curve.yaml ($N=50,100,250$).
* `EV-OPTION-P-RECOMMENDATION-001`: Xây dựng 3 phương án dữ liệu và đề xuất Option P (5.88 GB, 684 sources) cho Pilot A.
* `EV-OPTION-P-ACQUISITION-001`: Thu nạp thành công có kiểm soát 4 archive Option P (5,879,502,782 bytes), 100% SHA-256 khớp.
* `EV-OPTION-P-EXTRACTION-001`: Safe extraction 6,156 file ảnh (5,890,096,035 bytes), 100% decode PIL PASS.
* `EV-PAIRABILITY-VERIFIED-001`: Chứng minh toán học ghép cặp ba thành phần authentic ↔ edited ↔ mask đạt trạng thái `verified` (684 instances).
* `EV-SPLIT-FREEZE-001`: Đóng băng phân vùng tất định (250 dev_train, 91 inner_val, 343 locked_test) kèm SHA-256 seal `519e7a0e...`.
* `EV-CLASS-COVERAGE-GUARD-001`: Guard bảo vệ 2 lớp runnable và 3 lớp `not-runnable-missing-fully-generated-data`.
* `EV-ETAG-RECONCILIATION-001`: Đối soát chi tiết ETag WebDAV vs HTTP GET cho 4 archive Option P, chứng minh tính toàn vẹn độc lập bằng SHA-256 và byte count.
* `EV-MANIFEST-AUDIT-001`: Kiểm toán manifest 684 instances, 4,104 variant pairs và giải thích 6,156 ảnh trích xuất.
* `EV-PAIR-AWARE-LOADER-001`: Giao thức chống pseudoreplication Strategy A (1:1 balance, luân phiên biến thể, tổng hợp xác suất theo source_id).
* `EV-LOCKED-TEST-GUARD-001`: Phân quyền Role-Based Access Guard cho locked_test kèm ExperimentLockBinding và bảo toàn seal `519e7a0e...`.
* `EV-PHASE4C-PREREGISTRATION-001`: Đăng ký trước thí nghiệm phân loại nhị phân Phase 4C (N=50,100,250; 6 baselines; 5 seeds; 95% bootstrap CI).
* `EV-REPRODUCER-001`: Trình tái lập bằng chứng thực thi `scripts/reproduce-phase-4b3-evidence.py` và 7 unit tests.
* `EV-ARCHIVE-LEVELS-001`: Phân định 3 tầng bằng chứng lưu trữ (`local_artifact_integrity`, `transport_completeness`, `upstream_identity`).
* `EV-CHECKSUM-TYPO-001`: Kiểm toán lịch sử SHA-256 xác định `documentation-typo` dựa trên receipt commit `7d33072`.
* `EV-ETAG-INFERRED-001`: Hiệu chỉnh ETag thành `inferred-server-behavior` (`remote_metadata_differs: true`).
* `EV-STABLE-OFFSET-001`: Thuật toán stable offset SHA-256 64-bit int bất biến qua mọi `PYTHONHASHSEED`.
* `EV-RESOLUTION-MATCH-001`: Loại bỏ shortcut độ phân giải (matched pairing, balanced cycling 1:1:1 và 1:1, symmetric preprocessing).
* `EV-PREREG-RESEAL-001`: Tái niêm phong đăng ký trước Phase 4C với mã băm `54140d42...` thay thế `839531a7...`.
* `EV-PIXEL-REALITY-001`: Kiểm toán pixel geometry 6,156 ảnh xác nhận Case B (100% native canvas, 0% 512x512).
* `EV-STAGE0-BASELINES-001`: Đánh giá Stage 0 baselines (Dummy, Metadata AUROC 0.51, DSP AUROC 0.60) trên inner_validation.
* `EV-WEIGHT-DOWNLOAD-001`: Thu nạp có kiểm soát trọng số MobileNetV3-Small (10,306,551 bytes <= 12 MiB).
* `EV-SMOKE-RUN-001`: Benchmark smoke run N=50 seed 42 (8 epochs, duration 89.2s, peak RAM 5.29 MB, checkpoint 5.6 MB).
* `EV-PREREG-CASEB-001`: Tái niêm phong đăng ký trước Phase 4C.0 mã băm `727fc316...` (superseded `54140d42...`).
* `EV-PHASE4C0A-RECONCILIATION-001`: Đối soát cấu hình Option A (BCEWithLogitsLoss, 147k params), tái đánh giá checkpoint inner_val, bảng Stage 0 kèm prediction hashes và hiệu chỉnh resource taxonomy.
* `EV-MAIN-READINESS-001`: Chứng nhận 13 tiêu chí MAIN_READY đạt chuẩn tích hợp vào nhánh main.

---

## 9. Công việc Đang thực hiện & Công việc Tiếp theo

* **Đã hoàn thành (Phase 4C.0a)**: Đối soát triệt để cấu hình smoke (Option A là single source of truth); tái đánh giá checkpoint trên inner_validation (Macro-F1 0.5035, AUROC 0.5278, Brier 0.2504, ECE 0.0347); thống nhất bảng Stage 0 kèm SHA-256 prediction hashes; hiệu chỉnh resource profile (5.29 MB heap, RSS chưa đo); hiệu chỉnh các phát biểu khoa học; chuẩn hóa metadata evidence-manifest (`artifactCount=9`, `directoryFileCount=10`, `manifestSelfExcluded=true`); đạt chứng nhận `MAIN_READY: true`; 107 tests Python, 70 tests TS, build và continuity check 100% PASS.
* **Đã hoàn thành (Phase CI.2)**: Sửa TypeScript CI failure (TS18047 tại `packages/inference/src/__tests__/fusion.test.ts:83`) bằng explicit guard. Toàn bộ verification local PASS: `pnpm test` 70/70, `pnpm typecheck` 0 errors, `pnpm build` success, `pytest ml/tests` 107/107, validators 4/4, registry 7/7, evidence reproducer PASS. Commit pushed lên feature branch, PR #1 tự động cập nhật.
* **Đã hoàn thành (Phase 4C.1B.6R - Evidence Integrity, Statistical Contract and Repository Cleanup)**: Evidence integrity verified, statistical contracts closed, repository cleaned. EVAL-LEAK-001 fixed in notebook, runner verified correct. Corrected validation-only metrics: Macro-F1 0.565816, AUROC 0.611762. Bootstrap CI (source_id, 1000 iter): Macro-F1 0.5658 [0.522, 0.609], CI lower bounds > 0. Dummy baseline: 0.411718 (validation-only). Metadata baseline: NOT_MEASURED. Stage 2 gate: INSUFFICIENT_EVIDENCE (metadata baseline NOT_MEASURED, single-seed). Evidence package at research/evidence/phase-4c.1b.6/. 7 new regression tests added in ml/tests/test_eval_leakage_regression.py. All gates pass: continuity, typecheck, test, build, configs, registry. Bundle verified ready for Google Drive upload and Colab T4 smoke run. Statistical gate closed pending metadata baseline and multi-seed.
* **Đã hoàn thành (Phase 4C.1B.6R.2 - Measured Paired Bootstrap Closure)**: All placeholder statistical evidence replaced with measured results. Stage 1 validation-only artifact created (182 samples, Macro-F1 0.565816095425034). Dummy prediction artifact verified exact (Macro-F1 0.49395551257253384). Paired source-cluster bootstrap executed (91 clusters, 1000 iter, seed=42). Delta Macro-F1 = 0.07186058285250016, CI [-0.011494, 0.150825] includes 0 → superiority NOT statistically verified. Delta Balanced Acc CI [-0.010989, 0.148489] includes 0. Delta AUROC CI [0.035849, 0.200495] lower > 0. Metadata baseline NOT_MEASURED, gate_participation=false. Stage 2 gate: INSUFFICIENT_EVIDENCE. 8 regression tests added. Evidence manifest 16 artifacts. All gates pass: continuity, typecheck, test (152/1), build, configs, registry.
* **Đã hoàn thành (Phase 4C.1B.6R.3 - Final Evidence Provenance and Test-Seal Reconciliation)**: All placeholder statistical evidence replaced with measured paired bootstrap provenance. Historical root cause audit: EVAL-LEAK-001 documented as HISTORICAL_EXECUTION_PATH_UNAVAILABLE (run_smoke_local.py never in git), fault injection reproduces 141/282 symptom. Measured paired bootstrap: 91 clusters, 1000 iter, seed=42. Delta Macro-F1 = 0.07186058285250016, CI [-0.011494, 0.150825] includes 0 → superiority NOT verified. 8 new regression tests added (10 total). Historical placeholder CI [0.110, 0.198] and dummy point 0.346199 removed. Metadata baseline: NOT_MEASURED, gate_participation=false, label=HISTORICAL_PLACEHOLDER_EXCLUDED_FROM_GATE. Stage 2 gate: INSUFFICIENT_EVIDENCE. Evidence manifest 15 artifacts verified. All gates pass: continuity, typecheck, test (155/1), build, configs, registry.
* **Đã hoàn thành (Phase 4C.1B.6R.3.1 - Evidence Metadata, Dependency and T4 Execution Contract Closure)**: Manifest duplicate removed (14→13 artifacts). Notebook hash audit: file_sha256 (f26576cb...) + git_blob_oid (17c913e4...). Root cause wording: HISTORICAL_EXECUTION_PATH_UNAVAILABLE, synthetic fault injection. Dependency audit: all satisfied, pip check clean. Clean environment: 156/155/1 tests, runner dry-run OK. T4 multi-seed contract: 5 seeds (42, 1337, 2025, 3407, 9001), seed 42 re-run on T4 (EXPLORATORY_LOCAL_SMOKE excluded from T4 aggregate). Evidence manifest 9 artifacts.
* **Đã hoàn thành (Phase 4C.1B.6R.3.2 - True Clean Environment and Dependency Seal)**: Created true clean venv from requirements.txt only. JSON validation: 258 files, 0 errors. Clean venv: isolated, pip check clean, all imports OK. Hermetic tests: 149 passed, 1 skipped. Full suite: 155 passed, 1 skipped. Dependency audit: 17 deps, 12 verified, 1 runtime-provided, 3 optional, 2 missing-blocking. T4 multi-seed contract: 5 seeds (42, 1337, 2025, 3407, 9001), seed 42 re-run on T4 (EXPLORATORY_LOCAL_SMOKE excluded from T4 aggregate). Colab requirements: CUDA torch via --index-url, google.colab runtime-provided. JSON/Metadata: fixed escaping, UTC timestamps, git_head corrected, manifest counts fixed.
* **Đã hoàn thành (Phase 4C.1B.6R.3.2 - True Clean Environment and Dependency Seal)**: Created true clean venv from requirements.txt only. JSON validation: 258 files, 0 errors. Clean venv: isolated, pip check clean, all imports OK. Hermetic tests: 149 passed, 1 skipped. Full suite: 155 passed, 1 skipped. Dependency audit: 17 deps, 12 verified, 1 runtime-provided, 3 optional, 2 missing-blocking. T4 multi-seed contract: 5 seeds (42, 1337, 2025, 3407, 9001), seed 42 re-run on T4 (EXPLORATORY_LOCAL_SMOKE excluded from T4 aggregate). Colab requirements: CUDA torch via --index-url, google.colab runtime-provided. JSON/Metadata: fixed escaping, UTC timestamps, git_head corrected, manifest counts fixed.
* **Đã hoàn thành (Phase 4C.1B.6R.3.2a - Dependency Declaration Amendment and Clean Venv Recreation)**: Created requirements-dev.txt with nbformat and ipython. Recreated clean venv from requirements-dev.txt. Verified all imports, tests pass (149 hermetic / 155 full). Lint debt: 483 errors (352 fixable), classified as KNOWN_PREEXISTING_NONBLOCKING_LINT_DEBT, does not block training. T4 environment policy documented. All JSON valid (277 files, 0 errors). Ready for Gate B T4 execution.
* **Đã hoàn thành (Phase 4C.1C.0c - Build Verified One-Command T4 Training Operator)**: Khám phá CLI contract của canonical runner (`ml.training.run_phase_4c1`), xác nhận 9 flags chuẩn tắc. Ghi nhận notebook cũ là `SUPERSEDED_NOT_EXECUTABLE` (giữ nguyên byte hash `f26576cb70b6c892faaade7b225872d4bafbcfe1424b64df29564dd9e1cd28b9`). Xây dựng operator script Linux duy nhất `phase_4c1_t4_execute.sh` (19,280 bytes, SHA-256 `aae92f01e889ba0f5688c5e85c1809b8644f90ac24aac9a1884e0cd8aa174529`) tại `D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\t4_transfer\` (ngoài Git), kiểm thử tĩnh 13/13 tiêu chí an toàn PASS, niêm phong `t4-operator-script-receipt.json`. Trạng thái operator: `READY_FOR_REMOTE_EXECUTION`. Training runs mới = 0, locked-test accesses = 0.
* **Đã hoàn thành (Phase 4C.1C.0d - Build Reusable N250 Bundle and Resumable 15-Run T4 Operator)**: Nâng cấp exporter `ml/datasets/export_phase_4c1_bundle.py` tạo bundle tái sử dụng N=250 (`phase_4c1_binary_n250_reusable.tar`, 724,633,600 bytes, SHA-256 `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`) chứa đúng 250 development sources và 91 inner validation sources (0 dev/val overlap, 0 locked-test rows/sources/files, bất biến lồng nhau N50 ⊂ N100 ⊂ N250); nâng cấp validator fail-closed `ml/datasets/validate_phase_4c1_bundle.py` với 14 tiêu chí an toàn; nâng cấp canonical runner `ml/training/run_phase_4c1.py` hỗ trợ frozen cohorts 3 cỡ mẫu (`--sample-size 50, 100, 250`), tạo đủ 9 required artifacts; bổ sung 10 unit/fault-injection tests cho bundle và 2 regression tests cho runner (167 Python tests PASS); xây dựng operator Linux resumable 15 runs `phase_4c1_t4_execute_all_stage1.sh` ngoài Git. Training runs mới = 0, locked-test accesses = 0.
* **Đã hoàn thành (Phase 4C.1C.0d.1 - Pre-Execution Receipt Reconciliation and Colab Notebook V2)**: Đối soát trực tiếp bundle, manifest và assets: làm rõ `unique_source_ids` (250 dev, 91 val, tổng 341), `manifest_source_rows` (250 dev, 91 val, tổng 341), `image_samples` (500 dev, 182 val, tổng 682) và `image_assets` (500 dev, 182 val, tổng 682); sửa `reusable-bundle-receipt.json` nhất quán 100%; chuẩn hóa số đo kiểm thử tĩnh operator thành 18/18 PASS trong `t4-operator-all-stage1-receipt.json`; kiểm toán diff commit `79bb115..HEAD` xác nhận 0 file mã nguồn thực thi thay đổi (`execution-snapshot-binding-audit.json`), giữ nguyên archive code `79bb115`; niêm phong phán quyết `PRE_EXECUTION_GO_NO_GO.json` đạt verdict `GO`. Training runs mới = 0, locked-test accesses = 0.
* **Đã hoàn thành (Phase 4C.1C.0d.2 - Consolidate Phase 4C.1 Colab Notebook to One Canonical File)**: Hợp nhất toàn bộ launcher Colab vào một file chuẩn tắc duy nhất `notebooks/phase_4c1_learning_curve_colab.ipynb`. Xóa bỏ hoàn toàn notebook trùng lặp `notebooks/phase_4c1_learning_curve_colab_v2.ipynb`. Lịch sử khoa học của notebook cũ được bảo tồn qua Git history và historical SHA-256 `f26576cb...` ghi nhận trong evidence audit.
* **Đã hoàn thành (Phase 4C.1C.1 - Colab Launcher Persistent Drive Architecture)**: Nâng cấp trực tiếp notebook chuẩn tắc duy nhất `notebooks/phase_4c1_learning_curve_colab.ipynb` (15,213 bytes, SHA-256 `afcd9261af6b8049020f8e734d4367ebbfb470d3db62d562e67b2107b1f713a6`, 1 markdown + 5 code cells) sang kiến trúc Google Drive lưu trữ bền vững: (1) Drive chỉ lưu inputs (`/content/drive/MyDrive/forensics-web-lab/phase_4c1/inputs`) và persistent outputs (`.../runs/execution_79bb115`); (2) Staging 3 artifacts từ Drive xuống local `/content/forensics-transfer/` bằng atomic copy `.part` kèm kiểm tra streaming SHA-256 1 MiB chunk; (3) Dataset giải nén và đọc từ local `/content`, không train trực tiếp trên Drive; (4) Thiết lập symlink an toàn `/content/phase_4c1_outputs` trỏ vào `DRIVE_OUTPUT_DIR` (fail-closed nếu thư mục vật lý tồn tại, không dùng `rm -rf`); (5) Hỗ trợ resume an toàn qua cơ chế kiểm định 9 required artifacts của operator sau khi Colab reset; (6) Mặc định `EXECUTE=False`, `DOWNLOAD_FINAL_ARCHIVE=False`; (7) Bộ test `ml/tests/test_phase_4c1_notebook.py` đạt 14/14 PASS. Các hash của code archive `79bb115`, reusable bundle N250 và operator script giữ nguyên bất biến. Training runs mới = 0, locked-test accesses = 0, Stage 2 invocations = 0.
* **Đã hoàn thành (Phase 4C.1C.2 - Final Canonical Colab Execution Gate)**: Hoàn thiện lần cuối notebook chuẩn tắc duy nhất `notebooks/phase_4c1_learning_curve_colab.ipynb` (12,826 bytes, SHA-256 `463ae56390f93f50e246d0a1703f42f5effb3bc51fdee4cc90b5cc2d7697e7cc`, 6 markdown cells + 5 code cells = 11 cells tổng cộng) theo cấu trúc rút gọn 6 phần: (1) `# Phase 4C.1 — Learning Curve`, (2) `## Thiết lập`, (3) `## Kiểm tra môi trường và dữ liệu`, (4) `## Chuẩn bị phiên huấn luyện`, (5) `## Huấn luyện`, (6) `## Kết quả`. Loại bỏ hoàn toàn comment đánh số `# Cell X`, chuẩn hóa output chỉ dùng trạng thái ngắn (`[PASS]`, `[COPY]`, `[RUN]`, `[SKIP]`, `[READY]`). Bổ sung audit khoa học kiểm tra 5 output archives (kích thước > 0, sidecar `.sha256` khớp) và 15 `run_receipt.json` theo đúng schema thực tế (`status == "completed"`, `sample_size == size`, `seed == seed`, `stage == "frozen"`, `locked_test_access == 0`, `stage2_invocations == 0`, `validation_source_count == 91`). Bộ test `ml/tests/test_phase_4c1_notebook.py` đạt 16/16 PASS, toàn bộ test suite Python (175 PASS / 1 SKIPPED) và TypeScript (70/70 PASS) đạt chuẩn. Trạng thái: `READY_FOR_USER_COLAB_DRY_RUN`.
* **Đã hoàn thành (Phase 4C.1C.3 - Reconcile Operator Receipt Contract and Final Execution Readiness)**: Khắc phục triệt để contract gap giữa operator và schema runner thực tế: runner snapshot 79bb115 không xuất trường `execution_code_sha` trong `run_receipt.json`; operator `phase_4c1_t4_execute_all_stage1.sh` (21,662 bytes, SHA-256 `1a7570d757ccfc1b471c636f64d0f001c3b6fcc9306b9894f94186f909f1f3c4`) được cập nhật để kiểm tra binding chuẩn tắc từ `phase4c1_environment_lock.json` kèm sidecar `.sha256`, xác minh `execution_code_sha == 79bb11527d900fd387de1f41f2010c4152b7fea7` và `code_archive_sha256 == 5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5`; đồng bộ hóa kiểm tra `stage2_invocations == 0` và hỗ trợ checksum dictionary schema; bổ sung 7 fault-injections bắt buộc (binding đúng PASS, binding thiếu FAIL, binding sai FAIL, valid run resumable, partial run fail-closed, locked-test access FAIL, stage 2 invocation FAIL) đạt 21/21 assertions PASS; notebook chuẩn tắc đạt 16/16 tests PASS; cập nhật `PRE_EXECUTION_GO_NO_GO.json` verdict `GO`; 185 tests Python (10 tests mới trong `test_phase_4c1_operator.py`) và 70 tests TypeScript PASS. Trạng thái: `READY_FOR_USER_COLAB_DRY_RUN`. Training runs mới = 0, locked-test accesses = 0, stage 2 invocations = 0.
* **Đã hoàn thành (Phase 4C.1C.4 - Align Colab Drive Path with Physical Layout)**: Điều chỉnh duy nhất đường dẫn Google Drive trong notebook chuẩn tắc `notebooks/phase_4c1_learning_curve_colab.ipynb` (12,842 bytes, SHA-256 `22a251b6b784fdf614b7a256a63fc908d6fe799f0ce46148d14e7932db03d8e8`, 11 cells) sang đúng bố cục thực tế của người dùng: `/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1/` (bổ sung cấp thư mục `Colab Notebooks`); giữ nguyên toàn bộ logic streaming SHA-256, staging `/content/forensics-transfer`, symlink output và 3 artifacts bất biến; bộ test `ml/tests/test_phase_4c1_notebook.py` đạt 16/16 PASS; cập nhật `PRE_EXECUTION_GO_NO_GO.json`. Training runs mới = 0, locked-test accesses = 0, stage 2 invocations = 0. Trạng thái: `READY_FOR_USER_COLAB_DRY_RUN`.
* **Đã hoàn thành (Phase 4C.1C.5 - Make Colab Input Preflight Read-Only and Fail-Closed)**: Khắc phục lỗi thiết kế trong notebook chuẩn tắc `notebooks/phase_4c1_learning_curve_colab.ipynb` (13,360 bytes, SHA-256 `736f7a77d2559835a527833afe50d96022baa793c658141b38f001443513030b`, 11 cells): xóa bỏ hoàn toàn `DRIVE_ROOT.mkdir()`, `DRIVE_INPUT_DIR.mkdir()` và `DRIVE_OUTPUT_DIR.mkdir()` khỏi cell preflight; thiết lập kiểm tra fail-closed chỉ đọc với `DRIVE_ROOT.is_dir()` và `DRIVE_INPUT_DIR.is_dir()`; nếu thiếu artifact thì thông báo lỗi kèm đường dẫn chính xác và danh sách file thực tế hiện có trong `DRIVE_INPUT_DIR`; việc tạo thư mục đầu ra `DRIVE_OUTPUT_DIR.mkdir()` chỉ được phép thực thi ở cell chuẩn bị (staging) sau khi toàn bộ 3 artifacts đã vượt qua kiểm tra toàn vẹn; bộ test `ml/tests/test_phase_4c1_notebook.py` bổ sung 2 test mới đạt 18/18 PASS, toàn bộ test suite Python (187 PASS / 1 SKIPPED) và TypeScript (70/70 PASS) đạt chuẩn; cập nhật `PRE_EXECUTION_GO_NO_GO.json`. Training runs mới = 0, locked-test accesses = 0, stage 2 invocations = 0. Trạng thái: `READY_FOR_USER_COLAB_DRY_RUN`.
* **Đã hoàn thành (Phase 4C.1C.7 - Fix Colab Runtime Operator Venv with Without-Pip)**: Khắc phục triệt để lỗi tạo venv trên Colab T4 do `ensurepip` thất bại; cập nhật operator chuẩn tắc `phase_4c1_t4_execute_all_stage1.sh` (23,344 bytes, SHA-256 `6da81e2bf449f7d98492f8960a4b6330487bdcc5f96a39a2f02ab4423af2d9dc`) sử dụng `"$SYS_PY3" -m venv --system-site-packages --without-pip "$VENV_ROOT"`.
* **Đã hoàn thành (Phase 4C.1C.8 - Eliminate Venv, Minimal Colab Operator, and Sync Canonical Notebook)**: Loại bỏ hoàn toàn virtual environment khỏi operator `phase_4c1_t4_execute_all_stage1.sh` (25,470 bytes, SHA-256 `16cc4655c77ca5931290d5dd3c2612e40af1084cd0f0c1320e8fc68260e63daa`); sử dụng trực tiếp Python và PyTorch CUDA sẵn có của Colab; đồng bộ canonical notebook 5 cells; cập nhật toàn bộ test suite.
* **Đã hoàn thành (Phase 4C.1C.9 — Reconcile Bundle Hash Contract, Verify Resume End-to-End, and Harden Colab Notebook)**: Khắc phục triệt để và dứt điểm lỗi nhầm lẫn ngữ nghĩa giữa TAR archive SHA-256 (`BUNDLE_ARCHIVE_SHA256`: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`) và canonical extracted-bundle content SHA-256 (`BUNDLE_CONTENT_SHA256`: `c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`); nâng cấp operator `phase_4c1_t4_execute_all_stage1.sh` (29,794 bytes, SHA-256 `e105441ed20a40da55cc8db4fd7f83440182cae8f951d46f03939b08504efc14`) với: (1) kiểm tra content SHA trích xuất từ `bundle_receipt.json`, (2) hỗ trợ backward-compatible environment lock (coi legacy `bundle_sha256` là archive hash mà không phá hủy lock đã tồn tại), (3) verifier đối chiếu `run_receipt.bundle_sha256 == BUNDLE_CONTENT_SHA256` và `lock_archive_sha == BUNDLE_ARCHIVE_SHA256`, (4) tập trung hóa xử lý lỗi qua hàm `fail_operator` ghi đầy đủ metadata vào `OPERATOR_FAILURE.json` và trap ERR, (5) ghi log console bền vững qua `exec > >(tee -a "$CONSOLE_LOG") 2>&1`; nâng cấp canonical notebook `notebooks/phase_4c1_learning_curve_colab.ipynb` (11,705 bytes, SHA-256 `e8e17a48120e83ecfd26ff49a07a3ad5126df3a103e384c21a3f84348b673792`, 5 cells, `EXECUTE = True`) bắt `CalledProcessError` để in console log path, nội dung `OPERATOR_FAILURE.json` và tối thiểu 100 dòng cuối log trước khi raise error; bảo toàn nguyên vẹn run thật N=50, seed=42 (`phase4c1-stage1-n50-seed42`, 23 epochs, best epoch 18, ckpt SHA `b5fa53bf...`, 91 validation sources, Macro-F1 0.5415, AUROC 0.5607, Brier 0.2478, ECE 0.0265); chứng minh cơ chế resume công nhận run là `COMPLETED_VALID`, skip N50 seed42 và chuẩn bị chạy tiếp N50 seed1337; cập nhật toàn diện test suites (25 operator tests, 17 notebook tests, 201 Python tests PASS); receipts cập nhật UTC thật; audited HEAD commit `b842acc...`. Trạng thái: `READY_FOR_USER_COLAB_RESUME`.
* **Hiện trạng nghiên cứu**: Checkpoint smoke đã lưu trong `models/research/phase-4c.0/`; 1 remote run completed trên Colab T4 (`phase4c1-stage1-n50-seed42`, COMPLETED_VALID); 14 remaining runs; locked-test evaluations = 0; stage 2 invocations = 0. Reusable N250 bundle và code archive giữ nguyên bất biến. Trạng thái: `READY_FOR_USER_COLAB_RESUME`.
* **Công việc tiếp theo**: Người dùng upload lại đúng 2 file lên Google Drive:
  1. Operator `phase_4c1_t4_execute_all_stage1.sh` vào `Colab Notebooks/forensics-web-lab/phase_4c1/inputs/`
  2. Notebook `phase_4c1_learning_curve_colab.ipynb` vào `Colab Notebooks/forensics-web-lab/phase_4c1/`
  Mở notebook trên Colab T4 và chọn **Run all**. Cơ chế resume sẽ tự động phát hiện run N50 seed42 đã hoàn thành hợp lệ (SKIP) và tiếp tục thực thi từ run N50 seed1337 cho đến hết 15 runs. Tuyệt đối không cần upload lại code archive và dataset bundle.




