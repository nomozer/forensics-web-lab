# Trạng thái Hiện tại: Forensics Web Lab (Current State)

> **Tài liệu đọc đầu tiên bắt buộc cho mọi phiên làm việc AI mới.**  
> **Documented through substantive commit**: `110fef1`  
> **Ending commit Phase 4C.0**: `110fef1`  
> **Phase hoàn thành gần nhất**: Phase 4C.0a — Smoke Evidence Reconciliation and Main-Readiness Gate  
> **Branch**: `feat/production-ai-image-forensics`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không commit trực tiếp)  
> **Working tree**: clean  
> **Tuyên bố khoa học tối thượng**:  
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
* **Trọng số mô hình đã tải / huấn luyện**: Đã tải `mobilenet_v3_small-047dcff4.pth` (10,306,551 bytes) từ torchvision chính thức; 1 checkpoint smoke `smoke_mobilenetv3_small_seed42.pt` (5,624,201 bytes) tại `models/research/phase-4c.0/` (quarantined trong Research Track).
* **Số lượt huấn luyện (Training runs)**: `1` (đúng 1 single-seed smoke run N=50, seed 42; 0 full learning-curve runs; 0 runs mới trong Phase 4C.0a).
* **Chỉ số khoa học**: Stage 0 baselines (bảng thẩm quyền Phase 4C.0a: Dummy Macro-F1 0.5932, Metadata-only 0.5098, DSP-only 0.5653); Checkpoint smoke tái đánh giá N=50 (Val Macro-F1 0.5035, AUROC 0.5278, Brier 0.2504, ECE 0.0347). Toàn bộ được gắn nhãn `development-only exploratory baseline` và `research-smoke-only`. Full multi-seed evaluation: `not evaluated`. Trong cấu hình TGIF SD2-sp smoke đăng ký trước (N=50, seed=42), checkpoint frozen MobileNetV3-Small thể hiện near-chance discrimination trên inner-validation; single exploratory run này không suy diễn giới hạn tổng quát của biểu diễn ImageNet hay mô hình thị giác fine-tuned. Stratified Dummy dùng `random_state=42` là một stochastic reference realization (không phải theoretical chance level); cùng prediction vector xuất hiện ở N=50, 100, 250 do class prior 1:1 và dummy không học đặc trưng ảnh. Mọi so sánh confirmatory sau này bắt buộc dùng nhiều seeds hoặc bootstrap/permutation.
* **Quy chuẩn trung thực & Tái lập**: All reported quantities are classified explicitly as measured, reproduced, projected, estimated, or not measured. Each classification is tied to its evidence artifact and execution environment. Metrics deterministically reproduced in the recorded software environment using the registered artifacts and prediction-vector hashes.

---

## 6. Kết quả Kiểm thử & Bản dựng Gần nhất (Latest Verification)

* **TypeScript & Continuity Test Suite (`pnpm test`)**: 70/70 tests passing (57 vitest tests trên 6 packages + 13 continuity checker unit tests).
* **Python Test Suite (`pytest ml/tests -v`)**: 107/107 tests passing (bao gồm 9 Phase 4C.0 gate tests và 8 Phase 4C.0a reconciliation & main-readiness tests).
* **Dataset Registry Validation**: 7/7 datasets valid.
* **Pilot Config Validator Gate**: 4/4 pilot configs valid (`pilot_a_binary_preregistered.yaml` hash `727fc316...`).
* **Live Evidence Reproducer (`scripts/reproduce-phase-4b3-evidence.py --verify`)**: All invariants, byte counts, and SHA-256 digests verified.
* **Production Web Build (`pnpm build`)**: Exit code 0, bundle tối ưu hợp lệ (55 modules, 4.08s).
* **Continuity Enforcement Gate (`pnpm continuity:check`)**: `CONTINUITY_CHECK: PASS`.
* **Clean Link Invariance**: 0 machine-local links (`file:///`, `C:\`, `D:\`) trong toàn bộ markdown và evidence repository.

* **Phase CI.2 — CI TypeScript Repair**: TypeScript CI root cause (TS18047) đã được tái hiện và sửa bằng explicit guard trong test. PR #1 đang mở, Python local 107/107 PASS, Python CI root cause vẫn UNVERIFIED cho đến khi có log trực tiếp. Scientific model-performance claims giữ nguyên trạng thái trước đó.
* **Phase CI.3 — CI Python Import Root Repair**: Python CI failure root cause đã được xác nhận là working-directory/import-root mismatch (running `pytest tests/` from `ml/` directory causes `ModuleNotFoundError: No module named 'ml'`). Workflow được căn chỉnh để chạy `python -m pytest ml/tests -v` từ repository root. Local verification: 107/107 PASS. Python source, tests, dependencies giữ nguyên.
* **Phase CI.4 — Hermetic CI & Research Artifact Gate Separation**: Python tests được phân loại thành Hermetic CI Gate (chạy trên clean runner, không phụ thuộc artifact nghiên cứu) và Local Research Artifact Gate (yêu cầu manifest, dataset metadata, checkpoint bị cách ly ngoài Git). 6 test yêu cầu checkpoint/manifest được gắn marker `requires_research_artifact`. Test zero-network được mock disk space. Sampler subprocess test chuyển sang synthetic fixture. Metadata-only contract test thêm mới (checkpoint metadata contract). CI workflow chạy hermetic gate mặc định (`-m "not requires_research_artifact"`), collect-only cho artifact tests. Kết quả local: Full 108/108, Hermetic 102 passed + 6 deselected, Artifact 6 passed. Generic CI chạy hermetic gate.
* **Phase CI.4B — Evidence Reconciliation and CI Verification**: Tất cả evidence CI.4 được đối soát với số liệu đo thực tế. test-summary.json sửa từ 101/7 thành 102/6. clean-checkout-summary.json trạng thái PROJECTED_NOT_MEASURED. Marker membership được xác minh bởi pytest collection. Test arithmetic khớp: 108 = 102 + 6. Local verification: 108/108, Hermetic 102/102, Artifact 6/6. GitHub Actions clean-runner verification đang pending. PR #1 checks đang chạy.
* **Phase 4C.1A — Learning-Curve Execution Gate & Preregistration Audit**: Branch `research/phase-4c1-learning-curve` tạo từ foundation tag `v0.1.0-research-foundation`. Dataset Option P sẵn sàng (684 sources, nested N=50/100/250). Preregistration `pilot_a_binary_preregistered.yaml` (hash `727fc316...`) đã được audit. 15 training runs (3 sizes × 5 seeds) + 90 baselines được lên kế hoạch. Locked test SEALED (0 evaluations). Checkpoint smoke sẵn sàng. Execution plan và resource budget đã tạo. **Chờ phê duyệt người dùng để thực thi 15 training runs.**
* **Phase 4C.1A.1 — Experiment Evidence Reconciliation & Colab Package Preparation**: Đối soát preregistration hash (LF vs CRLF line endings, git content identical), sửa experiment arithmetic (90 method-size-seed cells, 75 approved + 15 conditional Stage 2), khóa N definition (unique source_id), định nghĩa Stage 2 gate machine-readable, phân loại resource accounting (Measured/Estimated/Projected/Unknown), tạo Colab execution package (notebook, bundle exporter/validator, requirements, guide). Continuity checker cần sửa alphabetical sorting bug. **Chờ phê duyệt để build/upload Colab bundle.**

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
* **Hiện trạng nghiên cứu**: Checkpoint smoke đã lưu trong `models/research/phase-4c.0/`; model weights và checkpoint được cách ly trong Research Track; locked-test evaluations = 0; full learning curve (15 runs) chưa thực thi.
* **Công việc tiếp theo**: Theo dõi CI mới trên PR #1. Python CI root cause vẫn UNVERIFIED cho đến khi có log trực tiếp. Sau khi CI ổn định, tag release `v0.1.0-research-foundation` và chuẩn bị branch `research/phase-4c1-learning-curve` cho Phase 4C.1.


