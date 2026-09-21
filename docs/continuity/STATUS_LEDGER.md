# Sổ Quản lý Tiến độ Dự án: Forensics Web Lab (Status Ledger)

> **Thư mục**: `docs/continuity/STATUS_LEDGER.md`  
> **Mục đích**: Biên niên sử cô đọng từng giai đoạn phát triển và nghiên cứu từ Phase 0 đến nay.  
> **Quy ước**: Giai đoạn mới nhất nằm ở trên cùng; mỗi giai đoạn không quá 20 dòng; chi tiết kiểm chứng nằm tại `research/evidence/`.

## Phase 4B.2 — Controlled Option-P Acquisition, Pairability Verification and Split Freeze
- **Mục tiêu**: Thu nạp có kiểm soát Option P (4 archives, 5.88 GB), safe extraction, kiểm toán decode và pairability ba thành phần, đóng băng deterministic splits ($N=50,100,250$) và class-coverage guard.
- **Starting commit**: `82a5266`
- **Thay đổi chính**: Ban hành `pilot-a-tgif-option-p.v1.json` và `user-approval.json`; tải an toàn 4 archive Option P (5,879,502,782 bytes) từ duy nhất `cloud.ilabt.imec.be`, khớp 100% SHA-256; safe tar extraction 6,156 ảnh (5,890,096,035 bytes) chống Tar Slip; `verify_option_p.py` kiểm toán PIL decode 100% PASS; pairability status `verified` (684 instances, 0 missing, 0 dimension mismatch); đóng băng tất định seed 42: `development_train` (250 sources), `inner_validation` (91 sources), `locked_test` (343 sources, SHA-256 seal `519e7a0e...`); zero cross-split leakage; learning curve lồng nhau $N=50 \subset N=100 \subset N=250$; sinh `manifest_pilot_a_option_p.csv`; bổ sung 11 unit tests (`test_option_p_protocol.py`, 66/66 Python tests passed).
- **Kiểm tra kỹ thuật**: `pnpm test` (70/70 passed), `pytest ml/tests` (66/66 passed), `pnpm build` (exit 0), `pnpm continuity:check` (PASS), dataset registry (7/7 valid), model weights: 0 bytes, training runs: 0.
- **Kết quả khoa học**: Xác minh toàn diện dữ liệu Option P cho bài toán con authentic–AI-edited và localization; khóa hoàn toàn locked-test; class-coverage guard chặn pipeline 3 lớp khi chưa có fully_generated; detection metrics: `not evaluated`.
- **Evidence**: `research/evidence/phase-4b.2/` (`acquisition-binding.json`, `user-approval.json`, `download-receipt.json`, `archive-inventory.json`, `extraction-summary.json`, `content-manifest-summary.json`, `pairability-audit.json`, `split-lock.json`, `class-coverage-audit.json`, `test-summary.json`, `build-summary.json`, `evidence-manifest.json`, `PHASE_REPORT.md`).
- **Giới hạn**: Dữ liệu chỉ gồm 2 lớp (authentic, ai_edited); 3 lớp chưa runnable; model weights = 0 byte; training runs = 0.
- **Quyết định tiếp theo**: Chuyển sang Phase 4C để huấn luyện baseline binary classifier và thực thi learning curve trên frozen splits.

---

## Phase 4B.1 — Small-Data Feasibility and Pairability Audit
- **Mục tiêu**: Kiểm toán căn nguyên chênh lệch 2.242/3.124, khóa đơn vị độc lập là `source_id`, khảo sát metadata remote `orig`/`sd2-sp`, thiết lập giao thức ít dữ liệu và 3 phương án tải.
- **Starting commit**: `3be606c`
- **Thay đổi chính**: Phát triển `ml/datasets/audit_source_ids.py` xác minh 31.238 masks xuất phát từ 2.242 unique COCO sources (1.558 train, 341 val, 343 test) và 3.124 category instances; zero cross-split leakage; khảo sát WebDAV remote 10 requests (41.549 bytes < 5 MiB ceiling) xác định mỗi component gồm 3 split archives độc lập; pairability status `pending-content-acquisition`; ban hành `docs/SMALL_DATA_PROTOCOL.md` và `ml/configs/pilot_a_learning_curve.yaml` ($N=50,100,250$); mở rộng `ml/configs/validator.py`; xây dựng 3 phương án dữ liệu (Option S/P/F) và đề xuất Option P (5.88 GB, 684 sources); bổ sung 9 tests pytest (55/55 passed).
- **Kiểm tra kỹ thuật**: `pnpm test` (70/70 passed), `pytest ml/tests` (55/55 passed), `pnpm build` (exit 0), `pnpm continuity:check` (PASS), config validator (3/3 valid), dataset content: 0 bytes, model weights: 0 bytes.
- **Kết quả khoa học**: Xác minh nguyên nhân 2.242/3.124 do 571 ảnh train có đa danh mục; chuẩn hóa đơn vị độc lập `source_id`; khóa ranh giới chống shortcut; detection metrics: `not evaluated`.
- **Evidence**: `research/evidence/phase-4b.1/` (`environment.json`, `network-accounting.json`, `source-id-audit.json`, `source-id-collision-report.json`, `tgif-orig-remote-inventory.json`, `tgif-sd2-sp-remote-inventory.json`, `remote-metadata-accounting.json`, `pairability-audit.json`, `small-data-options.json`, `test-summary.json`, `build-summary.json`, `evidence-manifest.json`, `PHASE_REPORT.md`).
- **Giới hạn**: Chưa tải ảnh orig và sd2-sp; pairability content pending; model chưa huấn luyện; metrics chưa đánh giá.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt Option P (tải validation + test split: 5.88 GB) cho Pilot A.

---

## Phase 4B.0 — TGIF Masks Live Acquisition Smoke
- **Mục tiêu**: Tải thực nghiệm duy nhất `tgif-masks`, thực thi trần mạng 64 MiB, kiểm toán toàn diện cardinality và thuộc tính mask.
- **Starting commit**: `8de2151`
- **Implementation snapshot commit**: `84aa127`
- **Ending commit**: `3be606c`
- **Thay đổi chính**: Mở rộng `ml/datasets/acquire.py` với `--component` và `--max-download-bytes`; bổ sung `HostnameRestrictedRedirectHandler` (giới hạn `cloud.ilabt.imec.be`); kiểm tra trần mạng trước và trong stream; bổ sung 8 unit tests an toàn (23 tests suite); ghi nhận phê duyệt máy đọc `user-approval.json`; thực hiện live smoke tải `tgif-masks` (42,327,429 bytes nhận, archive SHA-256 `62c89a65...`); trích xuất an toàn 3 archive con; xây dựng `ml/datasets/audit_masks.py` kiểm toán 31,238 file mask PNG (141,559,934 bytes); sinh manifest cục bộ `masks-manifest.jsonl` (31,238 bản ghi) và receipt `acquisition-receipt.json`.
- **Kiểm tra kỹ thuật**: `pnpm test` (70/70 passing), `pytest ml/tests` (46/46 passing), `pnpm build` (exit 0), `pnpm continuity:check` (PASS), 0 file nhị phân trong Git.
- **Kết quả khoa học**: Xác minh cardinality mask TGIF là 31,238 files (12,495 bbox, 12,495 segm, 6,248 generic_mask) trên 2,242 `source_id`; chuyển `mask_count`, `filename_convention`, `source_id_extraction_method` sang verified; các thành phần `orig`, `sd2-sp` tiếp tục locked; detection metrics: `not evaluated`.
- **Evidence**: `research/evidence/phase-4b.0/` (`user-approval.json`, `network-accounting.json`, `acquisition-summary.json`, `tgif-masks-inventory.json`, `tgif-masks-validation.json`, `mask-manifest-summary.json`, `test-summary.json`, `build-summary.json`, `evidence-manifest.json`, `PHASE_REPORT.md`).
- **Giới hạn**: Chỉ `tgif-masks` được tải; external dataset content: 42,327,429 bytes; `orig` và `sd2-sp` chưa tải; model weights: 0 bytes; training runs: 0.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt tải hai component còn lại (`tgif-orig`: ~7.30 GB, `tgif-sd2-sp`: ~18.58 GB) để hoàn thiện Pilot A.

---

## Phase 4A.5 — Model-Agnostic Continuity Enforcement
- **Mục tiêu**: Thiết lập cơ chế continuity tự động, chuẩn hóa hợp đồng agent, tích hợp checker CI, đóng chính xác Phase 4A.4.
- **Starting commit**: `a480776`
- **Ending commit**: `8de2151`
- **Implementation snapshot commit**: `311cfd3`
- **Thay đổi chính**: Đóng Phase 4A.4 với ending commit `a480776`; đổi thuật ngữ sang group-isolation invariant; xác thực SHA-256 acquisition plan; ban hành Continuity Contract trong `AGENTS.md` (read order, change matrix, phase closure); phát triển `scripts/continuity-check.mjs` (kiểm tra diff, 3 file chuẩn, duplicate files, local links, placeholders, line limits); viết 13 unit tests trong `scripts/__tests__/continuity-check.test.mjs`; tích hợp CI `pnpm continuity:check`.
- **Kiểm tra kỹ thuật**: `pnpm continuity:check` (PASS), `pnpm test` (70/70 passing: 57 vitest + 13 continuity tests), `pytest ml/tests` (38/38 passing), `pnpm build` (exit 0), config validator passing, acquisition dry-run passing (0 byte external network).
- **Kết quả khoa học**: Cơ chế continuity bảo toàn trung thực khoa học, ngăn ngừa AI hallucination về trạng thái dự án, bảo toàn ranh giới Zero-Egress và Dual-Track.
- **Evidence**: `research/evidence/phase-4a.5/` (`EV-PHASE4A4-CLOSURE-001`, `EV-CONTINUITY-CONTRACT-001`, `EV-CONTINUITY-CHECKER-001`, `EV-CONTINUITY-TESTS-001`, `EV-CI-CONTINUITY-001`).
- **Giới hạn**: External dataset content: 0 bytes; model content: 0 bytes; training runs: 0.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt lệnh tải dữ liệu thật cho Pilot A (`pilot-a-tgif.v1.json`) tại Phase 4B.

---

## Phase 4A.4 — Evidence Correction and Acquisition Safety Gate
- **Mục tiêu**: Hiệu chỉnh minh chứng và tuyên bố khoa học, kiểm toán cardinality TGIF, lập acquisition plan có SHA-256, hoàn thiện bộ tải dữ liệu an toàn offline.
- **Starting commit**: `9ab0e67`
- **Ending commit**: `a480776`
- **Implementation snapshot commit**: `a480776`
- **Thay đổi chính**: Sửa commit reference Phase 4A.3 (`9ab0e67`); loại bỏ các tuyên bố triệt tiêu 100% shortcut; chuẩn hóa Metadata Guard sang Paired Stratified Bootstrap 95% CI ($\Delta\text{Macro-F1} > 0$); kiểm toán cardinality TGIF (`tgif-cardinality-audit.json`); ban hành schema và plan máy đọc (`datasets/acquisition-plans/pilot-a-tgif.v1.json`, SHA-256: `7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`); nâng cấp safety gates trong `ml/datasets/acquire.py` (free disk, `.part`, resume, checksum, safe zip extraction, staging, receipt); bổ sung 15 unit tests offline.
- **Kiểm tra kỹ thuật**: `pnpm test` (57/57 passing), `pytest ml/tests` (38/38 passing), `pnpm build` (exit 0), validator passing, external dataset/model content: 0 bytes.
- **Kết quả khoa học**: Xác minh TGIF 3,124 orig, 18,744 sd2-sp; masks ước tính ~6,248; matched-pair claim được chuẩn hóa; metadata guard có cơ sở thống kê chặt chẽ.
- **Evidence**: `research/evidence/phase-4a.4/` (`EV-PHASE4A3-CORRECTION-001`, `EV-TGIF-CARDINALITY-001`, `EV-ACQUISITION-PLAN-001`, `EV-DOWNLOADER-SAFETY-001`, `EV-BOOTSTRAP-GUARD-001`).
- **Giới hạn**: External dataset content: 0 bytes; model content: 0 bytes; training runs: 0.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt acquisition plan `pilot-a-tgif.v1.json` cho live acquisition.

---

## Phase 4A.3 — Scientific Pilot Protocol & Label-Semantics Gate
- **Mục tiêu**: Đóng băng định nghĩa 3 nhãn, giải quyết dứt điểm ngữ nghĩa TGIF `sp`/`fr`, thiết kế pilot hai nhánh độc lập (Pilot A/B/C), xây dựng cấu hình máy đọc và validator.
- **Starting commit**: `0c42f3c`
- **Ending commit**: `9ab0e67`
- **Thay đổi chính**: Tạo `docs/PILOT_PROTOCOL.md` (label gate, `sp` vs `fr` audit, anti-shortcut protocol); tạo `ml/configs/pilot_tgif_edit.yaml` và `ml/configs/pilot_genimage_generated.yaml`; tạo validator `ml/configs/validator.py`; cập nhật `ml/datasets/acquire.py` với lệnh `--pilot`; backfill báo cáo Phase 4A.2; viết unit test kiểm định 3 lớp, chống rò rỉ và dry-run.
- **Kiểm tra kỹ thuật**: `pnpm test` (57/57 passing), `pytest` (23/23 passing), `pnpm build` (exit 0), 0 machine-local links, 0 byte external data.
- **Kết quả khoa học**: Xác minh `sp` là `ai_edited`; loại trừ `fr` khỏi `fully_generated` (quarantined); tách Pilot A (TGIF matched pairs) và Pilot B (GenImage); Pilot C kèm metadata baseline; toàn bộ metric là `not evaluated`.
- **Evidence**: `research/evidence/phase-4a.3/` (`EV-LABEL-GATE-001`, `EV-TGIF-SEMANTICS-001`, `EV-PILOT-DESIGN-001`, `EV-SHORTCUT-PROTOCOL-001`, `EV-PILOT-CONFIGS-001`, `EV-ACQUISITION-DRYRUN-001`).
- **Giới hạn**: Chưa tải dữ liệu thật; chờ phê duyệt `NEXT APPROVAL REQUEST`.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt tải dữ liệu thật cho Pilot A hoặc Pilot B.

---

## Phase 4A.2 — Research Scope Freeze, Continuity Protocol & TGIF Audit
- **Mục tiêu**: Hợp nhất tài liệu continuity sang bộ 3 file thống nhất, đóng băng phạm vi khoa học (3 lớp + uncertain), khảo sát TGIF/TGIF2 và đề xuất phương án pilot.
- **Starting commit**: `863a491`
- **Ending commit**: `0c42f3c`
- **Thay đổi chính**: Tạo `docs/continuity/` (CODE_INDEX, CURRENT_STATE, STATUS_LEDGER); xóa các file continuity cũ; cập nhật `AGENTS.md`; đóng băng 4 câu hỏi nghiên cứu và hệ thống chỉ số trong `docs/RESEARCH_PLAN.md` và `docs/EVALUATION.md`; thêm TGIF/TGIF2 vào `datasets/registry.json`; khảo sát 3 share Nextcloud TGIF (65.4 GB, 110 GB, 73 GB; hỗ trợ tải lẻ từng thư mục); đề xuất Pilot B (3,000 ảnh) cho khóa luận.
- **Kiểm tra kỹ thuật**: `pnpm test` (56/56 passing), `pnpm build` (exit code 0), `pytest` (15/15 passing), 0 machine-local links, 0 byte external data.
- **Kết quả khoa học**: Khóa phạm vi 3 lớp (`authentic`, `fully_generated`, `ai_edited`); localization là mục tiêu phụ; toàn bộ metric tiếp tục là `not evaluated`.
- **Evidence**: `research/evidence/phase-4a.2/` (`EV-CONTINUITY-001`, `EV-RESEARCH-SCOPE-001`, `EV-RQ-METRIC-MAP-001`, `EV-TGIF-METADATA-001`, `EV-TGIF-LICENSE-001`, `EV-DATASET-SHORTCUT-RISK-001`, `EV-PILOT-PROPOSAL-001`).
- **Giới hạn**: Chưa tải dữ liệu thật; chờ người dùng phê duyệt phương án Pilot B.
- **Quyết định tiếp theo**: Người dùng xem xét phê duyệt tải thư mục con của TGIF cho Pilot B ở phase tiếp theo.

---

## Phase 4A.1 — Acquisition Feasibility & Residual Evidence Correction
- **Mục tiêu**: Đính chính mức độ minh chứng và giấy phép tồn dư, khảo sát Google Drive GenImage, xác định tính khả thi của tập con và thiết lập đề xuất 3 cấp.
- **Starting commit**: `a0e71d5`
- **Ending commit**: `863a491`
- **Thay đổi chính**: Tách bạch điều khoản GenImage (`derivativeWeights: unclear`, `productionPromotion: prohibited-by-project-policy`); đổi `synthetic-smoke` sang `fixture-only`; khóa dataset chưa xác minh (`sagi-d`, `raid`, `realhd`); khảo sát Google Drive GenImage (BigGAN 8 file split zip ~24 GB nén, Kết luận B: không tải lẻ được); bổ sung `--metadata-only` và khóa tải ngoài trong `acquire.py`.
- **Kiểm tra kỹ thuật**: `pnpm test` (56/56 passing), `pytest` (15/15 passing), build exit 0, 0 byte external data.
- **Kết quả khoa học**: Bác bỏ giả định `<50 MB` cho GenImage chính thức; BigGAN là first fully inventoried archive.
- **Evidence**: `research/evidence/phase-4a.1/` (`EV-LICENSE-WEIGHTS-INTERPRETATION-001`, `EV-FIXTURE-TRACK-001`, `EV-GENIMAGE-REMOTE-METADATA-001`, `EV-GENIMAGE-SUBSET-FEASIBILITY-001`, `EV-EXTERNAL-DOWNLOAD-LOCK-001`, `EV-LOCAL-LINK-SCAN-001`).
- **Giới hạn**: Nguồn GenImage yêu cầu dung lượng tải lớn (~24 GB).
- **Quyết định tiếp theo**: Khảo sát nguồn dữ liệu inpainting có khả năng tải lẻ (TGIF).

---

## Phase 4A.0 — Data License Correction, Track Isolation & Acquisition Dry-Run
- **Mục tiêu**: Đính chính giấy phép dataset, thiết lập chính sách Dual-Track (ADR-0006), xây dựng Dataset Registry và bộ sinh fixture nội bộ.
- **Starting commit**: `e36cdb6`
- **Ending commit**: `a0e71d5`
- **Thay đổi chính**: Ban hành ADR-0006; phân chia `data/research/` và `data/product/`; tạo `datasets/registry.json` và schema manifest; tạo `ml/tests/fixtures/smoke_generator.py`; phát triển `ml/datasets/acquire.py` (dry-run only); bổ sung contamination guards trong TypeScript và Python.
- **Kiểm tra kỹ thuật**: `pnpm test` (51/51 passing), `pytest` (11/11 passing), build exit 0, 0 byte external data.
- **Kết quả khoa học**: Loại bỏ ngộ nhận về "GenImage Mini"; xác lập ranh giới dữ liệu không thương mại.
- **Evidence**: `research/evidence/phase-4a.0/` (`EV-LICENSE-GENIMAGE-001`, `EV-LICENSE-REALHD-001`, `EV-DATA-REGISTRY-001`, `EV-DATA-ISOLATION-001`, `EV-CONTAMINATION-001`, `EV-SMOKE-FIXTURE-001`).
- **Giới hạn**: Chưa kiểm tra trực tiếp kho lưu trữ từ xa của GenImage.
- **Quyết định tiếp theo**: Triển khai Phase 4A.1 khảo sát metadata remote.

---

## Phase 3.6 — Evidence Hardening & Claim Correction
- **Mục tiêu**: Loại bỏ đường dẫn máy cá nhân, sửa các tuyên bố quá mức về model và parity, thiết lập Evidence Register máy đọc.
- **Starting commit**: `cd59136`
- **Ending commit**: `e36cdb6`
- **Thay đổi chính**: Chuẩn hóa 8 trạng thái minh chứng; loại bỏ toàn bộ absolute path; đổi tên `CAND-C2-INHOUSE-MNV3` thành `ARCH-C2-INHOUSE-MNV3`; ghi nhận model size là `estimated`, parity là `pipeline-only`, metrics là `unverified`; tạo schema `evidence-manifest.v1.schema.json` và `docs/EVIDENCE_REGISTER.md`.
- **Kiểm tra kỹ thuật**: 40/40 TS tests, 5/5 Py tests, 0 machine-local links.
- **Kết quả khoa học**: Đảm bảo 100% tính trung thực khoa học, không có kết luận khống.
- **Evidence**: `research/evidence/phase-3.6/evidence-manifest.json` (`EV-GIT-001`, `EV-TEST-TS-001`, `EV-NOMODEL-001`, `EV-REGISTRY-001`).
- **Giới hạn**: Chưa có dataset thật để kiểm chứng mô hình.
- **Quyết định tiếp theo**: Thẩm định bản quyền và phân luồng dữ liệu tại Phase 4A.0.

---

## Phase 3.5 — Persistent Memory, Truth Audit & Honest No-Model State
- **Mục tiêu**: Thiết lập trí nhớ dự án bền vững trong Git, kiểm chứng tuyên bố triển khai và bảo đảm trạng thái trung thực khi không có model.
- **Starting commit**: `6555b02`
- **Ending commit**: `cd59136`
- **Thay đổi chính**: Tạo `AGENTS.md`, các tài liệu tracking ban đầu (sau này hợp nhất thành `docs/continuity/`), `docs/MODEL_ACQUISITION_GATE.md`; cập nhật `FusionCalibrator` và UI hiển thị "Model not installed", verdict `uncertain`, `confidence: null`.
- **Kiểm tra kỹ thuật**: Unit test hợp đồng no-model vượt qua; không sinh xác suất ngẫu nhiên.
- **Kết quả khoa học**: Hệ thống không giả mạo kết quả giám định khi chưa có model.
- **Evidence**: `docs/IMPLEMENTATION_TRUTH_AUDIT.md`, `packages/shared/src/__tests__/contracts.test.ts`.
- **Giới hạn**: Không có model AI hoạt động trong production.
- **Quyết định tiếp theo**: Chuẩn hóa minh chứng và loại bỏ đường dẫn cá nhân tại Phase 3.6.

---

## Phase 3 — Dataset & Research Pipeline
- **Mục tiêu**: Xây dựng pipeline dữ liệu và huấn luyện PyTorch có thể tái tạo cho nghiên cứu.
- **Starting commit**: `a882eee`
- **Ending commit**: `6555b02`
- **Thay đổi chính**: Tạo `ml/datasets/manifest.py`, adapters cho các dataset nghiên cứu, thuật toán pHash dedup, split chống rò rỉ theo `source_id`, pipeline tăng cường suy giảm chất lượng, kiến trúc `MobileNetV3Forensics`, class-weighted Focal loss, và Temperature Scaling.
- **Kiểm tra kỹ thuật**: Unit tests Python cho adapter, split và forward pass thành công.
- **Kết quả khoa học**: Pipeline kỹ thuật hoàn thiện; chưa huấn luyện checkpoint thật.
- **Evidence**: `ml/tests/test_model_pipeline.py`.
- **Giới hạn**: Chạy trên synthetic tensor fixtures, chưa có dữ liệu thật.
- **Quyết định tiếp theo**: Thiết lập trí nhớ dự án và kiểm định tính trung thực tại Phase 3.5.

---

## Phase 2 — Local Browser Forensic Pipeline
- **Mục tiêu**: Xây dựng pipeline giám định hình ảnh chạy trực tiếp trên trình duyệt bằng TypeScript.
- **Starting commit**: `f2a7fb4`
- **Ending commit**: `a882eee`
- **Thay đổi chính**: Cài đặt defensive file validation (magic bytes, bomb guard), trích xuất EXIF/XMP/IPTC metadata, C2PA adapter, phân tích 2D FFT, 2D DCT, Laplacian noise residual, JPEG grid ELA, và điều phối Web Worker bất đồng bộ.
- **Kiểm tra kỹ thuật**: Unit tests cho validation, metadata và DSP vượt qua.
- **Kết quả khoa học**: Pipeline DSP hoạt động như các heuristics khám phá.
- **Evidence**: `packages/forensics/src/__tests__/dsp.test.ts`, `packages/provenance/src/__tests__/exif-parser.test.ts`.
- **Giới hạn**: Tín hiệu DSP chỉ mang tính chỉ báo sơ bộ, chưa có suy luận học sâu.
- **Quyết định tiếp theo**: Xây dựng ML pipeline phía Python tại Phase 3.

---

## Phase 1 — Production Monorepo Foundation
- **Mục tiêu**: Khởi tạo cấu trúc monorepo phân tán chuẩn sản xuất và môi trường phát triển.
- **Starting commit**: `625e584`
- **Ending commit**: `f2a7fb4`
- **Thay đổi chính**: Thiết lập pnpm workspace (`@forensics/*`, `apps/web`, `ml/`), cấu hình TypeScript strict, Vite React web app, ESLint, Prettier, CI GitHub Actions và Docker container.
- **Kiểm tra kỹ thuật**: Lệnh test, lint và build toàn monorepo thành công.
- **Kết quả khoa học**: Nền tảng hạ tầng hoàn thiện, chưa có logic pháp chứng.
- **Evidence**: `.github/workflows/ci.yml`, `package.json`.
- **Giới hạn**: Khung sườn ban đầu, chưa có thuật toán phân tích.
- **Quyết định tiếp theo**: Phát triển browser forensic pipeline tại Phase 2.

---

## Phase 0 — Audit & Architecture Specification
- **Mục tiêu**: Khảo sát yêu cầu, đặc tả kiến trúc, kế hoạch nghiên cứu, threat model và ADRs ban đầu.
- **Starting commit**: `460f6d5`
- **Ending commit**: `625e584`
- **Thay đổi chính**: Soạn thảo `docs/ARCHITECTURE.md`, `docs/RESEARCH_PLAN.md`, `docs/DATASETS.md`, `docs/EVALUATION.md`, `docs/PRIVACY.md`, `docs/SECURITY.md`, `docs/LIMITATIONS.md`, `docs/LICENSING.md`, ADR 0001–0005, Model Registry schema.
- **Kiểm tra kỹ thuật**: Kiểm tra tính nhất quán tài liệu và commit hợp lệ.
- **Kết quả khoa học**: Định hình đề tài khoa học và không gian bài toán 3 lớp.
- **Evidence**: Các tài liệu nền tảng trong thư mục `docs/`.
- **Giới hạn**: Hoàn toàn là tài liệu đặc tả, chưa có mã nguồn triển khai.
- **Quyết định tiếp theo**: Khởi tạo monorepo tại Phase 1.
