# Sổ Quản lý Tiến độ Dự án: Forensics Web Lab (Status Ledger)

> **Thư mục**: `docs/continuity/STATUS_LEDGER.md`  
> **Mục đích**: Biên niên sử cô đọng từng giai đoạn phát triển và nghiên cứu từ Phase 0 đến nay.  
> **Quy ước**: Giai đoạn mới nhất nằm ở trên cùng; mỗi giai đoạn không quá 20 dòng; chi tiết kiểm chứng nằm tại `research/evidence/`.

---

## Phase 4A.2 — Research Scope Freeze, Continuity Protocol & TGIF Audit
- **Mục tiêu**: Hợp nhất tài liệu continuity sang bộ 3 file thống nhất, đóng băng phạm vi khoa học (3 lớp + uncertain), khảo sát TGIF/TGIF2 và đề xuất phương án pilot.
- **Starting commit**: `863a491`
- **Ending commit**: See repository HEAD
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
