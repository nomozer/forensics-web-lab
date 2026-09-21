# Sổ Quản lý Minh chứng Nghiên cứu và Kỹ thuật (Evidence Register)

> **Dự án**: `forensics-web-lab`  
> **Phiên bản Schema**: `docs/schemas/evidence-manifest.v1.schema.json`  
> **Manifest máy đọc Phase 3.6**: `research/evidence/phase-3.6/evidence-manifest.json`  
> **Manifest máy đọc Phase 4A.0**: `research/evidence/phase-4a.0/evidence-manifest.json`  
> **Manifest máy đọc Phase 4A.1**: `research/evidence/phase-4a.1/evidence-manifest.json`  
> **Manifest máy đọc Phase 4A.2**: `research/evidence/phase-4a.2/evidence-manifest.json`  
> **Manifest máy đọc Phase 4A.3**: `research/evidence/phase-4a.3/evidence-manifest.json`  
> **Cập nhật lần cuối**: Phase 4A.3 (Scientific Pilot Protocol and Label-Semantics Gate)

---

## 1. Nguyên tắc Quản lý Minh chứng & Phân loại Trạng thái

Hệ sinh thái nghiên cứu và sản phẩm của **Forensics Web Lab** tuân thủ nguyên tắc trung thực khoa học tuyệt đối (*Scientific Honesty*). Mọi tuyên bố kỹ thuật, hiệu năng, kích thước hoặc độ chính xác đều phải được phân loại theo tập 8 trạng thái minh chứng chuẩn:

| Trạng thái | Định nghĩa chuẩn | Điều kiện công nhận |
| :--- | :--- | :--- |
| `verified` | Đã có minh chứng trực tiếp, đo lường thật và có thể tái tạo 100%. | Phải có artifact cụ thể, lệnh chạy tái tạo, kết quả đo thật trong repository. |
| `reported` | Được báo cáo trong y văn/tài liệu tham khảo trước đây nhưng chưa kiểm chứng độc lập. | Có trích dẫn tài liệu gốc, ghi rõ chưa kiểm chứng trên codebase nội bộ. |
| `estimated` | Giá trị ước lượng lý thuyết (tính toán toán học), chưa đo trên artifact thật. | Công thức tính toán rõ ràng, chỉ rõ sai số biên khả dĩ. |
| `architecture-only` | Mới tồn tại ở tầng thiết kế kiến trúc hoặc mã nguồn PyTorch/TS, chưa huấn luyện. | Mã nguồn khởi tạo module có thể chạy, không gọi là checkpoint. |
| `pipeline-only` | Chỉ chứng minh pipeline kỹ thuật/dòng chảy dữ liệu hoạt động với dữ liệu giả định. | Unit test pipeline đạt; không suy diễn tính đúng đắn trên trọng số thật. |
| `unverified` | Chưa có minh chứng hoặc chưa thực hiện đo lường. | Ghi nhận trung thực `unverified` hoặc `not measured`. |
| `blocked` | Bị chặn do thiếu dữ liệu, trọng số, giấy phép bản quyền hoặc tài nguyên phần cứng. | Nêu rõ phụ thuộc đang chặn và điều kiện gỡ chặn. |
| `rejected` | Đã đánh giá toàn diện và kết luận không phù hợp với tiêu chuẩn dự án. | Nêu lý do loại trừ (kích thước quá lớn, giấy phép hạn chế, v.v.). |

---

## 2. Bảng Đăng ký Minh chứng Phase 4A.3 (Scientific Pilot Protocol & Label-Semantics Gate)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-LABEL-GATE-001** | Cổng kiểm định nhãn khoa học đóng băng không gian 3 lớp chuẩn tắc (`authentic`, `fully_generated`, `ai_edited`); từ chối mọi nhãn không hợp lệ. | `science` | `verified` | [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md) | `ml/.venv/Scripts/python -m pytest ml/tests/test_pilot_protocol.py -k test_label_taxonomy` | Định nghĩa 3 nhãn chuẩn tắc được kiểm thử tự động; validator từ chối nhãn nằm ngoài tập chuẩn. | Áp dụng cho phân loại ảnh tĩnh; không bao gồm video deepfake. |
| **EV-TGIF-SEMANTICS-001** | Kiểm toán ngữ nghĩa TGIF: thành phần `sp` (spliced) được xác minh là `ai_edited` kèm mask; thành phần `fr` (fully regenerated) canvas có điều kiện từ ảnh gốc, bị cấm gán `fully_generated` và được cách ly. | `dataset` | `verified` | [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md) | `ml/.venv/Scripts/python -m pytest ml/tests/test_pilot_protocol.py -k test_tgif_fr` | Đã chứng minh `fr` không đủ căn cứ gán `fully_generated`; validator tự động bắt lỗi vi phạm ngữ nghĩa nếu cố tình gán. | Dựa trên tài liệu chính thức TGIF (arXiv:2407.11566) và kiểm toán Nextcloud. |
| **EV-PILOT-DESIGN-001** | Thiết kế kiến trúc pilot hai nhánh độc lập (Pilot A trên TGIF matched pairs, Pilot B trên GenImage ImageNet) nhằm triệt tiêu nguy cơ shortcut nguồn dữ liệu; Pilot C ba lớp chỉ mở khi vượt qua audit shortcut. | `science` | `verified` | [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md) | `ml/.venv/Scripts/python -m ml.configs.validator --validate-all` | Pilot A và B cấu hình độc lập; Pilot C yêu cầu kèm metadata-only baseline guard. | Thí nghiệm đang ở giai đoạn giao thức trước huấn luyện; chưa tải dữ liệu thật. |
| **EV-SHORTCUT-PROTOCOL-001** | Giao thức chống rò rỉ và kiểm soát shortcut codified: deduplication (SHA-256 + pHash), group split theo `source_id`, class-level distribution audit, và metadata-only baseline guard. | `science` | `verified` | [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md) | `ml/.venv/Scripts/python -m pytest ml/tests/test_pilot_protocol.py -k test_manifest_group_isolation` | Zero leakage giữa các split được kiểm chứng bằng unit test; nhóm ảnh gốc và bản inpainting không bao giờ cắt chéo split. | Deduplication đầy đủ trên toàn bộ dataset sẽ chạy sau khi tải dữ liệu thật. |
| **EV-PILOT-CONFIGS-001** | Cấu hình pilot YAML máy đọc (`pilot_tgif_edit.yaml`, `pilot_genimage_generated.yaml`) và module validator tự động (`ml/configs/validator.py`) được triển khai và kiểm thử 100%. | `pipeline` | `verified` | [validator.py](ml/configs/validator.py) | `ml/.venv/Scripts/python -m ml.configs.validator --validate-all` | 13 trường bắt buộc, quy tắc nhãn và ràng buộc anti-leakage được tự động kiểm định đạt 100%. | Sẵn sàng nạp DataLoader khi manifest được sinh từ dữ liệu thật. |
| **EV-ACQUISITION-DRYRUN-001** | CLI `acquire.py` hỗ trợ dry-run chuyên biệt cho Pilot A và Pilot B, in đầy đủ 12 trường thông tin kiểm toán với 0 request mạng và 0 byte tải về. | `pipeline` | `verified` | [acquire.py](ml/datasets/acquire.py) | `ml/.venv/Scripts/python -m ml.datasets.acquire --pilot pilot-a` | Lệnh dry-run in chi tiết từng component, dung lượng xác minh, ổ đĩa yêu cầu và giữ mạng ở mức 0 bytes. | Thao tác tải thật bị khóa fail-closed chờ người dùng phê duyệt. |

---

## 3. Bảng Đăng ký Minh chứng Phase 4A.2 (Scope Freeze, Continuity & TGIF Feasibility Audit)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-CONTINUITY-001** | Dự án thiết lập giao thức continuity 3 file thống nhất trong `docs/continuity/` (`CODE_INDEX.md`, `CURRENT_STATE.md`, `STATUS_LEDGER.md`), giải quyết dứt điểm phân mảnh tài liệu. | `git-state` | `verified` | [CURRENT_STATE.md](docs/continuity/CURRENT_STATE.md) | `git status docs/continuity/` | 3 tài liệu continuity được tạo với cấu trúc cô đọng; cập nhật thứ tự đọc bắt buộc trong `AGENTS.md`. | Cung cấp ngữ cảnh xuyên suốt cho các phiên AI; log chi tiết tiếp tục nằm tại `research/evidence/`. |
| **EV-RESEARCH-SCOPE-001** | Phạm vi nghiên cứu được đóng băng vào 3 lớp huấn luyện (`authentic`, `fully_generated`, `ai_edited`); xác định `uncertain` là trạng thái quyết định sau calibration và localization là mục tiêu phụ. | `science` | `verified` | [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | `git log -1 docs/RESEARCH_PLAN.md` | Đóng băng không gian nhãn và vai trò của từng tác vụ; bản đồ nhiệt patch score được gắn nhãn là heuristic khám phá. | Năng lực định vị chỉ được công nhận sau khi đo mIoU/Dice so với ground-truth mask thật. |
| **EV-RQ-METRIC-MAP-001** | Khóa 4 câu hỏi nghiên cứu (RQ1–RQ4) và hệ thống chỉ số toàn diện (Primary Macro-F1/Balanced Accuracy, Calibration ECE/Brier, Robustness, Browser runtime). | `science` | `verified` | [EVALUATION.md](docs/EVALUATION.md) | `git log -1 docs/EVALUATION.md` | Hệ thống chỉ số và giả thuyết được chính thức hóa; mọi chỉ số tiếp tục mang trạng thái `not evaluated`. | Đòi hỏi huấn luyện và đánh giá trên dữ liệu thật ở các phase sau. |
| **EV-TGIF-METADATA-001** | Metadata kho lưu trữ chính thức TGIF và TGIF2 trên Nextcloud được khảo sát trực tiếp: TGIF (65.4 GB across 6 folders), TGIF2 FLUX (110 GB), TGIF2 random (73 GB); hỗ trợ tải lẻ từng thư mục con. | `dataset` | `verified` | [tgif-remote-inventory.json](research/evidence/phase-4a.2/tgif-remote-inventory.json) | `git log -1 research/evidence/phase-4a.2/tgif-remote-inventory.json` | Xác định dung lượng từng thư mục con (ví dụ: masks 40.4 MB, orig 6.8 GB); kiểm chứng khả năng tải độc lập qua Nextcloud share. | Khảo sát bằng truy vấn metadata/HTML; 0 byte content được tải về máy. |
| **EV-TGIF-LICENSE-001** | TGIF phân phối theo `CC BY-SA 4.0`, ảnh gốc MS-COCO `CC BY 4.0`; trạng thái trọng số mô hình là `unclear`, chính sách dự án là `prohibited-by-project-policy` đối với sản phẩm web. | `license` | `verified` | [registry.json](datasets/registry.json) | `python -m ml.datasets.acquire --validate-registry` | Đăng ký `tgif` và `tgif2` vào registry ở luồng `research-only`, xác thực giấy phép chính thức. | Điều khoản Share-Alike tiềm ẩn ràng buộc lên weights; cách ly tuyệt đối trong Research Track. |
| **EV-DATASET-SHORTCUT-RISK-001** | Nhận diện nguy cơ mô hình học shortcut đặc trưng của nguồn dữ liệu; thiết lập ma trận nghiên cứu–dữ liệu và các biện pháp kiểm soát bắt buộc (group split, matched pairs). | `dataset` | `verified` | [DATASETS.md](docs/DATASETS.md) | `git log -1 docs/DATASETS.md` | Ma trận 5 cột kiểm soát rủi ro shortcut được lập; quy định group split theo `source_id` và dùng matched pairs. | Cần kiểm chứng thực nghiệm bằng cross-dataset evaluation ở Phase 4. |
| **EV-PILOT-PROPOSAL-001** | Xây dựng 3 phương án thử nghiệm (Pilot A Smoke, Pilot B Exploratory 3-Class, Pilot C Benchmark); kết luận khuyến nghị lựa chọn Pilot B (~3,000 mẫu) cho khóa luận. | `dataset` | `verified` | [DATASETS.md](docs/DATASETS.md) | `git log -1 docs/DATASETS.md` | Pilot B xác định rõ dung lượng nén (~15 GB), dung lượng giải nén (~18 GB), yêu cầu ổ đĩa ($\ge 35\text{ GB}$) và cấu hình compute. | Phương án đề xuất chờ phê duyệt; 0 byte external data được tải trong phase này. |

---

## 3. Bảng Đăng ký Minh chứng Phase 4A.1 (Acquisition Feasibility & Residual Claim Correction)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-LICENSE-WEIGHTS-INTERPRETATION-001** | GenImage cấm sử dụng thương mại dataset và derivative works; tình trạng pháp lý của checkpoint là `unclear`, chính sách dự án là `prohibited-by-project-policy` đối với sản phẩm. | `license` | `verified` | [DATA_LICENSES.md](docs/DATA_LICENSES.md) | `git log -1 docs/DATA_LICENSES.md` | Tách biệt nguồn điều khoản (`prohibited_for_commercial_use`), giải thích pháp lý (`unclear`) và chính sách dự án (`production_use: prohibited`). | Chính sách bảo thủ của dự án; chưa phải ý kiến tư vấn pháp lý chính thức. |
| **EV-FIXTURE-TRACK-001** | `synthetic-smoke` được định danh lại vào luồng `fixture-only` (`purpose: fixture`), chỉ dùng kiểm thử kỹ thuật pipeline và tuyệt đối không tham gia huấn luyện hay đánh giá sản phẩm. | `dataset` | `verified` | [registry.json](datasets/registry.json) | `pnpm --filter @forensics/shared test` | Validator và schema từ chối `fixture-only` trong model lineage sản phẩm; `commercialUse: internal-testing-only`. | Chỉ là dữ liệu hình học giả lập sinh bằng script nội bộ. |
| **EV-GENIMAGE-REMOTE-METADATA-001** | Metadata thư mục Google Drive chính thức của GenImage (`1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS`) được khảo sát trực tiếp: chứa 8 thư mục generator lưu dưới dạng file nén phân mảnh multi-part zip (`.z01` .. `.zip`). | `dataset` | `verified` | [genimage-remote-inventory.json](research/evidence/phase-4a.1/genimage-remote-inventory.json) | `git log -1 research/evidence/phase-4a.1/genimage-remote-inventory.json` | 16 mục được ghi nhận; thư mục BigGAN (`1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt`) gồm 8 file split volumes `imagenet_ai_0419_biggan.z01` .. `.zip`. | Khảo sát bằng browser inspection; không tải file zip nội dung. |
| **EV-GENIMAGE-SUBSET-FEASIBILITY-001** | Nguồn chính thức GenImage không hỗ trợ tải trực tiếp từng ảnh hoặc tập con 50-100 ảnh; bắt buộc phải tải archive nén phân mảnh hoàn chỉnh rồi mới lấy mẫu (Kết luận B). | `dataset` | `verified` | [GENIMAGE_ACQUISITION_PROPOSAL.md](docs/GENIMAGE_ACQUISITION_PROPOSAL.md) | `git log -1 docs/GENIMAGE_ACQUISITION_PROPOSAL.md` | Archive nhỏ nhất phù hợp là BigGAN (~24 GB compressed, estimated ~26 GB uncompressed, yêu cầu ổ đĩa >= 55 GB); loại bỏ hoàn toàn con số <50 MB cho nguồn chính thức. | Việc lấy mẫu 10-50 ảnh chỉ thực hiện được sau khi đã tải và giải nén toàn bộ archive. |
| **EV-EXTERNAL-DOWNLOAD-LOCK-001** | Thao tác tải dataset ngoài được khóa chặt chẽ (`acquisitionEnabled: false`, `approvalStatus: pending-user-approval`), yêu cầu sự phê duyệt tường minh kèm tên archive và dung lượng byte chính xác từ người dùng. | `dataset` | `verified` | [acquire.py](ml/datasets/acquire.py) | `ml/.venv/Scripts/python -m ml.datasets.acquire --dataset genimage --track research --execute` | Lệnh execute trả về thông báo khóa và yêu cầu phê duyệt; riêng `synthetic-smoke --execute` sinh fixture cục bộ (0 network, 0 external bytes). | Ngăn chặn việc vô tình kích hoạt tải dữ liệu lớn khi chưa cấp phép. |
| **EV-LOCAL-LINK-SCAN-001** | Toàn bộ tài liệu, manifest và code trong repository đạt 0 liên kết máy cá nhân (`file:///`, `C:\`, `D:\`, `/Users/`, `/home/`). | `test-ts` | `verified` | [dataset-registry.test.ts](packages/shared/src/__tests__/dataset-registry.test.ts) | `pnpm --filter @forensics/shared test -t "Repository Clean Link"` | 0 liên kết máy cá nhân trong toàn bộ cây thư mục; phân biệt rõ ví dụ minh họa và liên kết thực. | Duy trì kiểm tra tự động trong CI/test suite. |

---

## 3. Bảng Đăng ký Minh chứng Phase 4A.0 (Data Governance & Track Isolation)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-LICENSE-GENIMAGE-001** | Giấy phép chính thức của GenImage là `CC BY-NC-SA 4.0 with additional dataset terms`; cấm sử dụng thương mại đối với cả dataset lẫn trọng số phái sinh. | `license` | `verified` | [DATA_LICENSES.md](docs/DATA_LICENSES.md) | `git log -1 docs/DATA_LICENSES.md` | Đã thẩm định văn bản từ GitHub tác giả; xác định thuộc luồng `research-only`. | Chỉ áp dụng trong luồng nghiên cứu; cấm đưa vào sản phẩm thương mại. |
| **EV-LICENSE-REALHD-001** | RealHD hiển thị trạng thái "Coming soon" trên kho mã nguồn chính thức và chưa có văn bản giấy phép công khai; việc tiếp nhận dữ liệu bị khóa. | `license` | `blocked` | [DATA_LICENSES.md](docs/DATA_LICENSES.md) | `git log -1 docs/DATA_LICENSES.md` | Đặt trạng thái `unavailable-or-pending`, giấy phép `unverified`, tình trạng `blocked`. | Khóa cho đến khi tác giả phát hành bản chính thức kèm giấy phép rõ ràng. |
| **EV-DATA-REGISTRY-001** | Sổ đăng ký dataset máy đọc tồn tại tại `datasets/registry.json` và tuân thủ chặt chẽ JSON Schema v1.0.0. | `registry` | `verified` | [registry.json](datasets/registry.json) | `python -m ml.datasets.acquire --validate-registry` | 5 dataset được đăng ký và kiểm định hợp lệ (genimage, realhd, sagi-d, raid, synthetic-smoke). | Kiểm tra cấu trúc dữ liệu và ranh giới bản quyền; không tải dữ liệu mạng. |
| **EV-DATA-ISOLATION-001** | ADR-0006 thiết lập kiến trúc cách ly tuyệt đối giữa dữ liệu nghiên cứu (`data/research/`) và dữ liệu sản phẩm (`data/product/`). | `dataset` | `verified` | [0006-research-product-data-isolation.md](docs/adr/0006-research-product-data-isolation.md) | `git log -1 docs/adr/0006-research-product-data-isolation.md` | Quy chuẩn phân luồng rõ ràng; cấm nhiễm chéo dữ liệu và checkpoint giữa hai luồng. | Quy định kiến trúc và cấu trúc thư mục; thực thi qua validator và test. |
| **EV-ACQUIRE-DRYRUN-001** | CLI `ml.datasets.acquire` thực thi ở chế độ `--dry-run` với 0 request mạng, 0 byte tải về và 0 file ảnh được tạo. | `dataset` | `verified` | [acquire.py](ml/datasets/acquire.py) | `python -m ml.datasets.acquire --dataset genimage --track research --dry-run` | Báo cáo dry-run hiển thị đầy đủ thông tin pháp lý; cấm kích hoạt lệnh tải thật. | Chỉ mô phỏng kiểm tra trước khi tải; tải thật đòi hỏi sự phê duyệt của người dùng. |
| **EV-CONTAMINATION-001** | Bộ kiểm soát ô nhiễm tự động từ chối nạp dataset `research-only` vào luồng sản phẩm và từ chối các dataset bị khóa (`blocked`). | `dataset` | `verified` | [test_contamination_guard.py](ml/tests/test_contamination_guard.py) | `ml/.venv/Scripts/python -m pytest ml/tests/test_contamination_guard.py -v` | 6/6 test kiểm soát ô nhiễm vượt qua; từ chối vi phạm ranh giới với mã thoát 1. | Hoạt động ở tầng CLI và validator; mô hình thực tế cần kiểm định trước khi nạp web. |
| **EV-SMOKE-FIXTURE-001** | Bộ dữ liệu giả lập hình học/nhiễu toán học được sinh hoàn toàn bằng code nội bộ mà không cần tải ảnh ngoài hay chịu ràng buộc bản quyền. | `dataset` | `verified` | [smoke_generator.py](ml/tests/fixtures/smoke_generator.py) | `ml/.venv/Scripts/python -m pytest ml/tests/test_contamination_guard.py -k test_synthetic_smoke_fixture_generation_and_provenance -v` | Sinh 8 mẫu hình học có nhãn authentic, fully_generated, ai_edited kèm mask và manifest. | Hoàn toàn là hình vẽ nhân tạo; không phải ảnh thật; không dùng đo độ chính xác. |
| **EV-ZEROEGRESS-001** | Tuyên bố Zero Server Egress được hỗ trợ bởi thiết kế kiến trúc client-only; việc kiểm chứng mạng khi chạy thực tế chưa được đo. | `no-model` | `architecture-only` | [PRIVACY.md](docs/PRIVACY.md) | `git log -1 docs/PRIVACY.md` | Kiến trúc loại bỏ backend API phân tích; kiểm chứng network interception trên trình duyệt ghi nhận `unverified`. | Đang có backlog kiểm thử E2E tự động chặn và ghi nhận network frames. |
| **EV-WASM-BUDGET-001** | Bản dựng Vite tạo file engine ONNX Runtime WebAssembly đo được 28.3 MB; ghi nhận là tài nguyên engine runtime, không phải kích thước model. | `build` | `verified` | [build-summary.json](research/evidence/phase-4a.0/build-summary.json) | `pnpm build` | File `ort-wasm-simd-threaded.jsep.wasm` đo được 28,312.03 kB; xác định là engine suy luận. | Đo lường trên file bundle chưa nén; nén gzip truyền tải và lazy loading có trong backlog. |
| **EV-SCIENCE-002** | Toàn bộ chỉ số độ chính xác phát hiện AI, khả năng tổng quát hóa trên generator chưa thấy và định vị inpainting vẫn chưa đo lường. | `science` | `unverified` | [EVALUATION.md](docs/EVALUATION.md) | `git log -1 docs/EVALUATION.md` | Mọi chỉ số Macro F1, ECE, mIoU tiếp tục ghi nhận `not measured` / `not evaluated`. | Đòi hỏi huấn luyện mô hình trên dữ liệu chuẩn trong các phase tiếp theo. |

---

## 3. Bảng Đăng ký Minh chứng Phase 3.6 (Lịch sử)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-GIT-001** | Trạng thái Git nằm trên nhánh `feat/production-ai-image-forensics`, `main` không bị can thiệp tại `460f6d5`, remote nguyên vẹn. | `git-state` | `verified` | [environment.json](research/evidence/phase-3.6/environment.json) | `git status --short && git branch --show-current && git rev-parse HEAD && git rev-parse main && git remote -v` | Nhánh `feat/production-ai-image-forensics`, HEAD `cd59136`, main `460f6d5`, working tree sạch. | Chỉ kiểm toán workspace cục bộ. |
| **EV-TEST-TS-001** | Bộ kiểm thử TypeScript đạt 100% trên toàn bộ 6 package monorepo. | `test-ts` | `verified` | [test-summary.json](research/evidence/phase-3.6/test-summary.json) | `pnpm test` | 40/40 tests vượt qua trên toàn bộ monorepo. | Chạy trên Node.js/jsdom. |
| **EV-TEST-PY-001** | Bộ kiểm thử Python ML đạt 100% trên kiến trúc mô hình, hàm mất mát, manifest và calibration. | `test-py` | `verified` | [test-summary.json](research/evidence/phase-3.6/test-summary.json) | `ml/.venv/Scripts/python -m pytest ml/tests -v` | 5/5 tests vượt qua trong 5.73s. | Chạy trên fixture giả lập. |
| **EV-BUILD-001** | Bản dựng production web bundle thành công với TypeScript strict mode và Vite. | `build` | `verified` | [build-summary.json](research/evidence/phase-3.6/build-summary.json) | `pnpm build` | Thoát mã 0 trong 2.99s. | Xác minh đóng gói tài nguyên tĩnh. |
| **EV-NOMODEL-001** | Hệ thống trả về kết luận `uncertain`, `confidence: null` khi chưa cài đặt mô hình; cấm sinh kết luận AI giả. | `no-model` | `verified` | [contracts.test.ts](packages/shared/src/__tests__/contracts.test.ts) | `pnpm test` | Contract tests xác minh trạng thái no-model trung thực. | Áp dụng ở tầng contract, worker, UI và report. |
| **EV-REPORT-001** | Báo cáo xuất bản công bố rõ ràng trạng thái "Model not installed" và không tạo điểm tin cậy AI giả. | `no-model` | `verified` | [report.test.ts](packages/report/src/__tests__/report.test.ts) | `pnpm test` | Module báo cáo sinh cấu trúc hợp lệ theo schema JSON v1.0. | Kiểm thử tính logic hiển thị. |
| **EV-REGISTRY-001** | Bộ kiểm định Registry từ chối mô hình chưa xác minh và cấm chuyển `ready` nếu thiếu 7 tiêu chí. | `registry` | `verified` | [registry-validator.ts](packages/shared/src/registry-validator.ts) | `pnpm test` | Validator ép buộc đủ 7 điều kiện cho trạng thái `ready`. | Xác minh cấu trúc schema. |
| **EV-ONNX-001** | Pipeline xuất ONNX và kiểm thử sai số số học (parity) hoạt động bình thường trên kiến trúc mạng nơ-ron. | `onnx-pipeline` | `pipeline-only` | [export_onnx.py](ml/export/export_onnx.py) | `ml/.venv/Scripts/python -m pytest ml/tests/test_model_pipeline.py -v` | Mã pipeline tồn tại và chạy trơn tru trên tensor kích thước chuẩn. | Chưa có checkpoint huấn luyện thật. |
| **EV-RUNTIME-001** | Tài nguyên ONNX Runtime Web WASM đã được đóng gói và kiểm chứng thực thi trên Chrome, Edge, Firefox, Safari. | `browser-runtime` | `unverified` | [ort-session.ts](packages/inference/src/ort-session.ts) | `pnpm build` | File WASM nằm trong bundle; việc thực thi suy luận trực tiếp chưa được kiểm chứng vật lý. | Bị chặn do chưa có file mô hình ONNX. |
| **EV-SIZE-001** | Kích thước mô hình MobileNetV3 nội bộ sau lượng tử hóa INT8 đạt khoảng 2.6 MB. | `model-size` | `estimated` | [PRETRAINED_MODEL_CANDIDATES.md](docs/PRETRAINED_MODEL_CANDIDATES.md) | `ml/.venv/Scripts/python -c "from ml.training.mobilenetv3_forensics import MobileNetV3Forensics; m = MobileNetV3Forensics(); p = sum(p.numel() for p in m.parameters()); print(f'Params: {p}')"` | Ước lượng lý thuyết 1 byte/param cho ra ~2.6 MB; chưa có file ONNX INT8 thực tế. | Hoàn toàn là ước lượng toán học. |
| **EV-LICENSE-001** | Repository và trọng số mô hình tương lai có giấy phép mở được xác minh. | `license` | `unverified` | [LICENSING.md](docs/LICENSING.md) | `git log -1 docs/LICENSING.md` | Giấy phép repository đang chờ quyết định; giấy phép trọng số mô hình hiện tại là `not-applicable`. | Quyền phân phối phụ thuộc mã nguồn, backbone và dataset. |
| **EV-DATA-001** | Các bộ dữ liệu huấn luyện học thuật đã được tải xuống và thẩm định trong repository. | `dataset` | `blocked` | [DATASETS.md](docs/DATASETS.md) | `git status --short data/` | 0 byte dữ liệu được tải về. | Bị khóa chờ người dùng cấp phép. |
| **EV-SCIENCE-001** | Hệ thống sở hữu độ chính xác phát hiện hoặc định vị inpainting đã đo lường. | `science` | `unverified` | [EVALUATION.md](docs/EVALUATION.md) | `git log -1 docs/EVALUATION.md` | Các chỉ số đều được ghi nhận trung thực là `not measured` / `not evaluated`. | Không có tuyên bố khoa học nào được đưa ra. |

---

## 4. Quy trình Tái tạo và Kiểm chứng bằng Máy

Mọi minh chứng trong bảng trên đều có thể được xác minh tự động thông qua công cụ kiểm thử hợp đồng trong `@forensics/shared` và bộ test Python:

```bash
# Kiểm tra toàn bộ hợp đồng TypeScript và manifest trên đĩa
pnpm test

# Kiểm tra bộ kiểm soát ô nhiễm và fixture trong Python
ml/.venv/Scripts/python -m pytest ml/tests -v

# Kiểm tra sổ đăng ký dataset máy đọc
ml/.venv/Scripts/python -m ml.datasets.acquire --validate-registry
```
