# Phase 4C.7B — Independent Cohort Acquisition: Protocol Amendment, Automated Pipeline, and Feasibility Alignment

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment (Pilot v2 Feasibility Closure and Existing Dataset Continuation Proposal)<br>
> **Status**: `TGIF_TRAIN_COHORT_INTAKE_APPROVED_PRE_EXECUTION` — Người dùng chính thức phê duyệt Phương án $N=400$ cặp (14 Large [3,5%], 221 Medium [55,25%], 165 Small [41,25%]) và chấp thuận phạm vi INTAKE ONLY bằng Colab CPU (tải 2 archive 18,63 GiB từ Nextcloud chính thức của IMEC, trích xuất chọn lọc 400 cặp, đóng gói ZIP ~120-160 MB trả về local; chưa cho phép detector/evaluation hoặc model training). Ghi nhận các nguyên tắc khoa học: (1) Không coi phân bổ này là tỷ lệ tự nhiên; nhóm Large chỉ có 14 nguồn nên không đưa ra kết luận mạnh riêng cho nhóm đó. (2) Đính chính thống kê: căn cứ "N=400 đảm bảo ME<0,01" từ Phase 4C.7A thuộc về thiết kế cân bằng cũ, CHƯA XÁC LẬP cho thiết kế mới; bắt buộc kiểm tra lại căn cứ thống kê trước evaluation. (3) Bổ sung SHA-256 từng mask được chọn (100% khớp hash trên đĩa cục bộ) và niêm phong danh mục 400 dòng đã khóa (`tgif_train_clean_subset_manifest_locked_n400.json`, SHA-256 `53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c`; CSV SHA-256 `dc9584b31899e5484a4eb40dd6ec653909e39f08de48ada17ba138f8dabd03d4`; pool manifest JSON `9e4ef2c9...`, CSV `ba122a4f...`). (4) Khóa quy tắc: bước kiểm tra bộ ba thực hiện chính thức tại LOCAL sau intake (sử dụng 100% masks có sẵn); khóa preprocessing (Lanczos 512x512 RGB, Nearest 512x512 mask binary, cấm letterbox) và Technical QC; quy tắc xử lý mẫu thiếu/hỏng: FAIL-CLOSED, cấm tự chọn mẫu thay thế (zero automatic replacement); an toàn trích xuất và bảo toàn benchmark (cấm sửa/recomposite ảnh). (5) Hoàn thiện launcher Colab CPU (`notebooks/tgif_train_cohort_acquisition_colab.ipynb`), worker (`acquire_tgif_train_subset_colab.py`), runner audit local (`audit_tgif_train_subset_local.py`), và bộ test (`ml/tests/test_tgif_train_intake.py`, 8/8 PASS). 26 quyết định Human Content QC lịch sử tiếp tục PENDING. Full cohort locked, detector calls = 0, independent performance `NOT_MEASURED`.<br>
> **Findings status**: `NOT_MEASURED` (0 detector calls, 0 cohort evaluation)<br>
> **Current TGIF intake execution commit**: `bf0f20a08a71246f9c441569aae3ae601486ab7f` (fail-closed N=400 worker/auditor execution snapshot; canonical intake notebook pins this full SHA by detached checkout). Historical pilot/diagnostic/calibration binding `79775250ee5393c9f193bdc0485f77998cefaddf` remains unchanged.<br>
> **Real pilot / diagnostic / calibration**: `PILOT_V2_FEASIBILITY_CONCLUDED_8_PAIRS_PENDING_HUMAN_CONTENT_QC` (8 pairs in `pilot-20261009T111247Z` evaluated, Agent recommends 1 ACCEPT / 7 REJECT, Human Content QC PENDING; 2 calibration attempts in `calib-20261009T015749Z` Agent recommends REJECT, Human Content QC PENDING; 6 diagnostic attempts in `diag-20261008T154628Z` 6/6 REJECT; 8 historical pilot pairs in `pilot-20261008T113700Z` PENDING; full 400-pair run remains `NOT_RUN`)<br>
> **Training runs**: 0 fits, 0 refits; frozen models untouched; retired locked-test not accessed (only its 343 source IDs are read for the disjoint guard)<br>
> **Evidence classification**: run/commit/artifact observations are `internal-empirical`; model cards/documentation are `external-source`; latent-capacity/context-bias explanations remain `unverified-hypothesis`. Citation keys resolve through `docs/references.bib`.<br>

## MM. Đăng Ký Phê Duyệt Tiếp Nhận TGIF Train N=400, Đính Chính Thống Kê & Bàn Giao Colab CPU Launcher (2026-10-09)

Hồ sơ đăng ký quyết định phê duyệt chính thức của người dùng cho việc tiếp nhận dữ liệu tập kiểm định độc lập `TGIF-Train-Clean-Subset` ($N=400$ cặp) thông qua môi trường Colab CPU:

### 1. Quyết Định Phê Duyệt & Giới Hạn Quyền Hạn (Human Approval & Scope)
- **Người duyệt**: Dũng Phạm `<valdung04@gmail.com>`.
- **Thời điểm duyệt**: `2026-10-09T14:19:44Z` (Giờ địa phương: `2026-10-09T21:19:44+07:00`).
- **Phương án lựa chọn**: Cỡ mẫu **$N = 400$ cặp** (14 Large, 221 Medium, 165 Small), ghép cặp 1:1 nguồn duy nhất.
- **Phạm vi phê duyệt**: **CHỈ DÀNH CHO TIẾP NHẬN DỮ LIỆU (INTAKE ONLY)** trên Colab CPU.
  - Cho phép tải 2 archive từ Nextcloud chính thức của IMEC (`orig_training.tar.gz` 5.26 GiB và `sd2-sp_training.tar.gz` 13.37 GiB, tổng 18.63 GiB) vào ổ đĩa tạm `/content/`.
  - Cho phép trích xuất chọn lọc đúng 400 cặp đã khóa và đóng gói thành `tgif_train_clean_subset_package.zip` (~120–160 MB) để tải về máy local.
  - **TUYỆT ĐỐI CHƯA CHO PHÉP**: Chạy mô hình detector, đánh giá kiểm định (evaluation pipeline) hoặc huấn luyện mô hình (model training). Toàn bộ mô hình tiếp tục ở trạng thái đóng băng và khóa chặt.

### 2. Các Cam Kết Khoa Học & Đính Chính Thống Kê Bắt Buộc
1. **Không coi phân bổ là tỷ lệ tự nhiên**: Nhóm Large chỉ có 14 nguồn khả dụng trong toàn bộ split ($3,5\%$). Đây là đặc thù dữ liệu phân đoạn của MS-COCO/TGIF, không phải tỷ lệ can thiệp tự nhiên. **Tuyệt đối không đưa ra kết luận mạnh riêng cho nhóm Large**.
2. **Đính chính căn cứ sai số ME < 0,01**: Tuyên bố trước đó cho rằng "N=400 đảm bảo $\text{ME} < 0,01$" xuất phát từ mô phỏng Monte Carlo ở Phase 4C.7A trên giả định phân bổ cân bằng $50/50$. Đối với thiết kế phân tầng mới có độ lệch lớn, **căn cứ này CHƯA XÁC LẬP** và bắt buộc phải kiểm tra lại căn cứ thống kê trước khi tiến hành evaluation.
3. **Đính chính cách diễn đạt ngưỡng diện tích `0.02/0.15`**: Hai giá trị `0.02` và `0.15` không phải ranh giới phân tầng của cohort đã khóa. Hợp đồng chuẩn tắc vẫn là `0.01 <= P < 0.10` (Small), `0.10 <= P < 0.30` (Medium), và `0.30 <= P <= 0.50` (Large), tương ứng **1–10%, 10–30%, 30–50%** sau preprocessing mask 512×512. Đính chính này chỉ sửa diễn đạt báo cáo; không tính lại, thay thế hoặc chọn lại bất kỳ mẫu nào trong 400 cặp đã khóa.
4. **Bước kiểm tra bộ ba thực hiện tại LOCAL sau intake**: Colab chỉ trích xuất ảnh authentic và edited. Bước kiểm tra bộ ba (tripartite alignment audit) được thực hiện chính thức trên máy LOCAL sau intake bằng `audit_tgif_train_subset_local.py`, đối chiếu với 100% kho mask đã có sẵn trên đĩa cục bộ (0 MB mạng trên Colab).
5. **Khóa Preprocessing & QC; Cấm tự chọn mẫu thay thế (FAIL-CLOSED)**:
   - Preprocessing đã khóa: center-crop Lanczos về $512 \times 512$ RGB không nén cho ảnh; center-crop Nearest-neighbor về $512 \times 512$ binarize $\{0, 255\}$ cho mask; diện tích mask tính sau biến đổi; cấm letterboxing.
   - QC đã khóa: PIL decode 100%, $\text{std} \ge 2,0$, $\text{inside\_mean\_l1} \ge 1,0$.
   - **Quy tắc xử lý mẫu thiếu/hỏng**: Nếu bất kỳ mẫu nào bị thiếu hoặc lỗi, worker ghi nhận lỗi vào receipt và dừng quy trình. **CẤM TỰ ĐỘNG CHỌN MẪU THAY THẾ (zero automatic replacement)** từ pool nếu chưa có sự phê duyệt của người dùng.
6. **An toàn trích xuất & Bảo toàn tính nguyên bản benchmark**:
   - Sử dụng phương pháp đọc luồng tar (`tarfile` stream) để trích xuất trực tiếp 400 tệp mà không giải nén 18,63 GiB ra đĩa. Kiểm tra nghiêm ngặt chống TarSlip/ZipSlip (chặn `..`, absolute paths, symlinks).
   - **Tuyệt đối không sửa hoặc composite lại ảnh benchmark** để ép outside L1 về 0. Giữ nguyên độ biến thiên pixel thực tế của tác giả TGIF.

### 3. Hồ Sơ Manifest Đã Khóa (Cryptographic Locked Manifests)
- **Manifest tuyển chọn 400 dòng đã khóa (Locked Selection Manifest)**:
  - JSON: [`research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json) (SHA-256: `53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c`)
  - CSV: [`research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv) (SHA-256: `dc9584b31899e5484a4eb40dd6ec653909e39f08de48ada17ba138f8dabd03d4`)
  - Toàn bộ 400 mask trên đĩa đã được băm SHA-256 và kiểm tra toàn vẹn bit 100% khớp với manifest khi tính diện tích.
- **Manifest hồ sơ ứng viên (Candidate Pool Manifest)**:
  - JSON: [`research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json) (SHA-256: `9e4ef2c9f89ad7dc1316d764c0acfff3b55dbc60dcc666d8928ff49a04dcd7b6`)
  - CSV: [`research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv) (SHA-256: `ba122a4f934c18dd49d59f33fda2c790101eae8eaf9d4a354d18ae4d39a1f6f3`)

### 4. Bàn Giao Launcher Colab CPU & Bộ Công Cụ
- **Colab CPU Launcher Notebook**: [`notebooks/tgif_train_cohort_acquisition_colab.ipynb`](notebooks/tgif_train_cohort_acquisition_colab.ipynb) (6 cells tự động: kiểm tra đĩa >= 25 GB, fetch branch để lấy object rồi checkout detached full execution SHA `bf0f20a08a71246f9c441569aae3ae601486ab7f`, xác minh HEAD sạch và mã băm locked manifest trước tải, tải 2 archive từ Nextcloud, chạy worker trích xuất chọn lọc, bàn giao ZIP và dọn archive/partial tạm).
- **Colab Worker Script**: [`scripts/research/acquire_tgif_train_subset_colab.py`](scripts/research/acquire_tgif_train_subset_colab.py).
- **Local Forensic Audit Script**: [`scripts/research/audit_tgif_train_subset_local.py`](scripts/research/audit_tgif_train_subset_local.py).
- **Bộ Kiểm Thử Tự Động**: [`ml/tests/test_tgif_train_intake.py`](ml/tests/test_tgif_train_intake.py) (8/8 PASS).

### 5. Kế Hoạch Sau Intake (Post-Intake Protocol)
1. Người dùng khởi chạy notebook trên Colab CPU để tải và đóng gói `tgif_train_clean_subset_package.zip`.
2. Tải package ZIP về máy LOCAL vào thư mục `data/research/local-artifacts/phase-4c.7b/`.
3. Chạy lệnh audit tại LOCAL:
   ```bash
   python scripts/research/audit_tgif_train_subset_local.py --package-zip data/research/local-artifacts/phase-4c.7b/tgif_train_clean_subset_package.zip
   ```
4. Kiểm toán an toàn ZIP, đối soát bộ ba, kiểm tra leakage Disjoint Guard (0 overlap), và mở HTML contact sheet tự chứa để người dùng thẩm định.
5. **DỪNG LẠI TRƯỚC EVALUATION** để bàn giao kết quả intake cho người dùng.

---

## LL. Xác Minh Thực Địa Tiếp Tục Nghiên Cứu bằng Benchmark Có Sẵn (TGIF/TGIF2 Priority Audit & Option P Cross-Disjointness) (2026-10-09)

Hồ sơ xác minh kỹ thuật và siêu dữ liệu thực tế tại chỗ (Local Metadata & Benchmark Verification) nhằm đánh giá khả năng tiếp tục kiểm định độc lập bằng bộ dữ liệu benchmark có sẵn, sau khi đã chốt dừng hoàn toàn việc tự sinh ảnh inpainting thủ công.

### 1. Khảo Sát Nguồn Sơ Cấp & Siêu Dữ Liệu Thực Tế (Primary Sources & Empirical Metadata)

Dự án tái sử dụng toàn diện các tài nguyên đã thẩm định trong repository: Dataset Registry (`datasets/registry.json`), Bibliography chuẩn tắc (`docs/references.bib`: `[@mareen2024tgif]`, `[@mareen2026tgif2]`, `[@giakoumoglou2025sagi]`, `[@zhu2023genimage]`), và Bảng đăng ký minh chứng (`docs/EVIDENCE_REGISTER.md`: `EV-MASK-CARDINALITY-RESOLVED-001`, `EV-TGIF-CARDINALITY-001`, `EV-TGIF-SEMANTICS-001`).

1. **Khả năng lấy bộ ba tương ứng (Tripartite Tuple)**:
   - Hệ thống hoàn toàn **LẤY ĐƯỢC 100%** bộ ba tương ứng gồm `authentic` (ảnh máy ảnh MS-COCO val2017 gốc), `ai_edited` (ảnh inpainting), và `ground_truth_mask` (mặt nạ nhị phân vùng can thiệp).
   - *Mask*: Toàn bộ kho mặt nạ của TGIF gồm 31,238 file mask PNG (141.5 MB giải nén, 2,242 unique COCO `source_id`, 3,124 task instances) **ĐÃ CÓ SẴN 100% TRÊN MÁY CỤC BỘ** tại `data/research/tgif/masks/` từ Phase 4B.0 (manifest tại `data/research/tgif/manifests/masks-manifest.jsonl`).
   - *Authentic*: Tập ảnh gốc MS-COCO val2017 có sẵn trong kho lưu trữ của TGIF (`orig`) và trên CDN chính thức của MS-COCO (`http://images.cocodataset.org/val2017/`).
   - *Edited*: Có sẵn trong các thư mục inpainting của TGIF (`sd2-sp` - Stable Diffusion 2 spliced, `ps-sp` - Adobe Firefly spliced).
2. **Phân biệt bản chất `sp` (Spliced) vs `fr` (Fully-Regenerated) và ý nghĩa Ground-Truth Mask**:
   - Tuân thủ nghiêm ngặt kiểm toán ngữ nghĩa nhãn trong `docs/PILOT_PROTOCOL.md` (Mục 2.2):
     - **`sp` (Spliced inpainting)**: Chỉ vùng bên trong mask được mô hình AI sinh ra và ghép đè (*spliced*) trở lại ảnh chụp MS-COCO gốc; 100% pixel bên ngoài mask là pixel máy ảnh thật nguyên bản. Đây là đại diện chuẩn mực cho nhãn khoa học **`ai_edited`**. Ground-truth mask của `sp` phản ánh **chính xác 100% ranh giới giữa pixel thật và pixel AI**, phục vụ trực tiếp cho bài toán định vị (Auxiliary RQ5 Localization: mIoU, Dice, Heatmap IoU).
     - **`fr` (Fully-Regenerated)**: Toàn bộ canvas đi qua quá trình khuếch tán/VAE của mô hình inpainting mà không ghép đè trở lại. Mọi pixel đều mang dấu vết sinh số, nhưng nội dung và bố cục không gian bị điều kiện hóa bởi ảnh MS-COCO gốc. Theo `PILOT_PROTOCOL.md`, `fr` **KHÔNG ĐƯỢC COI LÀ `fully_generated`** (để tránh mâu thuẫn nhãn phân loại); đồng thời mask trong `fr` chỉ là bounding box / request prompt lúc chạy inpainting, **KHÔNG PHẢI là ranh giới giữa pixel thật và pixel AI**. Do đó `fr` không có ranh giới ghép nối cục bộ và không dùng làm ground-truth localization theo nghĩa biên thật/giả.
3. **Danh sách Source IDs để lọc trước khi tải ảnh**:
   - **ĐÃ CÓ SẴN CỤC BỘ 100%**. Tệp `data/research/tgif/manifests/masks-manifest.jsonl` (31,238 dòng) chứa đầy đủ danh mục `source_id` (12 chữ số chuẩn COCO) và `instance_id` (`{category}_{source_id}`) cho toàn bộ 3 split:
     - Split `train`: **1,558 unique COCO `source_id`** (tạo thành 2,440 task instances do 571 ảnh COCO chứa nhiều nhãn đối tượng khác nhau).
     - Split `val`: **341 unique COCO `source_id`** (341 task instances).
     - Split `test`: **343 unique COCO `source_id`** (343 task instances).
   - Nhờ có sẵn manifest cục bộ, hệ thống có thể lọc và lập danh sách ID cần lấy **TRƯỚC KHI** thực hiện bất kỳ lệnh tải mạng nào.
4. **Cách tải chọn lọc, dung lượng dự kiến, giấy phép và attribution**:
   - *Cách tải và dung lượng*:
     - Kho Nextcloud công khai của TGIF (`https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o`) đóng gói theo split archive: `orig_training.tar.gz` (5.26 GiB), `sd2-sp_training.tar.gz` (13.37 GiB).
     - Để tránh tải toàn bộ 18.6 GiB về máy trạm cá nhân (tránh nguy cơ nghẽn đĩa và vi phạm trần mạng):
       * Nguồn authentic: Tải trực tiếp $N$ ảnh gốc từ official MS-COCO CDN (`http://images.cocodataset.org/val2017/{source_id}.jpg`) $\to$ Dung lượng cực nhỏ: chỉ ~40–60 MB cho 400 ảnh!
       * Nguồn mask: Đã có sẵn 100% trên đĩa local $\to$ **0 MB tải thêm**.
       * Nguồn edited: Khuyến nghị thực thi một tác vụ trích xuất trên Google Colab (tải `sd2-sp_training.tar.gz` về môi trường Colab tạm thời, trích xuất chính xác $N$ ảnh theo danh sách ID đã preregister, đóng gói thành package ZIP nhỏ ~60 MB tải về local). Tổng dung lượng nhận về local chỉ ~100–120 MB.
   - *Giấy phép & Attribution*:
     - TGIF được cấp phép dưới `CC BY-SA 4.0`. Ảnh gốc MS-COCO cấp phép dưới `CC BY 4.0` (và Flickr per-image terms).
     - Attribution chuẩn tắc: Mareen et al., *"TGIF: Text-Guided Inpainting Forgery Dataset"*, IEEE WIFS 2024 (`@mareen2024tgif`).
     - Tuân thủ ADR-0006: Thuộc Research Track (`data/research/`), chỉ dùng cho nghiên cứu học thuật/khóa luận, cấm đưa vào sản phẩm thương mại Product Track.

### 2. Đối Chiếu Metadata Với 684 Nguồn Option P Lịch Sử (Disjointness Audit)

Nhằm bảo đảm tính độc lập khoa học tuyệt đối, hệ thống thực hiện đối chiếu chéo metadata giữa kho TGIF và toàn bộ 684 nguồn ảnh Option P đã sử dụng trong các Phase trước (tại `data/research/tgif/manifests/manifest_pilot_a_option_p.csv`):

1. **Tuyệt đối không coi "thư mục chưa dùng" là bằng chứng disjoint**:
   - Trong TGIF, toàn bộ 4 thư mục inpainting (`sd2-sp`, `ps-sp`, `sd2-fr`, `sdxl-fr`) và thư mục ảnh gốc `orig` đều bắt nguồn từ cùng một tập 2,242 ảnh nguồn MS-COCO.
   - Option P lịch sử đã sử dụng toàn bộ 684 ảnh nguồn của hai split `validation` (341 sources: 250 `development_train` + 91 `inner_validation`) và `testing` (343 sources: `locked_test`).
   - Nếu lấy ảnh từ các thư mục "chưa từng tải" như `ps-sp` (Photoshop) hay `sdxl-fr` thuộc split `validation` hoặc `testing`, thì những ảnh này **CHIA SẺ 100% CÙNG NGUỒN GỐC ẢNH VÀ NỀN MS-COCO** với tập dev và locked-test mà mô hình đã học hoặc niêm phong. Chúng chỉ là các biến thể (*variants*) khác nhau của cùng một bức ảnh gốc!
   - Việc kiểm tra trên các thư mục này sẽ gây rò rỉ danh tính ảnh nguồn (*source identity leakage*), phá vỡ nguyên tắc Group Split.
2. **Nhóm mọi biến thể của cùng ảnh nguồn (Source-Level Grouping)**:
   - Mọi tệp ảnh mang cùng tiền tố COCO image ID 12 chữ số (ví dụ `000000002261`) đều được gom chung vào một nhóm nguồn thống kê duy nhất (`source_id`).
   - Bất kỳ ảnh nào thuộc 684 `source_id` của Option P đều bị **CHẶN TUYỆT ĐỐI** khỏi tập kiểm định mới.
3. **Phân vùng Disjoint thực tế trong TGIF**:
   - Nguồn duy nhất trong TGIF chưa từng xuất hiện trong Option P là **split `training` gốc** của TGIF.
   - Kết quả kiểm toán giao tập giữa các split của TGIF và Option P (684 sources):
     - Giao giữa Option P và TGIF Val (341 sources): **341 / 341 (100% trùng)**.
     - Giao giữa Option P và TGIF Test (343 sources): **343 / 343 (100% trùng)**.
     - Giao giữa Option P và TGIF Train (1,558 unique sources): **0 / 1,558 (0.0% trùng lặp — 100% DISJOINT TUYỆT ĐỐI)**.
4. **Phân định các kiểm tra đã hoàn thành vs cần tải ảnh mới làm được**:
   - *Kiểm tra ID/Metadata đã hoàn thành 100%*: Đối chiếu 1,558 `source_id` của split Train với 684 Option P IDs đạt 0 collision. Toàn bộ 24,400 file mask trong Train đã có sẵn mã băm SHA-256 cục bộ.
   - *Kiểm tra cần tải ảnh mới làm được*: Mã băm SHA-256 của file ảnh gốc tải về (`raw_sha256`), mã băm ảnh sau chuẩn hóa 512x512 (`master_sha256`), và khoảng cách perceptual hash (`pHash`).
5. **Giới hạn tuyên bố khoa học về Pretraining**:
   - MS-COCO val2017 là tập dữ liệu mở phổ biến từ năm 2014. Các mô hình thị giác lớn (MobileNetV3 pretrained trên ImageNet) và các mô hình khuếch tán (Stable Diffusion 2, SDXL) có thể đã tiếp xúc với phân phối ảnh COCO trong các tập dữ liệu tiền huấn luyện công khai khổng lồ (LAION, v.v.).
   - Dự án **KHÔNG TUYÊN BỐ và KHÔNG CÓ BẰNG CHỨNG** để khẳng định dữ liệu MS-COCO chưa từng xuất hiện trong pretraining của các mô hình nền tảng bên ngoài.
   - Tính "độc lập" ở đây **chỉ là tính độc lập thực nghiệm (experimental disjointness)** đối với quy trình huấn luyện, tinh chỉnh, chọn mô hình và hiệu chuẩn nội bộ của dự án này (hoàn toàn tách biệt khỏi 684 ảnh Option P).

### 3. Đề Xuất Đúng Một Tập Kiểm Định Khả Thi (Proposed Independent Benchmark Cohort)

Dựa trên kết quả kiểm toán, đề xuất đúng MỘT tập kiểm định độc lập khả thi kỹ thuật:

- **Tên tập kiểm định**: `TGIF-Train-Independent-Benchmark-Subset` (viết tắt: **`TGIF-Train-Clean-Subset`**).
- **Nguồn dữ liệu tuyển chọn**:
  - Tuyển chọn từ **split `training` gốc của TGIF**:
    * Nhánh Authentic: Ảnh gốc MS-COCO val2017 tương ứng với các task đã chọn.
    * Nhánh AI-Edited: Ảnh `sd2-sp_training` (Stable Diffusion 2 inpainting ghép đè).
    * Nhánh Mask: Mặt nạ phân đoạn đối tượng tương ứng từ `masks/training/` (đã có sẵn).
- **Phạm vi kiểm định (Validation Scope)**:
  - *Về ảnh nguồn*: Kiểm định trên **ảnh nguồn mới chưa từng thấy trong cùng phân phối dataset** (**Unseen In-Distribution Sources**). Cả 1,558 ảnh nguồn này hoàn toàn mới đối với mô hình của dự án.
  - *Về công cụ sinh*:
    * Nhánh chính: Kiểm định trên **cùng công cụ sinh (Same Generator - Inpainting SD2)** trên ảnh nguồn mới độc lập.
    * Nhánh mở rộng (nếu tải thêm được `ps-sp_training` từ TGIF): Kiểm định trên **công cụ sinh mới chưa từng học (Cross-Generator Generalization - Adobe Firefly vs SD2)**.
- **Quy mô và Điều chỉnh Giao thức (Protocol Amendment)**:
  - Phase 4C.7A trước đó thiết kế ma trận $2 \times 2$ gồm 400 cặp (COCO + Unsplash $\times$ SD2 + SDXL).
  - Tuy nhiên, việc tạo inpainting thủ công trên Unsplash và SDXL đã bị chốt dừng sau thất bại kỹ thuật của Pilot v2. Trong benchmark có sẵn TGIF, toàn bộ ảnh nguồn là MS-COCO và công cụ ghép có mask là SD2-sp (và PS-sp).
  - Do đó, **không thể giữ nguyên ma trận nguồn cũ có Unsplash**.
  - **Đề xuất điều chỉnh protocol trình duyệt**:
    * Giữ quy mô $N_{\text{target}} = 400$ cặp (400 authentic vs 400 ai_edited), tuyển chọn ngẫu nhiên có kiểm soát từ 1,558 ảnh nguồn sạch của TGIF Train.
    * Cân bằng trực giao theo **3 khoảng diện tích mask**: Small (1%–10%), Medium (10%–30%), Large (30%–50%).
    * (Hoặc phương án tinh gọn $N=200$ cặp nếu người dùng muốn tối thiểu hóa dung lượng tiếp nhận).
    * Ban hành Protocol Amendment điều chỉnh cơ cấu phân tầng trước khi tiếp nhận dữ liệu.

### 4. Đăng Ký Tiền Kiểm Chuẩn (Preregistration Protocol)

Nhằm triệt tiêu hoàn toàn nguy cơ thiên vị chọn mẫu (*selection bias*) và gian lận khoa học, toàn bộ quy trình được đăng ký trước (Preregistered):

1. **Tiêu chuẩn tuyển chọn mẫu (A Priori Selection Criteria)**:
   - *Mặt nạ phân đoạn thực tế*: Chỉ chọn mặt nạ phân đoạn (`segm` - segmentation mask), **loại bỏ 100% mặt nạ hộp chữ nhật (`bbox`)** để bảo đảm vùng chỉnh sửa bám sát đường biên vật thể tự nhiên, không tạo ra các cạnh cắt hộp nhân tạo.
   - *Biến thể inpainting cố định*: Chọn cố định biến thể `_0.png` (variation 0) của mỗi instance.
   - *Ghép cặp 1:1 nghiêm ngặt*: Mỗi bức ảnh nguồn MS-COCO chỉ xuất hiện đúng 1 lần duy nhất trong tập kiểm định (1 authentic ↔ 1 ai_edited ↔ 1 mask). Nếu một ảnh có nhiều category task, chỉ chọn 1 task instance ngẫu nhiên bằng PRNG seed cố định (`seed = 20261010`).
   - *Phân bổ diện tích mask cân bằng*:
     * Small ($1\% \le \text{mask\_area} < 10\%$): chiếm ~33.3% quota.
     * Medium ($10\% \le \text{mask\_area} < 30\%$): chiếm ~33.3% quota.
     * Large ($30\% \le \text{mask\_area} \le 50\%$): chiếm ~33.3% quota (chặn trên 50% để giữ bản chất can thiệp cục bộ).
2. **Quy trình tiền xử lý ảnh và mask (Standardized Preprocessing)**:
   - Chuẩn hóa canvas: Resize/crop bảo toàn tỷ lệ về $512 \times 512$ RGB bằng Lanczos / bicubic letterboxing trung tính theo đúng contract của `ml/evaluation/`.
   - Mask: Binarize nghiêm ngặt thành ảnh 1 kênh $L$ với giá trị $\in \{0, 255\}$ (ngưỡng 128).
   - Lưu trữ: Định dạng lossless PNG cho cả ảnh và mask. Không nén lại JPEG trong quá trình nạp dữ liệu gốc.
3. **Tiêu chí loại trừ định trước (A Priori Exclusion Criteria)**:
   - Trùng `source_id` với bất kỳ nguồn nào trong 684 Option P sources (loại ngay lập tức ở bước lọc ID).
   - Lỗi giải mã PIL (tệp hỏng/không đọc được).
   - Ảnh blank hoặc độ lệch chuẩn thấp ($\text{std} < 2.0$).
   - Diện tích mask ngoài dải quy định ($< 1\%$ hoặc $> 50\%$).
   - Mask rỗng hoặc không có sự thay đổi pixel giữa authentic và edited ($\text{inside\_mean\_L1} < 1.0$).
4. **Cam kết không chọn dữ liệu theo điểm mô hình (Zero Model-Informed Selection)**:
   - Danh mục candidate catalog và edit manifest được tạo và khóa bằng mã băm SHA-256 **HOÀN TOÀN ĐỘC LẬP VÀ TRƯỚC KHI** nạp mô hình vào bộ nhớ.
   - `assert_detector_isolation()` fail-closed: Tuyệt đối cấm xem điểm detector, xác suất dự đoán hay phân tích độ chính xác để chọn hoặc thay thế mẫu.
5. **Nhận thức về Shortcut của Benchmark Công Khai**:
   - Thừa nhận khoa học trung thực: TGIF sử dụng phương pháp dán vùng inpainting vào ảnh gốc (splicing). Quy trình này có thể để lại các forensic shortcuts như sai lệch mức nhiễu cảm biến giữa vùng dán và nền thật, hiện vật nén VAE decoder, hoặc bước nhảy biên vi mô.
   - Kết quả kiểm định trên tập này phản ánh năng lực phát hiện inpainting trong điều kiện của benchmark, không suy diễn thành năng lực phát hiện tuyệt đối trong môi trường mở thế giới thực.

### 5. Bàn Giao Ngắn & Kế Hoạch Tiếp Nhận Cụ Thể

- **Nguồn được chọn**: TGIF Training Split (`orig_training` + `sd2-sp_training`).
- **Số ảnh nguồn đủ điều kiện**:
  - *Đã xác minh thực tế*: **1,558 unique COCO sources** (2,440 inpainting tasks) trong kho mask `data/research/tgif/masks/training/`, 100% disjoint với 684 Option P sources.
  - *Còn UNKNOWN*: Số lượng file thực tế trong archive `ps-sp_training` của TGIF (cần kiểm toán PROPFIND chi tiết trên Nextcloud); dung lượng nén chi tiết từng split của TGIF2 (archive tổng >110 GB); tình trạng bản quyền của SAGI-D (đang `blocked`).
- **Cách tải & Dung lượng dự kiến**:
  - Mask: 0 MB (đã có sẵn 100% cục bộ).
  - Authentic: Tải 400 ảnh gốc từ MS-COCO CDN (`http://images.cocodataset.org/val2017/`) $\to$ ~40–60 MB.
  - Edited: Thực thi script trích xuất trên Colab từ `sd2-sp_training.tar.gz`, đóng gói ZIP package 400 ảnh $\to$ ~60 MB tải về local.
  - Tổng dung lượng nhận về local: **~100–120 MB** (hoàn toàn khả thi, không gây tràn đĩa).
- **Hạn chế chính**: Chỉ kiểm định trên miền ảnh MS-COCO; không có ảnh Unsplash; mô hình có thể phát hiện dựa trên đặc trưng ghép nối của TGIF.
- **Kế hoạch tiếp nhận 5 bước trình người dùng phê duyệt**:
  1. *Bước 1*: Người dùng duyệt đề xuất tập kiểm định `TGIF-Train-Clean-Subset` và phê duyệt Protocol Amendment điều chỉnh ma trận nguồn ($N=400$ hoặc $N=200$ cặp COCO Train $\times$ SD2-sp).
  2. *Bước 2*: Chạy script local tạo danh mục candidate catalog và manifest tuyển chọn từ `masks-manifest.jsonl`, khóa chặt bằng SHA-256.
  3. *Bước 3*: Tải ảnh authentic từ MS-COCO CDN (~50 MB) và đóng gói Colab worker trích xuất ảnh edited tương ứng (~60 MB).
  4. *Bước 4*: Ingest package, chạy Technical QC tự động (PIL decode, std check, outside/inside L1, mask bounds) và lập audit receipt.
  5. *Bước 5*: Thực hiện kiểm định độc lập với 5 outer-fold checkpoints theo đúng protocol Phase 4C.7A.

---

## KK. Chốt Đánh Giá Tính Khả Thi Phase 4C.7B sau Pilot v2 & Đề Xuất Nghiên Cứu Tiếp Tục bằng Dữ Liệu Sẵn Có (2026-10-09)

Hồ sơ chốt đánh giá tính khả thi kỹ thuật của Phase 4C.7B sau đợt chạy Pilot v2 (`pilot-20261009T111247Z`), tích hợp đầy đủ chuỗi thực chứng từ 4 đợt chạy độc lập (Pilot v1, Diagnostic, Calibration, Pilot v2) và đề xuất phương án tiếp tục nghiên cứu trên dữ liệu có sẵn. Không chạy thêm pilot, calibration, retry hay generation.

### 1. Phân Định Rạch Ròi Ba Cấp Độ Thẩm Định Kỹ Thuật và Nội Dung

Nhằm đảm bảo tính trung thực khoa học tuyệt đối (*Scientific Honesty*), hệ thống phân tách nghiêm ngặt ba cấp độ thẩm định:

1. **Technical QC (Tự động / Số học)**:
   - *Bản chất*: Bộ lọc tự động kiểm tra định dạng file ảnh PNG lossless $512 \times 512$, độ lệch chuẩn ảnh authentic $\ge 5.0$, delta tuyệt đối nội vùng mask L1 $\ge 3.0$, delta ngoại vùng mask $\le 0.5$, và mask nhị phân $\{0, 255\}$.
   - *Kết quả*: **PASS 100%** trên toàn bộ các attempt hợp lệ đã thực hiện (nhờ lớp ghép 1-bit binary request mask compositing cưỡng chế `outside_mean_l1 == 0.000000` và `outside_max_delta == 0.0`).
   - *Ý nghĩa pháp lý*: Technical QC chỉ xác nhận pipeline không lỗi crash, tệp ảnh hợp lệ và không biến đổi pixel ngoài mask; **tuyệt đối không đo lường độ chính xác ngữ nghĩa hay chất lượng thị giác**.
2. **Khuyến nghị của Agent (Sơ bộ / Heuristic)**:
   - *Bản chất*: Đánh giá định tính ban đầu của tác nhân AI dựa trên các tiêu chí kiểm tra trực quan (sự hiện diện vật thể, vị trí/tỷ lệ, độ chân thực/ánh sáng, chất lượng đường biên).
   - *Kết quả Pilot v2*: Agent khuyến nghị **1 ACCEPT** (Slot 2: tủ bếp navy-blue) / **7 REJECT** (Slot 1, 3, 4, 5, 6, 7, 8).
   - *Ý nghĩa pháp lý*: Khuyến nghị của Agent chỉ là thông tin hỗ trợ kỹ thuật; **tuyệt đối không có giá trị nghiệm thu chính thức và không được coi là tỷ lệ thành công của hệ thống**.
3. **Quyết định Human Content QC (Chính thức / Quyết định Người Thẩm Định)**:
   - *Bản chất*: Quyết định phê duyệt hoặc từ chối độc lập của người thẩm định con người (Dũng Phạm `<valdung04@gmail.com>`).
   - *Hiện trạng trong toàn bộ Phase 4C.7B*:
     - **Diagnostic run** (`diag-20261008T154628Z`): **6/6 REJECT** (Đã chốt chính thức tại `diagnostic_content_qc_determinations.json` lúc `2026-10-08T19:34:30Z`).
     - **Pilot v2** (`pilot-20261009T111247Z`): **8 cặp PENDING** (`PENDING_CONTENT_QC`, chưa có quyết định người duyệt).
     - **Calibration** (`calib-20261009T015749Z`): **2 attempts PENDING** (`PENDING_CONTENT_QC`, chưa có quyết định người duyệt).
     - **Follow-up pilot v1** (`pilot-20261008T113700Z`): **8 cặp PENDING** (`PENDING_CONTENT_QC`, chưa có quyết định người duyệt).
     - **Historical pilot v0** (`pilot-20261007T132003Z`): **8 cặp PENDING** (`PENDING_CONTENT_QC`).
   - *Quy tắc bất biến*: Chỉ ghi nhận ACCEPT/REJECT khi người dùng đã ra quyết định chính thức; tất cả các cặp chưa được duyệt tiếp tục giữ nguyên trạng thái **PENDING**.

### 2. Tổng Hợp Kết Quả Các Run Thực Nghiệm và Phân Loại Khiếm Khuyết

Tổng hợp ngắn các đợt chạy đã thực hiện trong Phase 4C.7B, bao quát đầy đủ các dạng thất bại kỹ thuật và chất lượng tạo sinh (liên kết trực tiếp các run, commit, và artifact hiện có mà không tạo dossier trùng lặp):

1. **Bỏ sót đối tượng (Semantic Omission)**:
   - *Cà chua nhỏ* (`IND_COCO_SDXL_002`, mask 3.78%): 4 attempts qua 3 đợt chạy (Pilot v1, Diagnostic Arm A, Calibration 2 attempts với guidance 7.5 và 9.5) đều thất bại hoàn toàn. Mô hình inpaint vân ruột bánh mì thay vì tạo quả cà chua bi.
   - *Cặp táp da* (`IND_COCO_SDXL_042`, mask 5.95%, Pilot v2 Slot 3): Omission 100% trên nền thảm đỏ phẳng đồng nhất (max delta vùng target chỉ 37.0).
   - *Chim bay* (`IND_COMMONS_SDXL_001`, mask 6.58%): Omission 100% trên bầu trời mây trong Pilot v1, Diagnostic Arm A, và Pilot v2 (Slot 7).
   - *Vali hành lý* (`IND_COMMONS_SD2_002`, mask 14.76%): Omission 100% trên đường đá cuội trong Pilot v1 và Diagnostic Arm A.
   - *Chậu hoa* (`IND_COMMONS_SD2_040`, mask 12.76%, Pilot v2 Slot 6): Tạo được khối thùng gỗ mộc nhưng bỏ sót hoàn toàn hoa (thùng rỗng).
2. **Sai ngữ nghĩa & Ảo giác (Semantic Hallucination)**:
   - *Vali hành lý* (`IND_COMMONS_SD2_002`, Diagnostic Arm B, crop 1.6x): Mô hình tạo ra một chiếc xe hơi đồ chơi cổ hoàn chỉnh có bánh và mui thay vì chiếc vali hành lý, dù inside L1 tăng vọt (43.39 vs 30.74).
   - *Đồng hồ treo tường* (`IND_COCO_SD2_001`, Pilot v2 Slot 1): Thất bại tạo hình đồng hồ tròn; sinh mảng phản xạ kính/kim loại đa diện méo mó.
   - *Bàn dã ngoại* (`COCO_EXT_SDXL_001`, Pilot v2 Slot 4): Dị dạng cấu trúc nặng nề, chân bàn phủ lớp màu xanh giống ống quần; nan sắt chìm dưới cát và chao đèn lơ lửng.
   - *Mũi đất biển* (`IND_COMMONS_SD2_001`, Pilot v2 Slot 5): Xóa sạch mũi đất tiền cảnh nhưng tự tái sinh một sườn vách đá phấn trắng dựng đứng mới thay thế đường bờ vịnh/thị trấn xa.
3. **Sai lệch vị trí & Đứt gãy hình học (Placement Deficit & Structural Severance)**:
   - *Chim bay* (`IND_COMMONS_SDXL_001`, Diagnostic Arm B): Sinh được bóng chim nhưng trôi lệch xuống dưới target box $dy = +41.0$ px ($dx = -30.5$ px, độ trùng khớp theo chiều dọc = 0 px).
   - *Cột đá kỷ niệm* (`IND_COMMONS_SDXL_005`, Pilot v2 Slot 8): Sinh được cột đá nhưng đỉnh ngọn tượng đồng cũ thò ra ngoài đỉnh mask ($y < 40$); rạp cam phía sau bị cắt cụt thẳng đứng tại $x=135$.
   - *Cắt cụt chảo đồng lân cận* tại Slot 1 (`IND_COCO_SD2_001`) do biên mask chữ nhật AABB cắt ngang chảo.
   - *Cắt vỡ chân cửa gỗ lịch sử* tại Slot 6 (`IND_COMMONS_SD2_040`) do mask chữ nhật chạm chân công trình.
   - *Cắt đứt lan can kim loại 210 px* tại `IND_COMMONS_SDXL_003` trong Pilot v1.
4. **Lỗi bậc biên & Lệch tông mảng lớn (Boundary Seams & Tonal Steps)**:
   - *Bước nhảy vi mô tại biên mask*: Hard binary compositing (1-bit) luôn tạo ra bậc nhảy tương phản vi mô tại biên (top/left edge L1 step 8.17–9.56 vs authentic 1.81–4.12 trên cà chua).
   - *Lệch tông diện tích lớn*: Mảng trời chữ nhật đậm màu bao quanh chim trong Diagnostic Arm B và Pilot v2 Slot 7 ($\Delta\text{RGB} \approx [-10, -11, -3]$); mảng trần nhà 30% area trong Pilot v1 (`IND_COCO_SDXL_041`); mảng thảm đỏ lệch tông trong Pilot v2 Slot 3.
   - *Thử nghiệm feathering cosine ($k=2$ px)*: Đã kiểm chứng thực nghiệm tại `derived_feathered_k2/` — chỉ làm mịn bậc vi mô 1–2 px nhưng không thể khắc phục sự đứt gãy hình học vĩ mô hay tonal mismatch diện tích lớn.
5. **Liên kết Lineage Artifacts Hiện Có**:
   - `pilot-20261009T111247Z`: Functional commit `79775250ee5393c9f193bdc0485f77998cefaddf`, artifact `data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z/`, review sheet `pilot_v2_content_qc_review_contact_sheet.html`.
   - `calib-20261009T015749Z`: Functional commit `045ea70cb9067ede3833f7a01562199869f4ae56`, artifact `data/research/local-artifacts/phase-4c.7b/calib-20261009T015749Z/`, review sheet `calibration_contact_sheet.html`.
   - `diag-20261008T154628Z`: Functional commit `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`, artifact `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/`, review sheet `diagnostic_contact_sheet.html`, dossier `diagnostic_content_qc_determinations.json`.
   - `pilot-20261008T113700Z`: Functional commit `d9d99b678053972436032a828056a76a6392fbb5`, artifact `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/`, review sheet `content_qc_contact_sheet.html`.
   - `pilot-20261007T132003Z`: Historical commit `7d2eea4027e2a17b51e6665ff81d481e4e333d48`, artifact `data/research/local-artifacts/phase-4c.7b/pilot-20261007T132003Z/`.

### 3. Kết Luận Có Giới Hạn (Bounded Feasibility Conclusion)

Dựa trên toàn bộ dữ liệu thực chứng thu được, kết luận chính thức về tính khả thi của quy trình hiện tại được xác lập như sau:

> **“Chưa đủ bằng chứng để mở rộng cấu hình hiện tại lên 400 cặp.”**

Các nguyên tắc khoa học giới hạn kết luận:
1. **Không coi 1/8 khuyến nghị ACCEPT là tỷ lệ nghiệm thu chính thức**:
   - Con số 1/8 chỉ là khuyến nghị sơ bộ của Agent từ quan sát thị giác; toàn bộ 8 cặp vẫn đang giữ trạng thái `PENDING_CONTENT_QC` chờ người dùng thẩm định.
   - Về mặt thống kê, một mẫu pilot $n=8$ mang tính thăm dò không đủ độ tin cậy để suy rộng thành tỷ lệ chấp nhận của toàn bộ cohort $N=400$.
2. **Không khẳng định local crop đã giải quyết chất lượng**:
   - Mặc dù Arm B trong diagnostic tạo được vật thể ở 2/2 ca SDXL, toàn bộ **6/6 attempts trong diagnostic đã bị Human Content QC chính thức REJECT** (do sai lệch placement $dy = +41.0$ px, bậc nhảy tương phản biên, texture disparity, và xe hơi đồ chơi hallucination trên SD2).
   - Local crop giải quyết được độ phân giải cục bộ nhưng gây mất ngữ cảnh toàn cục, làm phát sinh các lỗi nghiêm trọng về vị trí và độ đồng nhất tông màu.
3. **Không suy rộng thành kết luận mọi mô hình inpainting đều thất bại**:
   - Kết luận này **chỉ giới hạn nghiêm ngặt trong cấu hình kỹ thuật cụ thể đã kiểm thử thực nghiệm**: Stable Diffusion 2 Inpainting (qua mirror cộng đồng) và SDXL Inpainting 1.0 (Diffusers), hoạt động trên canvas toàn phần $512 \times 512$, sử dụng mặt nạ hình chữ nhật AABB ghép nhị phân 1-bit, trên tập prompt và candidate của catalog hiện hành.
   - Kết luận không khẳng định các công nghệ inpainting khác (như FLUX Inpainting, BrushNet, PowerPaint, contour-guided inpainting, hoặc multi-scale Poisson blending) đều thất bại.

### 4. Đề Xuất Tiếp Tục Nghiên Cứu bằng Dữ Liệu Sẵn Có

Để giải quyết bài toán kiểm định độc lập mà không tiếp tục tiêu tốn tài nguyên vào việc vi điều chỉnh inpainting thủ công, đề xuất phương án chuyển trọng tâm sang sử dụng các bộ dữ liệu / benchmark có sẵn.

#### 4.1. Rà Soát Dataset / Benchmark trong Registry và Bibliography Hiện Có

| Dataset / Benchmark | Nguồn Sơ Cấp (Primary Source) | Giấy Phép & Bản Quyền | Cấu Trúc Nhãn & Mask | Trạng Thái Registry | Đánh Giá Tính Sẵn Sàng cho Nghiên Cứu Hiện Tại |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TGIF** | Mareen et al., WIFS 2024 `[@mareen2024tgif]` | `CC BY-SA 4.0` (COCO base `CC BY 4.0`) | `ai_edited` vs `authentic`, ground-truth masks binary (`bbox`, `segm`) | `research-only`, `verified` | Đã có Option P trong repo (684 task instances). Kho Nextcloud còn các subfolder chưa dùng (`sd2-fr`, `sdxl-fr`, `ps-sp`). Có thể tuyển chọn tập con ngoại vi sạch với điều kiện vượt qua Disjoint Guard với 684 nguồn Option P. |
| **TGIF2** | Mareen et al., JIS 2026 `[@mareen2026tgif2]` | `CC BY-SA 4.0` | `ai_edited` vs `authentic`, FLUX.1 inpainting, ground-truth masks | `research-only`, `verified` | Mở rộng TGIF thêm FLUX.1 (schnell, dev, filldev). Dung lượng nén rất lớn (>180 GB). Cần lọc tập con nhỏ và kiểm toán disjointness với MS-COCO Option P. |
| **SAGI-D** | Giakoumoglou et al., ICCV 2025 `[@giakoumoglou2025sagi]` | Code: Apache-2.0 / Dataset: Academic Research | `ai_edited` (>95k ảnh inpainting) + binary masks | `blocked`, `proposed` | Cần xác minh nguồn sơ cấp về quyền phân phối lại của các ảnh nguồn gốc trước khi mở khóa tiếp nhận. |
| **GenImage** | Zhu et al., NeurIPS 2023 `[@zhu2023genimage]` | `CC BY-NC-SA 4.0` | `authentic` vs `fully_generated` (8 generators), **KHÔNG CÓ mask** | `research-only`, `verified` | Đã đăng ký trong registry. Tuy nhiên **chỉ có dữ liệu tạo sinh toàn phần (`fully_generated`)**, không có nhãn `ai_edited` và không có mask inpainting. |
| **RAID / RealHD** | RAID benchmark / RealHD | Code Apache-2.0 / Chưa công bố | Hỗn hợp, unreleased | `blocked` | Chưa đủ điều kiện pháp lý và kỹ thuật để tiếp nhận. |

#### 4.2. Đối Chiếu với Câu Hỏi Nghiên Cứu AI-Edited Hiện Tại

Câu hỏi nghiên cứu hiện tại yêu cầu:
1. **Không gian nhãn**: Nhị phân `authentic` vs `ai_edited` (hoặc mở rộng 3 lớp với `fully_generated`).
2. **Provenance & Giấy phép**: Nguồn gốc xác minh, giấy phép phù hợp với Research Track (ADR-0006: cấm đưa vào sản phẩm, chỉ dùng cho nghiên cứu/bài báo).
3. **Split & Chống Leakage (Tuyệt đối quan trọng)**:
   - Các mô hình hiện tại (5 outer-fold checkpoints) đã được huấn luyện/đánh giá trên 684 nguồn Option P lịch sử (COCO sources).
   - Bất kỳ bộ dữ liệu ngoại vi nào đưa vào đánh giá độc lập **BẮT BUỘC phải vượt qua 4-level Disjoint Guard**: Không trùng `source_id`, không trùng `origin_id`, không trùng mã băm ảnh gốc `raw_sha256`, và không trùng mã băm ảnh chuẩn hóa `master_sha256`.
4. **Mask vùng sửa (Ground-Truth Mask)**:
   - Cần thiết để đánh giá **Auxiliary RQ5 (Inpainting Heatmap Localization)** thông qua mIoU/Dice giữa patch heatmap của mô hình và mask thực tế.
5. **Khả năng đánh giá độc lập**: Dữ liệu chưa từng được mô hình tiếp xúc trong quá trình huấn luyện, chọn siêu tham số hay hiệu chuẩn.

#### 4.3. Phân Định Rõ Hai Hướng Lựa Chọn

- **Lựa chọn A: Giữ Nguyên Câu Hỏi Nghiên Cứu Hiện Tại (Khuyến Nghị)**:
  - *Phương pháp*: Tuyển chọn một tập con ngoại vi kiểm toán (**External Benchmark Inpainting Subset**) từ nguồn công khai sẵn có đã có ground-truth mask (ví dụ: các mẫu sạch từ subfolder chưa dùng của TGIF/TGIF2 sau khi chạy lọc loại trừ toàn bộ 684 COCO IDs Option P, hoặc SAGI-D sau khi hoàn thành xác minh bản quyền ảnh nguồn sơ cấp).
  - *Mục tiêu giữ nguyên*:
    * Giữ nguyên **RQ1** ở nhánh 2 lớp (`authentic` vs `ai_edited`).
    * Giữ nguyên **Auxiliary RQ5** (định vị vùng chỉnh sửa dựa trên ground-truth mask).
    * Giữ nguyên 5 outer-fold checkpoints đã niêm phong và giao thức đánh giá độc lập Phase 4C.7A.
  - *Ưu điểm*: Loại bỏ hoàn toàn công sức tự sinh ảnh inpainting; loại bỏ nguy cơ tạo ra các hiện vật vi mô (forensic shortcuts) do compositing nhân tạo; bảo toàn 100% mục tiêu nghiên cứu đã đăng ký.
- **Lựa chọn B: Cần Thay Đổi Giao Thức (Protocol Amendment & Scope Shift)**:
  - *Trường hợp 1 (Pivot sang GenImage)*:
    * Chuyển bài toán sang phân loại 3 lớp (`authentic`, `fully_generated`, `ai_edited`) hoặc 2 lớp `authentic` vs `fully_generated`.
    * *Thay đổi protocol bắt buộc*: Cần bổ sung dữ liệu `fully_generated`, mở khóa class-coverage guard (`EV-CLASS-COVERAGE-GUARD-001`), và huấn luyện/fine-tune lại classification head.
    * *Hệ quả*: Phải **hủy bỏ hoặc đình chỉ Auxiliary RQ5 (Localization)** vì GenImage là ảnh sinh toàn phần, không có mask cục bộ.
  - *Trường hợp 2 (Dùng inpainting dataset không có mask)*:
    * Phải ban hành Protocol Amendment sửa đổi loại bỏ Auxiliary RQ5 khỏi phạm vi đánh giá độc lập.

*Cam kết trung thực*: Dự án **CHƯA tải dữ liệu quy mô lớn (> 50 MB)** và **CHƯA chạy detector scoring / evaluation** trong phiên làm việc này.

---

## JJ. Pilot v2 run pilot-20261009T111247Z intake, audit, and empirical Content QC evaluation (2026-10-09)

- **Archive Intake & Safety Audit**:
  - ZIP package: `data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z_package.zip` (6,700,761 bytes, SHA-256 `a0521a16f3b5fd498aac5fc15ecc43def83f5bc64c7d907123a73e251ce1de0d`).
  - Pre-extraction safety check: 34 archive members inspected, 0 directory traversals (`..`), 0 leading slashes, 0 absolute paths, 0 symlinks, 0 filename collisions.
  - Archive structure: Flat top-level paths (`images/`, `masks/`, `run_binding.json`, `run_receipt.json`, `attempt_ledger.jsonl`, `provenance_ledger.jsonl`, `manifest_independent_cohort.csv`, `manifest_checksum.sha256`, `content_qc_contact_sheet.html`, `acquisition.log`). Safely extracted into dedicated directory `data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z/` without overwriting historical pilot, diagnostic, or calibration runs. All original ZIP, image, mask, receipt, and ledger files preserved intact.
- **Production CLI Audit & Binding Verification**:
  - Executed official CLI audit command at bound functional commit `79775250ee5393c9f193bdc0485f77998cefaddf`:
    `python scripts/research/run_cohort_acquisition.py --audit-run data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z --mode pilot --run-id pilot-20261009T111247Z --expected-commit 79775250ee5393c9f193bdc0485f77998cefaddf --edit-plan-path research/evidence/phase-4c.7b/content_grounded_pilot_plan_v2_proposal.json --catalog-extension research/evidence/phase-4c.7b/candidate_catalog_extension_v1.0.0.json` $\to$ **`PASS`**.
  - Verified run ID `pilot-20261009T111247Z`, source commit `79775250ee5393c9f193bdc0485f77998cefaddf`, approved plan SHA-256 `a95c96c594777025aac58049ca81978245035a54ad98c9298c0cf5a1ec0d8565`, catalog extension SHA-256 `0d875b81a8044e3285ecf0331beec6646d94b085915baaaa32247e86ee9023e7`, parent catalog SHA-256 `d85595c6b43d5acf8d312993a270278b4f17f481dca0f8286efdae07bcd281a5`.
  - Attempt accounting: exactly 8 attempts consumed across the 8 candidates in `attempt_ledger.jsonl`. All 8 attempts recorded status `ACCEPTED` (Technical QC PASS). Zero retries, zero automatic replacements (`automatic_replacement=false`).
  - Runtime telemetry: Python 3.13.15, Linux 6.6.122+, PyTorch 2.11.0+cu130, Diffusers 0.40.0, Transformers 5.18.0, CUDA 13.0, Tesla T4 (peak VRAM: SD2 3,249,408,000 bytes ~3.25 GB, SDXL 7,704,020,992 bytes ~7.70 GB; total elapsed time ~5.4 minutes).
- **Recalculated Pixel-Level Metrics from Extracted PNGs**:
  - Image geometry: 8 authentic PNGs ($512 \times 512$ RGB), 8 edited PNGs ($512 \times 512$ RGB), 8 mask PNGs ($512 \times 512$ L, values strictly $\{0, 255\}$).
  - Mask areas bit-exact: Slot 1: 9,350 px (3.57%), Slot 2: 79,360 px (30.27%), Slot 3: 15,600 px (5.95%), Slot 4: 82,500 px (31.47%), Slot 5: 66,654 px (25.43%), Slot 6: 33,440 px (12.76%), Slot 7: 17,250 px (6.58%), Slot 8: 100,050 px (38.17%).
  - Outside pixel invariance: `outside_mean_l1 == 0.000000`, `outside_max_delta == 0.0` on all 8 pairs (enforced strictly by 1-bit binary request mask compositing; does not prove diffusion model self-preservation of background).
  - Inside mask mean L1: Slot 1 = 37.23, Slot 2 = 44.09, Slot 3 = 14.03, Slot 4 = 45.80, Slot 5 = 54.34, Slot 6 = 43.99, Slot 7 = 11.08, Slot 8 = 61.74. (Reiterating: inside L1 does not correlate with semantic correctness; Slots 3 and 7 exhibit total semantic omission despite L1 deltas).
- **Empirical Evaluation Table across All 8 Allocation Slots**:

| Slot | Candidate ID & Stratum | Phân Loại & Diện Tích Mask | Đối Tượng Đăng Ký | Kết Quả Thực Nghiệm & Khiếm Khuyết Quan Sát | Khuyến Nghị Agent | Trạng Thái Human QC |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **1** | `IND_COCO_SD2_001`<br>`coco_sd2` | `object_replacement`<br>9,350 px (3.57%, Small) | Đồng hồ treo tường tròn bằng đồng thay chảo | Thất bại tạo hình đồng hồ; sinh ra mảng phản xạ kính/kim loại đa diện méo mó; viền mask chữ nhật cắt cụt mép các chảo đồng lân cận gây đứt gãy hình học thô bạo. | **REJECT** | **PENDING** |
| **2** | `IND_COCO_SD2_002`<br>`coco_sd2` | `object_replacement`<br>79,360 px (30.27%, Large) | Tủ bếp trên navy-blue và máy hút mùi | **Thành công tạo hình xuất sắc**. Tủ bếp màu xanh navy có chỉ dọc và máy hút mùi inox/trắng sắc nét; phối cảnh góc và tiếp giáp trần khớp tự nhiên; trụ nẹp gỗ đứng bên trái chuyển tiếp mềm mại. | **ACCEPT** | **PENDING** |
| **3** | `IND_COCO_SDXL_042`<br>`coco_sdxl` | `object_insertion`<br>15,600 px (5.95%, Small) | Cặp táp da đen đứng thẳng trên thảm đỏ | **Omission hoàn toàn 100%**. 0 cặp táp được sinh ra (max delta trong vùng target chỉ 37.0); vùng mask biến thành mảng thảm đỏ phẳng lệch tông, lộ rõ 4 cạnh viền chữ nhật và cắt cụt bóng chân người phía trên. | **REJECT** | **PENDING** |
| **4** | `COCO_EXT_SDXL_001`<br>`coco_sdxl` | `object_insertion`<br>82,500 px (31.47%, Large) | Bàn dã ngoại gỗ có ghế băng trên bờ cỏ/cát | **Dị dạng cấu trúc nặng nề**. Bàn dã ngoại bị vẹo, chân bàn phủ lớp màu xanh giống ống quần; sinh nan sắt/thang chìm dưới cát và chao đèn lơ lửng; bước nhảy biên ngang lộ rõ tại y=260 và dọc tại x=10; cát bị mờ nhạt. | **REJECT** | **PENDING** |
| **5** | `IND_COMMONS_SD2_001`<br>`commons_sd2` | `object_removal_and_infill`<br>66,654 px (25.43%, Medium) | Xóa mũi đất tiền cảnh mở ra biển và bờ xa | Đã xóa sạch mũi đất và rặng thông, nhưng tái sinh một sườn vách đá phấn trắng dựng đứng mới thay thế đường bờ vịnh/thị trấn xa ở góc phải; mặt nước nhẵn mờ dạng phơi sáng dài khác gợn sóng gốc. | **REJECT** | **PENDING** |
| **6** | `IND_COMMONS_SD2_040`<br>`commons_sd2` | `object_insertion`<br>33,440 px (12.76%, Medium) | Chậu hoa thùng gỗ mộc trên đường đá cuội | Tạo được hình khối thùng gỗ mộc vững chắc và bóng đổ, nhưng **bỏ sót hoa** (thùng rỗng); mép trên mask cắt vào chân cửa ra vào màu xanh của ngôi nhà cổ làm chân cửa bị nhòe vỡ, lệch trục nghiêm trọng. | **REJECT** | **PENDING** |
| **7** | `IND_COMMONS_SDXL_001`<br>`commons_sdxl` | `object_insertion`<br>17,250 px (6.58%, Small) | Chim nhỏ bay trên bầu trời mây mở | **Omission hoàn toàn 100%**. 0 con chim được sinh ra (tái hiện chính xác kết quả Arm A lịch sử); vùng trời inpaint có màu xanh tím/xám đậm hơn bầu trời authentic ($\Delta\text{RGB} \approx [-10, -11, -3]$), tạo khung chữ nhật lộ liễu. | **REJECT** | **PENDING** |
| **8** | `IND_COMMONS_SDXL_005`<br>`commons_sdxl` | `object_replacement`<br>100,050 px (38.17%, Large) | Cột đá cẩm thạch trắng thay tượng Triptych | Tạo hình đài tưởng niệm đồ sộ, nhưng **đỉnh ngọn tượng đồng cũ thò ra ngoài đỉnh mask** ($y < 40$); phía sau bên trái tự sinh cấu trúc rạp/mái hiên màu cam bị cắt cụt thẳng đứng tại x=135; chân bệ cột loang bóng trắng clashing với sỏi. | **REJECT** | **PENDING** |

- **Phân Tích Đối Chiếu 4 Candidate Giữ Lại vs 4 Candidate Thay Mới**:
  - _Bốn candidate giữ lại từ v1 (Slots 1, 2, 5, 7)_: Chỉ duy nhất Slot 2 (`IND_COCO_SD2_002`) đạt chất lượng cao; Slot 1 bị biến dạng kim loại và cắt chảo; Slot 5 tái sinh sai địa hình; Slot 7 tái hiện 100% lỗi omission và lệch tông nền trời của Arm A lịch sử.
  - _Bốn candidate thay mới trong v2 (Slots 3, 4, 6, 8)_: Đã khắc phục được các xung đột hình học cũ (loại bỏ lan can của `_003`, tránh vòm kính của `_020`, tránh xe đua của `_002`, tránh trần phẳng ép quota của `_041`), tuy nhiên kết quả thực nghiệm bộc lộ các giới hạn sinh thực tế mới:
    - Slot 3 chứng minh rằng việc chuyển sang mặt thảm phẳng đồng nhất không giải quyết được vấn đề omission của SDXL trên canvas 512x512.
    - Slot 4 chứng minh rằng mask large trống trên nền cát tự nhiên dẫn tới việc mô hình tự bịa đặt các cấu trúc kỳ dị (nan sắt chìm, chao đèn lơ lửng, chân bàn dạng ống quần).
    - Slot 6 chứng minh rằng ngay cả khi đơn giản hóa hình khối thành hình trụ đặc (thùng gỗ), việc mask chữ nhật chạm vào chân cửa kiến trúc cổ vẫn gây phá vỡ tính nguyên vẹn của công trình.
    - Slot 8 chứng minh rằng việc ước lượng bounding box hình học trên các khối tượng có đỉnh nhọn dễ bỏ sót các phần chóp nhô ra ngoài biên mask ($y < 40$).
- **Kết Luận Tính Khả Thi Dựa Trên Tiêu Chí Đăng Ký**:
  - Không tự đặt thêm ngưỡng mới sau khi thấy kết quả; không suy rộng quá mức từ tập mẫu $n=8$.
  - Thực nghiệm Pilot v2 xác nhận: **Chưa đủ căn cứ kỹ thuật để tiến hành thu thập quy mô lớn (Full Cohort $N=400$)** bằng phương pháp inpainting canvas toàn phần $512 \times 512$ hiện hành.
  - Tỷ lệ đáp ứng yêu cầu chất lượng nội dung thị giác (Content QC) ở mức rất thấp (chỉ 1/8 đạt chuẩn, 7/8 vi phạm các tiêu chí về omission, rách biên, hoặc biến dạng cấu trúc).
- **Contact Sheet Thẩm Định Độc Lập**:
  - Bản contact sheet gốc của Colab giữ nguyên vẹn tại: `data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z/content_qc_contact_sheet.html`.
  - Đã xuất bản contact sheet review tự chứa hoàn chỉnh: `data/research/local-artifacts/phase-4c.7b/pilot-20261009T111247Z/pilot_v2_content_qc_review_contact_sheet.html` (15,298,560 bytes, 100% Base64 embedded 32 panel hình ảnh authentic, overlay, composite, zoom 2x, pure UTF-8).
- **Kỷ Luật Quản Trị & Trạng Thái Quyết Định**:
  - Toàn bộ 8 cặp ảnh mới giữ nguyên trạng thái **`PENDING_CONTENT_QC`** cho đến khi có quyết định chính thức từ người dùng.
  - Tuyệt đối không tự ý thêm lượt chạy generation/retry, không gọi detector, không mở khóa full cohort.
  - Bảo toàn toàn bộ các quyết định lịch sử: Diagnostic 6/6 REJECT, Calibration 2 PENDING, Historical Pilot v1 8 PENDING.

---

## FF. Calibration run calib-20261009T015749Z intake, audit, and empirical evaluation (2026-10-09)

- **Archive Intake & Safety Audit**:
  - ZIP package: `data/research/local-artifacts/phase-4c.7b/calib-20261009T015749Z_package.zip` (3,628,851 bytes, SHA-256 `a547c47144f09bf875045e7a5e2673cd8b257742ab48ded17042c859da3aae90`).
  - Pre-extraction safety check: 14 archive members inspected, 0 directory traversals (`..`), 0 leading slashes, 0 absolute paths, 0 symlinks, 0 filename collisions.
  - Archive structure: Flat top-level paths (`images/`, `masks/`, `run_binding.json`, `calibration_receipt.json`, `attempt_ledger.jsonl`, `provenance_ledger.jsonl`, `calibration_contact_sheet.html`) without run directory prefix. Safely extracted into dedicated directory `data/research/local-artifacts/phase-4c.7b/calib-20261009T015749Z/` without overwriting historical pilot or diagnostic runs. All original ZIP, image, mask, receipt, and ledger files preserved intact.
- **Production CLI Audit & Binding Verification**:
  - Executed official CLI audit command at bound commit `045ea70cb9067ede3833f7a01562199869f4ae56`:
    `python scripts/research/run_cohort_acquisition.py --mode calibration --run-id calib-20261009T015749Z --expected-commit 045ea70cb9067ede3833f7a01562199869f4ae56 --audit-run data/research/local-artifacts/phase-4c.7b/calib-20261009T015749Z` $\to$ **`PASS`**.
  - Verified run ID `calib-20261009T015749Z`, source commit `045ea70cb9067ede3833f7a01562199869f4ae56`, approved plan SHA-256 `2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c`.
  - Attempt accounting: exactly 2 attempts on candidate `IND_COCO_SDXL_002` (`STARTED` and `ACCEPTED` in `attempt_ledger.jsonl` represent two lifecycle events within each attempt, not 4 attempts).
  - Candidate configuration: seed `20272319`, canvas 512×512, no crop, no feathering, EulerDiscreteScheduler, 30 steps, strength 1.0, guidance scale 7.5 (attempt 1) and 9.5 (attempt 2).
  - Telemetry: Python 3.13.15, Linux 6.6.122+, PyTorch 2.11.0+cu130, Diffusers 0.40.0, Transformers 5.18.0, CUDA 13.0, Tesla T4 (NumPy 2.1.3, Pillow 11.3.0).
- **Chuỗi Nguồn gốc Đầu vào (Input Provenance Chain) & Tính toàn vẹn**:
  - _Tệp niêm phong đầu vào trước chạy_: Kế hoạch `content_grounded_calibration_proposal.json` ràng buộc mã băm từ `pilot-20261008T113700Z`:
    - Authentic: `IND_COCO_SDXL_002_auth.png` (280,961 bytes, SHA-256 `7aefc1d1dff39ac5527ce47734421e34094af42d9763fe35ab3b94fee62e4571`).
    - Mask: `IND_COCO_SDXL_002_mask.png` (449 bytes, SHA-256 `2ff6b16571048015060095e9a240b28ed1073e9ea70a9d7870bfaf3b49fb9bee`).
    - Runner code (`verify_calibration_inputs`) kiểm tra fail-closed hai mã băm này trước khi thực thi. Tuy nhiên, `calibration_receipt.json` không ghi trường mã băm của tệp đầu vào trong JSON receipt (đây là một thiếu hụt ghi nhận receipt).
  - _Tệp đóng gói sau chạy trong archive_: Quá trình thực thi trên Colab mở ảnh qua Pillow và lưu lại bản sao (`auth_img.save(auth_dest)`). Quá trình re-encode bằng libpng mặc định làm thay đổi kích thước byte và mã băm tệp trên đĩa:
    - `CALIB_COCO_SDXL_002_PROMPT_G75_auth.png`: 288,206 bytes, SHA-256 `a35bef734b09b832c75aa0335f0de25adba54f37fff7ea4c4741e21517c75f46`.
    - `CALIB_COCO_SDXL_002_PROMPT_G75_mask.png`: 523 bytes, SHA-256 `e4721316af48fb49ca166a8b02541b4713e92abbafc7a16e84c752750b76f25a`.
  - _Phân biệt rõ ràng_: Hai tệp có SHA-256 khác nhau tuyệt đối không được gọi là byte-identical. Tuy nhiên, khi giải mã qua PIL thành mảng NumPy, mảng pixel của tệp trong archive khớp chính xác bit-to-bit với ảnh niêm phong (`max_auth_diff == 0`, `max_mask_diff == 0`). Mức độ trùng khớp pixel bảo đảm giá trị số không suy hao, nhưng không thay thế được việc lưu vết mã băm tệp trong receipt trước chạy.
- **Recalculated Pixel-Level Metrics & Formula Disclosures**:
  - _Kiểu dữ liệu & ROI_: Mảng ảnh kiểu `float32`, RGB 512×512; mask `uint8` tại `[345, 245, 455, 335]` (9,900 px, 3.776550%).
  - _Công thức độ lệch tuyệt đối_: `delta_abs = np.abs(edited.astype(np.float32) - authentic.astype(np.float32))`.
  - _Outside Invariance (Ghép nhị phân 1-bit)_: `outside_mean_l1 == 0.000000`, `outside_max_delta == 0.0` trên cả 2 attempts.
  - _Raw Diffusion Drift trước compositing_:
    - G7.5: mean L1 = `4.0973`, std của delta_abs = `3.4032`, max delta = `41.0`.
    - G9.5: mean L1 = `4.1187`, std của delta_abs = `3.4253`, max delta = `38.0`.
  - _Inside Mask Metrics (Đính chính định danh std)_:
    - Attempt 1 (Guidance 7.5): inside mean L1 = **`16.4117`**; std của delta_abs = **`13.6940`** (ddof=0; ddof=1 là `13.6943`); max delta = `89.0`. _(Đính chính: Con số 33.60 trong bản nháp trước là std của giá trị màu edited `np.std(g75[mask == 255]) = 33.6041`, không phải std của delta_abs)._
    - Attempt 2 (Guidance 9.5): inside mean L1 = **`16.9891`**; std của delta_abs = **`13.9863`** (ddof=0; ddof=1 là `13.9865`); max delta = `87.0`. _(Đính chính: Con số 33.02 trong bản nháp trước là std của giá trị màu edited `np.std(g95[mask == 255]) = 33.0174`)._
    - Delta nội vùng G9.5 vs G7.5: mean L1 = `5.6830`, std = `4.7690`, max delta = `41.0`.
  - _Số đo Bậc biên (Boundary Steps) tại `[345, 245, 455, 335]`_:
    - Cạnh trên (Top edge, $y=245$ vs $244$, $x \in [345, 455)$): Công thức `np.mean(np.abs(img[245, 345:455, :] - img[244, 345:455, :]))`. Authentic: **`1.8121`**; G7.5: **`8.1667`**; G9.5: **`9.2697`**.
    - Cạnh trái (Left edge, $x=345$ vs $344$, $y \in [245, 335)$): Công thức `np.mean(np.abs(img[245:335, 345, :] - img[245:335, 344, :]))`. Authentic: **`4.1185`**; G7.5: **`9.5593`**; G9.5: **`9.2852`**. _(Đính chính: Con số 3.24 trước đó là do sai lệch ROI đo; giá trị authentic thực tế trên ROI chuẩn tắc là **4.1185**)._
    - Cạnh phải (Right edge, $x=454$ vs $455$, $y \in [245, 335)$): Authentic: `3.3667`; G7.5: `8.3481`; G9.5: `10.2667`.
    - Cạnh dưới (Bottom edge, $y=334$ vs $335$, $x \in [345, 455)$): Authentic: `2.8667`; G7.5: `10.2758`; G9.5: `11.8667`.
- **Visual Analysis across Separated Criteria**:
  - _Tomato Presence_: Total semantic omission in both attempts. 0 cherry tomato synthesized. Both attempts generated infilled bread crumb texture.
  - _Position / Scale_: Target bbox `[375, 265, 430, 320]` contains no target object in either attempt.
  - _Realism / Lighting_: Inpainted region shows rough bread crumb texture, without the target object.
  - _Crumb Texture_: Guidance scale increase from 7.5 to 9.5 shifted crumb grain pattern slightly (internal L1 delta 5.68), but did not trigger object formation.
  - _Boundary Quality_: Hard binary compositing creates noticeable textural transition steps along registered mask boundary `[345, 245, 455, 335]`.
  - _Methodological & Scientific Insight_: "L1 tăng không bảo đảm thành công ngữ nghĩa" — inside L1 increased from 16.41 to 16.99, yet both attempts are complete omissions. Outside L1 = 0.000000 is an artifact of 1-bit compositing, not native diffusion behavior (~4.1 L1 drift).
- **Contact Sheet & Governance Dossier**:
  - Original Colab contact sheet backed up to `calibration_contact_sheet_raw_colab.html`.
  - Derived coordinate overlays and zoom panels rendered into `derived_overlays/`.
  - Enriched self-contained HTML contact sheet: `data/research/local-artifacts/phase-4c.7b/calib-20261009T015749Z/calibration_contact_sheet.html` (3,445,798 bytes, 15 Base64 embedded PNGs, authentic, raw, composite, overlay, and zoom panels).
  - 15/15 Base64 embeddings verified byte-for-byte against disk PNGs; table metrics match calculated values.
  - Governance separation: Agent recommends REJECT for both calibration attempts; formal Human Content QC status remains strictly **PENDING** (2 decisions awaiting human reviewer).
  - Historical diagnostic determinations (6/6 REJECT) and 8 historical pilot pairs (PENDING) preserved intact. Full cohort remains locked; detector calls = 0; independent performance `NOT_MEASURED`.
- **Scientific Bounds & Limitations**:
  - Phép thử chỉ so sánh guidance scale 7.5 với 9.5 trong cấu hình văn bản mới cố định, trên một candidate (`IND_COCO_SDXL_002`) và một seed (`20272319`).
  - So sánh lịch sử với pilot cũ chỉ mang tính tham khảo vì prompt và negative prompt đã thay đổi đồng thời.
  - Không kết luận cơ chế omission, hiệu quả tổng quát của guidance scale, hoặc lựa chọn pipeline production từ hai attempts.

## GG. Tổng hợp Tính khả thi Thực nghiệm Phase 4C.7B và Phân loại Khiếm khuyết Tạo sinh (2026-10-09)

Hồ sơ tổng hợp toàn diện các đợt chạy thực nghiệm độc lập trong Phase 4C.7B nhằm phục vụ đánh giá tính khả thi trước khi xem xét mở khóa cohort chính thức. Toàn bộ nhận định được phân tách nghiêm ngặt giữa quan sát thực nghiệm, lỗi phần mềm đã chứng minh, và các giả thuyết chưa kiểm chứng (liên kết với `docs/EVIDENCE_REGISTER.md` và `docs/references.bib`).

### 1. Bảng Tổng hợp Độc lập 4 Đợt Chạy Thực nghiệm

_Nguyên tắc kế toán khoa học: Tuyệt đối không gộp các lần chạy khác giao thức thành một tỷ lệ thành công chung; không tính trùng các sự kiện STARTED/terminal trong cùng một attempt; không tính các candidate tái dùng thành mẫu độc lập mới._

| Đợt chạy Thực nghiệm                            | Giao thức / Mục tiêu                                                                                                           | Quy mô & Ngân sách                                                                  | Technical QC (std / inside L1 / outside L1)                                                                                                                                          | Human Content QC                                                                                       | Lỗi / Khiếm khuyết Đã Chứng minh                                                                                                                                                                                                                                                                       |
| :---------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pilot lịch sử**<br>`pilot-20261007T132003Z`   | Pilot 8 cặp đầu tiên trên 4 strata; kiểm tra vận hành hệ thống thu thập tự động.                                               | 8 cặp kế hoạch<br>(2 cặp / stratum).<br>Ngân sách: 8 attempts.                      | **8/8 PASS (lịch sử)**<br>Mã tại binding commit `7d2eea4` bỏ qua kiểm tra `outside_l1 <= 0.5`.<br>Kiểm toán lại: outside L1 đo được **3.808–9.311** ở cả 8 cặp do thiếu compositing. | **8 PENDING**<br>_(Agent screening phát hiện 8/8 có vấn đề nội dung; chưa có quyết định người duyệt)_. | 1. Thiếu kiểm tra và bảo tồn pixel ngoài mask (outside L1 vượt ngưỡng 0.5 ở cả 8 cặp).<br>2. Thiếu đăng ký mục tiêu cụ thể (dùng prompt chung theo index và mask ellipse ngẫu nhiên trên canvas 512×512).                                                                                              |
| **Follow-up pilot**<br>`pilot-20261008T113700Z` | Thử nghiệm 8 cặp sau sửa đổi hướng dẫn nội dung (content-grounded instructions).                                               | 8 cặp kế hoạch<br>(2 cặp / stratum).<br>Ngân sách: 8 attempts.                      | **8/8 PASS**<br>(inside L1 11.08–76.14; outside L1 = 0.000000 nhờ 1-bit compositing).                                                                                                | **8 PENDING**<br>_(Chưa có quyết định người duyệt)_.                                                   | 1. Omission trên các mask: `IND_COCO_SDXL_002` (3.78%), `IND_COMMONS_SDXL_001` (6.58%), `IND_COMMONS_SD2_002` (14.76%).<br>2. Đứt gãy cấu trúc (lan can kim loại bị cắt cụt 210 px ở `IND_COMMONS_SDXL_003`).<br>3. Lệch tone mảng lớn (trần nhà 30% area ở `IND_COCO_SDXL_041`).                      |
| **Diagnostic**<br>`diag-20261008T154628Z`       | So sánh đối đầu Full Canvas 512×512 (Arm A) vs Local Crop có padding ở độ phân giải gốc (Arm B) trên 3 ca omission.            | 3 candidates $\times$ 2 arms = đúng 6 attempts.<br>Ngân sách: 6 attempts.           | **6/6 PASS**<br>(outside L1 = 0.000000 qua compositing).                                                                                                                             | **6/6 REJECT**<br>_(Đã chốt chính thức bởi Dũng Phạm lúc 2026-10-08T19:34:30Z)_.                       | 1. Arm A: 3/3 ca omission hoàn toàn (`IND_COCO_SDXL_002`, `IND_COMMONS_SD2_002`, `IND_COMMONS_SDXL_001`).<br>2. Arm B cà chua: bậc biên mask & vụn bánh mì.<br>3. Arm B vali: semantic hallucination (xe hơi đồ chơi).<br>4. Arm B chim: lệch placement ngoài target bbox 41 px & lệch tone mảng trời. |
| **Calibration**<br>`calib-20261009T015749Z`     | Can thiệp đơn biến: so sánh Guidance Scale 7.5 vs 9.5 với prompt/negative prompt mới trên `IND_COCO_SDXL_002` (seed 20272319). | 1 candidate $\times$ 2 guidance scales = đúng 2 attempts.<br>Ngân sách: 2 attempts. | **2/2 PASS**<br>(inside L1 16.4117 / 16.9891; outside L1 = 0.000000).                                                                                                                | **2 PENDING**<br>_(Agent đề xuất REJECT; chờ người duyệt)_.                                            | 1. Cả 2 attempt đều omission hoàn toàn (0 quả cà chua; tái tạo vân ruột bánh mì).<br>2. Bậc nhảy tương phản vi mô tại biên mask (top edge step 8.17 / 9.27 vs 1.81 authentic; left edge step 9.56 / 9.29 vs 4.12 authentic).                                                                           |

### 2. Phân loại 5 Nhóm Khiếm khuyết Chất lượng Tạo sinh (Defect Taxonomy)

1. **Hiện tượng Thiếu vật thể hoàn toàn (Semantic Omission)**:
   - _Biểu hiện_: Vùng mask không xuất hiện đối tượng được yêu cầu trong prompt mà bị lấp đầy bởi hoa văn nền xung quanh (infill).
   - _Bằng chứng thực nghiệm_: Quan sát thấy trên cả SDXL và SD2 full canvas 512×512 qua 3 ca thử nghiệm:
     - `IND_COCO_SDXL_002` (cà chua): 9.900 px, **3.776550%** canvas.
     - `IND_COMMONS_SDXL_001` (chim): 17.250 px, **6.580353%** canvas.
     - `IND_COMMONS_SD2_002` (vali): 38.700 px, **14.762878%** canvas.
       _Lưu ý khoa học_: Omission xuất hiện ở các mức diện tích từ 3.78% đến 14.76%; không kết luận rằng omission chỉ giới hạn dưới 5% và không tự đặt ngưỡng loại trừ mask < 5%.
   - Đợt calibration trên `IND_COCO_SDXL_002` (seed 20272319) chứng minh rằng việc tăng guidance scale (7.5 $\to$ 9.5) kèm negative prompt cụ thể (`bread crumb only`, `background infill`) không kích hoạt hình thành quả cà chua trong phạm vi cấu hình đã thử.
2. **Sai lệch Đối tượng Ngữ nghĩa (Semantic Hallucination)**:
   - _Biểu hiện_: Mô hình tạo ra một vật thể hoàn chỉnh nhưng hoàn toàn sai lệch so với prompt văn bản.
   - _Bằng chứng thực nghiệm_: Quan sát thấy ở SD2 inpainting khi áp dụng local crop 1.6x (`IND_COMMONS_SD2_002` Arm B trong `diag-20261008T154628Z`): mô hình sinh ra một chiếc xe hơi đồ chơi cổ thay vì chiếc vali hành lý, dù inside L1 tăng cao (43.39 vs 30.74).
3. **Sai lệch Vị trí và Tỷ lệ (Placement & Scale Deficit)**:
   - _Biểu hiện_: Mô hình sinh được đối tượng mục tiêu nhưng đặt sai vị trí hình học so với tọa độ kỳ vọng.
   - _Bằng chứng thực nghiệm_: Quan sát thấy ở SDXL native resolution local crop (`IND_COMMONS_SDXL_001` Arm B trong diagnostic): bóng chim xuất hiện nhưng bị dịch chuyển xuống dưới target bounding box $dy = +41.0$ px ($dx = -30.5$ px, độ trùng khớp theo chiều dọc = 0 px), nằm ngoài target box `[395, 75, 455, 125]` dù vẫn nằm 100% trong mask.
4. **Lệch Ánh sáng, Tông màu & Kết cấu (Lighting, Texture & Tone Mismatch)**:
   - _Biểu hiện_: Vùng can thiệp có mức độ phơi sáng, tông màu hoặc kết cấu không đồng nhất với ảnh authentic xung quanh.
   - _Bằng chứng thực nghiệm_: Mảng trời xung quanh chim Arm B bị lệch tone chữ nhật ($\Delta\text{RGB} \approx [-3.07, -3.79, -3.66]$), tạo thành một mảng chữ nhật xám mờ rõ rệt. Vùng vụn bánh mì infilled trong mask cà chua có độ tương phản và mật độ hạt mịn khác biệt với phần bánh mì authentic bên ngoài.
5. **Biên ghép Vi mô và Đứt gãy Cấu trúc (Boundary Seams & Structural Severance)**:
   - _Biểu hiện_: Bậc tương phản vi mô tại đường ranh giới mask và sự đứt đoạn vật lý của các thực thể hình học kéo dài.
   - _Bằng chứng thực nghiệm_:
     - Hard binary compositing (1-bit) luôn tạo ra bước nhảy tương phản vi mô tại biên (mép trên mask cà chua có step 8.17–9.27 so với authentic 1.81; mép trái có step 9.56–9.29 so với authentic 4.12).
     - Khi mask cắt ngang các cấu trúc liên tục (lan can kim loại ở `IND_COMMONS_SDXL_003`), phép ghép nhị phân cắt đứt cấu trúc vật lý 210 px.
     - Thử nghiệm feathering cosine ($k=2$ px) chứng minh chỉ làm mịn bậc chuyển tiếp 1-2 pixel nhưng không thể khắc phục sự đứt gãy hình học vĩ mô hay tonal mismatch diện tích lớn.

### 3. Phân tách Nghiêm ngặt: Thực nghiệm, Lỗi phần mềm và Giả thuyết

- **Quan sát Thực nghiệm (Empirical Observations - Đã đo đạc xác minh)**:
  - Tỷ lệ omission thực tế trên canvas 512×512 đối với 3 ca thử nghiệm (cà chua 3.78%, chim 6.58%, vali 14.76%).
  - Tọa độ bóng chim bị lệch khỏi target box ($dy = +41.0\text{ px}$).
  - Xe hơi đồ chơi thay thế vali dưới crop 1.6x.
  - Bước nhảy L1 tại biên mask và sự trôi lệch raw diffusion $\approx 4.1$ L1.
  - `outside_mean_l1 == 0.000000` hoàn toàn do lớp ghép 1-bit bảo đảm theo định nghĩa, không chứng minh mô hình diffusion tự bảo tồn nền.
- **Lỗi Phần mềm Đã Chứng minh (Proven Software Defects - Đã phân định)**:
  - Sự cố lịch sử trước pilot: 401 upstream checkpoint `sd2-inpainting` và lỗi thiếu tham số `height/width` trong pipeline SDXL (đã khắc phục bằng mirror cộng đồng và contract cứng 512×512).
  - Lỗi thiếu kiểm tra `outside_l1` trong runner cũ của `pilot-20261007T132003Z` (đã khắc phục từ `pilot-20261008T113700Z`).
  - Lỗ hổng quan sát trong receipt: `calibration_receipt.json` chưa lưu SHA-256 của file đầu vào tại thời điểm preflight.
- **Hồ sơ Lịch sử vs Catalog Chuẩn tắc Hiện hành**:
  - Các tệp `candidate_acquisition_plan.json` và `acquisition_smoke_receipt.json` (ngày 06/10) chứa `unsplash_*` và biên nhận `synthetic_smoke` là **hồ sơ lịch sử**, không đủ điều kiện nhập cohort.
  - Catalog chuẩn tắc hiện hành chứa `commons_*` là:
    - Catalog: `research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json` (SHA-256 `d85595c6b43d5acf8d312993a270278b4f17f481dca0f8286efdae07bcd281a5`).
    - Kế hoạch: `research/evidence/phase-4c.7b/candidate_acquisition_plan_v2.json` (SHA-256 `7c4190fbaae9605ad2ff462dd4d128f2707109f88bb9fe4dc272fab1963caa5c`).
    - Kiểm toán: `research/evidence/phase-4c.7b/catalog_eligibility_audit.json` (SHA-256 `1cc6b9b576a99ab21e72125b56424718753819408ba224c5bfff8ec5b2a8c3d4`).
- **Giả thuyết Chưa Kiểm chứng (Unverified Hypotheses - Giữ đúng trạng thái giả định)**:
  - _Giả thuyết Latent Downsampling Capacity_: Nhận định cho rằng diện tích latent nhỏ thiếu dung lượng biểu diễn chỉ là suy luận từ kiến trúc $8\times$ downsampling; chưa được cô lập thực nghiệm bằng phân tích tensor nội bộ.
  - _Giả thuyết Context Infill Attention Bias_: Giả định rằng embedding bối cảnh xung quanh lấn át prompt văn bản; chưa được đo đạc qua cross-attention maps.
  - _Nguy cơ Forensic Shortcut_: Nhận định cho rằng biên ghép 1-bit hoặc feathering nhân tạo có thể tạo shortcut cho mô hình detector là nguy cơ phương pháp luận cần kiểm soát; không khẳng định hiệu năng detector đã bị thổi phồng khi chưa đo thực tế (`NOT_MEASURED`).

---

## HH. Đề xuất Định hướng Tiếp theo & Bốn Phương án Kỹ thuật (2026-10-09)

Dựa trên bằng chứng tích lũy qua 4 đợt chạy (8 pilot attempts, 6 diagnostic attempts, 2 calibration attempts), việc tiếp tục thu thập full cohort $N=400$ theo quy trình hiện tại là **chưa khả thi về mặt chất lượng nội dung**, do các vấn đề về omission, sai placement, sai đối tượng và biên ghép đứt gãy vẫn tồn tại trên các candidate đã thử.

### 1. Phân tích Bốn Phương án Định hướng Kỹ thuật

#### Phương án 1: Tiếp tục vi điều chỉnh siêu tham số và prompt trên từng candidate (Micro-Tuning)

- _Vấn đề giải quyết_: Tìm kiếm tổ hợp prompt, negative prompt, seed và guidance scale trên canvas 512×512 cho từng candidate cụ thể.
- _Bằng chứng hỗ trợ & Phần chưa biết_: Đợt calibration `calib-20261009T015749Z` chứng minh việc tăng guidance scale 7.5 $\to$ 9.5 kèm negative prompt không giải quyết được omission trên `IND_COCO_SDXL_002` (seed 20272319). Trong phạm vi candidate và seed đã thử, việc thay đổi guidance không kích hoạt tạo vật thể.
- _Thay đổi phương pháp_: Không đổi allocation; giữ nguyên pipeline.
- _Điều kiện & Kế hoạch_: Yêu cầu một kế hoạch thử nghiệm mới được duyệt.
- _Đánh giá_: **Không khuyến nghị**. Dễ dẫn đến việc thử mò mẫm từng candidate vô hạn, khó mở rộng cho toàn bộ cohort.

#### Phương án 2: Chuyển đổi sang Pipeline Local-Crop có điều kiện ở độ phân giải gốc (Native-Resolution Cropped Pipeline)

- _Vấn đề giải quyết_: Khắc phục hiện tượng omission trên candidate nhỏ (đã chứng minh ở Arm B diagnostic: 2/2 ca SDXL Arm B tạo được quả cà chua và chim).
- _Bằng chứng hỗ trợ & Phần chưa biết_: Arm B tạo được vật thể nhưng làm phát sinh các lỗi mới: chim lệch vị trí $dy = +41\text{ px}$, mảng trời xám lệch tông, và SD2 hallucination xe đồ chơi. Chưa có cơ chế giải quyết placement và độ khớp tông màu mà không tạo vết biên nhân tạo.
- _Thay đổi phương pháp_: Yêu cầu thay đổi kiến trúc pipeline (crop, coordinate mapping, upscaling/downscaling, blending) và sửa đổi quy chuẩn thu thập.
- _Điều kiện & Kế hoạch_: Yêu cầu một Protocol Amendment mới, prototype kiểm thử và phê duyệt từ người dùng.
- _Đánh giá_: Tiềm năng về việc kích hoạt tạo vật thể, nhưng có độ phức tạp cao và chưa giải quyết được các khiếm khuyết nội dung phát sinh.

#### Phương án 3 (Khuyến nghị Cốt lõi): Rà soát Tiêu chuẩn Chọn mẫu và Đề tài Can thiệp (Content-Grounded Candidate & Mask Curation)

- _Vấn đề giải quyết_: Nhắm đến việc hạn chế các ca can thiệp có rủi ro nội dung cao ngay từ khâu tiền kiểm: tránh các vùng can thiệp cắt ngang cấu trúc hình học liên tục (như lan can, chân tường) và các vùng bối cảnh có nguy cơ xung đột ngữ nghĩa cao.
- _Bằng chứng hỗ trợ & Phần chưa biết_:
  - Bằng chứng thực nghiệm cho thấy việc cắt ngang lan can kim loại liên tục (`IND_COMMONS_SDXL_003`) gây đứt gãy không thể khắc phục bằng inpainting hay feathering.
  - Chưa biết: Việc thay đổi tiêu chí chọn mẫu có loại bỏ hoàn toàn omission trên canvas 512x512 hay không (cần kiểm chứng trên các candidate mới có nền phẳng/trơn). Không tự cam kết rằng mask lớn hơn sẽ bảo đảm sinh đúng vật thể.
- _Thay đổi phương pháp & Cohort_:
  - Giữ nguyên cơ cấu 4 strata (`coco_sd2`, `coco_sdxl`, `commons_sd2`, `commons_sdxl`), 8 allocation slots (3 replacement, 1 removal, 4 insertion | 3 small, 2 medium, 3 large).
  - Tiền kiểm nội dung từng candidate trước khi đưa vào kế hoạch chạy: ưu tiên vật thể tách biệt, có ranh giới tự nhiên; không nới mask chỉ để đủ quota nếu làm phá vỡ cấu trúc cảnh.
- _Điều kiện cần đạt trước khi cân nhắc Full Cohort_:
  - Người dùng phê duyệt định hướng curation và các quyết định trade-off cụ thể.
  - Soạn thảo Kế hoạch Thử nghiệm Curation mới được phê duyệt chính thức.
  - Đạt kết quả thẩm định con người khả quan trên gate pilot mới trước khi mở khóa cohort.
- _Đánh giá_: **Khuyến nghị**. Phương án này tập trung giải quyết các lỗi thiết kế kịch bản can thiệp, tránh lãng phí GPU trên các ca có xung đột hình học cố hữu.

#### Phương án 4: Tạm dừng thu thập cohort tạo sinh inpainting, chuyển trọng tâm sang Benchmark Ngoại vi Độc lập (External Benchmark Pivot)

- _Vấn đề giải quyết_: Tránh rủi ro về chất lượng tạo sinh và thời gian chuẩn bị dữ liệu inpainting nội bộ.
- _Đánh giá_: Lựa chọn dự phòng nếu người dùng muốn tập trung toàn bộ nguồn lực vào việc đánh giá mô hình trên các bộ dữ liệu công khai sẵn có (GenImage, TGIF, v.v.).

---

## II. Dossier Rà soát Candidate & Mask Tiền kiểm trước Generation (Pre-Generation Screening Dossier)

Hồ sơ tiền kiểm nội dung toàn diện cho 8 dòng phân bổ (allocation slots) của kế hoạch tạo sinh, tuân thủ nghiêm ngặt các nguyên tắc khoa học và chuẩn tắc dữ liệu:

- Sử dụng catalog chuẩn tắc hiện hành (`verified_candidate_catalog_v2.json`) và ảnh authentic thực tế chuẩn hóa $512 \times 512$ đã được xem xét trực tiếp.
- Phân định rõ ràng: Bbox target là hộp chữ nhật bao vùng đặt dự kiến, không đại diện cho diện tích phân đoạn (segmentation mask) thực tế của vật thể.
- Phân biệt rạch ròi giữa việc đạt quota diện tích ($\ge 30\%$) và việc bao trọn target đăng ký (target containment).
- Phân tách rạch ròi giữa kết quả thực nghiệm đã tạo sinh (quan sát lịch sử) và đánh giá tiền kiểm hình ảnh authentic (chưa sinh).
- Tuyệt đối không chọn candidate dựa trên bộ dò (detector calls = 0).
- Giữ nguyên định mức allocation đã khóa: 4 strata $\times$ 2 dòng = 8 dòng (3 replacement, 1 removal, 4 insertion | 3 small, 2 medium, 3 large).
- Ưu tiên đối tượng tách biệt và ranh giới tự nhiên; không mô tả vùng trời trống hoặc mảng màu liên tục là ranh giới tự nhiên nếu mask vẫn cắt ngang qua một bề mặt đồng nhất. Không nới mask chỉ để đủ quota nếu làm phá vỡ cấu trúc cảnh. Không khẳng định mask lớn hơn sẽ bảo đảm sinh đúng vật thể.
- Tất cả các dòng đề xuất mới giữ nguyên trạng thái **`PENDING_HUMAN_REVIEW`** (`reviewer: null`, `timestamp: null`). Không tự cấp ngân sách generation trong phiên này.
- **Tài liệu trực quan tự chứa**: Toàn bộ ảnh authentic, overlay target (đỏ) vs mask (vàng), zoom vùng tiếp giáp và phương án thay thế/thẩm tra hình học được tích hợp trong tệp HTML tự chứa: [`candidate_mask_screening_contact_sheet.html`](../../../data/research/local-artifacts/phase-4c.7b/candidate_mask_screening_contact_sheet.html) (15,057,350 bytes, nhúng Base64 hoàn chỉnh, mã hóa UTF-8 chuẩn).

### Bảng 1: Đối Chiếu Bản Quyền Chuẩn Tắc (Canonical License Evidence Table)

Tuyệt đối không suy diễn bản quyền; toàn bộ thông tin giấy phép được đối chiếu trực tiếp giữa Catalog (`verified_candidate_catalog_v2.json`), Báo cáo nghiên cứu, HTML Contact Sheet và metadata từ nguồn gốc chính thức (Flickr API / Wikimedia Commons ExtMetadata):

| Candidate ID           | Nguồn gốc & Origin ID | Tác giả Ghi công | Giấy phép Nguồn Chuẩn xác | Trạng thái Đối chiếu | Bằng chứng Xác minh (Verified Canonical Source)                                                                      |
| :--------------------- | :-------------------- | :--------------- | :------------------------ | :------------------: | :------------------------------------------------------------------------------------------------------------------- |
| `IND_COCO_SD2_001`     | `coco:397133`         | Pot Noodle       | `CC BY 2.0`               |       **KHỚP**       | Flickr photo 6255196340; license id 4 (`Attribution License`).                                                       |
| `IND_COCO_SD2_002`     | `coco:37777`          | larrylawfer      | `CC BY-NC-SA 2.0`         |       **KHỚP**       | Flickr photo 7839199426; license id 2 (`Attribution-NonCommercial-ShareAlike`).                                      |
| `IND_COCO_SDXL_002`    | `coco:293044`         | john-norris      | `CC BY-SA 2.0`            |       **KHỚP**       | Flickr photo 2544265538; license id 5 (`Attribution-ShareAlike License`).                                            |
| `IND_COCO_SDXL_041`    | `coco:189310`         | an iconoclast    | `CC BY 2.0`               |       **KHỚP**       | Flickr photo 7375894572; license id 4 (`Attribution License`).                                                       |
| `IND_COMMONS_SD2_001`  | `commons:92533678`    | Moahim           | `CC BY-SA 4.0`            |       **KHỚP**       | Wikimedia Commons File:"La Ciotat" - panoramio.jpg.                                                                  |
| `IND_COMMONS_SD2_002`  | `commons:81567907`    | Mr.choppers      | `CC BY-SA 3.0`            |       **KHỚP**       | Wikimedia Commons File:"1932" Pur Sang Bugatti Type 55.                                                              |
| `IND_COMMONS_SDXL_001` | `commons:166503140`   | Crisco 1492      | `CC BY-SA 4.0`            |       **KHỚP**       | Wikimedia Commons File:"Tower of Freedom" by Ed Dwight.                                                              |
| `IND_COMMONS_SDXL_003` | `commons:166529058`   | Crisco 1492      | `CC BY-SA 4.0`            |       **KHỚP**       | Wikimedia Commons File:"Tower Song" by Ted Bieler.                                                                   |
| `IND_COCO_SDXL_042`    | `coco:448076`         | luis.leao        | `CC BY 2.0`               |       **KHỚP**       | Flickr photo 2260856815; license id 4 (`Attribution License`).                                                       |
| `IND_COCO_SDXL_040`    | `coco:578489`         | laura47          | `CC BY-NC 2.0`            |    **ĐÍNH CHÍNH**    | Catalog ghi đúng `Attribution-NonCommercial License` (`CC BY-NC 2.0`), đính chính nhầm lẫn bản nháp ghi `CC BY 2.0`. |
| `IND_COMMONS_SD2_040`  | `commons:172876577`   | Chainwit.        | `CC BY 4.0`               |    **ĐÍNH CHÍNH**    | Wikimedia Commons API xác nhận `CC BY 4.0` (không có SA), đính chính nhầm lẫn bản nháp ghi `CC BY-SA 4.0`.           |
| `IND_COMMONS_SDXL_020` | `commons:192692840`   | Igor123121       | `CC BY 4.0`               |    **ĐÍNH CHÍNH**    | Wikimedia Commons API xác nhận `CC BY 4.0` (không có SA), đính chính nhầm lẫn bản nháp ghi `CC BY-SA 4.0`.           |
| `IND_COMMONS_SDXL_005` | `commons:171463547`   | Chris Woodrich   | `CC BY-SA 4.0`            |       **KHỚP**       | Wikimedia Commons File:"Triptych" by Gord Smith.                                                                     |
| `COCO_EXT_SDXL_001`    | `coco:460160`         | PratarPersilja   | `CC BY-SA 2.0`            |   **XÁC MINH MỚI**   | Flickr photo 9345977086; Flickr oEmbed xác nhận `CC BY-SA 2.0` (`https://flic.kr/p/feSznY`).                         |

---

### Bảng 2: 8 Dòng Dossier Tiền kiểm Lịch sử v1 (Historical Pilot v1 Allocation — 4 Slot Bị Chặn)

| Slot  | Candidate ID & Nguồn gốc Chuẩn tắc                                                                                      | Stratum & Phân loại Quota                                            | Target BBox, Mask BBox & Prompt                                                                                                                                                                                                                                                                                         | Raster Mask Area & Tỷ lệ Canvas                                       | Đánh giá Tiền kiểm & Rủi ro Kỹ thuật                                                                                                                                                                                                                                                                                                                                                                                              | Trạng thái Tiền kiểm                                                                                                                                                                      |
| :---: | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | `IND_COCO_SD2_001`<br>Origin: `coco:397133`<br>Pool index: 0<br>Tác giả: Pot Noodle<br>License: CC BY 2.0               | `coco_sd2`<br>`object_replacement`<br>`small_under_10pct`            | **Target:** Chảo đồng tròn treo trên tường bếp.<br>**Prompt:** _"a round brass wall clock mounted naturally on the kitchen wall, matching the warm indoor lighting"_<br>Target bbox: `[207, 117, 270, 182]`<br>Mask bbox: `[195, 95, 280, 205]`                                                                         | **9.350 px**<br>(**3.566742%**)<br>Thuần nhất nhị phân `{0, 255}`.    | Target bbox nằm hoàn toàn trong mask bbox. Ranh giới chảo tròn khép kín nhưng mask chữ nhật cắt ngang mạch ron gạch men ốp tường; inpaint đồng hồ có rủi ro lệch bước mạch gạch nếu phối cảnh không khớp.                                                                                                                                                                                                                         | **KEPT_WITH_DOCUMENTED_RISKS**<br>Chảo đồng là vật thể hiện hữu tách biệt. Giữ lại kèm cảnh báo rủi ro biên ron gạch men.                                                                 |
| **2** | `IND_COCO_SD2_002`<br>Origin: `coco:37777`<br>Pool index: 1<br>Tác giả: larrylawfer<br>License: CC BY-NC-SA 2.0         | `coco_sd2`<br>`object_replacement`<br>`large_over_30pct`             | **Target:** Máy hút mùi trắng và hệ tủ bếp trên màu vàng.<br>**Prompt:** _"matte navy-blue upper kitchen cabinets with a stainless-steel range hood, realistic residential interior photograph"_<br>Target bbox: `[145, 125, 410, 260]`<br>Mask bbox: `[95, 75, 415, 323]`                                              | **79.360 px**<br>(**30.273438%**)<br>Thuần nhất nhị phân `{0, 255}`.  | Target bbox nằm hoàn toàn trong mask bbox. Mask chiếm 30.27% diện tích, bao trùm cụm tủ trên và máy hút mùi. Rủi ro lệch đường tụ phối cảnh tại chân nóc tủ và tiếp giáp trần bếp.                                                                                                                                                                                                                                                | **KEPT_WITH_DOCUMENTED_RISKS**<br>Thay thế hệ tủ cũ bằng tủ màu navy giữ nguyên cấu trúc không gian bếp. Đạt định mức large (30.27%).                                                     |
| **3** | `IND_COCO_SDXL_002`<br>Origin: `coco:293044`<br>Pool index: 1<br>Tác giả: john-norris<br>License: CC BY-SA 2.0          | `coco_sdxl`<br>`object_insertion`<br>`small_under_10pct`             | **Target:** Quả cà chua bi đỏ đặt trên lát bánh mì.<br>**Prompt:** _"a small red cherry tomato resting on the slice of bread, matching the lunchbox lighting and camera angle"_<br>Target bbox: `[375, 265, 430, 320]`<br>Mask bbox: `[345, 245, 455, 335]`                                                             | **9.900 px**<br>(**3.776550%**)<br>Thuần nhất nhị phân `{0, 255}`.    | **Lịch sử thực nghiệm chính xác:** Omission xảy ra ở 4 attempts (pilot 113700Z, diagnostic Arm A, calibration G7.5 và G9.5) trên cùng candidate và seed. Riêng diagnostic Arm B **đã tạo sinh được quả cà chua**, nhưng đã bị **Human Content QC REJECT** chính thức bởi người duyệt (Dũng Phạm, `2026-10-08T19:34:30Z`) vì khuyết tật bước nhảy biên (boundary step) và kết cấu vụn bánh mì không đạt chuẩn chất lượng thị giác. | **BLOCKED / PENDING_TRADE_OFF**<br>Không có cơ sở kỳ vọng tiếp tục chạy full-canvas trên lát bánh mì này sẽ hết omission hoặc đạt chuẩn biên. Cần người dùng duyệt phương án thay thế.    |
| **4** | `IND_COCO_SDXL_041`<br>Origin: `coco:189310`<br>Pool index: 40<br>Tác giả: an iconoclast<br>License: CC BY 2.0          | `coco_sdxl`<br>`object_insertion`<br>`large_over_30pct`              | **Target:** Khoảng trần mở phía trên phòng khách cho đèn chùm.<br>**Prompt:** _"an elegant crystal chandelier hanging from the living room ceiling, warm interior illumination matching the residential lighting"_<br>Target bbox: `[180, 15, 332, 140]`<br>Mask bbox: `[0, 0, 512, 155]`                               | **79.360 px**<br>(**30.273438%**)<br>Thuần nhất nhị phân `{0, 255}`.  | Target bbox nằm hoàn toàn trong mask bbox. Đèn chùm thực tế chỉ chiếm ~7.86% diện tích, nhưng mask bị kéo ngang $512 \times 155$ để ép đủ quota $\ge 30\%$, gây bước nhảy tông độ sáng mặt phẳng trần tại $y=155$ và chạm vi mô mép rèm cửa trái ($x \in [12, 25]$). Rà soát toàn bộ 9 large + 12 medium insertion trong stratum `coco_sdxl` xác nhận không có ca nào trong catalog đạt chuẩn.                                    | **BLOCKED / PENDING_TRADE_OFF**<br>Ép mask trần phẳng 30% phá vỡ sự đồng nhất kết cấu của mảng trần. Cần người dùng duyệt phương án mở rộng catalog COCO.                                 |
| **5** | `IND_COMMONS_SD2_001`<br>Origin: `commons:92533678`<br>Pool index: 0<br>Tác giả: Moahim<br>License: CC BY-SA 4.0        | `commons_sd2`<br>`object_removal_and_infill`<br>`medium_10_to_30pct` | **Target:** Mũi đất rừng thông, chân vách đá và bãi đá ngầm nhô ra biển.<br>**Prompt:** _"open sea and distant coastline continuing naturally through the removed foreground headland, photorealistic sunset landscape"_<br>Target bbox: `[190, 308, 512, 512]`<br>Mask bbox: `[190, 305, 512, 512]`                    | **66.654 px**<br>(**25.426483%**)<br>Thuần nhất nhị phân `{0, 255}`.  | Target bbox nằm hoàn toàn trong mask bbox. Bờ vịnh lõm chéo khiến mask chữ nhật bắt buộc phải bao trùm một phần mặt nước vịnh ($x=190..340$) và chân dãy nhà/khách sạn sườn đồi ($x=440..512, y=305..335$). Rủi ro lặp vân sóng hoặc nhòe chân công trình.                                                                                                                                                                        | **KEPT_WITH_DOCUMENTED_RISKS**<br>Candidate duy nhất trong stratum cho removal. Option A bao trùm toàn bộ mũi đất và rặng thông. Chấp nhận rủi ro inpaint chân khách sạn.                 |
| **6** | `IND_COMMONS_SD2_002`<br>Origin: `commons:81567907`<br>Pool index: 1<br>Tác giả: Mr.choppers<br>License: CC BY-SA 3.0   | `commons_sd2`<br>`object_insertion`<br>`medium_10_to_30pct`          | **Target:** Vali du lịch da nâu đặt trên mặt đường đá cuội.<br>**Prompt:** _"a brown leather travel suitcase standing on the cobblestones beside the vintage car, realistic scale and daylight shadows"_<br>Target bbox: `[45, 355, 190, 495]`<br>Mask bbox: `[0, 340, 225, 512]`                                       | **38.700 px**<br>(**14.762878%**)<br>Thuần nhất nhị phân `{0, 255}`.  | Target bbox nằm hoàn toàn trong mask bbox. Quan sát thực nghiệm: SD2 sinh ra xe hơi đồ chơi ở Arm B và omission ở Arm A. Bối cảnh xe đua Bugatti quá mạnh lấn át token vali trong text prompt. Mép mask $x=225$ tiếp xúc sát trục bánh xe.                                                                                                                                                                                        | **BLOCKED / PENDING_TRADE_OFF**<br>Rủi ro ảo giác hoặc không tạo đối tượng cao khi đặt vali cạnh xe đua. Cần người dùng duyệt phương án thay thế.                                         |
| **7** | `IND_COMMONS_SDXL_001`<br>Origin: `commons:166503140`<br>Pool index: 0<br>Tác giả: Crisco 1492<br>License: CC BY-SA 4.0 | `commons_sdxl`<br>`object_insertion`<br>`small_under_10pct`          | **Target:** Con chim nhỏ bay trên nền trời mây mở.<br>**Prompt:** _"a small dark bird flying in the cloudy sky, distant scale and natural daylight"_<br>Target bbox: `[395, 75, 455, 125]`<br>Mask bbox: `[350, 45, 500, 160]`                                                                                          | **17.250 px**<br>(**6.580353%**)<br>Thuần nhất nhị phân `{0, 255}`.   | Target bbox nằm hoàn toàn trong mask bbox. Biên mask cắt giữa bầu trời mây có gradient độ sáng liên tục (không phải biên tự nhiên của chủ thể). Rủi ro: SDXL tạo mảng trời chữ nhật lệch tone ($\Delta\text{RGB} \approx -3.5$) và chim có thể bị lệch vị trí ($dy = +41\text{ px}$).                                                                                                                                             | **KEPT_WITH_DOCUMENTED_RISKS**<br>Vùng trời không cắt cấu trúc vật lý. Rủi ro thuần túy là khả năng định vị của SDXL và độ khớp màu trời. Đạt định mức small (6.58%).                     |
| **8** | `IND_COMMONS_SDXL_003`<br>Origin: `commons:166529058`<br>Pool index: 2<br>Tác giả: Crisco 1492<br>License: CC BY-SA 4.0 | `commons_sdxl`<br>`object_replacement`<br>`large_over_30pct`         | **Target:** Thân cột điêu khắc nhôm thẳng đứng 'Tower Song'.<br>**Prompt:** _"the visible shaft of a classical fluted Greco-Roman marble column standing naturally in the public park, realistic outdoor daylight and weathered stone texture"_<br>Target bbox: `[190, 0, 360, 512]`<br>Mask bbox: `[170, 0, 380, 512]` | **107.520 px**<br>(**41.015625%**)<br>Thuần nhất nhị phân `{0, 255}`. | Target bbox nằm hoàn toàn trong mask bbox. Mask dọc $x \in [170, 380]$ cắt ngang qua thanh lan can kim loại liên tục kéo dài 210 px ở hậu cảnh. Trong ảnh đã sinh, lan can bị đứt; điều này cho thấy inpainting tại vùng này có rủi ro đứt đoạn cấu trúc rất cao.                                                                                                                                                                 | **BLOCKED / PENDING_TRADE_OFF**<br>Xung đột hình học cắt ngang lan can kim loại là rủi ro thiết kế cố hữu. Bắt buộc người dùng duyệt thay thế candidate hoặc điều chỉnh phương pháp mask. |

---

### Phương Án Đề Xuất PENDING Cho 4 Slot Bị Chặn (Screening Proposals)

Các phương án đề xuất được rà soát nghiêm ngặt theo nội dung, xuất xứ bản quyền, disjointness đối với 684 nguồn Option P lịch sử, và hình học mask chuẩn tắc. **Tất cả các đề xuất đều ở trạng thái `PENDING_HUMAN_REVIEW`, không sửa catalog/plan đã niêm phong và không ghi nhận approval:**

#### 1. Slot 3 Đề Xuất PENDING (`coco_sdxl`, `object_insertion`, `small_under_10pct`)

- **Phương án Khuyến nghị (RECOMMENDED PENDING)**: `IND_COCO_SDXL_042` (Pool index: 41).
  - Origin ID: `coco:448076` | Nguồn ảnh trực tiếp: [Flickr luis.leao](https://www.flickr.com/photos/luisleao/2260856815/) | Tác giả: luis.leao | Giấy phép: `CC BY 2.0`.
  - Disjoint Guard: **PASS** (Zero overlap với 684 nguồn Option P lịch sử).
  - Mô tả cảnh authentic: Gian hàng triển lãm hàng không TAM. Khoảng thảm đỏ phía trước ($x \in [200, 360], y \in [360, 500]$) hoàn toàn rộng mở, sạch sẽ, phẳng đồng nhất, không có đồ vật che chắn (người đứng ở vách sau $y < 370$ và mép trái $x < 150$).
  - Target đề xuất: Cặp táp doanh nhân bằng da đen đứng thẳng trên thảm đỏ.
  - Prompt đề xuất: _"a professional black leather business briefcase standing upright on the red exhibition carpet, realistic studio floodlights and soft ground shadow"_.
  - Target bbox: `[235, 395, 335, 485]` | Mask bbox: `[220, 380, 350, 500]`.
  - Target containment: **100% bao trọn** ($220 \le 235 < 335 \le 350$ và $380 \le 395 < 485 \le 500$).
  - Diện tích raster mask: **15.600 px** (**5.950928%**), chuẩn `small_under_10pct` ($1\% - 10\%$).
  - Rủi ro & Cảnh báo: Mặt thảm đỏ phẳng đồng nhất tối ưu hóa điều kiện tiền kiểm, loại trừ va chạm vật thể lân cận; tuy nhiên không khẳng định nền đơn giản bảo đảm loại trừ hoàn toàn omission.
- **Phương án Dự phòng (BACKUP PENDING)**: `IND_COCO_SDXL_040` (Pool index: 39).
  - Origin ID: `coco:578489` | Nguồn ảnh: [Flickr laura47](https://www.flickr.com/photos/laura47/3181988690/) | Tác giả: laura47 | Giấy phép: `CC BY-NC 2.0` (Attribution-NonCommercial).
  - Target đề xuất: Đôi dép len xám trên sàn gỗ (`[40, 420, 130, 490]`, Mask `[30, 410, 140, 500]`, 9.900 px = 3.776550%).
  - Rủi ro: Giày da của người đàn ông trên sofa nằm sát mép mask tại $x=100..120$; sàn gỗ có vân phản xạ chéo.

#### 2. Slot 4 Đề Xuất PENDING (`coco_sdxl`, `object_insertion`, `large_over_30pct`) — Đề Xuất Mở Rộng Catalog COCO

- **Nguyên tắc**: Giữ nguyên stratum `coco_sdxl`, SDXL, thao tác `object_insertion`, định mức `large_over_30pct` ($\ge 30\%$) và allocation đã khóa. Không đổi sang replacement.
- **Phạm vi rà soát catalog hiện hành**: Rà soát chính xác toàn bộ **9 large insertion + 12 medium insertion** trong stratum `coco_sdxl` của catalog `verified_candidate_catalog_v2.json` (không suy rộng thành toàn bộ pool chung):
  1. Pool 0 (`_001`): Mèo trong bồn rửa — không có 30% canvas trống.
  2. Pool 4 (`_005`): Sân bay đêm — mask trùm nhân viên mặc áo phản quang, xe đẩy lồng sắt và bóng phản xạ mũi máy bay.
  3. Pool 28 (`_029`): Người vác chuối tại chợ — người và sọt chuối kín trung tâm cảnh.
  4. Pool 32 (`_033`): Bàn tiệc Đan Mạch — bàn đầy đĩa, ly, hoa và người chụp ảnh.
  5. Pool 40 (`_041`): Đèn chùm trần nhà — trần phẳng kéo $512 \times 155$ gây lệch tông độ sáng mặt phẳng trần.
  6. Pool 52 (`_053`): Bồn rửa mặt — dính gương phản chiếu và khung viền trầy xước.
  7. Pool 66 (`_067`): 4 bát đồ ăn vặt — đĩa bát chiếm 95% diện tích khung hình.
  8. Pool 69 (`_070`): Bánh mì trên tay — tay và bánh mì chéo kín khung hình.
  9. Pool 107 (`_108`): Phố Chicago — người đi bộ, cột đèn tín hiệu giữa ảnh, thùng rác, cây xanh.
  - Đồng thời kiểm tra 12 candidate medium insertion: 100% đều không có vùng trống $\ge 30\%$ tự nhiên mà không đè lên con người hoặc chủ thể cần bảo toàn.
- **Đề xuất Mở rộng Catalog COCO Riêng biệt (PROPOSED COCO EXTENSION PENDING)**:
  - Tạo đề xuất độc lập `slot4_coco_extension_proposal.json` mà **không chỉnh sửa catalog hay plan đã niêm phong**.
  - **Candidate đề xuất**: **`COCO_EXT_SDXL_001`**.
  - Origin ID: `coco:460160` | Flickr Photo ID: `9345977086` | Nguồn: [Flickr PratarPersilja](https://www.flickr.com/photos/12297627@N03/9345977086/) | Tác giả: PratarPersilja | Giấy phép: `CC BY-SA 2.0` (xác minh chính thức qua Flickr oEmbed URL `https://flic.kr/p/feSznY`).
  - Disjoint Guard: **PASS** (Zero overlap với 684 nguồn Option P lịch sử).
  - Đặc điểm cảnh authentic: Bờ biển mở rộng với thảm cỏ và bãi cát tự nhiên ở tiền cảnh ($y \in [260, 510]$), hậu cảnh là mặt nước và bầu trời mây overcast. Cảnh tĩnh mịch, hoàn toàn không có người, xe cộ hay kết cấu kiến trúc phức tạp.
  - Thao tác: `object_insertion` | Định mức: `large_over_30pct`.
  - Target đề xuất: Bàn dã ngoại bằng gỗ mộc có ghế băng gắn liền đặt trên bờ cỏ ven biển (`[20, 280, 330, 490]`, diện tích 65.100 px = 24.83%).
  - Prompt đề xuất: _"a weathered wooden picnic table with attached bench seating on the grassy coastal ground, natural overcast daylight and soft ground contact shadow"_.
  - Mask AABB đề xuất: `[10, 260, 340, 510]`.
  - Diện tích raster mask: **82.500 px** (**31.471252%**), đạt chuẩn `large_over_30pct` ($\ge 30\%$, dư +3.857 px).
  - **Phân biệt BBox Containment với Coverage Đối tượng Thực tế**:
    - _BBox Containment_: 100% target bbox nằm trong mask bbox ($10 \le 20 < 330 \le 340$ và $260 \le 280 < 490 \le 510$).
    - _Bản chất thao tác insertion_: Ảnh authentic là nền bờ biển tự nhiên, chưa có đối tượng trước khi sinh. Target bbox là hộp bao không gian cho bàn gỗ và ghế băng. Đối tượng vật lý được tạo sinh (mặt bàn, chân bàn, ghế băng, bóng đổ tiếp xúc) sẽ chiếm một diện mạo 3D thực tế là tập con bên trong target bbox.
    - _Kiểm tra bệ/chân & bóng đổ tiếp xúc_: Khoảng lề đáy $y \in [490, 510]$ (20 px) và lề ngang $x \in [10, 20]$ & $[330, 340]$ (10 px) nhằm dự phòng không gian cho chân bàn và bóng tiếp xúc trên nền đất. Tuy nhiên, lề biên tiền kiểm không bảo đảm mô hình sẽ tạo bóng đổ trải mềm hoặc triệt tiêu hoàn toàn bước nhảy biên.
    - _Kiểm tra vật thể lân cận_: Vùng bờ biển quang đãng, không có người, xe cộ hay công trình xây dựng. Bề mặt đất cỏ/cát gồ ghề giảm thiểu độ nhạy cảm so với mặt phẳng trần thạch cao đồng nhất của `_041`, song vẫn tiềm ẩn rủi ro sai lệch cấu trúc hoặc tông màu tại đường biên inpainting.

#### 3. Slot 6 Đề Xuất PENDING (`commons_sd2`, `object_insertion`, `medium_10_to_30pct`)

- **Phương án Khuyến nghị (RECOMMENDED PENDING)**: `IND_COMMONS_SD2_040` (Pool index: 39).
  - Origin ID: `commons:172876577` | Nguồn ảnh trực tiếp: [Wikimedia Commons Chainwit.](<https://commons.wikimedia.org/wiki/File:%22Eyes_of_Sibiu%22_at_house_Strada_Ocnei_2,_Sibiu_(2023)_-_img_07.jpg>) | Tác giả: Chainwit. | Giấy phép: `CC BY 4.0` (Creative Commons Attribution 4.0 International).
  - Disjoint Guard: **PASS** (Zero overlap với 684 nguồn Option P lịch sử).
  - Mô tả cảnh authentic: "Eyes of Sibiu" — Quảng trường nhà cổ Sibiu với mặt đường lát đá cuội rộng thoáng ở tiền cảnh ($y \in [360, 512]$), hoàn toàn tĩnh lặng và không có phương tiện cơ giới.
  - **Đơn giản hóa hình khối**: Thay vì xe đạp với nan hoa mỏng, xích và khoảng trống xuyên thấu phức tạp (dễ sinh lỗi đứt gãy hình học không cần thiết cho kiểm thử insertion), đề xuất chậu hoa thùng gỗ mộc có hình khối trụ đặc vững chắc, biên ranh giới rõ ràng, diện mạo ăn nhập hoàn hảo với quảng trường cổ Sibiu.
  - Target đề xuất: Chậu hoa thùng gỗ mộc trên mặt đường lát đá cuội (`[45, 375, 215, 495]`, 20.400 px = 7.78%).
  - Prompt đề xuất: _"a rustic wooden barrel planter filled with vibrant blooming flowers sitting naturally on the cobblestone pavement, realistic daylight shadows and weathered wood texture matching the historic town square"_.
  - Mask AABB đề xuất: `[20, 360, 240, 512]`.
  - Target containment: **100% bao trọn** ($20 \le 45 < 215 \le 240$ và $360 \le 375 < 495 \le 512$).
  - Diện tích raster mask: **33.440 px** (**12.756348%**), chuẩn `medium_10_to_30pct` ($10\% - 30\%$).
- **Phương án Tàu kéo `IND_COMMONS_SD2_021` — Chính thức BLOCKED**:
  - Origin ID: `commons:132687303` | Nguồn: [Wikimedia Commons JoachimKohler-HB](<https://commons.wikimedia.org/wiki/File:%22Cap_San_Diego%22_%26_Elbphilharmonie_(Hamburg,_2019).jpg>).
  - Rà soát authentic xác nhận mask `[0, 375, 240, 512]` chứa cầu đi bộ Überseebrücke, cọc trụ bê tông cảng và phao nổi đón khách. Phần nước trống bên dưới cầu chỉ đạt 20.500 px (7.82%), hụt định mức medium ($\ge 10\%$). Không tồn tại hình học mask nào đạt quota mà không phá hoại cầu cảng.

#### 4. Slot 8 Thẩm Tra Hình Học Offline & Đề Xuất Thay Thế Trong Cùng Allocation (`commons_sdxl`, `object_replacement`, `large_over_30pct`)

- **Thẩm tra hình học offline trên `IND_COMMONS_SDXL_020` (Żyletkowce)**:
  - Nguồn: `commons:192692840` | Tác giả: Igor123121 | Giấy phép: `CC BY 4.0`.
  - Target đăng ký: `[60, 145, 385, 365]` (71.500 px = 27.275391% canvas).
  - **Sửa sai lệch tọa độ và phân tích chính xác Mask AABB `[50, 130, 395, 360]`**:
    - Kích thước: $345 \times 230 =$ **79.350 px** (**30.269623%** canvas). Đạt định mức quota large ($\ge 30\%$, dư +707 px).
    - _Đính chính tọa độ_: Mask kết thúc tại $y=360$, **chỉ thiếu đúng 5 hàng đáy target** ($y \in [360, 365)$ trên chiều rộng 325 px, tương đương 1.625 px hay 2.27% diện tích target). Mask **hoàn toàn không thiếu cạnh phải** (trục X $x \in [50, 395]$ bao trọn target $x \in [60, 385]$ với 10 px padding mỗi bên).
    - _Tính lại giao với vòm kính từ raster_: Đỉnh khung kim loại vòm kính đạt $y=358$ tại $x \approx 208..212$. Mask $y \in [130, 360)$ cắt lẹm đúng **2 hàng đỉnh khung vòm kính** ($y=358, 359$, khoảng 14 px raster), **không tái sử dụng số đo 13 px của mask cũ** (13 px là độ sâu của mask cũ kéo tới $y=375$).
  - **Đo đạc Raster Contour Mask Nháp Offline (né đỉnh vòm kính và cấu trúc tiền cảnh)**:
    - Xuất raster nhị phân và overlay bám theo mép thực của thân tháp: `IND_COMMONS_SDXL_020_contour_mask.png` và `IND_COMMONS_SDXL_020_contour_overlay.png`.
    - Số pixel mask contour thực tế: **65.792 px** (**25.097656%** canvas).
    - Kết quả kiểm tra quota: **KHÔNG ĐẠT định mức large** ($\ge 30\%$, thiếu hụt nghiêm trọng -12.851 px hay -4.90%).
    - Kết quả che phủ target (Target Coverage): Chỉ bao phủ **65.496 px / 71.500 px** (**91.60%** diện tích target; bỏ sót 6.004 px = 8.40% diện tích target tại chân tháp).
    - Tuân thủ nguyên tắc trung thực khoa học: Tuyệt đối không thu hẹp hoặc tái đăng ký target post-hoc để hợp thức hóa mask.
  - **Kiểm tra khả năng hỗ trợ trong Schema & Runner hiện có**:
    - Runner (`ml/evaluation/independent_cohort_acquisition.py`) chỉ hỗ trợ trục tọa độ hộp chữ nhật `mask_bbox_xyxy: tuple[int, int, int, int]`, hoàn toàn không hỗ trợ raster mask polygon nếu không sửa code và ban hành amendment.
- **Rà soát Candidate Khác Trong Cùng Allocation (`commons_sdxl`, `object_replacement`, `large_over_30pct`)**:
  - **Phương án Khuyến nghị (RECOMMENDED PENDING)**: **`IND_COMMONS_SDXL_005`** (Pool index: 4).
    - Origin ID: `commons:171463547` | Nguồn: [Wikimedia Commons Chris Woodrich](<https://commons.wikimedia.org/wiki/File:%22Triptych%22_by_Gord_Smith_(rear_view),_Windsor,_Ontario,_2025-08-01.jpg>) | Tác giả: Chris Woodrich | Giấy phép: `CC BY-SA 4.0`.
    - Disjoint Guard: **PASS** (Zero overlap với 684 nguồn Option P lịch sử).
    - Target thay thế: Cột kỷ niệm bằng đá cẩm thạch trắng đứng trên bệ khối vững chắc trong công viên.
    - Prompt đề xuất: _"a classical white marble commemorative column standing on a solid plinth in the public park, realistic overcast daylight and weathered stone texture"_.
    - Target bbox: `[145, 45, 355, 465]` (diện tích 88.200 px = 33.647461% canvas).
    - Mask AABB đề xuất: `[135, 40, 365, 475]`.
    - Diện tích raster mask: **100.050 px** (**38.166046%** canvas), đạt chuẩn `large_over_30pct` ($\ge 30\%$).
    - **Phân tích BBox Containment với Coverage Đối tượng Thực tế**:
      - _BBox Containment_: 100% target bbox nằm trong mask bbox ($135 \le 145 < 355 \le 365$ và $40 \le 45 < 465 \le 475$).
      - _Che phủ đối tượng thực tế (Physical Object Coverage)_: Khối điêu khắc đồng "Triptych" trải dài từ $x=145$ đến $x=352$ và đỉnh cột bắt đầu tại $y=45$ đến chân bệ bê tông tại $y=465$ (bao gồm cả bệ bê tông chân tượng tại $y=445..465$). Target rectified `[145, 45, 355, 465]` bao trọn 100% cấu trúc vật lý thực tế của tượng và bệ đỡ; toàn bộ thân tượng và bệ đỡ đều nằm trọn 100% trong mask AABB.
      - _Kiểm tra bệ/chân & bóng đổ tiếp xúc_: Chân bệ bê tông kết thúc ở $y \approx 465$, thảm cỏ bắt đầu từ $y \approx 470$. Khoảng lề đáy $y \in [465, 475]$ (10 px) dự phòng không gian cho bệ móng và bóng đổ tiếp xúc trên mặt cỏ, song không bảo đảm loại trừ hoàn toàn nguy cơ vết sẹo biên hoặc bậc tương phản vi mô.
      - _Kiểm tra vật thể lân cận_: Hoàn toàn không có lan can kim loại (khắc phục lỗi của Tower Song `_003`), không có vòm kính tròn (khắc phục lỗi của Żyletkowce `_020`), không có người đi bộ. Hậu cảnh là bầu trời mây và rặng cây công viên ở xa.
      - _Tính tương thích Runner/Schema_: Sử dụng hộp chữ nhật AABB chuẩn, **tương thích 100% với schema và code runner hiện hành mà không cần sửa code!**
  - **Phương án Dự phòng (BACKUP PENDING)**: `IND_COMMONS_SDXL_002` (Pool index: 1, tượng Ed Dwight ôm trẻ, CC BY-SA 4.0; target `[180, 120, 400, 430]`, mask `[170, 110, 410, 440]`, 79.200 px = 30.21%). Lưu ý rủi ro đồng nhất danh tính và cánh tay tiếp xúc.

---

### Bảng Tổng Hợp 4 Slot Đề Xuất PENDING Trình Duyệt

| Slot  | Stratum & Thao Tác                     | Candidate Đề Xuất (PENDING)                            | Geometry (Target & Mask BBox)                                  | Diện Tích Mask & Quota                                  | Provenance & Giấy Phép Xác Minh                                                                                             | Rủi Ro Kỹ Thuật Còn Lại                                                                                                 | Nội Dung Chính Xác Cần Người Dùng Duyệt                                                              |
| :---: | :------------------------------------- | :----------------------------------------------------- | :------------------------------------------------------------- | :------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **3** | `coco_sdxl`<br>insertion<br>small      | **`IND_COCO_SDXL_042`**<br>(Pool 41, gian hàng TAM)    | Target: `[235, 395, 335, 485]`<br>Mask: `[220, 380, 350, 500]` | **15.600 px**<br>(**5.95%**)<br>Đạt `small` (1-10%)     | `coco:448076`<br>Tác giả: luis.leao<br>License: `CC BY 2.0`                                                                 | Thảm phẳng tối ưu tiền kiểm nhưng không bảo đảm 100% loại trừ omission của SDXL.                                        | Duyệt thay thế `IND_COCO_SDXL_002` bằng `IND_COCO_SDXL_042` (cặp táp da trên thảm đỏ).               |
| **4** | `coco_sdxl`<br>insertion<br>large      | **`COCO_EXT_SDXL_001`**<br>(Đề xuất mở rộng COCO)      | Target: `[20, 280, 330, 490]`<br>Mask: `[10, 260, 340, 510]`   | **82.500 px**<br>(**31.47%**)<br>Đạt `large` (&ge;30%)  | `coco:460160`<br>Tác giả: PratarPersilja<br>License: `CC BY-SA 2.0`<br>Extension: `candidate_catalog_extension_v1.0.0.json` | Bờ biển tự nhiên; cần theo dõi tiếp xúc chân bàn gỗ với nền cỏ/cát. Nền mặt đất tránh được lỗi bước nhảy tông mặt trần. | Duyệt mở rộng catalog COCO với candidate `COCO_EXT_SDXL_001` (bàn dã ngoại gỗ trên bờ cỏ).           |
| **6** | `commons_sd2`<br>insertion<br>medium   | **`IND_COMMONS_SD2_040`**<br>(Pool 39, phố cổ Sibiu)   | Target: `[45, 375, 215, 495]`<br>Mask: `[20, 360, 240, 512]`   | **33.440 px**<br>(**12.76%**)<br>Đạt `medium` (10-30%)  | `commons:172876577`<br>Tác giả: Chainwit.<br>License: `CC BY 4.0`                                                           | Nền đá cuội cổ; SD2 cần tạo bóng đổ tiếp xúc tự nhiên với mặt đường.                                                    | Duyệt thay thế `IND_COMMONS_SD2_002` bằng `IND_COMMONS_SD2_040` (chậu hoa thùng gỗ mộc).             |
| **8** | `commons_sdxl`<br>replacement<br>large | **`IND_COMMONS_SDXL_005`**<br>(Pool 4, tượng Triptych) | Target: `[145, 45, 355, 465]`<br>Mask: `[135, 40, 365, 475]`   | **100.050 px**<br>(**38.17%**)<br>Đạt `large` (&ge;30%) | `commons:171463547`<br>Tác giả: Chris Woodrich<br>License: `CC BY-SA 4.0`                                                   | Thay thế tượng đồng bằng cột đá; rủi ro khớp chân bệ với mặt cỏ công viên. Không vướng lan can hay vòm kính.            | Duyệt thay thế `IND_COMMONS_SDXL_003` bằng `IND_COMMONS_SDXL_005` (thay tượng Triptych bằng cột đá). |

---

### Bảng 3: 8 Dòng Chuẩn Tắc Pilot Plan v2 Đã Phê Duyệt (APPROVED Human Review)

| Slot | Candidate ID | Origin ID & Tác Giả / Giấy Phép | Prompt Đề Xuất Chuẩn Tắc | Target BBox [x1, y1, x2, y2] & Diện Tích | Mask BBox [x1, y1, x2, y2] & Diện Tích Tính Từ Tọa Độ | Trạng Thái Phê Duyệt |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | `IND_COCO_SD2_001` | `coco:397133`<br>Pot Noodle (`CC BY 2.0`) | _"a round brass wall clock mounted naturally on the kitchen wall, matching the warm indoor lighting"_ | `[207, 117, 270, 182]`<br>$63 \times 65 =$ **4.095 px** (1.5621%) | `[195, 95, 280, 205]`<br>$85 \times 110 =$ **9.350 px** (3.5667%) | **APPROVED** |
| **2** | `IND_COCO_SD2_002` | `coco:37777`<br>larrylawfer (`CC BY-NC-SA 2.0`) | _"matte navy-blue upper kitchen cabinets with a stainless-steel range hood, realistic residential interior photograph"_ | `[145, 125, 410, 260]`<br>$265 \times 135 =$ **35.775 px** (13.6471%) | `[95, 75, 415, 323]`<br>$320 \times 248 =$ **79.360 px** (30.2734%) | **APPROVED** |
| **3** | `IND_COCO_SDXL_042` | `coco:448076`<br>luis.leao (`CC BY 2.0`) | _"a professional black leather business briefcase standing upright on the red exhibition carpet, realistic studio floodlights and soft ground shadow"_ | `[235, 395, 335, 485]`<br>$100 \times 90 =$ **9.000 px** (3.4332%) | `[220, 380, 350, 500]`<br>$130 \times 120 =$ **15.600 px** (5.9509%) | **APPROVED** |
| **4** | `COCO_EXT_SDXL_001` | `coco:460160`<br>PratarPersilja (`CC BY-SA 2.0`) | _"a weathered wooden picnic table with attached bench seating on the grassy coastal ground, natural overcast daylight and soft ground contact shadow"_ | `[20, 280, 330, 490]`<br>$310 \times 210 =$ **65.100 px** (24.8337%) | `[10, 260, 340, 510]`<br>$330 \times 250 =$ **82.500 px** (31.4713%) | **APPROVED** |
| **5** | `IND_COMMONS_SD2_001` | `commons:92533678`<br>Moahim (`CC BY-SA 4.0`) | _"open sea and distant coastline continuing naturally through the removed foreground headland, photorealistic sunset landscape"_ | `[190, 308, 512, 512]`<br>$322 \times 204 =$ **65.688 px** (25.0580%) | `[190, 305, 512, 512]`<br>$322 \times 207 =$ **66.654 px** (25.4265%) | **APPROVED** |
| **6** | `IND_COMMONS_SD2_040` | `commons:172876577`<br>Chainwit. (`CC BY 4.0`) | _"a rustic wooden barrel planter filled with vibrant blooming flowers sitting naturally on the cobblestone pavement, realistic daylight shadows and weathered wood texture matching the historic town square"_ | `[45, 375, 215, 495]`<br>$170 \times 120 =$ **20.400 px** (7.7820%) | `[20, 360, 240, 512]`<br>$220 \times 152 =$ **33.440 px** (12.7563%) | **APPROVED** |
| **7** | `IND_COMMONS_SDXL_001` | `commons:166503140`<br>Crisco 1492 (`CC BY-SA 4.0`) | _"a small dark bird flying in the cloudy sky, distant scale and natural daylight"_ | `[395, 75, 455, 125]`<br>$60 \times 50 =$ **3.000 px** (1.1444%) | `[350, 45, 500, 160]`<br>$150 \times 115 =$ **17.250 px** (6.5804%) | **APPROVED** |
| **8** | `IND_COMMONS_SDXL_005` | `commons:171463547`<br>Chris Woodrich (`CC BY-SA 4.0`) | _"a classical white marble commemorative column standing on a solid plinth in the public park, realistic overcast daylight and weathered stone texture"_ | `[145, 45, 355, 465]`<br>$210 \times 420 =$ **88.200 px** (33.6456%) | `[135, 40, 365, 475]`<br>$230 \times 435 =$ **100.050 px** (38.1660%) | **APPROVED** |

---

### Các Bước Kế Tiếp & Phê Duyệt Chính Thức (Formal Approval & Actionable Status)

1. **Phê Duyệt Chính Thức Pilot Plan v2 & Protocol Amendment v1.7.0 (APPROVED)**:
   - Toàn bộ 8 instruction trong Pilot Plan v2 proposal (`content_grounded_pilot_plan_v2_proposal.json`) và Protocol Amendment v1.7.0 (`PROTOCOL_AMENDMENT_V1.7.0.md`) đã được phê duyệt chính thức bởi người duyệt: **Dũng Phạm <valdung04@gmail.com>** vào lúc **`2026-10-09T10:24:00Z`**.
   - SHA-256 Kế hoạch trước phê duyệt: `cd9086120b6b54a75d73db8482f1364ed7ee2643668c2a6fa44e66282355fb64` (đã đối chiếu khớp 100% trước khi ghi nhận).
   - SHA-256 Kế hoạch sau phê duyệt: `a95c96c594777025aac58049ca81978245035a54ad98c9298c0cf5a1ec0d8565`.
   - SHA-256 Catalog Extension: `0d875b81a8044e3285ecf0331beec6646d94b085915baaaa32247e86ee9023e7`.
   - Ngân sách thực thi: Tối đa 8 attempts, mỗi candidate đúng 1 attempt duy nhất; attempt lỗi vẫn tính ngân sách; tuyệt đối không retry hoặc automatic replacement.
2. **Cảnh báo Kỷ luật Quản trị & Phạm Vi Cho Phép**:
   - **Phê duyệt chỉ cho phép thực thi pilot trên GPU Colab**; tuyệt đối **không generation cục bộ**, không detector/training, không mở khóa full cohort ($N=400$ locked).
   - Giữ nguyên mọi quyết định Human Content QC lịch sử: Diagnostic run 6/6 REJECT, Calibration run 2 PENDING, Historical pilot 8 PENDING.
   - Phê duyệt này chỉ cho phép thực hiện pilot, hoàn toàn không đồng nghĩa với nghiệm thu ảnh đầu ra (ảnh sinh ra phải trải qua Human Content QC độc lập sau khi thu thập).
3. **Quy Trình Triển Khai Colab**:
   - Notebook được ghim chặt chẽ vào commit chứa approval chính thức.
   - Người dùng thực hiện chạy tuần tự từng cell trên môi trường GPU Colab (T4 / A100).


## EE. Approved calibration runner implementation & diagnostic determinations dossier (2026-10-08)

- **Calibration Execution Harness (`ml/evaluation/independent_cohort_calibration.py`)**:
  - Implemented dedicated calibration runner completely decoupled from diagnostic (6 attempts) and pilot (8 attempts).
  - Enforces fail-closed guards: plan review status must be strictly `APPROVED` (`CalibrationPlanNotApprovedError` on PENDING or null reviewer); plan SHA-256 must match approved proposal `2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c`; input authentic and mask PNG hashes verified against sealed files from `pilot-20261008T113700Z`; candidate binding locked to `IND_COCO_SDXL_002` at seed `20272319`; production mode bars mock engine (`CalibrationEngineError` when allow_mock=False).
  - Explicit parameter transmission: `negative_prompt` and `guidance_scale` (7.5 vs 9.5) passed explicitly to diffusers inpainting pipeline; fresh isolated torch RNG generator seeded per attempt; outside pixels strictly preserved by request mask compositing (`outside_mean_l1 == 0.000000`).
  - Budget & lifecycle accounting: exactly 2 attempts budget; failed attempts consume budget; resume treats started/accepted/failed attempts as consumed and skips them.
  - Comprehensive runtime telemetry: captures Python, PyTorch, Diffusers, Transformers, NumPy, Pillow, CUDA runtime version, and GPU device name/VRAM faithfully without speculation.
  - Outputs separate run directory `calib-<TIMESTAMP>`, self-contained HTML contact sheet (`calibration_contact_sheet.html`) with PENDING review status, durable ledgers (`attempt_ledger.jsonl`, `provenance_ledger.jsonl`), run binding (`run_binding.json`), and signed receipt (`calibration_receipt.json`).
- **Production CLI & Colab Notebook Integration**:
  - `scripts/research/run_cohort_acquisition.py`: Added `--mode calibration`, `--calibration-plan-path`, `--calibration-inputs-dir`, `--check-calibration-plan`, and execution dispatch.
  - Preflight verification verified: `python scripts/research/run_cohort_acquisition.py --check-calibration-plan` returned `status: PASS` with both sealed inputs verified.
  - `notebooks/independent_cohort_acquisition_colab.ipynb`: Configured explicit `EXECUTION_MODE = 'calibration'`; preflight input verification cell; 2-attempt calibration execution cell; `RUN_CONTEXT` logging; and Cell 4 audit/packaging creating `calib-<TIMESTAMP>_package.zip` without overwriting diagnostic or pilot archives.
- **Diagnostic Content QC Determinations Dossier**:
  - Created machine-readable dossier `research/evidence/phase-4c.7b/diagnostic_content_qc_determinations.json` and local addendum `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/content_qc_determinations_addendum.json`.
  - Records formal Human Content QC determination: 6/6 REJECT (Human reviewer: Dũng Phạm `<valdung04@gmail.com>`, timestamp: `2026-10-08T19:34:30Z`).
  - Preserves Technical QC (PASS) and original ledgers/receipts intact; does not delete diagnostic images; eight historical pilot pairs remain PENDING; full cohort remains LOCKED.
- **Targeted Test Coverage**:
  - `ml/tests/test_independent_cohort_calibration.py`: 12/12 PASS (testing approval gate, hash mismatches, explicit parameter passing, isolated generator seeds, 2-attempt budget, error accounting, resume skip, mock safeguard, telemetry, and outside L1 invariance).
  - `ml/tests/test_independent_cohort_bindings.py`: 62/62 PASS (including notebook calibration cell validation).
  - Zero local generation, zero GPU execution, zero training.

## DD. Diagnostic run diag-20261008T154628Z intake, audit, and empirical A/B evaluation (2026-10-08)

- **Archive Intake & Safety Audit**:
  - ZIP package: `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z_package.zip` (15,988,679 bytes, SHA-256 `f13d7baf458e9db86809bced5a20c27f37adb6f698353a5678b200262a481d20`).
  - Pre-extraction safety check: 30 archive members inspected, 0 directory traversals (`..`), 0 leading slashes, 0 absolute paths. Safely extracted into dedicated directory `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/` without overwriting historical pilot runs. All original ZIP, image, mask, receipt, and ledger files preserved intact.
- **Production Audit & Binding Verification**:
  - Executed CLI audit at binding commit `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`: `python scripts/research/run_cohort_acquisition.py --audit-run diag-20261008T154628Z` $\to$ **`PASS`**.
  - Verified run ID `diag-20261008T154628Z`, source commit `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`, approved plan SHA-256 `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157`.
  - Attempt accounting: exactly 6 attempts across 3 candidates $\times$ 2 arms (`STARTED` and `ACCEPTED` in `attempt_ledger.jsonl` represent two lifecycle events within each attempt, not 12 attempts).
  - Input bit-parity: all authentic and mask PNG hashes match approved plan JSON and match bit-identically to sealed normalized inputs from `pilot-20261008T113700Z` (`max_auth_diff == 0`, `max_mask_diff == 0`).
- **Recalculated Pixel-Level Metrics (Linear Scale vs Geometric Area Factor)**:
  - `IND_COCO_SDXL_002` (bread tomato, SDXL):
    - Arm A (512x512 full canvas): Linear scale 1.0x, Geometric area factor 1.0x; inside mean L1 = `16.7861` (std 13.73, max delta 81.0), outside mean L1 = `0.000000` (max delta 0.0).
    - Arm B (1024x1024 local crop): Linear scale 4.0x, Geometric area factor 16.0x; inside mean L1 = `28.2929` (std 32.96, max delta 176.0), outside mean L1 = `0.000000` (max delta 0.0).
  - `IND_COMMONS_SD2_002` (suitcase, SD2):
    - Arm A (512x512 full canvas): Linear scale 1.0x, Geometric area factor 1.0x; inside mean L1 = `30.7358` (std 24.12, max delta 146.0), outside mean L1 = `0.000000` (max delta 0.0).
    - Arm B (512x512 local crop): Linear scale 1.6x, Geometric area factor 2.56x; inside mean L1 = `43.3932` (std 40.49, max delta 226.0), outside mean L1 = `0.000000` (max delta 0.0).
  - `IND_COMMONS_SDXL_001` (sky bird, SDXL):
    - Arm A (512x512 full canvas): Linear scale 1.0x, Geometric area factor 1.0x; inside mean L1 = `11.0767` (std 9.00, max delta 43.0), outside mean L1 = `0.000000` (max delta 0.0).
    - Arm B (1024x1024 local crop): Linear scale 4.0x, Geometric area factor 16.0x; inside mean L1 = `13.0674` (std 32.45, max delta 214.0), outside mean L1 = `0.000000` (max delta 0.0).
  - Background invariance: strictly verified across all 6 attempts (`outside_mean_l1 == 0.000000`, `outside_max_delta == 0.0`).
- **Visual Analysis & Empirical Metric Dissociation (4 Criteria Separation)**:
  - _Tomato (`IND_COCO_SDXL_002`)_: Arm A completely omitted object (infilled bread crumb texture). Arm B successfully synthesized a plausible red cherry tomato with spherical highlight and green stem. Requires evaluation of sharpness, gloss/specular highlight (~[394, 276, 403, 286]), and background crumb texture inside mask [345, 245, 455, 335] vs authentic bread crumb. Bbox reconciliation: Crop bbox is `[256, 162, 512, 418]`; registered mask bbox is `[345, 245, 455, 335]` (9,900 px, 3.776550%). Overlay zoom confirms a subtle rectangular transition step in crumb texture along the **registered mask boundary `[345, 245, 455, 335]`** due to hard binary compositing with authentic bread (not at the crop bbox boundary). Agent recommends submitting for human quality review.
  - _Suitcase (`IND_COMMONS_SD2_002`)_: Arm A completely omitted object (infilled cobblestones). Arm B produced **semantic hallucination**: synthesized a miniature vintage automobile with roof and wheels instead of a suitcase. Crop bbox is `[0, 192, 320, 512]`; `[0, 237, 360, 512]` is the resized-mask raster bbox in inference resolution. Shadow observation: the car shadow is an oval pool directly beneath the vehicle and does not reach the canvas frame ($x=0, y=512$). Internal mask edges ($y=340, x=225$) blend naturally. Defect is strictly semantic hallucination, not a boundary compositing flaw. Inside L1 increased from 30.74 to 43.39 (max delta 226): **"L1 tăng không bảo đảm thành công ngữ nghĩa"** (không rút ra kết luận khái quát về tương quan từ 3 ca). Agent recommends REJECT.
  - _Bird (`IND_COMMONS_SDXL_001`)_: Arm A completely omitted object (grey sky patch). Arm B materialized bird object, but exhibits placement deficit and boundary tone mismatch (không mô tả là hòa nhập hoàn hảo). Target bbox registered: `[395, 75, 455, 125]`. Observed dark silhouette: ~`[375, 126, 414, 156]` ($dx = -30.5\text{ px}$, $dy = +41.0\text{ px}$; vertical overlap = $0\text{ px}$, strictly below target). Measured on PNG with RGB grayscale intensity thresholds and verified visually; color thresholds are not segmentation ground truth due to anti-aliasing/scattering, but silhouette is strictly detached from target box. Clearly distinguish "inside mask" (100% within mask `[350, 45, 500, 160]`) from "proper target placement" (strictly outside target box). Target or mask are strictly not shifted post-hoc to legitimize output. Rectangular sky patch inside mask exhibits tone step ($\sim -3$ to $-4$ RGB delta). Crop bbox is strictly `[256, 0, 512, 256]` ($256 \times 256$ px); `[187, 0, 443, 256]` is not used. Agent recommends REJECT for bird Arm B under current placement requirements.
- **Execution Evidence, Timing Scope & Runtime Profiling**:
  - Crop and native resolution scaling verified: SDXL raw crops are $1024 \times 1024$ (linear 4.0x, area 16.0x); SD2 raw crop is $512 \times 512$ with 1.6x padding (linear 1.6x, area 2.56x).
  - Timing measurement scope: `diagnostic_receipt.json` and `attempt_ledger.jsonl` record identical elapsed times for all 6 attempts (`114.319s`, `24.684s`, `71.542s`, `7.461s`, `37.614s`, `26.883s`). Code scope in `ml/evaluation/independent_cohort_diagnostic.py` starts `time.perf_counter()` before `get_engine(tool_key)`. For Arm A attempts (1, 3, 5), elapsed time includes pipeline instantiation from disk/cache into CUDA memory and initial graph warmup. For Arm B attempts (2, 4, 6), the engine is already memory-resident. Do NOT conclude inference speed of A/B from full attempt duration; isolated inference speed of A/B is not determined from full attempt time (chưa xác định tốc độ suy luận riêng của A/B từ thời gian toàn attempt).
  - Primary source bindings:
    - SDXL Inpainting: official Diffusers model card `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (pinned revision `115134f363124c53c7d878647567d04daf26e41e`) `[@sdxlInpaintingModelCard]`.
    - SD2 Inpainting: community mirror `sd2-community/stable-diffusion-2-inpainting` (pinned revision `5f74973cbb64c8568780732c17f43eb269d63a0d`; not an official Stability AI source) `[@sd2CommunityInpaintingModelCard]`.
  - Runtime environment recorded: Linux `6.6.122+-x86_64-with-glibc2.39`, Python `3.13.15`, NumPy `2.1.3`, Pillow `11.3.0`. Unrecorded in receipts: PyTorch, Diffusers, CUDA runtime version, GPU model (stated factually without speculation).
- **Methodological Bounds & Governance**:
  - Exactly 3 candidates, $n=1$ seed per candidate: observational A/B comparison. Causal mechanisms (latent tokens, context bias) remain unverified hypotheses. Distinguish visual observations from causal explanations.
  - Do NOT declare mask <3% filter or hybrid architecture as proven rules from this 3-case run.
  - Full cohort generation remains strictly **LOCKED**; zero new generation budget registered.
  - Contact sheet `diagnostic_contact_sheet.html` updated and self-contained; displays 4 separated criteria (presence, position, realism, boundary quality) and zoom overlays with target `[395, 75, 455, 125]`, mask `[350, 45, 500, 160]`, and silhouette `[375, 126, 414, 156]`; Human Content QC completed by human reviewer with formal determination **6/6 REJECT** across all diagnostic attempts. Eight historical pilot pairs (`pilot-20261008T113700Z`) remain strictly `PENDING_CONTENT_QC`. Calibration proposal `content_grounded_calibration_proposal.json` (SHA-256 `2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c`) is APPROVED for 2 attempts; generation authorized only on Colab, full cohort remains locked.

## CC. Independent cohort 6-attempt diagnostic plan human approval registered (2026-10-08)

- **Diagnostic Plan Approval & Hash Binding**:
  - Pre-approval hash verified: SHA-256 `8f2d980965a56cf8939d774e36321e583a99ba89faba4a502d59c0a25a9630e3`.
  - Human review status updated to `APPROVED` by human reviewer Dũng Phạm `<valdung04@gmail.com>` at `2026-10-08T15:17:27Z`.
  - Approved diagnostic plan SHA-256: `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157` (`research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json`).
  - Scope: Bounded 6 attempts across 3 candidates (tomato, suitcase, bird), 1 attempt per arm (Arm A: Full Canvas, Arm B: Local Crop with Padding), automatic_replacement=false.
  - Historical pilot-20261008T113700Z Human Content QC remains strictly PENDING; feathering remains unapproved for production; full cohort remains locked.
- **Sealed Input Bindings**:
  - Explicit bindings to sealed normalized PNGs from run `pilot-20261008T113700Z` (ZIP SHA-256 `3bdb1890...`, manifest `cf0e4300...`):
    - `IND_COCO_SDXL_002` (bread tomato): auth `7aefc1d1...` (`images/IND_COCO_SDXL_002_auth.png`), mask `2ff6b165...` (`masks/IND_COCO_SDXL_002_mask.png`), seed 20272319, steps 30, EulerDiscreteScheduler.
    - `IND_COMMONS_SD2_002` (suitcase): auth `98004bf7...` (`images/IND_COMMONS_SD2_002_auth.png`), mask `fee9ac7e...` (`masks/IND_COMMONS_SD2_002_mask.png`), seed 20283429, steps 50, DDIMScheduler.
    - `IND_COMMONS_SDXL_001` (sky bird): auth `7680e4ce...` (`images/IND_COMMONS_SDXL_001_auth.png`), mask `4981cedc...` (`masks/IND_COMMONS_SDXL_001_mask.png`), seed 20294438, steps 30, EulerDiscreteScheduler.
  - Remote redownload from web is strictly prohibited; runner checks both flat root (`base_dir / filename`) and standard package paths (`base_dir / images/` and `base_dir / masks/`).
- **Reframed Objective & Theoretical Grounding**:
  - Compares full-canvas inference (Arm A) vs local-crop padded inference (Arm B) across insertion omissions.
  - Does NOT claim to isolate latent downsampling alone, because cropping simultaneously modifies visual context (field of view) and token relative scale.
  - Identical PRNG seeds across different tensor resolutions ($64 \times 64$ vs $128 \times 128$) do not yield identical noise fields due to dimension-dependent sampling order in PyTorch generators.
- **Geometric Area Ratios & Coordinate Rounding Rules**:
  - SDXL candidates (tomato, bird): $(1024/256)^2 = 16.0\times$ geometric area ratio. Integer scale factor 4.0. Nearest-neighbor mask resize verified strictly binary $\{0, 255\}$.
  - SD2 suitcase candidate: $(512/320)^2 = 2.56\times$ geometric area ratio. Linear scale factor 1.6. Continuous mapping: $x \in [0.0, 360.0], y \in [236.8, 512.0]$. Nearest-neighbor sampling maps destination row $y_{dst}=236 \to$ source row 147 (0), $y_{dst}=237 \to$ source row 148 (255). Resized mask raster bbox: `[0, 237, 360, 512]`, pixel count: 99,000 px, values strictly in $\{0, 255\}$.
  - Latent pixel counts: methodologically estimated assuming 8x VAE downsampling ($13 \times 11 = 143$ vs $55 \times 45 = 2475$ for tomato; $28 \times 21 = 588$ vs $45 \times 34 = 1530$ for suitcase; $18 \times 14 = 252$ vs $75 \times 57 = 4275$ for bird), not directly measured tensor tokens.
- **Fail-Closed Diagnostic Runner & Mock Safeguards**:
  - Implemented `ml/evaluation/independent_cohort_diagnostic.py` and CLI `--mode diagnostic` in `scripts/research/run_cohort_acquisition.py`.
  - Fail-closed approval guard (`DiagnosticPlanNotApprovedError`) blocks execution before any attempt is started if status != APPROVED.
  - Preflight `--check-diagnostic-plan` validates plan hash, input hashes, and budget before run directory creation.
  - Mock engine is explicitly marked synthetic (`is_mock=True`, `is_synthetic=True`) and barred from production diagnostic (`DiagnosticEngineError` when allow_mock=False).
  - Generator is re-seeded independently per arm with candidate registered seed (no RNG state carry-over).
  - Preserves raw pre-composited generated images (`images/<arm_id>_raw_gen.png`) alongside final 512x512 composites.
  - Final compositing uses original registered canvas mask, guaranteeing bit-exact outside pixel invariance (`outside_mean_l1 == 0.000000`).
  - Hard limit of 6 attempts total; failed attempts are ledgered and count toward budget; no auto-retries or extra attempts.
  - Resume skips already consumed arms without repeating attempts.
  - Generates self-contained HTML contact sheet (`diagnostic_contact_sheet.html`) with Base64 embedded images and PENDING Content QC status.
- **Colab Handover Architecture**:
  - `notebooks/independent_cohort_acquisition_colab.ipynb` supports explicit selection `EXECUTION_MODE = "diagnostic"`.
  - Removed all guessing/fallback on `EXECUTION_MODE`; stops immediately before CLI if mode is missing or invalid.
  - Records successful execution context (`RUN_CONTEXT`) in Cell 3 for Cell 4 audit and packaging.
  - Preflight `--check-diagnostic-plan` verifies plan and Drive inputs (`RUNS_ROOT / "pilot-20261008T113700Z"`). Prioritizes reusing existing run directory on Drive if present.
  - Diagnostic results are isolated in dedicated run directory and do not enter the official independent cohort or alter historical pilot records.
- **Verification & Governance**:
  - `ml/tests/test_independent_cohort_diagnostic.py`: 11/11 PASS (100%).
  - Zero GPU runs executed in this session.
  - Human Content QC for `pilot-20261008T113700Z` remains PENDING (clarified as a distinct review decision, not a precondition for diagnostic execution).
  - Approval procedure registered: upon human review approval, commit the plan with `human_review_status = "APPROVED"` and reviewer metadata in a functional commit, then pin notebook to the resulting commit. Zero detector calls, independent performance NOT_MEASURED, full cohort NOT_RUN.

## BB. Follow-up pilot technical diagnosis, threshold audit, and remediation plan (2026-10-08)

- **Technical QC Threshold Reconciliation**:
  - Code at commit `d9d99b678053972436032a828056a76a6392fbb5` (`ml/evaluation/independent_cohort_acquisition.py`) and protocol (`CONTENT_GROUNDED_EDITING_AMENDMENT.md`) lock three explicit numerical gates:
    1. `NON_BLANK_STD_THRESHOLD = 5.0`: Image standard deviation of both authentic and edited arrays must be $\ge 5.0$.
    2. `MIN_MASKED_PIXEL_DELTA_L1 = 3.0`: Mean absolute pixel difference inside the mask (`mask == 255`) must be $\ge 3.0$.
    3. `MAX_UNMASKED_PIXEL_DELTA_L1 = 0.5`: Mean absolute pixel difference outside the mask (`mask == 0`) must be $\le 0.5$.
  - `diff_std` ($\text{std}(I_{\text{edit}} - I_{\text{auth}})$) is an exploratory descriptive statistic; no threshold exists in code or protocol.
  - Conflation resolved: earlier informal notes referencing "inside L1 >= 5.0" mistakenly merged the non-blank image std threshold (5.0) with the inside-mask L1 threshold (3.0). All 8 acquired pairs comfortably cleared `inside L1 >= 3.0` (range: 11.08 to 76.14).
- **Pipeline Transmission Audit**:
  - `prompt`: Passed directly from `spec.prompt` without automated templates or trigger tokens.
  - `negative_prompt`: Unset (`None`), relying on unconditioned null-string embedding `""`.
  - `mask polarity`: Mode `L`, `{0, 255}` (`255` = inpaint, `0` = preserve), aligned with Diffusers documentation `[@diffusersInpaintingDocs036]`. This source supports the API convention, not the content quality of internal outputs.
  - `preprocessing`: Aspect-preserving scale + center-crop to 512×512 via Lanczos (`normalize_image_to_canvas`).
  - `strength`: Unspecified (default 1.0 full denoising from $t=T$).
  - `guidance_scale`: Fixed at `7.5` for both tools.
  - `scheduler`: SD2 uses `DDIMScheduler` (50 steps); SDXL uses `EulerDiscreteScheduler` (30 steps).
  - `canvas`: 512×512 for both tools. SD2 operates at native resolution; SDXL operates below native 1024×1024.
- **Cause Classification: Proven vs Unverified Hypotheses**:
  - _Proven Software Fixes_: Fixed contact sheet reading non-existent `att['qc_details']` (now computed from RGB arrays); corrected threshold reporting.
  - _Proven Geometric / Physical Causes_: Zero-feathering 1-bit binary compositing (`Image.composite`) cutting continuous structures (railings, ceiling plaster, walls) inevitably creates 1-pixel edge discontinuities.
  - _Unverified Technical Hypotheses_:
    - Latent resolution bottleneck: assuming 8x downsampling maps small masks (tomato 3.78%, bird 6.58%) to estimated tiny grids ($13 \times 11$, $18 \times 14$ latent px), hypothesized to restrict structural object formation. `unverified-hypothesis`; latent tensors were not directly instrumented.
    - Context infill conditioning: prompts with prominent background descriptions ("on bread", "on cobblestones", "in cloudy sky") surrounded by unmasked context may bias UNet attention toward continuing background textures rather than generating salient objects. `unverified-hypothesis`; reverse sampling does not perform test-time gradient descent or loss optimization.
- **Remediation Proposals**:
  - _Object Insertion_:
    - Method A.1: Prompt token isolation (focusing on salient object features) + negative prompting (`"empty, blurry, missing object"`) + guidance tuning (9.0–11.0).
    - Method A.2: Local crop inference with padding margin $P$ at model-native resolution (1024×1024 for SDXL), downscaling with Lanczos and compositing strictly inside registered mask bbox. Evaluated risks: loss of global perspective/vanishing points, illumination misalignment, scale distortion, and resampling blur.
  - _Boundary Seam Handling_:
    - Inward-only edge feathering: Ramping $\alpha$ over $k = 2-3$ px strictly inside the registered mask ($mask == 255$), with $\alpha = 0$ outside, preserving exact `outside_mask_mean_l1 = 0.000000`. Solves 1-pixel high-frequency edge steps; cannot resolve macroscopic structural severance.
    - Natural-boundary mask alignment: Registering masks along natural architectural trim/moldings or replacing candidates with uncuttable linear foregrounds.
- **Controlled Next-Pilot Experimental Design**:
  - Structured into 4 distinct experimental arms (Arm 1: Prompt & Negative Prompt; Arm 2: Inward Feathering; Arm 3: Local Crop & Guidance; Arm 4: Natural Boundary / Candidate Substitution) to avoid confounding simultaneous variables.
  - Run `pilot-20261008T113700Z` consumed its 8 registered attempts; next pilot will be registered under a new dedicated run ID with transparent candidate tracking (new buffer candidates vs diagnostic re-attempts).
  - Scientific governance: Zero GPU generation executed; plan remains UNAPPROVED pending human review; Human Content QC for `pilot-20261008T113700Z` remains PENDING.
- **Empirical Inward Feathering (k=2 px) Evaluation**:
  - Generated derived feathered images for all 8 pairs in `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/derived_feathered_k2/` preserving 100% of outside pixels (`outside_mask_mean_l1 = 0.000000` verified 8/8).
  - Produced comprehensive self-contained HTML contact sheet: `feathering_comparison_contact_sheet.html`.
  - Visual inspection confirmed: 2-pixel cosine ramp successfully softens 1-pixel high-frequency edge steps (reducing tile seam sharpness in `IND_COCO_SD2_001` and plaster edge in `IND_COCO_SDXL_041`), with minimal ghosting. However, feathering **cannot fix macroscopic structural severance** (the 210-pixel severed railing gap in `IND_COMMONS_SDXL_003` remains severed with a blurred 2px stub; the 30% ceiling plane tonal mismatch in `IND_COCO_SDXL_041` remains obvious; internal generation blob/omission defects in `IND_COCO_SD2_001` are unaffected).
- **Registered 6-Attempt Diagnostic Plan (`content_grounded_diagnostic_plan.json`)**:
  - Registered dedicated calibration specification for the 3 omission candidates (`IND_COCO_SDXL_002` tomato, `IND_COMMONS_SD2_002` suitcase, `IND_COMMONS_SDXL_001` bird) under SHA-256 `cba87a66793351bcb2e7c21f88aeea03eab1c1f6fb7e1cc4a4483faf28b73498`.
  - Exactly 6 one-shot attempts (Arm A: Full Canvas 512x512 vs Arm B: Local Crop with Padding at native resolution).
  - Keeps identical authentic images, prompts, seeds from ledger, model revisions, guidance scales (7.5), schedulers, and steps; negative prompt and feathering are intentionally excluded to isolate latent spatial capacity.
  - Status: strictly `PENDING_USER_APPROVAL`; zero GPU execution; diagnostic results are exploratory and will not enter official cohort or overwrite pilot history.

---

## AA. Follow-up pilot intake, technical audit, and coordinate-grounded content review (pilot-20261008T113700Z) (2026-10-08)

- **Intake & Verification**:
  - Package ZIP SHA-256: `3bdb1890d2aa6d34bb2829e159408ef44ab30d630c6f85792fce5fb7ade9e2b4` (6,129,778 bytes; 34 archive members verified safe against path traversal). Original ZIP preserved untouched.
  - Binding commit: `d9d99b678053972436032a828056a76a6392fbb5`; approved plan SHA-256: `eeace0e57a34f9f3824a3ce1a52bfa374670dca3cc19c668554d62257141632d`.
  - Manifest SHA-256: `cf0e4300d1e6f760e82ab76fdd29ffd534b35577003ae8ccf4ef9a9c25799704`.
  - Production CLI `--audit-run` PASS: 8 attempts, 8 accepted pairs, 0 errors, 2 per stratum, 1 attempt/candidate, `automatic_replacement=false`.
- **Ledger Semantics Clarification**:
  - In `attempt_ledger.jsonl`, status `ACCEPTED` represents strictly **Technical QC Acceptance** (Step D/E in pipeline: 512x512 RGB canvas, binary L mask, non-blank, outside L1=0, inside L1 >= 3.0, non-blank image std >= 5.0, disjointness verified).
  - Canonical **Human Content QC** is tracked separately in `provenance_ledger.jsonl` and `run_receipt.json` as `PENDING_CONTENT_QC`.
- **Exact Pixel Metrics (Canonical Pipeline Formula)**:
  - Outside-mask mean L1: `0.000000` across all 8 pairs (achieved by construction via binary mask compositing `Image.composite(gen, auth, mask)`; not evidence of raw diffusion preserving outside pixels).
  - Inside-mask mean L1 verified across ZIP, extracted files, and HTML Base64 embeds:
    - `IND_COCO_SD2_001`: `37.228664` (diff std: 9.74)
    - `IND_COCO_SD2_002`: `44.094597` (diff std: 34.91)
    - `IND_COCO_SDXL_041`: `30.664345` (diff std: 24.06)
    - `IND_COCO_SDXL_002`: `16.786127` (diff std: 4.21)
    - `IND_COMMONS_SD2_001`: `54.337215` (diff std: 34.84)
    - `IND_COMMONS_SD2_002`: `30.735847` (diff std: 14.92)
    - `IND_COMMONS_SDXL_001`: `11.076676` (diff std: 3.62)
    - `IND_COMMONS_SDXL_003`: `76.143341` (diff std: 59.68)
  - All 8 pairs PASS technical quality thresholds.
- **Agent Visual Content Findings (Coordinate-Verified Advisory Screening)**:
  1. `IND_COCO_SD2_001` (clock replaces pan): Murky metallic/glass blob with harsh square boundary seam at $y=95, x=280$; inside wall color is cooler/lighter than outside; recommendation **`REJECT`**.
  2. `IND_COCO_SD2_002` (cabinet remodel): **Only upper cabinets and range hood** ($y \in [75, 323], x \in [95, 415]$) remodeled to matte navy blue; **lower cabinets and counter** ($y > 323$) remain 100% authentic honey-oak wood; minor synthetic gloss on right cabinet; recommendation **`NEEDS_REVIEW`**.
  3. `IND_COCO_SDXL_041` (chandelier): Crystal pendants terminate neatly above $y=140$ ($y \in [138, 140], x \in [180, 420]$) and are **not cut off**; horizontal seam at $y=155$ is the newly painted smooth white ceiling plaster/soffit cutting abruptly into darker authentic textured drywall; valance contact at $y=155, x \in [12, 25]$ is subtle; recommendation **`NEEDS_REVIEW`**.
  4. `IND_COCO_SDXL_002` (bread tomato): 0 tomatoes generated; infilled with blurry bread crumb texture; recommendation **`REJECT`**.
  5. `IND_COMMONS_SD2_001` (Option A headland removal): Original pine headland removed, but **not converted 100% to sea**; SD2 generated a newly synthesized coastal landscape: steep green grassy hillside ($x \in [380, 512], y \in [305, 480]$), sea stacks ($x \in [380, 430], y \in [340, 440]$), low reefs ($x \in [340, 390], y \in [450, 480]$), and foreground rocky mound ($x \in [200, 310], y \in [470, 512]$); distant town buildings in mask replaced by dark rock; recommendation **`NEEDS_REVIEW`**.
  6. `IND_COMMONS_SD2_002` (suitcase): 0 suitcases generated; infilled with cobblestones; recommendation **`REJECT`**.
  7. `IND_COMMONS_SDXL_001` (bird): 0 birds generated; infilled with purplish-gray sky patch; recommendation **`REJECT`**.
  8. `IND_COMMONS_SDXL_003` (column replacement): Severe semantic drift; generated psychedelic bottle-shaped pillar with neon reflections and severed railing at $x=170, 380$; recommendation **`REJECT`**.
- **Technical Diagnosis (Code & Model Behavior)**:
  - _Proven Cause (Boundary Seams)_: Zero-feathering 1-bit compositing (`Image.composite`) with axis-aligned boxes cutting continuous scene geometry causes immediate 1-pixel color/texture steps.
  - _Proven Cause (Contact Sheet Display Bug)_: Previous script referenced non-existent `att['qc_details']` key in `attempt_ledger.jsonl`, defaulting to `0.0`. Fixed by computing L1 directly from RGB arrays.
  - _High-Probability Hypotheses (Small Insertion Deficits)_: Latent space downsampling (8x) gives tiny spatial capacity (e.g., $13 \times 11$ latent px for tomato, $18 \times 14$ for bird); combined with strong background prompt conditioning ("bread", "cobblestones", "cloudy sky"), UNet inpainting strongly prioritizes context infilling over object synthesis.
- **Human Content QC Status**: Strictly **`PENDING_CONTENT_QC`** awaiting user per-pair determination.
- **Review Artifact**: Self-contained contact sheet at `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/content_qc_contact_sheet.html`.
- **Scientific boundary**: Detector calls remain 0; independent performance `NOT_MEASURED`; full cohort `NOT_RUN`.

---

## Z. Follow-up pilot plan approval by human reviewer (2026-10-08)

- **Formal human approval recorded**: User reviewed the complete 8-instruction follow-up pilot plan (pre-approval hash `0614fd5fd316e91df545fe940a853fea3bfb9de34142a1f2b8c6c3ae2d0f6fa3`) and formally approved execution:
  - Reviewer identity: `Dũng Phạm <valdung04@gmail.com>`.
  - Approval timestamp: `2026-10-08T07:34:38Z`.
  - Approved plan SHA-256: `eeace0e57a34f9f3824a3ce1a52bfa374670dca3cc19c668554d62257141632d` (`content_grounded_pilot_plan.json`).
- **Explicit authorizations**:
  1. `IND_COCO_SDXL_041` substituting candidate 001 (chandelier insertion), with accepted minor localized boundary contact with curtain valance apex at $x \in [12, 25], y=155$.
  2. Option A `[190, 305, 512, 512]` (25.426483%) for `IND_COMMONS_SD2_001`, with accepted inpainting infill/regeneration of shoreline structures within mask.
  3. `IND_COMMONS_SDXL_003` substituting candidate 002 ("Tower Song" Ted Bieler isolated column replacement), with accepted inpainting alteration risk for background trees/river/railing within $x \in [170, 380]$.
  4. Remaining five instructions approved according to plan: `IND_COCO_SD2_001` (pan), `IND_COCO_SD2_002` (cabinets/hood), `IND_COCO_SDXL_002` (tomato), `IND_COMMONS_SD2_002` (suitcase), `IND_COMMONS_SDXL_001` (bird).
- **Execution bounds**: Maximum 8 one-shot attempts (2 per stratum), 1 attempt/candidate, `automatic_replacement=false`.
- **Scientific boundary**: Approval applies strictly to the plan; generated images remain subject to Technical QC and Human Content QC upon acquisition. Zero generation in local session; zero training, detector calls, evaluator execution, PR, or merge to main.

---

## Y. Follow-up pilot instruction resolution, optical boundary review, and exact area synchronization (2026-10-08)

- **Instruction resolution and candidate proposals**: Resolved the three open follow-up instructions on authentic normalized 512×512 images with honest geometric disclosures:
  1. `IND_COMMONS_SD2_001`: Proposed Option A mask `[190, 305, 512, 512]` ($322 \times 207 = 66,654$ px = **25.426483%**, strictly `medium_10_to_30pct`) unifies with target bbox `[190, 308, 512, 512]` and envelops 100% of registered foreground headland, cliff base, offshore sea stacks ($x \ge 190$), and pine canopy ($y \ge 308$). Disclosed that the concave coastline forces inclusion of bay water ($x \in [190, 340], y \in [305, 340]$) and lower hotel/town edge ($x \in [440, 512], y \in [305, 335]$), which will be infilled/regenerated by inpainting. Option B (`[220, 340, 512, 512]`, 19.158936%) is documented for reference only and does not satisfy full removal because it severs pine crowns ($y \in [308, 340]$) and leaves offshore rocks ($x \in [190, 220]$) stranded in sea; historical mask `[220, 285, 512, 512]` area corrected to 25.285339% (not 27.88%).
  2. `IND_COCO_SDXL_041`: Substituted blocked cat-in-sink candidate `IND_COCO_SDXL_001` with candidate `IND_COCO_SDXL_041` (living room ceiling chandelier, origin `coco:189310`, author `an iconoclast`, CC BY 2.0; pool index 40) within `coco_sdxl`, preserving `sdxl_inpainting`, `object_insertion`, and `large_over_30pct`. Mask `[0, 0, 512, 155]` ($512 \times 155 = 79,360$ px = **30.273438%**, strictly `large_over_30pct`) and target bbox `[180, 15, 332, 140]` clear ceiling-wall line ($y \ge 173$) and living room furniture; pixel analysis confirms localized boundary contact at $y=155$ with the highest ornamental curve of left curtain valance across $x \in [12, 25]$ (contact depth ~3-4 px from fold apex $y \approx 151$), registered explicitly for human decision.
  3. `IND_COMMONS_SDXL_003`: Substituted intertwined adult/child limbs candidate `IND_COMMONS_SDXL_002` with candidate `IND_COMMONS_SDXL_003` ("Tower Song" Ted Bieler isolated vertical sculpture column, CC BY-SA 4.0; pool index 2) within `commons_sdxl`, preserving `sdxl_inpainting`, `object_replacement`, and `large_over_30pct`. Registers visible vertical shaft cropped by canvas boundaries (`target_bbox: [190, 0, 360, 512]`); mask `[170, 0, 380, 512]` ($210 \times 512 = 107,520$ px = **41.015625%**) completely eliminates limb confounds, while documenting explicit inpainting alteration risk for background trees/river/railing within $x \in [170, 380]$ (outside preserved by compositing). Option 2B retaining candidate 002 corrected to 30.212402% (not 30.8716%).
- **Exact coordinate-derived area synchronization**: Synchronized exact areas across plan, amendment, report, and contact sheet:
  - Pan `IND_COCO_SD2_001` `[195, 95, 280, 205]`: $85 \times 110 = 9,350$ px = **3.566742%** (`small_under_10pct`).
  - Kitchen upper-cabinets/range-hood `IND_COCO_SD2_002` `[95, 75, 415, 323]`: $320 \times 248 = 79,360$ px = **30.273438%** (`large_over_30pct`).
  - Chandelier `IND_COCO_SDXL_041` `[0, 0, 512, 155]`: $512 \times 155 = 79,360$ px = **30.273438%** (`large_over_30pct`).
  - Bread tomato `IND_COCO_SDXL_002` `[345, 245, 455, 335]`: $110 \times 90 = 9,900$ px = **3.776550%** (`small_under_10pct`).
  - Headland `IND_COMMONS_SD2_001` (Option A) `[190, 305, 512, 512]`: $322 \times 207 = 66,654$ px = **25.426483%** (`medium_10_to_30pct`).
  - Suitcase `IND_COMMONS_SD2_002` `[0, 340, 225, 512]`: $225 \times 172 = 38,700$ px = **14.762878%** (`medium_10_to_30pct`).
  - Monument sky bird `IND_COMMONS_SDXL_001` `[350, 45, 500, 160]`: $150 \times 115 = 17,250$ px = **6.580353%** (`small_under_10pct`).
  - Column `IND_COMMONS_SDXL_003` `[170, 0, 380, 512]`: $210 \times 512 = 107,520$ px = **41.015625%** (`large_over_30pct`).
- **Self-contained HTML dossier**: Regenerated `next_pilot_edit_plan_contact_sheet.html` with all 8 authentic base64 PNGs, SVG overlays, summary table, and three zoomed-in boundary inspection crops (curtain valance apex at $y=155$, shoreline structures at $y=305$, and offshore rocks at $x=190$ vs $x=220$). `human_review_status` remains `PENDING`.
- **Scientific boundary**: No image generation, training, detector execution, evaluator scoring, or automatic candidate replacement occurred.

---

## X. Pilot dossier geometry correction and test-temp cleanup (2026-10-08)

- Recomputed the exact normalized-canvas rectangular mask for every proposed instruction: `IND_COCO_SD2_001` 3.566742%, `IND_COCO_SD2_002` 30.273438% after correction, `IND_COCO_SDXL_001` 34.332275%, `IND_COCO_SDXL_002` 3.776550%, `IND_COMMONS_SD2_001` 25.285339%, `IND_COMMONS_SD2_002` 14.762878%, `IND_COMMONS_SDXL_001` 6.580353%, and `IND_COMMONS_SDXL_002` 30.212402%.
- The former `IND_COCO_SD2_002` bbox `[95,75,415,315]` occupied only 29.296875% and contradicted `large_over_30pct`. The corrected `[95,75,415,323]` bbox occupies 30.273438%; the added 8-pixel strip stays in the same backsplash/upper-fixture context and above the foreground bowl. No quota label or QC threshold was relaxed.
- Registered rectangular plans now fail closed against protocol ranges (1–10%, 10–30%, 30–50%) rather than borrowing the broader Technical QC raster tolerance. A regression reproduces and rejects the former 29.296875% large bbox.
- `IND_COCO_SDXL_001` remains unresolved: the current image has no safe region that is both a plausible insertion support and large-class mask. Recommended handling is a separately reviewed source-candidate substitution inside `coco_sdxl` that preserves tool, modification, mask class, stratum quota, budget, and one-attempt policy; no replacement candidate was selected automatically.
- `IND_COMMONS_SDXL_002` remains a human risk decision: its child-statue target touches the adult statue's hands and arms, so diffusion can alter adult anatomy, pose, or perceived identity inside the registered mask. Compositing only guarantees the outside-mask pixels by construction.
- Regenerated the existing `next_pilot_edit_plan_contact_sheet.html` in place with eight embedded authentic PNGs, eight overlays, an eight-row decision table, and exact mask percentages. It has zero relative image references and remains `human_review_status=PENDING`.
- Traced six stale direct-child pytest basetemp roots to explicit workspace `--basetemp` commands and synthetic test fixtures. They contained 9,916 files, 873 directories, and 2,375,435,081 bytes. No tracked fault-injection test was found to modify ACL. Only those six exact verified roots had inheritance restored recursively before deletion; parent ACL and all real artifacts remained untouched.
- Added marker-, prefix-, and direct-parent-locked pytest session cleanup in `finally`. Regression tests refuse unmarked and real-run-like names and demonstrate cleanup after a deliberately failing session as well as normal success.
- Reverified sealed real artifacts after cleanup: ZIP SHA-256 `1fcd1de57373c7250583d87a4a4050a91d918b8bd62ec105d4aab737d340053d`; manifest SHA-256 `caa0766addff4d87d9d11206f1e398a29ca04c58f0013866fac749fac09bc416`; attempt and provenance ledgers remain present.
- Verification: 83 related tests PASS / 2 artifact-network-gated skips; post-pin binding/notebook suite 57 PASS / 2 artifact-network-gated skips. Both marked workspace basetemp roots were absent afterward. Python compile, continuity checker, and `git diff --check` pass.
- No generation, training, detector, evaluator, full acquisition, or Human Content QC approval occurred.

---

## W. Content diagnosis and controlled corrective preparation (2026-10-07)

- Follow-up review on 2026-10-08 rechecked all eight planned instructions against the normalized authentic images. Five target/mask/prompt records were tightened or reframed, three remained aligned, and `IND_COCO_SDXL_001` is explicitly `NEEDS_USER_DECISION` because its large insertion rectangle overlaps the cat and sink; generation remains blocked.
- The existing `next_pilot_edit_plan_contact_sheet.html` outside Git was regenerated in place as a self-contained dossier with 8 embedded authentic PNGs, an 8-row review table, and no relative image references. `human_review_status` remains `PENDING`; agent instruction screening is not Human Content QC.
- Plan/runtime guards now enforce exactly 8 one-shot attempts, 2 per stratum, matching aggregate modification/mask quotas, and `automatic_replacement=false`. Resume cannot retry a ledgered candidate; full mode stops before run setup. Focused verification: 80 PASS / 2 artifact-network-gated skips.
- Inspected all 8 authentic–mask–edited tuples at 512×512 and reconciled prompts, mask geometry, crop transforms, and recorded engine configuration. Model revisions, schedulers, steps, guidance, and output dimensions match their model-specific protocol.
- Demonstrated that prompts came from a generic index and masks from seeded random ellipses on the normalized canvas. No source target coordinates existed, so crop/resize drift is not demonstrated; missing target registration is the proven defect.
- Measured outside-mask mean L1 3.808–9.311 on all 8 pairs versus locked maximum 0.5. Fixed future execution by compositing only inside the binary mask and enforcing locked std/masked/unmasked thresholds (5.0/3.0/0.5). An outside-mask L1 of exactly 0 after this step is by construction because authentic pixels are copied there; it is not evidence that the raw diffusion output preserved those pixels.
- Added fail-closed content-grounded instructions (prompt, target, rationale, target bbox, mask bbox), a maximum-eight-attempt amendment/plan preserving the pilot's source/tool/modification/mask quotas, and a review-only overlay contact sheet outside Git. There are no automatic replacements, and the CLI refuses generation while human review status remains `PENDING`.
- Standardized new run storage to `MyDrive/Colab Notebooks/forensics-web-lab/independent_cohort_acquisition/runs/<RUN_ID>` while preserving and documenting both historical phase-coded roots.
- Corrected the old incident wording from 110 SDXL attempts to 220 SDXL attempt records / 110 unique candidates. The original ledger is not locally available, so no unsupported explanation for the two records per candidate is asserted.
- Verification before commit: acquisition tests 21/21 PASS; binding/notebook tests 56 PASS / 2 artifact-gated skips; hermetic smoke 6/6 PASS; 440-candidate allocation verification PASS; full target rejects the 8-entry plan; `pnpm continuity:check` PASS.
- No generation, full acquisition, training, detector scoring, or evaluator execution occurred. All existing Content QC statuses remain `PENDING_CONTENT_QC`.

Evidence: `PILOT_CONTENT_DIAGNOSIS.md`, `CONTENT_GROUNDED_EDITING_AMENDMENT.md`, `content_grounded_pilot_plan.json`, and `docs/RESEARCH_WORKSTREAM_INDEX.md`.

---

## V. Real pilot intake and audit (`pilot-20261007T132003Z`)

- ZIP SHA-256 `1fcd1de57373c7250583d87a4a4050a91d918b8bd62ec105d4aab737d340053d`; archive paths passed traversal/absolute-path/duplicate/symlink checks and were extracted beside the untouched ZIP without overwrite.
- Exact production audit at binding commit `7d2eea4027e2a17b51e6665ff81d481e4e333d48`: `PASS`, 8 pairs, manifest SHA-256 `caa0766addff4d87d9d11206f1e398a29ca04c58f0013866fac749fac09bc416`.
- Direct attempt-ledger result: 8 attempts, 8 accepted, 0 rejected/failed, no rejection reasons; exactly 2 accepted pairs per stratum.
- Contract verification: 16/16 images are 512x512 RGB; 8/8 masks are 512x512 `L`, binary `{0,255}`, and within declared area brackets; hashes and inventories match receipts, manifest, and ledgers.
- Agent-only visual screening: concerns on all 8 pairs (7 likely reject, 1 inconclusive/recommend reject). Canonical statuses remain `PENDING_CONTENT_QC`; no agent observation is represented as human approval.
- Full acquisition gate: **LOCKED pending human Content QC**. Detector calls remain 0; full acquisition, training, and evaluator were not run. Summary: `PILOT_AUDIT_SUMMARY.md`; detailed artifacts and per-pair notes remain outside Git.

---

## U. Revision 5 — Generation Contract Enforcement & Systematic Failure Gate Hotfix (2026-10-07)

### U1. Incident Report & Root Cause Analysis (`pilot-20261007T093824Z`)

- **Sự cố thực địa**: Pilot run `pilot-20261007T093824Z` (commit `cdacb41a5501568c1a11435dfcb00d327bc69a6d`) trên Google Colab T4 GPU dừng với lỗi:
  `StratumQuotaDeficitError: Stratum 'coco_sdxl' exhausted all candidates without reaching quota: 0/2 valid pairs acquired.`
- **Phân tích Ledgers**:
  - `provenance_ledger.jsonl`: Có đúng 2 records thuộc stratum `coco_sd2` (đều pass technical QC, resolution 512x512, status `ACCEPTED`).
  - `attempt_ledger.jsonl`: Có 222 records tổng cộng.
  - Phân tích theo stratum và tool:
    - Stratum `coco_sd2` (`stable_diffusion_2_inpainting`): 2 attempts đều `ACCEPTED` (đạt quota 2/2).
    - Stratum `coco_sdxl` (`sdxl_inpainting`): 220 attempt records trên 110 unique candidates đều ghi cùng lý do (đối soát từ 222 tổng records trừ 2 `coco_sd2` accepted records; ledger gốc không có trong local artifacts hiện tại để xác định vì sao mỗi candidate có hai records):
      `QC_FAILED — Edited image size/mode invalid: (1024, 1024), RGB`.
- **Root Cause Kỹ thuật**:
  - Trong `DiffusersInpaintingEngine.inpaint()`, `self.pipeline(...)` được gọi mà không truyền tham số `height` và `width`.
  - Trong thư viện `diffusers`, `StableDiffusionXLInpaintPipeline.__call__` tính toán:
    `height = height or self.default_sample_size * self.vae_scale_factor`
    Với SDXL: `default_sample_size = 128`, `vae_scale_factor = 8` $\to 128 \times 8 = 1024$. Do đó pipeline sinh ảnh $1024 \times 1024$ mặc dù ảnh đầu vào và mask là $512 \times 512$.
    Với SD2: `default_sample_size = 64`, `vae_scale_factor = 8` $\to 64 \times 8 = 512$. Do đó SD2 tình cờ sinh đúng $512 \times 512$.
  - Lỗi generation contract hệ thống này bị hàm `evaluate_technical_qc` trả về `False, "Edited image size/mode invalid: (1024, 1024), RGB"`, khiến runner ghi nhận `status: QC_FAILED` và tiếp tục qua toàn bộ 110 unique candidates của `coco_sdxl` trước khi dừng với `StratumQuotaDeficitError`; incident transcription ghi 220 SDXL attempt records.

### U2. Giải pháp Generation Contract & Systematic Failure Gate

1. **Khóa Kích thước Tường minh**:
   - Thêm `"target_height": 512, "target_width": 512` vào `INPAINTING_MODEL_REGISTRY` cho cả hai model SD2 và SDXL.
   - `DiffusersInpaintingEngine.inpaint()` nhận tường minh `height=512, width=512` và truyền trực tiếp vào `self.pipeline(..., height=height, width=width, ...)`.
   - Tuyệt đối không resize ảnh 1024 xuống 512 để vượt QC; pipeline phải tạo sinh trực tiếp ở kích thước 512x512.
2. **Kiểm tra Contract Canvas Trước và Sau Inpainting**:
   - Trước inpainting: kiểm tra `auth_canvas` (512x512 RGB) và `mask_canvas` (512x512 L). Vi phạm ném `GenerationContractError`.
   - Sau inpainting: kiểm tra `edited_canvas` (512x512 RGB). Vi phạm ném `GenerationContractError`.
3. **Systematic Failure Gate (Ngăn Cháy Pool Ứng Viên)**:
   - Phân biệt triệt để lỗi contract/cấu hình hệ thống với lỗi content QC của từng ứng viên:
     - Vi phạm generation contract (`GenerationContractError`): Ghi attempt `status: GENERATION_CONTRACT_ERROR`, xuất `out_path / "failure_receipt.json"`, ghi log lỗi với full traceback và **dừng ngay lập tức** mà không thử tiếp ứng viên khác.
     - Content QC rejection hợp lệ (ảnh đen, biến thiên thấp, mask ratio ngoài ngưỡng): Ghi `status: QC_FAILED`, tiếp tục cơ chế thay thế ứng viên trong stratum để đảm bảo quota.
   - `audit_acquisition_run` fail-closed: Từ chối ngay lập tức nếu phát hiện `failure_receipt.json`.
4. **Logging Bền vững & Colab Notebook**:
   - Tạo file log bền vững `run_dir / "acquisition.log"` ghi nhận timestamp, candidate id, stratum id, trạng thái và full exception traceback.
   - Hàm `run()` trong Colab notebook dùng `subprocess.Popen(..., stderr=subprocess.STDOUT)` stream trực tiếp cả stdout và stderr lên giao diện Colab.
   - Cập nhật đường dẫn mặc định: `/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c7b/runs/<RUN_ID>`.
   - Xóa cờ hoàn tất cũ `globals().pop("PILOT_RUN_COMPLETED", None)` tại đầu Cell 3 trước mỗi lần chạy; chỉ audit và đóng gói ZIP khi lần chạy hiện tại thành công.
   - Bảo toàn nguyên vẹn thư mục run cũ `pilot-20261007T093824Z` tại `MyDrive/forensics-web-lab/phase_4c7b_runs/`.

### U3. Kết quả Kiểm thử & Xác minh

- **Regression test suite**: Bổ sung 4 targeted tests trong `ml/tests/test_independent_cohort_bindings.py`:
  - `test_diffusers_engine_passes_explicit_height_width_and_validates_dimensions`: kiểm tra pipeline nhận đúng `height=512, width=512`, từ chối input/output sai kích thước.
  - `test_output_1024_halts_immediately_and_does_not_try_next_candidate`: kiểm tra output 1024 dừng ngay sau ứng viên đầu tiên (call count = 1), ghi failure receipt và attempt ledger `GENERATION_CONTRACT_ERROR`.
  - `test_output_correct_size_proceeds_through_qc_and_handles_content_qc_rejection`: kiểm tra output 512x512 đi tiếp qua QC, lỗi content QC tiếp tục thay thế ứng viên hợp lệ.
  - `test_generation_contract_failure_prevents_completion_receipt_and_blocks_audit`: kiểm tra run thất bại không tạo completion receipt, bị `audit_acquisition_run` từ chối và chặn đóng gói ZIP.
- **Toàn bộ test suites**: 75/75 tests PASS trong `ml/tests/test_independent_cohort_bindings.py` (58/58) và `ml/tests/test_independent_cohort_acquisition.py` (17/17).
- **Hermetic smoke test**: PASS trong 1.37s (`acquisition_smoke_receipt_v2.json`).
- **Plan verification**: PASS (440 candidates, 110 per stratum, 0 historical overlap).
- **Model preflight**: PASS cho cả hai model.
- **Tính trung thực khoa học**: Real generation pilot `NOT_RUN` ở local (do không có GPU); các chỉ số detector `NOT_MEASURED`.

---

## T. Revision 4 — Protocol Amendment v1.3.1 Model Source Hotfix, Community Mirror Qualification & Fail-Closed Preflight (2026-10-07)

### T1. Incident Report & Empirical Reproduction

- **Sự cố thực địa**: Pilot run `pilot-20261007T082516Z` (commit `3710762868a007af4fe79798bad79d086abcd5c8`) trên Google Colab T4 GPU vượt qua GPU policy check nhưng dừng tại:
  `HfApi().model_info("stabilityai/stable-diffusion-2-inpainting")`
  với ngoại lệ `RepositoryNotFoundError` / HTTP 401 Unauthorized (`{"error":"Invalid username or password."}`).
- **Nguyên nhân kỹ thuật**: Stability AI đã chuyển repo `stabilityai/stable-diffusion-2-inpainting` sang chế độ gated/restricted/deprecated. Hugging Face API trả về HTTP 401 đối với các truy vấn ẩn danh không token để tránh rò rỉ sự tồn tại của private/gated repo.
- **Trạng thái thực tế**: Attempt ledger trên đĩa được xác nhận là trống; 0 receipts được ghi nhận. Thư mục run cũ và log lỗi được bảo toàn nguyên vẹn, không chỉnh sửa binding cũ để gượng ép `--resume`.
- **Tái hiện thực nghiệm**: Tái hiện thành công qua API metadata mà không sinh ảnh cục bộ; xác nhận phân biệt giữa lỗi mạng, repo 404, repo gated/private 401 và token không hợp lệ. Không yêu cầu người dùng nhập token hoặc in token vào log/chat.

### T2. Thẩm định Community Mirror & Protocol Amendment v1.3.1

- **Thẩm định nguồn thay thế `sd2-community/stable-diffusion-2-inpainting`**:
  - Đọc model card, license, model configs và Git LFS hashes.
  - Giấy phép: `openrail++` (CreativeML OpenRAIL++).
  - Pipeline class: `StableDiffusionInpaintPipeline`.
  - Configs: `model_index.json`, `unet/config.json` (sample_size 64, in_channels 9, out_channels 4), `scheduler/scheduler_config.json` (DDIM).
  - Revision đầy đủ: `5f74973cbb64c8568780732c17f43eb269d63a0d`.
  - LFS OIDs & file sizes đã thẩm định:
    - `unet/diffusion_pytorch_model.fp16.safetensors`: `29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8` (1,731,933,536 bytes).
    - `512-inpainting-ema.safetensors`: `b29e2ed9a8fe58e76f7e801bda091d23738bd74c1da3f339bcbe2d40922fcb60` (5,214,662,094 bytes).
    - `vae/diffusion_pytorch_model.fp16.safetensors`: `3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a` (167,335,342 bytes).
    - `text_encoder/model.fp16.safetensors`: `681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668` (680,821,096 bytes).
  - **Định danh minh bạch**: Được ghi nhận là `community_mirror` trong tài liệu và specification `[@sd2CommunityInpaintingModelCard]`. Tuyệt đối không gọi là nguồn chính thức và không tuyên bố trọng số bit-exact mà không có bằng chứng đối sánh trực tiếp.
  - **Bảo toàn giao thức**: Giữ nguyên kiến trúc UNet 9-channel inpainting, DDIM scheduler, 50 inference steps, guidance scale 7.5, fixed seeds, exact prompts và quota ma trận $2 \times 2$ (100 cặp mục tiêu / 110 pool cho mỗi stratum).
  - **SDXL Inpainting**: Thẩm định nguồn chính thức `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (full revision `115134f363124c53c7d878647567d04daf26e41e`, UNet fp16 LFS `6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f`) `[@sdxlInpaintingModelCard]`.
  - **Bất biến**: Không thay thế bằng SD1.5 hoặc model khác; không cho phép fallback tự động giữa các model.

### T3. Cơ chế Preflight Gate & Load Binding

- **Hàm preflight**: `verify_model_access_preflight(tool_keys, timeout, http_opener)` kiểm tra metadata API, 3 config files chính và gửi HTTP HEAD request tới safetensors weight files.
- **Fail-closed boundary**: Chạy strictly TRƯỚC KHI tạo thư mục run trên đĩa (`run_dir = Path(args.output_root) / args.run_id`). Nếu preflight thất bại, tiến trình dừng ngay lập tức và 0 thư mục rác/mồ côi được tạo ra.
- **CLI flag**: Bổ sung `--check-models` cho phép kiểm tra độc lập nhanh chóng (< 3 giây, 0 byte weights tải về máy).
- **Revision binding**: Revision được resolve trong preflight và chuyển trực tiếp vào `DiffusersInpaintingEngine`, đảm bảo nạp đúng chính xác revision đã kiểm tra.

### T4. Kết quả Kiểm thử & Xác minh

- **Regression test suite**: Bổ sung 8 tests trong `ml/tests/test_independent_cohort_bindings.py`:
  - HTTP 401 handling (`ModelPreflightError` phân loại `AUTHENTICATION_OR_GATED_REPO`).
  - HTTP 404 handling (`REPOSITORY_NOT_FOUND`).
  - Lỗi mạng (`NETWORK_OR_TIMEOUT_ERROR`).
  - Gated repo detection.
  - Live mirror và SDXL resolution.
  - Diffusers engine binding kiểm tra revision.
  - Fail-closed preflight ngăn chặn tạo thư mục run khi thất bại.
  - CLI `--check-models` thực thi thành công.
- **Tổng kết test**: 71/71 tests PASS (`test_independent_cohort_bindings.py`: 54/54, `test_independent_cohort_acquisition.py`: 17/17).
- **Smoke test**: `python scripts/research/run_cohort_acquisition.py --smoke-test` PASS trong 2.58s (`acquisition_smoke_receipt_v2.json`).
- **Plan verification**: `python scripts/research/run_cohort_acquisition.py --verify-plan` PASS (440 candidates, 110/stratum).
- **Colab Launcher**: Cập nhật Cell 2 bổ sung `--check-models` và ghim notebook vào commit functional mới.

---

## S. Revision 3 — Protocol Amendment v1.3 & Local Verified Catalog Build (2026-10-07)

### S1. Protocol Amendment v1.3 Resolution

- **Thay thế Unsplash Lite bằng Wikimedia Commons**: Do điều khoản Unsplash Lite Dataset Terms §2.A/§3.A–B chỉ cho phép download/store cho nghiên cứu nội bộ và cấm tái phân phối dữ liệu, việc tạo và công bố tập dữ liệu chỉnh sửa (inpainting) không được bảo hộ. Protocol Amendment v1.3 chính thức thay thế Unsplash Lite bằng ảnh chụp từ **Wikimedia Commons** (`Category:Quality_images`).
- **Nguồn COCO 2017 Clean val split**: Giữ nguyên nguồn COCO với điều kiện kiểm tra creator và license qua Flickr oEmbed live API. Trích xuất metadata từ `captions_val2017.json` (chứa đầy đủ `flickr_url` và Flickr photo ID; phân vùng `test2017` không chứa trường này). Loại bỏ toàn bộ 684 Option P IDs (còn 4,316 ảnh hoàn toàn disjoint).
- **Bốn Strata đối xứng**:
  1. `coco_sd2`: 100 cặp mục tiêu (110 pool)
  2. `coco_sdxl`: 100 cặp mục tiêu (110 pool)
  3. `commons_sd2`: 100 cặp mục tiêu (110 pool)
  4. `commons_sdxl`: 100 cặp mục tiêu (110 pool)
     Tổng cộng: 400 cặp mục tiêu chính thức, 440 candidates trong buffer pool.
- **Bảo toàn hạn ngạch (Quota Preservation)**: 40% replacement (160/176), 30% removal (120/132), 30% insertion (120/132); 30% small (120/132), 40% medium (160/176), 30% large (120/132).
- **Bất biến Zero Detector Scoring**: Tuyệt đối không gọi hay suy luận bất kỳ mô hình detector nào trong quá trình tuyển chọn hoặc tạo dữ liệu.

### S2. Kết quả Chạy Production Catalog Builder v2 ở Local

Chạy `scripts/research/build_verified_candidate_catalog.py` qua mạng local có checkpointing tại `artifacts/catalog_checkpoints/`:

- **COCO 2017**: Dùng `RemoteZipFile` range-based streaming đọc trực tiếp `captions_val2017.json` (chỉ tải 805 KB thay vì toàn bộ archive annotations). Thẩm tra 220 ứng viên hợp lệ qua Flickr oEmbed live API với bounded retry và rate limiting (loại các ảnh bị xóa/private/lỗi license).
- **Wikimedia Commons**: Truy vấn MediaWiki API `Category:Quality_images`, đọc `imageinfo` extmetadata, hỗ trợ continuation token (`gcmcontinue`), lọc giấy phép CC BY / CC BY-SA, loại NoDerivs. Thêm header `User-Agent` chuẩn MediaWiki policy giải quyết lỗi HTTP 403 Forbidden.
- **Tổng kết catalog**:
  - Đạt chính xác **220 COCO + 220 Wikimedia Commons = 440 eligible candidates** trong `research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json`.
  - Ghi nhận 221 excluded candidates kèm lý do chi tiết (CREATOR_UNVERIFIED, LICENSE_ND_NO_EDITED_EXPORT, NO_CC_LICENSE_AT_SOURCE, HTTP_ERROR, v.v.).
- **Xác minh Kế hoạch**: Chạy `python scripts/research/run_cohort_acquisition.py --verify-plan` đạt kết quả **PASS** (440 candidates, đúng 110 per stratum, 0 historical overlap). Xuất file kế hoạch `research/evidence/phase-4c.7b/candidate_acquisition_plan_v2.json`.

### S3. Thực nghiệm Tải và Decode Mẫu Thật Ngoài Git

Thực hiện tải và chuẩn hóa canvas thật trên 4 mẫu ảnh đại diện cho 4 strata tại `artifacts/pilot_download_test/`:

- `IND_COCO_SD2_001` (COCO): Pot Noodle, Attribution License, 200,576 bytes, tỷ lệ mask 0.0773 (small).
- `IND_COCO_SDXL_001` (COCO): mike ambs, CC BY-NC-SA, 106,547 bytes, tỷ lệ mask 0.4451 (large).
- `IND_COMMONS_SD2_001` (Commons): Moahim, CC BY-SA 4.0, 4,205,835 bytes, tỷ lệ mask 0.2397 (medium).
- `IND_COMMONS_SDXL_001` (Commons): Crisco 1492, CC BY-SA 4.0, 3,355,898 bytes, tỷ lệ mask 0.0700 (small).
  Tất cả 4 mẫu giải mã PIL RGB hoàn hảo, chuẩn hóa canvas $512 \times 512$ PNG thành công.

### S4. Kết quả Kiểm thử Toàn diện

- Targeted & contract tests: 63/63 tests PASS trong `test_independent_cohort_bindings.py` (46/46) và `test_independent_cohort_acquisition.py` (17/17).
- Smoke test hermetic: `run_cohort_acquisition.py --smoke-test` PASS trong 1.22s (`acquisition_smoke_receipt_v2.json`).
- Continuity checker: `pnpm continuity:check` PASS.
- Monorepo test suite: `pnpm test` PASS (13/13 unit and contract tests).

---

## R. Revision 2 — Corrections (2026-10-07)

The revision-1 report below (kept unchanged as history) claimed `READY_FOR_COLAB_REAL_ACQUISITION_PILOT`, 17/17 tests PASS and a verified 440-candidate catalog. Re-checking the actual source showed these claims did not hold.

### R1. Defects reproduced from source before fixing

| #   | Defect (commit `dc00895`)                                                                                                                                                                                                                                                                                                                                                               | Evidence                                                                                                                                   |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Notebook declared `PINNED_COMMIT = 4ea65df…` but never used it; an existing clone was only fetched, so stale code could run                                                                                                                                                                                                                                                             | Cell 2 source                                                                                                                              |
| 2   | Notebook used `!` shell commands (exit status ignored) and fell back to `/content` scratch when the Drive mount failed                                                                                                                                                                                                                                                                  | Cells 1–3 source                                                                                                                           |
| 3   | Audit cell accepted any existing manifest/ZIP, unbound to the run, commit, protocol, catalog or plan                                                                                                                                                                                                                                                                                    | Cell 4 source                                                                                                                              |
| 4   | Disjoint guard required `data/research/tgif/manifests/manifest_pilot_a_option_p.csv`, which is excluded from Git: `--verify-plan`, the pilot and 7 tests crash on any fresh clone (Colab included)                                                                                                                                                                                      | CI run [37564295722](https://github.com/nomozer/forensics-web-lab/actions/runs/37564295722) on `dc00895`: **7 failed**, reproduced locally |
| 5   | All 220 COCO `author` values are generated labels `flickr_contributor_<coco_id>`; `published_date` holds COCO `date_captured`; no Flickr photo ID or page; 91 entries are BY-ND/BY-NC-ND                                                                                                                                                                                                | `catalog_eligibility_audit.json`                                                                                                           |
| 6   | Unsplash candidates came from the Unsplash Lite dataset (HF mirror). Dataset Terms §2.A grant download/store and _internal ML training_; §3.A–B prohibit disseminating/redistributing the data. Creating and sharing an edited cohort is not covered; the generic Unsplash License was assumed instead. Download URLs request a processed rendition (`auto=format&fit=crop&w=600&q=80`) | `https://raw.githubusercontent.com/unsplash/datasets/master/TERMS.md` (read 2026-10-07)                                                    |
| 7   | Per-stratum quota counters were updated only on resume, and modification/mask labels were permuted jointly over all 110 slots, so a full 400-pair run could not meet the 40/30/30 × 30/40/30 quotas                                                                                                                                                                                     | `test_full_mode_quota_preservation_counts_new_acceptances` (failed with 99/100, then fixed)                                                |
| 8   | `--engine mock` was accepted in production mode (real photos + mock edits labelled production); `--verify-plan` printed FAIL but exited 0; the smoke test wrote run directories into the system temp root                                                                                                                                                                               | CLI source                                                                                                                                 |

### R2. Changes

- **Notebook** (`notebooks/independent_cohort_acquisition_colab.ipynb`): `checkout_pinned` clones/fetches and checks out **detached** at the full `EXPECTED_COMMIT`, asserts `git rev-parse HEAD`, and stops (without modifying anything) when the clone has local or untracked changes. All Git/CLI calls use `subprocess.run(..., check=True)`. Drive-mount failure stops before any run directory exists. Each session gets a new `RUN_ID`; Cell 4 audits only a run completed in the same session and never overwrites a ZIP. Full mode is not launched from the notebook.
- **Eligibility (pre-registered, detector-independent)**: `evaluate_candidate_eligibility` requires a creator verified from a source response (marker + evidence URL; placeholder patterns rejected), versioned CC license, labelled date provenance, documented download rendition, namespaced source keys and a check timestamp. ND is excluded for edited-cohort export unless separate permission is recorded (this does not claim all private research use is prohibited). BY-NC / BY-SA / BY-NC-SA are eligible with an explicit usage policy (attribution, indicate changes, NonCommercial and/or ShareAlike). Channels whose terms do not cover editing+sharing (Unsplash Lite dataset) are excluded.
- **Disjoint guard**: historical keys `coco:<id>` from the committed `research/evidence/phase-4c.0/real-variant-map.json` (684 sources); candidates carry `coco:`/`flickr:`/`unsplash:` keys. Flickr-ID disjointness is reported `NOT_CHECKED` (historical Flickr IDs are not available); byte-hash comparison only runs when the local manifest exists and only detects byte-identical files.
- **Run binding & audit**: `run_binding.json` (run id, commit, protocol/catalog/plan SHA-256, mode, quota); resume refuses foreign/unbound artifacts; `run_receipt.json` is written only after the quota is met and records per-stratum wall time, peak VRAM (`torch.cuda.max_memory_allocated`, `NOT_MEASURED` without CUDA), engine checkpoint + resolved Hugging Face revision (loaded with that exact `revision`), prompt/seed/steps/guidance/scheduler; every attempt carries `run_id` and elapsed time; receipts add download time, raw size, rendition, license URL and usage policy. `audit_acquisition_run` fails on any binding, checksum, ledger or count mismatch.
- **Plan layout**: first 100 slots per stratum carry exactly 40/30/30 and 30/40/30; the 10 buffer slots hold each of the 9 (modification, mask) cells once plus one extra (object_replacement, medium) — marginals 4/3/3 and 3/4/3, totals 44/33/33 unchanged — so a failed core slot of any cell can be replaced without breaking a quota (tested with one QC failure per stratum in a SYNTHETIC full run).
- **Historical byte-hash guard**: only active when the Git-excluded Option P manifest is present; on Colab it is inactive, printed as a NOTICE and recorded in `run_receipt.json`. It only detects byte-identical files (TGIF PNG hashes vs downloaded JPEG bytes), so it was never a meaningful guard for re-encoded copies.
- **CLI**: `--mode pilot|full` requires `--run-id`, `--expected-commit` (clean checkout at that SHA) and `--output-root`; checks GPU policy (CUDA, ≥ 12 GiB VRAM); refuses existing run dirs unless `--resume`; `--audit-run`; mock engine refused without `--allow-synthetic`.
- **Catalog builder v2** (`scripts/research/build_verified_candidate_catalog.py`): Flickr photo ID from COCO `flickr_url`, creator and current license from Flickr oEmbed (request URL, response SHA-256, timestamp stored), license must agree at COCO and Flickr, excluded entries kept with reasons. Unit-tested only against SYNTHETIC responses; **not run against the live services** (egress to flickr.com, huggingface.co, unsplash.com and images.cocodataset.org is denied in this environment).

### R3. Catalog eligibility (superseded v1 → v2)

|                                                    |       COCO |         Unsplash | Total |
| :------------------------------------------------- | ---------: | ---------------: | ----: |
| v1 candidates evaluated                            |        220 |              220 |   440 |
| Eligible under pre-registered rules                |          0 |                0 | **0** |
| Excluded: creator placeholder / unverified         |        220 | 220 (unverified) |   440 |
| Excluded: ND licence (also)                        |         91 |                — |    91 |
| Excluded: channel terms do not cover edited export |          — |              220 |   220 |
| Replaced by verified candidates                    |          0 |                0 | **0** |
| Still unverified (v2 not built)                    | 220 needed |       220 needed |   440 |

`verified_candidate_catalog.json`, `candidate_acquisition_plan.json` and `acquisition_smoke_receipt.json` are kept unchanged as **SUPERSEDED** history; the eligibility gate rejects the v1 catalog. The default binding now points to `verified_candidate_catalog_v2.json`, which does not exist yet, so `--verify-plan` fails closed.

**Blockers (user decisions/actions):**

1. **Unsplash strata (220 candidates)**: the Lite-dataset channel is not usable for an exported edited cohort. Options: (a) protocol amendment v1.3 replacing the source with a channel whose terms cover editing+sharing and that exposes verifiable creator/license metadata (e.g. Flickr/Wikimedia Commons CC BY / BY-SA photos), or (b) obtain written permission from Unsplash, or (c) discover photos via the Unsplash API under the Unsplash License (needs an API key and a review of the API terms). Not decided here.
2. **COCO strata**: run the v2 builder on a host with network access (Colab, or this environment after allowing `images.cocodataset.org` and `www.flickr.com`), review the output and commit it; then re-pin the notebook.

### R4. Verification actually run (this environment, CPU, no GPU)

- Targeted: `ml/tests/test_independent_cohort_bindings.py` + `ml/tests/test_independent_cohort_acquisition.py` + `ml/tests/test_independent_validation_preparation.py` — see CURRENT_STATE for counts.
- Full hermetic suite `pytest ml/tests -m "not requires_research_artifact"` — see CURRENT_STATE.
- `python scripts/research/run_cohort_acquisition.py --smoke-test` (SYNTHETIC, writes `acquisition_smoke_receipt_v2.json`).
- `python scripts/research/build_verified_candidate_catalog.py --audit-legacy` (offline) → `catalog_eligibility_audit.json`.
- `pnpm continuity:check`, `git diff --check`.

### R5. NOT_MEASURED

Real download/provenance of any authentic image, inpainting on real images, generation time, peak VRAM, QC pass rates, contact sheet of real pairs, content QC, Flickr oEmbed behaviour against the live API, independent detector performance.

---

## Revision 1 (SUPERSEDED — kept as history; claims corrected in §R)

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Status**: `READY_FOR_COLAB_REAL_ACQUISITION_PILOT`<br>
> **Findings status**: `NOT_MEASURED` (zero detector scoring executed on candidate cohort)<br>
> **Evidence class**: `independent_validation_amendment_and_pipeline`<br>
> **Candidate models verified**: 5 / 5 outer-fold models bound and SHA-256 verified (unchanged)<br>
> **Training runs**: 0 new fits, 0 refits, 0 recalibrations, 0 hyperparameter tuning<br>
> **Detector scoring**: 0 predictions (Detector Isolation Invariant strictly enforced)<br>
> **Cohort size target**: $N_{\text{target}} = 400$ source pairs (800 images: 400 authentic, 400 ai_edited) + 400 binary masks<br>
> **Buffer candidate pool**: 440 pre-ordered candidates (110 per stratum: verified real photographic sources)<br>
> **Task scope**: Binary classification (`authentic` = 0 vs. `ai_edited` = 1)<br>
> **Exact blocker**: Full inpainting generation of 400 pairs using SDXL 1.0 requires GPU with $\ge 12$ GB VRAM (local workstation has NVIDIA GTX 1650 4.00 GB VRAM and 2.44 GB free RAM, empirically measured). Acquisition pipeline, real candidate catalog, fail-closed downloader, and canonical 5-cell Colab launcher are verified and ready for execution on Google Colab T4 GPU (defaulting to 8-pair Technical Pilot).

## 1. Bối Cảnh và Căn Cứ Điều Chỉnh (Protocol Amendment v1.2)

Tại Phase 4C.7A, quá trình chuẩn bị kiểm định độc lập kết thúc với trạng thái `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT` và blocker thực địa:

1. Thiếu 160 ảnh tự chụp từ người dùng (Field Collection).
2. Thiếu tài khoản trả phí và API thực thi khả dụng cho Adobe Firefly.

Người dùng đã chính thức đồng ý điều chỉnh kế hoạch thu thập theo nguồn lực thực tế, tuân thủ nguyên tắc trung thực khoa học:

- **Thay thế ảnh tự chụp** bằng ảnh chụp công cộng có giấy phép và provenance rõ ràng:
  - **COCO 2017 Dataset (Phân vùng sạch ngoài Option P)**: 50% (200 pairs). Thẩm tra giấy phép Flickr theo từng ảnh từ metadata chính thức `image_info_test2017.zip` (`CC-BY 2.0`, `CC-BY-SA 2.0`, `Attribution-NoDerivs`, `Attribution-NonCommercial`, v.v.), không mặc định toàn bộ ảnh cùng license.
  - **Unsplash Verified Open Collection (Công bố trước 2022)**: 50% (200 pairs). Giấy phép Unsplash License, lưu trữ photo ID, tác giả và ngày công bố ($\le 2021$). Nhận định khoa học rõ ràng: thời điểm trước 2022 và EXIF không phải bằng chứng tuyệt đối của việc chưa từng chỉnh sửa, mà là bộ lọc vận hành hợp lý chống can thiệp text-to-image AI hiện đại; ngày công bố trên Unsplash là submission date, không suy diễn thành capture date.
- **Thay thế Adobe Firefly** bằng 2 công cụ inpainting mã nguồn mở (open-weights) tiêu biểu:
  - **Stable Diffusion 2 Inpainting (`stabilityai/stable-diffusion-2-inpainting`)**: 50% (200 pairs), CreativeML OpenRAIL-M.
  - **SDXL Inpainting 1.0 (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`)**: 50% (200 pairs), CreativeML OpenRAIL++.
- **Ma trận phân bổ trực giao cân bằng $2 \times 2$ (Orthogonal Balanced Matrix)**:
  - COCO x SD2: 100 pairs (buffer: 110)
  - COCO x SDXL: 100 pairs (buffer: 110)
  - Unsplash x SD2: 100 pairs (buffer: 110)
  - Unsplash x SDXL: 100 pairs (buffer: 110)
  - Tổng: 400 pairs chính thức, 440 candidates trong buffer pool.
  - _Làm rõ về mặt khoa học_: Bỏ cách diễn đạt "triệt tiêu mọi confounding". Phân bổ cân bằng chỉ kiểm soát các yếu tố đã thiết kế giữa 2 nguồn và 2 công cụ; không triệt tiêu các biến số nhiễu tự nhiên tiềm ẩn.
- **Quy tắc thay thế mẫu lỗi bảo toàn đa chiều (Multi-Dimensional Quota-Preserving Replacement)**:
  Nếu một candidate trong ô $(S, T)$ bị loại do lỗi download hoặc QC kỹ thuật, hệ thống chỉ chọn candidate tiếp theo trong buffer cùng ô $(S, T)$ có loại thao tác (`modification_type`) và diện tích mask (`mask_area_class`) còn thiếu quota. Tuyệt đối không làm lệch ma trận quota đã khóa.
- **Ngữ nghĩa của Mask**: Mask nhị phân $512 \times 512$ PNG xác định **vùng yêu cầu chỉnh sửa (inpainting request area)** cung cấp cho diffusion pipeline, không mặc nhiên đồng nhất tuyệt đối với việc mọi pixel bên ngoài mask đều bất biến 100% do đặc tính nén của VAE/latent space.

---

## 2. Kết Quả Đo Lường Hạ Tầng Thực Tế & Ước Tính (Resource Assessment)

### 2.1. Đo lường thực tế trên máy trạm cục bộ (Empirically Measured)

- **GPU**: NVIDIA GeForce GTX 1650 (4.00 GB GDDR6 VRAM, Compute Capability 7.5).
- **System RAM**: 15.78 GB Total, **2.44 GB Available / Free**.
- **Storage Free**: Ổ C: 52.1 GB, Ổ D: 79.0 GB.
- **Kết luận**: Khả năng chạy SDXL fp16 cục bộ là không khả thi do thiếu VRAM ($\ge 6.6$ GB) và thiếu RAM ($\ge 8$ GB cho offload).

### 2.2. Thông số ước tính cho môi trường thực thi từ xa (Estimated)

- **Môi trường mục tiêu**: Google Colab Free Tier (Tesla T4 GPU 15.0 GB VRAM) hoặc A100 GPU.
- **Thời gian sinh dự kiến**: **37–45 phút** cho toàn bộ 400 pairs (ESTIMATED, chưa đo thực nghiệm trên Colab).
- **Dung lượng lưu trữ dự kiến**: ~610 MB cho toàn bộ 400 pairs (800 ảnh + 400 masks).

---

## 3. Kiến Trúc Pipeline Thu Thập và Quality Control (Acquisition Pipeline)

Mã nguồn thực thi gồm một runner chuẩn tắc duy nhất:

- `ml/evaluation/independent_cohort_acquisition.py`: Core pipeline, downloader thật, QC kỹ thuật, durable resume.
- `scripts/research/run_cohort_acquisition.py`: Production CLI runner duy nhất (`--export-plan`, `--verify-plan`, `--smoke-test`, `--mode [pilot|full]`, `--contact-sheet`).
- `scripts/research/build_verified_candidate_catalog.py`: Khám phá và đóng băng danh mục 440 candidates thật từ COCO và Unsplash.
- `notebooks/independent_cohort_acquisition_colab.ipynb`: Notebook Colab 5 cells ngắn gọn, gọi CLI chuẩn tắc, mount Drive bền vững, không sao chép logic.

### 3.1. Các Cơ Chế Bảo Vệ Được Cài Đặt (Guards & Controls)

1. **Four-Level Disjoint Guard**:
   - Đối chiếu với toàn bộ 684 historical Option P sources (bao gồm cả 343 retired locked-test sources).
   - Kiểm tra: `source_id`, `origin_id` (Flickr / Unsplash ID), `raw_sha256` (ảnh gốc trước chuẩn hóa), `master_sha256` (ảnh sau chuẩn hóa 512x512).
2. **Fail-Closed Real Image Downloader**:
   - Chỉ tải từ official URLs đã xác minh.
   - Tuyệt đối cấm fallback sang ảnh synthetic, random noise hoặc placeholder khi download thất bại.
   - Thất bại download được ghi ngay vào `attempt_ledger.jsonl` và thay thế bằng candidate tiếp theo trong buffer.
3. **Canvas and Codec Parity**:
   - Chuẩn hóa về $512 \times 512$ RGB PNG lossless.
   - Mask nhị phân $512 \times 512$ PNG với pixel $\in \{0, 255\}$.
4. **Rigorous Technical QC**:
   - Không lỗi decode PIL, không ảnh solid/blank ($\text{std} > 2.0$), không chứa NaN/Inf.
   - Diện tích mask khớp bracket chỉ định (Small $< 10\%$, Medium $10\% - 30\%$, Large $> 30\%$).
   - Delta check: masked mean absolute difference $> 3.0$.
5. **Detector Isolation Invariant**:
   - `assert_detector_isolation()` fail-closed nếu bất kỳ module detector nào được nạp vào bộ nhớ.
   - Zero detector scoring / zero predictions trong toàn bộ pipeline.
6. **Durable On-Disk Resume & Tamper Detection**:
   - Mọi attempt và receipt ghi trực tiếp xuống `attempt_ledger.jsonl` và `provenance_ledger.jsonl`.
   - Resume tự động kiểm tra SHA-256 toàn bộ ảnh và mask đã có; fail-closed với `TamperDetectedError` nếu phát hiện sai lệch.
7. **Content QC Preservation**:
   - Sinh HTML contact sheet (`content_qc_contact_sheet.html`) hiển thị authentic, mask và ai_edited song song.
   - Zero detector scores hiển thị; tất cả mẫu giữ nguyên trạng thái `PENDING_CONTENT_QC` cho đến khi con người thẩm định.

---

## 4. Bằng Chứng Thực Nghiệm và Kiểm Chứng Kỹ Thuật (Empirical Verification)

### 4.1. Khám Phá & Đóng Băng 440 Candidates Thật

- File catalog: `research/evidence/phase-4c.7b/verified_candidate_catalog.json` (440 candidates: 220 COCO, 220 Unsplash).
- File plan chuẩn tắc: `research/evidence/phase-4c.7b/candidate_acquisition_plan.json` (`evidence_class: "verified_real_catalog"`, `eligible_for_independent_cohort: true`).
- File fixture tổng hợp cách ly: `research/evidence/phase-4c.7b/fixtures/synthetic_candidate_plan_fixture.json` (`evidence_class: "synthetic"`, `eligible_for_independent_cohort: false`).
- Kiểm chứng download thật trên máy trạm: Tải và decode thành công ảnh COCO (`(640, 480) RGB`) và Unsplash (`(600, 336) RGB`) với 0 lỗi.

### 4.2. Candidate Acquisition Plan Verification

Lệnh thực thi:

```bash
python scripts/research/run_cohort_acquisition.py --verify-plan
```

Kết quả kiểm tra:

- `total_candidates`: 440 (110 per stratum: `coco_sd2`: 110, `coco_sdxl`: 110, `unsplash_sd2`: 110, `unsplash_sdxl`: 110).
- `modification_type_counts`: `object_replacement`: 176 (40%), `object_removal_and_infill`: 132 (30%), `object_insertion`: 132 (30%).
- `mask_area_counts`: `medium_10_to_30pct`: 176 (40%), `large_over_30pct`: 132 (30%), `small_under_10pct`: 132 (30%).
- `tool_counts`: SD2: 220, SDXL: 220.
- `source_origin_counts`: COCO 2017: 220, Unsplash Verified: 220.
- `historical_overlap_detected`: `false` (0 overlapping sources, 0 overlapping origins).
- `missing_provenance_detected`: `false`.

### 4.3. Technical Smoke Test

Lệnh thực thi:

```bash
python scripts/research/run_cohort_acquisition.py --smoke-test
```

Kết quả tại `research/evidence/phase-4c.7b/acquisition_smoke_receipt.json`:

- **Trạng thái**: `PASS` (thời gian chạy: 1.09 giây).
- **Detector Isolation Invariant**: PASSED (0 detector modules loaded).
- **Plan Verification**: PASSED (440 real candidates verified).
- **Fail-Closed Synthetic Guard**: PASSED (rejected synthetic fixture in production mode).
- **Pipeline Execution**: PASSED (16 images, all QC passed).
- **Manifest & Checksum Verification**: PASSED.
- **Durable Resume & Tamper Detection**: PASSED (phát hiện chính xác disk tampering).

### 4.4. Test Suites

- **Bộ test mới** `ml/tests/test_independent_cohort_acquisition.py`: **17 / 17 tests PASSED** (11.04s).
  - Bao gồm 7 regression tests mới: rejection of synthetic candidates, download failure fail-closed, missing provenance fail-closed, historical Option P disjoint guard, durable resume & tamper detection, provisional cohort evaluator verdict, notebook 5-cell production CLI contract.
- **Bộ test hồi quy** `ml/tests/test_independent_validation_preparation.py`: **25 / 25 tests PASSED** (15.96s).

---

## 5. Current handoff and Colab execution gate

- **Status**: `PILOT_DIAGNOSED_CORRECTIVE_PLAN_PENDING_HUMAN_REVIEW`.
- Review the existing eight-pair Content QC sheet and `next_pilot_edit_plan_contact_sheet.html`; record explicit human decisions separately from agent notes.
- The canonical notebook checks out a detached full commit SHA. New runs use `MyDrive/Colab Notebooks/forensics-web-lab/independent_cohort_acquisition/runs/<RUN_ID>`; historical roots remain in `docs/RESEARCH_WORKSTREAM_INDEX.md` and are not moved.
- The proposed content-grounded plan remains `PENDING`. The production CLI refuses generation until a human changes the review status through a reviewed commit. No follow-up pilot is authorized by this report.
- Full acquisition is not present as an executable notebook path and remains locked. Do not run the evaluator until a later pilot has completed and received explicit Human Content QC approval.

---

## 6. Trạng Thái Đóng và Đạo Đức Nghiên Cứu

| Hạng Mục                      | Trạng Thái Ghi Nhận                                                                                                    |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **Giao thức nghiên cứu**      | Historical v1.2/v1.3 model/source bindings retained; content-grounded amendment v1.4 proposed and pending human review |
| **Mô hình candidate**         | 5 outer-fold models giữ nguyên 100% trọng số và SHA-256 hash đã kiểm toán                                              |
| **Candidate sources**         | Verified real photographs from COCO 2017 and Wikimedia Commons; historical Option P disjoint guards retained           |
| **Detector isolation**        | Tuyệt đối tuân thủ, zero detector calls trong thu thập và QC                                                           |
| **Independent Performance**   | Tiếp tục giữ trạng thái **`NOT_MEASURED`** (chưa đánh giá)                                                             |
| **Cohort Acquisition Status** | **`NOT_ACQUIRED`** (in progress, ready for remote GPU pilot execution)                                                 |
| **Quy tắc Git**               | Làm việc trên branch `research/independent-cohort-acquisition`, **không tạo Pull Request**                             |
| **Trạng thái Phase**          | **`READY_FOR_COLAB_REAL_ACQUISITION_PILOT`**                                                                           |
