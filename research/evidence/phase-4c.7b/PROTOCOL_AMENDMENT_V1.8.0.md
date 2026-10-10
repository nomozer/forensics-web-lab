# Protocol Amendment v1.8.0 — TGIF-Train-Clean-Subset Independent Cohort Acquisition Plan

Metadata:

- Workstream: `independent_cohort_acquisition`
- Phase trace: `Phase 4C.7B`
- Amendment version: `1.8.0`
- Status: **`APPROVED_BY_HUMAN_REVIEWER (INTAKE_ONLY)`**
- Human reviewer: `Dũng Phạm <valdung04@gmail.com>`
- Human reviewed at: `2026-10-09T14:19:44Z` (Local: 2026-10-09T21:19:44+07:00)
- Human decision: `APPROVED_OPTION_N400_INTAKE_ONLY`
- Approval scope: `INTAKE_ONLY` (Cho phép tải 2 archive ~18,63 GiB bằng Colab CPU, trích xuất 400 cặp đã khóa và trả package ~120-160 MB về local. Tuyệt đối CHƯA CHO PHÉP detector/evaluation hoặc model training).
- Bound locked selection manifest (JSON): [`tgif_train_clean_subset_manifest_locked_n400.json`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json) (SHA-256: `53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c`)
- Bound locked selection manifest (CSV): [`tgif_train_clean_subset_manifest_locked_n400.csv`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv) (SHA-256: `dc9584b31899e5484a4eb40dd6ec653909e39f08de48ada17ba138f8dabd03d4`)
- Bound candidate pool manifest (JSON): [`tgif_train_candidate_manifest_pending.json`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json) (SHA-256: `9e4ef2c9f89ad7dc1316d764c0acfff3b55dbc60dcc666d8928ff49a04dcd7b6`)
- Bound candidate pool manifest (CSV): [`tgif_train_candidate_manifest_pending.csv`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv) (SHA-256: `ba122a4f934c18dd49d59f33fda2c790101eae8eaf9d4a354d18ae4d39a1f6f3`)
- Intake worker script (Colab CPU): [`scripts/research/acquire_tgif_train_subset_colab.py`](scripts/research/acquire_tgif_train_subset_colab.py)
- Local forensic audit script: [`scripts/research/audit_tgif_train_subset_local.py`](scripts/research/audit_tgif_train_subset_local.py)
- Canonical Colab CPU launcher: [`notebooks/tgif_train_cohort_acquisition_colab.ipynb`](notebooks/tgif_train_cohort_acquisition_colab.ipynb)
- Approved cohort: `TGIF-Train-Clean-Subset` ($N=400$ pairs: 14 Large, 221 Medium, 165 Small)

> [!IMPORTANT]
> **Quy chuẩn Quản trị & Trung thực Khoa học (Scientific Honesty Protocol):**
>
> 1. **Phạm vi phê duyệt**: Phê duyệt này **CHỈ DÀNH CHO TIẾP NHẬN DỮ LIỆU (INTAKE ONLY)** trên Colab CPU (tải 2 archive 18,63 GiB, trích xuất 400 cặp, gói ZIP trả về local). **CHƯA CHO PHÉP detector/evaluation hoặc model training**.
> 2. **Không coi phân bổ N=400 là tỷ lệ tự nhiên**: Nhóm Large chỉ có 14 nguồn khả dụng ($3,5\%$). Tuyệt đối **không đưa ra kết luận mạnh riêng cho nhóm Large**.
> 3. **Đính chính căn cứ thống kê**: Tuyên bố “N=400 đảm bảo $\text{ME} < 0,01$” từ Phase 4C.7A thuộc về thiết kế cân bằng cũ; đối với thiết kế mới phân bổ lệch (14 Large, 221 Medium, 165 Small), **CĂN CỨ NÀY CHƯA XÁC LẬP** và bắt buộc phải kiểm tra lại căn cứ thống kê (Monte Carlo / bootstrap power) trước khi chạy evaluation.
> 4. **Bước kiểm tra bộ ba thực hiện LOCAL sau intake**: Colab chỉ tải và trích xuất ảnh authentic và edited. Bước kiểm tra bộ ba (tripartite alignment audit) và leakage guard được thực hiện chính thức tại máy LOCAL sau khi nhận package, sử dụng 100% kho mask đã có sẵn trên đĩa cục bộ (0 MB mạng).
> 5. **Khóa Preprocessing & QC; Cấm tự chọn mẫu thay thế**: Preprocessing (Lanczos 512x512 RGB, Nearest 512x512 mask binary, cấm letterbox) và Technical QC đã khóa. Quy tắc xử lý mẫu thiếu/hỏng là **FAIL-CLOSED**: nếu có tệp thiếu hoặc hỏng, ghi nhận vào receipt và dừng lại; **TUYỆT ĐỐI KHÔNG TỰ ĐỘNG CHỌN MẪU THAY THẾ**.
> 6. **Bảo toàn tính nguyên bản benchmark**: Tuyệt đối **không sửa hoặc composite lại ảnh benchmark** để ép outside L1 về 0. Giữ nguyên độ biến thiên pixel tự nhiên của tác giả TGIF.
> 7. Toàn bộ 26 quyết định Human Content QC lịch sử (diagnostic 6/6 REJECT, calibration 2 PENDING, pilot v1 8 PENDING, pilot v2 8 PENDING) tiếp tục được bảo toàn ở trạng thái PENDING.
> 8. Full cohort ($N=400$) tiếp tục bị **KHÓA CHẶT TRƯỚC EVALUATION**; số lượt gọi detector = 0; hiệu năng độc lập = `NOT_MEASURED`.

---

## 1. Bối cảnh & Lý do Chuyển Hướng sang Benchmark Ngoại Vi

Sau khi hoàn thành Pilot v2 (`pilot-20261009T111247Z`), phán quyết giới hạn đã được xác lập: **“Chưa đủ bằng chứng để mở rộng cấu hình hiện tại lên 400 cặp.”** Việc tự sinh inpainting trên GPU tiêu tốn thời gian, liên tục gặp các lỗi ngữ nghĩa cố hữu (semantic omission, hallucination, placement deficit, và boundary seam).

Nghiên cứu chuyển hướng sang tiếp nhận một tập con sạch từ benchmark công khai có sẵn (**TGIF / TGIF2**), nơi hình ảnh authentic và edited đã được công bố chính thức, có ground-truth mask phân đoạn phục vụ cả bài toán phân loại 2 lớp (`authentic` vs `ai_edited`) và định vị vùng sửa (RQ5 Localization).

---

## 2. Quy Tắc Ghép Bộ Ba (Tripartite Binding) & Đối Chiếu MS-COCO CDN

### 2.1. Bản chất bộ ba trên đĩa và Photoshop Adapted Mask (`ps_mask`)
Kiểm tra thực nghiệm trên đĩa cục bộ đã chứng minh:
- **Ảnh authentic (`orig`)**: Nằm trong thư mục `orig/training/{category}/{raw_id}_orig.png` của TGIF.
- **Ảnh edited (`sd2-sp`)**: Nằm trong thư mục `sd2-sp/training/{category}/{raw_id}_mask_segm.png_ps_mask.png_sd2_0.png`. Tác giả TGIF đã sử dụng Stable Diffusion 2.0 Inpainting với variation 0.
- **Mặt nạ ground-truth (`mask`)**: Phân đoạn segmentation đã được xử lý thích ứng qua Photoshop (`_mask_segm.png_ps_mask.png`).
- **Phát hiện thực nghiệm cốt lõi**: Phép đo pixel delta giữa `sd2-sp` và `orig` cho thấy $100\%$ pixel thay đổi là tập con của `ps_mask.png`, trong khi không phải tập con của native mask `_mask_segm.png` thô (do tác giả áp dụng feathering/dilation nhẹ khi inpaint). Do đó, **bộ ba chuẩn tắc bắt buộc phải gắn với `..._ps_mask.png`**.

### 2.2. Tuyệt đối không ghép MS-COCO CDN với edited chỉ dựa trên `source_id`
- Ảnh trên MS-COCO CDN (`images.cocodataset.org`) là ảnh JPEG gốc có kích thước tự nhiên (thường là $640 \times 480$ hoặc $500 \times 375$).
- Tác giả TGIF khi tạo benchmark đã thực hiện phóng to/cắt canvas (native resolution với cạnh lớn nhất khoảng 1024 px) và lưu dưới dạng PNG không nén.
- Nếu tải ảnh trực tiếp từ COCO CDN mà không qua đúng quy trình của TGIF, kích thước pixel và hệ tọa độ đối tượng sẽ **lệch hoàn toàn so với `sd2-sp` và `mask`**.
- Do đó: **Ảnh authentic tương ứng BẮT BUỘC phải trích xuất từ `orig_training.tar.gz` của TGIF**, không lấy trực tiếp từ COCO CDN.

---

## 3. Preprocessing Chuẩn Tắc của Pipeline Đánh Giá Đã Khóa

Theo đúng quy định của pipeline đánh giá đã khóa (`ml/evaluation/independent_cohort_acquisition.py`, `ml/configs/independent_validation_protocol.yaml`):

1. **Phép biến đổi ảnh RGB (authentic và edited)**:
   - Chuẩn hóa canvas $512 \times 512$ RGB không nén.
   - Tỷ lệ co dãn: $\text{scale} = \max(512/w, 512/h)$.
   - Phép nội suy: `PIL.Image.Resampling.LANCZOS`.
   - Cắt trung tâm (center crop): lấy vùng trung tâm $512 \times 512$.
2. **Phép biến đổi mask ground-truth**:
   - Sử dụng cùng tỷ lệ $\text{scale}$ và tọa độ center-crop như ảnh RGB.
   - Phép nội suy: BẮT BUỘC sử dụng `PIL.Image.Resampling.NEAREST` để bảo toàn tính rời rạc của mặt nạ nhị phân.
   - Nhị phân hóa (binarization): Ngưỡng $> 128 \to 255$, còn lại $0$ (chế độ `L`).
3. **Tính diện tích mask**:
   - Diện tích mask và tỷ lệ phần trăm bắt buộc tính **sau phép biến đổi này** trên canvas $512 \times 512$ ($262.144$ pixel).
4. **Cấm chuyển sang letterboxing**: Tuyệt đối bảo toàn quy tắc center-crop Lanczos đã khóa, không tự ý chuyển sang letterbox có viền đen.

---

## 4. Kiểm Toán Lọc Trùng Lặp 4 Cấp (Multi-Level Disjoint Guard)

Đã đối soát toàn bộ 2.440 tác vụ segmentation trong split `training` của TGIF:

1. **Đối chiếu Option P lịch sử**: Toàn bộ 684 nguồn Option P (341 validation + 343 locked-test) được bảo vệ nghiêm ngặt: **0 collision (100% disjoint)**.
2. **Đối chiếu Phase 4C.7B phát triển phương pháp**: Loại bỏ toàn bộ 336 nguồn COCO đã xuất hiện trong danh mục ứng viên và các pilot/diagnostic/calibration runs của Phase 4C.7B (phát hiện 155 nguồn trùng lặp trong TGIF training và đã loại bỏ triệt để 100%).
3. **Tổng nguồn bị loại trừ**: $684 + 336 = 1.020$ unique source IDs.
4. **Số nguồn duy nhất còn lại**: Đúng **1.403 unique COCO sources** hoàn toàn sạch và chưa từng thấy.

---

## 5. Báo Cáo Thiếu Hụt Nhóm Large & Đề Xuất Phân Bổ Khả Thi

### 5.1. Định mức diện tích mask đăng ký
- `small_under_10pct`: $1,0\% \le P < 10,0\%$ ($2.621 - 26.214$ px trên canvas $512 \times 512$).
- `medium_10_to_30pct`: $10,0\% \le P < 30,0\%$ ($26.215 - 78.643$ px).
- `large_over_30pct`: $30,0\% \le P \le 50,0\%$ ($78.644 - 131.072$ px).
- Loại trừ: $P < 1,0\%$ ($< 2.621$ px, quá vi mô) hoặc $P > 50,0\%$ ($> 131.072$ px, chiếm đa số ảnh).

### 5.2. Thống kê thực tế trên 1.403 nguồn TGIF Training khả dụng
Sau khi loại bỏ các task $< 1\%$ (588 tasks), số nguồn duy nhất có ít nhất 1 task hợp lệ là **1.160 nguồn**:

| Nhóm Diện Tích | Ngưỡng Diện Tích (512x512) | Số Nguồn Duy Nhất (Mutually Exclusive) | Tỷ Lệ Trong TGIF | Nhận Định Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| **Large** | $30\% - 50\%$ ($78.644 - 131.072$ px) | **14 nguồn** | **1,2%** | **THIẾU HỤT NGHIÊM TRỌNG** (Quota v1 yêu cầu 120; thiếu 106 nguồn) |
| **Medium** | $10\% - 30\%$ ($26.215 - 78.643$ px) | **282 nguồn** | **24,3%** | Dư dả so với quota 160 nguồn |
| **Small** | $1\% - 10\%$ ($2.621 - 26.214$ px) | **864 nguồn** | **74,5%** | Dư dả so với quota 120 nguồn |
| **Tổng cộng** | **$1\% - 50\%$** | **1.160 nguồn** | **100,0%** | Đủ lớn để thiết lập tập kiểm định $N=400$ |

> [!CAUTION]
> **Nguyên nhân phân bố lệch:** Trong tập dữ liệu MS-COCO tự nhiên, các đối tượng đơn lẻ được phân đoạn (con mèo, cái cốc, xe buýt, quả táo) hiếm khi chiếm $> 30\%$ diện tích toàn bộ bức ảnh. Đa số đối tượng tự nhiên nằm trong khoảng $1\% - 30\%$.

### 5.3. Cam kết trung thực, Lựa chọn N=400 & Đính chính thống kê quan trọng
Không tự ý nới lỏng tiêu chí (không hạ ngưỡng Large xuống $20\%$).

Người dùng đã xem xét và **chính thức phê duyệt Phương án N = 400 cặp**:
- **Large**: Lấy toàn bộ **14 cặp** ($3,5\%$).
- **Medium**: Lấy **221 cặp** ($55,25\%$).
- **Small**: Lấy **165 cặp** ($41,25\%$).
- **Tổng số**: Đúng **400 cặp** ($1:1$ ghép cặp nguồn duy nhất).

> [!CAUTION]
> **Đính chính Căn cứ Thống kê & Giới hạn Kết luận Khoa học:**
> 1. **Căn cứ $\text{ME} < 0,01$ CHƯA XÁC LẬP cho thiết kế mới**: Tuyên bố ban đầu viện dẫn "N=400 đảm bảo $\text{ME} < 0,01$" là kết quả từ mô phỏng Monte Carlo ở Phase 4C.7A trên phân bổ giả định cân bằng $50/50$. Thiết kế phân tầng mới có phân bổ diện tích lệch rất lớn ($3,5\%$ Large, $55,25\%$ Medium, $41,25\%$ Small). Do đó, **căn cứ biên sai số $\text{ME} < 0,01$ chưa được xác lập cho thiết kế mới**; dự án bắt buộc phải phân tích lại công suất thống kê và kiểm tra lại căn cứ trước khi chạy evaluation.
> 2. **Không coi đây là tỷ lệ tự nhiên**: Phân bổ này phản ánh đặc thù của tập TGIF Training sau các bộ lọc kỹ thuật, không phải là tỷ lệ can thiệp tự nhiên trong thế giới thực.
> 3. **Nhóm Large không đưa ra kết luận mạnh riêng**: Nhóm Large chỉ có đúng 14 nguồn khả dụng ($14 < 120$); vì vậy, **tuyệt đối không đưa ra các kết luận mạnh, độc lập riêng cho phân tầng Large**.

---

## 6. Manifest Ứng Viên & Danh Mục 400 Dòng Đã Khóa (LOCKED SELECTION)

Tất cả mặt nạ ground-truth trên đĩa đã được băm SHA-256 và kiểm tra toàn vẹn bit 100% khi tính diện tích.

1. **Manifest Tuyển Chọn 400 Dòng Đã Khóa (Canonical Locked Selection)**:
   - File JSON: [`research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json)
     - **SHA-256**: `53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c`
   - File CSV: [`research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv`](research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv)
     - **SHA-256**: `dc9584b31899e5484a4eb40dd6ec653909e39f08de48ada17ba138f8dabd03d4`
   - Chứa đúng 400 dòng được sắp xếp tất định (14 Large, 221 Medium, 165 Small), mỗi dòng kèm theo `mask_sha256`, `mask_file_bytes`, kích thước gốc, tọa độ và đường dẫn archive.

2. **Manifest Toàn Bộ Hồ Sơ Ứng Viên Khả Dụng (Full Candidate Pool - 1.160 sources)**:
   - File JSON: [`research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json)
     - **SHA-256**: `9e4ef2c9f89ad7dc1316d764c0acfff3b55dbc60dcc666d8928ff49a04dcd7b6`
   - File CSV: [`research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv)
     - **SHA-256**: `ba122a4f934c18dd49d59f33fda2c790101eae8eaf9d4a354d18ae4d39a1f6f3`

---

## 7. Quy Trình Tiếp Nhận Colab CPU & Các Khóa Kỹ Thuật (Intake Protocol)

### 7.1. Phân định Môi trường: Colab CPU Trích Xuất $\leftrightarrow$ LOCAL Kiểm Toán Bộ Ba
- **Colab CPU (Worker)**:
  - Chỉ làm nhiệm vụ tải 2 archive nặng tạm thời vào `/content/`.
  - Trích xuất chọn lọc đúng 400 ảnh authentic và 400 ảnh edited tương ứng theo manifest đã khóa.
  - Áp dụng preprocessing chuẩn tắc (Lanczos center-crop về $512 \times 512$ RGB không nén).
  - Kiểm tra Technical QC cơ bản (PIL decode, $\text{std} \ge 2,0$).
  - Đóng gói thành `tgif_train_clean_subset_package.zip` (~120–160 MB) kèm receipt.
  - **Không cần tải mask lên Colab**: Giảm thiểu truyền dữ liệu qua mạng.
- **LOCAL (Forensic Audit sau intake)**:
  - **Toàn bộ bước kiểm tra bộ ba (Tripartite Alignment Audit) được thực hiện tại LOCAL sau intake**.
  - LOCAL đã có sẵn 100% kho mask (`data/research/tgif/masks/training/`, 0 MB tải thêm).
  - Script `audit_tgif_train_subset_local.py` sẽ đối soát từng cặp ảnh authentic ↔ edited với mask trên đĩa local:
    * Kiểm tra `mask_sha256` khớp khóa manifest.
    * Chuyển đổi mask (Nearest center-crop $512 \times 512$, binarize $\{0, 255\}$).
    * Đo delta pixel: kiểm tra pixel thay đổi có nằm trong mask hay không, đo `inside_mean_l1`, `outside_mean_l1`, `outside_max_delta`.
    * Kiểm toán Disjoint Guard với cả 684 nguồn Option P lịch sử và 336 nguồn phát triển Phase 4C.7B; fail-closed nếu registry/count/binding sai.
    * Ràng buộc bit-exact packaged manifest, Colab receipt, exact 800 image paths và SHA-256 từng ảnh trước khi kiểm toán pixel.
    * Lập self-contained review contact sheet.

### 7.2. Khóa Quy Tắc Xử Lý Mẫu Thiếu/Hỏng & Cấm Tự Chọn Mẫu Thay Thế
- **Nguyên tắc FAIL-CLOSED**:
  - Nếu bất kỳ tệp ảnh nào trong danh sách 400 mẫu bị thiếu trong archive của tác giả TGIF, bị lỗi giải mã PIL, hoặc vi phạm ngưỡng QC:
  - Worker **PHẢI GHI NHẬN LỖI RÕ RÀNG VÀO RECEIPT VÀ DỪNG QUY TRÌNH**.
  - **TUYỆT ĐỐI CẤM TỰ ĐỘNG CHỌN MẪU THAY THẾ (ZERO AUTOMATIC REPLACEMENT)** từ 760 mẫu còn lại trong pool nếu chưa có sự phê duyệt rõ ràng từ người dùng.

### 7.3. Đường Dẫn Archive Chuẩn Tắc & An Toàn Trích Xuất
- **Nguồn Nextcloud chính thức của IMEC**:
  - `orig_training.tar.gz`: `https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?path=%2Forig&files=orig_training.tar.gz` (5.652.563.073 bytes, etag `db7c6f701d3b5a6c6f375f2fd60a1bd2`)
  - `sd2-sp_training.tar.gz`: `https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?path=%2Fsd2-sp&files=sd2-sp_training.tar.gz` (14.359.902.282 bytes, etag `bbe41aac348ddd4d9f4d687a94390d77`)
- **Trích xuất chọn lọc (Selective Streaming Extraction)**:
  - Sử dụng `tarfile` đọc trực tiếp từng thành viên trong file tar.gz mà không giải nén toàn bộ 18,63 GiB ra đĩa. Giữ dung lượng ổ đĩa Colab an toàn tuyệt đối.
  - Chống TarSlip/ZipSlip: chặn mọi đường dẫn có `..`, dấu `/` đầu, ký tự phân vùng ổ đĩa hoặc symlink.
- **Không sửa/recomposite ảnh benchmark**:
  - Giữ nguyên độ biến thiên pixel tự nhiên của benchmark. Cấm can thiệp nhân tạo dán đè pixel để ép outside L1 về 0.

---

## 8. Trạng Thái Phê Duyệt & Bàn Giao (INTAKE APPROVED)

- **Trạng thái phê duyệt**: **`APPROVED_BY_HUMAN_REVIEWER (INTAKE_ONLY)`**.
- **Phạm vi được phép thực thi**:
  1. Chạy Colab CPU launcher (`notebooks/tgif_train_cohort_acquisition_colab.ipynb`) để tiếp nhận 400 cặp ảnh từ 2 archive chính thức.
  2. Tải package ZIP (~120–160 MB) về máy LOCAL.
  3. Chạy `audit_tgif_train_subset_local.py` tại máy LOCAL để kiểm toán bộ ba, disjoint guard, và tạo contact sheet.
  4. **DỪNG LẠI SAU BƯỚC AUDIT LOCAL** để bàn giao kết quả cho người dùng.
  5. Tuyệt đối **CHƯA ĐƯỢC CHẠY DETECTOR, EVALUATION HOẶC MODEL TRAINING**.
