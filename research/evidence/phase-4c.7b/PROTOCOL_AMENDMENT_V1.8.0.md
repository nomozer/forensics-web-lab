# Protocol Amendment v1.8.0 — TGIF-Train-Clean-Subset Independent Cohort Acquisition Plan

Metadata:

- Workstream: `independent_cohort_acquisition`
- Phase trace: `Phase 4C.7B`
- Amendment version: `1.8.0`
- Status: **`PENDING_HUMAN_REVIEW`**
- Human reviewer: `PENDING` (chờ người dùng phê duyệt)
- Human reviewed at: `PENDING`
- Bound candidate manifest (JSON): [`tgif_train_candidate_manifest_pending.json`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json) (SHA-256: `0197ffa2829a3a19bc35322b0e260db6e881d506b95ab5d5e49a3d881dfd571b`)
- Bound candidate manifest (CSV): [`tgif_train_candidate_manifest_pending.csv`](research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv) (SHA-256: `480bd98062264ba9776eee5db1e64457d3b8ab0d862f30e5b7f4d8aa0868f919`)
- Intake worker script (Colab CPU): [`scripts/research/acquire_tgif_train_subset_colab.py`](scripts/research/acquire_tgif_train_subset_colab.py)
- Local forensic audit script: [`scripts/research/audit_tgif_train_subset_local.py`](scripts/research/audit_tgif_train_subset_local.py)
- Proposed cohort target: `TGIF-Train-Clean-Subset` ($N=400$ pairs recommended, $N=200$ pairs alternative)

> [!IMPORTANT]
> **Quy chuẩn Quản trị & Trung thực Khoa học (Scientific Honesty Protocol):**
>
> 1. Chuẩn bị Amendment v1.8.0 và Manifest PENDING **KHÔNG đồng nghĩa được phép tải dữ liệu quy mô lớn (>50 MB)**. Toàn bộ bước này thực hiện trên metadata và mask cục bộ; việc tải archive trên Colab phải chờ người dùng phê duyệt.
> 2. **Không sửa/composite lại ảnh benchmark để ép outside L1 về 0**: Dữ liệu benchmark gốc từ tác giả TGIF phải được bảo toàn nguyên vẹn độ biến thiên thực tế, không can thiệp nhân tạo.
> 3. **Trung thực về thiếu hụt nhóm Large**: Nhóm Large ($\ge 30\%$) bị thiếu hụt nghiêm trọng trong TGIF ($14 < 120$). Tuyệt đối không tự nới lỏng ngưỡng diện tích mask; báo cáo chính xác số lượng và đề xuất phân bổ khả thi.
> 4. Toàn bộ 26 quyết định Human Content QC lịch sử (diagnostic 6/6 REJECT, calibration 2 PENDING, pilot v1 8 PENDING, pilot v2 8 PENDING) tiếp tục được bảo toàn ở trạng thái PENDING.
> 5. Full cohort ($N=400$) tiếp tục bị **KHÓA CHẶT**; số lượt gọi detector = 0; hiệu năng độc lập = `NOT_MEASURED`.

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

### 5.3. Cam kết trung thực & Đề xuất 2 phương án quy mô khả thi
Không tự ý nới lỏng tiêu chí (không hạ ngưỡng Large xuống $20\%$). Trình 2 phương án cụ thể:

#### Phương án 1 (KHUYẾN NGHỊ) — Cỡ mẫu $N = 400$ cặp (Bảo toàn công suất thống kê)
- **Large**: Lấy toàn bộ **14 cặp** ($3,5\%$).
- **Medium**: Lấy **221 cặp** ($55,25\%$).
- **Small**: Lấy **165 cặp** ($41,25\%$).
- **Tổng số**: Đúng **400 cặp** ($1:1$ ghép cặp nguồn duy nhất).
- **Lý do khuyến nghị**: Giữ nguyên cỡ mẫu $N=400$ đã qua kiểm chuẩn Monte Carlo (Phase 4C.7A), đảm bảo biên sai số $\text{ME} < 0,01$ và công suất thống kê cho primary endpoint tại `jpeg_q75`.

#### Phương án 2 (THAY THẾ) — Cỡ mẫu $N = 200$ cặp
- **Large**: Lấy toàn bộ **14 cặp** ($7,0\%$).
- **Medium**: Lấy **93 cặp** ($46,5\%$).
- **Small**: Lấy **93 cặp** ($46,5\%$).
- **Tổng số**: Đúng **200 cặp** ($1:1$ ghép cặp nguồn duy nhất).

---

## 6. Manifest Ứng Viên Máy Đọc Đã Niêm Phong (PENDING)

Manifest chính thức đã được xuất bằng seed cố định `20261010` (PCG64):
- File JSON: `research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json`
  - **SHA-256**: `0197ffa2829a3a19bc35322b0e260db6e881d506b95ab5d5e49a3d881dfd571b`
- File CSV: `research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.csv`
  - **SHA-256**: `480bd98062264ba9776eee5db1e64457d3b8ab0d862f30e5b7f4d8aa0868f919`
- Danh mục chứa đủ 1.160 nguồn, xếp hạng tất định; đánh dấu chính xác cờ `selected_in_n400` và `selected_in_n200`.

---

## 7. Quy Trình Tiếp Nhận Colab CPU & Ngân Sách Tải (Intake Budget)

Quy trình 5 bước bảo vệ băng thông và tài nguyên lưu trữ:

| Bước | Môi Trường | Hành Động | Ngân Sách / Dung Lượng |
| :---: | :---: | :--- | :--- |
| **1** | Google Colab | Khởi tạo Colab runtime CPU tiêu chuẩn (0 VRAM, 0 GPU). | Miễn phí tài nguyên tính toán |
| **2** | Google Colab | Tải 2 file tar.gz gốc từ Nextcloud vào ổ đĩa tạm `/content/`:<br>- `orig_training.tar.gz` (5,26 GiB)<br>- `sd2-sp_training.tar.gz` (13,37 GiB) | **18,63 GiB tải tạm trên Colab** (dung lượng đĩa Colab ~100 GB đáp ứng thoải mái) |
| **3** | Google Colab | Chạy `acquire_tgif_train_subset_colab.py`: đối chiếu SHA-256 manifest, trích xuất duy nhất 400 ảnh authentic và 400 ảnh edited được chọn, chuẩn hóa $512 \times 512$ PNG. | Giải nén cục bộ trên Colab ~600 MB |
| **4** | Google Colab | Đóng gói thành `tgif_train_clean_subset_package.zip` kèm biên nhận trích xuất và báo cáo kiểm toán căn chỉnh bộ ba. | **Dung lượng file ZIP tải về local: ~120 - 160 MB** |
| **5** | LOCAL | Chạy `audit_tgif_train_subset_local.py`: kiểm tra bảo mật ZIP, kiểm toán 4-cấp Disjoint Guard, ghép nối với 400 masks đã có sẵn trên đĩa local. | **Masks tải về: 0 MB** (24.400 masks đã có sẵn 100% trên đĩa) |

> [!NOTE]
> **Tổng dung lượng tải về máy LOCAL**: Chỉ khoảng **120 – 160 MB** (thay vì phải tải hơn 18 GiB về máy cá nhân).

---

## 8. Bàn Giao Kế Hoạch & Trạng Thái Chờ Duyệt (PENDING)

Amendment v1.8.0 và Kế hoạch Tiếp nhận TGIF được bàn giao ở trạng thái **`PENDING_HUMAN_REVIEW`**.

Mọi hoạt động tải dữ liệu trên Colab, giải nén và đánh giá độc lập chỉ được phép thực hiện sau khi người dùng phê duyệt:
1. Phê duyệt quy mô cohort ($N=400$ pairs hoặc $N=200$ pairs).
2. Chấp thuận phân bổ nhóm diện tích mask (chấp nhận 14 cặp Large và tái phân bổ cho Medium/Small).
3. Cho phép chạy worker tải 18,63 GiB tạm trên Colab và gói ZIP ~150 MB về máy.
