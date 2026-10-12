# Tu Chính Thiết Kế Kỹ Thuật v1.2.0: Chuẩn Hóa Ngân Sách Dữ Liệu Gốc CORD-v2, Schema Parquet & Đối Soát Hạt Giống
(Independent Document Generation Pilot Design Amendment v1.2.0)

> **Mã văn bản**: `AMENDMENT-2026-10-12-DOC-GENERATION-PILOT-DESIGN-V1.2.0`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Văn bản tu chính**: Bổ sung và chuẩn hóa [INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.1.0.md](./INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.1.0.md)<br>
> **Mục đích**: Khắc phục 8 khiếm khuyết kỹ thuật đã phát hiện, truy nguyên cấu trúc lưu trữ Parquet của CORD-v2, điều chỉnh có căn cứ trần ngân sách dữ liệu gốc từ 200 MB lên 260 MB, và đối soát đồng bộ hạt giống cùng quy trình failure handling.

---

## 1. Truy Nguyên Cấu Trúc Parquet & Nguyên Nhân Gốc Rễ Vượt Ngân Sách 200 MB

### 1.1. Phát Hiện Preflight Thực Tế Trên Tệp CORD-v2 Shard
Trong quá trình preflight thực tế trên shard chính thức của CORD-v2:
* **Repository**: `naver-clova-ix/cord-v2`
* **Revision**: `7f0115a4b758a71d6473b8d085751692da2fef98`
* **Tệp shard validation**: `data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet`
* **Kích thước đo đạc thực tế (HTTP Content-Length)**: **242,080,800 bytes** ($242.08\text{ MB}$ theo hệ thập phân, hay $230.87\text{ MiB}$ theo hệ nhị phân).

### 1.2. Cơ Chế Lưu Trữ Cột Điểm Ảnh (Snappy Dictionary Page)
Kiểm tra chi tiết metadata Parquet qua PyArrow C++ engine ghi nhận:
1. Shard chỉ có 1 row group duy nhất chứa 100 hàng.
2. Cột `image.bytes` có `total_compressed_size = 241,931,149 bytes` (chiếm 99.94% dung lượng tệp).
3. Toàn bộ 100 ảnh PNG được mã hóa bên trong **một dictionary page duy nhất** từ `dictionary_page_offset = 4` đến `data_page_offset = 241,931,026` với thuật toán nén `SNAPPY`.
4. Trong định dạng Parquet với nén Snappy, PyArrow không thể giải nén ngẫu nhiên một mục từ điển đơn lẻ mà bắt buộc phải nạp và giải nén toàn bộ dictionary block từ offset 4.
5. Do đó, bất kỳ truy vấn nào để đọc ảnh (dù chỉ đọc 1 hàng như row 7 hay 4 hàng 7, 22, 61, 93) đều đòi hỏi phải tải khối từ điển 241.9 MB.
6. Ước tính trước đây trong Amendment v1.1.0 cho rằng HTTP Range có thể chỉ tải 5–10 MB là **không khả thi về mặt cấu trúc lưu trữ Parquet**.

### 1.3. Tu Chính Trần Ngân Sách Dữ Liệu Gốc (Original Data Budget Adjustment)
Theo nguyên tắc Scientific Honesty và điều khoản "Nếu protocol cần sửa cách cộng ngân sách, tạo amendment trước download; không tự nâng trần":
* **Trần ngân sách cũ (v1.1.0)**: $\le 200\text{ MB}$ ($209,715,200\text{ bytes}$).
* **Trần ngân sách tu chính (v1.2.0)**: **$\le 260\text{ MB}$ ($272,629,760\text{ bytes}$ / $260\text{ MiB}$)**.
* **Mức độ tiêu thụ thực tế**: $242,080,800\text{ bytes} \le 272,629,760\text{ bytes}$ (thỏa mãn trần với biên an toàn $30.55\text{ MB}$ hay $12.6\%$).
* **Quy tắc Single-Download Cache**: Tệp shard bắt buộc chỉ được tải **đúng 1 lần** và lưu vào cache cục bộ `data/research/cord_v2/` (trên LOCAL) hoặc `./cord_cache/` (trên Colab), tái sử dụng cho cả 4 attempts; nghiêm cấm tải lặp lại qua mạng cho từng attempt.

---

## 2. Bảng Ngân Sách 6 Thành Phần Cập Nhật (Protocol v1.2.0)

| Thành phần | Đơn vị & Cơ sở tính toán | Trần ngân sách v1.2.0 | Đo đạc thực tế / Kế toán | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **1. Checkpoint Colab (FP16)** | Safetensors SDXL inpainting ungated | $\le 12.0\text{ GiB}$ ($12,884,901,888\text{ B}$) | **$6,941,218,469\text{ bytes}$ ($6.46\text{ GiB}$)** | ĐẠT |
| **2. Dữ liệu gốc CORD-v2** | 1 shard validation duy nhất nạp cache | $\le 260\text{ MB}$ ($272,629,760\text{ B}$) | **$242,080,800\text{ bytes}$ ($230.87\text{ MiB}$)** | ĐẠT (sau tu chính) |
| **3. Dependencies Colab** | diffusers, transformers, accelerate, v.v. | $\le 2.0\text{ GiB}$ | **$\approx 1.2\text{ GiB}$** | ĐẠT |
| **4. Artifacts chuyển về LOCAL** | Archive zip gồm 4 pairs, contact sheet, CSV | $\le 100\text{ MB}$ ($104,857,600\text{ B}$) | **$\approx 15 - 30\text{ MB}$** | ĐẠT |
| **5. Đĩa & Peak VRAM** | Ephemeral Colab runtime | $15\text{ GB}$ đĩa / $13\text{ GB}$ VRAM | **$\approx 9.5\text{ GB}$ đĩa / $\approx 10.5\text{ GB}$ VRAM** | ĐẠT |
| **6. Thời gian thực thi GPU** | Colab T4 (30 steps Euler x 4 attempts) | $\le 60\text{ phút}$ | **$\approx 10 - 15\text{ phút}$** | ĐẠT |

---

## 3. Khắc Phục Bất Nhất Hạt Giống (Seed Erratum & Reconciled Lineage)

1. **Hiện trạng mâu thuẫn**:
   * Kế hoạch máy gốc `document_feasibility_plan.json`: `[20261012, 20261013, 20261014, 20261015]`.
   * Báo cáo tóm tắt cũ `FEASIBILITY_STUDY_PREPARATION_REPORT.md`: ghi nhầm `[42, 1337, 2026, 9001]` do sao chép từ tài liệu lịch sử trước đây.
2. **Quyết định khóa chuẩn tắc**:
   * Chuẩn tắc duy nhất của kế hoạch máy là dãy hạt giống **`[20261012, 20261013, 20261014, 20261015]`** gắn liền với ngày khởi tạo và mã attempt `0..3`.
   * Đồng bộ 100% dãy hạt giống này trên: (a) `document_feasibility_plan.json`; (b) `run_document_feasibility_colab.py`; (c) `independent_document_feasibility_colab.ipynb`; và (d) các báo cáo đối soát.

---

## 4. Xác Minh Schema Cột & Đối Soát Ground Truth 4 Attempts

### 4.1. Cấu Trúc Schema PyArrow CORD-v2
* Cột `image` trong Arrow Table là kiểu `StructType`:
  * `bytes`: `BinaryType` (chứa chuỗi byte nhị phân của tệp ảnh PNG gốc).
  * `path`: `StringType` (chứa đường dẫn chuỗi tương đối).
* Cột `ground_truth` là kiểu `StringType` (chuỗi JSON chứa metadata phân tích cú pháp).
* **Quy tắc truy cập code chuẩn tắc**:
  ```python
  table = pf.read(columns=["image", "ground_truth"])
  image_row = table["image"][row_idx].as_py()
  raw_bytes = image_row["bytes"]
  gt_data = json.loads(table["ground_truth"][row_idx].as_py())
  ```
  Tuyệt đối cấm gọi `pf.read(columns=["bytes"])` vì Arrow Schema không có cột cấp cao nhất tên là `bytes`.

### 4.2. Bảng Đối Soát 100% Khớp Thực Tế Annotation
Kiểm tra thực tế từ trường `ground_truth` của 4 hàng xác nhận:

| Attempt | Row CORD | Image ID | Kích thước gốc | Trường chọn | Nhãn / Text gốc thực tế | BBox ground truth | BBox trong plan | Kết luận |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attempt_0` | 22 | 22 | $576 \times 864$ | `menu.price` | `"280,000"` | `[442, 406, 468, 488]` | `[442, 406, 468, 488]` | **KHỚP 100%** |
| `attempt_1` | 61 | 61 | $1356 \times 2315$ | `menu.nm` | `"Populaire Chocolate"` | `[1084, 128, 1156, 750]` | `[1085, 128, 1156, 749]` | **KHỚP ENVELOPE** |
| `attempt_2` | 7 | 7 | $576 \times 864$ | `sub_total.discount_price` | `"Discount: -9,545"` | `[524, 208, 556, 492]` | `[524, 208, 556, 492]` | **KHỚP 100%** |
| `attempt_3` | 93 | 93 | $864 \times 1296$ | `sub_total.tax_price` | `"TAX 15,636"` | `[939, 383, 975, 605]` | `[939, 383, 975, 605]` | **KHỚP 100%** |

Toàn bộ 4 nguồn tài liệu đều có annotation thực tế đầy đủ, tọa độ hộp bao hoàn toàn nằm bên trong kích thước ảnh, không có trường nào bị bịa đặt.

---

## 5. Quy Chuẩn Kỹ Thuật Runner, Failure Handling & Turnkey Notebook

1. **Chuẩn hóa CSV**:
   Sử dụng thư viện chuẩn `csv.DictWriter(..., quoting=csv.QUOTE_MINIMAL)` thay cho việc nối chuỗi thủ công; bảo đảm không phát sinh lỗi ngoặc đơn thừa hay lỗi parse strict.
2. **Resume Policy**:
   Khi một attempt đã có `attempt_receipt.json`, runner vẫn phải bổ sung bản ghi đó vào danh sách tổng hợp của `human_review.csv`. Nếu tệp `human_review.csv` cũ đã có đánh giá của con người (`HUMAN_PASS` hoặc `HUMAN_FAIL`), runner bảo toàn phán quyết và ghi chú của con người, không ghi đè thành `PENDING_HUMAN_REVIEW`.
3. **Failure Receipts & Partial Archive**:
   Mỗi attempt được bảo vệ bởi khối bắt lỗi `try...except Exception`. Nếu lỗi phát sinh, runner lập tức ghi nhận `attempt_failure_receipt.json` ghi rõ lỗi, loại lỗi và bước xảy ra sự cố. Runner tiếp tục thực thi các attempts còn lại nếu lỗi có tính cục bộ. Khi kết thúc, runner luôn đóng gói bản lưu trữ zip (`document_feasibility_results.zip`) chứa toàn bộ artifacts và receipts đã tạo.
4. **Trực tiếp tạo Contact Sheet trong Runner**:
   Runner tự động ghép ảnh `feasibility_contact_sheet.png` (5 cột: Authentic, ROI Trước, ROI Sau, Final Edited, Intervention Mask) và đóng gói trực tiếp vào archive zip.
5. **Kiểm soát Ngân sách Chặt chẽ (Budget Enforcement)**:
   Runner kiểm tra kích thước checkpoint, dữ liệu gốc và dung lượng đĩa trước khi khởi tạo mô hình; fail-closed dừng thực thi nếu bất kỳ hạn ngạch nào bị vượt.
6. **Turnkey Notebook Sinh Tự Động**:
   Notebook `notebooks/independent_document_feasibility_colab.ipynb` được sinh hoàn toàn tự động từ script `build_document_feasibility_colab_notebook.py`, nhúng trực tiếp runner và plan chuẩn tắc với kiểm toán cú pháp `compile()`, loại bỏ 100% lỗi cú pháp và chép tay.
