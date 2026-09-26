# Phase 4B.1 Report: Small-Data Feasibility and Pairability Audit

## 1. Executive Summary

| Thuộc tính | Giá trị |
| :--- | :--- |
| **Phase** | **4B.1 — Small-Data Feasibility and Pairability Audit** |
| **Starting Commit** | `3be606c` |
| **Implementation Snapshot Commit** | `e8db154` |
| **Main Branch Ref** | `460f6d5` (*Bảo toàn nguyên vẹn, không commit trực tiếp*) |
| **Remote Host Allowed** | `cloud.ilabt.imec.be` (Duy nhất host chính thức) |
| **Metadata Requests Executed** | `10` requests (GET root, PROPFIND, HEAD) |
| **Metadata Response Bytes** | **`41,549` bytes** (~0.04 MiB, trần cứng: 5 MiB) |
| **Dataset Content Downloaded** | **`0` bytes** (Không tải thêm bất kỳ byte ảnh/archive nào) |
| **Model Weights Downloaded** | **`0` bytes** |
| **Training Runs Executed** | **`0`** |
| **Independent Statistical Unit** | **`source_id`** (MS-COCO 12-digit image identifier) |
| **Source ID Audit Result** | **2,242** unique source images (1,558 train, 341 val, 343 test) |
| **Inpainting Task Instances** | **3,124** instances (2,440 train, 341 val, 343 test) |
| **Pairability Status** | `pending-content-acquisition` |
| **Recommended Acquisition Option**| **Option P — Small-Data Thesis Pilot** (5.88 GB download, 684 sources) |
| **Phán quyết (Verdict)** | **`PASS`** |

---

## 2. Giải trình Chênh lệch Cardinality: 2.242 vs 3.124

Bằng chứng kiểm toán thực nghiệm trên toàn bộ **31.238 file mask** tại [source-id-audit.json](source-id-audit.json) đã làm sáng tỏ hoàn toàn sự chênh lệch:

1. **Bản chất của con số 3.124**:
   * Trong bài báo TGIF (arXiv:2407.11566 Table 1), các tác giả công bố 3.124 ảnh authentic (2.440 train, 341 val, 343 test).
   * Con số 3.124 này là số lượng **nhiệm vụ chỉnh sửa theo danh mục** (*category inpainting task instances*).
   * Mỗi instance tương ứng với đúng 10 file mask (tổng cộng $2.440 \times 10 + 341 \times 10 + 342 \times 10 + 8 = 31.238$ file mask; trong đó duy nhất 1 instance tại test có 8 file).
2. **Bản chất của con số 2.242**:
   * Tập dữ liệu gốc MS-COCO val2017 có nhiều ảnh chứa cùng lúc nhiều nhãn đối tượng (ví dụ: một ảnh có cả `apple` và `bowl`).
   * Trong tập huấn luyện (`train`), có **571 ảnh MS-COCO** được tạo mask cho nhiều danh mục đối tượng khác nhau (ví dụ: ảnh `110999` có mask trong cả 5 thư mục: `apple`, `bowl`, `cup`, `orange`, `spoon`).
   * Do đó, 2.440 inpainting instances trong `train` thực chất bắt nguồn từ **1.558 ảnh chụp thực tế duy nhất**.
   * Tại `val`: 341 instances = 341 unique source images (1:1).
   * Tại `test`: 343 instances = 343 unique source images (1:1).
   * Tổng số ảnh chụp độc lập toàn bộ dataset: $1.558 + 341 + 343 = \mathbf{2.242}$.
3. **Bảo toàn Ranh giới Chống rò rỉ (Zero Cross-Split Leakage)**:
   * Giữa 3 phân vùng Train (1.558), Val (341) và Test (343), **tuyệt đối không có bất kỳ `source_id` nào trùng lặp** (giao thoa = 0).
   * Parser chuẩn tắc được xác lập là **Parser 3 (Canonical Dual-Key)**:
     * `source_id`: 12 chữ số COCO (`000000xxxxxx`) — **Đơn vị thống kê độc lập duy nhất** dùng cho phân chia split, bootstrap CI và báo cáo cỡ mẫu.
     * `instance_id`: `{category}_{coco_id}` — Đơn vị định danh nhiệm vụ inpainting dùng để ghép cặp mặt nạ đối ứng.

---

## 3. Khảo sát Siêu dữ liệu Từ xa (Remote Metadata Inventory)

Khảo sát qua WebDAV PROPFIND và HEAD trên máy chủ chính thức `cloud.ilabt.imec.be` thu được cấu trúc upstream chi tiết:

### 3.1. Thành phần `tgif-orig` (7.32 GB)
* Bao gồm đúng **3 split tar.gz archives** độc lập:
  * `orig_training.tar.gz`: 5,652,563,073 bytes (~5.26 GiB), ETag `"db7c6f701d3b5a6c6f375f2fd60a1bd2"`
  * `orig_validation.tar.gz`: 859,947,874 bytes (~820.11 MiB), ETag `"d95a1202a7445d79872735f60cbeaa95"`
  * `orig_testing.tar.gz`: 806,962,390 bytes (~769.58 MiB), ETag `"fc2934fed45674aa89479640d0030beb"`
* **Khả năng tải tập con**: Xác minh thành công. Máy chủ cho phép tải trực tiếp từng archive split riêng rẽ qua tham số `?path=%2Forig&files=orig_<split>.tar.gz` với `Content-Length` chính xác và mã `200 OK`.

### 3.2. Thành phần `tgif-sd2-sp` (18.57 GB)
* Bao gồm đúng **3 split tar.gz archives** độc lập:
  * `sd2-sp_training.tar.gz`: 14,359,902,282 bytes (~13.37 GiB), ETag `"bbe41aac348ddd4d9f4d687a94390d77"`
  * `sd2-sp_validation.tar.gz`: 2,172,017,290 bytes (~2.02 GiB), ETag `"44b9e62f224dd312f59d9a5bd79d1171"`
  * `sd2-sp_testing.tar.gz`: 2,040,575,228 bytes (~1.90 GiB), ETag `"2aa172ad2b1200973b7593259b37cc07"`
* **Khả năng tải tập con**: Tương tự, có thể tải lẻ `sd2-sp_validation.tar.gz` (2.17 GB) hoặc `sd2-sp_testing.tar.gz` (2.04 GB) mà không cần tải toàn bộ 18.57 GB.

---

## 4. Hiện trạng Ghép cặp (Pairability Audit)

* **Split-level symmetry**: Cả 3 component (`orig`, `sd2-sp`, `masks`) đều có cấu trúc đối xứng hoàn hảo qua 3 split: `training`, `validation`, `testing`.
* **Trạng thái pairability**: Ghi nhận **`pending-content-acquisition`**. Do remote chỉ công bố metadata ở cấp archive tar.gz, việc xác thực từng cặp ảnh pixel byte-to-byte chỉ có thể thực hiện sau khi tải và giải nén nội dung ảnh thực tế.

---

## 5. Thiết kế Giao thức Ít Dữ liệu & Đường cong Học tập

Đã ban hành văn bản quy chuẩn [docs/SMALL_DATA_PROTOCOL.md](../../docs/SMALL_DATA_PROTOCOL.md) và cấu hình máy đọc [ml/configs/pilot_a_learning_curve.yaml](../../ml/configs/pilot_a_learning_curve.yaml):

1. **Đường cong học tập**: Đăng ký trước 5 mức $N \in \{50, 100, 250, 500, 1000\}$ unique `source_id`.
   * Đối với Option P, mức $N=50, 100, 250$ ở trạng thái `runnable`.
   * Mức $N=500, 1000$ tự động chuyển trạng thái `not_runnable` do vượt quá dung lượng training pool của Option P (341 sources).
2. **Fixed Benchmarks**: Đánh giá mọi mức trên tập Test cố định (343 sources).
3. **Mô hình 2 giai đoạn**:
   * Stage 1: MobileNetV3-Small đóng băng 100% backbone, huấn luyện linear classification/localization head.
   * Stage 2: Mở khóa block cuối cùng chỉ khi Stage 1 vượt qua kiểm toán shortcut.
4. **Sáu baseline bắt buộc**:
   * Stratified Dummy, Metadata-Only (EXIF/C2PA), DSP-Only (FFT/DCT/Noise), Frozen Visual Backbone, Fine-Tuned Visual Model, Multimodal Fusion Calibrator.
5. **Tiêu chuẩn nghiệm thu**:
   * Cận dưới khoảng tin cậy 95% của $\Delta\text{Macro-F1}$ (Paired Stratified Bootstrap 1,000 resamples) phải lớn hơn 0 so với cả Dummy và Metadata-only baseline.
   * Độ lệch chuẩn qua các hạt giống $\sigma < 0.03$.
6. **Định nghĩa pháp chứng**:
   * Nhãn `no_ai_evidence`: *Không phát hiện đủ bằng chứng AI trong phạm vi mô hình và dữ liệu đánh giá* (tuyệt đối không tuyên bố ảnh thật 100%).
   * Mọi trường hợp độ tin cậy thấp hoặc xung đột tín hiệu chuyển sang `uncertain`.

---

## 6. Ba Phương án Dữ liệu & Đề xuất Duy nhất

Chi tiết xem [research/evidence/phase-4b.1/small-data-options.json](small-data-options.json):

* **Option S (Smallest Engineering Subset)**:
  * Tải: `orig_validation.tar.gz` (859.9 MB) + `sd2-sp_validation.tar.gz` (2.17 GB) = **3.03 GB**.
  * Quy mô: 341 unique sources. Mục đích: Kiểm tra loader và pipeline. Không tạo metric khoa học.
* **Option P (Small-Data Thesis Pilot — RECOMMENDED)**:
  * Tải: `orig_validation.tar.gz` (859.9 MB) + `orig_testing.tar.gz` (806.9 MB) + `sd2-sp_validation.tar.gz` (2.17 GB) + `sd2-sp_testing.tar.gz` (2.04 GB) = **5.88 GB** (`5,879,502,782` bytes).
  * Quy mô: **684 unique source images** (341 validation pool, 343 fixed test benchmark).
  * Giá trị khoa học: Đủ điều kiện chạy đường cong học tập $N=50, 100, 250$, kiểm định paired bootstrap 95% CI trên 343 test sources độc lập. Tiết kiệm 77.3% băng thông so với bản đầy đủ.
* **Option F (Full Pilot A)**:
  * Tải toàn bộ 6 archives = **25.89 GB** (`25,891,968,137` bytes).
  * Quy mô: 2,242 sources (3,124 instances). Dành cho benchmark quy mô lớn.

**Khuyến nghị chính thức của đề tài**: Chọn **Option P**.

---

## 7. Trạng thái Tuyên bố Khoa học

* **Chỉ số hiệu năng (Accuracy, Macro-F1, ECE, mIoU)**: Tiếp tục mang trạng thái **`not evaluated`**.
* **Định vị khoa học**: Phase 4B.1 đã giải quyết dứt điểm câu hỏi về cardinality và đơn vị thống kê độc lập, thiết lập giao thức ít dữ liệu chặt chẽ và khảo sát thành công phương án tải tập con tối ưu. **Chưa có mô hình nào được huấn luyện** và **chưa có tuyên bố nào về độ chính xác phân loại**.
