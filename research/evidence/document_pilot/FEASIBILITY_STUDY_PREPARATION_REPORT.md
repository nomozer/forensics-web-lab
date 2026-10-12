# Báo Cáo Kỹ Thuật: Chuẩn Bị, Sửa Lỗi Toàn Diện & Kiểm Chứng Dữ Liệu Thật CORD-v2 Cho Thử Nghiệm Khả Thi Tạo Dữ Liệu Tài Liệu AI
(Controlled AI-Document Generation Feasibility Study Repair & CORD Preflight Report)

> **Mã văn bản**: `REPORT-2026-10-12-FEASIBILITY-STUDY-REPAIR-AND-CORD-PREFLIGHT-V1.2.0`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Trạng thái thực hiện**: `PREPARED_SOURCE_PREFLIGHT_VERIFIED`<br>
> **Protocol**: Version 1.2.0 (theo [INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.2.0.md](./INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.2.0.md))<br>
> **Môi trường**: LOCAL CPU (Windows pwsh) hoàn thành toàn diện sửa code, preflight dữ liệu CORD thật và kiểm thử 13 unit tests; sẵn sàng chuyển giao notebook thực thi trên GOOGLE COLAB GPU.

---

## 1. Tóm Tắt Mục Tiêu & Ranh Giới Nghiên Cứu

Thử nghiệm khả thi tạo dữ liệu (Generation Feasibility Study) được triển khai trên đúng **bốn tài liệu CORD v2** theo kế hoạch máy tiền đăng ký:
- 2 attempts thay đổi chữ/số bằng AI inpainting (text replacement).
- 2 attempts xóa nội dung bằng AI inpainting (text removal).

### Ranh Giới Khoa Học Tuyệt Đối (Scientific Honesty)
1. **Đây là Generation Feasibility Study**: Mục tiêu duy nhất là kiểm chứng tính khả thi kỹ thuật của quy trình inpainting tài liệu (tải checkpoint, nạp dữ liệu, suy luận inpainting, bảo toàn hình học và kiểm chứng ngoài mask).
2. **Không phải Detector Training**: Tuyệt đối không huấn luyện mô hình phát hiện trong đợt này.
3. **Không phải Cohort Đánh Giá Độc Lập Hoàn Tất**: Bốn tài liệu này là mẫu phát triển kỹ thuật (`development feasibility sources`), được đánh dấu cách ly (`DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION`) và loại trừ vĩnh viễn khỏi bất kỳ panel đánh giá 40 nguồn nào trong tương lai.
4. **Không Thay Đổi Research Questions (RQs)**: Các câu hỏi nghiên cứu và 6 frozen detectors được bảo toàn nguyên vẹn.
5. **Không Dùng Detector Scores Để Chọn Ảnh**: Không chạy 6 detectors trên các mẫu sinh ra để lọc, xếp hạng hoặc chọn attempt thành công.
6. **Không Tuyên Bố Đã Generation Trên Colab**: LOCAL chưa có GPU tương thích SDXL FP16 (~10 GB VRAM), chưa chạy checkpoint weights; trạng thái chính thức được niêm phong là `PREPARED_SOURCE_PREFLIGHT_VERIFIED`.

---

## 2. Kết Quả Khắc Phục Tám Khiếm Khuyết Đã Phát Hiện

| # | Khiếm khuyết đã phát hiện | Nguyên nhân gốc rễ (Root Cause) | Giải pháp kỹ thuật đã kiểm chứng |
| :--- | :--- | :--- | :--- |
| **1** | Cell 2 IndentationError | Chép tay thiếu lệnh `print(f"...")` tại dòng 57-58. | Chuyển sang generator tự động `build_document_feasibility_colab_notebook.py` với `compile()` kiểm toán 100% code cells. |
| **2** | Loader CORD Parquet struct | Schema vật lý leaf chứa `bytes`, nhưng schema logic Arrow là `image: struct<bytes, path>`. | Đọc trực tiếp `table["image"][row]["bytes"].as_py()` và `table["ground_truth"][row]`, hỗ trợ cache cục bộ. |
| **3** | Lệch Hạt Giống (Seeds) | Báo cáo cũ gõ nhầm seeds; máy lưu `[20261012..20261015]`. | Khóa hạt giống canonical `[20261012, 20261013, 20261014, 20261015]` đồng bộ 100% giữa plan, runner, notebook và receipt. |
| **4** | CSV Ghép Chuỗi Lỗi `"" )` | Ghép chuỗi thủ công làm hỏng cú pháp strict CSV khi có dấu nháy/phẩy. | Dùng `csv.DictWriter` với `quoting=csv.QUOTE_MINIMAL`, kiểm thử strict round-trip Unicode. |
| **5** | Resume Bỏ Sót Review Rows | Câu lệnh `continue` bỏ qua bước thêm attempt đã hoàn thành vào danh sách review. | Cấu trúc lại luồng ghi review: nạp toàn bộ 4 attempts, giữ nguyên phán quyết con người cũ nếu file đã tồn tại. |
| **6** | Thiếu Contact Sheet trong Runner | Runner cũ chỉ ghi ảnh riêng lẻ, phụ thuộc verifier local để ghép ảnh. | Runner Colab tự sinh `feasibility_contact_sheet.png` (5 cột) và đóng gói trực tiếp vào archive zip. |
| **7** | Thiếu Failure Receipts & Partial Archive | Lỗi uncaught làm sập script, mất toàn bộ dữ liệu của các attempts trước. | Bổ sung `try...except` per attempt, sinh `attempt_failure_receipt.json`, luôn đóng gói partial zip archive. |
| **8** | Chưa Thực Thi Budget Limits | Ngân sách chỉ ghi trong markdown, không có assert trong mã nguồn. | Bổ sung kiểm tra fail-closed: shard size $\le 260$ MB, checkpoint $\le 12$ GiB, disk free $\ge 10$ GB, GPU time $\le 60$ phút, archive $\le 100$ MB. |

---

## 3. Kết Quả Preflight Dữ Liệu CORD-v2 Thật Trên LOCAL

Preflight được thực hiện độc lập qua `scripts/research/preflight_cord_sources.py` trên shard chính thức `naver-clova-ix/cord-v2` (`data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet`):
- **Trần ngân sách phê duyệt**: 260 MB (272,629,760 bytes) theo Protocol Amendment v1.2.0.
- **Kích thước file thực tế**: **242,080,800 bytes** ($230.87\text{ MiB}$ / $242.08\text{ MB}$) $\le 260\text{ MB}$.
- **Mã băm SHA-256 Shard**: `0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15`.
- **Lưu cache**: Đã tải và lưu tại `data/research/cord_v2/` (được cách ly bởi `.gitignore`).

### Kết Quả Đối Soát 4 Tài Liệu Nguồn CORD-v2 Thật

1. **Attempt #0 (`attempt_0_replacement_price`)**:
   - `source_row_index`: 22 | `source_image_id`: 22 | `source_doc_identifier`: `cord_v2_valid_row22_img22`
   - Kích thước gốc: $576 \times 864$ | EXIF Orientation: 1 (Normal)
   - Encoded Bytes SHA-256: `d27f10eeadc2f3a4f1b7e277f07fd2a2c4c06ed24378b5ec949786ec39949c7b`
   - Decoded RGB SHA-256: `faeb6395bae43b219bb2c8699fbf1c1023503e59b0372bd6bfd96b17d2f562fe`
   - Văn bản gốc: `"280,000"` | Danh mục GT: `menu.price` | Thao tác: `replacement` $\to$ `"350,000"`
   - Bounding Box: `[442, 406, 468, 488]` | Khớp 100% hình học và bao phủ trọn vẹn văn bản.

2. **Attempt #1 (`attempt_1_replacement_item_name`)**:
   - `source_row_index`: 61 | `source_image_id`: 61 | `source_doc_identifier`: `cord_v2_valid_row61_img61`
   - Kích thước gốc: $1356 \times 2315$ | EXIF Orientation: 1 (Normal)
   - Encoded Bytes SHA-256: `aa2de89a0a51024f43ed5a0579453354447fba47b7714f4a19605fd96a0a67c4`
   - Decoded RGB SHA-256: `0cb678f7671dbe3dd1ac1257ea7d5ddd1d6c29667644cb86a811c495b19dd933`
   - Văn bản gốc: `"Populaire Chocolate"` | Danh mục GT: `menu.nm` | Thao tác: `replacement` $\to$ `"Classic Tiramisu"`
   - Bounding Box: `[1085, 128, 1156, 749]` | Khớp envelope tọa độ.

3. **Attempt #2 (`attempt_2_removal_discount_line`)**:
   - `source_row_index`: 7 | `source_image_id`: 7 | `source_doc_identifier`: `cord_v2_valid_row7_img7`
   - Kích thước gốc: $576 \times 864$ | EXIF Orientation: 1 (Normal)
   - Encoded Bytes SHA-256: `52adf3336878fbd558951660d6ce52268dd3e636563ca1d7d7e004ccab96bff7`
   - Decoded RGB SHA-256: `a8af9134a268ee619e857843e79628d19e7f5a02d748e1f89706ee4d206cee40`
   - Văn bản gốc: `"Discount: -9,545"` | Danh mục GT: `sub_total.discount_price` | Thao tác: `removal` $\to$ `""`
   - Bounding Box: `[524, 208, 556, 492]` | Khớp 100% hình học.

4. **Attempt #3 (`attempt_3_removal_tax_line`)**:
   - `source_row_index`: 93 | `source_image_id`: 93 | `source_doc_identifier`: `cord_v2_valid_row93_img93`
   - Kích thước gốc: $864 \times 1296$ | EXIF Orientation: 1 (Normal)
   - Encoded Bytes SHA-256: `d7f3fb8f44ff9368d1469e354ae2ae05d7616688741369ef4f85e4ea7efc1e13`
   - Decoded RGB SHA-256: `87ea589178ee79a02fb4208a3854897e9eeea9583b6e828453ae660c159828e8`
   - Văn bản gốc: `"TAX 15,636"` | Danh mục GT: `sub_total.tax_price` | Thao tác: `removal` $\to$ `""`
   - Bounding Box: `[939, 383, 975, 605]` | Khớp 100% hình học.

- **Biên nhận preflight**: `research/evidence/document_pilot/cord_source_preflight_receipt.json`.
- **Contact sheet kiểm chứng hình học**: `research/evidence/document_pilot/preflight_inspections/cord_preflight_contact_sheet.png`.

---

## 4. Kiểm Thử Hệ Thống (Verification Evidence)

Toàn bộ 13 unit tests trong `ml/tests/test_document_feasibility_geometry.py` đạt **13/13 PASS**:
1. `test_pilot_design_amendments`: PASS (Xác minh Amendment v1.1.0 và v1.2.0).
2. `test_geometry_roundtrip_module`: PASS (Bảo toàn 100% bitwise ngoài mask).
3. `test_out_of_bounds_bbox_rejected`: PASS (Chặn đứng bounding box ngoài khung hoặc thoái hóa).
4. `test_document_feasibility_plan_schema`: PASS (Khớp Protocol v1.2.0, 260 MB ceiling, 4 seeds).
5. `test_notebook_compiles_all_cells`: PASS (Biên dịch toàn bộ code cells notebook và runner).
6. `test_notebook_payload_hash_parity`: PASS (Đối soát SHA-256 bit-exact giữa plan, runner và staging).
7. `test_real_cord_parquet_loader_and_sources`: PASS (Nạp đúng 4 tài liệu CORD thật từ cache).
8. `test_cord_source_preflight_receipt_verification`: PASS (Xác thực biên nhận preflight và ảnh kiểm tra).
9. `test_csv_strict_roundtrip`: PASS (Bảo toàn dấu ngoặc kép, dấu phẩy và Unicode tiếng Việt).
10. `test_resume_preserves_completed_attempts_and_human_verdicts`: PASS (Không mất hàng, không ghi đè verdict cũ).
11. `test_failure_handling_and_partial_archive`: PASS (Cô lập lỗi attempt, sinh failure receipt và zip partial).
12. `test_budget_and_schema_fail_closed_checks`: PASS (Dừng fail-closed khi vượt ngưỡng tải).
13. `test_verification_harness_fail_closed`: PASS (Bắt rò rỉ pixel ngoài mask 1 đơn vị L1).

Bộ kiểm thử hồi quy:
- `test_document_pilot_reporting_reconciliation.py`: 6/6 PASS.
- `test_document_observation_pilot.py`: 3/3 PASS.
- `pnpm test`: 64 vitest + 13 continuity tests = 77/77 PASS (100% pass).

---

## 5. Hướng Dẫn Thực Hiện Trên Google Colab GPU (T4)

1. Mở trình duyệt và truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn **Upload** và tải tệp `notebooks/independent_document_feasibility_colab.ipynb`.
3. Vào **Runtime** $\to$ **Change runtime type** $\to$ Chọn **T4 GPU** (miễn phí, không cần đăng ký tài khoản trả phí).
4. Thực thi tuần tự 4 cells:
   - **Cell 1**: Xác nhận GPU T4 sẵn sàng và cài đặt dependencies.
   - **Cell 2**: Staging tự động kế hoạch máy và runner script (đối soát SHA-256 bit-exact).
   - **Cell 3**: Thực thi chạy 4 attempts tạo dữ liệu khả thi (tự động tải checkpoint FP16, nạp CORD shard, suy luận inpainting và composite).
   - **Cell 4**: Tải tệp `document_feasibility_results.zip` về máy LOCAL.
5. Sau khi tải xong về máy LOCAL, đặt tệp vào thư mục dự án và chạy:
   ```bash
   python scripts/research/verify_document_feasibility_artifacts.py --zip document_feasibility_results.zip
   ```
   để tiến hành kiểm chứng kỹ thuật fail-closed và thẩm định nội dung.

---

## 6. Đúng Một Đề Xuất Tiếp Theo Được Phê Duyệt

Sau khi preflight dữ liệu CORD thật đạt chuẩn `PREPARED_SOURCE_PREFLIGHT_VERIFIED`, đề xuất duy nhất là người dùng mở và chạy notebook đã sửa trên Google Colab GPU (T4) để thực hiện bốn attempts tạo sinh.
