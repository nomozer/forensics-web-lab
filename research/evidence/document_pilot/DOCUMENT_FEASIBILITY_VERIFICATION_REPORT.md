# Báo Cáo Tiếp Nhận & Kiểm Chứng Dữ Liệu Tạo Sinh Tài Liệu Khả Thi (Controlled Document Generation Feasibility Study)

> **Mã định danh**: `Phase 4C.8 trace — Controlled Document Generation Feasibility Verification`  
> **Branch**: `research/independent-cohort-acquisition` (LOCAL CPU)  
> **Protocol**: Version 1.2.0 (Design Amendment v1.2.0)  
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”  
> **Nguyên tắc khoa học**: Kiểm chứng thực tế trước khi kết luận; bảo toàn bằng chứng lỗi; không tự gán HUMAN_PASS.

---

## 1. Tiếp Nhận Gói Dữ Liệu Lưu Trữ (Archive Intake)

- **Đường dẫn ZIP gốc**: `data/research/local-artifacts/document-feasibility/document_feasibility_results.zip`
- **Kích thước file ZIP**: **10,592,936 bytes** (~10.10 MB)
- **Mã băm SHA-256 ZIP**: `deadd9205207eaaa296a8d9eeb493ee48eeea76ee7afd210684187e4026ed684`
- **Danh mục ZIP**: Gồm 34 files (1 manifest, 3 preflight receipts, 1 human_review.csv, 1 contact sheet, và 4 thư mục attempt x 7 files/attempt = 28 files).
- **Thư mục giải nén độc lập**: `data/research/local-artifacts/document-feasibility/runs/run_20261013_051727_4320eb1e/` (được cách ly ngoài Git qua `.gitignore`, không ghi đè bất kỳ artifact cũ nào).

### Đối Soát Khóa Kế Hoạch Chuẩn Tắc (Plan Parity Audit)

| Thông số | Kế hoạch chuẩn tắc (`document_feasibility_plan.json`) | Bằng chứng Colab (`feasibility_manifest.json`) | Trạng thái đối soát |
| :--- | :--- | :--- | :--- |
| **Run ID** | Khóa động theo timestamp và hash plan | `run_20261013_051727_4320eb1e` | **MATCH** |
| **Protocol Version** | `1.2.0` | `1.2.0` | **MATCH** |
| **Plan Hash SHA-256** | `4320eb1e85149323afeed51b18296bb44823d5c86267c4464e7c3d996077256f` | `4320eb1e85149323afeed51b18296bb44823d5c86267c4464e7c3d996077256f` | **BIT-EXACT MATCH** |
| **Model Repository** | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | **MATCH** |
| **Model Revision** | Default unpinned trên HF | `115134f363124c53c7d878647567d04daf26e41e` (Khóa 18 files FP16) | **VERIFIED** |
| **Seeds (4 attempts)** | `[20261012, 20261013, 20261014, 20261015]` | `[20261012, 20261013, 20261014, 20261015]` | **MATCH** |
| **CORD-v2 Sources** | Rows `[22, 61, 7, 93]`, image IDs `[22, 61, 7, 93]` | Rows `[22, 61, 7, 93]`, image IDs `[22, 61, 7, 93]` | **MATCH** |
| **Prompts & Targets** | Khóa cứng 4 prompts | Khớp nguyên văn 4 prompts | **MATCH** |
| **Parameters** | steps=30, guidance=7.5, strength=0.99, canvas=1024 | steps=30, guidance=7.5, strength=0.99, canvas=1024 | **MATCH** |

> [!NOTE]
> **Ghi nhận thiếu sót cục bộ (Field Omission)**: Trong từng file `attempt_receipt.json` cục bộ trong thư mục con, các trường `seed`, `prompt`, `source_row_index` không được ghi trực tiếp (`null`). Tuy nhiên, các giá trị này được bảo toàn và lưu đầy đủ tại `source_preflight_receipt.json` và `feasibility_manifest.json`.

---

## 2. Phần Cứng, Thời Gian & VRAM Thực Tế

Toàn bộ các số đo được trích xuất trực tiếp từ các receipt thực nghiệm trên Colab:
- **Phần cứng**: 1 GPU **Tesla T4** (Tổng VRAM: 14.56 GB, Driver CUDA 13.0, PyTorch 2.11.0+cu130).
- **Dung lượng đĩa trống**: 69.92 GB (thoải mái so với yêu cầu tối thiểu 10 GB).
- **Thời gian nạp mô hình (Model Load Duration)**: **140.35 giây** (~2.34 phút).
- **Thời gian suy luận từng attempt**:
  - Attempt 0: **32.93 giây** (Peak VRAM: 9,178.50 MB = 8.96 GiB).
  - Attempt 1: **30.81 giây** (Peak VRAM: 9,178.50 MB = 8.96 GiB).
  - Attempt 2: **32.95 giây** (Peak VRAM: 9,178.50 MB = 8.96 GiB).
  - Attempt 3: **33.22 giây** (Peak VRAM: 9,178.50 MB = 8.96 GiB).
- **Tổng thời gian suy luận (Total Inference Duration)**: **129.91 giây** (~2.17 phút).
- **Tổng thời gian GPU Colab**: ~4.5 phút (rất an toàn so với hạn mức ngân sách 60 phút).
- **Bộ nhớ VRAM**: Mức đỉnh 8.96 GiB hoàn toàn nằm trong giới hạn 12 GiB định mức cho GPU T4.

---

## 3. Kết Quả Kiểm Chứng Kỹ Thuật (Technical QC)

Bộ kiểm chứng `scripts/research/verify_document_feasibility_artifacts.py` đã kiểm tra từng attempt dựa trên mảng pixel thật (`numpy` arrays) và mã băm cryptographic:

| Attempt | Nguồn | Thao tác | Kích thước gốc | Mask px | Outside Bitwise Equal | Inside changed px (%) | Mean L1 / Max L1 | Hashes Match | Technical QC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attempt 0** | CORD row 22 | Replacement | 576 × 864 | 2,132 | **True (Diff = 0)** | 2,131 (99.95%) | 8.93 / 68 | 5/5 Match | **PASS** |
| **Attempt 1** | CORD row 61 | Replacement | 1356 × 2315 | 44,091 | **True (Diff = 0)** | 33,567 (76.13%) | 65.64 / 255 | 5/5 Match | **PASS** |
| **Attempt 2** | CORD row 7 | Removal | 576 × 864 | 9,088 | **True (Diff = 0)** | 9,088 (100.0%) | 10.86 / 82 | 5/5 Match | **PASS** |
| **Attempt 3** | CORD row 93 | Removal | 864 × 1296 | 7,992 | **True (Diff = 0)** | 7,992 (100.0%) | 16.75 / 128 | 5/5 Match | **PASS** |

### Kết luận kỹ thuật
- **100% ngoài mask bảo toàn tuyệt đối**: `np.array_equal == True`, sai khác tối đa `max_outside_diff == 0` trên toàn bộ 4 attempts. Không có rò rỉ pixel (zero leakage).
- **Toàn vẹn kích thước và mask nhị phân**: Kích thước ảnh xuất ra khớp chính xác ảnh gốc, giá trị mask chỉ gồm `{0, 255}`.
- **Tính toàn vẹn mã băm (Cryptographic Integrity)**: 100% mã băm SHA-256 của các file ảnh trên đĩa khớp hoàn toàn với `attempt_receipt.json`.

---

## 4. Thẩm Định Nội Dung Trực Quan (Human Content QC Preparation)

Đã tạo bộ ảnh phóng đại 2x không can thiệp nội dung tại `research/evidence/document_pilot/qc_inspection/`. Dưới đây là đánh giá trực quan sơ bộ (Preliminary Visual Assessment) từ agent; trạng thái đánh giá chính thức được giữ ở `PENDING_HUMAN_REVIEW`:

### Bảng Đánh Giá Chi Tiết 4 Attempts

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Attempt 0: Thay thế đơn giá "280,000" -> "350,000" (CORD row 22, Bbox: [442, 406, 468, 488])                                           |
| - Nội dung mục tiêu:      CHƯA ĐẠT (FAILED TO RENDER). Mô hình không in ra chữ số "350,000".                                            |
| - Chữ có đọc được không:  KHÔNG CÓ CHỮ. Vùng inpainting biến thành một mảng xám trơn nhẵn.                                              |
| - Phần cần giữ:           Bảo toàn 100% (bitwise equal ngoài mask).                                                                     |
| - Nền và mép ghép:        Lộ rõ mảng hình chữ nhật xám hơi lệch màu so với nền giấy nhiệt xung quanh.                                   |
| - Phán quyết sơ bộ:       INPAINTING_TEXT_COLLAPSE (Mô hình chỉ xóa chữ cũ, không in được số mới).                                     |
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Attempt 1: Thay thế tên món "Populaire Chocolate" -> "Magnum Gold Almond" (CORD row 61, Bbox: [1085, 128, 1156, 749])                   |
| - Nội dung mục tiêu:      CHƯA ĐẠT CHÍNH XÁC (MISSPELLED). Sinh ra chữ "MAGNU ALONLD" thay vì "Magnum Gold Almond".                     |
| - Chữ có đọc được không:  ĐỌC ĐƯỢC RÕ RÀNG nhưng SAI CHÍNH TẢ (bỏ sót từ "Gold", thiếu 'm' ở Magnum, Almond biến thành ALONLD).          |
| - Phần cần giữ:           Bảo toàn 100% (dòng số lượng và dòng tên món bên dưới bitwise equal).                                         |
| - Nền và mép ghép:        Nền trắng phẳng; font chữ sans-serif in hoa lệch hoàn toàn phong cách chữ in nhiệt gốc của hóa đơn.           |
| - Phán quyết sơ bộ:       INPAINTING_HALLUCINATED_TYPO (Mô hình sinh sai chính tả và lệch kiểu chữ in ấn).                             |
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Attempt 2: Xóa bỏ dòng giảm giá "Discount: -9,545" (CORD row 7, Bbox: [524, 208, 556, 492])                                             |
| - Nội dung mục tiêu:      ĐẠT (REMOVED). Dòng giảm giá đã bị xóa hoàn toàn, không còn dấu vết số.                                       |
| - Chữ có đọc được không:  Không có chữ (đúng mục đích xóa bỏ).                                                                         |
| - Phần cần giữ:           Bảo toàn 100% (SubTotal và PB1 bitwise equal).                                                                |
| - Nền và mép ghép:        BẤT THƯỜNG RÕ BẰNG MẮT THƯỜNG. Ảnh gốc có dải bóng đổ chéo; mảng inpainting sinh ra nền phẳng sáng hơn,       |
|                           tạo thành một vệt hình chữ nhật sáng đè lên bóng đổ tự nhiên.                                                 |
| - Phán quyết sơ bộ:       REMOVAL_PATCH_SHADOW_MISMATCH (Xóa được chữ nhưng lộ ranh giới bóng đổ).                                     |
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Attempt 3: Xóa bỏ dòng thuế "TAX 15,636" (CORD row 93, Bbox: [939, 383, 975, 605])                                                      |
| - Nội dung mục tiêu:      ĐẠT (REMOVED). Dòng thuế và số tiền thuế đã biến mất.                                                         |
| - Chữ có đọc được không:  Không có chữ (đúng mục đích xóa bỏ).                                                                         |
| - Phần cần giữ:           Bảo toàn 100% (Subtotal và Total 172,000 bitwise equal).                                                      |
| - Nền và mép ghép:        BẤT THƯỜNG RÕ BẰNG MẮT THƯỜNG. Giấy hóa đơn gốc có nếp nhăn nhàu nát (crease) chạy dọc qua chữ. Vùng         |
|                           inpainting làm phẳng lì giấy, cắt đứt nếp nhăn và có sắc xanh lam nhạt (bluish tint), lộ rõ vết ghép.        |
| - Phán quyết sơ bộ:       REMOVAL_PATCH_TEXTURE_CREASE_BROKEN (Xóa được chữ nhưng phá vỡ cấu trúc nếp nhăn giấy).                       |
+-----------------------------------------------------------------------------------------------------------------------------------------+
```

---

## 5. Đường Dẫn Ảnh Để Người Dùng Thẩm Định (QC Artifact Paths)

Người dùng có thể mở trực tiếp các file ảnh sau trên máy LOCAL để kiểm tra:

1. **Bảng tổng hợp Contact Sheet (5 cột)**:
   - File: [feasibility_contact_sheet.png](./feasibility_contact_sheet.png)
2. **Ảnh so sánh phóng đại 2x cạnh nhau (Side-by-side 2x inspection)**:
   - Attempt 0: [attempt_0_replacement_price_side_by_side_2x.png](./qc_inspection/attempt_0_replacement_price_side_by_side_2x.png)
   - Attempt 1: [attempt_1_replacement_item_name_side_by_side_2x.png](./qc_inspection/attempt_1_replacement_item_name_side_by_side_2x.png)
   - Attempt 2: [attempt_2_removal_discount_line_side_by_side_2x.png](./qc_inspection/attempt_2_removal_discount_line_side_by_side_2x.png)
   - Attempt 3: [attempt_3_removal_tax_line_side_by_side_2x.png](./qc_inspection/attempt_3_removal_tax_line_side_by_side_2x.png)
3. **Thư mục chứa ảnh ROI chi tiết**:
   - Thư mục: [qc_inspection/](./qc_inspection/)

---

## 6. Kết Luận Khoa Học & Trả Lời Các Câu Hỏi Giới Hạn

1. **Generation đã chạy được bao nhiêu attempts?**
   - Đã chạy và hoàn thành đầy đủ **4 / 4 attempts** (100% hoàn tất thực thi trên GPU T4, không có attempt nào bị lỗi hoặc dừng giữa chừng).

2. **Replacement và removal có triển vọng đến đâu?**
   - **Replacement (Thay thế văn bản)**: Triển vọng **RẤT THẤP** với SDXL inpainting nguyên bản (vanilla text-to-image inpainting). Mô hình diffusion thiếu khả năng điều khiển chính xác cấp độ ký tự (glyph-level control), dẫn đến hiện tượng xóa trắng (Attempt 0) hoặc sinh sai chính tả và lệch kiểu chữ hoàn toàn (Attempt 1: "MAGNU ALONLD"). Nếu muốn tạo văn bản thay thế đạt chuẩn, bắt buộc cần các phương pháp chuyên biệt (ví dụ: TextDiffuser, AnyText, hoặc render văn bản quang học bằng font in nhiệt rồi inpaint hòa trộn).
   - **Removal (Xóa bỏ văn bản)**: Triển vọng kỹ thuật **CAO HƠN** (cả Attempt 2 và 3 đều xóa sạch chữ thành công), nhưng gặp thách thức lớn về **tính chân thực của nền giấy** (background artifact): mô hình sinh ra nền phẳng làm lộ ranh giới bóng đổ hoặc cắt đứt nếp nhăn vật lý của giấy. Tuy nhiên, các vết ghép này vẫn cung cấp tín hiệu giả mạo thực tế cho các mô hình phát hiện giả mạo.

3. **Có lỗi pipeline cần sửa trước lượt sau không?**
   - **Đã phát hiện và sửa trên verifier**: Verifier trước đó hardcode protocol v1.1.0 và thiếu bước kiểm tra mã băm ảnh; đã được nâng cấp để bind metadata từ manifest và kiểm tra mã băm an toàn (16/16 unit tests PASS).
   - **Khiếm khuyết runner**: Runner Colab hiện tại không gán `seed`, `prompt`, `source_row_index` trực tiếp vào `attempt_receipt.json` cục bộ (mặc dù có trong manifest). Cần bổ sung các trường này vào `attempt_receipt` ở lần chạy sau để đảm bảo tự chứa thông tin đầy đủ.

---

## 7. Đề Xuất Tiếp Theo Duy Nhất

**Đề xuất duy nhất**: Người dùng mở các ảnh kiểm tra trực quan tại `qc_inspection/` để thực hiện thẩm định con người (Human Content QC), sau đó quyết định hướng đi:
- *Hướng A*: Chấp nhận các hạn chế thị giác của SDXL để coi đây là bằng chứng chẩn đoán về giới hạn tạo sinh tài liệu của diffusion thuần túy.
- *Hướng B*: Điều chỉnh phương pháp tạo sinh tài liệu (chuyển sang text removal thuần túy hoặc nghiên cứu pipeline text-synthesis chuyên biệt) trước khi mở rộng quy mô.
