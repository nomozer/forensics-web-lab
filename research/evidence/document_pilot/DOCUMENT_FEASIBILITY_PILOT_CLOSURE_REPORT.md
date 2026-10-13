# Báo Cáo Chốt Pilot: Chẩn Đoán Đối Chứng Hình Học & Đối Soát Bằng Chứng Tạo Sinh Tài Liệu SDXL

> **Mã định danh**: `Phase 4C.8 trace — Document Feasibility Geometry Control Diagnosis & Pilot Closure`  
> **Branch**: `research/independent-cohort-acquisition` (LOCAL CPU)  
> **Protocol**: Version 1.2.0 (Design Amendment v1.2.0)  
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”  
> **Nguyên tắc khoa học**: Kiểm chứng thực tế trên mảng pixel; bảo toàn nguyên vẹn artifacts gốc; không suy diễn vượt bằng chứng; giữ human review PENDING.

---

## 1. Mục Tiêu & Phương Pháp Thực Nghiệm Khóa 3 Điều Kiện

Để phân tách dứt điểm sai lệch do xử lý hình học thông thường (crop, resize, letterbox, padding, inverse composite) khỏi can thiệp tạo sinh của mô hình AI, pilot thiết lập **3 điều kiện độc lập** trên cùng 4 nguồn CORD-v2 (`rows [22, 61, 7, 93]`):

1. **`I0` — Authentic Control**: Ảnh RGB gốc đã lưu trong kho lưu trữ Colab (`authentic_control.png`, native dimensions).
2. **`I1` — Geometry-Only Control**: Ảnh gốc đi qua toàn bộ quy trình: Context crop (với margin ratio 0.4–0.5) $\to$ Bicubic resize $\to$ Padding canvas vuông $1024 \times 1024$ $\to$ Unpadding $\to$ Bicubic restore về kích thước crop $\to$ Composite nghiêm ngặt bên trong target bounding box; **hoàn toàn bỏ qua mô hình AI** (dùng chính canvas đầu vào làm canvas đầu ra).
3. **`I2` — Existing Generated Output**: Ảnh kết quả chỉnh sửa hoàn chỉnh đã sinh bởi `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` trong lần chạy Colab GPU (`final_edited_document.png`). Giữ nguyên vẹn, không tạo lại hay chỉnh sửa.

> [!IMPORTANT]
> **Provenance & Nhãn Điều Kiện**: `I1` là đối chứng xử lý ảnh thông thường, **không phải ảnh AI**. `I1` tuyệt đối không được đưa vào cohort như mẫu AI-positive. Control này khảo sát riêng khâu crop/resize/composite; không kiểm soát VAE reconstruction nội bộ hay quy trình diffusion khử nhiễu.

---

## 2. Bảng Số Liệu Định Lượng Pixel Thật (SSOT Pixel Arrays)

Toàn bộ các chỉ số dưới đây được tính toán trực tiếp từ mảng số nguyên có dấu (`np.int32`) trên đĩa qua script chuẩn tắc `scripts/research/diagnose_document_feasibility_geometry.py`, loại bỏ hoàn toàn hiện tượng tràn số `uint8`:

| Attempt ID | Thao tác | Kích thước gốc | Mask Px | $I_1$ Outside Eq | $I_1 - I_0$ Inside Mean L1 / Max | $I_1 - I_0$ Changed (%) | $I_2 - I_0$ Inside Mean L1 / Max | $I_2 - I_0$ Changed (%) | $I_2 - I_1$ Inside Mean L1 / Max | $I_2 - I_1$ Changed (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attempt 0** (CORD r22) | Replacement (`280,000` $\to$ `350,000`) | $576 \times 864$ | 2,132 | **True (Diff = 0)** | 1.19 / 11 | 1,766 (82.83%) | **8.93 / 68** | 2,131 (99.95%) | 8.66 / 63 | 2,132 (100.0%) |
| **Attempt 1** (CORD r61) | Replacement (`Populaire Chocolate` $\to$ `Magnum Gold Almond`) | $1356 \times 2315$ | 44,091 | **True (Diff = 0)** | 2.96 / 74 | 22,049 (50.01%) | **65.64 / 255** | 33,567 (76.13%) | 65.57 / 255 | 33,647 (76.31%) |
| **Attempt 2** (CORD r7) | Removal (`Discount: -9,545` $\to$ `""`) | $576 \times 864$ | 9,088 | **True (Diff = 0)** | 0.98 / 14 | 7,145 (78.62%) | **10.86 / 82** | 9,088 (100.0%) | 10.81 / 74 | 9,087 (99.99%) |
| **Attempt 3** (CORD r93) | Removal (`TAX 15,636` $\to$ `""`) | $864 \times 1296$ | 7,992 | **True (Diff = 0)** | 1.89 / 13 | 7,273 (91.00%) | **16.75 / 128** | 7,992 (100.0%) | 16.40 / 121 | 7,992 (100.0%) |

### Kiểm Toán Ranh Giới Mép Mask (2px Transition Boundary Band)

Định nghĩa dải ranh giới 2-pixel được khóa trước khi đo:
- **Inner band (1px)**: $M \setminus \text{erode}(M, 3\times 3)$ (1 pixel nằm ngay sát mép trong mask).
- **Outer band (1px)**: $\text{dilate}(M, 3\times 3) \setminus M$ (1 pixel nằm ngay sát mép ngoài mask).

| Attempt ID | Inner Px | $I_1 - I_0$ Inner Mean / Max | $I_2 - I_0$ Inner Mean / Max | Outer Px | $I_1 - I_0$ Outer Mean / Max | $I_2 - I_0$ Outer Mean / Max | Outside Leakage Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Attempt 0** | 212 | 0.95 / 4 | 5.56 / 17 | 220 | **0.00 / 0** | **0.00 / 0** | **ZERO LEAKAGE (PASS)** |
| **Attempt 1** | 1,380 | 0.12 / 12 | 0.87 / 202 | 1,388 | **0.00 / 0** | **0.00 / 0** | **ZERO LEAKAGE (PASS)** |
| **Attempt 2** | 628 | 0.68 / 3 | 8.21 / 22 | 636 | **0.00 / 0** | **0.00 / 0** | **ZERO LEAKAGE (PASS)** |
| **Attempt 3** | 512 | 1.74 / 8 | 7.55 / 40 | 520 | **0.00 / 0** | **0.00 / 0** | **ZERO LEAKAGE (PASS)** |

**Nhận xét**: 100% pixel ở outer band có sai khác bằng đúng 0 (`max_diff = 0`), chứng minh thuật toán composite hoàn toàn sạch, không có rò rỉ pixel ra ngoài ranh giới đã định.

---

## 3. Đánh Giá Trực Quan Sơ Bộ (Preliminary Visual Assessment)

Bảng ảnh tổng hợp 4 hàng độ phân giải cao đã được xuất tại:
`research/evidence/document_pilot/qc_inspection/geometry_control_contact_sheet.png` (Kích thước: $1690 \times 680$, dung lượng: 289 KB).

### Trả lời 5 câu hỏi trọng tâm:

1. **$I_1$ có làm mờ chữ hoặc tạo mép vá không?**
   - **Hoàn toàn không mờ chữ**: Trên cả 4 mẫu, toàn bộ ký tự trong $I_1$ giữ nguyên độ sắc nét ban đầu khi quan sát bằng mắt thường. Sai lệch số học $I_1 - I_0$ rất nhỏ (Mean L1 chỉ từ **0.98 đến 2.96** trên thang 255), tập trung hoàn toàn ở các pixel viền ký tự có độ tương phản cao do phép làm tròn nội suy bicubic 2 chiều.
   - **Hoàn toàn không tạo mép vá**: Nền giấy, đường bóng đổ chéo (Attempt 2) và nếp nhăn giấy (Attempt 3) đi xuyên qua ranh giới mask một cách trơn tru, không có bước nhảy độ sáng hay gián đoạn cấu trúc vật lý nào.
2. **Bất thường nào quan sát được ở cả $I_1$ và $I_2$?**
   - Không có bất thường thị giác vĩ mô nào xuất hiện ở cả hai điều kiện. Điểm chung duy nhất là sự hiện diện của sai số nội suy ở cấp độ sub-pixel tại biên ký tự (thể hiện qua sai lệch L1 vi mô).
3. **Bất thường nào thấy ở $I_2$ nhưng chưa thấy ở $I_1$?**
   - **Attempt 0**: Vùng inpainting bị sụp đổ hoàn toàn thành một mảng hình chữ nhật xám trơn nhẵn (text collapse), làm mất chữ số gốc nhưng không in ra số mới.
   - **Attempt 1**: Ký tự sinh ra bị sai chính tả ("MAGNU ALONLD" thay vì "Magnum Gold Almond") và font chữ in hoa sans-serif đậm lệch hoàn toàn kiểu chữ in nhiệt chấm ma trận của tài liệu.
   - **Attempt 2**: Xuất hiện một ô sáng phẳng đè lên và cắt ngang dải bóng đổ chéo tự nhiên của tài liệu.
   - **Attempt 3**: Vùng inpainting làm phẳng lì nếp gấp giấy (crease) và có sắc xanh lam nhạt, tạo thành một mảng nhẵn thín bất thường giữa tờ hóa đơn nhàu nát.
4. **Hai replacement có đạt target text không?**
   - **0 / 2 đạt**: Attempt 0 không sinh được chữ; Attempt 1 sinh sai chính tả và lệch kiểu font chữ.
5. **Hai removal xóa được chữ nhưng còn vấn đề chất lượng gì?**
   - **2 / 2 xóa được chữ** (văn bản mục tiêu biến mất 100%), nhưng **chất lượng nền giấy không đạt chuẩn chân thực**: cả hai đều tạo ra các mảng vá hình chữ nhật phẳng lì phá vỡ kết cấu tự nhiên (bóng đổ, nếp nhăn).
   - **Trạng thái Human QC**: Tiếp tục giữ nguyên `PENDING_HUMAN_REVIEW` trong `human_review.csv`. Đánh giá này là preliminary visual assessment từ agent, không thay thế cho phán quyết của người dùng.

---

## 4. Sổ Cái Đối Soát Mâu Thuẫn Báo Cáo (Reconciliation Ledger)

Dưới đây là bảng đối soát chi tiết 5 điểm mâu thuẫn được đối chiếu trực tiếp từ các artifact thô và mã nguồn thực thi:

| Mục đối soát | Nguồn TXT/Tóm tắt | Nguồn Markdown trước | Ground Truth từ Raw Artifacts | Kết luận & Nguyên nhân kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| **1. Model Revision** | `115134f3c051a82da171358b5a0429f4ff896e08` | `115134f363124c53c7d878647567d04daf26e41e` | **`115134f363124c53c7d878647567d04daf26e41e`** | **Khớp Markdown & Artifacts**. Chuỗi SHA trong TXT là lỗi sao chép văn bản (transcription erratum). Cả plan, receipt preflight và manifest Colab đều ghi nhận đúng revision `115134f36312...`. |
| **2. Max Pixel Difference** | 66 / 247 / 100 / 106 | 68 / 255 / 82 / 128 | **68 / 255 / 82 / 128** | **Khớp Markdown & Mảng Pixel Thật**. Tính trực tiếp từ mảng `uint8` có dấu ($I_2 - I_0$ inside mask) xác nhận max channel diff là 68, 255, 82, 128. Số trong TXT là không chính xác. |
| **3. Driver, PyTorch & CUDA** | Gộp chung `CUDA 13.0, PyTorch 2.11.0` | Gộp chung | **Torch: `2.11.0+cu130`**<br>**GPU: `Tesla T4` (14.56 GB)**<br>**Driver & CUDA Driver: `NOT_RECORDED`** | **Phân tách rành mạch**. Preflight receipt ghi nhận runtime PyTorch `2.11.0+cu130`. Host driver version không được lưu trong JSON receipt. Không đánh đồng CUDA supported version trong nvidia-smi với runtime CUDA của PyTorch. |
| **4. Loading & Inference Time** | Ghi nhận tổng thời gian | Ghi nhận tổng thời gian GPU ~4.5 phút | **Model Load: 140.35s**<br>**Inference: 129.91s** (32.93s, 30.81s, 32.95s, 33.22s)<br>**Tổng 2 hạng mục: 270.26s** | **Minh bạch phạm vi đo**. 270.26s là tổng của khâu nạp model (download + init) và 4 lần suy luận. Thời gian toàn notebook và thời gian GPU active thực tế là `NOT_RECORDED`. |
| **5. Model Pinning Policy** | Không đề cập | Ghi: "Default unpinned trên HF" | **`EXACT_PINNED_REVISION_115134f36312...`** | **Đính chính nhãn**. Canonical plan `document_feasibility_plan.json` đã pin cố định revision `115134f36312...` trước khi chạy. Việc ghi "Default unpinned" trong bảng trước là sơ suất soạn thảo. |

---

## 5. Chốt Kết Luận Pilot SDXL Hiện Tại (Pilot Closure)

1. **Khả năng thực thi (Generation Pipeline Feasibility)**:
   - Pipeline Colab T4 đã chứng minh khả năng thực thi khép kín: hoàn thành 100% 4/4 attempts, VRAM đỉnh 8.96 GiB ($\le 12$ GiB), thời gian suy luận ~32 giây/ảnh, kiểm soát bitwise equality ngoài mask đạt chuẩn tuyệt đối (`max_diff = 0`).
2. **Khả năng đạt mục tiêu replacement/removal**:
   - **Replacement**: Thất bại (0/2). Vanilla SDXL inpainting không có khả năng sinh ký tự vi mô chính xác (glyph-level control) dựa trên text prompt thông thường.
   - **Removal**: Đạt mục tiêu xóa chữ (2/2), nhưng thất bại về tính chân thực của nền giấy (background realism).
3. **Chất lượng vùng sửa**:
   - Vùng sửa để lại các mảng vá phẳng làm lộ rõ vết ghép (cắt ngang bóng đổ, làm phẳng nếp nhăn).
4. **Ảnh hưởng quan sát được của hình học (Geometry Impact)**:
   - Đối chứng $I_1$ chứng minh khâu xử lý hình học thông thường (crop, resize, letterbox, composite) hoạt động chính xác và không gây mờ chữ hay tạo mép vá giả. Mọi khiếm khuyết thị giác ở $I_2$ xuất phát từ chính mô hình tạo sinh diffusion và VAE decoding.
5. **Giới hạn về nguồn và generator**:
   - Cỡ mẫu $N=4$ là feasibility pilot; không suy rộng thành khẳng định SDXL nói chung "không thể thay chữ".
   - Vết vá lộ rõ trong pilot này KHÔNG đồng nghĩa với việc các detector sẽ dễ dàng phát hiện các tài liệu AI chất lượng cao được tạo bởi các công cụ chuyên dụng.

---

## 6. Đề Xuất Duy Nhất Cho Lượt Tiếp Theo (Direction Toward Detection)

Do đối chứng $I_1$ xác nhận module hình học đã đạt chuẩn kỹ thuật, hạn chế cốt lõi nằm ở **chất lượng tạo sinh của cấu hình vanilla SDXL inpainting** đối với văn bản tài liệu.

**Đề xuất duy nhất**: **Chọn phương án dữ liệu / editor phù hợp hơn cho bài toán phát hiện giả mạo tài liệu** thay vì mở rộng SDXL vanilla thành panel 40 nguồn:
- **Căn cứ**: Vanilla diffusion inpainting không thể sinh chữ đạt chuẩn OCR mà không có module text-rendering hoặc conditioned adapter (như AnyText hay TextDiffuser). Việc mở rộng 40 mẫu với SDXL vanilla sẽ chỉ nhân bản các mẫu lỗi text collapse và hallucinated typo, không đại diện cho tài liệu giả mạo thực tế có độ tinh vi cao.
- **Thiết kế tối thiểu cho lượt sau**:
  1. *Hướng ưu tiên dữ liệu có sẵn*: Sử dụng panel tài liệu đã được tiếp nhận và kiểm toán cách ly độc lập (như RealText-V2 $N=40$ đã có sẵn trên LOCAL) để thử nghiệm trực tiếp quan sát hình học P1 Letterbox Canvas trên các detector, thay vì cố gắng tự tạo sinh bằng diffusion thuần túy.
  2. *Nếu tiếp tục tạo sinh*: Giới hạn bài toán ở tác vụ Text Removal / Inpainting Nền Sạch (Clean Inpainting), hoặc thiết kế quy trình 2 bước (Typesetting quang học bằng font in nhiệt chuẩn $\to$ Diffusion blend mép) trên quy mô pilot nhỏ.
  - *Ranh giới cam kết*: Không tự ý tải checkpoint mới, không tạo cohort mới, và không huấn luyện khi chưa có sự phê duyệt của người dùng.

---

## 7. Hồ Sơ Bằng Chứng & Danh Mục Artifacts Bàn Giao

1. **Script chẩn đoán**: [diagnose_document_feasibility_geometry.py](../../../scripts/research/diagnose_document_feasibility_geometry.py)
2. **Biên nhận chẩn đoán & đối soát**: [document_feasibility_geometry_diagnostic_receipt.json](./document_feasibility_geometry_diagnostic_receipt.json)
3. **Bảng ảnh so sánh 4 hàng**: [geometry_control_contact_sheet.png](./qc_inspection/geometry_control_contact_sheet.png)
4. **Báo cáo chốt pilot**: [DOCUMENT_FEASIBILITY_PILOT_CLOSURE_REPORT.md](./DOCUMENT_FEASIBILITY_PILOT_CLOSURE_REPORT.md)
5. **Unit test xác thực**: `ml/tests/test_document_feasibility_geometry.py::test_geometry_control_i1_diagnosis_receipt` (PASS).
