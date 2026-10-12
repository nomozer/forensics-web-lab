# Tu Chính Bản Thiết Kế Kỹ Thuật (Amendment v1.1.0)
## Thử Nghiệm Khả Thi Tạo Dữ Liệu Tài Liệu Giả Mạo AI Có Kiểm Soát
(Controlled AI-Document Generation Feasibility Study - Protocol Amendment v1.1.0)

> **Mã văn bản**: `DESIGN-AMENDMENT-2026-10-12-FEASIBILITY-STUDY-V1.1.0`<br>
> **Văn bản gốc được tu chính**: `INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN.md` (v1.0.0, bảo toàn nguyên bản làm lineage)<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Ngày phê duyệt và áp dụng**: 2026-10-12<br>
> **Trạng thái thực hiện**: PREPARED_NOT_EXECUTED (Đã hoàn thiện code, test, plan máy và notebook; chờ thực thi GPU trên Colab)

---

## 1. Mục Đích & Phạm Vi Tu Chính

Bản tu chính này đính chính các khẳng định vượt bằng chứng trong thiết kế v1.0.0, điều chỉnh ngân sách tài nguyên trung thực với việc tải checkpoint mô hình, và thiết lập hợp đồng hình học (Geometry Contract) cùng quy tắc QC hai tầng cho thử nghiệm khả thi (Feasibility Study) trên đúng **bốn tài liệu CORD v2** (2 text replacement, 2 text removal).

### Ranh Giới Nghiên Cứu Rõ Ràng
1. **Đây là Generation Feasibility Study**: Mục tiêu là kiểm chứng tính khả thi kỹ thuật của quy trình inpainting tài liệu (tải checkpoint, nạp dữ liệu, suy luận inpainting, bảo toàn hình học và kiểm chứng ngoài mask).
2. **Không phải Detector Training**: Tuyệt đối không huấn luyện mô hình phát hiện trong đợt này.
3. **Không phải Cohort Đánh Giá Độc Lập Hoàn Tất**: Bốn tài liệu này là mẫu phát triển kỹ thuật (`development feasibility sources`), được đánh dấu cách ly và loại trừ khỏi panel đánh giá 40 nguồn sau này.
4. **Không Thay Đổi Research Questions (RQs)**: Các câu hỏi nghiên cứu và 6 frozen detectors được bảo toàn nguyên vẹn.
5. **Không Dùng Detector Scores Để Chọn Ảnh**: Không chạy 6 detectors trên các mẫu sinh ra để lọc, xếp hạng hoặc chọn attempt thành công.

---

## 2. Đính Chính Các Khẳng Định Thiết Kế (Design Rectifications)

Bản tu chính v1.1.0 sửa đổi dứt khoát 7 khẳng định trong bản thiết kế v1.0.0 nhằm tuân thủ nguyên tắc trung thực khoa học (Scientific Honesty):

| STT | Khẳng định cũ trong v1.0.0 | Khẳng định đính chính trong Amendment v1.1.0 | Căn cứ khoa học |
| :--- | :--- | :--- | :--- |
| **1** | *"Không tồn tại dataset mở"* | **“Chưa tìm được nguồn đáp ứng điều kiện trong các ứng viên đã kiểm tra.”** | Tránh khái quát hóa tuyệt đối khi chỉ mới kiểm toán một tập con các ứng viên công khai. |
| **2** | *"Tự tạo dữ liệu là giải pháp độc lập duy nhất"* | **“Tự tạo dữ liệu là một trong các phương án kỹ thuật được xem xét khi các ứng viên đã kiểm tra không đáp ứng tiêu chí provenance.”** | Không tuyên bố đây là phương án duy nhất khả dĩ của toàn bộ lĩnh vực. |
| **3** | *"Cam kết thời gian < 5 phút và GPU T4/A100"* | **“Ước tính thời gian thực tế phụ thuộc hạ tầng Colab cấp phát ngẫu nhiên; thiết lập trần ngân sách thực thi là 60 phút.”** | Google Colab có thể cấp phát T4, V100 hoặc GPU khác tùy thời điểm, không cam kết trước. |
| **4** | *"PNG parity loại bỏ 100% confounding"* | **“PNG parity loại bỏ định dạng nén JPEG như một shortcut rõ ràng, nhưng không loại bỏ toàn bộ các confounding tiềm ẩn khác (nhiễu cảm biến, nội suy canvas).”** | Định dạng đồng nhất là điều kiện cần, không phải điều kiện đủ để triệt tiêu mọi biến gây nhiễu. |
| **5** | *"Zero hash collisions là độc lập tuyệt đối"* | **“Zero hash collisions chứng minh không trùng lặp bit với các tập kiểm tra đã có, là một chỉ báo cô lập dữ liệu cần thiết nhưng không thay thế cho phân tích semantic similarity.”** | Tránh đồng nhất hash uniqueness với tính độc lập thống kê hoàn toàn. |
| **6** | *"Biến đổi pixel (L1 >= 3.0) là thành công thay nội dung"* | **“Biến đổi pixel chỉ chứng minh mô hình có can thiệp số học; thành công ngữ nghĩa (thay đúng từ, xóa đúng dòng, văn bản đọc được) bắt buộc phải qua QC nội dung.”** | Thước đo L1 chỉ là descriptive statistic, không phản ánh tính chính xác ngữ nghĩa (semantic correctness). |
| **7** | *"Dữ liệu tự tạo không vướng bất kỳ điều kiện bản quyền nào"* | **“Dữ liệu kế thừa giấy phép nguồn CORD v2 (CC-BY 4.0) và tuân thủ giấy phép mô hình SDXL Inpainting (OpenRAIL++). Cần tuân thủ đầy đủ điều kiện ghi nhận tác giả (attribution) và phạm vi sử dụng hợp pháp.”** | Bản quyền và điều khoản sử dụng của tập dữ liệu gốc và trọng số mô hình vẫn áp dụng đầy đủ. |

---

## 3. Tách Ngân Sách Tài Nguyên 6 Thành Phần (Resource Budget Separation)

Thiết kế v1.0.0 ước tính sai tổng tải mạng là "85 MB" do bỏ sót trọng số mô hình diffusion. Bản tu chính v1.1.0 thiết lập ngân sách tách bạch 6 thành phần cho thử nghiệm feasibility:

| Thành phần ngân sách | Trần định mức tối đa | Ước tính thực tế (Feasibility 4 attempts) | Ghi chú & Ràng buộc |
| :--- | :--- | :--- | :--- |
| **1. Checkpoint trên COLAB** | **12.0 GiB** | **6.46 GiB (6,941,218,469 bytes)** | Chỉ tải variant `fp16` safetensors của `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Không tải FP32. |
| **2. Dữ liệu gốc CORD v2** | **200 MB** | **$\approx 5 - 10$ MB** | Đọc metadata và 4 ảnh CORD-v2 qua HTTP Range request từ Hugging Face Parquet. |
| **3. Cài đặt Dependencies** | **2.0 GiB** | **$\approx 1.2$ GiB** | `diffusers`, `transformers`, `accelerate`, `safetensors`, `pyarrow` trên môi trường Colab. |
| **4. Artifacts chuyển về LOCAL** | **100 MB** | **$\approx 15 - 30$ MB** | Zip archive gồm: 4 authentic, 4 edited, 4 masks, 4 contexts, 4 receipts, contact sheet và CSV. |
| **5. Dung lượng đĩa & Peak VRAM** | **15 GB đĩa / 13 GB VRAM** | **$\approx 9$ GB đĩa / $\approx 10.5$ GB VRAM** | SDXL FP16 pipeline yêu cầu $\approx 10$ GB VRAM cho canvas $1024 \times 1024$. |
| **6. Thời gian thực thi GPU** | **60 phút** | **$\approx 10 - 15$ phút** | Download checkpoint ($\approx 4-6$p) + load model ($\approx 1$p) + inference 4 attempts ($\approx 3-5$p). |

> **Nguyên tắc tuyệt đối**:
> - Không tải checkpoint nặng (6.46 GiB) về LOCAL CPU.
> - Không commit checkpoint hoặc ảnh nhị phân lớn vào Git repository.
> - Nếu quá trình preflight cho thấy bất kỳ thành phần nào vượt trần ngân sách, dừng ngay trước khi tải và ghi nhận blocker.

---

## 4. Đặc Tả Mô Hình & Checkpoint Đã Khóa (Frozen Editor Specification)

Thử nghiệm sử dụng đúng **một checkpoint duy nhất**, đã được thẩm tra và khóa cứng:
- **Repository**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- **Revision Git SHA**: `115134f363124c53c7d878647567d04daf26e41e`
- **Giấy phép**: `OpenRAIL++`
- **Trạng thái truy cập**: `gated: False`, `private: False` (hoàn toàn mở công khai, không cần Hugging Face token hay đăng nhập).
- **Pipeline Class**: `StableDiffusionXLInpaintPipeline` (`diffusers`)
- **Variant & Dtype**: `torch.float16`, `variant="fp16"`, định dạng `.safetensors`.
- **Scheduler**: `EulerDiscreteScheduler` (`scheduler_config.json` khóa cứng theo repo).
- **Generation Canvas Resolution**: $1024 \times 1024$ pixels (đúng độ phân giải native chuẩn của SDXL).
- **Tham số suy luận khóa trước**:
  * `num_inference_steps`: 30
  * `guidance_scale`: 7.5
  * `strength`: 0.99 (inpainting thay thế vùng mask)
  * `seeds`: Tất định theo từng attempt từ base seed `20261012`.

> **Quy tắc không thay thế**: Không tự ý đổi sang Stable Diffusion v2, AnyText hoặc checkpoint khác nếu gặp lỗi. Nếu model này gặp blocker, dừng và báo cáo phương án kỹ thuật cần khắc phục.

---

## 5. Bốn Tài Liệu Phát Triển CORD v2 (Development Feasibility Sources)

- **Nguồn dữ liệu**: `naver-clova-ix/cord-v2` (CC-BY 4.0), split: `validation`, revision: `7f0115a4b758a71d6473b8d085751692da2fef98`.
- **Phương pháp chọn**: Chọn tất định bằng seed `20261012` trên tập `validation` (100 receipts) dựa trên annotation đầy đủ (tọa độ bbox quadrilateral, nhãn trường OCR, văn bản trước can thiệp).
- **Phân bổ 4 attempts**:
  1. **Attempt 1 (Replacement)**: Thay đổi số tiền/đơn giá (`menu.price`) bằng số tiền giả mạo.
  2. **Attempt 2 (Replacement)**: Thay đổi tên món hàng (`menu.nm`) bằng tên món mới.
  3. **Attempt 3 (Removal)**: Xóa bỏ dòng chiết khấu / giảm giá (`subtotal.discount_price` hoặc `menu.discount`).
  4. **Attempt 4 (Removal)**: Xóa bỏ dòng thuế / phí phục vụ (`subtotal.tax_price` hoặc `subtotal.service_charge`).

> **Ghi chú cách ly dữ liệu**: Cả 4 tài liệu này được gán nhãn `DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION` và bị loại trừ vĩnh viễn khỏi bất kỳ cohort đánh giá độc lập 40 nguồn nào trong tương lai để ngăn ngừa rò rỉ dữ liệu (data contamination).

---

## 6. Hợp Đồng Hình Học (Geometry Contract)

Để khắc phục triệt để lỗi làm bẹp/méo tài liệu khi ép vào ảnh vuông, hợp đồng hình học thiết lập quy trình:
1. **Lưu trữ ảnh gốc nguyên bản**: Lưu một bản decoded, orientation-correct RGB ở kích thước và tỷ lệ gốc $(W_{orig}, H_{orig})$ cho mỗi tài liệu.
2. **Context Crop có đệm**:
   - Xác định bounding box can thiệp `[ymin, xmin, ymax, xmax]` trong tọa độ gốc.
   - Mở rộng vùng đệm ngữ cảnh (context margin $\approx 20\% - 50\%$) xung quanh vùng can thiệp để mô hình inpainting có đủ thông tin nền chữ và cấu trúc xung quanh: `[c_ymin, c_xmin, c_ymax, c_xmax]`.
3. **Canvas trung gian $1024 \times 1024$**:
   - Đưa context crop vào canvas $1024 \times 1024$ giữ nguyên tỷ lệ khung hình (aspect-ratio preserving resize), căn giữa và đệm viền (letterbox/padding).
   - Nội suy ảnh dùng Bicubic; nội suy binary mask dùng Nearest-Neighbor để bảo toàn giá trị nhị phân thuần túy $\{0, 255\}$.
4. **Ánh xạ ngược & Composite (Inverse Mapping & Binary Composite)**:
   - Cắt bỏ phần padding từ output của mô hình inpainting.
   - Resize patch trở lại kích thước context crop ban đầu $(W_{crop}, H_{crop})$ bằng Bicubic.
   - Dán ngược patch vào tọa độ gốc của ảnh tài liệu.
   - **Composite chỉ thực hiện bên trong Final Intervention Mask**:
     $$I_{edited}(x, y) = \begin{cases} I_{inpainted}(x, y) & \text{nếu } M_{final}(x, y) = 255 \\ I_{authentic}(x, y) & \text{nếu } M_{final}(x, y) = 0 \end{cases}$$
5. **Kiểm tra bất biến ngoài mask (Outside Invariant)**:
   - Bắt buộc kiểm tra `np.array_equal(I_{edited}[M == 0], I_{authentic}[M == 0]) == True`.
   - Nếu có bất kỳ pixel nào ngoài mask khác biệt (sai khác dù chỉ 1 đơn vị L1), attempt đó bị đánh dấu thất bại kỹ thuật (`TECHNICAL_QC_FAIL`).

Lưu riêng biệt 6 thành phần artifact cho mỗi attempt:
1. `authentic_control.png` (ảnh gốc kích thước chuẩn)
2. `generation_context.png` (context crop đưa vào canvas)
3. `raw_model_output.png` (ảnh sinh ra từ SDXL trước khi dán ngược)
4. `final_edited_document.png` (ảnh tài liệu hoàn thiện sau composite)
5. `final_intervention_mask.png` (mặt nạ nhị phân ở tọa độ ảnh gốc)
6. `geometry_receipt.json` (tọa độ crop, hệ số tỷ lệ, padding, offsets)

---

## 7. Tiêu Chuẩn Kiểm Soát Chất Lượng (QC) Hai Tầng

### A. Technical QC (Tự động - Fail-Closed)
- File ảnh đọc được, đúng hệ màu RGB và đúng kích thước gốc $(W_{orig}, H_{orig})$.
- Mặt nạ nhị phân $\{0, 255\}$, không rỗng ($\text{sum} > 0$), khớp tọa độ can thiệp.
- Outside-mask equality: `np.array_equal` tuyệt đối bằng `True` (bảo toàn 100% pixel gốc).
- Inside-mask statistics: Ghi nhận diện tích thay đổi, mean L1, max L1, số pixel biến đổi.
- Bắt buộc tạo **Contact Sheet** trực quan gồm 5 panel: (1) Authentic gốc, (2) ROI Crop trước sửa, (3) ROI Crop sau sửa, (4) Intervention Mask, (5) Final Edited Document.

### B. Content QC (Thẩm định Ngữ Nghĩa - Phân Biệt Human vs Preliminary)
- **Text Replacement**:
  * Target text có thực sự xuất hiện không?
  * Ký tự có đọc được (legible) không hay bị méo mó (glyph hallucination)?
  * Cấu trúc bảng/dòng xung quanh có bị phá hỏng không?
- **Text Removal**:
  * Nội dung mục tiêu đã được xóa hoàn toàn chưa?
  * Vùng nền thay thế có hòa nhập tự nhiên với kết cấu giấy/hóa đơn không?
  * Bố cục tài liệu có bị biến dạng không?
- **Trạng thái thẩm định**:
  * Hệ thống tự động chỉ đưa ra `preliminary_assessment` (Đánh giá sơ bộ của AI).
  * Khởi tạo file `human_review.csv` với trạng thái `PENDING_HUMAN_REVIEW`.
  * Tuyệt đối không tự ý gán `HUMAN_PASS` khi chưa có người thẩm tra thật xác nhận.

---

## 8. Hướng Dẫn Thực Thi Phân Tách LOCAL vs COLAB

1. **Trên LOCAL CPU**:
   - Chạy `python scripts/research/build_document_feasibility_plan.py` để sinh kế hoạch `document_feasibility_plan.json`.
   - Kiểm thử toàn bộ modules và unit tests: `pytest ml/tests/test_document_feasibility_geometry.py`.
   - Xuất bản notebook `notebooks/independent_document_feasibility_colab.ipynb`.
2. **Trên GOOGLE COLAB GPU**:
   - Người dùng mở notebook bằng tài khoản Google thông thường (không cần đăng nhập HF hay token).
   - Chọn runtime GPU (T4 hoặc tương đương).
   - Chạy các cell tuần tự: cài dependencies, tải checkpoint FP16, thực thi 4 attempts, và tải `document_feasibility_results.zip`.
3. **Kiểm chứng tại LOCAL sau khi tải archive**:
   - Chạy `python scripts/research/verify_document_feasibility_artifacts.py --zip document_feasibility_results.zip`.
   - Kiểm tra outside array equality, sinh contact sheet và xuất receipt kiểm toán.
