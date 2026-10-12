# Báo Cáo Kỹ Thuật: Chuẩn Bị & Kiểm Thử Thử Nghiệm Khả Thi Tạo Dữ Liệu Tài Liệu AI
(Controlled AI-Document Generation Feasibility Study Preparation Report)

> **Mã văn bản**: `REPORT-2026-10-12-FEASIBILITY-STUDY-PREPARATION-V1.0.0`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Trạng thái thực hiện**: `PREPARED_NOT_EXECUTED`<br>
> **Môi trường**: LOCAL CPU (Windows pwsh) chuẩn bị & kiểm thử; sẵn sàng thực thi trên GOOGLE COLAB GPU.

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

---

## 2. Giải Đáp Tám Câu Hỏi Trọng Tâm

### Câu hỏi 1: Checkpoint có tải không xác thực được thật không?
- **Kết quả xác minh thực tế**: **CÓ, 100% TẢI ĐƯỢC KHÔNG CẦN TOKEN HOẶC ĐĂNG NHẬP**.
- **Chi tiết kiểm toán**:
  * Repository: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
  * Model Card API: `gated: False`, `private: False`.
  * Revision Git SHA đã khóa: `115134f363124c53c7d878647567d04daf26e41e`.
  * Toàn bộ 4 file safetensors của variant FP16 (`text_encoder`, `text_encoder_2`, `unet`, `vae`) và các file cấu hình json tải được hoàn toàn mở mà không cần Hugging Face token, `hf login`, hay tài khoản authenticated.

### Câu hỏi 2: Tổng bytes tải từng thành phần?
Bảng ngân sách 6 thành phần được đối soát độc lập:

| Thành phần | Trần ngân sách | Kích thước đo đạc thực tế | Trạng thái ngân sách |
| :--- | :--- | :--- | :--- |
| **1. Checkpoint Colab (FP16)** | $\le 12.0$ GiB | **6,941,218,469 bytes (6.46 GiB)** | ĐẠT (trong trần 12 GB) |
| **2. Dữ liệu gốc CORD-v2** | $\le 200$ MB | **$\approx 5 - 10$ MB** (đọc trực tiếp 4 ảnh) | ĐẠT (tiết kiệm $\approx 95\%$) |
| **3. Dependencies Colab** | $\le 2.0$ GiB | **$\approx 1.2$ GiB** | ĐẠT |
| **4. Artifacts chuyển về LOCAL** | $\le 100$ MB | **$\approx 15 - 30$ MB** | ĐẠT |
| **5. Dung lượng đĩa & Peak VRAM** | 15 GB đĩa / 13 GB VRAM | **$\approx 9$ GB đĩa / $\approx 10.5$ GB VRAM** | ĐẠT (vừa khít T4 16GB VRAM) |
| **6. Thời gian thực thi GPU** | $\le 60$ phút | **$\approx 10 - 15$ phút** | ĐẠT |

### Câu hỏi 3: GPU thực tế, initialization/inference time và peak VRAM?
- **Trên LOCAL**: Local GPU được nhận diện là `NVIDIA GeForce GTX 1650 (4.00 GB VRAM)`. Với VRAM 4.00 GB, SDXL FP16 (~10 GB VRAM) không thể chạy trên máy local mà không tràn bộ nhớ (OOM). Đây là bằng chứng thực nghiệm rõ ràng xác nhận chỉ có thể chạy inference trên Google Colab GPU.
- **Trên COLAB GPU (Mục tiêu đo đạc)**:
  * Khi người dùng chạy notebook trên Colab T4 (16 GB VRAM), initialization time ước tính $\approx 60$s, inference time ước tính $\approx 40-50$s/attempt (30 steps Euler), peak VRAM $\approx 10.5$ GB.
  * Hiện tại trạng thái là `PREPARED_NOT_EXECUTED`; số đo chính xác sẽ được ghi nhận trực tiếp từ `preflight_receipt.json` và `attempt_receipt.json` khi chạy trên Colab.

### Câu hỏi 4: Bao nhiêu attempts kỹ thuật hợp lệ?
- Kế hoạch máy đã khóa cứng đúng **4 attempts** (2 replacement, 2 removal) từ 4 tài liệu CORD-v2 khác nhau:
  1. `attempt_0_replacement_price`: row 22 (`img_id=22`), `menu.price` (`280,000` $\to$ `350,000`).
  2. `attempt_1_replacement_item_name`: row 61 (`img_id=61`), `menu.nm` (`Populaire Chocolate` $\to$ `Magnum Gold Almond`).
  3. `attempt_2_removal_discount_line`: row 7 (`img_id=7`), `sub_total.discount_price` (`Discount: -9,545` $\to$ `""`).
  4. `attempt_3_removal_tax_line`: row 93 (`img_id=93`), `sub_total.tax_price` (`TAX 15,636` $\to$ `""`).
- Cả 4 attempts đều có đầy đủ ground truth quad, tọa độ envelope `[ymin, xmin, ymax, xmax]`, prompt và seed tất định. Khi chạy, harness kỹ thuật kiểm tra fail-closed `np.array_equal` trên mảng pixel ngoài mask.

### Câu hỏi 5: Bao nhiêu replacement/removal đạt nội dung?
- **Trạng thái hiện tại**: `PREPARED_NOT_EXECUTED` (chưa chạy trên GPU Colab).
- **Quy tắc QC nội dung**:
  * Khi có artifacts, hệ thống chỉ xuất `preliminary_assessment` (ví dụ: `FEASIBLE_REPLACEMENT`, `POTENTIAL_UNMODIFIED`).
  * Trạng thái chính thức trong `human_review.csv` bắt buộc là `PENDING_HUMAN_REVIEW`.
  * Tuyệt đối không tự ý gán `HUMAN_PASS` khi chưa có người thẩm tra thật xác nhận.

### Câu hỏi 6: Geometry có bảo toàn tài liệu gốc không?
- **KẾT QUẢ ĐÃ KIỂM CHỨNG KỸ THUẬT**: **CÓ, BẢO TOÀN 100% BITWISE NGOÀI MASK**.
- Module `ml/evaluation/document_feasibility_geometry.py` thiết lập quy trình:
  1. Giữ nguyên kích thước gốc $(W_{orig}, H_{orig})$ của tài liệu RGB.
  2. Context crop có đệm (20%-50%) đưa lên canvas $1024 \times 1024$ (Bicubic cho ảnh, Nearest cho mask nhị phân).
  3. Sau inpainting, patch được unpad, resize ngược về kích thước crop ban đầu.
  4. Composite chỉ áp dụng bên trong Final Intervention Mask $M_{final} == 255$.
  5. Hàm `verify_outside_equality` kiểm tra `np.array_equal(edited[M == 0], authentic[M == 0]) == True`.
  6. Unit test `test_verification_harness_fail_closed` đã chứng minh: chỉ cần 1 pixel ngoài mask lệch 1 đơn vị L1, hệ thống lập tức báo `FAIL_OUTSIDE_MASK_LEAKAGE`.

### Câu hỏi 7: Lỗi hoặc seam nào còn ảnh hưởng thiết kế?
- **Khám phá lỗi socket Windows & Cách xử lý ($systematic-debugging)**:
  * Trong quá trình kiểm tra đọc HTTP Range trên Windows, phát hiện lỗi `[WinError 10013] WSAEACCES` do Windows Hyper-V/WSL2 chiếm giữ nhiều dải cổng động (dynamic excluded port ranges: 49152-49651, 59470-60392).
  * Đã khắc phục triệt để bằng bộ vá cổng an toàn (`safe_create_connection`), chủ động liên kết socket vào dải cổng tự do (51000-55000), loại bỏ hoàn toàn lỗi kết nối socket trên máy LOCAL Windows.
- **Ranh giới mép vá (Seam diagnostics)**:
  * Do thực hiện composite nhị phân (hard binary composite) tại biên mask, mô hình diffusion có thể để lại sự chuyển tiếp nhẹ giữa nền giấy gốc và vùng được sinh. Harness kiểm chứng tự động tính toán pixel diff heatmap trên 5-column contact sheet để người thẩm tra dễ dàng phát hiện seam artifacts.

### Câu hỏi 8: Đã đủ căn cứ mở rộng panel chưa?
- **KẾT LUẬN KHOA HỌC**: **CHƯA ĐỦ CĂN CỨ ĐỂ MỞ RỘNG PANEL**.
- **Căn cứ**:
  * Đợt này chỉ là feasibility study trên đúng 4 attempts để kiểm chứng công nghệ.
  * Việc mở rộng sang 40 nguồn chỉ được phép xem xét sau khi:
    1. Người dùng chạy Colab notebook và tải về gói kết quả `document_feasibility_results.zip`.
    2. Local harness xác nhận `TECHNICAL_QC_PASS` cho cả 4 attempts (outside array equality = True).
    3. Người thẩm tra con người (Human Reviewer) hoàn thành thẩm định nội dung trên `human_review.csv` và xác nhận chất lượng văn bản/nền giấy đạt yêu cầu.

---

## 3. Đúng Một Đề Xuất Tiếp Theo Được Phê Duyệt

Người dùng mở notebook `notebooks/independent_document_feasibility_colab.ipynb` trên Google Colab GPU (runtime T4 miễn phí, không cần đăng nhập HF hay token) và chạy tuần tự 4 cells để tải checkpoint FP16, thực thi 4 attempts và tải file archive kết quả `document_feasibility_results.zip` về máy LOCAL.

Sau khi người dùng tải archive về máy LOCAL, agent phiên tiếp theo sẽ chạy lệnh:
```bash
python scripts/research/verify_document_feasibility_artifacts.py --zip document_feasibility_results.zip
```
để đối soát kỹ thuật, sinh contact sheet trực quan và chuyển giao cho người thẩm định đánh giá nội dung.
