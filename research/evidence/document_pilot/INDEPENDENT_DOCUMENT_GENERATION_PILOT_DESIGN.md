# Bản Thiết Kế Kỹ Thuật: Pilot Tự Xây Dựng Dữ Liệu Tài Liệu Giả Mạo AI Có Kiểm Soát
(Independent Controlled AI-Document Pilot Design)

> **Mã thiết kế**: `DESIGN-2026-10-12-INDEPENDENT-DOCUMENT-PILOT`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Trạng thái**: PENDING HUMAN REVIEW & DECISION (Chưa tải checkpoint, chưa chạy tạo ảnh hay huấn luyện)

---

## 1. Cơ Sở Thực Tiễn & Lý Do Thiết Kế Fallback

1. **Kết quả kiểm toán dữ liệu công khai (Audit Summary)**:
   - Toàn bộ các bộ dữ liệu tài liệu do Generative AI (Gemini 2.5 Flash, Ideogram v2, GPT-4o) tạo ra (`Scam-AI/AIForge-Doc-v1`, `v2`, `gpt4o-receipt`) đều nằm sau rào cản xác thực có điều kiện (**`gated: auto`**, yêu cầu người dùng gửi đơn thỏa thuận học thuật và cung cấp API token).
   - Người dùng đã đưa ra quyết định dứt khoát: **Không đăng nhập Hugging Face, không cấp token cá nhân, không tiếp tục xin mở quyền AIForge-Doc**.
   - Toàn bộ các bộ dữ liệu tài liệu công khai mở ungated (`TDoc-2.8M`, `RealText-V2-Syn25k`, `FSTS`, `DocTamper`) đều là **chỉnh sửa truyền thống** (copy-move, splicing, synthetic font rendering, Fourier PSD modeling), hoàn toàn không có nguồn gốc AI tạo sinh.
   - Bộ dữ liệu `RealText-V2` (ACM MM 2026) có 0/40 mẫu xác minh được AI, bị nhiễu định dạng 100% (JPEG vs PNG) và không ghép cặp nguồn.

2. **Nguyên tắc khoa học cốt lõi (Scientific Honesty)**:
   - Không thể trả lời câu hỏi *"khả năng phát hiện tài liệu giả mạo sinh bởi AI"* bằng các bộ dữ liệu chỉnh sửa Photoshop/copy-move truyền thống.
   - Để có một bộ dữ liệu **100% verified Generative AI provenance**, **100% paired 1:1 authentic-edited**, và **100% format-symmetric**, giải pháp độc lập duy nhất là **tự tạo dựng một cohort tài liệu inpainting có kiểm soát** theo phương pháp đã thành công ở Phase 4C.7B.
   - **Phân định ranh giới**: Tạo dữ liệu bằng mô hình AI (GenAI generation) là việc tạo ra mẫu thử nghiệm; hoàn toàn khác với việc huấn luyện mô hình phát hiện (detector training). Lượt này chỉ bàn giao thiết kế, tuyệt đối không tự ý tải weights hay tạo ảnh.

---

## 2. Đặc Tả Thiết Kế 10 Điểm Bắt Buộc

### 1. Nguồn Tài Liệu Gốc Công Khai & Bản Quyền
- **Bộ dữ liệu gốc**: **CORD v2** (*Consolidated Receipt Dataset for Post-OCR Parsing*, Park et al., NeurIPS 2019 / Clova AI).
  * **Giấy phép**: **CC-BY 4.0** (cho phép tự do sử dụng, sửa đổi và phân phối phục vụ nghiên cứu).
  * **Trạng thái tải**: Mở hoàn toàn (ungated), khả dụng trực tiếp từ official repository không cần token hay đăng nhập.
  * **Miền tài liệu**: Hóa đơn bán lẻ, biên lai thanh toán thương mại với bố cục bảng, dòng số, và chữ in chuẩn.
- **Nguồn đối chứng bổ trợ**: **SROIE 2019** (ICDAR 2019 Competition on Scanned Receipts OCR and Information Extraction), giấy phép học thuật mở.

### 2. Editor / Model Ứng Viên & Trọng Số Checkpoint
- **Ứng viên chuẩn tắc (Recommended)**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` hoặc community mirror `sd2-community/stable-diffusion-2-inpainting`.
  * **Giấy phép**: OpenRAIL++ / CreativeML Open RAIL.
  * **Tính khả dụng**: Đã được thẩm tra và cấu hình preflight thành công trong codebase dự án (`ml/evaluation/independent_cohort_acquisition.py`), có thể tải và chạy hoàn toàn tự động trên môi trường điện toán đám mây tạm thời mà không cần user authentication token.
- **Ứng viên thay thế**: `AnyText` (Alibaba DAMO Academy) - mô hình visual text generation/editing chuyên biệt có weights công khai trên ModelScope/GitHub.

### 3. Môi Trường Tạo Dữ Liệu Dự Kiến
- **Môi trường**: **Google Colab GPU** (NVIDIA T4 16GB VRAM hoặc A100) thông qua notebook chuẩn tắc tự động hóa (tương tự `notebooks/phase_4c7b_colab_launcher.ipynb`).
- **Nguyên tắc cách ly**: Toàn bộ quá trình tải checkpoint nặng và suy luận sinh ảnh diễn ra trên đĩa tạm thời (`/content/ephemeral/`) của máy ảo GPU; tuyệt đối không tải mô hình nặng về máy cá nhân (LOCAL CPU).

### 4. Ngân Sách GPU, Mạng & Lưu Trữ Dự Kiến
- **Tải mạng**:
  * Tải 40 ảnh gốc CORD: $\approx 15 - 25\text{ MB}$.
  * Chuyển giao kết quả hoàn chỉnh về LOCAL: 40 cặp (80 ảnh PNG $512 \times 512$ + 40 masks) $\approx 40 - 60\text{ MB}$.
  * Tổng chuyển giao mạng lũy kế: $\approx 85\text{ MB} \ll 1\text{ GB}$ hạn mức phiên.
- **Thời gian GPU**:
  * SD2 Inpainting (50 steps DDIM): $\approx 3.2\text{s}$ / attempt $\implies 40 \text{ attempts} \approx 128\text{s}$ (~2.2 phút).
  * SDXL Inpainting (30 steps Euler): $\approx 4.8\text{s}$ / attempt $\implies 40 \text{ attempts} \approx 192\text{s}$ (~3.2 phút).
  * Tổng thời gian GPU ước tính: $< 5$ phút.
- **Lưu trữ LOCAL**: Footprint ổ đĩa lưu trữ panel $\approx 50\text{ MB}$, hoàn toàn an toàn trong dung lượng trống của ổ đĩa.

### 5. Quy Mô Mẫu, Phân Tầng & Mặt Nạ
- **Quy mô panel**: Đúng **40 cặp nguồn (40 source pairs)**, tương đương **80 ảnh**:
  * 40 ảnh tài liệu nguyên bản (**40 authentic documents**);
  * 40 ảnh tài liệu bị sửa đổi bằng AI inpainting (**40 AI-inpainted documents**);
  * 40 mặt nạ nhị phân ground-truth (**40 binary masks** $512 \times 512$).
- **Phân tầng thao tác (Stratification)**:
  * 20 mẫu **Text Replacement** (thay đổi số tiền, ngày tháng, tên mặt hàng bằng inpainting có prompt định hướng);
  * 20 mẫu **Text Erasure / Inpainting Removal** (xóa bỏ dòng chữ hoặc dấu mộc bằng inpainting phông nền).
- **Phân bố diện tích mặt nạ**:
  * 20 mẫu Small / Minor ($< 5\%$ diện tích canvas: số tiền, ngày);
  * 20 mẫu Medium / Structural ($5\% - 20\%$ diện tích canvas: tiêu đề, bảng kê).

### 6. Quy Trình Khóa Prompt, Seed & Nhật Ký Tạo Dữ Liệu
- **Kế hoạch tiền đăng ký (Pre-registered Plan)**:
  * Xuất bản tệp kế hoạch máy trước khi chạy: `independent_document_pilot_plan.json`.
  * Mỗi attempt được khóa cứng: `sample_id`, `source_image_id`, `mask_bbox [ymin, xmin, ymax, xmax]`, `prompt`, `negative_prompt`, `guidance_scale=7.5`, `steps=50`, `seed` (khởi tạo từ PRNG cố định `SEED = 20261012 + index`).
- **Biên nhận thực thi máy (Generation Receipt)**:
  * Ghi lại chi tiết: phần cứng GPU, phiên bản PyTorch, Diffusers, CUDA, thời gian inference từng ảnh, peak VRAM, và SHA-256 từng tệp PNG sinh ra.

### 7. Quy Trình Tạo Đối Chứng Authentic Đạt Chuẩn Parity
- **Bảo toàn tính đối xứng định dạng (Format Parity)**:
  * Cả ảnh authentic và ảnh edited đều được cắt/chuyển đổi sang canvas $512 \times 512$ bằng cùng thuật toán nội suy Lanczos/Bicubic.
  * Cả hai đều được lưu dưới định dạng **PNG lossless 24-bit RGB**.
  * Loại trừ 100% định dạng shortcut (chấm dứt hiện tượng forged là JPEG còn authentic là PNG như RealText-V2).

### 8. Tiêu Chuẩn Kiểm Soát Chất Lượng (QC) Độc Lập
- **Technical QC (Tự động, Fail-Closed)**:
  * Ngoài vùng mask: `outside_mean_l1 = 0.000000` (pixel ngoài vùng can thiệp được bảo toàn bit-identical từ ảnh gốc qua composite mask).
  * Trong vùng mask: `inside_mean_l1 >= 3.0` (chứng minh mô hình thực sự can thiệp nội dung).
  * Kiểm tra ảnh không rỗng: `std >= 5.0`.
- **Human Content QC (Thẩm định độc lập)**:
  * Người thẩm tra kiểm tra trực quan tính hợp lý: không bị mờ nhòe dị dạng, không bị méo mó cấu trúc bảng.
  * **Tuyệt đối cấm**: Không sử dụng 6 frozen models để đánh giá và chọn ảnh đẹp/khó (ngăn chặn survivorship bias làm méo mó kết quả nghiên cứu).

### 9. Phân Chia Nguồn Tránh Rò Rỉ Dữ Liệu (Data Leakage Guards)
- Toàn bộ 40 ảnh CORD/SROIE độc lập tuyệt đối với COCO (Locked-Test, Dev train, Holdout) và ImageNet.
- Kiểm toán mã băm SHA-256 tự động đối soát với toàn bộ catalogs nội bộ trước khi cấp phép đánh giá (`ISOLATION_AUDIT_PASS`).

### 10. Giới Hạn Công Nghệ & Ranh Giới Khoa Học
- Các mô hình diffusion nguồn mở (SD2/SDXL) có thể gặp hiện tượng tạo chữ nhỏ không hoàn hảo (glyph distortion). Đây là đặc tính kỹ thuật thực tế của diffusion inpainting và là đối tượng nghiên cứu hợp lệ của bộ phát hiện giả mạo.
- Báo cáo kết quả sẽ ghi nhận trung thực generator được sử dụng, không ngụy tạo kết quả hoàn hảo.

---

## 3. Đúng Một Phương Án Đề Xuất Để Người Dùng Quyết Định

> **Đề xuất duy nhất**: Người dùng xem xét bản thiết kế trên. Nếu phê duyệt, phiên kế tiếp sẽ chuẩn bị mã nguồn kịch bản kế hoạch `build_document_pilot_plan.py` và Colab launcher notebook để người dùng chạy 5 phút trên Google Colab GPU miễn phí, xuất ra panel 40 cặp tài liệu chuẩn đối chứng hoàn toàn sạch sẽ và công khai.
