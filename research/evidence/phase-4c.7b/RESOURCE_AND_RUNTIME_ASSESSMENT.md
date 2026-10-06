# Resource and Runtime Assessment: Phase 4C.7B Independent Cohort Acquisition

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment
> **Status**: `ASSESSMENT_COMPLETED_EMPIRICALLY_VERIFIED`
> **Audit Date (UTC)**: 2026-10-07T01:45:00Z
> **Host Environment**: Windows 11 Host, Python 3.12 (`ml/.venv`)

---

## 1. Kết Quả Đo Lường Hạ Tầng Thực Tế (Empirical Hardware Audit)

Số liệu được đo trực tiếp tại phiên làm việc qua lệnh hệ thống và thư viện PyTorch/psutil:

| Thành Phần Phần Cứng | Thông Số Đo Được | Ngưỡng Khả Dụng / Trạng Thái |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce GTX 1650 | CUDA 12.1 khả dụng, Compute Capability 7.5 |
| **GPU VRAM** | **4.00 GB GDDR6** | Cố định, không thể mở rộng |
| **System RAM Total** | **15.78 GB** | DDR4 |
| **System RAM Available** | **2.44 GB** (dao động 2.1 - 2.5 GB) | Hạn chế do tiến trình nền hệ điều hành |
| **Ổ đĩa C: (System)** | **52.1 GB Free** (trên tổng 350 GB) | Đủ cho cache tạm thời |
| **Ổ đĩa D: (Data/Workspace)** | **79.0 GB Free** (trên tổng 134 GB) | Rất dồi dào cho dataset và artifacts |

---

## 2. Ước Tính Chi Tiết Từng Hạng Mục Tài Nguyên

Tuân thủ quy tắc trung thực khoa học: **Không tuyên bố GPU 4GB hoặc 2GB RAM là đủ cho SDXL khi chưa đo đạc**.

### 2.1. Dung Lượng Trọng Số và Bộ Nhớ Tải Về (Weights & HuggingFace Cache)
- **Stable Diffusion 2 Inpainting (`stabilityai/stable-diffusion-2-inpainting`)**:
  - UNet (fp16): ~1.7 GB
  - Text Encoder (OpenCLIP ViT-H/14): ~1.3 GB
  - VAE + Scheduler configs: ~350 MB
  - *Tổng dung lượng tải*: **~3.5 GB**
- **SDXL Inpainting 1.0 (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`)**:
  - UNet 2.6B params (fp16): ~5.1 GB
  - Dual Text Encoders (OpenCLIP ViT-bigG + CLIP ViT-L): ~1.8 GB
  - VAE + Tokenizers: ~350 MB
  - *Tổng dung lượng tải*: **~7.25 GB**
- **Tổng dung lượng tải hai checkpoints**: **~10.75 GB**.

### 2.2. Dung Lượng Ảnh Nguồn, Ảnh Master, Ảnh Edited và Masks
- 440 ảnh nguồn Authentic thô (JPEG/PNG, tải qua API/URL): ~250 MB
- 400 ảnh Master Authentic đã chuẩn hóa ($512 \times 512$ PNG RGB lossless): ~160 MB
- 400 ảnh Mask nhị phân ($512 \times 512$ PNG 8-bit / 1-bit lossless): ~15 MB
- 400 ảnh AI Edited đã chuẩn hóa ($512 \times 512$ PNG RGB lossless): ~180 MB
- Provenance manifests, receipts, attempt ledgers: ~5 MB
- *Tổng dung lượng lưu trữ Cohort*: **~610 MB** (hoàn toàn an toàn trong 79.0 GB đĩa D:).

### 2.3. Bộ Nhớ GPU/RAM Khi Chạy (Peak Inference Memory)
- **Stable Diffusion 2.0 Inpainting**:
  - Yêu cầu VRAM tối thiểu (fp16, batch size 1): ~3.8 GB – 4.5 GB.
  - Trên GTX 1650 (4.00 GB VRAM): Chạy ở ngưỡng báo động (critical border). Nếu bật `enable_sequential_cpu_offload`, pipeline chuyển từng layer sang CPU, nhưng đòi hỏi RAM hệ thống khả dụng $\ge 6.0$ GB.
  - Hiện tại RAM khả dụng chỉ có 2.44 GB, nguy cơ OS paging và swapping cực kỳ cao.
- **SDXL Inpainting 1.0**:
  - Yêu cầu VRAM tối thiểu (fp16): **~6.5 GB – 8.0 GB**.
  - Không thể chạy suy luận trực tiếp trên VRAM 4.00 GB của GTX 1650 (CUDA Out of Memory 100% xảy ra ngay bước load UNet).
  - Sử dụng CPU Offloading cho SDXL đòi hỏi tối thiểu **12 GB RAM hệ thống còn trống**, vượt xa 2.44 GB RAM thực tế của máy trạm local.

### 2.4. Thời Gian Tạo Sinh (Generation Latency)
- **Trên GTX 1650 (nếu dùng sequential offload cho SD2)**:
  - SD2 (50 steps DDIM): ~35 – 50 giây / ảnh. Tạo 200 ảnh mất ~2.5 giờ.
  - SDXL: Không khả thi (Infeasible).
- **Trên Google Colab T4 GPU (15.0 GB GDDR6 VRAM, 12.7 GB System RAM)**:
  - SD2 (50 steps DDIM, native GPU fp16): ~3.5 giây / ảnh $\rightarrow$ 200 ảnh mất ~12 phút.
  - SDXL (30 steps Euler, native GPU fp16): ~7.5 giây / ảnh $\rightarrow$ 200 ảnh mất ~25 phút.
  - *Tổng thời gian tạo sinh toàn bộ 400 cặp trên Colab T4*: **~37 – 45 phút**.

---

## 3. Kết Luận Phân Tuyến Thực Thi (Execution Strategy)

1. **Môi Trường Local (GTX 1650 4GB / RAM 2.44GB)**:
   - Dùng cho: Pipeline engineering, schemas, option P historical exclusion checks, sample selection seeds, technical smoke test trên mock/fixture pipeline, test suites, continuity checkers, CLI wrappers.
   - Không ép buộc tải 10.75 GB checkpoints và chạy SDXL trên local để tránh sập bộ nhớ (OOM/thrashing).
2. **Môi Trường Remote Khả Dụng (Google Colab T4)**:
   - Chuẩn bị sẵn file notebook tự động hóa `notebooks/independent_cohort_acquisition_colab.ipynb` và gói chỉ thị chuyển giao (handoff instructions).
   - Thao tác người dùng yêu cầu: Chỉ cần mở notebook trên Colab, chọn Runtime T4 GPU miễn phí, và nhấn "Run All".
   - Notebook sẽ tự động nạp danh sách 440 candidates đã được runner chuẩn bị sẵn, thực thi inpainting bằng weights chính thức, kiểm tra Technical QC, và xuất gói ZIP cohort cùng checksum.
3. **Tuyệt Đối Tuân Thủ Đạo Đức Nghiên Cứu**:
   - Không giả vờ đã chạy Colab khi chưa có runtime thực tế.
   - Không đăng ký dịch vụ đám mây trả phí.
   - Không yêu cầu credentials hoặc token bảo mật của người dùng trong chat.
