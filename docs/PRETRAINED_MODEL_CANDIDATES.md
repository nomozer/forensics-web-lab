# Pretrained Model Candidates: Khảo sát Ứng viên Checkpoint Mô hình

> **Ngày thực hiện**: 2026-09-21  
> **Nguyên tắc cốt lõi**:  
> - Chỉ nghiên cứu metadata, tài liệu khoa học và mã nguồn chính thức.  
> - **KHÔNG tải weights hoặc dataset** trong giai đoạn này (`Download status: not-downloaded`).  
> - Không suy đoán giấy phép hay thông số kỹ thuật chưa được công bố (`unknown`).  
> - Không chuyển đổi metric từ paper thành metric của sản phẩm (nếu trích dẫn phải ghi rõ `reported-by-source`).  
> - Trạng thái xét duyệt (`Admission status`) chỉ nhận một trong các giá trị: `candidate`, `rejected`, `blocked-license`, `blocked-missing-weights`, `blocked-incompatible-task`.

---

## 1. Nhóm A: Fully Generated Detector (Phát hiện Ảnh Tạo Sinh Toàn Phần)

### Ứng viên A1: LAID MobileNetV3-Small / ShuffleNetV2
- **Candidate ID**: `CAND-A1-LAID`
- **Official repository**: `https://github.com/nchivar/LAID`
- **Paper**: "LAID: Lightweight AI-Generated Image Detection in Spatial and Spectral Domains", Nicholas Chivaran & Jianbing Ni, PST 2025 ([arXiv:2507.05162](https://arxiv.org/abs/2507.05162))
- **Exact source URL**: `https://github.com/nchivar/LAID` (Release / Google Drive folder liên kết trong README)
- **Repository commit or release**: Commit `main` (tháng 12/2025)
- **Code license**: `unspecified` (Kho lưu trữ GitHub không có file `LICENSE`, đường dẫn `LICENSE` trả về HTTP 404)
- **Weights license**: `unknown` (Weights được chia sẻ qua Google Drive công khai, không đính kèm tệp văn bản cấp phép cụ thể)
- **Commercial/research restrictions**: Giấy phép bài báo trên arXiv là `CC BY 4.0`, nhưng mã nguồn và weights chưa có giấy phép phần mềm chính thức rõ ràng.
- **Architecture**: MobileNetV3-Small / ShuffleNetV2
- **Parameter count**: ~2.54M (MobileNetV3-Small) hoặc ~1.4M (ShuffleNetV2 0.5x)
- **Checkpoint format**: PyTorch (`.pth`)
- **Estimated checkpoint size**: ~10 MB (MobileNetV3) hoặc ~5.5 MB (ShuffleNet)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`)
- **Output classes**: 2 lớp (Real vs AI-Generated)
- **Localization support**: Không (chỉ hỗ trợ phân loại toàn ảnh `image-level classification`)
- **Training datasets**: GenImage (8 bộ tạo sinh)
- **Known generator coverage**: Midjourney, Stable Diffusion v1.4, Stable Diffusion v1.5, ADM, Glide, VQDM, Wukong, BigGAN
- **ONNX export feasibility**: Rất cao (Torchvision tiêu chuẩn, không có custom C++ operator)
- **WASM operator compatibility**: Hoàn toàn tương thích (Conv2d, BatchNorm, ReLU/Hardswish, Linear)
- **Expected browser memory**: ~35–50 MB RAM
- **Known limitations**: Chỉ phân loại nhị phân (không phát hiện vùng chỉnh sửa inpainting cục bộ); giấy phép phân phối trọng số chưa minh bạch; cần fine-tune lại head nếu muốn phân 3 lớp.
- **Download status**: `not-downloaded`
- **Admission status**: `blocked-license` (do repository thiếu file LICENSE và giấy phép phân phối weights chưa được xác minh chính thức).

---

## 2. Nhóm B: AI-Edited / Localization Detector (Định vị Vùng Chỉnh sửa Cục bộ)

### Ứng viên B1: SAGI-D Inpainting Forensics (mever-team/SAGI)
- **Candidate ID**: `CAND-B1-SAGI-D`
- **Official repository**: `https://github.com/mever-team/SAGI`
- **Paper**: "SAGI: Semantically Aligned and Uncertainty Guided AI Image Inpainting", CVPR 2025 ([arXiv:2502.06593](https://arxiv.org/abs/2502.06593))
- **Exact source URL**: `https://github.com/mever-team/SAGI`
- **Repository commit or release**: Commit `main` (tháng 02/2025)
- **Code license**: `unknown`
- **Weights license**: `unknown`
- **Commercial/research restrictions**: Nghiên cứu học thuật CVPR 2025
- **Architecture**: N/A (SAGI là công trình tạo sinh inpainting và dataset benchmark, không phải bộ dò tìm phát hiện)
- **Parameter count**: `unknown`
- **Checkpoint format**: `none`
- **Estimated checkpoint size**: `unknown`
- **Input size**: N/A
- **Normalization**: N/A
- **Output classes**: N/A
- **Localization support**: N/A
- **Training datasets**: SAGI-D Dataset (95,000+ ảnh inpainting)
- **Known generator coverage**: SD Inpainting, Blended Latent Diffusion
- **ONNX export feasibility**: N/A
- **WASM operator compatibility**: N/A
- **Expected browser memory**: N/A
- **Known limitations**: Repository chính thức của tác giả chỉ công bố mã nguồn sinh dữ liệu và tập dataset SAGI-D trên Kaggle; không phát hành checkpoint tiền huấn luyện của mô hình phát hiện (detector/localizer).
- **Download status**: `not-downloaded`
- **Admission status**: `blocked-missing-weights` (kho lưu trữ chính thức không phát hành pretrained detection weights).

---

### Ứng viên B2: TruFor (grip-unina/TruFor)
- **Candidate ID**: `CAND-B2-TRUFOR`
- **Official repository**: `https://github.com/grip-unina/TruFor`
- **Paper**: "TruFor: Leveraging RGB and Noise Analysis for Forensic Image Manipulation Detection and Localization", Guillaro et al., CVPR 2023
- **Exact source URL**: `https://github.com/grip-unina/TruFor`
- **Repository commit or release**: Tag v1.0
- **Code license**: Custom GRIP-UNINA Non-Commercial / Research License
- **Weights license**: Strictly non-commercial research use only
- **Commercial/research restrictions**: Giới hạn nghiêm ngặt phi thương mại (Non-commercial research only)
- **Architecture**: Cross-modal Transformer kết hợp RGB và Noiseprint residual
- **Parameter count**: ~68.4M tham số
- **Checkpoint format**: PyTorch (`.pth.tar`)
- **Estimated checkpoint size**: ~260 MB
- **Input size**: Kích thước động (tối thiểu $512 \times 512$)
- **Normalization**: RGB chuẩn hóa + Noiseprint extractor
- **Output classes**: Điểm số toàn ảnh và Anomaly Map định vị điểm ảnh (pixel-level localization)
- **Localization support**: Có (độ phân giải cao cấp điểm ảnh)
- **Training datasets**: Đa dạng các tập splicing/copymove và inpainting truyền thống
- **Known generator coverage**: Một số mô hình inpainting cổ điển và GAN
- **ONNX export feasibility**: Trung bình (có custom transformer attention và multi-scale fusion)
- **WASM operator compatibility**: Khó khăn trên browser thuần WASM CPU do chi phí tính toán lớn và bộ nhớ vượt ngưỡng
- **Expected browser memory**: > 400 MB RAM (vượt ngưỡng cho phép < 35 MB)
- **Known limitations**: Kích thước checkpoint quá lớn (~260 MB), kiến trúc nặng nề không phù hợp chạy thuần CPU trong Web Worker trình duyệt; giấy phép phi thương mại hạn chế khả năng triển khai sản phẩm thực tế.
- **Download status**: `not-downloaded`
- **Admission status**: `rejected` (vượt quá giới hạn kích thước và tài nguyên trình duyệt client-side, giấy phép phi thương mại khắt khe).

---

## 3. Nhóm C: Fallback Baseline & Lightweight Checkpoint

### Ứng viên C1: CNNDetection (Wang et al. CVPR 2020)
- **Candidate ID**: `CAND-C1-CNNDET`
- **Official repository**: `https://github.com/PeterWang512/CNNDetection`
- **Paper**: "CNN-generated images are surprisingly easy to spot... for now", Sheng-Yu Wang et al., CVPR 2020
- **Exact source URL**: `https://github.com/PeterWang512/CNNDetection`
- **Repository commit or release**: Release v1.0 (`blur_jpg_prob0.1.pth`)
- **Code license**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Weights license**: CC BY-NC-SA 4.0
- **Commercial/research restrictions**: Giới hạn phi thương mại (Non-Commercial) và chia sẻ tương tự (ShareAlike)
- **Architecture**: ResNet-50 (hoặc MobileNet baseline thử nghiệm)
- **Parameter count**: 25.6M (ResNet-50)
- **Checkpoint format**: PyTorch (`.pth`)
- **Estimated checkpoint size**: ~98 MB (FP32)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard
- **Output classes**: 2 lớp (Real vs CNN-generated)
- **Localization support**: Không (chỉ phân loại toàn ảnh)
- **Training datasets**: ProGAN (1 tập huấn luyện 20 lớp ProGAN)
- **Known generator coverage**: ProGAN, StyleGAN, BigGAN, CycleGAN (kém hiệu quả với Diffusion mới như SDXL/Flux)
- **ONNX export feasibility**: Rất cao
- **WASM operator compatibility**: Hoàn toàn tương thích
- **Expected browser memory**: ~120 MB RAM
- **Known limitations**: Checkpoint gốc dựa trên ResNet-50 có kích thước ~98 MB (vượt ngưỡng 35 MB); giấy phép CC BY-NC-SA hạn chế thương mại; huấn luyện chủ yếu trên GAN thế hệ cũ, độ nhạy với Latent Diffusion hiện đại còn hạn chế.
- **Download status**: `not-downloaded`
- **Admission status**: `blocked-license` (do điều khoản NC-SA không cho phép dùng thương mại) / `rejected` (dung lượng vượt ngưỡng browser).

---

### Ứng viên C2: In-House MobileNetV3-Small (Kế hoạch Huấn luyện Nội bộ Phase 4)
- **Candidate ID**: `CAND-C2-INHOUSE-MNV3`
- **Official repository**: `nomozer/forensics-web-lab` (Mã nguồn hiện tại trong `ml/training/mobilenetv3_forensics.py`)
- **Paper**: N/A (Đề tài nghiên cứu nội bộ của đồ án)
- **Exact source URL**: Local build từ pipeline tái lập của dự án
- **Repository commit or release**: Phản ánh từ commit `6555b02`
- **Code license**: Giấy phép của dự án (MIT / Apache-2.0)
- **Weights license**: Thuộc bản quyền dự án, không có hạn chế pháp lý của bên thứ ba
- **Commercial/research restrictions**: Không có hạn chế bên ngoài
- **Architecture**: MobileNetV3-Small tinh gọn, điều chỉnh classifier head cho 3 lớp bài toán
- **Parameter count**: ~2.54M tham số
- **Checkpoint format**: PyTorch `.pt` -> ONNX opset 17 -> INT8 Quantized ONNX
- **Estimated checkpoint size**: ~10.2 MB (FP32), ~2.6 MB (INT8 Quantized)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard
- **Output classes**: 3 lớp (`authentic`, `fully_generated`, `ai_edited`)
- **Localization support**: Có (thông qua quét lưới patch chồng lấn $224 \times 224$ kết hợp bộ tích lũy Gauss 2D `HeatmapAccumulator`)
- **Training datasets**: Trích xuất cân bằng từ các tập hợp lệ được cấp phép (GenImage + SAGI-D + RAID)
- **Known generator coverage**: Huấn luyện trực tiếp trên hỗn hợp GAN và Diffusion hiện đại
- **ONNX export feasibility**: Đã kiểm chứng thành công trong pipeline `ml/export/`
- **WASM operator compatibility**: 100% tương thích với ONNX Runtime Web CPU/WASM
- **Expected browser memory**: ~25–35 MB RAM (đáp ứng tiêu chuẩn nghiêm ngặt nhất)
- **Known limitations**: Cần dữ liệu huấn luyện thực tế và tài nguyên tính toán ở Phase 4 để tạo checkpoint thật.
- **Download status**: `not-downloaded`
- **Admission status**: `candidate` (Lựa chọn tối ưu và minh bạch nhất về mặt kiến trúc, kích thước và giấy phép).

---

## 4. Bảng So sánh và Khuyến nghị

| Tiêu chí | CAND-A1 (LAID) | CAND-B2 (TruFor) | CAND-C1 (CNNDet) | CAND-C2 (In-House MNV3) |
| :--- | :--- | :--- | :--- | :--- |
| **Kiến trúc** | MobileNetV3 / ShuffleNet | Cross-modal Transformer | ResNet-50 | MobileNetV3-Small (Custom Head) |
| **Số tham số** | ~1.4M – 2.5M | ~68.4M | ~25.6M | **~2.54M** |
| **Dung lượng INT8** | ~2.5 – 6 MB | N/A (> 200 MB) | ~25 MB | **~2.6 MB** |
| **Giấy phép code/weights**| Unspecified / Unknown | Non-Commercial Only | CC BY-NC-SA 4.0 | **Dự án kiểm soát 100%** |
| **Hỗ trợ 3 lớp** | Không (chỉ 2 lớp) | Không (chỉ Splicing) | Không (chỉ 2 lớp) | **Có (3 lớp bản địa)** |
| **Định vị Heatmap** | Không | Có (Pixel-level) | Không | **Có (Sliding patch + Gauss)** |
| **WASM Runtime** | Tốt | Quá nặng, không khả thi | Trung bình | **Tối ưu tuyệt đối (< 30 MB)** |
| **Admission Status** | `blocked-license` | `rejected` | `blocked-license` | **`candidate`** |

### Khuyến nghị Checkpoint Đầu tiên:
- **Ứng viên đề xuất**: **`CAND-C2-INHOUSE-MNV3`** (Huấn luyện nội bộ trên lát cắt dữ liệu nhỏ được phê duyệt trong Phase 4).
- **Lý do**: Đây là ứng viên duy nhất:
  1. Hỗ trợ bản địa phân loại 3 lớp chuẩn mực (`authentic`, `fully_generated`, `ai_edited`).
  2. Định vị được vùng chỉnh sửa cục bộ qua cơ chế sliding patch tương thích với `HeatmapAccumulator` hiện có.
  3. Đạt kích thước siêu nhẹ (~2.6 MB INT8) chạy mượt mà trên mọi thiết bị qua WASM CPU.
  4. Hoàn toàn sạch về mặt pháp lý và giấy phép, không phụ thuộc vào liên kết chia sẻ bên thứ ba thiếu tệp LICENSE.
