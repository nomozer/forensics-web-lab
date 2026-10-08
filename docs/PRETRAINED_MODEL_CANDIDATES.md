# Pretrained Model Candidates: Khảo sát Ứng viên Checkpoint và Kiến trúc Mô hình

> **Ngày cập nhật**: 2026-09-21 (Phase 3.6 Evidence Hardening & Claim Correction)  
> **Nguyên tắc cốt lõi**:  
> - Chỉ nghiên cứu metadata, tài liệu khoa học và mã nguồn chính thức.  
> - **KHÔNG tải weights hoặc dataset** trong giai đoạn này (`Download status: not-downloaded`).  
> - Không suy đoán giấy phép hay thông số kỹ thuật chưa được công bố (`unverified`).  
> - Không chuyển đổi metric từ paper thành metric của sản phẩm (nếu trích dẫn phải ghi rõ `reported-by-source`).  
> - Trọng số chưa huấn luyện thì **KHÔNG gọi là checkpoint**; chỉ ghi nhận là `architecture candidate`.  
> - Trạng thái xét duyệt (`Admission status`) tuân thủ nghiêm ngặt: `candidate`, `architecture-only`, `rejected`, `blocked-license`, `blocked-missing-weights`, `blocked-incompatible-task`.

---

## 1. Nhóm A: Fully Generated Detector (Phát hiện Ảnh Tạo Sinh Toàn Phần)

### Ứng viên A1: LAID MobileNetV3-Small / ShuffleNetV2
- **Candidate ID**: `CAND-A1-LAID`
- **Official repository**: `https://github.com/nchivar/LAID` `[@laidRepository]`
- **Paper**: "LAID: Lightweight AI-Generated Image Detection in Spatial and Spectral Domains", Nicholas Chivaran & Jianbing Ni, arXiv:2507.05162; repository tác giả ghi accepted at PST 2025 `[@chivaran2025laid; @laidRepository]`.
- **Exact source URL**: `https://github.com/nchivar/LAID` (Release / Google Drive folder liên kết trong README)
- **Repository commit or release**: Revision `9bd6f07db6c220e53d4f2e07ea00bbda761874e3` (đọc 2026-10-09).
- **Code license**: `unspecified` (Kho lưu trữ GitHub không có file `LICENSE`, đường dẫn `LICENSE` trả về HTTP 404)
- **Weights license**: `unverified` (Weights được chia sẻ qua Google Drive công khai, không đính kèm tệp văn bản cấp phép cụ thể)
- **Commercial/research restrictions**: Giấy phép bài báo trên arXiv là `CC BY 4.0`, nhưng mã nguồn và weights chưa có giấy phép phần mềm chính thức rõ ràng.
- **Architecture**: MobileNetV3-Small / ShuffleNetV2
- **Parameter count**: 2,542,856 cho torchvision MobileNetV3-Small `IMAGENET1K_V1` `[@torchvisionMobilenetV3Small]`; con số ~1.4M cho ShuffleNetV2 0.5x chưa được primary LAID page xác minh trong lần rà soát này.
- **Checkpoint format**: PyTorch (`.pth`)
- **Estimated checkpoint size**: ~10 MB (MobileNetV3) hoặc ~5.5 MB (ShuffleNet) (`estimated`)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`)
- **Output classes**: 2 lớp (Real vs AI-Generated)
- **Localization support**: Không (chỉ hỗ trợ phân loại toàn ảnh `image-level classification`)
- **Training datasets**: GenImage (8 bộ tạo sinh)
- **Known generator coverage**: Midjourney, Stable Diffusion v1.4, Stable Diffusion v1.5, ADM, Glide, VQDM, Wukong, BigGAN
- **ONNX export feasibility**: Rất cao (Torchvision tiêu chuẩn, không có custom C++ operator)
- **WASM operator compatibility**: Hoàn toàn tương thích về mặt lý thuyết (Conv2d, BatchNorm, ReLU/Hardswish, Linear)
- **Expected browser memory**: ~35–50 MB RAM (`estimated`)
- **Known limitations**: Chỉ phân loại nhị phân (không phát hiện vùng chỉnh sửa inpainting cục bộ); giấy phép phân phối trọng số chưa minh bạch; cần fine-tune lại head nếu muốn phân 3 lớp.
- **Download status**: `not-downloaded`
- **Admission status**: `blocked-license` (do repository thiếu file LICENSE và giấy phép phân phối weights chưa được xác minh chính thức).

---

## 2. Nhóm B: AI-Edited / Localization Detector (Định vị Vùng Chỉnh sửa Cục bộ)

### Ứng viên B1: SAGI-D Inpainting Forensics (mever-team/SAGI)
- **Candidate ID**: `CAND-B1-SAGI-D`
- **Official repository**: `https://github.com/mever-team/SAGI` `[@sagiRepository]`
- **Paper**: "SAGI: Semantically Aligned and Uncertainty Guided AI Image Inpainting", ICCV 2025 `[@giakoumoglou2025sagi]`.
- **Exact source URL**: `https://github.com/mever-team/SAGI`
- **Repository commit or release**: Revision `4ddc5cf1204e56690df97ed06b1e879df082402e` (đọc 2026-10-09).
- **Code license**: `Apache-2.0` tại repository revision đã ghi; quyền phân phối dataset/source images vẫn `unverified`.
- **Weights license**: `unverified`
- **Commercial/research restrictions**: Paper ICCV 2025; license code không tự động mở rộng sang dataset/source images.
- **Architecture**: N/A (SAGI là công trình tạo sinh inpainting và dataset benchmark, không phát hành checkpoint phát hiện)
- **Parameter count**: `unverified`
- **Checkpoint format**: `none`
- **Estimated checkpoint size**: `unverified`
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
- **Official repository**: `https://github.com/grip-unina/TruFor` `[@truforRepository]`
- **Paper**: "TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization", Guillaro et al., CVPR 2023 `[@guillaro2023trufor]`.
- **Exact source URL**: `https://github.com/grip-unina/TruFor`
- **Repository commit or release**: Revision `ae54475df6f41a491d7615100feb19263dec13f7` (không tìm thấy tag `v1.0` qua remote refs tại ngày đọc).
- **Code license**: Custom GRIP-UNINA Non-Commercial / Research License
- **Weights license**: Strictly non-commercial research use only
- **Commercial/research restrictions**: Giới hạn nghiêm ngặt phi thương mại (Non-commercial research only)
- **Architecture**: Cross-modal Transformer kết hợp RGB và Noiseprint residual
- **Parameter count**: ~68.4M tham số (`unverified` trong lần rà soát này; paper/repository pages đã đọc không cung cấp con số này).
- **Checkpoint format**: PyTorch (`.pth.tar`)
- **Estimated checkpoint size**: ~260 MB (`estimated`)
- **Input size**: Kích thước động (tối thiểu $512 \times 512$)
- **Normalization**: RGB chuẩn hóa + Noiseprint extractor
- **Output classes**: Điểm số toàn ảnh và Anomaly Map định vị điểm ảnh (pixel-level localization)
- **Localization support**: Có (độ phân giải cao cấp điểm ảnh)
- **Training datasets**: Đa dạng các tập splicing/copymove và inpainting truyền thống
- **Known generator coverage**: Một số mô hình inpainting cổ điển và GAN
- **ONNX export feasibility**: Trung bình (có custom transformer attention và multi-scale fusion)
- **WASM operator compatibility**: Khó khăn trên browser thuần WASM CPU do chi phí tính toán lớn và bộ nhớ vượt ngưỡng
- **Expected browser memory**: > 400 MB RAM (`estimated`, chưa có browser measurement artifact).
- **Known limitations**: Kích thước checkpoint quá lớn (~260 MB), kiến trúc nặng nề không phù hợp chạy thuần CPU trong Web Worker trình duyệt; giấy phép phi thương mại hạn chế khả năng triển khai sản phẩm thực tế.
- **Download status**: `not-downloaded`
- **Admission status**: `rejected` (vượt quá giới hạn kích thước và tài nguyên trình duyệt client-side, giấy phép phi thương mại khắt khe).

---

## 3. Nhóm C: Fallback Baseline & Thiết kế Kiến trúc Nội bộ

### Ứng viên C1: CNNDetection (Wang et al. CVPR 2020)
- **Candidate ID**: `CAND-C1-CNNDET`
- **Official repository**: `https://github.com/PeterWang512/CNNDetection` `[@cnnDetectionRepository]`
- **Paper**: "CNN-generated images are surprisingly easy to spot... for now", Sheng-Yu Wang et al., CVPR 2020 `[@wang2020cnndetection]`.
- **Exact source URL**: `https://github.com/PeterWang512/CNNDetection`
- **Repository commit or release**: Revision `ea0b5622365e3a9cd31d1b54b6b5971131a839ab` (không tìm thấy tag `v1.0` qua remote refs tại ngày đọc); repository links checkpoint `blur_jpg_prob0.1.pth`.
- **Code license**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Weights license**: CC BY-NC-SA 4.0
- **Commercial/research restrictions**: Giới hạn phi thương mại (Non-Commercial) và chia sẻ tương tự (ShareAlike)
- **Architecture**: ResNet-50 (hoặc MobileNet baseline thử nghiệm)
- **Parameter count**: 25.6M (ResNet-50; architecture-level value, chưa đối chiếu một artifact checkpoint cục bộ).
- **Checkpoint format**: PyTorch (`.pth`)
- **Estimated checkpoint size**: ~98 MB (FP32) (`estimated`)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard
- **Output classes**: 2 lớp (Real vs CNN-generated)
- **Localization support**: Không (chỉ phân loại toàn ảnh)
- **Training datasets**: ProGAN (1 tập huấn luyện 20 lớp ProGAN)
- **Known generator coverage**: ProGAN, StyleGAN, BigGAN, CycleGAN (kém hiệu quả với Diffusion mới như SDXL/Flux)
- **ONNX export feasibility**: Rất cao
- **WASM operator compatibility**: Hoàn toàn tương thích
- **Expected browser memory**: ~120 MB RAM (`estimated`)
- **Known limitations**: Checkpoint gốc dựa trên ResNet-50 có kích thước ~98 MB (vượt ngưỡng 35 MB); giấy phép CC BY-NC-SA hạn chế thương mại; huấn luyện chủ yếu trên GAN thế hệ cũ, độ nhạy với Latent Diffusion hiện đại còn hạn chế.
- **Download status**: `not-downloaded`
- **Admission status**: `blocked-license` (do điều khoản NC-SA không cho phép dùng thương mại) / `rejected` (dung lượng vượt ngưỡng browser).

---

### Ứng viên C2: Thiết kế Kiến trúc Nội bộ MobileNetV3-Small (Đề xuất Thực nghiệm Phase 4)
- **Candidate ID**: `ARCH-C2-INHOUSE-MNV3` *(Kiến trúc đề xuất, KHÔNG PHẢI checkpoint đã huấn luyện)*
- **Official repository**: `nomozer/forensics-web-lab` (Mã kiến trúc trong `ml/training/mobilenetv3_forensics.py`)
- **Paper**: Đề tài nghiên cứu nội bộ của đồ án
- **Exact source URL**: Local build từ pipeline mã nguồn của repository
- **Repository commit or release**: Phản ánh từ commit `cd59136`
- **Code license**: MIT / Apache-2.0 (mã nguồn dự án)
- **Weights license**: `not-applicable` *(Chưa có trọng số. Quyền phân phối trọng số sau này phụ thuộc vào pretrained backbone khởi tạo, mã nguồn và giấy phép của các dataset được dùng để huấn luyện)*
- **Commercial/research restrictions**: Phụ thuộc vào dữ liệu huấn luyện thực tế trong tương lai
- **Architecture**: MobileNetV3-Small tinh gọn, sửa đổi classifier head cho 3 lớp bài toán
- **Parameter count**: 2,542,856 cho backbone torchvision chuẩn `[@torchvisionMobilenetV3Small]`; tổng custom-head configuration phải lấy từ artifact/run tương ứng (`architecture-only`, không suy thành checkpoint forensic).
- **Checkpoint format**: `none` *(Chưa huấn luyện, chưa xuất file checkpoint thật)*
- **Estimated checkpoint size**: ~10.2 MB (FP32), ~2.6 MB (INT8 Quantized) (`estimated`, chưa đo trên artifact thật)
- **Input size**: $224 \times 224 \times 3$
- **Normalization**: ImageNet standard
- **Output classes**: 3 lớp (`authentic`, `fully_generated`, `ai_edited`) (`architecture-only`)
- **Localization support**: `architecture-only` (mã quét lưới patch chồng lấn $224 \times 224$ và bộ tích lũy Gauss 2D `HeatmapAccumulator` đã được lập trình, nhưng khả năng định vị thực tế là `unverified` do chưa có trọng số mô hình)
- **Training datasets**: Chưa tải dataset nào (`none`)
- **Known generator coverage**: `unverified` (chưa có mô hình để đánh giá độ phủ)
- **ONNX export feasibility**: `pipeline-only` (đã kiểm chứng mã xuất ONNX trên mô hình khởi tạo ngẫu nhiên/chưa huấn luyện trong unit test, chưa kiểm thử trên checkpoint thật)
- **WASM operator compatibility**: `pipeline-only` (ONNX Runtime Web documentation liệt kê WebAssembly browser support `[@onnxRuntimeWebDocs]`, nhưng độ tương thích của model thật trên Chrome/Edge/Firefox/Safari vẫn `unverified` trong dự án).
- **Expected browser memory**: ~25–35 MB RAM (`estimated`)
- **Known limitations**: Chưa có dữ liệu huấn luyện; chưa có trọng số; chưa từng đo đạc chỉ số khoa học thực tế; bộ dữ liệu nhỏ (< 50 MB) chỉ đủ cho pipeline smoke test, không đủ để tạo ra mô hình phát hiện đáng tin cậy hay đưa ra tuyên bố khoa học.
- **Download status**: `not-downloaded`
- **Admission status**: `architecture-only` (chỉ là ứng viên kiến trúc mã nguồn, chưa đủ điều kiện chuyển thành checkpoint sẵn sàng).

---

## 4. Bảng So sánh và Tình trạng Minh chứng

| Tiêu chí | CAND-A1 (LAID) | CAND-B2 (TruFor) | CAND-C1 (CNNDet) | ARCH-C2-INHOUSE-MNV3 |
| :--- | :--- | :--- | :--- | :--- |
| **Loại ứng viên** | Checkpoint bên ngoài | Checkpoint bên ngoài | Checkpoint bên ngoài | **Architecture candidate** |
| **Kiến trúc** | MobileNetV3 / ShuffleNet | Cross-modal Transformer | ResNet-50 | MobileNetV3-Small (Custom Head) |
| **Số tham số** | ~1.4M – 2.5M | ~68.4M | ~25.6M | **~2.54M (`architecture-only`)** |
| **Dung lượng INT8** | ~2.5 – 6 MB (`estimated`) | N/A (> 200 MB) | ~25 MB (`estimated`) | **~2.6 MB (`estimated`, chưa đo thật)** |
| **Giấy phép code** | Unspecified | Non-Commercial Only | CC BY-NC-SA 4.0 | **MIT / Apache-2.0 (mã nguồn)** |
| **Giấy phép weights**| Unverified | Non-Commercial Only | CC BY-NC-SA 4.0 | **`not-applicable` (chưa có weights)** |
| **Hỗ trợ 3 lớp** | Không (chỉ 2 lớp) | Không (chỉ Splicing) | Không (chỉ 2 lớp) | **`architecture-only`** |
| **Định vị Heatmap** | Không | Có (Pixel-level) | Không | **`architecture-only` (chưa đo thật)** |
| **WASM Runtime** | Lý thuyết tương thích | Quá nặng, không khả thi | Trung bình | **`pipeline-only`** |
| **Khả năng phát hiện**| Reported by source | Reported by source | Reported by source | **`unverified` (chưa huấn luyện)** |
| **Admission Status** | `blocked-license` | `rejected` | `blocked-license` | **`architecture-only`** |

### Định hướng Đề xuất cho Phase 4:
- **Kiến trúc đề xuất để huấn luyện**: **`ARCH-C2-INHOUSE-MNV3`** (Sử dụng kiến trúc MobileNetV3-Small đã lập trình trong repository để thực hiện huấn luyện thực nghiệm trong Phase 4 khi có dataset được duyệt).
- **Lưu ý trung thực**:
  1. Hiện tại **chưa có checkpoint nào được duyệt hoặc sẵn sàng nạp** (`none`).
  2. Việc huấn luyện mô hình thực sự ở Phase 4 cần một tập dữ liệu đa dạng được phê duyệt rõ ràng về giấy phép và nguồn gốc.
  3. Mọi chỉ số đánh giá khoa học (F1, ECE, AUROC, mIoU) hiện đều ở trạng thái `not evaluated` và sẽ chỉ được đo đạc trên artifact thật sau khi hoàn thành huấn luyện.
