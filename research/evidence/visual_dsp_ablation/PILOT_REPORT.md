# Báo cáo Pilot: Thí nghiệm Visual / DSP Ablation

> **Branch**: `research/visual-dsp-ablation`<br>
> **Base main commit**: `9516341e7d7deaf61be4e501d6ce2eba0741c313`<br>
> **Trạng thái**: `PREFLIGHT_AND_PILOT_VERIFIED`<br>
> **Đối tượng dữ liệu**: Đúng 341 development sources (682 samples: 341 authentic + 341 ai_edited)<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access, 0 evaluation, fail-closed guard active)<br>

---

## 1. Mục tiêu và Thiết kế Thí nghiệm

Thí nghiệm so sánh đối đầu prospective 3 cấu hình (recipes) trên đúng 341 nguồn phát triển:
1. **`visual_control`**: MobileNetV3-Small frozen backbone pooled features (576-dim).
2. **`dsp_only`**: Vector 16 đặc trưng DSP chuẩn tắc (4 FFT radial + 3 DCT energy bands + 5 Noise residuals + 3 JPEG BAG + 1 Laplacian variance) ánh xạ trực tiếp từ các module trong `packages/forensics`.
3. **`visual_dsp_fusion`**: Vector đặc trưng kết hợp sớm (early fusion) 592-dim ($576 + 16$).

### Nguyên tắc Bất biến
- **Cùng cấu trúc phân chia fold**: 5 outer folds $\times$ 4 inner folds, nhóm theo `source_id` (Zero-leakage giữa các ảnh của cùng một source).
- **Cùng họ classifier**: L2-regularized Logistic Regression (solver L-BFGS, max_iter=1000).
- **Quy trình chọn siêu tham số đồng nhất**: Lưới $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$ tối ưu hóa trên Macro-F1 trung bình của 4 inner folds; sau đó refit toàn bộ outer-train fold bằng $C^*$ tốt nhất và suy luận một lần trên outer-test fold.
- **Dự đoán ảnh đơn độc lập**: Mỗi ảnh được trích xuất và phân loại riêng biệt, không dùng ảnh cặp, source ID, đường dẫn hay nhãn thật.
- **Tiền xử lý độc lập hoàn toàn**: `StandardScaler` được fit độc lập trên từng training fold và transform trên test fold; tuyệt đối không fit trên toàn bộ dataset trước.

---

## 2. Kết quả Preflight

Preflight thực thi qua lệnh:
```bash
python -m ml.training.run_visual_dsp_ablation --mode preflight
```
- **Dữ liệu**: Nạp thành công 341 pairs (682 images) từ `data/research/tgif/manifests/manifest_pilot_a_option_p.csv`.
- **Toàn vẹn SHA-256**: 100% (682/682) file ảnh trên đĩa khớp hoàn toàn mã băm SHA-256 đã đăng ký trong manifest.
- **Pretrained weights**: `models/research/pretrained/mobilenet_v3_small-047dcff4.pth` (SHA-256 `047dcff4...`) khớp xác thực.
- **Cơ chế fail-closed**: Kiểm thử chặn đứng mọi nỗ lực nạp partition `locked_test` (15/15 unit tests PASS).

---

## 3. Kết quả Technical Pilot

Pilot thực thi kỹ thuật trên outer fold 0 của cấu hình `visual_control`:
```bash
python -m ml.training.run_visual_dsp_ablation --mode pilot --output-dir data/research/local-artifacts/visual_dsp_ablation
```

### Thông số Runtime & Caching
- **Visual feature extraction (682 samples, CUDA)**: ~21.0 giây (lưu tại `shared/visual_features.pt`).
- **DSP feature extraction (682 samples, CPU)**: ~91.5 giây (lưu tại `shared/dsp_features.pt`).
- **Inner CV (7 grid points $\times$ 4 inner folds = 28 fits)**: 1.916 giây.
- **Best $C$ được chọn**: $0.001$ (Inner-CV Mean Macro-F1: $0.5734$).
- **Outer Test Metrics (outer fold 0, 138 samples)**:
  - Macro-F1: `0.5775`
  - Balanced Accuracy: `0.5797`
  - AUROC: `0.6127`
  - Brier Score: `0.2411`
  - ECE: `0.0404`
  - FPR: `0.3478` (24/69)
  - FNR: `0.4928` (34/69)
- **Kiểm tra Resume**: Chạy lại pilot mode nhận diện fit receipt đã tồn tại và hoàn thành kiểm tra trong 2.3 giây mà không chạy lại tính toán.

> [!IMPORTANT]
> **Quy tắc Trung thực Khoa học (Scientific Honesty)**: Kết quả của lượt chạy pilot là kiểm tra kỹ thuật về đường ống dữ liệu, thời gian thực thi và tính toàn vẹn của artifact schema. Tuyệt đối **không** dùng điểm số pilot để thay đổi protocol, điều chỉnh không gian siêu tham số hoặc lựa chọn mô hình.

---

## 4. Hướng dẫn Thực thi Full Matrix & Phân tích OOF

### Chạy Local (Toàn bộ 15 outer fits = 3 recipes $\times$ 5 folds)
```bash
python -m ml.training.run_visual_dsp_ablation --mode full --output-dir data/research/local-artifacts/visual_dsp_ablation
```

### Chạy qua Google Colab
1. Sử dụng notebook: `notebooks/visual_dsp_ablation_colab.ipynb`.
2. Đặt code snapshot `phase_visual_dsp_ablation_code.tar.gz` (SHA-256 `fda4feb4...`) và dataset `phase_4c1_binary_n250_reusable.tar` vào thư mục Drive:
   `MyDrive/Colab Notebooks/forensics-web-lab/visual_dsp_ablation/`
3. Thực thi notebook. Khi kết quả hoàn tất, archive kết quả tự động lưu về Drive.

### Chạy Phân tích OOF (Out-Of-Fold Analysis)
Sau khi hoàn tất 15 outer fits:
```bash
python scripts/research/analyze_visual_dsp_ablation.py \
  --output-dir data/research/local-artifacts/visual_dsp_ablation \
  --report-dir research/evidence/visual_dsp_ablation
```
Phân tích sẽ tự động tính toán:
- Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE, FPR, FNR cho cả 3 recipes trên toàn bộ 341 sources (682 samples).
- Paired deltas giữa Visual vs DSP, Visual+DSP vs Visual, Visual+DSP vs DSP.
- Tạo biểu đồ SVG và PNG độc lập không phụ thuộc thư viện đồ họa ngoài.
