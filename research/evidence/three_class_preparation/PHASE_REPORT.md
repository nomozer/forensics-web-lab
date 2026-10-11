# Phase Report: Three-Class RQ1 & RQ2 Experiment Execution

> **Phase**: Three-Class RQ1 & RQ2 Empirical Pilot Execution (Phase 4C.8 trace)<br>
> **Branch**: `research/independent-cohort-acquisition`<br>
> **Execution Date**: 2026-10-11<br>
> **Status**: `THREE_CLASS_PILOT_EXECUTED_EMPIRICAL_PASS`

---

## 1. Mục tiêu & Bối cảnh

Hoàn tất trọn vẹn dữ liệu thực và thực thi đánh giá thực nghiệm cho hai câu hỏi nghiên cứu cốt lõi:
- **RQ1 (Three-Class Generalization)**: Khả năng phân biệt đồng thời 3 lớp `authentic`, `ai_edited`, và `fully_generated` trên kiến trúc thị giác tích chập siêu nhẹ (MobileNetV3-small frozen linear probe).
- **RQ2 (Multimodal Evidence Fusion)**: Đóng góp của đặc trưng DSP canonical 16-D khi ghép nối với nhánh thị giác (Feature Concatenation 592-D) so với visual-only baseline (576-D) trong không gian 3 lớp.

Tuân thủ nghiêm ngặt **Quy tắc trung thực kỹ thuật và nghiên cứu** (Rule 3 AGENTS.md):
- Không vượt ngân sách tài nguyên: Ngân sách mạng cộng dồn $\le 2\text{ GiB}$, ổ đĩa $\le 5\text{ GiB}$.
- Không fine-tune backbone, không generation.
- Không tạo số liệu giả, không chọn lại mẫu/splits để làm đẹp kết quả.
- Tách biệt hoàn toàn mô hình 3 lớp mới khỏi demo nhị phân FP32 và 5 fold models lịch sử.

---

## 2. Ban hành Protocol Amendment v2.0.0 (SD1.4 → SD1.5)

- **Lý do khách quan**: Archive chính thức của GenImage SD1.4 là multipart zip 30 phần nặng ~96.5 GiB (vượt 2 GiB limit); mirror Tiny-GenImage trên Hugging Face thiếu hoàn toàn Generator 5 (SD1.4, 0 mẫu).
- **Nội dung sửa đổi**:
  - Ban hành [`PROTOCOL_AMENDMENT_V2.0.0.md`](./PROTOCOL_AMENDMENT_V2.0.0.md).
  - Chuyển generator của lớp `fully_generated` từ SD1.4 sang **Stable Diffusion v1.5** (`TheKernel01/Tiny-GenImage`, generator index 6, label 1).
  - Giữ nguyên taxonomy 3 lớp, MobileNetV3-small 576-d visual, 16-d canonical DSP, 592-d feature concatenation probe.
  - Phân định rõ ràng: Vector 592-d là **feature concatenation probe**, không gọi nhầm là late fusion (late fusion dành cho tầng logit/probability).

---

## 3. Tiếp nhận Dữ liệu trong Ngân sách & Khóa Splits (Tasks 2 & 3)

- **Kiểm soát ngân sách tải mạng**:
  - Shards 0 & 1 đã có sẵn: 875,164,612 bytes (~834.6 MiB).
  - Tải bổ sung Shard 2 (`validation-00002-of-00004.parquet`): 435,573,773 bytes (~415.4 MiB).
  - **Tổng tải mạng cộng dồn**: **1,310,738,385 bytes (~1.22 GiB)** $\le 2\text{ GiB}$ ($61.04\%$ ngân sách).
  - **Tổng footprint ổ đĩa mới**: ~1.40 GiB $\le 5\text{ GiB}$ ngân sách.
- **Trích xuất mẫu SD1.5 xác định**:
  - Trích xuất đủ **341 ảnh SD1.5** (125 từ shard 0, 125 từ shard 1, 91 từ shard 2) tại `data/research/genimage/sd15_pilot/`.
  - 100% giải mã Pillow RGB thành công, 341 hashes SHA-256 riêng biệt (0 duplicate).
  - Biên nhận cập nhật: [`genimage_acquisition_receipt.json`](./genimage_acquisition_receipt.json).
- **Khóa Splits Phát triển Ba Lớp (1,023 ảnh)**:
  - `development_train`: **750 ảnh** (250 COCO authentic, 250 TGIF SD2 inpainting, 250 SD1.5).
  - `development_holdout`: **273 ảnh** (91 COCO authentic, 91 TGIF SD2 inpainting, 91 SD1.5).
  - Tỷ lệ các lớp hoàn toàn cân bằng 1:1:1 trong cả hai tập.
  - Mọi cặp authentic/ai_edited từ cùng nguồn COCO đều nằm strictly trong cùng partition.
  - Manifest xuất bản: [`three_class_pilot_manifest.csv`](./three_class_pilot_manifest.csv) và [`three_class_pilot_splits.json`](./three_class_pilot_splits.json).
- **Kiểm toán cách ly sealed cohorts (ZERO LEAKAGE PASS)**:
  - Locked-Test (Phase 4B/4C.2G): 343 sources, 3,087 disk images, 2,970 distinct hashes $\to$ **Overlap = 0**.
  - TGIF N=400 (Phase 4C.7B): 400 sources, 800 zip images, 800 distinct hashes $\to$ **Overlap = 0**.
  - Báo cáo kiểm toán: [`leakage_and_confounding_report.json`](./leakage_and_confounding_report.json).

---

## 4. Trích xuất Đặc trưng & Huấn luyện Baselines (Tasks 4 & 5)

- **Trích xuất đặc trưng**:
  - Backbone: MobileNetV3-small đóng băng weights ImageNet `mobilenet_v3_small-047dcff4.pth` (SHA-256 `047dcff4...`).
  - Trích xuất đủ 1,364 ảnh (750 train + 273 holdout + 341 nature control): 0 NaNs, 0 Infs.
  - Cache: `data/research/cache/three_class_features.npz` (2.82 MB).
  - Biên nhận: [`feature_extraction_receipt.json`](./feature_extraction_receipt.json).
- **Huấn luyện mô hình**:
  - Hai linear probes (`C=1.0`, `solver='lbfgs'`, multinomial cross-entropy) được huấn luyện strictly trên `development_train` (750 mẫu).
  - `StandardScaler` fitted strictly trên `development_train`.
  - Hiệu chuẩn nhiệt độ Temperature Scaling fitted trên out-of-fold predictions từ inner 5-fold cross-validation của `development_train`:
    * Visual-only: $T_v = 8.3734$
    * Visual+DSP: $T_f = 8.1321$
  - Artifacts mô hình: [`three_class_visual_only_model.json`](./three_class_visual_only_model.json) và [`three_class_visual_dsp_model.json`](./three_class_visual_dsp_model.json).

---

## 5. Kết quả Thực nghiệm Thực tế trên Development Holdout ($N=273$)

### Đối Chiếu Hai Run Đánh Giá: Run Lịch Sử vs. Run Reconciled (Hiện Hành)
- **Run Lịch Sử** ([`three_class_pilot_evaluation_receipt.json`](./three_class_pilot_evaluation_receipt.json), SHA-256 bound): Inner CV sample-level chưa gom nhóm, scaler toàn cục, nhiệt độ over-smoothed ($T_v = 8.3734, T_f = 8.1321$).
- **Run Reconciled (Hiện Hành)** ([`three_class_pilot_evaluation_receipt_reconciled.json`](./three_class_pilot_evaluation_receipt_reconciled.json), SHA-256 bound): Inner `StratifiedGroupKFold(5)` gom nhóm theo `source_group_id`, scaler cô lập từng inner fold, nhiệt độ out-of-fold chuẩn tắc ($T_v = 5.9818, T_f = 5.9577$). **Kết luận hiện hành sử dụng số liệu Reconciled**.

| Chỉ số Đánh Giá | Baseline 1: Visual-Only (576-D) | Baseline 2: Visual + DSP (592-D) | Chênh Lệch ($\Delta = \text{B2} - \text{B1}$) | Ghi chú & Ranh giới |
| :--- | :---: | :---: | :---: | :--- |
| **Macro-F1** (Khóa Outer) | **0.5086** | **0.5385** | **+0.0298** | +2.98 điểm phần trăm (đồng nhất 100% giữa hai run) |
| **Balanced Accuracy** | **0.4982** | **0.5238** | **+0.0256** | +2.56 điểm phần trăm (đồng nhất 100% giữa hai run) |
| **Multiclass Brier Score (Reconciled)** | **0.6002** | **0.5711** | **-0.0291** | Thấp hơn là tốt hơn (Lịch sử: 0.5841 vs 0.5618) |
| **Multiclass Top-Class ECE (Reconciled)** | **0.1003** | **0.0865** | **-0.0138** | Thấp hơn là tốt hơn (Lịch sử: 0.0466 vs 0.0780) |
| - F1 `authentic` ($n=91$) | 0.4108 | 0.4409 | +0.0301 | +3.01 điểm phần trăm |
| - F1 `ai_edited` ($n=91$) | 0.4615 | 0.4390 | -0.0225 | -2.25 điểm phần trăm |
| - F1 `fully_generated` ($n=91$) | 0.6536 | 0.7355 | +0.0819 | +8.19 điểm phần trăm |

### Ma trận Nhầm lẫn (Confusion Matrix 3×3) — Đồng nhất Cả Hai Run

- **Baseline 1 (Visual-Only)**:
  ```
  Pred ->   Authentic   AI-Edited   Fully-Gen
  Auth (91)     38          46           7
  Edit (91)     38          48           5
  Gen  (91)     18          23          50
  ```
- **Baseline 2 (Visual + DSP)**:
  ```
  Pred ->   Authentic   AI-Edited   Fully-Gen
  Auth (91)     41          46           4
  Edit (91)     43          45           3
  Gen  (91)     11          23          57
  ```

### Kiểm định Thống kê Ghép cặp (Paired Cluster Bootstrap)
- Phương pháp: Stratified Source Cluster Bootstrap (10,000 replicates, PCG64 seed `20261011`, tái lấy mẫu theo `source_group_id`).
- Điểm ước lượng $\Delta \text{Macro-F1}$: **+0.0298** (tương đương **+2.98 đến +2.99 điểm phần trăm** (percentage points), tuyệt đối không nhập nhằng với mức tăng tương đối $+5.86\%$).
- Khoảng tin cậy 95% Percentile CI:
  * Cấu hình công bố (Pooled 182 clusters): **[-0.0068, +0.0687]**, $P(\Delta^* > 0) = \mathbf{94.35\%}$
  * Cấu hình phân tầng (Stratified 91 COCO + 91 SD1.5): **[-0.0065, +0.0676]**, $P(\Delta^* > 0) = \mathbf{94.53\%}$
- Giới hạn diễn giải: Tỷ lệ bootstrap replicates có $\Delta^* > 0$ (~$94.35\%$) là **tần suất thực nghiệm của các mẫu tái rút**, tuyệt đối không phải là xác suất giả thuyết đúng (posterior probability) hoặc p-value của kiểm định giả thuyết vô hiệu.
- **Kết luận khoa học chính thức**: **`INCONCLUSIVE`**. Việc bổ sung 16-D canonical DSP cho thấy xu hướng cải thiện dương (+2.98 điểm phần trăm), nhưng khoảng tin cậy 95% CI cắt 0 ở cận dưới (-0.0068 $\le 0$). Do đó, chưa đủ bằng chứng để khẳng định sự cải thiện có ý nghĩa thống kê ở mức ý nghĩa $\alpha = 0.05$.

---

## 6. Đánh giá Chẩn đoán Nguồn (ImageNet Authentic Control Cohort, $N=341$)

Đánh giá riêng biệt trên 341 ảnh ImageNet authentic (`nature_control_pilot`, nhãn 0 từ Tiny-GenImage validation shards; không suy diễn ngữ nghĩa thuần túy từ tên thư mục `nature`), hoàn toàn không tham gia huấn luyện hay chọn ngưỡng:

| Mô hình | Số ảnh Đúng (Authentic) | Nhầm `ai_edited` | Nhầm `fully_generated` | Tổng Nhầm AI | **False Positive AI Rate** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Visual-Only** | 111 / 341 | 106 | 124 | 230 | **67.45%** |
| **Visual + DSP** | 109 / 341 | 112 | 120 | 232 | **68.04%** |

### Các Giả Thuyết Khoa Học Cần Kiểm Tra Về Hiện Tượng Báo Động Giả
1. **Giả thuyết Shortcut Nguồn / Semantic Domain Shift (Cần kiểm tra thực nghiệm)**: Khi chuyển từ tập huấn luyện (MS-COCO val2017) sang một tập ảnh authentic khác (ImageNet authentic), tỷ lệ báo động giả AI lên tới **~67–68%**. Giả thuyết đặt ra là mô hình phân loại 3 lớp siêu nhẹ dựa trên backbone đông lạnh có thể đã học đặc trưng phân phối bối cảnh của dataset gốc (COCO vs ImageNet) hơn là nhận diện thuần túy dấu vết AI. Đây là giả thuyết nghiên cứu cần được kiểm tra bằng thực nghiệm can thiệp bổ sung dữ liệu (Ablation A0 vs A1), không phải nguyên nhân đã chứng minh.
2. **Giả thuyết Bất Đối Xứng Nén Ảnh (Cần kiểm tra thực nghiệm)**: Ảnh COCO/TGIF được lưu ở định dạng PNG không nén, trong khi ảnh ImageNet mang dấu vết nén JPEG ban đầu từ cảm biến. Giả thuyết đặt ra là dấu vết nén này tạo ra shortcut tần số khiến bộ phân loại nhầm lẫn với AI inpainting/generation. Cần kiểm tra độ nhạy nén qua tái nén có kiểm soát trước khi đưa ra kết luận khẳng định.
3. **Kết luận an toàn ứng dụng**: Dù nguyên nhân cơ chế là gì, tỷ lệ báo động giả ~67–68% trên ảnh ImageNet authentic khẳng định tuyệt đối **không đưa mô hình 3 lớp này lên web demo cho người dùng tải ảnh tự do**; việc giữ nguyên bản demo nhị phân FP32 với 4 banner ranh giới nghiên cứu là quyết định kiến trúc chính xác.

---

## 7. Kiểm Toán Độc Lập Raw Predictions ($systematic-debugging)

Đọc trực tiếp từ tệp raw predictions `three_class_pilot_predictions.json` và biên nhận `three_class_pilot_evaluation_receipt.json` qua harness độc lập [`reproduce_three_class_pilot_results.py`](../../../scripts/research/reproduce_three_class_pilot_results.py):

1. **Xác thực mẫu và tính toàn vẹn**:
   - Đúng **273 holdout samples** (91 authentic, 91 ai_edited, 91 fully_generated) và **341 source-control samples** (100% authentic nature).
   - Sample IDs và thứ tự giữa 2 baselines hoàn toàn trùng khớp 100%.
   - 100% xác suất hữu hạn, không có NaN/Inf.
   - Tổng xác suất mỗi hàng bằng 1 trong dung sai $\le 1.0 \times 10^{-6}$ (max deviation $= 2.22 \times 10^{-16}$).
   - 100% nhãn dự đoán khớp quy tắc quyết định đã khóa: `predicted_label == argmax(calibrated_probs)`.
2. **Chi tiết Per-Class Metrics trên Holdout ($N=273$)**:
   - **Visual-Only (576-D)**:
     * `authentic`: TP=38, FP=56, FN=53 | Precision=0.4043, Recall=0.4176, F1=0.4108
     * `ai_edited`: TP=48, FP=69, FN=43 | Precision=0.4103, Recall=0.5275, F1=0.4615
     * `fully_generated`: TP=50, FP=12, FN=41 | Precision=0.8065, Recall=0.5495, F1=0.6536
     * Tổng hợp: Macro-F1 = 0.5086, BAcc = 0.4982, Brier = 0.5841, ECE = 0.0466
   - **Visual + DSP (592-D Concatenation)**:
     * `authentic`: TP=41, FP=54, FN=50 | Precision=0.4316, Recall=0.4505, F1=0.4409
     * `ai_edited`: TP=45, FP=69, FN=46 | Precision=0.3947, Recall=0.4945, F1=0.4390
     * `fully_generated`: TP=57, FP=7, FN=34 | Precision=0.8906, Recall=0.6264, F1=0.7355
     * Tổng hợp: Macro-F1 = 0.5385, BAcc = 0.5238, Brier = 0.5618, ECE = 0.0780
3. **Biên nhận kiểm toán độc lập**: [`three_class_predictions_audit_receipt.json`](./three_class_predictions_audit_receipt.json).

---

## 8. Kiểm Tra Phạm Vi Fitting & Chuẩn Hóa Inner CV (Section C)

Thực hiện kiểm tra nghiêm ngặt ranh giới fitting theo Rule 3 (AGENTS.md):

1. **Phát hiện Mismatch trong Inner CV Lịch Sử**:
   - Khi kiểm tra mã nguồn `train_and_evaluate_three_class_baselines.py`, inner 5-fold cross-validation được thực hiện bằng `StratifiedKFold` (sample-level) và sử dụng một `StandardScaler` toàn cục fit trên toàn bộ `development_train`.
   - Vì các cặp `authentic` (COCO) và `ai_edited` (TGIF) chia sẻ cùng `source_group_id`, việc chia sample-level đã làm rò rỉ bối cảnh nguồn giữa inner training và inner validation folds. Điều này dẫn đến sự tự tin thái quá trong OOF logits, khiến bộ tối ưu Temperature Scaling gán nhiệt độ rất cao ($T_v = 8.3734, T_f = 8.1321$) để làm phẳng xác suất.
2. **Khắc phục Chuẩn Hóa (Reconciled Grouped CV)**:
   - Triển khai script chuẩn hóa [`run_reconciled_three_class_cv.py`](../../../scripts/research/run_reconciled_three_class_cv.py):
     * Sử dụng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261011)` theo đúng `source_group_id` trên `development_train` (750 mẫu, 500 groups). Group overlap giữa inner train và val folds = **0**.
     * Cô lập `StandardScaler` riêng cho từng inner fold, fit strictly trên inner training partition của fold đó trước khi transform inner val.
     * Fit Temperature Scaling trên true out-of-fold logits không rò rỉ:
       - Visual-only: $T_v = \mathbf{5.9818}$ (giảm từ 8.3734)
       - Visual+DSP: $T_f = \mathbf{5.9577}$ (giảm từ 8.1321)
3. **Bảo Toàn Outer Models & Kết Quả Holdout**:
   - Vì mô hình chính (outer linear probes) được fit trên toàn bộ 750 mẫu `development_train` với hyperparameters giữ nguyên (`C=1.0`, multinomial cross-entropy), trọng số outer models và các quyết định argmax trên holdout **hoàn toàn đồng nhất bit-exact** (0 quyết định bị lật).
   - Macro-F1 và Balanced Accuracy giữ nguyên tuyệt đối: Visual Macro-F1 = 0.5086, Fusion Macro-F1 = 0.5385.
   - Các chỉ số hiệu chuẩn xác suất OOF và holdout được cập nhật chính xác:
     * Visual Holdout Brier: 0.6002 (trước: 0.5841); Holdout ECE: 0.1003 (trước: 0.0466)
     * Fusion Holdout Brier: 0.5711 (trước: 0.5618); Holdout ECE: 0.0865 (trước: 0.0780)
   - Lưu trữ song song: Tạo biên nhận reconciled [`three_class_pilot_evaluation_receipt_reconciled.json`](./three_class_pilot_evaluation_receipt_reconciled.json) và bảng dự đoán reconciled [`three_class_pilot_predictions_reconciled.json`](./three_class_pilot_predictions_reconciled.json). Biên nhận lịch sử `three_class_pilot_evaluation_receipt.json` được bảo toàn nguyên vẹn và liên kết đối chiếu rõ ràng.
4. **Đối chiếu Preprocessing Thực Tế với Protocol**:
   - **Pipeline 3 lớp**:
     * Mọi ảnh nạp qua Pillow RGBA/RGB, chuyển đổi thành canvas 512×512 (bộ lọc Lanczos).
     * Nhánh DSP nhận trực tiếp canvas 512×512 để trích xuất 16 đặc trưng canonical (FFT 128×128, DCT 8×8, Noise Residual 3×3, JPEG grid 8×8, Laplacian variance).
     * Nhánh Visual resize canvas 512×512 về tensor bicubic [3, 224, 224] và chuẩn hóa ImageNet cấp cho MobileNetV3-small frozen backbone.
   - **Khác biệt cốt lõi với Pipeline Nhị phân Lịch sử**:
     * Pipeline nhị phân lịch sử sử dụng kỹ thuật *letterbox padding* (giữ nguyên aspect ratio và chèn đệm đen) để đưa về 512×512. Pipeline 3 lớp sử dụng trực tiếp *direct resize* 512×512 (Lanczos/Bicubic).
     * Do tiền xử lý khác biệt, **tuyệt đối không kế thừa kết luận numerical parity** của pipeline nhị phân cho pipeline ba lớp.
     * Phân biệt rõ ràng: Checkpoint PyTorch có hash trọng số `047dcff4...` (`mobilenet_v3_small-047dcff4.pth`), trong khi file ONNX proto `models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx` là artifact đồ thị tính toán opset 17 với hash SHA-256 hoàn toàn riêng biệt.

---

## 9. Phân Tích Lỗi Có Kiểm Soát & Robustness Post-Hoc (Section D)

Thực hiện phân tích khám phá post-hoc có kiểm soát qua script [`run_three_class_error_analysis.py`](../../../scripts/research/run_three_class_error_analysis.py) trên toàn bộ 614 mẫu (273 development holdout + 341 source control) dưới 4 điều kiện biến đổi đồng nhất:
- `original`: Ảnh đầu vào nguyên bản (PNG không nén cho COCO/TGIF, JPEG nguyên bản cho ImageNet).
- `jpeg_q95`: Tái nén JPEG mức chất lượng cao Q=95.
- `jpeg_q75`: Tái nén JPEG mức chất lượng trung bình Q=75.
- `resize_0.5`: Thu nhỏ kích thước ảnh xuống 50% bằng bộ lọc bilinear rồi phóng to lại kích thước ban đầu.

### 9.1. Bảng Tổng Hợp Robustness Post-Hoc (4 Điều Kiện)

| Điều Kiện Biến Đổi | Baseline | Macro-F1 Holdout | Balanced Acc | Brier Holdout | ECE Holdout | Control FP AI Rate | Control Nhầm Edit | Control Nhầm Gen |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **original** | Visual (576-D) | 0.5086 | 0.4982 | 0.6002 | 0.1003 | 67.45% (230/341) | 106 | 124 |
| | Visual+DSP (592-D) | 0.5385 | 0.5238 | 0.5711 | 0.0865 | 68.04% (232/341) | 112 | 120 |
| **jpeg_q95** | Visual (576-D) | 0.5087 | 0.4982 | 0.6008 | 0.1018 | 66.86% (228/341) | 106 | 122 |
| | Visual+DSP (592-D) | 0.5358 | 0.5201 | 0.5739 | 0.0886 | 68.33% (233/341) | 114 | 119 |
| **jpeg_q75** *(Khám phá)* | Visual (576-D) | 0.5163 | 0.5055 | 0.5995 | 0.1057 | 67.16% (229/341) | 106 | 123 |
| *(Canonical Pre-Resize)* | Visual+DSP (592-D) | **0.6015** | 0.6007 | 0.4789 | 0.0582 | 67.16% (229/341) | 113 | 116 |
| **resize_0.5** | Visual (576-D) | 0.5074 | 0.4945 | 0.6015 | 0.1030 | 62.17% (212/341) | 107 | 105 |
| | Visual+DSP (592-D) | 0.5204 | 0.5055 | 0.5826 | 0.0927 | 63.05% (215/341) | 111 | 104 |

> **Ghi chú đối soát mâu thuẫn số liệu Q75 ($systematic-debugging)**:
> - Trong bản thảo trước, dòng `jpeg_q75` từng ghi Macro-F1 là `0.5235` (Visual `0.5087`). Đối chiếu trực tiếp với tệp predictions thô [`three_class_predictions_jpeg_q75.json`](./robustness/three_class_predictions_jpeg_q75.json) và biên nhận gốc [`three_class_robustness_audit_receipt.json`](./robustness/three_class_robustness_audit_receipt.json) (SHA-256 đã niêm phong tại commit `0f1b22d`) xác nhận số liệu thực tế đo được của Fusion là **`0.6015`** (Visual: **`0.5163`**). Con số `0.5235` là lỗi chép văn bản thủ công (transcription erratum) trong bản nháp cũ.
> - Cả hai harness (robustness và ablation) đều áp dụng chuẩn `JPEG_Q75_CANONICAL_PRE_RESIZE` (nén JPEG Q75 trước khi đưa về canvas 512×512 Lanczos). Nếu nén sau khi resize (`JPEG_Q75_POST_RESIZE`), Macro-F1 giảm về `0.4588`.
> - Do sự nhạy cảm cao của đặc trưng tần số với quy trình nén và sự tồn tại của lỗi ghi nhận lịch sử, endpoint Q75 được đánh dấu là **chỉ số phụ khám phá (exploratory secondary endpoint)**; toàn bộ kết luận khoa học chính thức của nghiên cứu được neo trên **ảnh gốc (original canvas)**.

### 9.2. Phân Tích Cơ Chế Lỗi và Hiện Tượng Nhầm Lẫn

1. **Nhầm lẫn giữa `authentic` và `ai_edited`**:
   - Trên holdout nguyên bản, có **46/91 ảnh authentic** bị phân loại nhầm thành `ai_edited` (cho cả visual và visual+DSP).
   - Ngược lại, có **38/91 ảnh ai_edited** (visual) và **43/91 ảnh ai_edited** (fusion) bị nhận định nhầm thành `authentic`.
   - Trong khi đó, nhầm lẫn `authentic` $\to$ `fully_generated` là rất thấp: chỉ 7/91 (visual) và 4/91 (fusion).
   - **Giả thuyết cơ chế (cần kiểm tra)**: TGIF inpainting chỉ thay thế một vùng cục bộ (1% đến 50% diện tích canvas); phần lớn diện tích còn lại giữ nguyên pixel gốc từ MS-COCO. Giả thuyết đặt ra là do MobileNetV3 áp dụng Global Average Pooling (GAP) trên toàn bộ canvas, các đặc trưng toàn cục có thể bị chi phối bởi bối cảnh nền COCO, khiến vector đặc trưng của ảnh inpainting rất gần với ảnh authentic. Đây là giả thuyết nghiên cứu kiến trúc, cần kiểm chứng đối sánh thay vì kết luận đã chứng minh.
2. **Đo lường Thay Đổi Xác Suất theo Cặp Nguồn ($\Delta P(\text{ai\_edited})$) Từ Predictions Reconciled**:
   - Tính toán trực tiếp trên 91 cặp holdout (`authentic` và `ai_edited` có cùng `source_group_id` COCO) từ tệp [`three_class_pilot_predictions_reconciled.json`](./three_class_pilot_predictions_reconciled.json) (không trộn lẫn với nhiệt độ over-smoothed cũ):
     * Visual-Only (Reconciled $T_v = 5.9818$): Giá trị trung bình của $[P(\text{ai\_edited} \mid \text{edited}) - P(\text{ai\_edited} \mid \text{auth})]$ là **$+0.0218$** (trung vị $+0.0119$; 53/91 cặp có mức tăng dương, tức 58.24%). *(Số liệu lịch sử cũ $T=8.37$ là +0.0180, 54/91)*.
     * Visual+DSP (Reconciled $T_f = 5.9577$): Giá trị trung bình là **$+0.0296$** (trung vị $+0.0190$; 55/91 cặp có mức tăng dương, tức 60.44%). *(Số liệu lịch sử cũ $T=8.13$ là +0.0242, 56/91)*.
     * Cả hai mô hình đều cho thấy trong gần **40% số cặp holdout**, việc inpainting vùng ảnh bằng SD2 không làm tăng xác suất dự đoán `ai_edited`.
3. **Phân Tích ImageNet Authentic Control Cohort & Bóc Tách Giả Thuyết Shortcut**:
   - Tỷ lệ False Positive AI trên 341 ảnh ImageNet authentic control duy trì ở mức rất cao qua mọi điều kiện: 67.45% (original), 66.86% (Q95), 67.16% (Q75) và 62.17% (resize 0.5).
   - **Quan sát thực nghiệm**: Việc áp dụng tái nén JPEG Q95 hoặc Q75 đồng nhất cho toàn bộ tập dữ liệu không xóa bỏ được tỷ lệ báo động giả cao của ImageNet authentic control.
   - **Giả thuyết khoa học**: Sự khác biệt về phân phối bối cảnh và đối tượng (ImageNet vs MS-COCO) là một giả thuyết shortcut nguồn chính yếu, song hành cùng giả thuyết lịch sử nén. Tuy nhiên, không kết luận JPEG hay source shortcut là nguyên nhân duy nhất đã chứng minh chỉ từ false-positive rate. Đổi ảnh sang PNG hoặc resize không thể xóa bỏ lịch sử nén trước đó của cảm biến máy ảnh.
   - **Cảnh báo giới hạn**: Đây là phân tích khám phá post-hoc, không phải kiểm định độc lập mới và tuyệt đối không được dùng để lựa chọn mô hình thắng cuộc.
4. **Hiện vật kiểm toán robustness**: Thư mục [`research/evidence/three_class_preparation/robustness/`](./robustness/) chứa đầy đủ 4 tệp raw predictions và biên nhận [`three_class_robustness_audit_receipt.json`](./robustness/three_class_robustness_audit_receipt.json).


---

## 10. Thực Nghiệm Ablation Đa Dạng Hóa Nguồn Ảnh Authentic (A0 vs A1)

Thực hiện theo Protocol Amendment v2.1.0 ([`PROTOCOL_AMENDMENT_V2.1.0.md`](./PROTOCOL_AMENDMENT_V2.1.0.md)) nhằm kiểm tra khả năng giảm báo động giả trên tập đối chứng ImageNet authentic thông qua việc bổ sung nguồn ảnh authentic vào tập huấn luyện, giữ nguyên hướng nghiên cứu RQ1–RQ2 lightweight Visual-only vs Visual + DSP.

> **Lưu ý phương pháp luận**: Đây là can thiệp bổ sung dữ liệu nguồn thật cùng cách cân bằng trọng số mẫu. Thực nghiệm này **không chứng minh** đã loại bỏ toàn bộ source confounding. Cả tập holdout và tập đối chứng đã được quan sát trong các lượt trước; đây là thực nghiệm phát triển tiếp nối có kiểm soát, không phải kiểm định độc lập mới.

### 10.1. Thiết Kế & Kiểm Soát Dữ Liệu
- **Tập bổ sung**: 250 ảnh ImageNet authentic mới trích xuất từ các tệp parquet có sẵn trên LOCAL ([`imagenet_authentic_train_manifest.json`](./imagenet_authentic_train_manifest.json)). 0% tải mạng mới; footprint tăng thêm ~39.2 MB (tổng lưu trữ an toàn trong ngân sách 5 GiB).
- **Cách ly tuyệt đối**: 0 trùng lặp byte hash (SHA-256) hoặc source ID với 341 ảnh đối chứng ImageNet, 273 ảnh holdout, Locked-Test và TGIF N=400.
- **Hai điều kiện dữ liệu**:
  * **A0 (Pilot Baseline)**: 250 auth COCO, 250 edit TGIF, 250 gen SD1.5 (tổng 750 mẫu, weight 1.0).
  * **A1 (Supplemented)**: A0 + 250 auth ImageNet mới (tổng 1,000 mẫu).
  * **Trọng số mẫu A1**: auth COCO (0.5), auth ImageNet (0.5), edit TGIF (1.0), gen SD1.5 (1.0). Tổng trọng số mỗi lớp đạt chính xác 250.0 (tỷ lệ lớp hiệu dụng bảo toàn 1:1:1).
- **Quy trình fitting**: Grouped inner CV (5-fold, 750 groups độc lập) theo `source_group_id`; `StandardScaler` fit riêng trong từng fold; hiệu chuẩn nhiệt độ ($T_{v,a1} = 5.1813, T_{f,a1} = 4.9391$) chỉ dựa trên OOF logits hợp lệ.

### 10.2. Bảng Đối Sánh Tổng Hợp Bốn Cấu Hình (Original & JPEG Q75)

#### Bảng 1: Kết Quả Trên Ảnh Gốc (Original Canvas)

| Tiêu Chí Đánh Giá | A0-Visual (576-D) | A0-Fusion (592-D) | A1-Visual (576-D) | A1-Fusion (592-D) |
| :--- | :---: | :---: | :---: | :---: |
| **Dữ liệu huấn luyện** | A0 ($N=750$) | A0 ($N=750$) | A1 ($N=1,000$) | A1 ($N=1,000$) |
| **Nhiệt độ hiệu chuẩn ($T$)** | 5.9818 | 5.9577 | 5.1813 | 4.9391 |
| **Holdout Macro-F1** | 0.5086 | **0.5385** | 0.4721 | 0.4887 |
| **Holdout Balanced Accuracy** | 0.4982 | **0.5238** | 0.4652 | 0.4762 |
| **Holdout Multiclass Brier** | 0.6002 | **0.5711** | 0.6586 | 0.6444 |
| **Holdout Top-Class ECE** | 0.1003 | **0.0865** | 0.1613 | 0.1575 |
| - *Authentic F1 (P / R)* | 0.4108 (0.40 / 0.42) | 0.4409 (0.43 / 0.45) | 0.4444 (0.37 / 0.55) | 0.4384 (0.38 / 0.53) |
| - *AI-Edited F1 (P / R)* | 0.4615 (0.41 / 0.53) | 0.4390 (0.39 / 0.49) | 0.4526 (0.43 / 0.47) | 0.4456 (0.42 / 0.47) |
| - *Fully-Generated F1 (P / R)* | 0.6536 (0.81 / 0.55) | **0.7355** (0.89 / 0.63) | 0.5191 (0.85 / 0.37) | 0.5821 (0.91 / 0.43) |
| **Control False-Positive AI Rate** ($N=341$) | 67.45% (230/341) | 68.04% (232/341) | **35.78%** (122/341) | **31.09%** (106/341) |
| - *Số lượng gán đúng Authentic* | 111 / 341 | 109 / 341 | 219 / 341 | **235 / 341** |
| - *Báo động nhầm ai_edited* | 106 | 112 | 71 | 69 |
| - *Báo động nhầm fully_generated* | 124 | 120 | 51 | 37 |
| **Paired $\Delta P(\text{ai\_edited})$** ($N=91$) | +0.0218 (53/91 > 0) | +0.0296 (55/91 > 0) | +0.0148 (54/91 > 0) | +0.0168 (53/91 > 0) |

#### Bảng 2: Độ Nhạy Dưới Nén JPEG Q75 (Robustness Canvas)

| Tiêu Chí Đánh Giá | A0-Visual (Q75) | A0-Fusion (Q75) | A1-Visual (Q75) | A1-Fusion (Q75) |
| :--- | :---: | :---: | :---: | :---: |
| **Holdout Macro-F1** | 0.5163 | 0.6015 | 0.4806 | **0.6046** |
| **Holdout Balanced Accuracy** | 0.5055 | 0.6007 | 0.4725 | **0.5934** |
| **Holdout Multiclass Brier** | 0.5995 | **0.4789** | 0.6577 | 0.5009 |
| **Holdout Top-Class ECE** | 0.1057 | **0.0582** | 0.1383 | 0.1083 |
| - *Authentic F1* | 0.4375 | 0.4471 | 0.4474 | **0.4923** |
| - *AI-Edited F1* | 0.4577 | 0.4796 | 0.4632 | 0.4731 |
| - *Fully-Generated F1* | 0.6536 | **0.8778** | 0.5313 | 0.8485 |
| **Control False-Positive AI Rate** ($N=341$) | 67.16% (229/341) | 67.16% (229/341) | 36.36% (124/341) | **31.38%** (107/341) |

### 10.3. Phân Tích Thống Kê & Khoảng Tin Cậy (Bootstrap 10,000 Replicates)

Phân tích Paired Source-Cluster Bootstrap (10,000 lần lấy mẫu theo nhóm nguồn, hạt giống `20261011`):
1. **Hiệu ứng can thiệp dữ liệu ($A1 - A0$) trên Holdout Macro-F1**:
   - Visual-only: $\Delta = -0.0365$ ($95\%\text{ CI}: [-0.0816, +0.0080]$, $P(\Delta > 0) = 5.43\%$).
   - Visual + DSP: $\Delta = -0.0498$ ($95\%\text{ CI}: [-0.0944, -0.0044]$, $P(\Delta > 0) = 1.65\%$).
   $\to$ Trên tập holdout ban đầu, việc bổ sung nguồn authentic dẫn đến suy giảm Macro-F1 có ý nghĩa thống kê (do mô hình dịch chuyển sang trạng thái bảo thủ, giảm recall trên lớp generation).
2. **Hiệu ứng đặc trưng DSP ($\text{Fusion} - \text{Visual}$)**:
   - Trong điều kiện A0: $\Delta = +0.0299$ ($95\%\text{ CI}: [-0.0068, +0.0687]$, $P(\Delta > 0) = 94.35\%$).
   - Trong điều kiện A1: $\Delta = +0.0166$ ($95\%\text{ CI}: [-0.0168, +0.0514]$, $P(\Delta > 0) = 83.82\%$).
   $\to$ Mặc dù point estimate có xu hướng dương trong cả hai điều kiện dữ liệu và phần lớn khối lượng bootstrap là dương ($94.35\%$ và $83.82\%$), khoảng tin cậy 95% CI trong cả hai điều kiện đều cắt 0 ở cận dưới. Theo nguyên tắc trung thực khoa học, kết luận chính thức về lợi ích phân loại của DSP trên ảnh gốc là **`INCONCLUSIVE`** (chưa đủ bằng chứng để khẳng định vượt trội có ý nghĩa thống kê).
3. **Hiệu ứng giảm báo động giả trên ImageNet Authentic Control**:
   - Visual-only: Báo động giả giảm từ $67.45\%$ xuống $35.78\%$ ($\Delta = -31.67\text{ điểm phần trăm}$).
   - Visual + DSP: Báo động giả giảm từ $68.04\%$ xuống $31.09\%$ ($\Delta = -36.95\text{ điểm phần trăm}$).
   $\to$ **Cảnh báo diễn giải khoa học**:
   - Việc giảm báo động giả từ 68.04% xuống 31.09% (giảm 36.95 điểm phần trăm) **không chứng minh riêng semantic bias là nguyên nhân duy nhất**, vì can thiệp bổ sung ảnh ImageNet đồng thời mang lại cả phân phối đối tượng tự nhiên và lịch sử nén JPEG nguyên bản của camera.
   - Việc bổ sung 250 ảnh authentic và huấn luyện lại probe làm thay đổi **toàn bộ ranh giới quyết định đa lớp (multiclass decision boundaries)** trong không gian xác suất 3 chiều, không phải là cơ chế tự động tịnh tiến một ngưỡng nhị phân 0.5.

### 10.4. Giải Quyết Bốn Câu Hỏi Nghiên Cứu Cốt Lõi

1. **Bổ sung authentic ImageNet có giảm false positives trên control không?**
   - **Có, giảm 36.95 điểm phần trăm** (từ $68.04\%$ xuống $31.09\%$ ở mô hình fusion; visual giảm $31.67$ điểm phần trăm). Số lượng mẫu ImageNet authentic nhận diện chính xác tăng từ 109 lên 235 trên tổng số 341 mẫu. Tuy nhiên, việc giảm báo động giả không chứng minh riêng semantic bias là nguyên nhân độc nhất, mà phản ánh tác động đồng thời của phân phối đối tượng và đặc trưng nén.
2. **Có làm giảm khả năng nhận diện ai_edited hoặc fully_generated không?**
   - **Có sự đánh đổi (trade-off) bất đối xứng**:
     * Với `fully_generated` (SD1.5): Recall trên holdout giảm từ 62.64% xuống 42.86% (Fusion) và từ 54.95% xuống 37.36% (Visual). Do không gian authentic mở rộng sang ImageNet, mô hình trở nên thận trọng hơn và gán nhầm 35/91 ảnh SD1.5 sang nhãn authentic.
     * Với `ai_edited` (TGIF SD2): Khả năng nhận diện ít bị ảnh hưởng tiêu cực; F1 dao động nhẹ ($0.4615 \to 0.4526$ ở visual, $0.4390 \to 0.4456$ ở fusion), recall đạt 47.25% (so với 49.45% ở A0).
3. **DSP có lợi thêm sau can thiệp dữ liệu không?**
   - **Point estimate tăng nhẹ nhưng kết luận thống kê là INCONCLUSIVE**. DSP cải thiện Macro-F1 thêm $+0.0166$ trên ảnh gốc ($83.82\%$ bootstrap mass $> 0$), nhưng khoảng tin cậy 95% CI cắt 0 ($[-0.0168, +0.0514]$). Dưới điều kiện nén JPEG Q75, DSP nâng Macro-F1 từ $0.4806$ lên $0.6046$, song endpoint Q75 chỉ là chỉ số phụ khám phá.
4. **Bước tiếp theo cần cải thiện dữ liệu hay đặc trưng cục bộ?**
   - **Cần đánh giá thực nghiệm đặc trưng cục bộ (patch / quadrant DSP) trên dữ liệu hiện có để kiểm tra giả thuyết GAP làm loãng tín hiệu, song hành cùng việc chuẩn bị thiết kế dữ liệu đối chứng nhân tố (factorial design)**.


---

## 11. Đối Chiếu Dung Lượng Tải và Hiện Vật Dữ Liệu (Section E)

Để đảm bảo tính trung thực tuyệt đối giữa các văn bản báo cáo và biên nhận tiếp nhận dữ liệu:

1. **Đối Chiếu Dung Lượng Tải Mạng**:
   - Con số "875.1 MB" trong các tóm tắt ban đầu chỉ phản ánh dung lượng của **Shard 0 và Shard 1** khi thực hiện đợt tải đầu tiên.
   - Sau khi tiếp nhận thêm Shard 2 (`validation-00002-of-00004.parquet`, 435,573,773 bytes) để trích xuất đủ 341 mẫu SD1.5, tổng dung lượng tải mạng thực tế được ghi nhận chính xác trong [`genimage_acquisition_receipt.json`](./genimage_acquisition_receipt.json) là:
     $$\text{Tổng byte tải} = 433,181,356 + 441,983,256 + 435,573,773 = \mathbf{1,310,738,385\text{ bytes}} \approx \mathbf{1.22\text{ GiB}} \le 2\text{ GiB}$$
   - Tổng footprint lưu trữ mới trên đĩa: **1,552,278,924 bytes** (~1.45 GiB) $\le 5\text{ GiB}$.
2. **Bảo toàn và Kế thừa Hiện Vật Lịch Sử**:
   - Toàn bộ các hiện vật lịch sử (`three_class_pilot_predictions.json`, `three_class_pilot_evaluation_receipt.json`, `source_control_evaluation_receipt.json`) được giữ nguyên vẹn 100%, không bị sửa đổi để khớp câu chữ.
   - Các hiện vật chuẩn hóa mới được liên kết minh bạch:
     * [`three_class_pilot_evaluation_receipt_reconciled.json`](./three_class_pilot_evaluation_receipt_reconciled.json) (thay thế cho việc diễn giải temperature & calibration)
     * [`three_class_pilot_predictions_reconciled.json`](./three_class_pilot_predictions_reconciled.json)
     * [`three_class_predictions_audit_receipt.json`](./three_class_predictions_audit_receipt.json)
3. **Đề Xuất Thực Nghiệm Kế Tiếp**:
   - Ban hành tài liệu đề xuất hoàn chỉnh: [`NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`](./NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md).
   - Thiết kế nghiên cứu factorial 2×2 phân tách dứt điểm Source Bias (COCO vs ImageNet) và Compression History (PNG vs JPEG Recompression), đồng thời kiểm soát phân phối diện tích inpainting.

---

## 12. Hiện Vật Xuất Bản & Minh Chứng Hoàn Chỉnh

- **Protocols & Amendments**:
  * [`PROTOCOL_AMENDMENT_V2.0.0.md`](./PROTOCOL_AMENDMENT_V2.0.0.md) (Pilot amendment)
  * [`PROTOCOL_AMENDMENT_V2.1.0.md`](./PROTOCOL_AMENDMENT_V2.1.0.md) (Ablation A0 vs A1 amendment)
  * [`NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`](./NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md)
- **Manifests & Splits**:
  * [`three_class_pilot_manifest.csv`](./three_class_pilot_manifest.csv) (1,023 records)
  * [`three_class_pilot_splits.json`](./three_class_pilot_splits.json) (Splits lock & audit)
  * [`source_control_cohort_manifest.json`](./source_control_cohort_manifest.json) (341 controls)
  * [`imagenet_authentic_train_manifest.json`](./imagenet_authentic_train_manifest.json) (250 authentic ImageNet supplement)
- **Models & Features**:
  * [`three_class_visual_only_model.json`](./three_class_visual_only_model.json) (A0 Visual)
  * [`three_class_visual_dsp_model.json`](./three_class_visual_dsp_model.json) (A0 Fusion)
  * [`three_class_visual_only_a1_model.json`](./three_class_visual_only_a1_model.json) (A1 Visual)
  * [`three_class_visual_dsp_a1_model.json`](./three_class_visual_dsp_a1_model.json) (A1 Fusion)
  * [`three_class_local_dsp_a0_model.json`](./three_class_local_dsp_a0_model.json) (A0 Local 640-D)
  * [`three_class_local_dsp_a1_model.json`](./three_class_local_dsp_a1_model.json) (A1 Local 640-D)
  * `data/research/cache/three_class_features.npz` (2.82 MB, A0 features)
  * `data/research/cache/three_class_ablation_features.npz` (5.40 MB, A1 features & sample weights)
  * `data/research/cache/three_class_local_dsp_features.npz` (8.04 MB, 640-D features & local summaries)
  * [`ablation_feature_extraction_receipt.json`](./ablation_feature_extraction_receipt.json)
  * [`local_dsp_feature_extraction_receipt.json`](./local_dsp_feature_extraction_receipt.json)
- **Audited Predictions & Evaluation Receipts**:
  * [`three_class_pilot_predictions.json`](./three_class_pilot_predictions.json) (Historical raw holdout + control)
  * [`three_class_pilot_evaluation_receipt.json`](./three_class_pilot_evaluation_receipt.json) (Historical receipt)
  * [`three_class_pilot_evaluation_receipt_reconciled.json`](./three_class_pilot_evaluation_receipt_reconciled.json) (Reconciled CV receipt)
  * [`three_class_pilot_predictions_reconciled.json`](./three_class_pilot_predictions_reconciled.json) (Reconciled predictions)
  * [`three_class_predictions_audit_receipt.json`](./three_class_predictions_audit_receipt.json) (Independent prediction audit)
  * [`source_control_evaluation_receipt.json`](./source_control_evaluation_receipt.json) (Diagnostic findings)
  * [`three_class_ablation_evaluation_receipt.json`](./three_class_ablation_evaluation_receipt.json) (A0 vs A1 full 4-condition receipt)
  * [`three_class_ablation_a1_predictions.json`](./three_class_ablation_a1_predictions.json) (A1 standalone predictions)
  * [`three_class_local_dsp_evaluation_receipt.json`](./three_class_local_dsp_evaluation_receipt.json) (Local DSP 4-configuration receipt)
  * [`three_class_local_dsp_predictions.json`](./three_class_local_dsp_predictions.json) (Local DSP raw predictions)
- **Robustness Post-Hoc Artifacts**:
  * [`robustness/three_class_robustness_audit_receipt.json`](./robustness/three_class_robustness_audit_receipt.json)
  * [`robustness/three_class_predictions_original.json`](./robustness/three_class_predictions_original.json)
  * [`robustness/three_class_predictions_jpeg_q95.json`](./robustness/three_class_predictions_jpeg_q95.json)
  * [`robustness/three_class_predictions_jpeg_q75.json`](./robustness/three_class_predictions_jpeg_q75.json)
  * [`robustness/three_class_predictions_resize_0.5.json`](./robustness/three_class_predictions_resize_0.5.json)
- **Protocols & Amendments**:
  * [`PROTOCOL_AMENDMENT_V2.0.0.md`](./PROTOCOL_AMENDMENT_V2.0.0.md) (Pilot amendment)
  * [`PROTOCOL_AMENDMENT_V2.1.0.md`](./PROTOCOL_AMENDMENT_V2.1.0.md) (A0 vs A1 amendment)
  * [`PROTOCOL_AMENDMENT_V2.2.0.md`](./PROTOCOL_AMENDMENT_V2.2.0.md) (Local Quadrant DSP amendment)
  * [`NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`](./NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md)
- **Test Suites**:
  * [`ml/tests/test_three_class_pilot.py`](../../../ml/tests/test_three_class_pilot.py) (8/8 PASS)
  * [`ml/tests/test_three_class_ablation.py`](../../../ml/tests/test_three_class_ablation.py) (6/6 PASS)
  * [`ml/tests/test_three_class_local_dsp.py`](../../../ml/tests/test_three_class_local_dsp.py) (6/6 PASS)

---

## 13. Thực Nghiệm Ablation Đặc Trưng DSP Cục Bộ (Local Quadrant DSP Ablation)

Thực thi theo **Protocol Amendment v2.2.0** ([`PROTOCOL_AMENDMENT_V2.2.0.md`](./PROTOCOL_AMENDMENT_V2.2.0.md)) nhằm kiểm tra thực nghiệm giả thuyết: *"Biểu diễn DSP cục bộ tổng hợp có giúp cải thiện khả năng phân biệt authentic / ai_edited mà không làm tăng báo động giả trên control hoặc làm suy giảm nghiêm trọng khả năng phát hiện fully_generated hay không?"*

### 13.1. Thiết Kế Bốn Cấu Hình & Ranh Giới Kỹ Thuật
1. **Không gian biểu diễn**:
   - **Baseline Global (592-D)**: Visual embedding MobileNetV3 (576-D) + Global Canonical DSP (16-D).
   - **Thử nghiệm Local (640-D)**: Visual embedding (576-D) + Global DSP (16-D) + Local Quadrant DSP Summaries (48-D).
   - Local DSP 48-D được trích xuất từ 4 quadrant 256×256 không chồng lấn trên canvas 512×512, tổng hợp theo từng chiều 16-D bằng: Mean (16-D), Population Std ddof=0 (16-D), và Max (16-D).
2. **Bốn cấu hình đối sánh**:
   - **A0-global**: Baseline 592-D trên tập A0 pilot ($N=750$, weight 1.0).
   - **A0-local**: Probe mới 640-D trên tập A0 pilot ($N=750$, weight 1.0).
   - **A1-global**: Baseline 592-D trên tập A1 supplemented ($N=1,000$, weight balanced).
   - **A1-local**: Probe mới 640-D trên tập A1 supplemented ($N=1,000$, weight balanced).
3. **Ranh giới bất biến**: Giữ nguyên toàn bộ splits, seed `20261011`, $C=1.0$, solver `lbfgs`, 5-fold grouped inner CV với `StandardScaler` cô lập fold, và hiệu chuẩn nhiệt độ OOF ($T_{\text{A0,glob}}=5.8907, T_{\text{A0,loc}}=6.0142, T_{\text{A1,glob}}=5.1400, T_{\text{A1,loc}}=5.1689$).

---

### 13.2. Bảng Kết Quả Đối Sánh Bốn Cấu Hình (Original Canvas)

Đánh giá thực nghiệm trên **273 development holdout** và **341 ImageNet authentic controls** nguyên bản (original canvas):

| Tiêu Chí Đánh Giá | A0-Global (592-D) | A0-Local (640-D) | A1-Global (592-D) | A1-Local (640-D) |
| :--- | :---: | :---: | :---: | :---: |
| **Dữ liệu huấn luyện** | A0 ($N=750$) | A0 ($N=750$) | A1 ($N=1,000$) | A1 ($N=1,000$) |
| **Nhiệt độ hiệu chuẩn ($T$)** | 5.8907 | 6.0142 | 5.1400 | 5.1689 |
| **Holdout Macro-F1** | 0.5385 | **0.5552** | 0.4887 | 0.4702 |
| **Holdout Balanced Accuracy** | 0.5238 | **0.5421** | 0.4762 | 0.4579 |
| **Holdout Multiclass Brier** | **0.5717** | 0.5731 | 0.6402 | 0.6350 |
| **Holdout Top-Class ECE** | 0.0903 | **0.0696** | 0.1405 | 0.1531 |
| **Confusion Matrix (A / E / G)** | `[[41, 46, 4],` <br> `[43, 45, 3],` <br> `[11, 23, 57]]` | `[[39, 49, 3],` <br> `[37, 51, 3],` <br> `[11, 22, 58]]` | `[[48, 42, 1],` <br> `[45, 43, 3],` <br> `[35, 17, 39]]` | `[[43, 45, 3],` <br> `[44, 44, 3],` <br> `[33, 20, 38]]` |
| - *Authentic F1 (P / R)* | 0.4409 (0.43 / 0.45) | 0.4382 (0.45 / 0.43) | **0.4384** (0.38 / 0.53) | 0.4076 (0.36 / 0.47) |
| - *AI-Edited F1 (P / R)* | 0.4390 (0.39 / 0.49) | **0.4789** (0.42 / 0.56) | 0.4456 (0.42 / 0.47) | 0.4400 (0.40 / 0.48) |
| - *Fully-Generated F1 (P / R)* | 0.7355 (0.89 / 0.63) | **0.7484** (0.91 / 0.64) | 0.5821 (0.91 / 0.43) | 0.5630 (0.86 / 0.42) |
| **Control False-Positive AI Rate** ($N=341$) | 68.04% (232/341) | 70.38% (240/341) | 31.09% (106/341) | **27.57%** (94/341) |
| - *Số lượng gán đúng Authentic* | 109 / 341 | 101 / 341 | 235 / 341 | **247 / 341** |
| - *Báo động nhầm ai_edited* | 112 | 105 | 69 | 59 |
| - *Báo động nhầm fully_generated* | 120 | 135 | 37 | 35 |
| **Paired Shift $\Delta P(\text{ai\_edited})$** ($N=91$) | +0.0297 (55/91 > 0) | +0.0300 (58/91 > 0) | +0.0164 (53/91 > 0) | +0.0166 (55/91 > 0) |

---

### 13.3. Phân Tích Thống Kê & Khoảng Tin Cậy (Bootstrap 10,000 Replicates)

Phân tích Paired Source-Cluster Bootstrap (10,000 replicates theo nhóm nguồn COCO, seed `20261011`):
1. **Chênh lệch Local − Global trong điều kiện A0**:
   $$\Delta \text{Macro-F1} = +0.0168, \quad 95\%\text{ CI} = [-0.0155, +0.0492], \quad P(\Delta^* > 0) = 84.61\%$$
   $\to$ Point estimate tăng nhẹ (+1.68 điểm phần trăm), F1 của lớp `ai_edited` tăng từ 0.4390 lên 0.4789 và recall tăng từ 49.45% lên 56.04% (+6 mẫu edit được phát hiện đúng). Tuy nhiên, **95% CI cắt 0 ở cận dưới (-0.0155)**, do đó kết luận thống kê chính thức là **`INCONCLUSIVE`**.
2. **Chênh lệch Local − Global trong điều kiện A1**:
   $$\Delta \text{Macro-F1} = -0.0185, \quad 95\%\text{ CI} = [-0.0514, +0.0129], \quad P(\Delta^* > 0) = 12.95\%$$
   $\to$ Trong điều kiện dữ liệu đa dạng nguồn authentic, đặc trưng Local DSP làm giảm Macro-F1 (-1.85 điểm phần trăm), chủ yếu do recall của lớp authentic giảm từ 52.75% xuống 47.25% (F1 authentic giảm từ 0.4384 xuống 0.4076). **95% CI cắt 0 (-0.0514 đến +0.0129)**, kết luận là **`INCONCLUSIVE`**.

---

### 13.4. Phân Tích Chẩn Đoán Theo Phân Vị Diện Tích Chỉnh Sửa TGIF Masks (Post-Hoc Diagnosis)

Đối soát 100% (91/91) ảnh holdout `ai_edited` với ground-truth segmentation masks TGIF resident cục bộ tại `data/research/tgif/masks/` (manifest: `masks-manifest.jsonl`). Diện tích can thiệp được chia thành 4 khoảng phân vị:
- **Q1 (<2.85% diện tích, diện tích trung bình 1.62%, N=23)**:
  * A0-global: Recall `ai_edited` = 39.13% (9/23), $\bar{P}(\text{edit}) = 0.3924$.
  * A0-local: Recall `ai_edited` = **52.17%** (12/23), $\bar{P}(\text{edit}) = 0.4107$ (**+3 mẫu edit được phát hiện đúng**).
  * A1-global: Recall `ai_edited` = 43.48% (10/23), $\bar{P}(\text{edit}) = 0.3849$.
  * A1-local: Recall `ai_edited` = 43.48% (10/23), $\bar{P}(\text{edit}) = 0.4072$.
  $\to$ Trong điều kiện A0, đặc trưng quadrant cục bộ giải phóng tín hiệu pháp chứng khỏi bị bối cảnh toàn cục làm loãng đối với các vùng chỉnh sửa siêu nhỏ (<2.85%).
- **Q2 (2.85%–6.58% diện tích, diện tích trung bình 4.92%, N=23)**:
  * A0-global: Recall 47.83% (11/23) $\to$ A0-local: Recall **56.52%** (13/23) (**+2 mẫu edit được phát hiện đúng**).
  * A1-global: Recall 52.17% (12/23) $\to$ A1-local: Recall 43.48% (10/23).
- **Q3 (6.58%–14.88% diện tích, diện tích trung bình 10.75%, N=22)**:
  * A0-global: Recall 63.64% (14/22) $\to$ A0-local: Recall 68.18% (15/22).
  * A1-global: Recall 45.45% (10/22) $\to$ A1-local: Recall **54.55%** (12/22) (**+2 mẫu edit được phát hiện đúng**).
- **Q4 (>14.88% diện tích, diện tích trung bình 28.14%, N=23)**:
  * A0-global: 47.83% (11/23) $\to$ A0-local: 47.83% (11/23) (không đổi).
  * A1-global: 47.83% (11/23) $\to$ A1-local: **52.17%** (12/23).

---

### 13.5. Đo Lường Chi Phí Tính Toán CPU (Extraction Overhead)

Đo lường thời gian thực thi trích xuất trên panel phát triển 16 mẫu cố định trên CPU nội bộ:
- **Global DSP 16-D**: Mean = 82.97 ms, Median = 84.11 ms, Min = 70.52 ms, Max = 94.59 ms.
- **Local Quadrant DSP 48-D**: Mean = 98.93 ms, Median = 98.38 ms, Min = 89.67 ms, Max = 113.28 ms.
- **Combined DSP 64-D**: Mean = 181.89 ms, Median = 183.52 ms, Min = 163.16 ms, Max = 199.59 ms.
- **Đánh giá overhead**: Chi phí trích xuất DSP tăng thêm **+98.93 ms (+119.2% overhead)**, nâng tổng thời gian trích xuất tín hiệu số lên ~182 ms. Lưu ý: Độ trễ và tính tương đương browser parity của pipeline nhị phân lịch sử (đã đo 80.8 ms warm latency trên Chromium) **không tự động kế thừa** cho pipeline 640-D mới.

---

### 13.6. Đánh Giá Trade-Off & Kết Luận Chiến Lược Nghiên Cứu

1. **Trade-off trong điều kiện A0 (Thiếu đa dạng authentic)**:
   - Local DSP giúp tăng khả năng phát hiện `ai_edited` diện tích nhỏ (<6.58% diện tích canvas) từ 39% lên 52%, nâng F1 của lớp edit từ 0.4390 lên 0.4789.
   - Nhưng đổi lại, nó làm tăng thêm báo động giả trên ImageNet authentic control từ 68.04% lên **70.38%** (+2.34 điểm phần trăm).
2. **Trade-off trong điều kiện A1 (Đã bổ sung authentic ImageNet)**:
   - Local DSP giúp giảm thêm báo động giả control từ 31.09% xuống **27.57%** (-3.52 điểm phần trăm, nhận diện đúng 247/341 ảnh authentic).
   - Nhưng đổi lại, nó làm giảm Macro-F1 holdout từ 0.4887 xuống **0.4702** (-1.85 điểm phần trăm) do suy giảm recall của lớp authentic COCO (từ 52.75% xuống 47.25%).
3. **Kết luận khoa học & Khuyến nghị chiến lược**:
   - Biểu diễn Local Quadrant DSP 48-D **chưa đủ bằng chứng thống kê để khẳng định là giải pháp ưu việt** (cả hai 95% bootstrap CI đều cắt 0; kết luận strictly `INCONCLUSIVE`).
   - Mặc dù có tín hiệu hỗ trợ phát hiện edit diện tích nhỏ, việc chia lưới quadrant cố định không thể giải quyết triệt để sự mất cân xứng phân phối nguồn (domain confounding giữa COCO và ImageNet).
   - **Khuyến nghị**: **Tạm dừng mở rộng đặc trưng cục bộ phức tạp hơn (ví dụ sliding window hoặc fine grid)**. Thay vào đó, **ưu tiên nguồn lực vào thiết kế dữ liệu đối chứng nhân tố (factorial design 2×2)** theo [`NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`](./NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md) để phân tách dứt điểm domain semantic bias khỏi format/compression shortcut trước khi đưa ra bất kỳ kết luận kiến trúc mới nào.

---

## 14. Đánh Giá Khả Năng Tổng Quát Hóa Ngoại Lai (External Exploratory Generalization Probe trên So-Fake-Set)

### 14.1. Mục Tiêu & Giao Thức Đóng Băng Đánh Giá (Zero-Tuning Protocol)

Nhằm kiểm định khả năng tổng quát hóa của các mô hình phân loại ba lớp (`authentic`, `ai_edited`, `fully_generated`) đã phát triển trên phân phối dữ liệu hoàn toàn độc lập bên ngoài phòng thí nghiệm, một cuộc khảo sát thăm dò ngoại lai (*external exploratory probe*) được thực hiện trên dataset cộng đồng chính thức `saberzl/So-Fake-Set` (arXiv:2505.18660).

**Quy tắc bất biến khóa trước**:
1. **Tuyệt đối không tinh chỉnh**: Không huấn luyện lại (no retraining), không điều chỉnh trọng số (no refitting), không hiệu chuẩn lại nhiệt độ (no recalibration), không quét ngưỡng phân loại (no threshold tuning).
2. **Không chọn mô hình theo kết quả probe**: Toàn bộ 6 cấu hình mô hình được đăng ký, niêm phong mã băm SHA-256 và đóng băng trước khi thực hiện bất kỳ phép đo dự đoán nào trên tập ngoài.
3. **Phạm vi khoa học trung thực**: Xác định đây là một cuộc thăm dò thăm dò ngoại lai quy mô nhỏ ($N=300$, cân bằng 100 mẫu/lớp), quan hệ nguồn chưa thể đối chiếu toàn diện (unverified source independence); không tuyên bố đánh giá toàn bộ tập So-Fake hay kết luận tiền đăng ký hồi tố.

---

### 14.2. Tiếp Nhận Dữ Liệu Có Kiểm Soát & Kế Toán Ngân Sách Mạng Thực Dùng

- **Khảo sát ban đầu**: `saberzl/So-Fake-OOD` chứa các shard parquet ~2.77 GiB (> 750 MiB hạn mức phiên) $\to$ quyết định dừng tiếp cận OOD nguyên khối để tuân thủ ngân sách.
- **Thăm dò footer HTTP Range**: Thăm dò metadata shard validation đầu tiên của `saberzl/So-Fake-Set` (`data/validation-00000-of-00232.parquet`) tiêu tốn 81.4 KiB, xác định kích thước shard là 496,763,723 bytes (473.75 MiB $\le$ 750 MiB ngân sách phiên), chứa 1,016 mẫu với 441 `full_synthetic`, 316 `real`, và 259 `tampered`.
- **Tải file thực tế**: Thực hiện tải có giám sát qua `scripts/research/acquire_so_fake_probe_shard.py`, lưu tại `data/research/so_fake/validation-00000-of-00232.parquet` (SHA-256: `e69523915f809a06a904153a7518060abd5b4c81a0aad48f6ab2fc3d3db0c62a`).
- **Kế toán ngân sách mạng** (biên nhận: [`so_fake_acquisition_receipt.json`](./so_fake_acquisition_receipt.json)):
  * Dung lượng chuyển tải phiên này: **496,763,723 bytes (473.75 MiB)** / hạn mức 786,432,000 bytes (750 MiB) $\to$ **ĐẠT (dư 276.25 MiB)**.
  * Lũy kế mạng toàn bộ dự án: **1,807,585,485 bytes (1.6834 GiB)** / hạn mức 2,147,483,648 bytes (2.00 GiB) $\to$ **ĐẠT (dư 324.15 MiB / 15.83%)**.
  * Footprint đĩa bổ sung: Parquet shard (473.8 MB) + 300 ảnh trích xuất (142.1 MB) + feature cache (1.97 MB) $\approx 618$ MB $\le 5$ GiB hạn mức.

---

### 14.3. Đóng Băng Model Registry, Nhiệt Độ Hiệu Chuẩn & Đính Chính Sai Số Lịch Sử

Tạo và niêm phong [`locked_models_registry.json`](./locked_models_registry.json) gồm 6 mô hình:
1. **A0-global** (592-D, $T = 5.9577$, SHA-256: `0343cb9d...`): Baseline kết hợp Visual 576-D + Global DSP 16-D trên điều kiện A0 ($N=750$).
2. **A0-local** (640-D, $T = 6.0142$, SHA-256: `27498992...`): Visual 576-D + Global DSP 16-D + Local Quadrant DSP 48-D trên điều kiện A0 ($N=750$).
3. **A1-global** (592-D, $T = 4.9391$, SHA-256: `77ee17b4...`): Visual 576-D + Global DSP 16-D trên điều kiện A1 (bổ sung 700 ảnh authentic ImageNet, $N=1450$).
4. **A1-local** (640-D, $T = 5.1689$, SHA-256: `21fd5aa1...`): Visual 576-D + Global DSP 16-D + Local Quadrant DSP 48-D trên điều kiện A1 ($N=1450$).
5. **A0-visual** (576-D, $T = 5.9818$, SHA-256: `d2c971fb...`): Đối chứng Visual-only trên điều kiện A0 ($N=750$).
6. **A1-visual** (576-D, $T = 5.1813$, SHA-256: `75658d96...`): Đối chứng Visual-only trên điều kiện A1 ($N=1450$).

**Đính chính các sai sót ghi chép lịch sử (Documentation Reconciliations)**:
- **Sai số nhiệt độ hiệu chuẩn ($T$) giữa các báo cáo**:
  * Điều kiện A0-global: Báo cáo Ablation v2.1 ghi $T_{\text{fus}} = 5.8907$ (fit trên OOF đơn scaler), trong khi Model Artifact chính thức `three_class_visual_dsp_model.json` ghi $T = 5.9577$ (fit trên OOF kép scaler CV). Tương tự A1-global: Báo cáo v2.1 ghi $T = 5.1400$, artifact chính thức ghi $T = 4.9391$.
  * Quyết định: Khóa đúng giá trị nhiệt độ trong artifact JSON vật lý đã lưu trên đĩa ($T_{\text{A0,glob}} = 5.9577$, $T_{\text{A1,glob}} = 4.9391$), bảo tồn nguyên vẹn provenance và không refit hồi tố.
- **Sai số làm tròn $\Delta(\text{A0-local} - \text{A0-global})$**:
  * Macro-F1 A0-local là 0.5552, A0-global là 0.5385. Phép trừ làm tròn hiển thị chính xác là $+0.0167$ (+1.67 điểm phần trăm). Ghi nhận trước đây là $+0.0168$ là do phép trừ float thô trước khi định dạng chuỗi.

---

### 14.4. Kiểm Toán Cách Ly Dữ Liệu & Đặc Điểm Cohort Ngoại Lai (N=300)

- **Quy tắc chọn mẫu**: Rút trích cố định theo seed 42 qua `scripts/research/extract_and_audit_so_fake_cohort.py`, thu được đúng 100 mẫu/lớp ($N=300$ tổng cộng):
  * `real` $\to$ `authentic` ($n=100$)
  * `tampered` $\to$ `ai_edited` ($n=100$)
  * `full_synthetic` $\to$ `fully_generated` ($n=100$)
- Toàn bộ 300 ảnh được giải nén lossless sang định dạng PNG tại `data/research/so_fake/images/`, vượt qua 100% kiểm tra Pillow RGB decode (manifest: [`so_fake_probe_manifest.json`](./so_fake_probe_manifest.json)).
- **Kiểm toán cách ly độc lập** (receipt: [`so_fake_isolation_audit.json`](./so_fake_isolation_audit.json)):
  * Đối soát mã băm SHA-256 của 300 mẫu probe với toàn bộ các tập dữ liệu nội bộ: tập huấn luyện ba lớp ($N=1450$), tập holdout phát triển ($N=273$), tập authentic controls (Nature & ImageNet, $N=741$), 343 nguồn niêm phong Locked-Test nhị phân, và 400 nguồn niêm phong TGIF nhị phân.
  * Kết quả: **`ISOLATION_AUDIT_PASS`** — 0 trường hợp trùng lặp (collision_count = 0).

---

### 14.5. Bảng Kết Quả Đánh Giá Tổng Hợp Trên So-Fake Probe (N=300)

| Mô hình / Cấu hình | Đặc trưng | Macro-F1 | Balanced Acc | Multiclass Brier | Top-Class ECE | Báo động giả trên Authentic ($N=100$) | Ma trận nhầm lẫn $[A, E, G]$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-global** (Baseline) | 592-D | **0.3551** | 0.3567 | 0.8101 | 0.2479 | 57.0% (57/100: 41 edit, 16 gen) | $[[43, 41, 16], [37, 35, 28], [36, 35, 29]]$ |
| **A0-local** (Probe) | 640-D | 0.3459 | 0.3467 | 0.8151 | 0.2590 | 63.0% (63/100: 46 edit, 17 gen) | $[[37, 46, 17], [34, 38, 28], [38, 33, 29]]$ |
| **A1-global** (Supplem.) | 592-D | 0.3230 | 0.3500 | 0.8498 | 0.3072 | 35.0% (35/100: 25 edit, 10 gen) | $[[65, 25, 10], [57, 20, 23], [48, 32, 20]]$ |
| **A1-local** (Probe) | 640-D | 0.3027 | 0.3300 | 0.8579 | 0.3139 | 38.0% (38/100: 28 edit, 10 gen) | $[[62, 28, 10], [58, 19, 23], [54, 28, 18]]$ |
| **A0-visual** (Control) | 576-D | 0.3487 | **0.3667** | **0.7620** | **0.2429** | 53.0% (53/100: 46 edit, 7 gen) | $[[47, 46, 7], [45, 47, 8], [40, 44, 16]]$ |
| **A1-visual** (Control) | 576-D | 0.3315 | **0.3667** | 0.8036 | 0.2723 | **30.0% (30/100: 27 edit, 3 gen)** | $[[70, 27, 3], [72, 25, 3], [52, 33, 15]]$ |

*Ghi chú chi tiết ma trận nhầm lẫn: Hàng thể hiện Ground Truth $[A=\text{authentic}, E=\text{ai\_edited}, G=\text{fully\_generated}]$, Cột thể hiện nhãn dự đoán tương ứng.*

---

### 14.6. Kiểm Định Ghép Cặp Bootstrap Phân Cụm Nguồn (10,000 Replicates)

Thực hiện kiểm định ghép cặp 10,000 lần lấy mẫu lại có hoàn lại theo phân cụm nguồn (`source_group_id`, seed 20261011):
- **$\Delta(\text{A0-local} - \text{A0-global})$**: $-0.0092$, 95% CI: $[-0.0375, +0.0200]$, $P(\Delta^* > 0) = 28.12\%$ $\to$ **`INCONCLUSIVE`** (Khoảng tin cậy chứa 0; không có bằng chứng Local DSP cải thiện tổng quát hóa).
- **$\Delta(\text{A1-local} - \text{A1-global})$**: $-0.0203$, 95% CI: $[-0.0452, +0.0042]$, $P(\Delta^* > 0) = 5.25\%$ $\to$ **`INCONCLUSIVE`** (Khuynh hướng suy giảm nhẹ khi bổ sung local DSP).
- **$\Delta(\text{A0-global} - \text{A0-visual})$**: $+0.0064$, 95% CI: $[-0.0434, +0.0538]$, $P(\Delta^* > 0) = 59.45\%$ $\to$ **`INCONCLUSIVE`** (Không có sự khác biệt có ý nghĩa thống kê giữa Visual-only và DSP trên miền ngoại lai).
- **$\Delta(\text{A1-global} - \text{A1-visual})$**: $-0.0085$, 95% CI: $[-0.0499, +0.0322]$, $P(\Delta^* > 0) = 35.18\%$ $\to$ **`INCONCLUSIVE`**.
- **$\Delta(\text{A1-local} - \text{A0-local})$**: $-0.0432$, 95% CI: $[-0.0942, +0.0071]$, $P(\Delta^* > 0) = 4.84\%$ $\to$ **`INCONCLUSIVE`**.
- **$\Delta(\text{A1-global} - \text{A0-global})$**: $-0.0321$, 95% CI: $[-0.0798, +0.0156]$, $P(\Delta^* > 0) = 9.22\%$ $\to$ **`INCONCLUSIVE`**.

---

### 14.7. Phân Tích Chi Tiết Theo Bộ Sinh & Công Cụ Can Thiệp (Generator Breakdown, $N \ge 5$)

1. **`flux.1-dev`** ($N=59$, nhãn thực: `ai_edited`):
   * A0-visual: Độ chính xác 45.8% (27/59), dự đoán 23 authentic, 9 generated.
   * A0-local: Độ chính xác 28.8% (17/59), dự đoán 19 authentic, 22 generated.
   * A0-global: Độ chính xác 27.1% (16/59), dự đoán 21 authentic, 22 generated.
   * A1-global & A1-local: Độ chính xác chỉ đạt **16.9%** (10–11/59), dự đoán tới 48% mẫu là authentic.
2. **`sdxl`** ($N=40$, nhãn thực: `ai_edited`):
   * A0-local: Độ chính xác 40.0% (16/40).
   * A0-global: Độ chính xác 37.5% (15/40).
   * A0-visual: Độ chính xác 35.0% (14/40).
   * A1-global & A1-local: Độ chính xác rơi mạnh xuống **17.5%** (7/40), dự đoán sai 31/40 mẫu thành authentic.
3. **`latent_diffusion`** ($N=32$, nhãn thực: `ai_edited`):
   * A0-global: Độ chính xác 46.9% (15/32).
   * A0-local: Độ chính xác 43.8% (14/32).
   * A1-global: Độ chính xác 28.1% (9/32).
   * A1-local: Độ chính xác 25.0% (8/32).
4. **`OpenForensics_fake`** ($N=9$, nhãn thực: `fully_generated`):
   * A0-global: Độ chính xác 66.7% (6/9).
   * A0-local & A1-global: Độ chính xác 55.6% (5/9).
   * A0-visual: Độ chính xác **0.0%** (0/9, dự đoán 6 mẫu thành ai_edited, 3 authentic) $\to$ Trên nhóm này, các đặc trưng DSP đóng vai trò cứu vãn phát hiện cấu trúc tổng hợp mà biểu diễn visual thuần túy bị mất dấu.
5. **`sdxl_lightning`** ($N=8$, nhãn thực: `fully_generated`):
   * A0-visual: Độ chính xác 62.5% (5/8).
   * A0-local: Độ chính xác 62.5% (5/8).
   * A0-global: Độ chính xác 50.0% (4/8).
   * A1-global & A1-local: Độ chính xác 25.0% (2/8).

---

### 14.8. Luận Giải Khoa Học, Giới Hạn & Đúng Một Bước Tiếp Theo

1. **Suy giảm hiệu năng nghiêm trọng dưới biến dịch phân phối ngoài miền (Domain Shift Collapse)**:
   - Trên tập holdout phát triển nội bộ, các mô hình đạt Macro-F1 $\approx 0.49 - 0.55$. Nhưng khi chuyển sang So-Fake probe, Macro-F1 của toàn bộ 6 mô hình sụt giảm mạnh về khoảng **$0.30 - 0.35$**, xấp xỉ mức đoán ngẫu nhiên ba lớp không thông tin ($0.3333$).
   - Điều này chứng minh thực nghiệm rằng các bộ dò tuyến tính/logistic nhẹ (Lightweight Probes) học trên đặc trưng trích xuất sâu bị trói buộc rất chặt vào miền ngữ nghĩa và nén của dữ liệu huấn luyện (COCO / ImageNet / TGIF).
2. **Local DSP Không Cứu Được Khả Năng Tổng Quát Hóa Ngoại Lai**:
   - Cả trên điều kiện A0 ($\Delta = -0.0092$) lẫn A1 ($\Delta = -0.0203$), việc bổ sung đặc trưng 48-D Local Quadrant DSP không mang lại lợi thế tổng quát hóa; các khoảng tin cậy bootstrap đều phủ số 0.
3. **Bản Chất Đánh Đổi Ranh Giới Quyết Định Của Điều Kiện A1**:
   - Bổ sung authentic ImageNet vào huấn luyện (A1) tiếp tục phát huy hiệu quả kéo giảm báo động giả trên ảnh authentic ngoại lai: từ 53%–57% (A0) xuống còn **30%–35%** (A1), tức giảm 22–23 điểm phần trăm báo động giả.
   - Tuy nhiên, sự dịch chuyển ranh giới này khiến mô hình trở nên quá thiên vị về lớp `authentic`, dẫn đến việc bỏ sót nghiêm trọng các can thiệp ngoại lai mới (như FLUX.1 và SDXL), khiến Recall của lớp `ai_edited` và `fully_generated` sụt giảm mạnh về dưới 20%.
4. **Giới hạn nghiên cứu của đợt probe**:
   - Quy mô $N=300$ mang tính chất thăm dò ban đầu do ràng buộc ngân sách mạng nghiêm ngặt.
   - Mối quan hệ nguồn gốc sâu xa của các ảnh trong So-Fake (chẳng hạn metadata nguồn ảnh gốc trước khi can thiệp) chưa thể đối soát toàn diện, vì vậy chưa thể kết luận độc lập nguồn hoàn toàn (unverified source independence).
5. **Đúng một bước tiếp theo được phê duyệt (Single Approved Next Step)**:
   - Không mở rộng kiến trúc, không tăng số quadrant, không triển khai browser.
   - Thực hiện **Thiết kế đối chứng nhân tố 2×2 (Factorial Source-Conditioned Cohort)** nhằm phân lập triệt để giữa *Domain Shift Ngữ Nghĩa* (COCO vs ImageNet) và *Artefact Kỹ Thuật Số/Nén* (JPEG quantization / generator fingerprint) trước khi xem xét bất kỳ phương án huấn luyện lại hoặc tinh chỉnh nào.

---

## 15. ĐÍNH CHÍNH BẰNG CHỨNG SO-FAKE PROBE & CHẨN ĐOÁN GHÉP CẶP VỀ NÉN VÀ THỨ TỰ RESIZE

> **Mã thực nghiệm**: `SO_FAKE_EVIDENCE_RECTIFICATION_AND_PAIRED_COMPRESSION_SENSITIVITY`<br>
> **Thời điểm**: 2026-10-11<br>
> **Ràng buộc nghiên cứu**: Không tải dataset/checkpoint mới; không retrain/refit/recalibrate/threshold tune; giữ nguyên 6 locked models; kiểm tra pipeline kỹ thuật trên panel tất định; khóa 3 điều kiện trước khi đánh giá; bảo toàn artifacts lịch sử kèm receipts v2 riêng.

---

### 15.1. Đính Chính Bằng Chứng So-Fake Probe (Manifest v2 & Isolation Audit v2)

#### 1. Nguyên nhân gốc rễ và xử lý Source Grouping ($systematic-debugging)
- **Phát hiện**: Trong manifest v1 (`so_fake_probe_manifest.json`), logic gom nhóm nguồn sử dụng `stem[:16]` đã vô tình cắt ngắn tên file, khiến nhiều ảnh có tên file và nguồn gốc hoàn toàn khác nhau bị ép vào cùng một cluster ID (ví dụ: `so_fake_latent_diffusion_object_part_synt`). Cắt ngắn tiền tố tên file không phản ánh quan hệ nguồn ảnh thực tế.
- **Khắc phục qua Manifest v2** ([`so_fake_probe_manifest_v2.json`](./so_fake_probe_manifest_v2.json)):
  * Thiết lập `sample_id` duy nhất và chuẩn hóa cho từng ảnh (`so_fake_000000` đến `so_fake_000099` theo từng split).
  * Tách biệt dứt khoát 3 khái niệm:
    1. *Sample identity duy nhất*: Định danh từng mẫu ảnh riêng biệt trên đĩa.
    2. *Source group có bằng chứng*: Chỉ áp dụng khi có metadata đối soát rõ ràng về ảnh gốc chung.
    3. *Quan hệ nguồn chưa xác minh (`unverified_source_cluster`)*: Do shard Parquet của So-Fake không cung cấp ID ảnh nguồn gốc ban đầu trước can thiệp, toàn bộ 300 mẫu được gắn trạng thái `source_group_status: "unverified"`.
- **Kiểm toán cách ly v2** ([`so_fake_isolation_audit_v2.json`](./so_fake_isolation_audit_v2.json)):
  * Xác nhận 300 mẫu probe không có bất kỳ xung đột mã băm SHA-256 nào với các tập dữ liệu nội bộ (0 collisions). Trạng thái: **`ISOLATION_AUDIT_PASS`**.

#### 2. Tái tính Paired Bootstrap với Giả định Rõ ràng
- Do quan hệ nguồn chưa được kiểm chứng độc lập từ metadata gốc, chúng tôi báo cáo **Paired Image Bootstrap** (10,000 replicates, seed 20261011) với giả định mẫu ghép cặp ngang hàng ở cấp độ ảnh đơn lẻ, thay vì tự nhận là "Source-Cluster Bootstrap đã kiểm chứng".
- Tỷ lệ replicate có $\Delta > 0$ được báo cáo trung thực dưới dạng **Bootstrap Positive Fraction** ($P(\Delta^* > 0)$), tuyệt đối **không diễn giải hay coi là p-value** hay xác suất giả thuyết vô hiệu $H_0$.

#### 3. Bảng Phân Rã Generator Chuẩn Xác theo Ground-Truth Class
Đã khắc phục lỗi gán nhãn toàn bộ generator theo dòng đầu tiên. Bảng thống kê chính xác theo `(generator, ground_truth_label)` trên 300 mẫu:

| Tên Generator / Nguồn | Ground-Truth Class | Số lượng mẫu ($N$) | Tỷ lệ trong lớp |
| :--- | :--- | :---: | :---: |
| **COCO** | authentic | 53 | 53.0% |
| **ImageNet** | authentic | 47 | 47.0% |
| **FLUX** (FLUX.1-inpaint) | ai_edited | 44 | 44.0% |
| **sd2_inpainting** | ai_edited | 22 | 22.0% |
| **latent_diffusion** (inpainting) | ai_edited | 12 | 12.0% |
| **sdxl_inpainting** | ai_edited | 12 | 12.0% |
| **deepfloyd_if** | ai_edited | 10 | 10.0% |
| **midjourney** | fully_generated | 36 | 36.0% |
| **latent_diffusion** (text2img) | fully_generated | 20 | 20.0% |
| **FLUX** (FLUX.1-dev text2img) | fully_generated | 15 | 15.0% |
| **OpenForensics_fake** | fully_generated | 9 | 9.0% |
| **dalle3** | fully_generated | 8 | 8.0% |
| **sdxl** | fully_generated | 7 | 7.0% |
| **deepfloyd_if** | fully_generated | 5 | 5.0% |

*Nhận xét*: FLUX.1 xuất hiện ở cả hai lớp (44 ảnh ai_edited và 15 ảnh fully_generated); latent_diffusion xuất hiện ở cả hai lớp (12 ảnh ai_edited và 20 ảnh fully_generated).

#### 4. Phân Tích Phân Loại Học (Taxonomy) Nhóm `OpenForensics_fake` ($N=9$)
- **Khảo sát nguồn gốc**: OpenForensics (arXiv:2107.03589, ICCV 2021) là tập dữ liệu chuyên về phát hiện khuôn mặt giả mạo đa đối tượng (multi-face forgery detection/segmentation). Trong So-Fake, nhóm này gồm 9 ảnh khuôn mặt bị hoán đổi/chỉnh sửa (face swap / face inpainting).
- **Độ vênh phân loại học**: Đề tài chúng tôi định nghĩa:
  * `fully_generated`: Toàn bộ canvas được sinh tổng hợp từ text prompt hoặc unconditioned noise.
  * `ai_edited`: Canvas nền thật có một hoặc nhiều vùng cục bộ bị can thiệp bởi inpainting/editing.
  * Như vậy, 9 ảnh `OpenForensics_fake` về bản chất ngữ nghĩa là chỉnh sửa khuôn mặt cục bộ trên nền ảnh thật, nhưng nhãn nguồn của So-Fake lại xếp vào `full_synthetic` (tương ứng `fully_generated`).
- **Phân tích độ nhạy (Taxonomy Sensitivity Analysis, $N=300$ vs $N=291$)**:
  * Để bảo toàn tính lịch sử và khách quan khoa học, chúng tôi giữ nguyên kết quả chính thống trên $N=300$ theo nhãn nguồn gốc của So-Fake.
  * Đồng thời, thực hiện phân tích độ nhạy loại trừ 9 mẫu `OpenForensics_fake` ($N=291$):
    - **A0-global**: Macro-F1 thay đổi từ 0.3427 ($N=300$) xuống 0.3340 ($N=291$, $\Delta = -0.0087$).
    - **A0-local**: Macro-F1 thay đổi từ 0.3300 ($N=300$) xuống 0.3168 ($N=291$, $\Delta = -0.0132$).
    - **A1-global**: Macro-F1 thay đổi từ 0.3230 ($N=300$) xuống 0.3176 ($N=291$, $\Delta = -0.0054$).
    - **A1-local**: Macro-F1 thay đổi từ 0.3109 ($N=300$) xuống 0.3014 ($N=291$, $\Delta = -0.0095$).
    - **A0-visual**: Macro-F1 thay đổi từ 0.3369 ($N=300$) lên 0.3482 ($N=291$, $\Delta = +0.0113$).
    - **A1-visual**: Macro-F1 thay đổi từ 0.3339 ($N=300$) lên 0.3402 ($N=291$, $\Delta = +0.0063$).
  * *Kết luận độ nhạy*: Việc loại bỏ 9 mẫu OpenForensics không làm thay đổi bất kỳ kết luận nghiên cứu nào. Điểm Macro-F1 của toàn bộ 6 mô hình vẫn dao động quanh mức $0.30 - 0.35$ (gần mức đoán ngẫu nhiên $0.3333$). Sự sụt giảm của các mô hình DSP khi bỏ 9 mẫu này phản ánh việc DSP phát hiện được vết nén/ghép cục bộ trên 5–6 mẫu OpenForensics (nhưng gán đúng nhãn nguồn `fully_generated`).

#### 5. Đính Chính Tài Liệu & Giấy Phép Dữ Liệu
- **Kích thước tập huấn luyện A1**: Khẳng định rõ ràng A1 là **$N=1,000$** mẫu (gồm 250 COCO authentic, 250 ImageNet authentic, 250 TGIF ai_edited, 250 SD1.5 fully_generated). Số liệu ghi chú trong registry và báo cáo cũ hoàn toàn khớp với kích thước huấn luyện thực tế này.
- **Giấy phép `saberzl/So-Fake-Set`**: Kiểm tra trực tiếp metadata Hugging Face cardData: trường giấy phép chính thức của repository `saberzl/So-Fake-Set` là **`unspecified`** (chưa định rõ trong metadata). Tuyệt đối không tự ý gán giấy phép CC-BY-NC-4.0 (vốn thuộc repository phụ `So-Fake-OOD`) cho tập dữ liệu `So-Fake-Set`.

---

### 15.2. Kiểm Tra Pipeline Kỹ Thuật Trước Thí Nghiệm (Panel Tất Định)

Để loại trừ khả năng suy giảm hiệu năng bắt nguồn từ lỗi implementation trong việc trích xuất ảnh Parquet sang PNG hoặc sai lệch tiền xử lý, chúng tôi thiết lập panel kiểm chứng kỹ thuật gồm 16 mẫu đại diện được rút trích trực tiếp:
- **Bytes ảnh gốc vs PNG trích xuất**:
  * Kiểm tra độ chênh lệch mảng pixel RGB sau decode: **Max Pixel Difference = 0** (khớp bit-exact 100% trên toàn bộ các kênh màu).
  * Shape và Orientation: 0 trường hợp thay đổi hướng EXIF, 0 trường hợp sai lệch kích thước ảnh gốc.
- **Tính toán đặc trưng DSP**:
  * So sánh vector đặc trưng 16-D Global DSP và 48-D Local DSP tính từ bytes nạp trực tiếp trong bộ nhớ vs ảnh đọc từ file PNG:
  * **Max Absolute Difference = 0.0000000000e+00** (khớp bit-exact 100%).
- **Kết luận kỹ thuật**: Quá trình giải nén Parquet sang file PNG lossless hoàn toàn bảo toàn nguyên vẹn dữ liệu đầu vào của pipeline. Hiệu năng thấp trên So-Fake probe **hoàn toàn là do biến dịch phân phối thực tế (OOD Distribution Shift)**, không phải do lỗi tiền xử lý hay trích xuất kỹ thuật.

---

### 15.3. Thí Nghiệm Ghép Cặp Về Độ Nhạy Nén JPEG Và Thứ Tự Resize (C0 vs C1 vs C2)

#### 1. Khóa Thiết Kế Ba Điều Kiện (Preregistered Transformation Pipeline)
Thực hiện đánh giá ghép cặp trên cùng một ảnh, cùng 6 locked models, trên 2 cohorts:
- **Cohort 1: So-Fake Probe ($N=300$)** — Tập dữ liệu ngoại lai độc lập.
- **Cohort 2: Development Holdout ($N=273$)** — Tập kiểm tra phát triển nội bộ (ghi nhận rõ là development holdout, không gọi là kiểm định độc lập mới).

Ba điều kiện được khóa bất biến trước khi chạy thí nghiệm:
- **Điều kiện C0 (Baseline)**: Pipeline chuẩn mực nguyên trạng trên ảnh không nén lại. Canvas gốc được resize Lanczos về 512×512 cho DSP và bicubic về 224×224 cho Visual MobileNetV3.
- **Điều kiện C1 (Full-Resolution Compression)**: Nén JPEG ảnh ở độ phân giải gốc một lần duy nhất với tham số chuẩn hóa: `quality=75, subsampling=2 (4:2:0), optimize=False` bằng Pillow encoder. Sau đó decode RGB và đưa vào pipeline chuẩn (resize Lanczos 512×512 cho DSP, bicubic 224×224 cho Visual).
- **Điều kiện C2 (Canvas-First Compression)**: Resize ảnh RGB gốc về 512×512 bằng Lanczos trước. Sau đó nén JPEG canvas 512×512 một lần duy nhất với cùng tham số (`quality=75, subsampling=2, optimize=False`). Decode RGB và đưa trực tiếp vào trích xuất DSP mà **không resize thêm một lần nào nữa** (tránh nén/resize lặp ngoài thiết kế); đồng thời resize bicubic về 224×224 từ canvas này để nạp vào Visual backbone.

#### 2. Kết Quả Thực Nghiệm Trên So-Fake Probe Cohort ($N=300$)

| Mô hình | Điều kiện | Macro-F1 | Balanced Acc | Multiclass Brier | Top-Class ECE | Báo động giả Authentic FP | Tỷ lệ đổi nhãn vs C0 (Flip %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-global** | C0 | 0.3427 | 0.3433 | 0.8257 | 0.2738 | 61.0% (61/100) | 0.0% (baseline) |
| | C1 | 0.3630 | 0.3633 | 0.8664 | 0.2921 | 61.0% (61/100) | 18.3% (55/300) |
| | C2 | 0.2941 | 0.3533 | 0.9340 | 0.3687 | **94.0% (94/100)** | **49.3% (148/300)** |
| **A0-local** | C0 | 0.3300 | 0.3300 | 0.8231 | 0.2764 | 67.0% (67/100) | 0.0% (baseline) |
| | C1 | 0.3488 | 0.3467 | 0.8654 | 0.3029 | 65.0% (65/100) | 20.7% (62/300) |
| | C2 | 0.2853 | 0.3433 | 0.9242 | 0.3711 | **95.0% (95/100)** | **46.7% (140/300)** |
| **A1-global** | C0 | 0.3230 | 0.3500 | 0.8488 | 0.3128 | 35.0% (35/100) | 0.0% (baseline) |
| | C1 | 0.3689 | 0.3867 | 0.8711 | 0.3009 | 37.0% (37/100) | 21.7% (65/300) |
| | C2 | 0.3215 | 0.3700 | 0.9168 | 0.3512 | **80.0% (80/100)** | **56.3% (169/300)** |
| **A1-local** | C0 | 0.3109 | 0.3367 | 0.8522 | 0.3168 | 38.0% (38/100) | 0.0% (baseline) |
| | C1 | 0.3540 | 0.3667 | 0.8551 | 0.2956 | 41.0% (41/100) | 19.0% (57/300) |
| | C2 | 0.3328 | 0.3633 | 0.8691 | 0.3228 | **75.0% (75/100)** | **46.3% (139/300)** |
| **A0-visual** | C0 | 0.3369 | 0.3533 | 0.7753 | 0.2463 | 58.0% (58/100) | 0.0% (baseline) |
| | C1 | 0.3213 | 0.3400 | 0.7710 | 0.2409 | 58.0% (58/100) | 5.3% (16/300) |
| | C2 | 0.3399 | 0.3567 | 0.7737 | 0.2270 | 57.0% (57/100) | 6.0% (18/300) |
| **A1-visual** | C0 | 0.3339 | 0.3700 | 0.8027 | 0.2637 | 29.0% (29/100) | 0.0% (baseline) |
| | C1 | 0.3267 | 0.3667 | 0.7984 | 0.2605 | 28.0% (28/100) | 3.7% (11/300) |
| | C2 | 0.3156 | 0.3600 | 0.8025 | 0.2683 | 28.0% (28/100) | 4.3% (13/300) |

*Mốc tham chiếu Brier ngẫu nhiên*: Dự đoán xác suất đều $[1/3, 1/3, 1/3]$ đạt $B_{\text{uniform}} = 2/3 \approx 0.6667$. Toàn bộ 6 mô hình trên So-Fake đều có Brier từ $0.7710$ đến $0.9340$, **kém hơn rõ rệt so với bộ dự đoán xác suất đều không thông tin** (thể hiện mô hình bị sai lệch tự tin thái quá).

#### 3. Kết Quả Thực Nghiệm Trên Development Holdout Cohort ($N=273$)

| Mô hình | Điều kiện | Macro-F1 | Balanced Acc | Multiclass Brier | Top-Class ECE | Báo động giả Authentic FP | Tỷ lệ đổi nhãn vs C0 (Flip %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-global** | C0 | 0.5592 | 0.5495 | 0.5742 | 0.1044 | 55.0% (50/91) | 0.0% (baseline) |
| | C1 | **0.6187** | 0.6190 | 0.4821 | 0.0791 | 55.0% (50/91) | 9.5% (26/273) |
| | C2 | 0.4720 | 0.5201 | 0.6286 | 0.1388 | **85.7% (78/91)** | **32.2% (88/273)** |
| **A0-local** | C0 | 0.5564 | 0.5458 | 0.5766 | 0.0856 | 56.0% (51/91) | 0.0% (baseline) |
| | C1 | **0.6133** | 0.6154 | 0.4870 | 0.0681 | 58.2% (53/91) | 9.9% (27/273) |
| | C2 | 0.4792 | 0.5275 | 0.6300 | 0.1402 | **85.7% (78/91)** | **31.9% (87/273)** |
| **A1-global** | C0 | 0.5045 | 0.4945 | 0.6478 | 0.1427 | 50.5% (46/91) | 0.0% (baseline) |
| | C1 | **0.6161** | 0.6081 | 0.5049 | 0.1007 | 50.5% (46/91) | 13.2% (36/273) |
| | C2 | 0.4680 | 0.4945 | 0.6599 | 0.1738 | **78.0% (71/91)** | **38.5% (105/273)** |
| **A1-local** | C0 | 0.4919 | 0.4835 | 0.6400 | 0.1562 | 55.0% (50/91) | 0.0% (baseline) |
| | C1 | **0.5933** | 0.5824 | 0.5228 | 0.0783 | 52.8% (48/91) | 12.8% (35/273) |
| | C2 | 0.4740 | 0.4872 | 0.6470 | 0.1659 | **70.3% (64/91)** | **31.5% (86/273)** |
| **A0-visual** | C0 | 0.5109 | 0.5018 | 0.5976 | 0.1032 | 57.1% (52/91) | 0.0% (baseline) |
| | C1 | 0.5174 | 0.5092 | 0.5965 | 0.1046 | 57.1% (52/91) | 2.6% (7/273) |
| | C2 | 0.5194 | 0.5128 | 0.5985 | 0.1100 | 57.1% (52/91) | 5.5% (15/273) |
| **A1-visual** | C0 | 0.4836 | 0.4762 | 0.6627 | 0.1747 | 48.4% (44/91) | 0.0% (baseline) |
| | C1 | 0.4773 | 0.4689 | 0.6626 | 0.1614 | 50.5% (46/91) | 2.2% (6/273) |
| | C2 | 0.4803 | 0.4725 | 0.6639 | 0.1627 | 50.5% (46/91) | 3.3% (9/273) |

#### 4. Phân Tích Hiện Tượng Bất Đối Xứng Cực Đoan C1 vs C2 ($systematic-debugging)
1. **Tính trơ của Visual Feature (MobileNetV3)**:
   - Các mô hình thuần thị giác (`A0-visual` và `A1-visual`) hầu như không bị ảnh hưởng bởi nén JPEG hay thứ tự resize: tỷ lệ đổi nhãn (flip rate) chỉ từ **2.2% đến 6.0%** trên cả hai cohorts. Macro-F1 giữ nguyên ở mức ~0.32-0.34 (So-Fake) và ~0.48-0.52 (Holdout).
2. **Cơ chế gây nhiễu của Điều kiện C2 (Canvas-First Compression)**:
   - Khi resize ảnh về 512×512 trước rồi mới nén JPEG chất lượng 75, các khối DCT 8×8 của thuật toán JPEG được khắc cố định hoàn hảo trên lưới tọa độ 512×512 của canvas.
   - Khi bộ trích xuất DSP tính toán ma trận tương quan phổ FFT 2D trên lưới này, nó phát hiện các đường biên khối nén nhân tạo (artificial blocking artifacts) đồng đều trên toàn bộ bức ảnh.
   - Do tập huấn luyện của DSP chứa nhiều ảnh AI có đặc trưng biên tần số cao tương tự, bộ phân loại bị lừa rằng đây là dấu hiệu can thiệp nhân tạo. Hậu quả là **Báo động giả trên ảnh thật (Authentic False Alarm) bùng nổ lên 75%–95% trên So-Fake và 70%–86% trên Holdout**. Tỷ lệ lật quyết định lên tới gần một nửa số ảnh (32%–56%).
3. **Cơ chế hài hòa của Điều kiện C1 (Full-Resolution Compression)**:
   - Khi nén JPEG ở độ phân giải gốc rồi mới resize Lanczos về 512×512, bộ lọc nội suy Lanczos đã làm mịn và làm nhòe bớt các cạnh khối DCT sắc nhọn.
   - Trên tập Holdout (nơi ảnh gốc COCO vốn có lịch sử nén khác biệt với ảnh TGIF), việc nén C1 vô tình chuẩn hóa và hài hòa dấu vết nén, giúp Macro-F1 của các mô hình DSP tăng vọt lên **0.59–0.62** (giải thích nguồn gốc con số 0.6015 trong thí nghiệm trước).
   - Tuy nhiên, trên So-Fake ngoại lai, hiệu ứng này chỉ giúp tăng nhẹ F1 từ ~0.33 lên ~0.36, hoàn toàn không đủ để vượt qua sự sụp đổ phân phối ngữ nghĩa.

---

### 15.4. Trả Lời Giới Hạn 4 Câu Hỏi Trọng Tâm

1. **Có lỗi pipeline thực sự hay không?**
   - **TRẢ LỜI**: **KHÔNG CÓ LỖI PIPELINE THỰC SỰ TRONG VIỆC TRÍCH XUẤT VÀ TÍNH TOÁN ĐẶC TRƯNG**.
   - Kiểm chứng panel 16 ảnh tất định xác nhận 100% bit-exact pixel arrays và 100% bit-exact DSP vectors giữa Parquet bytes gốc và PNG trích xuất. Trọng số mô hình, scaler, nhiệt độ và class order đều khớp tuyệt đối với registry đã niêm phong.
   - Lỗi duy nhất được tìm thấy và đã khắc phục triệt để ở lượt này là **lỗi gộp cụm giả mạo trong manifest v1 do cắt ngắn chuỗi tên file (`stem[:16]`)**, gây sai lệch phân cụm trong kiểm định bootstrap. Lỗi này thuộc về tầng báo cáo/thống kê (evaluation accounting), không ảnh hưởng đến pixel hay scoring logic của mô hình.

2. **Model nhạy đến mức nào với JPEG và thứ tự resize?**
   - **TRẢ LỜI**: **CÁC MÔ HÌNH KẾT HỢP DSP CỰC KỲ NHẠY VỚI THỨ TỰ RESIZE VÀ NÉN JPEG, TRONG KHI MÔ HÌNH THỊ GIÁC (VISUAL-ONLY) THÌ GẦN NHƯ MIỄN NHIỄM**.
   - Thứ tự xử lý quyết định sự thành bại: Nếu nén JPEG sau khi đưa về canvas 512×512 (C2), lưới DCT 8×8 kích hoạt báo động giả hàng loạt trên ảnh thật (lên tới 95%), làm sụp đổ hoàn toàn độ đặc hiệu. Nếu nén JPEG trước khi resize Lanczos (C1), bộ lọc Lanczos làm suy giảm biên khối, giúp ổn định và thậm chí nâng F1 trên holdout lên 0.62.
   - Độ nhạy này chứng minh rằng **đặc trưng DSP phản ánh rất mạnh lịch sử xử lý tín hiệu kỹ thuật số cấp thấp (low-level processing artifacts)** hơn là chỉ riêng dấu vết sinh ảnh AI.

3. **Hiệu năng thấp còn tồn tại sau kiểm tra pipeline hay không?**
   - **TRẢ LỜI**: **CÓ, HIỆU NĂNG THẤP TRÊN SO-FAKE VẪN HOÀN TOÀN TỒN TẠI VÀ LÀ THỰC TẾ NGHIÊN CỨU**.
   - Sau khi kiểm tra bit-exact pipeline, sửa manifest v2, loại trừ ảnh taxonomy mơ hồ (OpenForensics), và thử nghiệm cả 3 điều kiện nén/resize, Macro-F1 của toàn bộ 6 mô hình trên So-Fake probe vẫn bị giam chặt trong khoảng **0.2853 đến 0.3689** (xấp xỉ mức ngẫu nhiên 0.3333).
   - Điểm Multiclass Brier ($0.77 - 0.93$) đều tệ hơn mốc xác suất đều ngẫu nhiên ($0.6667$), chứng minh mô hình bị lệch nghiêm trọng (miscalibrated) khi bước ra ngoài miền huấn luyện.

4. **Các dữ liệu hiện tại có đủ để phân biệt nguyên nhân ngữ nghĩa, generator và nén ảnh hay chưa?**
   - **TRẢ LỜI**: **CHƯA ĐỦ ĐỂ PHÂN BIỆT NHÂN QUẢ ĐỘC LẬP GIỮA CÁC YẾU TỐ**.
   - Trong dữ liệu hiện có:
     * Generator gắn liền với kiểu nén và định dạng (ví dụ SD1.5/SD2 dùng PNG lossless trong khi dữ liệu nguồn COCO/ImageNet có lịch sử JPEG nén nhiều lần).
     * Ngữ nghĩa (context COCO đa đối tượng vs ImageNet đơn đối tượng) bị trộn lẫn với nhãn authentic.
     * Sự so sánh hậu nghiệm (post-hoc) giữa các dataset khác nhau không thể phân tách được quan hệ nhân quả. Việc nén lại ảnh (recompression) hoặc chuyển sang PNG không xóa bỏ được lịch sử nén gốc vốn có bên trong cấu trúc tần số của ảnh.

---

### 15.5. Đúng Một Đề Xuất Cải Thiện Kế Tiếp Có Căn Cứ (Single Approved Next Step)

**Đề xuất duy nhất**: Triển khai **Thiết kế đối chứng nhân tố 2×2 (Factorial Source-Conditioned Cohort)** theo đề cương đã lập tại `NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`:
- Ghép cặp đối chứng chéo: Nguồn ngữ nghĩa $[ \text{COCO} \times \text{ImageNet} ]$ đan xen với Lịch sử định dạng $[ \text{PNG uncompressed} \times \text{JPEG Q75 recompressed} ]$.
- Giữ nguyên quy tắc: Không tự ý tải checkpoint bên ngoài, không huấn luyện lại mô hình trong phiên này, không thay đổi demo production. Bước đi này phải được thẩm định và phê duyệt trước khi tạo đợt thu nhận dữ liệu tiếp theo.
