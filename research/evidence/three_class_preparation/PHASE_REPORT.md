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
