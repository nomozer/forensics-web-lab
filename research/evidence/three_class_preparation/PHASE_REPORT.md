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

| Chỉ số Đánh Giá | Baseline 1: Visual-Only (576-D) | Baseline 2: Visual + DSP (592-D) | Chênh Lệch ($\Delta = \text{B2} - \text{B1}$) |
| :--- | :---: | :---: | :---: |
| **Macro-F1** | **0.5086** | **0.5385** | **+0.0299** (+2.99%) |
| **Balanced Accuracy** | 0.4982 | 0.5238 | +0.0256 |
| **Multiclass Brier Score** | 0.5841 | 0.5618 | -0.0223 (cải thiện) |
| **Multiclass Top-Class ECE** | 0.0466 | 0.0780 | +0.0314 |
| - F1 `authentic` ($n=91$) | 0.4108 | 0.4409 | +0.0301 |
| - F1 `ai_edited` ($n=91$) | 0.4615 | 0.4390 | -0.0225 |
| - F1 `fully_generated` ($n=91$) | 0.6536 | 0.7355 | +0.0819 |

### Ma trận Nhầm lẫn (Confusion Matrix 3×3)

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
- Điểm ước lượng $\Delta \text{Macro-F1}$: **+0.0299**
- Khoảng tin cậy 95% Percentile CI: **[-0.0068, +0.0686]**
- Tỷ lệ replicate có cải thiện $P(\Delta^* > 0)$: **94.32%**
- **Kết luận khoa học chính thức**: **`INCONCLUSIVE`**. Việc bổ sung 16-D canonical DSP cho thấy xu hướng cải thiện dương (+2.99%, với 94.32% bootstrap mass > 0), nhưng khoảng tin cậy 95% CI cắt 0 ở cận dưới (-0.0068 $\le 0$). Do đó, chưa đủ bằng chứng để khẳng định sự cải thiện có ý nghĩa thống kê ở mức ý nghĩa $\alpha = 0.05$.

---

## 6. Đánh giá Chẩn đoán Nguồn (Source Control Cohort, $N=341$)

Đánh giá riêng biệt trên 341 ảnh ImageNet authentic (`nature`), hoàn toàn không tham gia huấn luyện hay chọn ngưỡng:

| Mô hình | Số ảnh Đúng (Authentic) | Nhầm `ai_edited` | Nhầm `fully_generated` | Tổng Nhầm AI | **False Positive AI Rate** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Visual-Only** | 111 / 341 | 106 | 124 | 230 | **67.45%** |
| **Visual + DSP** | 109 / 341 | 112 | 120 | 232 | **68.04%** |

### Ý nghĩa Khoa học & Vạch trần Shortcut Phân phối Nguồn
1. **Source bias nghiêm trọng**: Khi chuyển từ miền dữ liệu huấn luyện (MS-COCO) sang miền dữ liệu khác (ImageNet nature), tỷ lệ báo động giả AI lên tới **~67–68%**.
2. **Domain Shortcut**: Mô hình phân loại 3 lớp siêu nhẹ dựa trên backbone đông lạnh bị phụ thuộc nặng nề vào các đặc trưng phân phối bối cảnh của dataset gốc (COCO vs ImageNet) hơn là nhận diện thuần túy dấu vết tạo sinh/chỉnh sửa của AI.
3. **Bất đối xứng nén ảnh**: Ảnh COCO/TGIF được đánh giá ở định dạng PNG không nén, trong khi ảnh ImageNet mang dấu vết nén JPEG ban đầu của máy ảnh số. Dấu vết nén này tạo ra shortcut tần số khiến bộ phân loại nhầm lẫn với AI inpainting/generation.
4. **Kết luận an toàn ứng dụng**: Tuyệt đối **không đưa mô hình 3 lớp này lên web demo cho người dùng tải ảnh tự do**; việc giữ nguyên bản demo nhị phân FP32 với 4 banner ranh giới nghiên cứu là quyết định kiến trúc chính xác.

---

## 7. Hiện vật Xuất bản & Minh chứng
- [`PROTOCOL_AMENDMENT_V2.0.0.md`](./PROTOCOL_AMENDMENT_V2.0.0.md)
- [`three_class_pilot_manifest.csv`](./three_class_pilot_manifest.csv) (1,023 records)
- [`three_class_pilot_splits.json`](./three_class_pilot_splits.json) (Splits lock & audit)
- [`three_class_visual_only_model.json`](./three_class_visual_only_model.json)
- [`three_class_visual_dsp_model.json`](./three_class_visual_dsp_model.json)
- [`three_class_pilot_predictions.json`](./three_class_pilot_predictions.json) (Chi tiết 273 holdout + 341 nature predictions)
- [`three_class_pilot_evaluation_receipt.json`](./three_class_pilot_evaluation_receipt.json) (Metrics & bootstrap)
- [`source_control_evaluation_receipt.json`](./source_control_evaluation_receipt.json) (Diagnostic findings)
- [`ml/tests/test_three_class_pilot.py`](../../../ml/tests/test_three_class_pilot.py) (5/5 PASS)
