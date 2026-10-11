# Protocol Amendment v2.1.0: Authentic Source Diversification Ablation (A0 vs A1)

> **Protocol ID**: `PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT`<br>
> **Date**: 2026-10-11<br>
> **Status**: `LOCKED_AUTHENTIC_IMAGENET_ABLATION_AMENDMENT`<br>
> **Supercedes**: Extends `PROTOCOL_AMENDMENT_V2.0.0.md` without modifying historical A0 artifacts.

---

## 1. Motivation & Scientific Rationale

Trong thực nghiệm Pilot Ba Lớp (RQ1–RQ2) và phân tích chẩn đoán, mô hình ghi nhận tỷ lệ báo động giả AI cao (~67%–68%) trên tập đối chứng 341 ảnh ImageNet authentic (`nature_control_pilot`).
Hai giả thuyết được đặt ra:
1. **Giả thuyết Shortcut Nguồn (Source / Semantic Domain Hypothesis)**: Mô hình học đặc trưng bối cảnh cụ thể của MS-COCO val2017 (sinh hoạt, nội thất, đô thị) cho lớp `authentic`, nên khi gặp ảnh thuộc miền phân phối khác (ImageNet vật thể, thiên nhiên) thì dự đoán lệch sang `ai_edited` hoặc `fully_generated`.
2. **Giả thuyết Lịch sử Nén (Compression History Hypothesis)**: Dấu vết nén JPEG ban đầu từ cảm biến của ảnh ImageNet tạo ra tín hiệu tần số bị bộ phân loại nhầm lẫn với dấu vết inpainting/generation.

Để kiểm tra thực nghiệm giả thuyết (1) mà không thay đổi kiến trúc hoặc hyperparameters, Protocol Amendment v2.1.0 thiết lập một cuộc thử nghiệm ablation bổ sung dữ liệu có kiểm soát: bổ sung thêm **250 ảnh ImageNet authentic** vào tập huấn luyện và áp dụng cơ chế cân bằng trọng số lớp (`sample_weight`), đánh giá tác động trực tiếp lên tỷ lệ báo động giả trên tập đối chứng và các đánh đổi trên tập holdout.

---

## 2. Thiết Kế Hai Điều Kiện Dữ Liệu (A0 vs A1)

### 2.1. Điều kiện A0 (Pilot Baseline hiện có, N=750)
- Tập huấn luyện:
  * 250 ảnh `authentic` (MS-COCO val2017)
  * 250 ảnh `ai_edited` (TGIF SD2 inpainting)
  * 250 ảnh `fully_generated` (Stable Diffusion v1.5)
- Tổng mẫu: 750 ảnh. Trọng số mỗi mẫu: $1.0$.
- Mô hình: Tái sử dụng các mô hình A0 reconciled đã được audit ở commit `0f1b22d` (`three_class_visual_only_model.json`, `three_class_visual_dsp_model.json` với nhiệt độ $T_v = 5.9818, T_f = 5.9577$).

### 2.2. Điều kiện A1 (Authentic Source Supplemented, N=1,000)
- Tập huấn luyện:
  * 250 ảnh `authentic` (MS-COCO val2017) — **weight = 0.5** (tổng trọng số = 125.0)
  * 250 ảnh `authentic` (ImageNet authentic, mới trích xuất) — **weight = 0.5** (tổng trọng số = 125.0)
  * 250 ảnh `ai_edited` (TGIF SD2 inpainting) — **weight = 1.0** (tổng trọng số = 250.0)
  * 250 ảnh `fully_generated` (Stable Diffusion v1.5) — **weight = 1.0** (tổng trọng số = 250.0)
- **Tổng trọng số hiệu dụng mỗi lớp trong A1**:
  * Lớp 0 (`authentic`): $125.0 + 125.0 = 250.0$
  * Lớp 1 (`ai_edited`): $250.0$
  * Lớp 2 (`fully_generated`): $250.0$
  $\to$ Tỷ lệ lớp hiệu dụng được bảo toàn tuyệt đối **1:1:1** (cân bằng 250:250:250).

---

## 3. Quy Tắc Cách Ly Dữ Liệu & Ranh Giới Bất Biến

1. **Không Tải Dữ Liệu Mới**: Trích xuất đúng 250 ảnh ImageNet authentic từ 3 tệp parquet đã lưu trữ cục bộ (`validation-00000`, `00001`, `00002`). Tổng dung lượng lưu trữ trên đĩa duy trì $\le 5\text{ GiB}$.
2. **Cách Ly Nghiêm Ngặt Tập Đối Chứng**: 341 ảnh ImageNet authentic control hiện có trong `source_control_cohort_manifest.json` **tuyệt đối không được sử dụng** vào tập huấn luyện A1.
3. **Cách Ly Tập Holdout & Sealed Sets**: 273 ảnh `development_holdout`, 343 nguồn Locked-Test và 400 nguồn TGIF N=400 không có bất kỳ chồng lấn nào về source ID hoặc SHA-256 byte hash.
4. **Không Thay Đổi Kiến Trúc & Hyperparameters**:
   - Backbone: MobileNetV3-small đóng băng weights ImageNet `047dcff4...`.
   - Tiền xử lý: Direct resize canvas 512×512 (Lanczos) cho 16-D canonical DSP; bicubic resize [3, 224, 224] chuẩn hóa ImageNet cho Visual branch.
   - Bộ phân loại: Multinomial logistic regression (`C=1.0`, solver `'lbfgs'`).
   - Grouped inner CV: `StratifiedGroupKFold(n_splits=5)` gom nhóm theo `source_group_id`.
   - Scaler isolation: Mỗi inner fold fit `StandardScaler` độc lập trên inner training partition.
   - Temperature Scaling: Hiệu chuẩn nhiệt độ chỉ dùng true out-of-fold logits.

---

## 4. Bốn Cấu Hình Đánh Giá Đối Sánh

| Mã Cấu Hình | Điều Kiện Dữ Liệu | Không Gian Đặc Trưng | Kích Thước Vector |
| :--- | :--- | :--- | :--- |
| **A0-Visual** | A0 (Pilot N=750) | Visual-only (MobileNetV3-small) | 576-D |
| **A0-Fusion** | A0 (Pilot N=750) | Visual + Canonical 16-D DSP | 592-D (Concatenation) |
| **A1-Visual** | A1 (Supplemented N=1,000) | Visual-only (MobileNetV3-small) | 576-D |
| **A1-Fusion** | A1 (Supplemented N=1,000) | Visual + Canonical 16-D DSP | 592-D (Concatenation) |

---

## 5. Tiêu Chí Đánh Giá & Các Endpoint Thống Kê

1. **Trên Development Holdout ($N=273$)**:
   - Macro-F1, Balanced Accuracy.
   - Per-class Precision, Recall, F1 (`authentic`, `ai_edited`, `fully_generated`).
   - Multiclass Brier Score, Multiclass Top-Class ECE.
   - Confusion Matrix 3×3.
   - Độ dịch chuyển xác suất theo cặp nguồn trên 91 cặp COCO holdout: $\Delta P(\text{ai\_edited}) = P(\text{ai\_edited} \mid \text{edited}) - P(\text{ai\_edited} \mid \text{auth})$.
2. **Trên ImageNet Authentic Control Cohort ($N=341$)**:
   - False-Positive AI Rate: $\frac{\text{FP}_{\text{edited}} + \text{FP}_{\text{generated}}}{341}$.
   - Tách biệt số lượng nhầm lẫn thành `ai_edited` vs `fully_generated`.
3. **Phân Tích Độ Nhạy Nén (Robustness Sensitivity)**:
   - Đánh giá trên `original` và `jpeg_q75` (cùng phép biến đổi đồng nhất).
4. **Phân Tích Thống Kê**:
   - Paired source-cluster bootstrap (10,000 replicates, seed `20261011`) cho chênh lệch $A1 - A0$ và $\text{Fusion} - \text{Visual}$.
   - Lưu ý giới hạn: Đây là thực nghiệm phát triển tiếp nối có kiểm soát, không phải kiểm định độc lập mới. Không coi các ngưỡng hiệu năng là tiêu chuẩn tiền đăng ký.
