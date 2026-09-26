# Giao thức Thử nghiệm Ít Dữ liệu & Đường cong Học tập (Small-Data Protocol)

> **Tài liệu**: `docs/SMALL_DATA_PROTOCOL.md`  
> **Phiên bản**: `1.0.0` (Phase 4B.1)  
> **Mục tiêu**: Chuẩn hóa phương pháp luận ít dữ liệu (*sample-efficient learning*), quy tắc phân tách đơn vị độc lập, đường cong học tập và tiêu chí khoa học nghiệm thu trước khi huấn luyện mô hình.

---

## 1. Nguyên tắc Cốt lõi: Đơn vị Thống kê Độc lập (Independent Statistical Unit)

1. **Định nghĩa đơn vị độc lập**:
   * **Đơn vị thống kê độc lập duy nhất** của tập dữ liệu là **`source_id`** (đại diện cho một ảnh chụp thực tế từ MS-COCO val2017).
   * File mặt nạ (`mask`), biến thể phân giải (512, 1024), loại mask (`bbox`, `segm`), hay các bản chỉnh sửa khác nhau của cùng một ảnh nguồn **tuyệt đối không được coi là các mẫu độc lập**.
2. **Quy tắc công bố quy mô dữ liệu**:
   Mọi báo cáo kết quả và bảng số liệu khoa học bắt buộc phải đồng thời công bố 4 chỉ số:
   * `file_count`: Tổng số file ảnh vật lý.
   * `variant_count`: Số biến thể sinh ra từ cùng ảnh nguồn.
   * `unique_source_id_count`: Số lượng ảnh nguồn thực tế duy nhất.
   * `paired_source_id_count`: Số lượng ảnh nguồn có đủ cặp đối ứng (`orig ↔ sd2-sp ↔ mask`).
3. **Bảo toàn phân vùng nguồn (Group-Isolation Invariant)**:
   Toàn bộ biến thể (authentic, edit variations, masks) của cùng một `source_id` bắt buộc phải nằm trọn vẹn trong **cùng một phân vùng** (`train`, `val`, hoặc `test`). Tuyệt đối cấm để các biến thể của cùng một ảnh xuất hiện ở hai phân vùng khác nhau.

---

## 2. Thiết kế Đường cong Học tập (Learning Curve Protocol)

### 2.1. Các mức quy mô dữ liệu đăng ký trước (Pre-registered Sample Sizes)

Hệ thống đăng ký 5 mức quy mô ảnh nguồn:
$$N \in \{50, 100, 250, 500, 1000\} \quad \text{unique } source\_id$$

* **Cơ chế fail-closed**: Nếu mức $N$ vượt quá số lượng `eligible_source_id` thực tế trong tập huấn luyện, mức đó tự động chuyển trạng thái `not-runnable` và không được thực thi.
* **Cấu hình Option P Đóng băng (Phase 4B.2 Frozen State)**:
  * Official Validation Pool: 341 unique `source_id`.
  * Official Testing Pool: 343 unique `source_id`.
  * **`development_train`**: 250 unique `source_id` (hạt giống cố định seed 42).
  * **`inner_validation`**: 91 unique `source_id` (dành cho early stopping, calibration, threshold).
  * **`locked_test`**: 343 unique `source_id` (chỉ evaluator cuối cùng đọc khi cấu hình đã đóng băng hoàn toàn).
  * **Mã băm niêm phong (Split-Lock SHA-256)**: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9` (tại `research/evidence/phase-4b.2/split-lock.json`).
  * **Cross-Split Isolation**: Giao giữa các split theo `source_id` bằng đúng 0 (`overlap = 0`).
  * **Nested Learning Curve Invariant**: $N=50 \subset N=100 \subset N=250$ (các tập con lồng nhau nghiêm ngặt theo chỉ mục source_id).
  * **Mức hợp lệ trong Phase 4C**: $N = 50, 100, 250$.
  * **Mức không hợp lệ**: $N = 500, 1000$ chuyển trạng thái `not-runnable-insufficient-independent-sources` (do `development_train` có 250 sources).
  * **Class-Coverage Guard**: Phase 4B.2 / Phase 4C hiện chỉ hỗ trợ 2 lớp (`authentic`, `ai_edited`). Chế độ 3 lớp bị chặn với mã `not-runnable-missing-fully-generated-data`.

### 2.2. Kiểm soát Tính ngẫu nhiên & Cố định Tập Đánh giá

1. **Fixed Validation & Test Benchmarks**:
   * Mọi mức $N$ đều được đánh giá trên cùng một tập Test cố định (343 `source_id`) và tập Validation cố định (341 `source_id` hoặc tập con validation cố định).
   * Tập Test tuyệt đối không tham gia vào bất kỳ khâu chọn checkpoint, dừng sớm (early stopping) hay điều chỉnh siêu tham số nào.
2. **Lặp hạt giống (Multi-Seed Protocol)**:
   * Giai đoạn thăm dò (exploratory): Chạy tối thiểu **3 seeds** ngẫu nhiên độc lập ($S \in \{42, 1337, 2026\}$) cho mỗi mức $N$.
   * Giai đoạn công bố kết quả khóa luận: Chạy tối thiểu **5 seeds** ($S \in \{42, 100, 1337, 2024, 2026\}$).
   * Báo cáo mean $\pm$ standard deviation trên toàn bộ hạt giống.
3. **Cân bằng biến thể theo vòng lặp (Variant Epoch Cycling)**:
   * Không đưa đồng thời 6 biến thể của cùng một `source_id` vào một epoch như 6 mẫu độc lập.
   * Tại mỗi epoch huấn luyện, hệ thống luân phiên chọn ngẫu nhiên 1 biến thể duy nhất cho mỗi `source_id` để duy trì tỷ lệ 1:1 giữa authentic và edited.

---

## 3. Kiến trúc Huấn luyện Hai Giai đoạn (Two-Stage Transfer Learning)

Mô hình mục tiêu: **MobileNetV3-Small** (định hướng triển khai Zero-Egress trên trình duyệt web).

### Giai đoạn 1 — Frozen Transfer Learning
* **Backbone**: MobileNetV3-Small pretrained trên ImageNet-1k; **đóng băng 100% trọng số backbone**.
* **Classification Head**: Global Average Pooling $\rightarrow$ Linear(1024, 256) $\rightarrow$ Hardswish $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(256, 2) cho Pilot A.
* **Localization Head**: Patch-based sliding window hoặc 1x1 conv map chiếu ground-truth mask.
* **Tối ưu hóa**: AdamW, learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$, Cosine Annealing, Early Stopping (patience 5 epochs trên Validation Macro-F1).
* **Lưu vết**: Config hash, seed, model checkpoint.

### Giai đoạn 2 — Limited Fine-Tuning
* Chỉ được kích hoạt khi Giai đoạn 1 hoàn thành và vượt qua kiểm toán chống shortcut.
* Chỉ mở khóa (unfreeze) block tích chập cuối cùng (`features[-1]`).
* Learning rate giảm xuống $5 \times 10^{-5}$.

---

## 4. Hệ thống Sáu Baseline Bắt buộc (Mandatory Baselines)

Để chứng minh mô hình nơ-ron thực sự học đặc trưng pháp chứng, bắt buộc phải so sánh với 6 baseline đối chứng:

1. **Stratified Dummy Classifier**: Phân loại ngẫu nhiên theo phân bố tiên nghiệm của nhãn trong tập huấn luyện.
2. **Metadata-Only Classifier**: Logistic Regression dựa trên siêu dữ liệu EXIF/C2PA (phần mềm, ISO, kích thước, camera make/model).
3. **DSP-Only Classifier**: Random Forest / Logistic Regression dựa trên đặc trưng DSP (2D FFT radial energy distribution, DCT block variance, ELA noise residuals).
4. **Frozen Visual Backbone + Linear Head**: MobileNetV3 đóng băng toàn bộ backbone (Stage 1).
5. **Fine-Tuned Visual Model**: MobileNetV3 mở khóa block cuối (Stage 2).
6. **Multimodal Fusion Classifier**: Bộ hiệu chuẩn hợp nhất (FusionCalibrator) kết hợp Visual Logits + DSP Score + Metadata Score.

---

## 5. Định nghĩa Kết luận Pháp chứng & Cơ chế Quyết định (Decision Semantics)

### 5.1. Bốn trạng thái đầu ra của hệ thống

| Trạng thái kỹ thuật | Định nghĩa ngữ nghĩa chuẩn tắc | Ràng buộc phát ngôn |
| :--- | :--- | :--- |
| `no_ai_evidence` | **Không phát hiện đủ bằng chứng AI trong phạm vi mô hình và dữ liệu đánh giá** | Tuyệt đối **không gọi là ảnh thật 100%**; không tuyên bố tính nguyên bản tuyệt đối |
| `fully_generated` | Dấu vết tạo sinh AI toàn phần từ bộ sinh văn bản sang ảnh | Chỉ áp dụng khi có bằng chứng tạo sinh từ đầu đến cuối |
| `ai_edited` | Dấu vết chỉnh sửa, thay thế nội dung cục bộ bằng inpainting / splicing | Áp dụng cho các vùng can thiệp cục bộ (TGIF SD2-sp) |
| `uncertain` | Không chắc chắn; mức tin cậy thấp, tín hiệu mâu thuẫn, hoặc **chưa cài đặt mô hình** | Trạng thái từ chối quyết định (*selective abstention*) |

### 5.2. Tiêu chí Từ chối Quyết định (Selective Abstention Criteria)

Hệ thống bắt buộc chuyển sang nhãn `uncertain` khi gặp một trong các điều kiện:
1. Xác suất dự đoán sau hiệu chuẩn Temperature Scaling nằm dưới ngưỡng tin cậy ($P(\hat{y}) < \tau_{\text{abstain}}$, mặc định $\tau = 0.65$).
2. Xung đột đa phương thức: Visual model dự đoán `ai_edited` nhưng DSP residual và Metadata C2PA chỉ thị ảnh nguyên bản an toàn với độ lệch vượt ngưỡng dung sai.
3. Mẫu ngoại lai ngoài phân phối (OOD): Khoảng cách biểu diễn embedding vượt ngưỡng $3\sigma$ của tập huấn luyện.
4. Trạng thái không có mô hình (No-Model State): Trả về `uncertain` với thông báo trung thực `Model not installed`.

---

## 6. Tiêu chí Khoa học Nghiệm thu (Scientific Acceptance Criteria)

### 6.1. Chỉ số Phân loại (Classification)
* Macro-F1 (chỉ số chính, tránh bias do mất cân bằng lớp).
* Balanced Accuracy.
* Precision và Recall cho từng lớp.
* AUROC (Area Under Receiver Operating Characteristic).
* Ma trận nhầm lẫn (Confusion Matrix).

### 6.2. Chỉ số Hiệu chuẩn Xác suất (Calibration)
* ECE (Expected Calibration Error, 10 bins).
* Brier Score.
* NLL (Negative Log-Likelihood).
* Tỷ lệ từ chối quyết định (Abstention Rate) và Coverage-Risk curve.

### 6.3. Chỉ số Định vị Vùng Chỉnh sửa (Localization)
* Dice Coefficient.
* mIoU (mean Intersection over Union).
* Pixel-level AUROC trên ground-truth masks.

### 6.4. Tiêu chí Xác nhận Tuyên bố "Hiệu quả Ít Dữ liệu" (Few-Data Claim Acceptance)
Một mô hình chỉ được công nhận là "học hiệu quả với ít dữ liệu" khi thỏa mãn đồng thời:
1. Visual model đạt Macro-F1 vượt trội có ý nghĩa thống kê so với Dummy Baseline ($p < 0.01$).
2. Cận dưới khoảng tin cậy 95% của chênh lệch Macro-F1 qua Paired Stratified Bootstrap lớn hơn 0 ($\text{CI}_{95\%}[\Delta\text{Macro-F1}] > 0$).
3. Visual model vượt trội Metadata-only baseline qua Paired Bootstrap.
4. Kết quả ổn định qua các seeds huấn luyện ($\sigma_{\text{Macro-F1}} < 0.03$).
5. Đường cong học tập thể hiện xu hướng tăng trưởng đơn điệu theo kích thước $N$.
6. Tập Test hoàn toàn độc lập và không bị rò rỉ bất kỳ `source_id` nào từ tập huấn luyện.

---

## 7. Phân tách Ba Nhiệm vụ Thử nghiệm (Pilot Experiments Separation)

* **Pilot A (Inpainting Forensics)**:
  * Phân loại 2 lớp: `authentic` vs `ai_edited`.
  * Dữ liệu: TGIF `orig` + `sd2-sp` + `masks`.
  * Nhiệm vụ: Phân loại nhị phân và định vị vùng can thiệp.
* **Pilot B (Full Generation Forensics)**:
  * Phân loại 2 lớp: `authentic` vs `fully_generated`.
  * Dữ liệu: Nguồn độc lập có ảnh tạo hoàn toàn (GenImage hoặc nguồn thay thế đã kiểm duyệt).
  * Nhiệm vụ: Phân loại ảnh tạo sinh toàn phần.
* **Pilot C (Three-Class Exploratory Forensics)**:
  * Phân loại 3 lớp: `authentic` vs `fully_generated` vs `ai_edited`.
  * Chỉ được kích hoạt sau khi cả Pilot A và Pilot B vượt qua kiểm toán shortcut độc lập.
  * Tuyệt đối không sử dụng TGIF `fr` làm đại diện cho `fully_generated`.
  * **Trạng thái thực thi hiện tại (Phase 4B.2 / 4C)**: `not-runnable-missing-fully-generated-data`. Cơ chế Class-Coverage Guard tự động từ chối chạy chế độ 3 lớp cho đến khi có tập dữ liệu `fully_generated` hợp lệ được nạp vào Research Track.

