# Research Plan & Scientific Protocol: Forensics Web Lab

> **Phiên bản**: Phase 4A.2 (Scope Freeze)  
> **Trạng thái tài liệu**: Đã đóng băng phạm vi khoa học (*Research Scope Frozen*)  
> **Nguyên tắc cốt lõi**: Trung thực khoa học (*Scientific Honesty*), tách biệt tuyệt đối Research Track và Product Track (ADR-0006), mọi chỉ số chưa đo ghi nhận `not evaluated`.

---

## 1. Tiêu đề và Mục tiêu Nghiên cứu

* **Tên đề tài**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Tên sản phẩm minh chứng**: `Forensics Web Lab`
* **Mục tiêu tổng quát**:
  * Phát triển mô hình mạng nơ-ron tích chập nhẹ phân loại ảnh 3 lớp (`authentic`, `fully_generated`, `ai_edited`).
  * Xây dựng cơ chế hiệu chuẩn xác suất và từ chối dự đoán khi thiếu độ tin cậy (trạng thái `uncertain`).
  * Đánh giá độ bền vững (robustness) trước các suy giảm phổ biến trên môi trường mạng.
  * Hiện thực hóa pipeline suy luận trực tiếp trên trình duyệt web người dùng (In-Browser Inference qua ONNX Runtime Web WASM/WebGPU, Zero Server Egress).
  * Khảo sát năng lực định vị vùng chỉnh sửa (localization) như một mục tiêu phụ khi có ground-truth mask.

---

## 2. Định nghĩa Lớp Dữ liệu và Trạng thái Phân loại

### 2.1. Ba Lớp Huấn luyện Mục tiêu (Three-Class Ground-Truth)
Hệ thống học máy được huấn luyện trên không gian nhãn 3 lớp duy nhất:
1. `authentic` (Ảnh thông thường / Chụp từ máy ảnh thật): Ảnh nguyên bản từ máy ảnh hoặc ảnh qua chỉnh sửa truyền thống (Photoshop không dùng generative AI).
2. `fully_generated` (Ảnh tạo hoàn toàn bởi AI): Ảnh được sinh 100% từ mô hình tạo sinh (GAN, Diffusion, Flow-Matching).
3. `ai_edited` (Ảnh có vùng chỉnh sửa bằng AI): Ảnh gốc có một hoặc nhiều vùng nội dung bị thay thế, vẽ bù (inpainting), xóa hoặc thêm đối tượng bằng AI.

### 2.2. Trạng thái Quyết định Hậu Kiểm chuẩn: `uncertain`
* `uncertain` **không phải là lớp dữ liệu thứ tư** trong ground-truth hay quá trình huấn luyện mô hình.
* `uncertain` là **trạng thái quyết định sau calibration và selective abstention**:
  * Khi xác suất dự đoán sau hiệu chuẩn (calibrated confidence) nằm dưới ngưỡng tin cậy tối thiểu ($P < \tau_{\text{abstain}}$).
  * Hoặc khi có sự mâu thuẫn đối kháng gay gắt giữa tín hiệu thị giác của mô hình deep learning và các tín hiệu pháp chứng tín hiệu số DSP (FFT/DCT/Noise/JPEG) hoặc siêu dữ liệu C2PA/EXIF.
* Việc gán `uncertain` giúp hệ thống tránh dự đoán sai nghiêm trọng (catastrophic false certainty) trên dữ liệu bị suy giảm nặng hoặc chưa rõ ràng.

---

## 3. Phân định Mục tiêu Chính và Mục tiêu Phụ

### 3.1. Mục tiêu Khoa học Chính (Primary Scientific Objectives)
1. **Phân loại 3 lớp bằng mô hình nhẹ**: Đánh giá năng lực của kiến trúc gọn nhẹ ($< 15\text{M}$ tham số) trong việc phân biệt `authentic`, `fully_generated`, `ai_edited` trên tập kiểm thử độc lập, không rò rỉ nhóm ảnh gốc (`source_id`).
2. **Hiệu chuẩn độ tin cậy (Confidence Calibration)**: Ứng dụng Temperature Scaling để đảm bảo xác suất dự báo phản ánh sát xác suất đúng thực tế (tối thiểu hóa Expected Calibration Error - ECE).
3. **Đánh giá Robustness thực tế**: Kiểm tra độ bền vững của mô hình trước nén JPEG, thay đổi kích thước (resizing), làm mờ (blur), ảnh chụp màn hình (screenshot) và ảnh định dạng WebP.
4. **Triển khai In-Browser Runtime CPU/WASM**: Thực hiện chuyển đổi sang ONNX, lượng tử hóa INT8 và kiểm chứng hiệu năng suy luận client-side an toàn, không gửi ảnh về máy chủ.

### 3.2. Mục tiêu Khoa học Phụ (Secondary Exploratory Objectives)
1. **Định vị vùng AI chỉnh sửa khi có ground-truth mask**: Khi dataset có mask nhị phân chính thức (như TGIF/TGIF2), đánh giá khả năng trích xuất bản đồ nhiệt (heatmap) từ các sliding window patches thông qua chỉ số mIoU, Dice Score và Pixel AUROC.
2. **Khảo sát bản đồ nhiệt suy diễn (Inference Heatmap)**:
   > **Lưu ý khoa học**: Bản đồ nhiệt sinh ra từ patch scores của mô hình phân loại được phân loại là **khám phá sơ bộ (*exploratory heuristics*)**. Chỉ khi nào được kiểm chứng thực nghiệm bằng chỉ số mIoU/Dice so với ground-truth mask trên dữ liệu thật, năng lực định vị mới được công nhận là tính năng khoa học đã kiểm chứng.

---

## 4. Câu hỏi Nghiên cứu (Research Questions - RQs)

Hệ thống nghiên cứu tập trung giải quyết 4 câu hỏi trọng tâm và 1 câu hỏi phụ:

* **RQ1 (Three-Class Generalization on Unseen Data)**:  
  *Mô hình tích chập gọn nhẹ có phân biệt được ba lớp `authentic`, `fully_generated` và `ai_edited` trên dữ liệu chưa thấy (unseen generators / held-out sources) với độ chính xác vượt trội baseline ngẫu nhiên hay không?*

* **RQ2 (Multi-Modal Evidence Fusion vs Visual-Only Baseline)**:  
  *Việc kết hợp kết quả mô hình thị giác (visual model) với siêu dữ liệu xuất xứ (provenance) và tín hiệu pháp chứng số (DSP frequency/noise/JPEG) có cải thiện Macro-F1 hoặc độ hiệu chuẩn xác suất (ECE) so với mô hình thị giác đơn lẻ hay không?*

* **RQ3 (ONNX Quantization Trade-Offs)**:  
  *Quá trình xuất sang ONNX và lượng tử hóa INT8 (post-training quantization) ảnh hưởng như thế nào đến chất lượng phân loại (Macro-F1 drop), dung lượng lưu trữ (model size bytes) và thời gian suy luận (latency)?*

* **RQ4 (Browser CPU/WASM Execution Feasibility)**:  
  *Mô hình tối ưu hóa có thể vận hành ổn định trong môi trường trình duyệt web thông qua ONNX Runtime Web WASM (đa luồng SIMD) với độ trễ suy luận và mức chiếm dụng bộ nhớ (peak memory) đáp ứng trải nghiệm tương tác của người dùng hay không?*

* **Auxiliary RQ5 (Patch-Based Localization Feasibility)**:  
  *Khi có ground-truth mask nhị phân, phương pháp nội suy patch scores từ backbone phân loại có đạt được mức độ định vị chấp nhận được (mIoU, Pixel AUROC) so với mask thật mà không cần thêm mô hình phân đoạn độc lập hay không?*

---

## 5. Giả thuyết Khoa học (Testable Hypotheses)

Các giả thuyết được thiết lập theo nguyên tắc có thể bác bỏ (*falsifiable*), các ngưỡng cụ thể chưa có cơ sở đo lường được ghi nhận `TBD before confirmatory experiment`:

* **Hypothesis 1 ($H_1$)**: Trên tập kiểm thử held-out cách ly nhóm ảnh gốc (`source_id`), mô hình gọn nhẹ đạt chỉ số Macro-F1 và Balanced Accuracy vượt trội có ý nghĩa thống kê so với stratified dummy baseline.
* **Hypothesis 2 ($H_2$)**: Cơ chế kết hợp bằng chứng (Evidence Fusion) kết hợp Temperature Scaling làm giảm ECE và giảm tỷ lệ dự đoán sai tự tin (overconfident errors) so với mô hình thị giác thuần túy (visual-only).
* **Hypothesis 3 ($H_3$)**: Mô hình ONNX INT8 duy trì chất lượng dự đoán trong biên độ không thua kém định trước ($\Delta \text{Macro-F1} \le \epsilon_{\text{margin}}$, với $\epsilon_{\text{margin}}$ được định rõ trước thí nghiệm xác nhận), đồng thời giảm dung lượng ít nhất $60\%$ so với bản FP32.
* **Hypothesis 4 ($H_4$)**: Trên môi trường trình duyệt chuẩn (Chrome/Firefox trên máy tính thông dụng), runtime WASM SIMD đạt thời gian suy luận trung vị (median latency) và đỉnh RAM trong ngân sách cho phép tương tác (ngân sách trần $\text{Latency} \le T_{\text{max}}$ và $\text{RAM} \le M_{\text{max}}$ được chốt trước benchmark).

---

## 6. Ma trận Vai trò Dataset trong Nghiên cứu

| Dataset | Vai trò nghiên cứu | Nhãn đóng góp | Rủi ro chính | Biện pháp kiểm soát | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Smoke** | Fixture kiểm thử kỹ thuật pipeline | authentic, fully_generated, ai_edited (hình học giả lập) | Dữ liệu vẽ bằng code, không có giá trị học máy | Đặt track `fixture-only`, cấm huấn luyện model sản phẩm | `verified` |
| **GenImage** | Huấn luyện & benchmark ảnh tạo hoàn toàn | `fully_generated` (8 generator), `authentic` (ImageNet val) | Chỉ có ảnh fully generated, không có mask inpainting; archive lớn | Phân luồng Research Track; First fully inventoried archive (BigGAN ~24 GB) | `verified` |
| **TGIF / TGIF2** | Huấn luyện & benchmark ảnh chỉnh sửa | `ai_edited` (spliced `sp`), `authentic` (MS-COCO pairs), masks (`fr` quarantined) | Dung lượng lớn (65.4 GB - 110 GB); điều khoản CC BY-SA 4.0; `fr` cấm gán `fully_generated` | Nextcloud cho phép tải độc lập thư mục nhỏ; cách ly trong Research Track; `sp` dùng cho Pilot A | `verified` |
| **RAID** | Đánh giá độ bền vững đối kháng (Adversarial Robustness) | Đa dạng generator và độ suy giảm nặng | Bản quyền dataset card chưa chứng minh quyền phân phối ảnh | Trạng thái `blocked`, chỉ xem xét làm candidate kiểm định sau | `blocked` |
| **RealHD** | Ứng viên tương lai cho inpainting đa dạng | AI-edited đa dạng | Tác giả chưa phát hành archive và license công khai | Trạng thái `blocked` | `blocked` |

---

## 7. Rủi ro Shortcut Nguồn Dữ liệu và Giao thức Thí nghiệm Pilot (docs/PILOT_PROTOCOL.md)

> **Cảnh báo khoa học**: Nếu mỗi lớp nhãn được lấy từ một dataset hoàn toàn tách biệt (ví dụ: `authentic` lấy từ ImageNet, `ai_edited` lấy từ MS-COCO, `fully_generated` lấy từ GenImage), mô hình nơ-ron có xu hướng ghi nhớ các đặc trưng riêng của từng nguồn dữ liệu (độ phân giải gốc, camera color gamut, profile nén JPEG, bộ lọc tiền xử lý) thay vì học bản chất dấu vết thuật toán AI.

### Giao thức Thí nghiệm Pilot Hai Nhánh Độc lập (Two-Branch Architecture):
Chi tiết xem [PILOT_PROTOCOL.md](docs/PILOT_PROTOCOL.md):
1. **Pilot A (Authentic vs AI-Edited + Localization)**: Sử dụng các cặp matched pairs của TGIF (`orig` authentic MS-COCO + `sd2-sp` inpainting + `masks`). Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec, quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh vẫn phải được đo bằng baseline và source-held-out evaluation.
2. **Pilot B (Authentic vs Fully-Generated)**: Sử dụng cặp đối chứng trong cùng phân phối ImageNet của GenImage (val nature vs BigGAN ai).
3. **Pilot C (Three-Class Exploratory)**: Chỉ được mở khi Pilot A và Pilot B vượt qua bài kiểm toán shortcut; bắt buộc kèm **Metadata-Only Baseline Guard**.

### Biện pháp kiểm soát bắt buộc:
1. **Group Split theo ảnh gốc**: Đảm bảo toàn bộ biến thể phái sinh từ một ảnh gốc (`source_id`) luôn nằm trọn vẹn trong một split duy nhất (Train, Val hoặc Test).
2. **Dùng Matched Pairs**: Khi dùng TGIF, sử dụng chính các cặp ảnh gốc authentic (MS-COCO) đi kèm với các biến thể inpainting của chính ảnh đó.
3. **Chuẩn hóa Tiền xử lý (Standardized Preprocessing)**: Toàn bộ ảnh đầu vào đều đi qua pipeline đồng nhất (letterbox padding, resize về $224 \times 224$ px hoặc $512 \times 512$ px, chuẩn hóa kênh màu theo ImageNet mean/std).
4. **Đánh giá Cross-Dataset & Source-Held-Out**: Báo cáo kết quả chi tiết theo từng nguồn dataset và từng generator cụ thể; không gộp chung số liệu để che giấu hiện tượng sụt giảm độ chính xác trên nguồn ảnh lạ.
