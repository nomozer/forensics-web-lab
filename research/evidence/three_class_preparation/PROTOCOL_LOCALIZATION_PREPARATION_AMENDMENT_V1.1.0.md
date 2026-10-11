# Protocol Amendment Specification v1.1.0: Localization Protocol Clarification & Epistemic Boundaries

> **Protocol ID**: `PROTO-LOCALIZATION-PREPARATION-AMENDMENT-V1.1.0`<br>
> **Supersedes/Amends**: `PROTO-LOCALIZATION-PREPARATION-V1.0.0` (preserving v1.0.0 untouched for lineage audit)<br>
> **Date**: 2026-10-12<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Status**: `AMENDED_SPECIFICATION_NO_TRAINING_AUTHORIZED`<br>
> **Model Status**: `NOT_RUN / NOT_MEASURED` (Zero training, zero model checkpoint acquisition in this turn)

---

## 1. Mục Đích Tu Chỉnh & Đính Chính Ranh Giới Khoa Học

Văn bản này bổ sung và đính chính các phát biểu trong `PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md` nhằm bảo đảm tính trung thực khoa học và tuân thủ ranh giới nghiên cứu của đề tài:

1. **Khôi Phục Tên Đề Tài Chính Thức**:
   - Tên đề tài chính thức bất biến: **“Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”**
   - Loại bỏ cách đặt tên cũ gây hiểu nhầm về phạm vi.
2. **Định Vị Localization Là Hướng Ứng Viên Khám Phá (Candidate Exploratory Direction)**:
   - Bài toán định vị cấp điểm ảnh (Pixel-Level Localization) được xác định là một **hướng nghiên cứu ứng viên bổ trợ**, **chưa được xác nhận bởi các pilot thực nghiệm hiện tại**.
   - Đề tài KHÔNG tự ý thay đổi câu hỏi nghiên cứu (Research Questions - RQ), không chuyển trọng tâm cốt lõi sang bài toán phân đoạn ảnh (segmentation), và không coi localization là yêu cầu bắt buộc của đề tài.
3. **Đếm Lại Số Lượng Ảnh Có Mask Chuẩn Xác Từ Manifests**:
   - Trong `PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md`, việc ghi nhận "Development Train: $N=750$ cặp" và "Development Holdout: $N=91$ cặp" là sai lệch:
     * Tập `development_train` ba lớp ($N=750$ ảnh) thực chất gồm: **250 ảnh COCO authentic + 250 ảnh TGIF SD2 inpainting edited + 250 ảnh SD1.5 fully generated**. Do đó, chỉ có duy nhất **250 ảnh edited có mặt nạ (ground-truth mask)**!
     * Tập `development_holdout` ba lớp ($N=273$ ảnh) gồm: **91 ảnh authentic + 91 ảnh TGIF SD2 edited + 91 ảnh SD1.5 fully generated**. Do đó, chỉ có duy nhất **91 ảnh edited có mặt nạ (ground-truth mask)**!
   - Tuyệt đối không gọi $N=750$ hay $N=273$ là số lượng cặp localization.
4. **Xác Định Rõ Bản Chất Của Development Holdout**:
   - Tập `development_holdout` ($N=273$) là tập phát triển/thăm dò đã được quan sát và đánh giá nhiều lần qua các pha trước.
   - **Không được coi Development Holdout là tập kiểm định độc lập cuối cùng (final independent test set)**.
5. **Ngân Sách Tài Nguyên Là Mục Tiêu Thiết Kế Chưa Đo Lường**:
   - Các chỉ số ngân sách mô hình (< 10 MB, < 2.5M params, CPU < 50 ms, WebGL/WebGPU < 35 ms) chỉ là **mục tiêu thiết kế dự kiến (unmeasured design targets)**, chưa được đo lường thực tế trên bất kỳ mô hình localization nào.
6. **Xóa Bỏ Cam Kết Tự Động Huấn Luyện**:
   - **Xóa bỏ hoàn toàn cam kết tự động huấn luyện U-Net trên Colab GPU ở lượt kế tiếp**.
   - Mọi quyết định huấn luyện phải có sự phê duyệt rõ ràng từ người dùng và chỉ được thực hiện khi dữ liệu đáp ứng đầy đủ điều kiện về nguồn gốc AI, tính ghép cặp và sự cô lập confounding.

---

## 2. Bảng Đối Chiếu Thay Đổi Cụ Thể (Clause-by-Clause Amendment Matrix)

| Mục trong v1.0.0 | Nội dung cũ (v1.0.0) | Nội dung tu chỉnh (v1.1.0 Amendment) |
| :--- | :--- | :--- |
| **Header & Title** | *"Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web"* | **“Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”** |
| **Mục 1.3** | *"Chuyển dịch trọng tâm sang Định vị Điểm ảnh (Pixel-Level Localization)"* | **Định vị điểm ảnh là hướng nghiên cứu ứng viên khám phá (candidate direction)**, không thay thế bài toán phát hiện cấp ảnh và không phải yêu cầu bắt buộc của đề tài. |
| **Mục 2.2** | Ngân sách tài nguyên ghi như tiêu chí đã thiết lập | **Mục tiêu thiết kế chưa đo lường (unmeasured design targets)**. |
| **Mục 4.2** | *"Development Train: N=750 cặp... Development Holdout: N=91 cặp edited + 91 auth + 91 gen"* | **Development Train chỉ có 250 ảnh edited có mask** (trong tổng số 750 ảnh ba lớp); **Development Holdout chỉ có 91 ảnh edited có mask** (trong tổng số 273 ảnh ba lớp). |
| **Mục 4.2** | Development Holdout ghi như tập đánh giá cuối cùng | **Development Holdout là tập phát triển nội bộ đã xem nhiều lần**, không phải final independent test. |
| **Mục 7.4** | *"Cam kết lượt tiếp theo: Triển khai huấn luyện kiến trúc U-Net mini đã đặc tả trên Colab GPU"* | **Xóa bỏ cam kết huấn luyện tự động**. Mọi hoạt động huấn luyện dừng lại chờ phê duyệt kế hoạch nghiên cứu chính thức. |

---

## 3. Trạng Thái Pháp Lý & Tính Bất Biến

- File văn bản gốc `PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md` được niêm phong nguyên vẹn để phục vụ truy vết lịch sử (lineage audit).
- Bản tu chỉnh v1.1.0 này có hiệu lực ràng buộc đối với mọi hoạt động nghiên cứu tiếp theo liên quan đến phân đoạn điểm ảnh và đánh giá mask.
