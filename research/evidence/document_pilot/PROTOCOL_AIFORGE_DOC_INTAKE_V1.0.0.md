# Protocol Specification v1.0.0: AIForge-Doc Cohort Intake & Zero-Shot Evaluation

> **Protocol ID**: `PROTO-AIFORGE-DOC-INTAKE-V1.0.0`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Mục tiêu**: Thiết lập quy trình tiền đăng ký (pre-registered protocol) để tiếp nhận panel tài liệu có nguồn gốc Generative AI xác minh 100% từ benchmark `AIForge-Doc` (Scam-AI, arXiv:2602.20569) và đánh giá zero-shot trên 6 locked frozen models.<br>
> **Ranh giới khoa học**: Không tự ý huấn luyện, không vượt rào cản xác thực (no bypass of gated access), không giả lập dữ liệu, tuân thủ nghiêm ngặt ngân sách mạng $\le 1\text{ GB}$.

---

## 1. Cơ Sở Lý Luận & Lựa Chọn Nguồn Dữ Liệu Sơ Cấp

1. **Khắc phục triệt để khiếm khuyết của RealText-V2**:
   - Kiểm toán thực nghiệm tại Phase 4C.8 xác nhận: RealText-V2 có **0/40 mẫu** xác minh được nguồn gốc Generative AI (toàn bộ 20 mẫu giả mạo thuộc nhóm can thiệp văn bản số chưa rõ công cụ), đồng thời bị nhiễu định dạng 100% (`.jpg` 4:2:0 cho forged vs `.png` cho authentic).
   - Do đó, để trả lời câu hỏi cốt lõi của đề tài về *khả năng phát hiện tài liệu giả mạo sinh bởi AI*, hệ thống bắt buộc phải tiếp nhận một cohort có **100% verified Generative AI provenance** và **cấu trúc ghép cặp 1:1 nguồn gốc**.

2. **Hồ sơ nguồn dữ liệu AIForge-Doc (Scam-AI)**:
   - **Repository**: `Scam-AI/AIForge-Doc-v1` (và v2 bổ trợ).
   - **Công bố khoa học**: Wu, Zhou, Xu et al., *“AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents”*, arXiv:2602.20569 (2026).
   - **Giấy phép**: **CC-BY-NC-SA 4.0** (phi thương mại, cho phép nghiên cứu khoa học).
   - **Công cụ tạo sinh (Generators/Inpainters)**:
     * Gemini 2.5 Flash Image inpainting;
     * Ideogram v2 Edit inpainting;
     * GPT-Image-2 (trên v2).
   - **Tài liệu gốc (Source Corpora)**: CORD v2 (hóa đơn), WildReceipt (biên lai), SROIE (hóa đơn bán lẻ), XFUND (biểu mẫu).
   - **Cấu trúc lưu trữ**: DocTamper-compatible, chia sẵn `TrainingSet` (3,249 ảnh) và `TestingSet` (812 ảnh forged + 812 ảnh authentic đối chứng gốc + 812 ground-truth masks).
   - **Định dạng**: Đồng nhất PNG lossless cho cả ảnh gốc, ảnh bị sửa và mặt nạ (loại bỏ hoàn toàn format-label shortcut).

---

## 2. Quy Tắc Chọn Mẫu Tiền Đăng Ký (Pre-Registered Cohort Sampling)

1. **Mục tiêu quy mô panel**:
   - Đúng **40 cặp nguồn (40 source pairs)**, tương đương **80 ảnh tài liệu**:
     * 40 ảnh nguyên bản (**40 authentic documents**);
     * 40 ảnh bị can thiệp AI inpainting (**40 AI-inpainted documents**);
     * 40 mặt nạ điểm ảnh ground-truth (**40 binary masks** phục vụ kiểm toán che phủ hình học).
2. **Quy tắc phân vùng (Split Binding)**:
   - Độc quyền lấy mẫu từ tập kiểm định tác giả định nghĩa (**`TestingSet/`**), không lấy từ `TrainingSet/`.
3. **Quy tắc chọn mẫu tất định (Deterministic Sampling Rule)**:
   - Lọc danh sách 812 mẫu trong `TestingSet/images/`, sắp xếp theo thứ tự bảng chữ cái của `sample_id`.
   - Áp dụng bộ tạo số ngẫu nhiên giả với seed cố định: `SEED = 20261012`.
   - Lấy mẫu phân tầng (stratified sampling) bảo đảm đại diện cân bằng giữa các nguồn tài liệu gốc (CORD, SROIE, WildReceipt) và công cụ inpainting (Gemini vs Ideogram).
   - Tuyệt đối không chọn mẫu dựa trên độ khó/dễ thị giác hoặc kết quả thử nghiệm của mô hình.
4. **Giới hạn ngân sách**:
   - Dung lượng tải tối đa: $\le 1\text{ GB}$ (thực tế 80 ảnh PNG độ phân giải tài liệu ước tính $\approx 60 - 120\text{ MB}$, hoàn toàn nằm trong hạn mức).
   - Tuyệt đối không tải toàn bộ 12,187 files của dataset.

---

## 3. Khóa Cấu Hình Đánh Giá Zero-Shot

1. **Mô hình**: Giữ nguyên vẹn 6 locked frozen models từ `ml/training/locked_models_registry.json`:
   - `A0-visual` (576-D MobileNetV3 ImageNet);
   - `A0-global` (592-D MobileNetV3 + Global DSP 16-D);
   - `A0-local` (640-D MobileNetV3 + Global DSP 16-D + Local DSP 48-D);
   - `A1-visual`, `A1-global`, `A1-local` (huấn luyện bổ sung ImageNet authentic).
2. **Không fine-tuning/recalibration**:
   - $T_{\text{scaler}}$, $C=1.0$, trọng số tuyến tính và scaling coefficients giữ nguyên 100%.
3. **Hình học quan sát**:
   - **P0**: Canonical Center-Crop $\min(w, h) \to 224 \times 224$ (Visual) và $512 \times 512$ (DSP).
   - **P1**: Orientation-Correct Aspect-Preserving Letterbox Canvas với đệm xám 128 $\to 224 \times 224$ và $512 \times 512$.
4. **Chỉ số tính điểm**:
   - Điểm số phát hiện: $s_{\text{AI}} = p(\text{ai\_edited}) + p(\text{fully\_generated})$.
   - Quyết định nhị phân: Positive nếu $s_{\text{AI}} \ge 0.5$, Negative nếu $s_{\text{AI}} < 0.5$.
5. **Thống kê ghép cặp**:
   - Resampling bootstrap 10,000 lần theo **Source Pair** (cặp nguồn giữ nguyên 2 ảnh authentic và tampered trong cùng một đơn vị rút mẫu).

---

## 4. Rào Cản Xác Thực & Cơ Chế Fail-Closed

1. **Trạng thái thực tế trên Hugging Face**:
   - Dataset `Scam-AI/AIForge-Doc-v1` và `v2` được cấu hình **`gated: auto`**.
   - Việc tải file trực tiếp qua HTTPS mà không có Hugging Face user token đã ký chấp thuận thỏa thuận học thuật sẽ trả về mã lỗi **HTTP 401 Unauthorized**.
2. **Quy định đạo đức & pháp lý**:
   - AI agent **tuyệt đối không tìm cách vượt rào cản xác thực (authentication wall)**, không tìm bản sao rò rỉ trên các kênh không chính thống, không cố gắng sinh ảnh giả lập bằng API thương mại để thay thế.
   - Khi môi trường chưa có `HF_TOKEN` hợp lệ:
     * Kích hoạt cơ chế **FAIL-CLOSED**;
     * Xuất biên nhận kiểm toán điểm chặn truy cập (`aiforge_doc_access_blocker_audit.json`);
     * Dừng luồng suy luận và báo cáo minh bạch cho người dùng;
     * Cung cấp hướng dẫn chính xác để người dùng phê duyệt quyền trên Hugging Face.
