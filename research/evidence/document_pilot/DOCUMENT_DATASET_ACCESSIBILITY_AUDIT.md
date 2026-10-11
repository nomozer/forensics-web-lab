# Kiểm Toán Khả Năng Tiếp Cận Dữ Liệu Tài Liệu Giả Mạo AI (Document AI-Tampering Data Accessibility Audit)

> **Ngày thực hiện**: 2026-10-11
> **Môi trường**: LOCAL CPU, branch `research/independent-cohort-acquisition`
> **Mục tiêu**: Đánh giá tính khả dụng, điều kiện pháp lý, rào cản truy cập và cấu trúc ghép cặp của các bộ dữ liệu tài liệu bị chỉnh sửa / giả mạo bằng AI trong y văn.

---

## 1. Rà Soát Năm Mục Tiêu Y Văn Sơ Cấp (Primary Literature Targets)

| Bộ dữ liệu / Nghiên cứu | Nguồn công bố & Tác giả | Trạng thái tiếp cận thực tế | Rào cản truy cập (Access Blockers) | Bản quyền / Giấy phép | Định dạng & Cấu trúc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DocTamper / DTD** | CVPR 2023, Qu et al. (`qcf-568/DocTamper`) | **Bị khóa / Cần phê duyệt** | Kho lưu trữ Baidu/Kaggle được nén mật khẩu; yêu cầu gửi email từ địa chỉ cơ sở đào tạo (.edu) tới `202221012612@mail.scut.edu.cn` hoặc nộp đơn qua form `http://121.41.49.212:9000/apply/doctamper`. | Non-commercial research only | Định dạng LMDB (`.mdb`). Dữ liệu can thiệp bằng phương pháp sinh văn bản tổng hợp truyền thống (STG) / copy-move / splicing, không phải diffusion inpainting hiện đại. |
| **OSTF** | AAAI 2025, Wang et al. (`qcf-568/OSTF`) | **Bị khóa / Cần phê duyệt** | Link Google Drive và Baidu Drive yêu cầu nộp đơn qua email giáo dục có thông tin người hướng dẫn tới `202221012612@mail.scut.edu.cn`. | Unspecified / Research application | Dữ liệu chữ trong cảnh tự nhiên (**Scene Text**: biển hiệu, cửa hàng, menu từ ArT, ICDAR, LSVT, TextOCR), **không phải tài liệu hành chính / biểu mẫu / hóa đơn**. |
| **AIForge-Doc** (v1 & v2) | arXiv:2602.20569, Scam-AI (`Scam-AI/AIForge-Doc-v1`, `v2`) | **Gated trên Hugging Face** | Gated dataset (`gated: auto`); trả về mã lỗi **HTTP 401 Unauthorized** khi tải trực tiếp qua script không có API token đã được duyệt. Không có public mirror mở. | CC-BY-NC-SA 4.0 | 4,061 ảnh hóa đơn / biểu mẫu (CORD, WildReceipt, SROIE, XFUND) bị inpainting bằng Gemini 2.5 Flash Image và Ideogram v2 Edit; có mask điểm ảnh và cặp authentic. Rất phù hợp về mặt khoa học nhưng bị chặn bởi rào cản tài khoản cá nhân. |
| **Fake or JPEG?** | arXiv:2403.17608, Grommelt et al. (2024) | **Nghiên cứu phương pháp luận (Công khai)** | Mã nguồn và dữ liệu công khai tại `keuper-labs.org`. | Open research | Chứng minh các mô hình học sâu dễ học các shortcut nén JPEG và phân phối kích thước ảnh thay vì đặc trưng tạo sinh thật. Đóng vai trò cơ sở lý luận đối chứng cho các phát hiện nén C1 vs C2 và chẩn đoán nhân tố 2×2 của đề tài. |
| **GADNet / DEA-Net** | ESWA 2025, DOI: 10.1016/j.eswa.2025.130500, Li et al. | **Mã nguồn công khai** (`cecret3350/DEA-Net`) | Mã nguồn mô hình khả dụng trên GitHub; bài báo tập trung vào phân loại ảnh tạo sinh tổng quát, không cung cấp dataset tài liệu chuyên biệt. | Open research | Kiến trúc CNN nhẹ khai thác đặc trưng dual-domain (không phải dataset tài liệu). |

---

## 2. Rà Soát Bộ Dữ Liệu Công Khai Thay Thế: RealText-V2 (ACM MM 2026)

Để thực hiện pilot thực nghiệm có kiểm soát mà không vi phạm quy tắc không tự ý tải dữ liệu mật khẩu / trả phí, chúng tôi đã khảo sát và xác minh bộ dữ liệu **RealText-V2** (`vankey/RealText-V2`):

- **Nguồn gốc**: Cuộc thi quốc tế *ACM MM 2026 MGC: GenText-Forensics (Challenge on Explainable Forensics and Adversarial Generation for Text-Centric Images)*.
- **Giấy phép**: **CC-BY-NC-4.0** (phi thương mại, cho phép nghiên cứu khoa học).
- **Trạng thái truy cập**: **Công khai hoàn toàn (Ungated, `gated: false`)**, cho phép tải trực tiếp qua HTTP URL mà không cần đăng nhập hay khóa API.
- **Quy mô**: 13,500 ảnh tài liệu trong split huấn luyện (7,500 ảnh bị làm giả `black` có mask điểm ảnh; 6,000 ảnh nguyên bản `white`).
- **Lĩnh vực tài liệu**: Tài chính (financial filings, corporate records), biểu mẫu, y tế, hợp đồng, tài liệu kỹ thuật, e-commerce.
- **Đặc điểm pháp chứng then chốt**:
  1. *Không ghép cặp 1:1 cùng ảnh nguồn (Unpaired)*: Tập `black` (forged) và tập `white` (authentic) là các tài liệu độc lập riêng biệt (Stem overlap = 0).
  2. *Bất đối xứng định dạng lưu trữ*: Ảnh forged được lưu dưới dạng `.jpg`, trong khi ảnh authentic được lưu dưới dạng `.png`. Đây là một **domain/compression shortcut tiềm ẩn** tương tự cảnh báo của *Fake or JPEG?*.
  3. *Tiền xử lý chuẩn tắc bắt buộc*: Để tránh mô hình nhận diện dựa vào đuôi file hoặc lưới JPEG, toàn bộ ảnh phải được giải mã về mảng pixel RGB thuần túy và đưa qua quy trình Canonical Preprocessing trước khi trích xuất đặc trưng.

---

## 3. Kết Luận Điểm Chặn & Giải Pháp Pilot

1. **Điểm chặn (Data Access Blocker)**: Không có bộ dữ liệu tài liệu bị chỉnh sửa bằng AI nào có **ghép cặp 1:1 hoàn hảo giữa authentic và edited** mà **hoàn toàn mở (ungated)**, miễn phí và không yêu cầu thủ tục đăng ký cá nhân qua email cơ quan giáo dục hoặc Hugging Face token.
2. **Giải pháp thực nghiệm khoa học**:
   - Khóa một panel tài liệu tất định gồm **$N=60$ tài liệu** (30 authentic + 30 AI/forged có mask phân đoạn điểm ảnh) từ nguồn công khai RealText-V2.
   - Giới hạn tải mạng $\le 50$ MB (chỉ tốn ~35 MB, bảo đảm an toàn ngân sách mạng còn lại 324 MB).
   - Thực thi đối chiếu 3 điều kiện hình học quan sát (**P0 Center-crop**, **P1 Letterbox Canvas**, **P2 Multi-Tile 5 views**) trên 6 locked models để kiểm tra trực tiếp đánh đổi giữa độ trễ, khả năng giữ vùng sửa và tỷ lệ báo động giả.
