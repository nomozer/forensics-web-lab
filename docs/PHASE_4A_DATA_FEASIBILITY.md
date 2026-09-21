# Khảo sát Khả thi Dữ liệu và Kế hoạch Chuẩn bị (Phase 4A Data Feasibility)

> **Mục đích tài liệu**: Lập kế hoạch phân tích tính khả thi và tiêu chuẩn dữ liệu trước Phase 4A.  
> **Nguyên tắc bất biến**:  
> - Tài liệu này **CHỈ LẬP KẾ HOẠCH, TUYỆT ĐỐI KHÔNG TẢI DỮ LIỆU**.  
> - Bắt buộc phân biệt rạch ròi giữa **Pipeline Smoke Test** và **Scientific Training & Evaluation**.  
> - Mọi thông tin giấy phép chưa có văn bản chính thức phải ghi `unverified`.  
> - Phase 4A **chỉ được bắt đầu sau khi người dùng phê duyệt rõ ràng bằng văn bản**.

---

## 1. Phân định Hai Mục tiêu Sử dụng Dữ liệu

| Đặc điểm | Mục tiêu 1: Pipeline Smoke Test | Mục tiêu 2: Scientific Training & Evaluation |
| :--- | :--- | :--- |
| **Mục đích cốt lõi** | Xác minh thông suốt đường ống kỹ thuật đầu-cuối: `manifest -> dedup -> group split -> train step -> export ONNX -> quantize -> registry -> WASM worker`. | Huấn luyện mô hình có năng lực tổng quát hóa, đo đạc chỉ số khoa học thực tế, kiểm định độ bền vững trước suy giảm và phát hiện can thiệp. |
| **Quy mô dữ liệu** | Rất nhỏ (khoảng vài chục đến vài trăm ảnh mẫu, $< 50\text{ MB}$). | Lớn và đa dạng (hàng chục nghìn ảnh, nhiều bộ tạo sinh khác nhau, ước tính $10\text{ GB} - 50\text{ GB}$). |
| **Giá trị kết luận** | **KHÔNG CÓ GIÁ TRỊ KHOA HỌC**. Tuyệt đối không dùng mô hình từ smoke test để tuyên bố khả năng phát hiện hay độ chính xác. | Có giá trị công bố khoa học; đo đạc độ chính xác, Macro F1, ECE và mIoU trên tập kiểm thử độc lập. |
| **Rủi ro rò rỉ dữ liệu** | Thấp (chỉ nhằm kiểm tra lỗi cú pháp và shape tensor). | Rất cao nếu không kiểm soát nghiêm ngặt ranh giới `source_id` giữa các tập. |
| **Yêu cầu GPU** | Có thể chạy nhanh trên CPU (vài epochs ngắn). | Bắt buộc GPU rời (CUDA/T4/A100) để hội tụ gradient và tối ưu hóa hàm mất mát Focal Loss. |

---

## 2. Tiêu chuẩn Khoa học cho Tập Dữ liệu Huấn luyện Thật (Scientific Training)

Để một tập dữ liệu được chấp thuận đưa vào huấn luyện mô hình sản phẩm ở Phase 4, tập dữ liệu đó bắt buộc phải đáp ứng 8 tiêu chuẩn:

1. **Đa dạng nguồn phát sinh (Generator Diversity)**: Bao phủ cả họ mô hình GAN cổ điển (ProGAN, StyleGAN) và họ Latent Diffusion hiện đại (Stable Diffusion v1.5, SDXL, Midjourney, Flux).
2. **Cách ly rò rỉ nhóm ảnh gốc (Group-Based Split)**: Mọi biến thể từ cùng một `source_id` ảnh gốc bắt buộc phải nằm trọn vẹn trong một tập duy nhất (Train hoặc Val hoặc Test).
3. **Đánh giá Unseen-Generator**: Phải trích xuất ít nhất một họ tạo sinh hoàn toàn độc lập (ví dụ: Wukong hoặc Midjourney) chỉ để kiểm thử mù, không xuất hiện trong tập huấn luyện.
4. **Bộ mẫu thử thách (Hard Negatives)**: Bắt buộc có ảnh thông thường được chỉnh sửa bằng Photoshop truyền thống (crop, color grading, tone mapping, frequency separation) để mô hình học cách không báo động giả khi gặp ảnh chỉnh sửa phi AI.
5. **Dữ liệu suy giảm thực tế (Realistic Degradations)**: Tập kiểm thử phải có các mẫu ảnh bị nén lại JPEG nhiều lần ($Q=40-75$), ảnh bị resize giảm độ phân giải, và ảnh chụp lại màn hình (screenshot).
6. **Nhánh riêng cho bài toán tạo sinh toàn phần (`fully_generated`)**: Các ảnh toàn cảnh được sinh trực tiếp từ text-to-image prompt.
7. **Nhánh riêng cho bài toán chỉnh sửa cục bộ (`ai_edited`)**: Ảnh gốc có các vùng được inpainting hoặc thay thế bằng AI, **bắt buộc phải có ground-truth binary mask** (`mask_path`) để phục vụ kiểm định bản đồ nhiệt localization.
8. **Kiểm tra Giấy phép và Điều khoản Tái phân phối**: Xác minh tính hợp pháp của việc nạp dữ liệu và xuất bản trọng số sau này.

---

## 3. Bảng Quyết định và Đánh giá Khả thi Dataset

| Dataset | Nhiệm vụ Hướng tới | Nguồn Chính thức | Tình trạng Giấy phép | Quyền Phân phối Trọng số | Ước tính Dung lượng Tải | Rủi ro Kỹ thuật & Bản quyền | Quyết định Đề xuất |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Phát hiện toàn ảnh (`fully_generated`) | [GitHub: GenImage-Dataset](https://github.com/GenImage-Dataset/GenImage) | Academic / Research Only (`reported-by-source`) | `unverified` (phụ thuộc điều khoản ImageNet gốc) | ~30 GB – 120 GB (nhiều subset) | Dung lượng rất lớn; ảnh thật lấy từ ImageNet có bản quyền phức tạp | **Chờ duyệt lát cắt nhỏ** |
| **SAGI-D** | Phát hiện và định vị inpainting (`ai_edited`) | [GitHub: mever-team/SAGI](https://github.com/mever-team/SAGI) | Research Benchmark (`unverified`) | `unverified` | ~10 GB – 25 GB | Tác giả chỉ công bố trên Kaggle; cần tạo API key Kaggle để tải | **Chờ duyệt lát cắt nhỏ** |
| **RealHD** | Benchmark đa dạng AI và Face Manipulation | ACM Multimedia 2025 Paper | `unverified` | `unverified` | `unverified` | Chưa có link tải tự động công khai tập trung | **blocked** (chưa đủ nguồn rõ ràng) |
| **RAID** | Đánh giá tổng quát hóa đa generator | [GitHub: RAID-Benchmark](https://github.com/raid-benchmark/raid) | Non-commercial Research (`reported`) | `unverified` | ~20 GB – 40 GB | Dung lượng lớn, chỉ dùng cho đánh giá held-out | **Chờ duyệt Phase 4B** |
| **Tập Mẫu Smoke Test Nội bộ** | Kiểm thử kỹ thuật pipeline (Smoke Test) | Cung cấp thủ công hoặc sinh từ synthetic scripts | Giấy phép nội bộ dự án | 100% kiểm soát | $< 50\text{ MB}$ (50–100 ảnh) | Không thể dùng để tuyên bố chỉ số phát hiện | **Khuyến nghị cho Phase 4A** |

---

## 4. Cổng Phê duyệt Bước sang Phase 4A (Approval Gate)

Giai đoạn Phase 4A chỉ được kích hoạt khi và chỉ khi:

1. **Người dùng phê duyệt chính thức bằng văn bản** chọn phương án dữ liệu:
   - *Phương án 1 (Khuyến nghị)*: Chuẩn bị một tập mẫu nhỏ phục vụ **Pipeline Smoke Test** ($< 50\text{ MB}$, 50–100 ảnh) để kiểm chứng mã huấn luyện, xuất ONNX và đo đạc latency kỹ thuật mà không tuyên bố độ chính xác.
   - *Phương án 2*: Phê duyệt tải một lát cắt chính thức (subset nhỏ có chọn lọc) của GenImage hoặc SAGI-D kèm tài nguyên GPU để bắt đầu huấn luyện khoa học.
2. **Xác nhận hạn mức lưu trữ và mạng**: Người dùng chấp thuận dung lượng tải và phân vùng lưu trữ local.
3. **Cam kết phân loại kết quả**: Mọi kết quả đo đạc từ tập smoke test bắt buộc phải gắn nhãn `pipeline-only`, không được công bố là hiệu năng thực của sản phẩm `Forensics Web Lab`.
