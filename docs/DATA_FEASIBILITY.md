# Khảo sát Khả thi Dữ liệu và Kế hoạch Chuẩn bị (Phase 4A Data Feasibility)

> **Mục đích tài liệu**: Lập kế hoạch phân tích tính khả thi và tiêu chuẩn dữ liệu trước Phase 4A theo chính sách Dual-Track.  
> **Nguyên tắc bất biến**:  
> - Tài liệu này **CHỈ LẬP KẾ HOẠCH, TUYỆT ĐỐI KHÔNG TẢI DỮ LIỆU**.  
> - Bắt buộc phân biệt rạch ròi giữa **Research Track** (nghiên cứu) và **Product Track** (sản phẩm thương mại).  
> - Phân biệt giữa **Synthetic Smoke Fixture** (không dữ liệu ngoài) và **Scientific Training & Evaluation**.  
> - Mọi thông tin giấy phép chưa có văn bản chính thức phải ghi `unverified`.  
> - Phase 4A **chỉ được bắt đầu sau khi người dùng phê duyệt rõ ràng bằng văn bản**.

---

## 1. Phân định Hai Mục tiêu Sử dụng Dữ liệu

| Đặc điểm | Mục tiêu 1: Pipeline Smoke Test | Mục tiêu 2: Scientific Training & Evaluation |
| :--- | :--- | :--- |
| **Mục đích cốt lõi** | Xác minh thông suốt đường ống kỹ thuật đầu-cuối: `manifest -> dedup -> group split -> train step -> export ONNX -> quantize -> registry -> WASM worker`. | Huấn luyện mô hình có năng lực tổng quát hóa, đo đạc chỉ số khoa học thực tế, kiểm định độ bền vững trước suy giảm và phát hiện can thiệp. |
| **Quy mô dữ liệu** | Rất nhỏ (50–100 ảnh mẫu, $< 50\text{ MB}$). Ưu tiên fixture sinh từ code (`ml/tests/fixtures/generated-smoke/`). | Lớn và đa dạng (hàng chục nghìn ảnh, nhiều bộ tạo sinh khác nhau, ước tính $10\text{ GB} - 50\text{ GB}$). |
| **Giá trị kết luận** | **KHÔNG CÓ GIÁ TRỊ KHOA HỌC**. Tuyệt đối không dùng mô hình từ smoke test để tuyên bố khả năng phát hiện hay độ chính xác. | Có giá trị công bố khoa học; đo đạc độ chính xác, Macro F1, ECE và mIoU trên tập kiểm thử độc lập. |
| **Rủi ro rò rỉ dữ liệu** | Thấp (chỉ nhằm kiểm tra lỗi cú pháp và shape tensor). | Rất cao nếu không kiểm soát nghiêm ngặt ranh giới `source_id` giữa các tập. |
| **Yêu cầu GPU** | Chạy nhanh trên CPU (vài step/epochs ngắn). | Bắt buộc GPU rời (CUDA/T4/A100) để hội tụ gradient và tối ưu hóa hàm mất mát Focal Loss. |

---

## 2. Tiêu chuẩn Khoa học cho Tập Dữ liệu Huấn luyện Thật (Scientific Training)

Để một tập dữ liệu được chấp thuận đưa vào huấn luyện mô hình sản phẩm hoặc nghiên cứu khoa học, tập dữ liệu đó bắt buộc phải đáp ứng 8 tiêu chuẩn:

1. **Đa dạng nguồn phát sinh (Generator Diversity)**: Bao phủ cả họ mô hình GAN cổ điển và họ Latent Diffusion hiện đại.
2. **Cách ly rò rỉ nhóm ảnh gốc (Group-Based Split)**: Mọi biến thể từ cùng một `source_id` ảnh gốc bắt buộc phải nằm trọn vẹn trong một tập duy nhất (Train hoặc Val hoặc Test).
3. **Đánh giá Unseen-Generator**: Phải trích xuất ít nhất một họ tạo sinh hoàn toàn độc lập chỉ để kiểm thử mù, không xuất hiện trong tập huấn luyện.
4. **Bộ mẫu thử thách (Hard Negatives)**: Bắt buộc có ảnh thông thường được chỉnh sửa bằng Photoshop truyền thống để mô hình học cách không báo động giả khi gặp ảnh chỉnh sửa phi AI.
5. **Dữ liệu suy giảm thực tế (Realistic Degradations)**: Tập kiểm thử phải có các mẫu ảnh bị nén lại JPEG nhiều lần ($Q=40-75$), ảnh bị resize giảm độ phân giải, và ảnh screenshot.
6. **Nhánh riêng cho bài toán tạo sinh toàn phần (`fully_generated`)**: Các ảnh toàn cảnh được sinh trực tiếp từ text-to-image prompt.
7. **Nhánh riêng cho bài toán chỉnh sửa cục bộ (`ai_edited`)**: Ảnh gốc có các vùng được inpainting, **bắt buộc phải có ground-truth binary mask** (`mask_path`).
8. **Tuân thủ Bản quyền và Ranh giới Hai Luồng (Track Compliance)**:
   - Dữ liệu mang điều khoản phi thương mại (như GenImage) chỉ được huấn luyện trong **Research Track**.
   - Trọng số tạo ra từ luồng nghiên cứu tuyệt đối không được đưa vào `models/product/` hoặc phân phối trên client web.

---

## 3. Bảng Quyết định và Đánh giá Khả thi Dataset

| Dataset | Nhiệm vụ Hướng tới | Nguồn Chính thức | Giấy phép Dataset | Quyền Trọng số Phái sinh | Khả dụng | Rủi ro Kỹ thuật & Bản quyền | Quyết định Đề xuất |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Phát hiện toàn ảnh (`fully_generated`) | [GitHub: GenImage](https://github.com/GenImage-Dataset/GenImage) | `CC BY-NC-SA 4.0 with additional dataset terms` | Cấm thương mại (`prohibited`) | Khả dụng | Ràng buộc phi thương mại; cấm thương mại hóa trọng số phái sinh | **`research-only`** (Chờ duyệt tải mẫu) |
| **SAGI-D** | Phát hiện và định vị inpainting (`ai_edited`) | [GitHub: SAGI](https://github.com/mever-team/SAGI) | Phi thương mại / Học thuật | Chưa rõ (`unclear`) | Khả dụng trên Kaggle | Tác giả chỉ công bố trên Kaggle; quyền phân phối lại trọng số chưa rõ | **`research-only`** (Chờ duyệt tải mẫu) |
| **RealHD** | Benchmark đa dạng AI và Face Manipulation | [GitHub: RealHD](https://github.com/Hanzhe-yu/RealHD) | `unverified` | Chưa rõ (`unclear`) | `unavailable-or-pending` ("Coming soon") | Chưa có bản phát hành dữ liệu chính thức | **`blocked`** (Chưa đủ điều kiện) |
| **RAID** | Đánh giá tổng quát hóa đa generator | [GitHub: RAID](https://github.com/raid-benchmark/raid) | Non-commercial Research | Chưa rõ (`unclear`) | Khả dụng | Dung lượng lớn (20–40 GB); chỉ dùng cho đánh giá held-out | **`research-only`** (Chờ Phase 4B) |
| **Synthetic Smoke Fixture** | Smoke test pipeline kỹ thuật không tải mạng | Mã sinh nội bộ `ml/tests/fixtures/` | Project-Internal | Cho phép (`allowed`) | Khả dụng ngay | Không phải ảnh thật; không có giá trị đo độ chính xác | **`product-eligible`** (Khuyến nghị cho smoke test) |

> [!IMPORTANT]
> **Làm rõ về thuật ngữ tập mẫu**: Không có bộ dữ liệu chính thức nào mang tên "GenImage Mini". Mọi đề xuất trích xuất tập con nhỏ từ GenImage được định danh chính xác là **`Custom smoke subset sampled from GenImage`** (hoặc `Project-defined GenImage smoke subset`). Tập con này hoàn toàn kế thừa điều khoản `CC BY-NC-SA 4.0 with additional dataset terms` và chỉ được lưu trữ tại `data/research/`.

---

## 4. Cổng Phê duyệt Bước sang Phase 4A (Approval Gate)

Giai đoạn Phase 4A chỉ được kích hoạt khi và chỉ khi:

1. **Người dùng phê duyệt chính thức bằng văn bản** chọn phương án dữ liệu:
   - *Phương án 1 (Khuyến nghị tuyệt đối)*: Sử dụng **Synthetic Smoke Fixture** tự sinh bằng code trong `ml/tests/fixtures/generated-smoke/` để kiểm chứng kỹ thuật toàn bộ pipeline (split, train step, export, registry, wasm worker) với **0 byte tải mạng và 0 rủi ro pháp lý**.
   - *Phương án 2*: Phê duyệt trích xuất một **Custom smoke subset sampled from GenImage** (ước tính 30–50 MB, lưu tại `data/research/`) cho Research Track kèm cam kết không đưa trọng số vào sản phẩm.
   - *Phương án 3*: Phê duyệt tải tập dữ liệu đầy đủ (>5 GB) cho nghiên cứu học thuật khi có hạ tầng GPU.
2. **Xác nhận hạn mức lưu trữ và mạng**: Người dùng chấp thuận dung lượng tải và phân vùng lưu trữ local.
3. **Cam kết phân loại kết quả**: Mọi kết quả đo đạc từ tập smoke test bắt buộc phải gắn nhãn `pipeline-only`, không được công bố là hiệu năng thực của sản phẩm `Forensics Web Lab`.
