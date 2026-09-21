# Data Licenses, Intellectual Property & Track Governance

> **Tài liệu kiểm toán bản quyền dữ liệu và cách ly luồng**: `forensics-web-lab`  
> **Cập nhật chính thức**: Phase 4A.0  
> **Nguyên tắc cốt lõi**: Tách bạch 100% giữa **Research Track** (nghiên cứu học thuật) và **Product Track** (sản phẩm thương mại / web client).

---

## 1. Chính sách Hai Luồng Dữ liệu (Dual-Track Policy)

Dự án áp dụng quy định cách ly tuyệt đối giữa nghiên cứu khoa học và sản phẩm thương mại:

| Tiêu chí | Research Track | Product Track |
| :--- | :--- | :--- |
| **Mục đích** | Thí nghiệm khoa học, benchmark so sánh, kiểm chứng thuật toán. | Tích hợp vào ứng dụng web, phân phối trọng số ONNX đến người dùng cuối. |
| **Dữ liệu cho phép** | Dataset học thuật phi thương mại (kể cả có điều khoản hạn chế NC/SA). | Dữ liệu do dự án tự sở hữu hoặc có giấy phép rõ ràng cho phép thương mại hóa và phân phối trọng số phái sinh. |
| **Trọng số mô hình** | Lưu trong `models/research/`; **nghiêm cấm** đưa vào production bundle. | Lưu trong `models/product/`; phải vượt qua đầy đủ 17 tiêu chí tại [MODEL_ACQUISITION_GATE.md](MODEL_ACQUISITION_GATE.md). |
| **Ranh giới** | Không làm nhiễm dữ liệu hoặc trọng số sang Product Track. | Phải có đầy đủ provenance, license evidence và checksum lưu trong sổ đăng ký. |

---

## 2. Thẩm định Giấy phép Dataset Chi tiết (Dataset License Audit)

Mỗi tập dữ liệu được thẩm định trực tiếp từ nguồn chính thức của tác giả (không sử dụng thông tin từ blog hay bên thứ ba):

### 2.1 GenImage
* **Tác giả / Công bố**: Zhu et al. (NeurIPS 2023)
* **Nguồn chính thức**: [GitHub: GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
* **Văn bản giấy phép chính thức**: [GenImage License File](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
* **Giấy phép mã nguồn (Code License)**: Apache-2.0
* **Giấy phép dữ liệu (Dataset License)**: `CC BY-NC-SA 4.0 with additional dataset terms`
* **Điều khoản bổ sung bắt buộc**:
  1. Chỉ được dùng cho mục đích phi thương mại (nghiên cứu khoa học, giảng dạy, công bố học thuật).
  2. Nghiêm cấm sử dụng tập dữ liệu và các **sản phẩm phái sinh (derivative works)** cho bất kỳ mục đích thương mại nào.
  3. Bắt buộc bảo lưu ghi công (attribution) và chia sẻ tương tự (share-alike).
* **Quyền phân phối trọng số phái sinh**: `prohibited` đối với sản phẩm thương mại; chỉ được dùng nội bộ trong nghiên cứu.
* **Phân loại luồng**: **`research-only`**. Tuyệt đối không đưa mô hình huấn luyện từ GenImage vào Product Track.
* **Lưu ý về tập con**: Không tồn tại dataset chính thức tên "GenImage Mini". Mọi tập con mẫu nhỏ dùng để thử nghiệm kỹ thuật được định danh là **`Custom smoke subset sampled from GenImage`** (hoặc `Project-defined GenImage smoke subset`), vẫn kế thừa đầy đủ điều khoản `CC BY-NC-SA 4.0 with additional dataset terms`.

### 2.2 RealHD
* **Tác giả / Công bố**: ACM Multimedia 2025 Paper
* **Trang dự án**: [real-hd.github.io](https://real-hd.github.io)
* **Kho mã nguồn**: [GitHub: Hanzhe-yu/RealHD](https://github.com/Hanzhe-yu/RealHD)
* **Tình trạng khả dụng (Availability)**: `unavailable-or-pending` (Kho mã nguồn chính thức tại thời điểm kiểm tra hiển thị trạng thái "Coming soon").
* **Giấy phép mã nguồn**: `unverified`
* **Giấy phép dữ liệu**: `unverified` (Chưa công bố văn bản giấy phép chính thức).
* **Quyền phân phối trọng số**: `unclear`
* **Quyết định thẩm định**: **`blocked`**. Không tự ý suy đoán RealHD là CC-BY-NC hay bất kỳ giấy phép nào khi tác giả chưa phát hành chính thức.

### 2.3 SAGI-D (Synthetic and AI-Generated Inpainting Dataset)
* **Tác giả / Tổ chức**: MEVER Team
* **Kho mã nguồn**: [GitHub: mever-team/SAGI](https://github.com/mever-team/SAGI)
* **Kênh phân phối dữ liệu**: Kaggle ([sagitdataset/sagi-d](https://www.kaggle.com/datasets/sagitdataset/sagi-d))
* **Giấy phép mã nguồn**: MIT License
* **Giấy phép dữ liệu**: Phi thương mại / Nghiên cứu học thuật (`unverified` về quyền phân phối lại ma trận trọng số phái sinh).
* **Quyền phân phối trọng số**: `unclear`
* **Phân loại luồng**: **`research-only`**. Cần thẩm định thêm trước khi sử dụng cho bất kỳ mục đích nào ngoài đo lường inpainting mask.

### 2.4 RAID Benchmark
* **Tác giả / Tổ chức**: Academic Consortium
* **Kho mã nguồn**: [GitHub: raid-benchmark/raid](https://github.com/raid-benchmark/raid)
* **Trang dự án**: [raid-benchmark.com](https://raid-benchmark.com)
* **Giấy phép mã nguồn**: Apache-2.0
* **Giấy phép dữ liệu**: Đánh giá nghiên cứu phi thương mại (tổng hợp từ nhiều mô hình sinh ảnh công cộng).
* **Quyền phân phối trọng số**: `unclear`
* **Phân loại luồng**: **`research-only`**. Chỉ dùng làm tập kiểm thử held-out đánh giá độ bền vững, không dùng để huấn luyện.

---

## 3. Bảng Tổng hợp Thẩm định Giấy phép

| Dataset ID | Nguồn Chính thức | Code License | Dataset License | Thương mại? | Trọng số phái sinh | Khả dụng | Quyết định Track |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `genimage` | GitHub / Paper | Apache-2.0 | `CC BY-NC-SA 4.0 with additional dataset terms` | Cấm (`prohibited`) | Cấm thương mại (`prohibited`) | Khả dụng (`available`) | **`research-only`** |
| `realhd` | GitHub / Project page | `unverified` | `unverified` | Chưa rõ (`unclear`) | Chưa rõ (`unclear`) | Chưa phát hành (`pending`) | **`blocked`** |
| `sagi-d` | GitHub / Kaggle | MIT | Research / Academic (`unverified`) | Cấm (`prohibited`) | Chưa rõ (`unclear`) | Khả dụng (`available`) | **`research-only`** |
| `raid` | GitHub / Project page | Apache-2.0 | Research Evaluation Only | Cấm (`prohibited`) | Chưa rõ (`unclear`) | Khả dụng (`available`) | **`research-only`** |
| `synthetic-smoke` | Mã nguồn nội bộ repo | Project-Internal | Project-Internal | Cho phép (`allowed`) | Cho phép (`allowed`) | Khả dụng (`available`) | **`product-eligible`** |

---

## 4. Giao thức Tải Dữ liệu và Kiểm soát Ô nhiễm (Contamination Gatekeeper)

1. **Không tải tự động**: Mọi script tải dữ liệu trong `ml/datasets/` mặc định chỉ chạy ở chế độ `--dry-run`.
2. **Cấm vượt luồng (Strict Track Isolation)**:
   - Tuyệt đối từ chối nạp dataset `research-only` vào cấu hình huấn luyện `product`.
   - Tuyệt đối từ chối đăng ký checkpoint bắt nguồn từ `research-only` vào `models/registry.json` của sản phẩm.
3. **Điều kiện tải thật (Future Execution)**:
   - Phải có cờ `--execute`.
   - Phải gõ xác nhận chính xác mã giấy phép qua `--accept-license <exact-license-id>`.
   - Phải khai báo ngưỡng dung lượng tối đa qua `--expected-bytes <number>`.
