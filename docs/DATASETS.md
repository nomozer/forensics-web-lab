# Dataset Specification & Provenance Protocol: Forensics Web Lab

> **Dự án**: `forensics-web-lab`  
> **Phiên bản cập nhật**: Phase 4A.0 (Dual-Track Data Governance)  
> **Nguyên tắc**: Cách ly tuyệt đối dữ liệu nghiên cứu và dữ liệu sản phẩm; kiểm soát rò rỉ ranh giới (`source_id`); cấm commit binary ảnh vào Git.

---

## 1. Chính sách Quản lý Dữ liệu và Phân luồng Kiến trúc (Dual-Track Isolation)

Hệ thống thiết lập phân định rõ ràng giữa hai vùng dữ liệu vật lý và pháp lý:

```text
data/
├── research/              # CHỈ DÀNH CHO NGHIÊN CỨU: Dataset học thuật phi thương mại (CC-BY-NC-SA, v.v.)
│   └── README.md          # Lưu ý pháp lý và hướng dẫn nạp dữ liệu nghiên cứu
└── product/               # DÀNH CHO SẢN PHẨM: Dữ liệu tự sở hữu / có quyền thương mại và xuất bản trọng số
    └── README.md          # Điều kiện nghiệm thu dữ liệu sản phẩm
```

### Quy tắc bất biến
1. **Quarantine khỏi Git**: Thư mục `data/research/` và `data/product/` bị chặn hoàn toàn trong `.gitignore`. Không commit bất kỳ file ảnh hoặc archive nào.
2. **Không nhiễm chéo (Zero Cross-Contamination)**: Dữ liệu trong `data/research/` tuyệt đối không được tham gia vào pipeline huấn luyện sản phẩm (`models/product/`).
3. **Thẩm định nguồn chính thức**: Chỉ tiếp nhận thông tin từ repository hoặc website chính thức của tác giả; không dùng nguồn thứ cấp để suy diễn giấy phép.
4. **Human Gatekeeper**: Bắt buộc có sự chấp thuận của người dùng trước khi tải dữ liệu thật.

---

## 2. Danh mục Dữ liệu Khảo sát (Dataset Catalog)

### 2.1 GenImage
* **Mục đích**: Nghiên cứu phát hiện toàn ảnh nhân tạo từ nhiều bộ tạo sinh khác nhau.
* **Nguồn chính thức**: [GitHub: GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
* **Văn bản giấy phép**: [GenImage License](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
* **Giấy phép chính thức**: `CC BY-NC-SA 4.0 with additional dataset terms`
* **Hạn chế pháp lý**: Chỉ dùng cho mục đích phi thương mại (nghiên cứu, giảng dạy). Nghiêm cấm sử dụng dataset và **sản phẩm phái sinh (derivative works)** cho mục đích thương mại.
* **Quyền phân phối trọng số**: `prohibited` cho bản thương mại.
* **Phân loại luồng**: **`research-only`** (Lưu tại `data/research/genimage/`).
* **Lưu ý quan trọng**: Không tồn tại dataset chính thức tên "GenImage Mini". Mọi tập con mẫu nhỏ do dự án tự lấy mẫu phục vụ thử nghiệm kỹ thuật được gọi là **`Custom smoke subset sampled from GenImage`** (hoặc `Project-defined GenImage smoke subset`), vẫn kế thừa đầy đủ điều khoản `CC BY-NC-SA 4.0 with additional dataset terms`.

### 2.2 RealHD
* **Mục đích**: Khảo sát phát hiện can thiệp cục bộ độ phân giải cao.
* **Trang dự án**: [real-hd.github.io](https://real-hd.github.io)
* **Kho mã nguồn**: [GitHub: Hanzhe-yu/RealHD](https://github.com/Hanzhe-yu/RealHD)
* **Tình trạng khả dụng**: `unavailable-or-pending` (Kho mã nguồn tác giả tại thời điểm kiểm tra hiển thị "Coming soon").
* **Giấy phép**: `unverified` (Chưa công bố văn bản giấy phép chính thức).
* **Quyết định**: **`blocked`**. Nghiêm cấm suy đoán giấy phép khi chưa có văn bản công khai.

### 2.3 SAGI-D (Synthetic and AI-Generated Inpainting Dataset)
* **Mục đích**: Nghiên cứu phát hiện và định vị inpainting cục bộ (`ai_edited`).
* **Nguồn chính thức**: [GitHub: mever-team/SAGI](https://github.com/mever-team/SAGI) và Kaggle.
* **Giấy phép mã nguồn**: MIT | **Giấy phép ảnh**: Phi thương mại / Nghiên cứu học thuật (`unverified` quyền phân phối lại ma trận trọng số).
* **Phân loại luồng**: **`research-only`** (Lưu tại `data/research/sagi-d/`).

### 2.4 RAID Benchmark
* **Mục đích**: Tập đánh giá độ bền vững trước suy giảm và nén thực tế.
* **Nguồn chính thức**: [raid-benchmark.com](https://raid-benchmark.com) | [GitHub: raid-benchmark/raid](https://github.com/raid-benchmark/raid)
* **Giấy phép**: Apache-2.0 (code) / Nghiên cứu học thuật (ảnh).
* **Phân loại luồng**: **`research-only`** (Dành riêng cho held-out evaluation, không dùng để huấn luyện).

### 2.5 Synthetic Smoke Fixture (Dữ liệu Giả lập Nội bộ)
* **Mục đích**: Kiểm thử thông suốt pipeline kỹ thuật (manifest, split, loader, export interface) mà không phụ thuộc dữ liệu bên ngoài.
* **Nguồn gốc**: Sinh 100% bằng script nội bộ dự án (`ml/tests/fixtures/generated-smoke/`).
* **Đặc tính**: Các hình khối hình học, gradient màu và nhiễu toán học đơn giản; **tuyệt đối không phải ảnh chụp thật**.
* **Phân loại luồng**: **`product-eligible`** (về mặt bản quyền kỹ thuật; không được dùng để đo lường độ chính xác).

---

## 3. Đặc tả Manifest Chuẩn hóa và Nguồn gốc (Provenance Contract)

Mỗi mẫu ảnh trong hệ thống bắt buộc phải được theo dõi nguồn gốc qua định dạng manifest (JSON/CSV) chuẩn hóa theo schema `docs/schemas/dataset-manifest.v1.schema.json`:

| Trường dữ liệu | Kiểu dữ liệu | Ý nghĩa và Ràng buộc |
| :--- | :--- | :--- |
| `sample_id` | `string` | Định danh duy nhất toàn cầu cho mẫu ảnh (ví dụ: `genimage_sd15_000123`). Không dùng đường dẫn file làm ID. |
| `dataset_id` | `string` | ID tập dữ liệu xuất xứ (khớp với `datasets/registry.json`, ví dụ: `genimage`). |
| `source_id` | `string` | ID ảnh gốc cha (parent image). Tất cả ảnh biến thể từ cùng ảnh gốc phải có chung `source_id`. |
| `original_filename` | `string` | Tên tệp gốc khi tải về. |
| `sha256` | `string` | Mã băm SHA-256 (64 ký tự hex thường) của tệp ảnh. |
| `label` | `string` | Nhãn phân loại: `authentic`, `fully_generated`, `ai_edited`. |
| `task` | `string` | Nhiệm vụ: `classification` hoặc `localization`. |
| `generator_family` | `string` | Họ mô hình tạo sinh (ví dụ: `stable-diffusion`, `midjourney`, `camera`, `geometric-fixture`). |
| `generator_version` | `string` | Phiên bản bộ tạo (ví dụ: `1.5`, `xl`, `v5`, `none`). |
| `edit_method` | `string` | Phương thức can thiệp: `none`, `full_synthesis`, `inpainting`, `face_swap`, `synthetic_fixture`. |
| `mask_path` | `string` | Đường dẫn tương đối đến mask nhị phân (bắt buộc với `ai_edited` localization, rỗng với toàn cảnh). |
| `license_track` | `string` | Phân luồng bản quyền: `research-only` hoặc `product-eligible`. |
| `split_group` | `string` | Nhóm chia tập: `train`, `val`, `test_indomain`, `test_unseen_generator`, `test_degraded`. |
| `acquired_at` | `string` | Thời điểm nạp dữ liệu (ISO 8601 UTC). |

---

## 4. Giao thức Chống Rò rỉ Dữ liệu (Anti-Leakage Protocol)

1. **Ràng buộc Nhóm `source_id`**:
   - Ảnh gốc và mọi ảnh phái sinh (ảnh inpainting, ảnh crop, ảnh nén JPEG) bắt buộc phải nằm cùng một tập split.
   - Tuyệt đối không cho phép ảnh gốc ở tập `train` trong khi ảnh inpainting của nó ở tập `val` hoặc `test`.
2. **Cách ly Họ Tạo sinh Chưa Thấy (Unseen Generator Isolation)**:
   - Ít nhất một họ tạo sinh độc lập (ví dụ: Midjourney hoặc Wukong) phải được giữ lại riêng cho tập `test_unseen_generator`.
3. **Cấm Pha trộn Luồng (Strict Cross-Track Prohibition)**:
   - Không được phép huấn luyện một checkpoint dùng hỗn hợp mẫu từ cả `research-only` và `product-eligible`.
