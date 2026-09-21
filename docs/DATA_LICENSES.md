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
* **Tình trạng trọng số phái sinh và cách diễn đạt chuẩn mực**:
  - `source_terms.dataset_derivative_works`: `prohibited_for_commercial_use`
  - `legal_interpretation.trained_weights_status`: `unclear`
  - `project_policy.production_use`: `prohibited`
  - `project_policy.research_use`: `allowed_subject_to_terms`
  - `productionPromotion`: `prohibited-by-project-policy`
  > *GenImage cấm sử dụng thương mại dataset và derivative works. Việc checkpoint có được xem là derivative work hay không chưa được dự án xác lập bằng ý kiến pháp lý. Chính sách bảo thủ của dự án là giữ toàn bộ checkpoint học từ GenImage trong research track.*
* **Phân loại luồng**: **`research-only`** (`purpose: scientific-benchmark`). Tuyệt đối không đưa mô hình huấn luyện từ GenImage vào Product Track.
* **Lưu ý về tập con**: Không tồn tại dataset chính thức tên "GenImage Mini". Mọi tập con mẫu nhỏ dùng để thử nghiệm kỹ thuật được định danh là **`Custom smoke subset sampled from GenImage`** (hoặc `Project-defined GenImage smoke subset`), vẫn kế thừa đầy đủ điều khoản `CC BY-NC-SA 4.0 with additional dataset terms`.

### 2.2 RealHD
* **Tác giả / Công bố**: ACM Multimedia 2025 Paper
* **Trang dự án**: [real-hd.github.io](https://real-hd.github.io)
* **Kho mã nguồn**: [GitHub: Hanzhe-yu/RealHD](https://github.com/Hanzhe-yu/RealHD)
* **Tình trạng khả dụng (Availability)**: `unavailable-or-pending` (Kho mã nguồn chính thức tại thời điểm kiểm tra hiển thị trạng thái "Coming soon").
* **Giấy phép mã nguồn**: `unverified`
* **Giấy phép dữ liệu**: `unverified` (Chưa công bố văn bản giấy phép chính thức).
* **Quyền phân phối trọng số**: `unclear`
* **Quyết định thẩm định**: **`blocked`** (`status: blocked`, `acquisitionEnabled: false`). Không tự ý suy đoán RealHD là CC-BY-NC hay bất kỳ giấy phép nào khi tác giả chưa phát hành chính thức.

### 2.3 SAGI-D (Synthetic and AI-Generated Inpainting Dataset)
* **Tác giả / Tổ chức**: MEVER Team
* **Kho mã nguồn**: [GitHub: mever-team/SAGI](https://github.com/mever-team/SAGI)
* **Kênh phân phối dữ liệu**: Kaggle ([sagitdataset/sagi-d](https://www.kaggle.com/datasets/sagitdataset/sagi-d))
* **Giấy phép mã nguồn**: MIT License
* **Giấy phép dữ liệu**: Phi thương mại / Nghiên cứu học thuật (`unverified` về bằng chứng giấy phép chính thức và quyền phân phối lại ma trận trọng số phái sinh).
* **Quyền phân phối trọng số**: `unclear`
* **Quyết định thẩm định**: **`blocked`** (`status: proposed`, `licenseStatus: unverified`, `acquisitionEnabled: false`). Bị khóa cho đến khi thu thập đủ URL bằng chứng bản quyền chính thức.

### 2.4 RAID Benchmark
* **Tác giả / Tổ chức**: Academic Consortium
* **Kho mã nguồn**: [GitHub: raid-benchmark/raid](https://github.com/raid-benchmark/raid)
* **Trang dự án**: [raid-benchmark.com](https://raid-benchmark.com)
* **Giấy phép mã nguồn**: Apache-2.0
* **Giấy phép dữ liệu**: Đánh giá nghiên cứu phi thương mại (`unverified` về quyền phân phối lại dữ liệu).
* **Quyền phân phối trọng số**: `unclear`
* **Quyết định thẩm định**: **`blocked`** (`status: proposed`, `licenseStatus: unverified`, `acquisitionEnabled: false`). Bị khóa cho đến khi thu thập đủ URL bằng chứng bản quyền chính thức.

### 2.5 Synthetic Smoke Fixture
* **Nguồn**: Mã nguồn dự án tại `ml/tests/fixtures/smoke_generator.py`
* **Quyền sở hữu (Ownership)**: `project-generated`
* **Trạng thái giấy phép**: `pending-project-license-decision`
* **Mục đích thương mại**: `internal-testing-only`
* **Phân loại luồng**: **`fixture-only`** (`purpose: fixture`).
* **Phạm vi áp dụng**:
  - Cho phép: Kiểm tra manifest, kiểm tra loader, kiểm tra split, kiểm tra forward/backward kỹ thuật, kiểm tra export contract.
  - Cấm: Không tham gia huấn luyện checkpoint sản phẩm, không đánh giá accuracy hay năng lực phát hiện AI thực tế, không thuộc product model lineage.

---

## 3. Bảng Tổng hợp Thẩm định Giấy phép & Phân luồng Mục đích

| Dataset ID | Purpose | Nguồn Chính thức | Code License | Dataset License | Thương mại? | Trọng số phái sinh | Production Promotion | Quyết định Track |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `genimage` | `scientific-benchmark` | GitHub / Paper | Apache-2.0 | `CC BY-NC-SA 4.0 with terms` | Cấm (`prohibited`) | Chưa rõ (`unclear`) | `prohibited-by-project-policy` | **`research-only`** |
| `realhd` | `scientific-benchmark` | GitHub / Project page | `unverified` | `unverified` | Chưa rõ (`unclear`) | Chưa rõ (`unclear`) | `prohibited-by-project-policy` | **`blocked`** |
| `sagi-d` | `scientific-benchmark` | GitHub / Kaggle | MIT | `unverified` | Cấm (`prohibited`) | Chưa rõ (`unclear`) | `prohibited-by-project-policy` | **`blocked`** |
| `raid` | `scientific-benchmark` | GitHub / Project page | Apache-2.0 | `unverified` | Cấm (`prohibited`) | Chưa rõ (`unclear`) | `prohibited-by-project-policy` | **`blocked`** |
| `synthetic-smoke` | `fixture` | Mã nguồn nội bộ repo | Project-Internal | `pending-project-license-decision` | Nội bộ (`internal-testing-only`) | Chưa rõ (`unclear`) | `prohibited-by-project-policy` | **`fixture-only`** |

---

## 4. Giao thức Tải Dữ liệu và Kiểm soát Ô nhiễm (Contamination Gatekeeper)

1. **Mặc định không tải nội dung**: Mọi script tải dữ liệu trong `ml/datasets/acquire.py` chỉ hoạt động ở chế độ `--metadata-only` hoặc `--dry-run`.
2. **Khóa tải ngoài (External Acquisition Lock)**:
   - Toàn bộ dataset ngoài duy trì `acquisitionEnabled: false` và `approvalStatus: pending-user-approval`.
   - Lệnh gọi tải với `--execute` trả về: `Acquisition is prepared. User approval with exact archive and byte size is required.`
3. **Thực thi cục bộ (Local Fixture Generation)**:
   - Riêng dataset `synthetic-smoke` với cờ `--execute` thực hiện sinh dữ liệu thử nghiệm cục bộ thuần túy từ code:
     - `operation: local deterministic fixture generation`
     - `networkRequests: 0`
     - `externalBytesDownloaded: 0`
     - `classification: local-test-execution`
4. **Mức độ bảo vệ của `.gitignore`**:
   - Khẳng định bảo vệ được: **verified against the tested extension and path matrix**.
   - Ma trận phần mở rộng đã kiểm thử: `.png`, `.jpg`, `.jpeg`, `.webp`, `.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`, `.zip`, `.tar.gz`.
   - Ma trận đường dẫn đã kiểm thử: `data/research/*`, `data/product/*`, `models/product/*`, `models/research/*`, `artifacts/research/*`, `artifacts/product/*`, `ml/tests/fixtures/generated-smoke/*`.

