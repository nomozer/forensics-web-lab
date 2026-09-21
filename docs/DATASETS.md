# Dataset Specification & Provenance Protocol: Forensics Web Lab

> **Dự án**: `forensics-web-lab`  
> **Phiên bản cập nhật**: Phase 4A.2 (Scope Freeze & TGIF Integration)  
> **Nguyên tắc**: Cách ly tuyệt đối dữ liệu nghiên cứu và dữ liệu sản phẩm (Dual-Track ADR-0006); kiểm soát rò rỉ ranh giới (`source_id`); cấm commit binary ảnh vào Git; 0 byte external data khi chưa có người dùng phê duyệt.

---

## 1. Chính sách Quản lý Dữ liệu và Phân luồng Kiến trúc (Dual-Track Isolation)

Hệ thống thiết lập phân định rõ ràng giữa hai vùng dữ liệu vật lý và pháp lý:

```text
data/
├── research/              # CHỈ DÀNH CHO NGHIÊN CỨU: Dataset học thuật phi thương mại (CC-BY-NC-SA, CC-BY-SA, v.v.)
│   └── README.md          # Lưu ý pháp lý và hướng dẫn nạp dữ liệu nghiên cứu
└── product/               # DÀNH CHO SẢN PHẨM: Dữ liệu tự sở hữu / có quyền thương mại và xuất bản trọng số
    └── README.md          # Điều kiện nghiệm thu dữ liệu sản phẩm
```

### Quy tắc bất biến
1. **Quarantine khỏi Git**: Thư mục `data/research/` và `data/product/` bị chặn hoàn toàn trong `.gitignore`. Không commit bất kỳ file ảnh hoặc archive nào.
2. **Không nhiễm chéo (Zero Cross-Contamination)**: Dữ liệu trong `data/research/` tuyệt đối không được tham gia vào pipeline huấn luyện sản phẩm (`models/product/`).
3. **Thẩm định nguồn chính thức**: Chỉ tiếp nhận thông tin từ repository hoặc website chính thức của tác giả; không dùng nguồn thứ cấp để suy diễn giấy phép.
4. **Human Gatekeeper**: Bắt buộc có sự chấp thuận tường minh của người dùng trước khi tải dữ liệu thật.

---

## 2. Danh mục Dữ liệu Khảo sát (Dataset Catalog)

### 2.1 GenImage
* **Mục đích**: Nghiên cứu phát hiện toàn ảnh nhân tạo từ nhiều bộ tạo sinh khác nhau (`purpose: scientific-benchmark`).
* **Nguồn chính thức**: [GitHub: GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
* **Văn bản giấy phép**: [GenImage License](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
* **Giấy phép chính thức**: `CC BY-NC-SA 4.0 with additional dataset terms`.
* **Hạn chế pháp lý**: Chỉ dùng cho mục đích phi thương mại (nghiên cứu, giảng dạy). Nghiêm cấm sử dụng dataset và **sản phẩm phái sinh (derivative works)** cho mục đích thương mại.
* **Tình trạng trọng số phái sinh và chính sách dự án**:
  - `source_terms.dataset_derivative_works`: `prohibited_for_commercial_use`
  - `legal_interpretation.trained_weights_status`: `unclear`
  - `project_policy.production_use`: `prohibited`
  - `project_policy.research_use`: `allowed_subject_to_terms`
  - `productionPromotion`: `prohibited-by-project-policy`
  > *GenImage cấm sử dụng thương mại dataset và derivative works. Việc checkpoint có được xem là derivative work hay không chưa được dự án xác lập bằng ý kiến pháp lý. Chính sách bảo thủ của dự án là giữ toàn bộ checkpoint học từ GenImage trong research track.*
* **Phân loại luồng**: **`research-only`** (Lưu tại `data/research/genimage/`).
* **Khảo sát Metadata**: Lưu trữ theo chuỗi nén multi-part zip theo từng generator. Thư mục BigGAN là archive đầu tiên được kiểm kê toàn diện (*first fully inventoried archive*): 23,516,377,048 bytes nén, ước tính ~26 GB giải nén, ~16k ảnh.

### 2.2 TGIF & TGIF2 (Text-Guided Inpainting Forgery Dataset)
* **Mục đích**: Nghiên cứu phát hiện và định vị inpainting cục bộ có hướng dẫn bằng văn bản (`ai_edited`, `purpose: scientific-benchmark`).
* **Nguồn chính thức**: [GitHub: IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) | [Paper WIFS 2024](https://arxiv.org/abs/2407.11566) | [Paper JIS 2026](https://arxiv.org/abs/2603.28613)
* **Giấy phép dataset**: `CC BY-SA 4.0` (tác giả phân phối công khai).
* **Nguồn ảnh gốc**: MS-COCO (`CC BY 4.0`).
* **Bộ tạo sinh**: SD2, SDXL, Adobe Photoshop/Firefly (TGIF) và FLUX.1 schnell/dev/filldev (TGIF2).
* **Ground-truth masks**: Cung cấp đầy đủ mask nhị phân (segmentation, bounding box, random rectangle).
* **Khảo sát Metadata Nextcloud**:
  - TGIF: 65.4 GB (masks: 40.4 MB, orig: 6.8 GB, ps-sp: 16.8 GB, sd2-fr: 7.1 GB, sd2-sp: 17.3 GB, sdxl-fr: 17.3 GB).
  - TGIF2 FLUX: 110 GB (8 thư mục: masks-flux: 7.7 MB, orig-flux: 5.7 GB, các thư mục FLUX 16–18 GB).
  - TGIF2 random: 73 GB (13 thư mục: masks 900 KB, metadata: 1.8 MB, model subfolders).
* **Khả năng tải độc lập**: Nextcloud public shares hỗ trợ tải độc lập từng thư mục con dưới dạng zip nén động (ví dụ: chỉ tải `masks` 40.4 MB, hoặc `orig` 6.8 GB) mà không bắt buộc tải toàn bộ 65.4 GB.
* **Hiện trạng thu nạp (Phase 4B.0 Smoke)**:
  - `tgif-masks`: **Đã tải và kiểm toán thành công** (`status: verified`). Tải về: 42,327,429 bytes (~40.37 MiB, SHA-256 archive `62c89a65...`). Giải nén an toàn: 141,559,934 bytes across 31,238 mask PNG files (12,495 bbox, 12,495 segm, 6,248 generic_mask) trên 2,242 MS-COCO `source_id`. Manifest lưu tại `data/research/tgif/manifests/masks-manifest.jsonl`, biên nhận tại `data/research/tgif/acquisition-receipt.json`.
  - `tgif-orig` (~7.30 GB) và `tgif-sd2-sp` (~18.58 GB): **Chưa tải, tiếp tục bị khóa** (`status: locked`, chờ phê duyệt cho Phase 4B.1+).
* **Phân loại luồng**: **`research-only`** (`derivativeWeights: unclear`, `productionPromotion: prohibited-by-project-policy` do điều khoản Share-Alike).

### 2.3 RealHD
* **Mục đích**: Khảo sát phát hiện can thiệp cục bộ độ phân giải cao (`purpose: scientific-benchmark`).
* **Nguồn chính thức**: [real-hd.github.io](https://real-hd.github.io) | [GitHub: Hanzhe-yu/RealHD](https://github.com/Hanzhe-yu/RealHD)
* **Tình trạng khả dụng**: `unavailable-or-pending` ("Coming soon" trên GitHub).
* **Giấy phép**: `unverified`.
* **Quyết định**: **`blocked`** (`status: blocked`, `acquisitionEnabled: false`). Khóa hoàn toàn cho đến khi có bản phát hành chính thức.

### 2.4 SAGI-D (Synthetic and AI-Generated Inpainting Dataset)
* **Mục đích**: Nghiên cứu phát hiện và định vị inpainting cục bộ (`ai_edited`, `purpose: scientific-benchmark`).
* **Nguồn chính thức**: [GitHub: mever-team/SAGI](https://github.com/mever-team/SAGI)
* **Giấy phép**: `unverified` về bằng chứng phân phối lại ảnh gốc.
* **Quyết định**: **`blocked`** (`status: proposed`, `licenseStatus: unverified`, `acquisitionEnabled: false`).

### 2.5 RAID Benchmark
* **Mục đích**: Đánh giá độ bền vững đối kháng và suy giảm chất lượng (`purpose: scientific-benchmark`).
* **Nguồn chính thức**: [raid-benchmark.com](https://raid-benchmark.com) | [GitHub: raid-benchmark/raid](https://github.com/raid-benchmark/raid)
* **Giấy phép**: Mã nguồn Apache-2.0, nhưng bản quyền hình ảnh chưa có dataset card chứng minh.
* **Quyết định**: **`blocked`** (`status: proposed`, `licenseStatus: unverified`, `acquisitionEnabled: false`).

### 2.6 Synthetic Smoke Fixture (Dữ liệu Giả lập Nội bộ)
* **Mục đích**: Kiểm thử thông suốt pipeline kỹ thuật (manifest, split, loader, export contract) mà không phụ thuộc dữ liệu bên ngoài (`purpose: fixture`).
* **Nguồn gốc**: Sinh 100% bằng script nội bộ (`ml/tests/fixtures/generated-smoke/`).
* **Đặc tính**: Hình học, gradient màu và nhiễu toán học; **tuyệt đối không phải ảnh chụp thật**.
* **Phân loại luồng**: **`fixture-only`** (`commercialUse: internal-testing-only`). Không tham gia huấn luyện sản phẩm hay đo accuracy.

---

## 3. Ma trận Nghiên cứu – Dữ liệu (Research-Data Matrix)

| Research Question | Task | Candidate Dataset | Required Labels | Split Strategy | Metric | Main Risk | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ1 (3-Class Generalization)** | 3-Class Classification | **GenImage** + **TGIF** | `authentic`, `fully_generated`, `ai_edited` | Group-held-out by `source_id` + Unseen generator holdout | Macro-F1, Balanced Accuracy, Per-Class F1, AUROC | Dataset-source shortcut learning | Planning (0 bytes downloaded) |
| **RQ2 (Multi-Modal Fusion)** | Fusion & Calibration | **GenImage** + **TGIF** + Heuristics | 3 classes + DSP/C2PA signals | Validation calibration split | ECE, Brier Score, NLL, Macro-F1 gain | Overfitting to calibration split | Planning |
| **RQ3 (ONNX Quantization)** | Quantization Parity | **GenImage** / **TGIF** test split | 3 classes | Held-out Test split | Macro-F1 drop, Model size, $L_\infty$ parity | INT8 accuracy degradation | Planning |
| **RQ4 (Browser Runtime)** | In-Browser Execution | Standard test samples | 3 classes | 50 benchmark iterations | Latency (median/P95), Peak RAM, WASM size | Browser out-of-memory under high res | Planning |
| **Auxiliary RQ5 (Localization)** | Inpainting Localization | **TGIF** (`masks`, `ps-sp`, `sd2-sp`) | Binary Ground-Truth Mask | Source-held-out pairs | mIoU, Dice/F1 mask, Pixel AUROC | Patch heatmap resolution mismatch | Planning |

---

## 4. Nguy cơ Shortcut Nguồn Dữ liệu và Biện pháp Kiểm soát

> **Nguy cơ Dataset-Source Shortcut**:  
> Nếu mỗi lớp nhãn được lấy từ một dataset khác nhau (ví dụ: `authentic` từ ImageNet, `fully_generated` từ GenImage, `ai_edited` từ TGIF/MS-COCO), mô hình deep learning có thể học đặc trưng riêng của nguồn dữ liệu (profile nén, kích thước gốc, color tone, cảm biến camera) thay vì học dấu vết tạo sinh AI.

### Biện pháp kiểm soát bắt buộc trong thiết kế thử nghiệm:
1. **Group Split theo ảnh gốc (`source_id`)**: Mọi biến thể từ cùng một ảnh gốc luôn được giữ trọn vẹn trong cùng một split (Train, Val hoặc Test).
2. **Dùng Matched Pairs**: Trong TGIF, sử dụng chính các cặp ảnh gốc authentic (MS-COCO) đi kèm với các biến thể inpainting của chính ảnh đó.
3. **Theo dõi đa chiều (Multi-Factor Tracking)**: Manifest ghi nhận rõ `dataset_id`, `generator_family`, `generator_version`, `edit_method`.
4. **Chuẩn hóa Tiền xử lý**: Mọi ảnh đều được đưa về cùng một chuẩn tiền xử lý đồng nhất (letterbox padding, resize, chuẩn hóa kênh màu).
5. **Đánh giá Source-Held-Out & Cross-Dataset**: Báo cáo kết quả phân tách theo từng generator và từng nguồn dữ liệu để phát hiện rớt độ chính xác.

---

## 5. Đề xuất Phương án Thử nghiệm Pilot (docs/PILOT_PROTOCOL.md) — Chờ Duyệt

Dựa trên kết quả kiểm toán ngữ nghĩa nhãn tại Phase 4A.3, thành phần `fr` (fully regenerated) của TGIF **không được gán nhãn là `fully_generated`** mà phải thuộc `ai_edited` hoặc cách ly. Để giảm đáng kể nguy cơ shortcut nguồn dữ liệu, dự án phân tách thành kiến trúc hai nhánh độc lập trước khi mở rộng sang 3 lớp:

### Phương án Pilot A — Authentic vs AI-Edited & Localization (TGIF Matched Pairs)
* **Cấu hình máy đọc**: `ml/configs/pilot_tgif_edit.yaml`
* **Nguồn dữ liệu**: TGIF Nextcloud (`masks` 40.4 MB, `orig` 6.8 GB, `sd2-sp` 17.3 GB).
* **Dung lượng nén**: 25,919,643,647 bytes (~24.1 GB).
* **Dung lượng giải nén ước tính**: ~27 GB.
* **Ổ đĩa trống yêu cầu**: $\ge 55\text{ GB}$.
* **Số ảnh**: 3,124 authentic (verified) + 18,744 inpaintings (verified trong `sd2-sp`, thuộc 74,976 tổng toàn bộ TGIF) + ~6,248 binary masks (ước tính 2 mask per source image: segm & bbox).
* **Ưu điểm khoa học**: Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec, quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh vẫn phải được đo bằng baseline và source-held-out evaluation.
* **Mục tiêu khoa học**: Phân loại nhị phân `authentic` vs `ai_edited` và định vị vùng chỉnh sửa với ground-truth mask (Macro-F1, Balanced Acc, mIoU, Dice, Pixel AUROC).
* **Phương án rút gọn tối thiểu (Low-Bandwidth Option)**: Tải trước `masks` (40.4 MB) + `orig` (6.8 GB) = 6.84 GB để kiểm thử pipeline định vị trước khi tải `sd2-sp`.

### Phương án Pilot B — Authentic vs Fully-Generated (GenImage Controlled Pairs)
* **Cấu hình máy đọc**: `ml/configs/pilot_genimage_generated.yaml`
* **Nguồn dữ liệu**: GenImage BigGAN split archive (`imagenet_ai_0419_biggan.z01` – `.zip`).
* **Dung lượng nén**: 23,516,377,048 bytes (~21.9 GB / ~24 GB).
* **Dung lượng giải nén ước tính**: ~26 GB.
* **Ổ đĩa trống yêu cầu**: $\ge 60\text{ GB}$.
* **Số ảnh**: ~16,000–20,000 ảnh (~2,000 ảnh trong validation split cân bằng 1,000 real / 1,000 fake).
* **Ưu điểm khoa học**: Phân loại nhị phân trong cùng phân phối ImageNet của GenImage tác giả.
* **Mục tiêu khoa học**: Phân loại `authentic` vs `fully_generated` và đo lường suy giảm khi kiểm thử trên generator chưa thấy (cross-generator drop).

### Phương án Pilot C — Thử nghiệm Khám phá Ba Lớp (Three-Class Exploratory)
* **Điều kiện mở cổng**: Chỉ thực hiện sau khi Pilot A và Pilot B vượt qua bài kiểm toán rò rỉ và shortcut.
* **Nguồn dữ liệu**: Kết hợp `authentic` (TGIF `orig`), `fully_generated` (GenImage BigGAN), `ai_edited` (TGIF `sd2-sp`).
* **Ràng buộc khoa học**: Bắt buộc huấn luyện kèm **Metadata-Only Baseline Guard** (đo lường khả năng đoán nhãn chỉ từ resolution, aspect ratio, file size, codec). Đánh giá qua Paired Stratified Bootstrap 95% CI; chỉ nghiệm thu khi cận dưới 95% CI của $\Delta\text{Macro-F1} > 0.0$.
* **Trạng thái**: Gắn nhãn bắt buộc là `exploratory pilot`, chưa dùng làm kết luận khẳng định cho đến khi kiểm chứng cross-dataset.

---

### Kết luận Khuyến nghị (Recommended Acquisition Strategy):
> **Dự án đề xuất lựa chọn Phương án Pilot A làm bước tải đầu tiên** vì:
> 1. TGIF Nextcloud hỗ trợ tải riêng từng thư mục con độc lập.
> 2. Cặp ảnh gốc MS-COCO và phiên bản inpainting `sd2-sp` chia sẻ cùng source image, giảm đáng kể nguy cơ học đặc trưng nguồn ảnh khác nhau.
> 3. Cung cấp đồng thời ground-truth binary mask cho bài toán định vị (localization).
> 4. Nếu người dùng muốn tối thiểu hóa lần tải đầu, có thể phê duyệt gói rút gọn `masks` (40.4 MB) + `orig` (6.8 GB) = 6.84 GB. Hoặc phê duyệt toàn bộ Pilot A (~24.1 GB nén, yêu cầu $\ge 55\text{ GB}$ đĩa trống).
