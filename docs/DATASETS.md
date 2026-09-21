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

## 5. Đề xuất Phương án Thử nghiệm (Pilot Proposals) — Chờ Duyệt

Dự án chuẩn bị 3 phương án quy mô thử nghiệm để người dùng phê duyệt trước khi tải dữ liệu thật:

### Phương án Pilot A — Pipeline Smoke (Kiểm thử Kỹ thuật)
* **Sample Count**: 10–50 ảnh.
* **Compressed Bytes**: 0 byte (dùng `synthetic-smoke` nội bộ) hoặc ~40.4 MB (chỉ tải thư mục `masks` của TGIF).
* **Estimated Extracted Bytes**: ~50 MB.
* **Required Disk Space**: $\ge 1\text{ GB}$.
* **Required Compute**: CPU thông thường (không cần GPU).
* **Expected Runtime**: $< 1$ phút.
* **License**: Project-Internal / CC BY-SA 4.0.
* **Scientific Purpose**: `pipeline-only` (kiểm tra acquisition adapter, mã băm SHA-256, manifest parser, DataLoader batching). Không tạo metric khoa học.
* **Limitations**: Không có giá trị đánh giá độ chính xác hay khả năng phát hiện AI.

### Phương án Pilot B — Exploratory Three-Class (Đề xuất Khuyến nghị cho Khóa luận)
* **Sample Count**: ~3,000 ảnh (cân bằng 3 lớp: ~1,000 `authentic`, ~1,000 `fully_generated`, ~1,000 `ai_edited`).
  - `authentic`: Lấy từ TGIF `orig` (MS-COCO camera authentic).
  - `ai_edited`: Lấy từ TGIF `sd2-sp` hoặc `ps-sp` (cùng source ID với ảnh authentic).
  - `fully_generated`: Lấy từ TGIF `sd2-fr` / `sdxl-fr` hoặc GenImage BigGAN sample.
* **Compressed Bytes**: ~14–15 GB (tải thư mục `masks` 40.4 MB, `orig` 6.8 GB, và `sd2-fr` 7.1 GB từ Nextcloud TGIF).
* **Estimated Extracted Bytes**: ~17–18 GB.
* **Required Disk Space**: $\ge 35\text{ GB}$ trống.
* **Required Compute**: GPU cá nhân (VRAM $\ge 6\text{ GB}$) hoặc Google Colab T4 / CPU đa luồng (chạy trong vài giờ).
* **Expected Runtime**: 2–4 giờ huấn luyện thử nghiệm.
* **License**: CC BY-SA 4.0 (MS-COCO CC BY 4.0) — Quarantined trong Research Track.
* **Scientific Purpose**: `exploratory pilot` (chạy thử nghiệm training loop 3 lớp hoàn chỉnh, đo CPU/GPU time thật, kiểm tra leakage qua group split, kiểm tra loss convergence và xuất ONNX).
* **Limitations**: Số lượng mẫu giới hạn ở mức pilot khám phá; toàn bộ metric gắn nhãn `exploratory, not publication-grade`.

### Phương án Pilot C — Confirmatory Benchmark (Đánh giá Đầy đủ)
* **Sample Count**: Toàn bộ split chính thức (>80,000 ảnh từ GenImage và TGIF).
* **Compressed Bytes**: ~90 GB (BigGAN ~24 GB + TGIF 65.4 GB).
* **Estimated Extracted Bytes**: ~100 GB.
* **Required Disk Space**: $\ge 220\text{ GB}$ trống.
* **Required Compute**: GPU chuyên dụng (NVIDIA A100 / RTX 3090/4090, VRAM $\ge 16\text{ GB}$).
* **Expected Runtime**: 24–48 giờ.
* **License**: CC BY-NC-SA 4.0 / CC BY-SA 4.0 — Research Track.
* **Scientific Purpose**: `scientific-benchmark` (đánh giá chính thức in-domain, cross-generator, unseen generator, calibration ECE, báo cáo khoảng tin cậy 95% phục vụ bài báo khoa học).
* **Limitations**: Đòi hỏi tài nguyên tính toán và lưu trữ vượt quá giới hạn máy cá nhân hiện tại.

---

### Kết luận Khuyến nghị (Single Recommended Pilot):
> **Dự án đề xuất lựa chọn Phương án Pilot B (Exploratory Three-Class)** vì:
> 1. Đây là phương án nhỏ nhất vừa đủ để tạo ra một không gian bài toán **ba lớp cân bằng hoàn chỉnh** (`authentic`, `fully_generated`, `ai_edited`).
> 2. Khai thác tính năng của Nextcloud TGIF cho phép **tải riêng từng thư mục con** (chỉ tải `masks`, `orig` và `sd2-fr` hoặc `sd2-sp`), tiết kiệm hơn 75% băng thông so với việc tải toàn bộ 65.4 GB.
> 3. Kiểm soát được nguy cơ **shortcut learning** nhờ sử dụng matched pairs từ cùng nguồn MS-COCO của TGIF.
> 4. Phù hợp hoàn toàn với giới hạn tài nguyên máy tính và thời gian thực hiện khóa luận.
