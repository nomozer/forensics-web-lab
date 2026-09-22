# Phase 4B.3 — Evidence Reconciliation and Binary Experiment Preregistration

> **Phase**: 4B.3  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting Commit**: `1ad8151`  
> **Verdict**: **`PASS`**  
> **Execution Date**: 2026-09-22  
> **Scope**: Research Track Only — Evidence Reconciliation and Binary Experiment Preregistration  

---

## 1. Executive Summary & Verdict

Phase 4B.3 hoàn thành toàn diện việc đối soát tính nhất quán của bằng chứng Phase 4B.2, kiểm toán chi tiết ngữ nghĩa manifest huấn luyện, thiết lập cơ chế chống ngụy tạo cỡ mẫu (anti-pseudoreplication) và phân quyền chặt chẽ cho locked-test, đồng thời đăng ký trước (preregister) toàn bộ giao thức thí nghiệm phân loại nhị phân cho Phase 4C:

1. **Hiệu chỉnh Tính nhất quán Bằng chứng Phase 4B.2**:
   * Đồng bộ hóa đầy đủ lịch sử commit: starting commit `82a5266`, implementation snapshots `7d33072` & `6bb088b`, ending commit `1ad8151`.
   * Sửa lỗi sao chép nhầm mã băm SHA-256 của archive `sd2-sp_validation.tar.gz` trong báo cáo markdown về giá trị đo đạc thực tế chính xác: `bd9eb4399f60166a09209d8a66a5df2b5eaecfbcc1a9d52f694e6812e24c5ad7`.
   * Khẳng định tính trung thực khoa học: Phase 4B.2 có **0 training run** và **chưa hỗ trợ bất kỳ tuyên bố nào về hiệu năng mô hình**.
2. **Đối soát ETag và Tính toàn vẹn Archive**:
   * Phân định rõ ràng: ETag khảo sát qua WebDAV PROPFIND (Nextcloud tính mã băm MD5) khác biệt với ETag nhận được khi tải trực tiếp qua HTTP GET (Apache sinh mã hex theo inode/mtime của file). Do đó `etag_match: false` và `remote_metadata_changed: true`.
   * Tính toàn vẹn của 4 archive được bảo đảm độc lập tuyệt đối 100% bằng **dung lượng byte count** (5,879,502,782 bytes) và **mã băm SHA-256 cục bộ** đã kiểm chứng.
3. **Kiểm toán Manifest & Phân biệt Đơn vị Mẫu**:
   * Kiểm toán trực tiếp `manifest_pilot_a_option_p.csv`: có đúng **684 dòng**, tương ứng với 684 task instances độc lập (`source_id`). 684/684 đường dẫn authentic và 684/684 đường dẫn canonical edit đều tồn tại, 0 lỗi, 0 trùng lặp, 0 rò rỉ chéo phân vùng.
   * Giải thích toán học chính xác mối liên hệ giữa các con số:
     * **684** instance $\times$ 3 độ phân giải authentic (native, 512, 1024) = **2,052 ảnh authentic**.
     * **684** instance $\times$ 6 biến thể chỉnh sửa (3 bbox + 3 segm) = **4,104 ảnh ai_edited**.
     * Tổng số ảnh giải nén trên đĩa = $2,052 + 4,104 =$ **6,156 ảnh**.
     * Sinh manifest biến thể mở rộng `manifest_pilot_a_variant_pairs.csv` gồm đúng **4,104 cặp mẫu**.
4. **Cơ chế Chống Ngụy tạo Cỡ mẫu (Strategy A: Pair-Aware Sampler)**:
   * Thiết kế và triển khai `ml/datasets/pair_aware_loader.py` theo Phương án A: Mỗi epoch, mỗi `source_id` đóng góp đúng 1 ảnh authentic và 1 ảnh edited (cân bằng 1:1, tổng $2N$ mẫu/epoch).
   * Luân phiên biến thể tất định qua từng epoch: $\text{variant\_idx} = (\text{epoch} + \text{offset}(\text{source\_id})) \pmod 6$.
   * Đơn vị thống kê tối thượng cho primary metrics là **`source_id`**: tổng hợp xác suất dự đoán theo source trước khi tính Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE.
5. **Phân quyền Locked-Test (Role-Based Access Guard)**:
   * Triển khai `ml/evaluation/locked_test_guard.py` thay thế quy tắc chung chung bằng ba quyền truy cập tường minh:
     * `trainingAccessAllowed: false`
     * `developmentAccessAllowed: false`
     * `finalEvaluationAccessAllowed: true only with frozen experiment lock`
   * Yêu cầu ràng buộc mã hóa đầy đủ: `experiment_lock_hash`, `config_hash`, `checkpoint_hash`, `preprocessing_hash`, `threshold_calibration_hash`, danh sách seed, timestamp UTC và đối soát seal `519e7a0e...`.
6. **Đăng ký trước Toàn diện Phase 4C**:
   * Ban hành `docs/PHASE_4C_PREREGISTRATION.md`, `ml/configs/pilot_a_binary_preregistered.yaml` và `docs/schemas/experiment-lock.v1.schema.json`.
   * Khóa chặt bài toán phân loại nhị phân `authentic` vs `ai_edited`; pipeline 3 lớp tiếp tục ở trạng thái `not-runnable-missing-fully-generated-data`.
   * Đăng ký 3 điểm learning curve lồng nhau ($N \in \{50, 100, 250\}$), 6 baselines, 5 seeds, và 95% paired stratified bootstrap CI.
7. **Kế hoạch Pretrained Weights & Tài nguyên**:
   * Lập kế hoạch chi tiết cho MobileNetV3-Small (~10.3 MB, 1.78M backbone params, BSD-3-Clause). Trong Phase 4B.3: **0 byte trọng số được tải về**, **0 training run thực hiện**.
   * Đệ trình các yêu cầu phê duyệt rõ ràng cho người dùng trước khi kích hoạt Phase 4C.

---

## 2. Archive Inventory & ETag Reconciliation

| Component | Archive | Expected Bytes | Actual Bytes | Local SHA-256 | Phase 4B.1 ETag (WebDAV) | Phase 4B.2 ETag (HTTP GET) | ETag Match | Content Integrity Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| `tgif-orig` | `orig_validation.tar.gz` | 859,947,874 | 859,947,874 | `c9f02a343a5ac759f1e7aae6b62b4d1cd3c2e8f01d6ffe182fd5815674945ade` | `"d95a1202a7445d79872735f60cbeaa95"` | `"69420b-62589574d6c41"` | `false` | **VERIFIED (SHA-256 + Bytes)** |
| `tgif-orig` | `orig_testing.tar.gz` | 806,962,390 | 806,962,390 | `8020c2f2080b349f68b9c22d4594c0df0d0722255bba981bdf18b47c291df52c` | `"fc2934fed45674aa89479640d0030beb"` | `"69420b-62589573eedc1"` | `false` | **VERIFIED (SHA-256 + Bytes)** |
| `tgif-sd2-sp` | `sd2-sp_validation.tar.gz` | 2,172,017,290 | 2,172,017,290 | `bd9eb4399f60166a09209d8a66a5df2b5eaecfbcc1a9d52f694e6812e24c5ad7` | `"44b9e62f224dd312f59d9a5bd79d1171"` | `"69420b-62589574d6089"` | `false` | **VERIFIED (SHA-256 + Bytes)** |
| `tgif-sd2-sp` | `sd2-sp_testing.tar.gz` | 2,040,575,228 | 2,040,575,228 | `c346af3cb85b00ac2b944d2e47e71d0e531652142ea844b63c337f95671b82aa` | `"2aa172ad2b1200973b7593259b37cc07"` | `"69420b-62589573ed651"` | `false` | **VERIFIED (SHA-256 + Bytes)** |
| **Tổng** | **4 archives** | **5,879,502,782** | **5,879,502,782** | *(100% dung lượng và mã băm SHA-256 khớp tuyệt đối)* | | | | **PASS** |

### Kết luận Khoa học về ETag
1. `remote_metadata_changed`: `true`
2. `content_integrity`: `verified-by-local-sha256-and-byte-count`
3. `etag_match`: `false`
4. **Nguyên nhân kỹ thuật**: ETag trong Nextcloud được sinh bởi module WebDAV dưới dạng mã hash MD5 của file (`"d95a..."`), trong khi lệnh tải trực tiếp HTTP GET đi qua web server Apache phía trước sinh ETag hex theo định dạng `inode-size-mtime` (`"69420b-6258957..."`). ETag không phải là mã băm nội dung bất biến trên hạ tầng máy chủ này.

---

## 3. Manifest Audit & Phân loại Đơn vị Mẫu

### 3.1 Định nghĩa Tường minh các Khái niệm
* **`source_id`**: Đơn vị thống kê độc lập tối thượng (mã ảnh gốc MS-COCO gồm 12 chữ số zero-padded). Toàn bộ phép tính phân chia tập dữ liệu, bootstrap CI và primary metrics đều tính trên đơn vị này.
* **`instance_id`**: Nhiệm vụ chỉnh sửa ảnh theo danh mục COCO (ví dụ: `surfboard_000000002261`).
* **`image_variant`**: File ảnh cụ thể tồn tại trên đĩa (native, 512, 1024 đối với authentic; bbox 0..2, segm 0..2 đối với edited).
* **`training_sample`**: Mẫu tensor được DataLoader trả về trong quá trình huấn luyện (ảnh, nhãn nhị phân, metadata).
* **`paired_sample`**: Một cặp gồm đúng 1 ảnh authentic và 1 ảnh edited cùng chung `source_id`.

### 3.2 Bảng Thống kê Kiểm toán
| Tiêu chí | Đo lường thực tế | Kỳ vọng | Kết luận |
| :--- | :---: | :---: | :---: |
| Số dòng trong `manifest_pilot_a_option_p.csv` | **684** | 684 | Khớp 100% |
| Số cột | **14** | 14 | Chuẩn schema |
| Đường dẫn authentic tồn tại trên đĩa | **684 / 684** | 684 | 0 thiếu sót |
| Đường dẫn canonical edit tồn tại trên đĩa | **684 / 684** | 684 | 0 thiếu sót |
| Unique `source_id` | **684** | 684 | Không trùng |
| Unique `instance_id` | **684** | 684 | Không trùng |
| Phân vùng `development_train` | **250** | 250 | Khớp |
| Phân vùng `inner_validation` | **91** | 91 | Khớp |
| Phân vùng `locked_test` | **343** | 343 | Khớp |
| Learning curve points ($N=50, 100, 250$) | **50, 100, 250** | 50, 100, 250 | Lồng nhau hoàn hảo |
| Dòng trùng lặp / Đường dẫn trùng lặp | **0** | 0 | PASS |
| Rò rỉ chéo phân vùng theo `source_id` | **0** | 0 | PASS |
| Số biến thể authentic trên mỗi source | **3** (native, 512, 1024) | 3 | Khớp |
| Số biến thể edited trên mỗi source | **6** (3 bbox + 3 segm) | 6 | Khớp |
| Tổng số ảnh authentic trên đĩa | **2,052** | 2,052 | $684 \times 3$ |
| Tổng số ảnh edited trên đĩa | **4,104** | 4,104 | $684 \times 6$ |
| Tổng số ảnh giải nén trên đĩa | **6,156** | 6,156 | $2,052 + 4,104$ |
| Số dòng trong `manifest_pilot_a_variant_pairs.csv` | **4,104** | 4,104 | Khớp 100% |

---

## 4. Anti-Pseudoreplication & Sampling Strategy

Dự án lựa chọn và đăng ký trước **Strategy A: Pair-Aware Sampler**:

```text
Mỗi Epoch (2N samples):
├── Lặp qua N unique source_id trong partition:
│   ├── Lấy 1 mẫu Authentic (native unresized variant 0)
│   └── Lấy 1 mẫu AI-Edited (luân phiên tất định: variant_idx = (epoch + offset) % 6)
└── Cân bằng tỷ lệ lớp 1:1 chính xác trong từng epoch
```

* **Lý do lựa chọn**: Đối với dữ liệu pilot quy mô nhỏ, Phương án A loại bỏ hoàn toàn nguy cơ ngụy tạo cỡ mẫu (pseudoreplication). Mỗi source chỉ đóng góp đúng 1 ảnh thật và 1 ảnh giả trong mỗi epoch, ngăn chặn việc một ảnh có nhiều biến thể can thiệp làm thiên lệch trọng số gradient hoặc giả định tăng cỡ mẫu độc lập.
* **Đơn vị Đánh giá Chính**: Khi đánh giá mô hình, toàn bộ xác suất dự đoán của các biến thể thuộc cùng một `source_id` được tổng hợp (mean pooling) thành một điểm số duy nhất trước khi tính Macro-F1, Balanced Accuracy, AUROC, Brier Score và ECE.
* **Đánh giá Biến thể**: Variant-level metrics chỉ đóng vai trò phân tích phụ tá (secondary/diagnostic).

---

## 5. Phân quyền Locked-Test (Role-Based Access Guard)

Quy tắc khóa chung chung được thay thế bằng module `ml/evaluation/locked_test_guard.py` thực thi 3 quyền độc lập:

| Vai trò gọi module (`AccessRole`) | Quyền truy cập `locked_test` | Hành vi thực thi |
| :--- | :---: | :--- |
| `TRAINING_LOADER` | **`false`** | Chặn đứng vô điều kiện, ném ngoại lệ `PermissionError` |
| `DEVELOPMENT_EVALUATOR` | **`false`** | Chặn đứng vô điều kiện, ném ngoại lệ `PermissionError` |
| `FINAL_EVALUATOR` | **`true`** *(có điều kiện)* | Chỉ cấp quyền khi cung cấp đủ `ExperimentLockBinding` hợp lệ |

### Ràng buộc Mã hóa Bắt buộc để Mở Khóa Đánh giá Cuối cùng:
1. `experiment_lock_hash`: Mã băm SHA-256 tổng hợp của toàn bộ cấu hình thực nghiệm.
2. `config_hash`: Mã băm SHA-256 của file cấu hình YAML.
3. `checkpoint_hash`: Mã băm SHA-256 của checkpoint mô hình đã đóng băng.
4. `preprocessing_hash`: Mã băm SHA-256 của pipeline tiền xử lý ảnh.
5. `threshold_calibration_hash`: Mã băm SHA-256 của tham số hiệu chuẩn nhiệt độ và ngưỡng phân loại.
6. `seeds`: Danh sách hạt giống ngẫu nhiên đã đăng ký trước `[42, 1337, 2025, 3407, 9001]`.
7. `timestamp_utc`: Thời điểm niêm phong khóa đánh giá.
8. `locked_split_seal`: Phải khớp 100% với seal đóng băng từ Phase 4B.2: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9`.

---

## 6. Đăng ký trước Thí nghiệm Phase 4C (Preregistration Summary)

* **Tài liệu đăng ký trước**: `docs/PHASE_4C_PREREGISTRATION.md`
* **File cấu hình chuẩn**: `ml/configs/pilot_a_binary_preregistered.yaml` (SHA-256: `839531a700b211e5adc4a145e811c4533dbb45a044b63e816c474a462e88ddd1`)
* **Phạm vi bài toán**: Phân loại nhị phân `authentic` vs `ai_edited`. Không gian 3 lớp tiếp tục khóa.
* **Learning curve**: $N \in \{50, 100, 250\}$ unique sources lồng nhau trên `development_train` (250 sources). `inner_validation` cố định 91 sources. `locked_test` cố định 343 sources.
* **Hạt giống (Seeds)**: 5 seeds: `[42, 1337, 2025, 3407, 9001]`.
* **6 Baselines**:
  1. Stratified Random Dummy
  2. Metadata-only Logistic Regression
  3. DSP-only Classifier (2D FFT / DCT / Noise Residual)
  4. Frozen MobileNetV3-Small Visual Head
  5. Fine-tuned MobileNetV3-Small (Block 12)
  6. Calibrated Multimodal Fusion (Visual + DSP + Metadata)
* **Chỉ số chính (Source-Level)**: Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE, Selective Risk & Coverage.
* **Phương pháp thống kê**: Báo cáo mean $\pm$ std qua 5 seeds; khoảng tin cậy 95% Paired Stratified Bootstrap với 1.000 lần lấy mẫu lại theo đơn vị `source_id`.
* **Stage Gates**:
  * Stage 0: Khảo sát Dummy, Metadata-only và DSP-only trên dev_train/inner_val.
  * Stage 1: Huấn luyện classification head với backbone đóng băng; chọn checkpoint bằng inner_val.
  * Stage 2: Mở block 12 chỉ khi Stage 1 vượt trội Dummy và không bị Metadata-only giải thích.
  * Final Evaluation: Đóng băng toàn bộ checkpoint và tham số, sinh experiment-lock, đánh giá duy nhất 1 lần trên locked_test.

---

## 7. Kế hoạch Pretrained Weights và Tài nguyên Tính toán

### 7.1 Thông số Kỹ thuật Pretrained Weights (Kế hoạch)
* **Phiên bản PyTorch / Torchvision**: PyTorch `2.14.0+cpu`, Torchvision `0.29.0+cpu`
* **Mô hình**: MobileNetV3-Small
* **Enum trọng số torchvision**: `torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1`
* **URL tải về**: `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`
* **Số tham số**: 1,782,648 tham số trong feature backbone; 2,542,856 tham số đầy đủ
* **Kích thước tải dự kiến**: ~10.3 MB (10,830,000 bytes)
* **Giấy phép**: BSD 3-Clause (hoàn toàn phù hợp cho Research Track)
* **Trạng thái Phase 4B.3**: **0 bytes trọng số tải về, 0 training run thực hiện**.

### 7.2 Kế hoạch Tính toán & Ranh giới Web
* **Giai đoạn huấn luyện**: Thực hiện trên máy cục bộ bằng CPU (hoặc CUDA GPU nếu người dùng cho phép). Thời gian huấn luyện ước tính $< 45$ phút trên CPU thông thường.
* **Sản phẩm Web**: Bản web thương mại và minh chứng hoàn toàn suy luận bằng **CPU/WASM SIMD client-side** (Zero-Egress), không phụ thuộc vào GPU hay tài nguyên máy chủ. Mọi tuyên bố hiệu năng web chỉ được công nhận sau benchmark thực tế trên trình duyệt.

---

## 8. Báo cáo Kiểm thử Tự động & Bản dựng

| Bộ kiểm thử | Lệnh thực thi | Tổng số test | Kết quả | Trạng thái |
| :--- | :--- | :---: | :---: | :---: |
| **Python Test Suite** | `ml/.venv/Scripts/python -m pytest ml/tests -v` | 79 | 79 passed, 0 failed | **PASS** |
| **TypeScript / Vitest** | `pnpm -r run test` | 57 | 57 passed, 0 failed | **PASS** |
| **Continuity Unit Tests** | `node --test scripts/__tests__/continuity-check.test.mjs` | 13 | 13 passed, 0 failed | **PASS** |
| **Tổng số bài test tự động** | | **149** | **149 passed, 0 failed** | **PASS (100%)** |
| **Dataset Registry Validation** | `python -m ml.datasets.acquire --validate-registry` | 7 datasets | 7/7 valid | **PASS** |
| **Config Validator** | `python -m ml.configs.validator --validate-all` | 4 configs | 4/4 valid | **PASS** |
| **Web Production Build** | `pnpm build` | 55 modules | 2.99s, exit code 0 | **PASS** |
| **Continuity Checker Gate** | `pnpm continuity:check` | Toàn bộ repo | `CONTINUITY_CHECK: PASS` | **PASS** |

---

## 9. Danh mục Artifacts Tạo mới trong Phase 4B.3

Tất cả artifacts đã được tạo đầy đủ trong `research/evidence/phase-4b.3/`:
1. `environment.json`: Cấu hình chi tiết nền tảng, runtime (Python 3.12, Node v24, Torch 2.14, Torchvision 0.29).
2. `git-state.json`: Ghi nhận trạng thái Git khởi đầu (`1ad8151`), base main (`460f6d5`), remote configuration.
3. `continuity-reconciliation.json`: Ghi nhận việc đồng bộ commit Phase 4B.2, sửa mã băm `sd2-sp_val` và xác nhận 0 training runs.
4. `etag-reconciliation.json`: Đối soát chi tiết ETag WebDAV vs HTTP GET cho 4 archive, chứng minh tính toàn vẹn bằng SHA-256 và byte count.
5. `manifest-audit.json`: Kiểm toán toàn diện 684 dòng manifest, giải thích con số 4,104 và 6,156 ảnh.
6. `source-balance-audit.json`: Chứng minh toán học của Strategy A (Pair-Aware Sampler) và luân phiên biến thể tất định.
7. `locked-test-access-audit.json`: Kiểm toán cơ chế phân quyền Role-Based cho `locked_test` và bảo toàn seal `519e7a0e...`.
8. `experiment-preregistration.json`: Bảng đăng ký máy đọc cho thí nghiệm phân loại nhị phân Phase 4C.
9. `test-summary.json`: Báo cáo chi tiết kết quả thực thi 149 bài kiểm thử tự động.
10. `build-summary.json`: Báo cáo build production web bundle tối ưu (2.99s).
11. `evidence-manifest.json`: Danh mục kiểm kê mã băm SHA-256 của toàn bộ evidence artifacts Phase 4B.3.
12. `PHASE_REPORT.md`: Báo cáo khoa học chính thức của Phase 4B.3.

---

## 10. Yêu cầu Phê duyệt cho Phase 4C

Để bắt đầu Phase 4C, tác nhân AI đệ trình các yêu cầu phê duyệt cụ thể sau đây:

1. **Phê duyệt tải trọng số Pretrained MobileNetV3-Small**: Cho phép tải ~10.3 MB từ `download.pytorch.org` về môi trường huấn luyện Research Track (`models/research/pretrained/`).
2. **Phê duyệt thực thi Stage 0**: Cho phép chạy huấn luyện và đánh giá 3 baselines (Dummy, Metadata-only, DSP-only) trên `development_train` và `inner_validation`.
3. **Phê duyệt thực thi Stage 1**: Cho phép huấn luyện classification head của MobileNetV3-Small trên 3 tập learning curve ($N \in \{50, 100, 250\}$) với 5 seeds, sử dụng CPU cục bộ trong thời gian ước tính $< 45$ phút.
