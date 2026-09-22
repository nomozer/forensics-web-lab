# Phase 4B.2 — Controlled Option-P Acquisition, Pairability Verification and Split Freeze

> **Phase**: 4B.2  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting Commit**: `82a5266`  
> **Implementation Snapshots**: `7d33072`, `6bb088b`  
> **Ending Commit**: `1ad8151`  
> **Verdict**: **`PASS`**  
> **Execution Date**: 2026-09-22  
> **Scope**: Research Track Only — Small-Data Feasibility & Pilot Splitting (TGIF Option P)

---

## 1. Executive Summary & Verdict

Phase 4B.2 hoàn thành toàn bộ mục tiêu thu nạp có kiểm soát Option P, xác minh tính toàn vẹn và decode cho 100% ảnh, kiểm toán ghép cặp ba thành phần (*authentic ↔ AI-edited ↔ ground-truth mask*), và đóng băng phân vùng split xác định cho nghiên cứu ít dữ liệu của khóa luận:

1. **Thu nạp 4 Archive Phê duyệt**: Tải thành công 4 archive theo phê duyệt (`orig_validation`, `orig_testing`, `sd2-sp_validation`, `sd2-sp_testing`) với tổng cộng **5,879,502,782 bytes (~5.88 GB)** từ duy nhất host `cloud.ilabt.imec.be`. 100% dung lượng byte count và SHA-256 cục bộ được xác minh tuyệt đối. ETag trên luồng streaming HTTP GET là định dạng inode/mtime của Apache (`69420b...`), khác với ETag từ WebDAV PROPFIND của Phase 4B.1; tính toàn vẹn nội dung được bảo đảm độc lập bằng SHA-256 và byte count.
2. **Safe Extraction & Decodability 100%**: Giải nén an toàn với cơ chế chống Tar Slip, path traversal, absolute path và escaping symlinks, trích xuất **6,156 file ảnh (5,890,096,035 bytes)**. Kiểm toán Pillow decode cho 100% ảnh đạt 0 lỗi hỏng, 0 dimension mismatch.
3. **Pairability Audit `verified`**: Khớp nối thành công 100% (684 category instances) giữa ảnh thật, ảnh AI-edited và mask. Không có ảnh thật bị thiếu, không có ảnh sửa bị thiếu, không có mask bị thiếu.
4. **Đóng băng Split Nguồn Độc Lập**:
   * `development_train`: **250 unique `source_id`**
   * `inner_validation`: **91 unique `source_id`**
   * `locked_test`: **343 unique `source_id`** (niêm phong mã hóa SHA-256 seal: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9`)
   * Giao cắt chéo split theo `source_id`: **0 tuyệt đối**.
   * Learning curve lồng nhau: $N=50 \subset N=100 \subset N=250$.
5. **Class-Coverage Guard**: Pipeline 2 lớp (`authentic` vs `ai_edited`) ở trạng thái `runnable`. Pipeline 3 lớp ở trạng thái `not-runnable-missing-fully-generated-data`.
6. **Scientific Integrity**: 0 byte trọng số model tải về, 0 training run thực hiện trong Phase 4B.2. Phase 4B.2 chưa hỗ trợ bất kỳ tuyên bố nào về hiệu năng mô hình.

---

## 2. Archive Inventory & Checksum Accounting

| Component | Archive | Expected Bytes | Actual Bytes | SHA-256 | Inventory ETag (4B.1) | Acquisition ETag (4B.2) | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| `tgif-orig` | `orig_validation.tar.gz` | 859,947,874 | 859,947,874 | `c9f02a343a5ac759f1e7aae6b62b4d1cd3c2e8f01d6ffe182fd5815674945ade` | `"d95a1202a7445d79872735f60cbeaa95"` | `"69420b-62589574d6c41"` | **VERIFIED** |
| `tgif-orig` | `orig_testing.tar.gz` | 806,962,390 | 806,962,390 | `8020c2f2080b349f68b9c22d4594c0df0d0722255bba981bdf18b47c291df52c` | `"fc2934fed45674aa89479640d0030beb"` | `"69420b-62589573eedc1"` | **VERIFIED** |
| `tgif-sd2-sp` | `sd2-sp_validation.tar.gz` | 2,172,017,290 | 2,172,017,290 | `bd9eb4399f60166a09209d8a66a5df2b5eaecfbcc1a9d52f694e6812e24c5ad7` | `"44b9e62f224dd312f59d9a5bd79d1171"` | `"69420b-62589574d6089"` | **VERIFIED** |
| `tgif-sd2-sp` | `sd2-sp_testing.tar.gz` | 2,040,575,228 | 2,040,575,228 | `c346af3cb85b00ac2b944d2e47e71d0e531652142ea844b63c337f95671b82aa` | `"2aa172ad2b1200973b7593259b37cc07"` | `"69420b-62589573ed651"` | **VERIFIED** |
| **Tổng** | **4 archives** | **5,879,502,782** | **5,879,502,782** | *(Tất cả archive nguyên vẹn tại `data/research/tgif/archives/`)* | | | **PASS** |

---

## 3. Extraction & Image Decodability Audit

Tất cả 4 archive được giải nén an toàn thông qua hàm `safe_extract_tar()`:
* Ngăn chặn path traversal (`..` Tar Slip).
* Ngăn chặn đường dẫn tuyệt đối (`/`).
* Ngăn chặn escaping symlinks trỏ ra ngoài thư mục đích.
* Kiểm tra tính đóng gói thư mục đích (`resolve().is_relative_to()`).

### Thống kê file giải nén

| Phân vùng | Thư mục trích xuất | Số file ảnh | Tổng dung lượng (bytes) | Trạng thái decode PIL |
| :--- | :--- | :---: | :---: | :---: |
| `orig/validation` | `data/research/tgif/orig/validation/` | 1,023 | 861,683,700 | 1,023 / 1,023 PASS (100%) |
| `orig/testing` | `data/research/tgif/orig/testing/` | 1,029 | 808,321,851 | 1,029 / 1,029 PASS (100%) |
| `sd2-sp/validation` | `data/research/tgif/sd2-sp/validation/` | 2,046 | 2,176,267,677 | 2,046 / 2,046 PASS (100%) |
| `sd2-sp/testing` | `data/research/tgif/sd2-sp/testing/` | 2,058 | 2,043,822,807 | 2,058 / 2,058 PASS (100%) |
| **Tổng cộng** | | **6,156** | **5,890,096,035** | **6,156 / 6,156 PASS (100%)** |

* Ảnh bị hỏng / decode failure: **0**
* Định dạng: 100% PNG hợp lệ.
* Trùng lặp SHA-256 bất thường: **0**.

---

## 4. Tripartite Pairability Audit (`verified`)

* **Authentic (Orig)**: Mỗi instance chứa 3 biến thể phân giải (ảnh gốc unresized, 512x512, 1024x1024).
* **AI-Edited (SD2-sp)**: Mỗi instance chứa 6 biến thể (3 variants cho `bbox` inpainting + 3 variants cho `segm` inpainting).
* **Ground-Truth Masks**: Cung cấp đầy đủ các mask tương ứng tại `data/research/tgif/masks/`.

### Bảng đối soát ghép cặp

| Tiêu chí | Đo lường thực tế | Kỳ vọng | Kết luận |
| :--- | :---: | :---: | :---: |
| **Matched Task Instances** | **684** | 684 (341 val + 343 test) | **Khớp 100%** |
| **Missing Originals** | **0** | 0 | **PASS** |
| **Missing Edited Images** | **0** | 0 | **PASS** |
| **Missing Masks** | **0** | 0 | **PASS** |
| **Dimension Mismatches** | **0** | 0 | **PASS** |
| **Corrupt Images** | **0** | 0 | **PASS** |
| **Ambiguous Mappings** | **0** | 0 | **PASS** |
| **Tình trạng Pairability** | **`verified`** | `verified` | **PASS** |

---

## 5. Frozen Splits & Partition Isolation

### 5.1 Cấu trúc Phân vùng

Phân chia dựa trên đơn vị độc lập bắt buộc là **MS-COCO `source_id`** (12 chữ số zero-padded):

```text
Validation Pool (341 unique sources)
├── development_train: 250 sources (seed 42)
└── inner_validation:   91 sources (seed 42, early stopping/threshold)

Testing Pool (Official TGIF Test)
└── locked_test:       343 sources (niêm phong SHA-256 seal)
```

### 5.2 Kiểm toán Cô lập (Zero-Leakage)

* Giao cắt `development_train` $\cap$ `inner_validation`: **0 sources**
* Giao cắt `development_train` $\cap$ `locked_test`: **0 sources**
* Giao cắt `inner_validation` $\cap$ `locked_test`: **0 sources**
* **Cross-Split Overlap**: **0 tuyệt đối**.

### 5.3 Learning Curves Lồng nhau (Nested Invariant)

Từ tập `development_train` (250 sources), các tập con huấn luyện cho learning curve được sinh tất định:
* $N=50$: 50 unique sources
* $N=100$: 100 unique sources
* $N=250$: 250 unique sources
* Kiểm chứng toán học: $N=50 \subset N=100 \subset N=250$ (**PASS**).
* $N=500$ và $N=1000$: Gắn nhãn bắt buộc `not-runnable-insufficient-independent-sources`.

### 5.4 Niêm phong Locked Test

* File niêm phong: `research/evidence/phase-4b.2/split-lock.json`
* **SHA-256 Seal**: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9`
* Quy tắc bảo vệ: `trainingAccessAllowed: false`. Bất kỳ loader huấn luyện nào cố truy cập `locked_test` sẽ bị chặn và ném ngoại lệ `PermissionError`.

---

## 6. Class-Coverage Guard

* **Dữ liệu hiện có**:
  * `authentic`: Có (2,052 ảnh, TGIF orig validation & testing)
  * `ai_edited`: Có (4,104 ảnh, TGIF sd2-sp validation & testing)
  * `fully_generated`: **CHƯA CÓ** (0 ảnh trong Option P)
* **Trạng thái Pipeline**:
  * Binary Classification (`authentic` vs `ai_edited`): **`runnable`**
  * Localization (`ai_edited` + masks): **`runnable`**
  * 3-Class Generalization (`authentic` / `fully_generated` / `ai_edited`): **`not-runnable-missing-fully-generated-data`**

---

## 7. Verification & Test Suite Accounting

| Bộ kiểm thử | Tổng số test | PASS | FAIL | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **TypeScript / Vitest** | 57 | 57 | 0 | Packages: shared, forensics, provenance, report, inference, web |
| **Continuity Suite** | 13 | 13 | 0 | Quy tắc diff, token parsing, scenario isolation |
| **Python Unit Tests** | 66 | 66 | 0 | Bao gồm 11 targeted tests cho Option P (Tar Slip, resume, split lock) |
| **Dataset Registry** | 7 | 7 | 0 | Validate thành công 7 dataset schema |
| **Web Production Build** | 1 | 1 | 0 | `pnpm build` hoàn tất không lỗi (Vite v6, 55 modules) |
| **Continuity Checker** | 1 | 1 | 0 | `pnpm continuity:check` PASS |

---

## 8. Artifacts Generated in Phase 4B.2

Tất cả artifacts đã được tạo đầy đủ trong `research/evidence/phase-4b.2/`:
1. `environment.json`: Cấu hình môi trường thực thi (Python 3.12.10, Node v24.13.0, pnpm 10.33.0).
2. `acquisition-binding.json`: Ràng buộc pháp lý và kỹ thuật trước khi tải.
3. `user-approval.json`: Bằng chứng phê duyệt của người dùng cho Option P.
4. `download-receipt.json`: Biên nhận chi tiết 4 archive tải về.
5. `archive-inventory.json`: Kiểm toán kích thước, SHA-256 và ETag từng archive.
6. `extraction-summary.json`: Báo cáo trích xuất an toàn và đếm file giải nén.
7. `content-manifest-summary.json`: Tổng kết decode PIL, số lượng ảnh và phân vùng.
8. `pairability-audit.json`: Chứng minh toán học ghép cặp ba thành phần.
9. `split-lock.json`: Đóng băng danh sách 343 `source_id` test kèm SHA-256 seal.
10. `class-coverage-audit.json`: Kiểm toán phân loại 2 lớp vs 3 lớp.
11. `test-summary.json`: Báo cáo tổng hợp kiểm thử tự động.
12. `build-summary.json`: Báo cáo build production web bundle.
13. `evidence-manifest.json`: Danh mục và checksum của toàn bộ evidence artifacts.
14. `PHASE_REPORT.md`: Báo cáo khoa học chính thức của Phase 4B.2.

---

## 9. Đề xuất Công việc cho Phase 4C

Phase 4B.2 hoàn thành trọn vẹn việc chuẩn bị và đóng băng dữ liệu Option P. Hoạt động huấn luyện và thực nghiệm chuyển sang **Phase 4C**:
1. **Pilot Baseline Model (Binary Feasibility)**:
   * Huấn luyện mô hình nhẹ (MobileNetV3 / EfficientNet-B0) phân loại nhị phân `authentic` vs `ai_edited`.
   * Sử dụng đúng tập `development_train` (250 sources) và `inner_validation` (91 sources).
2. **Learning Curve Execution**:
   * Đo lường hiệu năng (Balanced Accuracy, F1-macro, ECE) tại 3 điểm dữ liệu lồng nhau: $N=50$, $N=100$, $N=250$.
   * Kiểm tra giả thuyết: Liệu mô hình có học đặc trưng thực tế hay bị bão hòa shortcut.
3. **Sealed Evaluation**:
   * Chỉ mở `locked_test` (343 sources) sau khi đã đóng băng checkpoint và siêu tham số.
4. **Binary Localization Experiment**:
   * Thử nghiệm baseline định vị vùng can thiệp trên tập mask đi kèm.
