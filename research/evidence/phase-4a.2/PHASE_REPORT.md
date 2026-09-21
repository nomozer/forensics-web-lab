# Báo cáo Nghiệm thu Giai đoạn Phase 4A.2 (Phase 4A.2 Completion Report)

> **Giai đoạn**: Phase 4A.2 — Research Scope Freeze, Continuity Documents and TGIF Feasibility Audit  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting commit**: `863a491`  
> **Ending commit**: `0c42f3c`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không thay đổi)  
> **Thời điểm hoàn thành**: 2026-09-21  
> **Tuyên bố trung thực khoa học**: `No scientific model-performance claim is currently supported.`

---

## 1. Mục tiêu Giai đoạn

1. Thiết lập giao thức continuity 3 file chính thức (`docs/continuity/CODE_INDEX.md`, `docs/continuity/CURRENT_STATE.md`, `docs/continuity/STATUS_LEDGER.md`), giải quyết dứt điểm phân mảnh tài liệu và loại bỏ các file tracking cũ.
2. Đóng băng phạm vi nghiên cứu khoa học: 3 lớp mục tiêu (`authentic`, `fully_generated`, `ai_edited`), xác định `uncertain` là trạng thái quyết định sau calibration và selective abstention, định vị vùng chỉnh sửa (localization) là mục tiêu phụ.
3. Chốt 4 câu hỏi nghiên cứu cốt lõi (RQ1–RQ4), 1 câu hỏi phụ (RQ5) và hệ thống chỉ số phân cấp toàn diện trong `docs/RESEARCH_PLAN.md` và `docs/EVALUATION.md`.
4. Khảo sát metadata chính thức của TGIF và TGIF2 trên GitHub và Nextcloud public shares.
5. Cập nhật Dataset Registry, chuẩn hóa ma trận nghiên cứu–dữ liệu và nhận diện rủi ro dataset-source shortcut.
6. Xây dựng đề xuất pilot 3 cấp (A/B/C) và chọn phương án khuyến nghị phù hợp với nguồn lực khóa luận.

---

## 2. Công việc Đã Hoàn thành

### 2.1. Bộ ba tài liệu Continuity Thống nhất (`docs/continuity/`)
- Tạo `docs/continuity/CODE_INDEX.md` (157 dòng, ngân sách $\le 350$ dòng): Bản đồ kiến trúc, sơ đồ luồng xử lý, trách nhiệm từng package, contracts, sổ đăng ký và sổ tay lệnh.
- Tạo `docs/continuity/CURRENT_STATE.md` (154 dòng, ngân sách $\le 300$ dòng): Nguồn trạng thái duy nhất cho session mới, tuyên bố trung thực khoa học, phân loại 5 trạng thái triển khai, nguy cơ shortcut và danh sách evidence.
- Tạo `docs/continuity/STATUS_LEDGER.md`: Biên niên sử cô đọng tiến độ từ Phase 0 đến Phase 4A.2 (mỗi phase $\le 20$ dòng, phase mới nhất ở đầu file).
- Di chuyển nội dung và xóa an toàn 3 file cũ: `docs/CODE_MAP.md`, `docs/PROJECT_STATE.md`, `docs/SESSION_HANDOFF.md`.
- Cập nhật `AGENTS.md` bắt buộc tác nhân AI đọc theo thứ tự: `CURRENT_STATE.md` $\rightarrow$ `CODE_INDEX.md` $\rightarrow$ `STATUS_LEDGER.md`.

### 2.2. Đóng băng Phạm vi Khoa học và Hệ thống Chỉ số
- **3 Lớp huấn luyện**: `authentic` (ảnh máy ảnh thật), `fully_generated` (ảnh sinh toàn phần từ GAN/Diffusion), `ai_edited` (ảnh có vùng can thiệp bằng inpainting/splicing).
- **Trạng thái `uncertain`**: Xác định là trạng thái quyết định sau calibration khi confidence thấp hoặc tín hiệu mâu thuẫn; không phải nhãn dữ liệu thứ tư.
- **Mục tiêu chính**: Phân loại 3 lớp bằng mạng tích chập nhẹ ($<15\text{M}$ tham số), hiệu chuẩn độ tin cậy (Temperature Scaling), kiểm tra robustness, và suy luận in-browser CPU/WASM.
- **Mục tiêu phụ**: Định vị vùng AI inpainting khi có ground-truth mask. Bản đồ nhiệt từ patch scores ghi nhận là heuristic khám phá sơ bộ cho đến khi có kiểm chứng mask thật.
- **Chốt 4 câu hỏi nghiên cứu (RQ1–RQ4)**: RQ1 (3-class generalization on unseen data), RQ2 (multimodal evidence fusion vs visual-only), RQ3 (ONNX INT8 quantization trade-offs), RQ4 (browser CPU/WASM execution feasibility), và Auxiliary RQ5 (patch localization).
- **Hệ thống chỉ số**: Primary (Macro-F1, Balanced Accuracy), Secondary (Per-class P/R/F1, AUROC, Confusion Matrix), Calibration (ECE, Brier, NLL, Coverage, Selective Risk), Localization (mIoU, Dice, Pixel AUROC), Robustness, Browser runtime. Toàn bộ mang trạng thái `not evaluated`.

### 2.3. Khảo sát Metadata và Tính Khả thi của TGIF / TGIF2
- Khảo sát trực tiếp qua web inspection các public share Nextcloud chính thức của IMEC IDLab:
  1. **TGIF Share (`xEeAzrY7ES9KA8o`)**: Tổng dung lượng **65.4 GB** (6 thư mục: `masks` 40.4 MB, `orig` 6.8 GB, `ps-sp` 16.8 GB, `sd2-fr` 7.1 GB, `sd2-sp` 17.3 GB, `sdxl-fr` 17.3 GB).
  2. **TGIF2 FLUX Share (`KG48tLZZzifC5WE`)**: Tổng dung lượng **110 GB** (8 thư mục: `masks-flux` 7.7 MB, `orig-flux` 5.7 GB, 6 thư mục model FLUX 16–18 GB).
  3. **TGIF2 Random Share (`GDGewtTFcHccaNj`)**: Tổng dung lượng **73 GB** (13 thư mục: `masks-*` ~900 KB, `metadata` 1.8 MB, model subfolders).
- **Kết luận khả thi**: Nền tảng Nextcloud cho phép tải riêng từng thư mục con dưới dạng zip nén động (khác với GenImage Google Drive yêu cầu tải toàn bộ chuỗi split archive ~24 GB).
- Thẩm định bản quyền: Dataset TGIF phân phối theo `CC BY-SA 4.0`, ảnh gốc MS-COCO `CC BY 4.0`. Trọng số phái sinh tiềm ẩn điều kiện Share-Alike nên dự án áp dụng chính sách bảo thủ cấm đưa vào sản phẩm (`research-only`).
- Đăng ký `tgif` và `tgif2` vào `datasets/registry.json`.

### 2.4. Đính chính Kết luận GenImage
- Đính chính BigGAN từ cụm từ "smallest usable archive" thành **"first fully inventoried archive"** do chưa có bảng kích thước của cả 8 generator.
- Ghi nhận chính xác: dung lượng nén 23,516,377,048 bytes là `verified`, dung lượng giải nén ~26 GB là `estimated`, số ảnh ~16,000–20,000 là `estimated`.

### 2.5. Ma trận Nghiên cứu – Dữ liệu và Kiểm soát Shortcut
- Lập ma trận nghiên cứu–dữ liệu trong `docs/DATASETS.md`.
- Nhận diện rủi ro *Dataset-Source Shortcut*: Mô hình có thể học đặc trưng của camera/compression từng dataset thay vì dấu vết AI.
- Đề ra 5 biện pháp kiểm soát: Group split theo `source_id`, dùng matched pairs (MS-COCO authentic + inpainted version), theo dõi đa chiều lineage, chuẩn hóa tiền xử lý, và đánh giá source-held-out.

### 2.6. Đề xuất Phương án Thử nghiệm 3 Cấp (Pilot Proposals)
- **Pilot A (Pipeline Smoke)**: 10–50 ảnh, 0 byte mạng hoặc ~40.4 MB (`masks`), trạng thái `pipeline-only`.
- **Pilot B (Exploratory Three-Class — Khuyến nghị cho Khóa luận)**: ~3,000 ảnh cân bằng 3 lớp, tải các thư mục `masks` (40.4 MB) + `orig` (6.8 GB) + `sd2-fr` (7.1 GB) hoặc `sd2-sp` (17.3 GB) từ TGIF Nextcloud. Dung lượng nén ~15 GB, giải nén ~18 GB, yêu cầu $\ge 35\text{ GB}$ đĩa trống. Toàn bộ metric gắn nhãn `exploratory`.
- **Pilot C (Confirmatory Benchmark)**: Toàn bộ split official (>80,000 ảnh), ~90 GB nén, $\ge 220\text{ GB}$ đĩa trống.

---

## 3. Kết quả Kiểm thử & Bản dựng Thực tế

| Bộ kiểm thử | Lệnh thực thi | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :--- |
| **TypeScript Monorepo Tests** | `pnpm test` | 57/57 tests passing trên 6 package (shared: 34, provenance: 3, report: 4, forensics: 5, inference: 8, web: 3) | `PASS` |
| **Python ML Tests** | `pytest ml/tests -v` | 15/15 tests passing (contamination guard: 10, model pipeline: 5) | `PASS` |
| **Production Web Build** | `pnpm build` | Exit code 0 trong 2.93s; bundle Vite hợp lệ (index.html: 1.08 kB, worker: 437.93 kB, WASM runtime: 28.3 MB) | `PASS` |
| **Clean Link Invariance** | `pnpm test -t "Repository Clean Link"` | 0 machine-local links (`file:///`, `C:\`, `D:\`) trong markdown links của repo | `PASS` |
| **Dataset Registry Validation**| `acquire.py --validate-registry` | 7/7 datasets kiểm định hợp lệ theo schema | `PASS` |

---

## 4. Hạch toán Tài nguyên & Dữ liệu Ngoại vi

- **Metadata Requests**: 3 yêu cầu inspect HTML/DOM metadata tới các share Nextcloud của TGIF.
- **External Dataset Bytes Downloaded**: `0 bytes`.
- **Model Checkpoint Bytes Downloaded**: `0 bytes`.
- **Training Runs Executed**: `0`.

---

## 5. Tình trạng Tuyên bố Khoa học (Scientific Claims Audit)

- **Supported Claims**: Không có tuyên bố độ chính xác nào được hỗ trợ. Toàn bộ pipeline deep learning chưa có trọng số mô hình.
- **Unsupported Claims**: Độ chính xác phân loại 3 lớp, Macro-F1, Balanced Accuracy, inpainting localization mIoU/Dice, và Expected Calibration Error (ECE) tiếp tục được công bố trung thực là `unverified` / `not evaluated`.
- **Trạng thái Web App**: Hiển thị trung thực `verdict: uncertain`, `confidence: null`, kèm thông báo rõ ràng "Model not installed".

---

## 6. Danh mục Minh chứng Bổ sung trong Evidence Register

- `EV-CONTINUITY-001` (`git-state`, `verified`): Hệ thống 3 file continuity thống nhất tại `docs/continuity/`.
- `EV-RESEARCH-SCOPE-001` (`science`, `verified`): Đóng băng phạm vi nghiên cứu 3 lớp và trạng thái `uncertain`.
- `EV-RQ-METRIC-MAP-001` (`science`, `verified`): Khóa 4 câu hỏi nghiên cứu và hệ thống chỉ số phân cấp.
- `EV-TGIF-METADATA-001` (`dataset`, `verified`): Kiểm kê metadata TGIF/TGIF2 qua Nextcloud.
- `EV-TGIF-LICENSE-001` (`license`, `verified`): Thẩm định giấy phép TGIF CC BY-SA 4.0 và MS-COCO CC BY 4.0.
- `EV-DATASET-SHORTCUT-RISK-001` (`dataset`, `verified`): Nhận diện nguy cơ shortcut và thiết lập cơ chế kiểm soát.
- `EV-PILOT-PROPOSAL-001` (`dataset`, `verified`): Xây dựng đề xuất 3 cấp và khuyến nghị Pilot B.

---

## 7. Các Blocker và Quyết định Cần Người dùng Phê duyệt

1. **Khóa tải dữ liệu ngoài**: Lệnh `python -m ml.datasets.acquire --execute` đang ở chế độ fail-closed và yêu cầu sự phê duyệt cụ thể.
2. **Quyết định tiếp theo**: Người dùng xem xét phê duyệt tải các thư mục con của TGIF cho Phương án Pilot B (~14.5–15 GB nén) để tiến hành huấn luyện thử nghiệm ở các giai đoạn sau.
