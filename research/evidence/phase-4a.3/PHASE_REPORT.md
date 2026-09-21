# Báo cáo Nghiệm thu Giai đoạn Phase 4A.3 (Phase 4A.3 Completion Report)

> **Giai đoạn**: Phase 4A.3 — Scientific Pilot Protocol and Label-Semantics Gate  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting commit**: `0c42f3c`  
> **Ending commit**: See repository HEAD  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không thay đổi)  
> **Thời điểm hoàn thành**: 2026-09-21  
> **Tuyên bố trung thực khoa học**: `No scientific model-performance claim is currently supported.`  
> **Phán quyết giai đoạn (Verdict)**: **`PASS`**

---

## 1. Mục tiêu Giai đoạn

1. Xác minh và đóng băng ý nghĩa khoa học của ba nhãn phân loại: `authentic`, `fully_generated`, `ai_edited`.
2. Xác định chính xác thành phần nào của TGIF/TGIF2 phù hợp với từng nhãn, giải quyết dứt điểm tranh cãi về mặt phương pháp luận đối với hai thành phần `sp` (spliced) và `fr` (fully regenerated).
3. Thiết kế kiến trúc pilot hai nhánh độc lập (Pilot A: Authentic vs AI-Edited + Localization; Pilot B: Authentic vs Fully-Generated; Pilot C: Three-Class Exploratory) để triệt tiêu nguy cơ mô hình học shortcut từ nguồn dataset (*dataset-source shortcut*).
4. Xây dựng giao thức kiểm soát shortcut và chống rò rỉ dữ liệu (Deduplication SHA-256/pHash, Group Split `source_id`, Class-level profile audit, Metadata-only baseline guard).
5. Xây dựng cấu hình pilot máy đọc (`ml/configs/pilot_tgif_edit.yaml`, `ml/configs/pilot_genimage_generated.yaml`), module kiểm định cấu hình (`ml/configs/validator.py`), và nâng cấp dry-run acquisition trong `ml/datasets/acquire.py`.
6. Bổ sung báo cáo nghiệm thu Phase 4A.2 còn thiếu (`research/evidence/phase-4a.2/PHASE_REPORT.md`).
7. Đảm bảo toàn bộ mạng ngoại vi giữ ở mức `0 bytes` (Zero-Egress), không tải model, không chạy huấn luyện khống.
8. Kết thúc bằng một yêu cầu phê duyệt tải dữ liệu duy nhất (`NEXT APPROVAL REQUEST`) có tên file, dung lượng, và mục đích cụ thể.

---

## 2. Công việc Đã Hoàn thành

### 2.1. Backfill Báo cáo Phase 4A.2 Còn Thiếu
- Đã tạo `research/evidence/phase-4a.2/PHASE_REPORT.md` (112 dòng) tổng hợp trung thực toàn bộ kết quả khảo sát metadata TGIF/TGIF2, đóng băng phạm vi 3 lớp, hệ thống câu hỏi RQ1–RQ4 và bằng chứng `EV-CONTINUITY-001` đến `EV-PILOT-PROPOSAL-001`.

### 2.2. Audit Ngữ nghĩa Nhãn và Thẩm định Thành phần TGIF
- **Cổng thẩm định nhãn (Label-Semantics Gate)**:
  - `authentic`: Ảnh chụp từ cảm biến thực tế, không chứa pixel tạo sinh (MS-COCO `orig`, ImageNet val `nature`).
  - `fully_generated`: Toàn bộ nội dung ảnh ($100\%$ pixel) được sinh từ mô hình tạo sinh từ nhiễu hoặc văn bản, **không bắt đầu từ một ảnh thật cần giữ nguyên danh tính nội dung**.
  - `ai_edited`: Ảnh bắt đầu từ một ảnh thật, sau đó một phần hoặc toàn bộ canvas được biến đổi bằng generative inpainting/editing có điều kiện từ ảnh nguồn.
- **Thẩm định `sp` vs `fr` của TGIF**:
  - `sp` (Spliced): Vùng trong mask là inpainting của SD2/Firefly, vùng ngoài mask là ảnh MS-COCO thật $100\%$. Được xác minh tuyệt đối ($100\%$) là nhãn **`ai_edited`** và dùng cho bài toán localization với mask chuẩn.
  - `fr` (Fully Regenerated): Inpainter xuất toàn bộ canvas mà không ghép lại. Mặc dù mọi pixel đều đi qua diffusion/VAE, ảnh này **vẫn bắt đầu từ ảnh MS-COCO thật và giữ nguyên cấu trúc/danh tính nội dung của ảnh thật**.
  - **Kết luận phương pháp luận**: `fr` **KHÔNG ĐỦ CĂN CỨ để gọi là `fully_generated`**. Gán `fr` là `fully_generated` sẽ gây mâu thuẫn nhận thức cho mô hình. Thành phần `fr` được phân loại là `ai_edited (conditional regeneration)` và **tạm thời bị cách ly (quarantined) khỏi manifest huấn luyện ban đầu**.

### 2.3. Thiết kế Kiến trúc Pilot Hai Nhánh Độc lập
- **Pilot A (Authentic vs AI-Edited + Localization)**:
  - Nguồn: TGIF `orig` (3,124 ảnh thật), TGIF `sd2-sp` (74,976 ảnh chỉnh sửa), TGIF `masks` (3,124 binary mask).
  - Ưu điểm: Triệt tiêu hoàn toàn rủi ro cross-dataset shortcut vì cả thật và giả đều xuất phát từ cùng ảnh MS-COCO (matched pairs).
  - Ràng buộc: Chia tập Train/Val/Test strictly theo khóa `source_id`.
  - Chỉ số: Macro-F1, Balanced Accuracy, AUROC, mIoU, Dice, Pixel AUROC.
- **Pilot B (Authentic vs Fully-Generated)**:
  - Nguồn: GenImage BigGAN (ImageNet val authentic vs BigGAN fake).
  - Ưu điểm: Phân loại nhị phân trong cùng phân phối ImageNet.
  - Chỉ số: Macro-F1, Balanced Accuracy, AUROC, Cross-Generator F1 Drop.
- **Pilot C (Three-Class Exploratory)**:
  - Điều kiện: Chỉ được mở khi Pilot A và Pilot B vượt qua bài kiểm toán shortcut.
  - Ràng buộc: Bắt buộc đo kèm **Metadata-Only Baseline**. Nếu visual model không vượt trội baseline metadata $\ge 15\%$, kết quả bị gắn cờ shortcut learning.

### 2.4. Triển khai Cấu hình Máy đọc và Validator
- Đã tạo `ml/configs/pilot_tgif_edit.yaml` (Pilot A) và `ml/configs/pilot_genimage_generated.yaml` (Pilot B).
- Đã xây dựng `ml/configs/validator.py` với khả năng kiểm định tự động 13 trường bắt buộc, ràng buộc chống rò rỉ `group_key: source_id`, kiểm tra nhãn chuẩn tắc, và từ chối nghiêm ngặt việc gán `fr` sang `fully_generated`.
- Đã cập nhật `ml/datasets/acquire.py` hỗ trợ lệnh dry-run chuyên biệt `--pilot pilot-a` và `--pilot pilot-b` in đầy đủ 12 trường thông tin cần thiết.

---

## 3. Kết quả Kiểm thử & Bản dựng Thực tế

| Bộ kiểm thử | Lệnh thực thi | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :--- |
| **TypeScript Monorepo Tests** | `pnpm test` | 57/57 tests passing trên 6 package (@forensics/shared: 34, @forensics/provenance: 3, @forensics/report: 4, @forensics/forensics: 5, @forensics/inference: 8, web: 3) | `PASS` |
| **Python ML Tests** | `pytest ml/tests -v` | 23/23 tests passing (contamination guard: 10, model pipeline: 5, pilot protocol & label gate: 8) | `PASS` |
| **Pilot Config Validation** | `python -m ml.configs.validator --validate-all` | 2/2 pilot configs valid, passed all label semantics gates | `PASS` |
| **Pilot A Dry-Run** | `python -m ml.datasets.acquire --pilot pilot-a` | Exit code 0, 12 fields printed, 0 bytes network | `PASS` |
| **Pilot B Dry-Run** | `python -m ml.datasets.acquire --pilot pilot-b` | Exit code 0, 12 fields printed, 0 bytes network | `PASS` |
| **Production Web Build** | `pnpm build` | Exit code 0 trong 3.33s; bundle Vite hợp lệ (worker: 437.93 kB, WASM runtime: 28.3 MB) | `PASS` |
| **Clean Link Invariance** | Link grep scan | 0 machine-local links (`file:///`, `C:\`, `D:\`) trong markdown links của repository | `PASS` |

---

## 4. Hạch toán Mạng & Dữ liệu Ngoại vi

- **Metadata Requests**: `0` (sử dụng inventory đã kiểm kê từ Phase 4A.1 và Phase 4A.2).
- **External Dataset Bytes Downloaded**: `0 bytes`.
- **Model Checkpoint Bytes Downloaded**: `0 bytes`.
- **Training Runs Executed**: `0`.

---

## 5. Tình trạng Tuyên bố Khoa học (Scientific Claims Audit)

- **Supported Claims**:
  - Giao thức thí nghiệm hai nhánh độc lập (Pilot A, Pilot B) và điều kiện mở Pilot C.
  - Phân định ranh giới phương pháp luận của 3 nhãn và lý do loại trừ `fr` khỏi `fully_generated`.
  - Cơ chế kiểm soát rò rỉ dữ liệu `source_id` group split và deduplication.
- **Unsupported Claims (Tiếp tục giữ trạng thái unverified / not evaluated)**:
  - Không có bất kỳ tuyên bố độ chính xác (Accuracy), F1-score, hay ECE nào được đưa ra.
  - Localization mIoU/Dice tiếp tục mang trạng thái `not evaluated` cho đến khi có dữ liệu thật.

---

## 6. Danh mục Minh chứng Bổ sung (Evidence Register)

- `EV-LABEL-GATE-001` (`science`, `verified`): Cổng thẩm định 3 nhãn khoa học chuẩn tắc.
- `EV-TGIF-SEMANTICS-001` (`dataset`, `verified`): Thẩm định ngữ nghĩa `sp` (ai_edited) và `fr` (quarantined).
- `EV-PILOT-DESIGN-001` (`science`, `verified`): Thiết kế kiến trúc pilot hai nhánh độc lập A/B/C.
- `EV-SHORTCUT-PROTOCOL-001` (`science`, `verified`): Giao thức chống rò rỉ và kiểm soát shortcut.
- `EV-PILOT-CONFIGS-001` (`pipeline`, `verified`): Cấu hình pilot YAML máy đọc và validator tự động.
- `EV-ACQUISITION-DRYRUN-001` (`pipeline`, `verified`): Dry-run thu nạp dữ liệu cho Pilot A và Pilot B đạt 0 bytes ngoại vi.

---

## 7. Các Blocker và Quyết định Cần Người dùng Phê duyệt

1. **Khóa tải nội dung dữ liệu (Content Download Lock)**: Lệnh thu nạp dữ liệu thật đang bị khóa (`content download execution: DISABLED`).
2. **Quyết định tiếp theo**: Người dùng xem xét và phê duyệt `NEXT APPROVAL REQUEST` dưới đây để tiến hành tải dữ liệu thật phục vụ Pilot A hoặc Pilot B trong giai đoạn tiếp theo.
