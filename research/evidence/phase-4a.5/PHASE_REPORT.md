# Báo cáo Nghiệm thu Giai đoạn Phase 4A.5 (Phase 4A.5 Completion Report)

> **Giai đoạn**: Phase 4A.5 — Model-Agnostic Continuity Enforcement  
> **Repository**: `nomozer/forensics-web-lab`  
> **Branch**: `feat/production-ai-image-forensics`  
> **Starting commit**: `a480776`  
> **Implementation snapshot commit**: `311cfd3`  
> **Base main commit**: `460f6d5` (bảo toàn nguyên vẹn, không commit trực tiếp)  
> **Evidence generated at UTC**: `2026-09-21T14:48:00Z`  
> **Working tree**: clean  
> **Tuyên bố khoa học tối thượng**: `No scientific model-performance claim is currently supported.`  
> **Phán quyết giai đoạn (Verdict)**: **`PASS`**

---

## 1. Mục tiêu Giai đoạn

1. **Đóng chính xác Phase 4A.4**: Ghi nhận ending commit thực tế `a480776` trên toàn bộ tài liệu continuity và báo cáo Phase 4A.4; thay thế thuật ngữ "Zero-Leakage Invariant" bằng "group-isolation invariant"; đối chiếu mã băm SHA-256 của `datasets/acquisition-plans/pilot-a-tgif.v1.json` (`7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`).
2. **Thiết lập Continuity Contract trong `AGENTS.md`**: Quy định thứ tự đọc bắt buộc khi bắt đầu phiên làm việc, ma trận cập nhật tài liệu khi thay đổi mã nguồn, và quy trình kết thúc phase chuẩn mực.
3. **Triển khai Continuity Checker độc lập**: Xây dựng công cụ cross-platform `scripts/continuity-check.mjs` chạy qua `pnpm continuity:check`, hỗ trợ kiểm tra uncommitted changes, `--staged`, và `--base <commit> --head <commit>`.
4. **Kiểm thử Toàn diện Bộ Kiểm tra**: Viết 13 bài test unit và contract test trong `scripts/__tests__/continuity-check.test.mjs` trên các fixture git cô lập (bao phủ 10 kịch bản kiểm thử bắt buộc).
5. **Tích hợp CI GitHub Actions**: Bổ sung bước kiểm tra continuity tự động vào `.github/workflows/ci.yml` với `fetch-depth: 0`.
6. **Duy trì Cấu trúc Ba File Duy nhất**: Đảm bảo `docs/continuity/` chỉ chứa đúng 3 file Markdown tiêu chuẩn, không có bản sao đánh số hay session pack trùng lặp.

---

## 2. Kết quả Triển khai Chi tiết

### 2.1. Đóng Chính xác Phase 4A.4 & Chuẩn hóa Thuật ngữ
- Đã cập nhật ending commit thực tế `a480776` vào:
  - `research/evidence/phase-4a.4/PHASE_REPORT.md`
  - `research/evidence/phase-4a.4/evidence-manifest.json`
  - `docs/continuity/CURRENT_STATE.md`
  - `docs/continuity/STATUS_LEDGER.md`
- Đã thay thế "Zero-Leakage Invariant" thành "group-isolation invariant" vì group split chỉ chứng minh các partition không chia sẻ `source_id`, không tuyên bố loại bỏ 100% mọi dạng rò rỉ dữ liệu hay bias tiềm ẩn.
- Đã xác thực SHA-256 của `datasets/acquisition-plans/pilot-a-tgif.v1.json` đạt khớp 100% với:
  `7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e`.

### 2.2. Ban hành Continuity Contract trong `AGENTS.md`
Mục 4 của `AGENTS.md` đã được ban hành với các điều khoản ràng buộc:
- **Thứ tự đọc bắt buộc khi bắt đầu session**:
  1. `docs/continuity/CURRENT_STATE.md`
  2. `docs/continuity/CODE_INDEX.md`
  3. `docs/continuity/STATUS_LEDGER.md`
  4. `research/evidence/<latest-phase>/PHASE_REPORT.md`
- **Ma trận cập nhật file theo loại thay đổi**:
  - Trạng thái model/dataset/metric/test/blocker: yêu cầu `CURRENT_STATE.md`.
  - Thêm/xóa/đổi tên module/schema/contract/CLI/architecture: yêu cầu `CODE_INDEX.md` và `CURRENT_STATE.md`.
  - Tạo mới `PHASE_REPORT.md`: yêu cầu `CURRENT_STATE.md` và `STATUS_LEDGER.md`.
  - Sửa typo/format đơn thuần ngoài phạm vi trạng thái: không yêu cầu.
- **Quy trình kết thúc phase**: Test/build $\rightarrow$ Cập nhật 3 file $\rightarrow$ Tạo evidence $\rightarrow$ Chạy continuity checker $\rightarrow$ Commit khi PASS.

### 2.3. Triển khai Continuity Checker (`scripts/continuity-check.mjs`)
- Công cụ Node.js ESM thuần túy, hoạt động đồng nhất trên Windows, Linux và CI.
- Tự động nhận diện ngữ cảnh:
  - Mặc định: kiểm tra working tree (uncommitted + untracked) hoặc commit gần nhất.
  - `--staged`: kiểm tra các thay đổi trong git index.
  - `--base <commit> --head <commit>`: so sánh diff giữa 2 commits (dùng cho CI PR/Push).
- Rà soát toàn bộ các ràng buộc:
  - Đúng 3 file trong `docs/continuity/`.
  - 0 bản sao đánh số (`CURRENT_STATE(1).md`, v.v.).
  - Phase mới nhất trên `STATUS_LEDGER.md` khớp thư mục evidence mới nhất.
  - `CURRENT_STATE.md` tham chiếu đúng phase mới nhất.
  - `PHASE_REPORT.md` mới nhất tồn tại và không rỗng.
  - 0 liên kết máy cá nhân (`file:///`, `C:\`, `D:\`).
  - 0 placeholder chưa giải quyết (`See repository HEAD`, `<commit>`, `Xác định sau commit báo cáo`).
  - Mỗi phase trên ledger không vượt quá 20 dòng.

### 2.4. Kiểm thử Tự động (`scripts/__tests__/continuity-check.test.mjs`)
13 bài test unit cô lập đã kiểm thử thành công 10/10 kịch bản:
1. Sửa typo trên file non-state (`README.md`) $\rightarrow$ PASS mà không yêu cầu cập nhật file continuity.
2. Sửa file trạng thái model trong `ml/` $\rightarrow$ FAIL khi thiếu `CURRENT_STATE.md`.
3. Thêm module mới trong `packages/` $\rightarrow$ FAIL khi thiếu `CURRENT_STATE.md` hoặc `CODE_INDEX.md`.
4. Tạo phase report mới $\rightarrow$ FAIL khi thiếu `CURRENT_STATE.md` hoặc `STATUS_LEDGER.md`.
5. Tạo phase report kèm module mới $\rightarrow$ FAIL khi thiếu bất kỳ file nào trong cả ba file.
6. Thiếu latest phase report $\rightarrow$ FAIL.
7. Có file continuity trùng lặp/đánh số $\rightarrow$ FAIL.
8. Có placeholder `See repository HEAD` $\rightarrow$ FAIL.
9. Có đường dẫn máy cá nhân $\rightarrow$ FAIL.
10. Cập nhật đầy đủ hợp lệ $\rightarrow$ PASS.
11. Bổ sung: Đoạn phase trong `STATUS_LEDGER.md` vượt quá 20 dòng $\rightarrow$ FAIL.
12. Bổ sung: Thuật toán so sánh và sắp xếp phase tokens (`comparePhaseTokens`) đạt độ chính xác tuyệt đối.

### 2.5. Tích hợp CI GitHub Actions
Cập nhật `.github/workflows/ci.yml`:
- Cấu hình `fetch-depth: 0` để bảo đảm cây commit đầy đủ cho lệnh diff.
- Thêm bước kiểm tra tự động `Model-Agnostic Continuity Check` chạy `pnpm continuity:check -- --base <base> --head <head>`.

---

## 3. Tổng hợp Kiểm thử & Đo đạc Kỹ thuật

```text
============================= TEST EXECUTION SUMMARY =============================
Vitest (Workspace Packages & Web App):
  * Total Test Files:        6 passed (6)
  * Total Tests:             57 passed (57)
  * Failed Tests:            0
  * Test Duration:           ~7.5s

Continuity Checker Test Suite (Node.js Native Test Runner):
  * Total Test Files:        1 passed (1)
  * Total Tests:             13 passed (13)
  * Failed Tests:            0
  * Test Duration:           ~2.67s

Pytest (Python ML Suite):
  * Total Test Files:        4 passed (4)
  * Total Tests:             38 passed (38)
    - ml/tests/test_acquisition_safety.py:  15 passed
    - ml/tests/test_contamination_guard.py: 10 passed
    - ml/tests/test_model_pipeline.py:       5 passed
    - ml/tests/test_pilot_protocol.py:       8 passed
  * Failed Tests:            0
  * Test Duration:           13.61s

Config Validator:
  * Pilot Configs Validated: 2 (pilot_tgif_edit.yaml, pilot_genimage_generated.yaml)
  * Status:                  All valid, label semantics gate passed.

Downloader Safety Dry-Run:
  * Plan:                    datasets/acquisition-plans/pilot-a-tgif.v1.json
  * SHA-256:                 7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e
  * Status:                  PASS (0 bytes downloaded, execution locked)

Web Application Build:
  * Exit Code:               0
  * Build Duration:          3.08s
  * Dist Artifacts:
    - dist/index.html:                                 1.08 kB
    - dist/assets/forensics.worker-D3P75d9E.js:      437.93 kB
    - dist/assets/ort-wasm-simd-threaded.jsep-*.wasm: 28.31 MB
    - dist/assets/index-oqrv6H5y.js:                 683.69 kB
    - dist/assets/index-CJ6LiDni.css:                  3.97 kB

Continuity Checker Execution:
  * Command:                 pnpm continuity:check
  * Exit Code:               0
  * Status:                  CONTINUITY_CHECK: PASS
==================================================================================
```

### Hạch toán Mạng (Network Accounting):
* Metadata requests: `0`
* Metadata response bytes: `0`
* Dataset content requests: `0`
* Dataset content bytes: `0`
* Model weights requests: `0`
* Model weights bytes: `0`
* Training runs executed: `0`

---

## 4. Đánh giá Tiêu chí Nghiệm thu (Pass/Fail Verdict)

| Tiêu chí | Yêu cầu | Thực tế đạt được | Phán quyết |
| :--- | :--- | :--- | :--- |
| **Phase 4A.4 Ending Commit** | Ending commit ghi đúng `a480776` | `a480776` đã cập nhật đầy đủ | **PASS** |
| **Acquisition Plan SHA-256** | Khớp `7da36f450fe4...` | Khớp chính xác 100% | **PASS** |
| **Group-Isolation Invariant** | Thay thế Zero-Leakage Invariant | Đã thay thế chuẩn xác | **PASS** |
| **Three Continuity Files** | Đúng 3 file trong `docs/continuity/` | Đúng 3 file, 0 file thừa | **PASS** |
| **No Duplicate Files** | 0 file đánh số/sao lưu | Đã quét toàn repo: 0 vi phạm | **PASS** |
| **Continuity Contract** | Ban hành tại `AGENTS.md` | Đã ban hành Mục 4 rõ ràng | **PASS** |
| **Continuity Checker** | `scripts/continuity-check.mjs` | Hoàn thiện, hỗ trợ 3 chế độ | **PASS** |
| **Checker Unit Tests** | 10 kịch bản bắt buộc | 13/13 unit tests passing | **PASS** |
| **CI Integration** | Tích hợp `.github/workflows/ci.yml` | Đã cấu hình chạy tự động | **PASS** |
| **External Dataset Bytes** | Bắt buộc `0 bytes` | `0 bytes` | **PASS** |
| **Model Weights Bytes** | Bắt buộc `0 bytes` | `0 bytes` | **PASS** |
| **Training Runs** | Bắt buộc `0` | `0` | **PASS** |
| **Full Test Suite** | 70 TS/JS tests + 38 Python tests | 108 tests passing (100%) | **PASS** |
| **Web Build** | Exit code 0, không lỗi compile | Exit code 0, hoàn tất trong 3.08s | **PASS** |
| **Git Safety** | Main `460f6d5` giữ nguyên, không push/deploy | Main bảo toàn nguyên vẹn, working tree clean | **PASS** |

**KẾT LUẬN GIAI ĐOẠN**: **`PASS`**

---

## 5. Yêu cầu Phê duyệt Tiếp theo (NEXT APPROVAL REQUEST)

```text
Plan ID: pilot-a-tgif
Plan path: datasets/acquisition-plans/pilot-a-tgif.v1.json
Expected SHA-256:
7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e

Dataset content: TGIF orig + masks + sd2-sp
Download size: 25,919,643,647 bytes (~24.14 GB)
Required free disk: >= 59,055,800,320 bytes (~55 GB)
Destination: data/research/tgif
Track: research-only
Purpose: Pilot A classification and localization
```

Lệnh tải chỉ được trình bày để chờ người dùng phê duyệt:

```bash
python -m ml.datasets.acquire \
  --plan datasets/acquisition-plans/pilot-a-tgif.v1.json \
  --execute \
  --approved-plan-sha256 7da36f450fe424970e4676fc0c35047ea756385843dd2fb1c656f1fa45deac4e
```
