# Erratum: Phase 4C.2C.0 Completed Stage 2 Results Report

> **Document**: `research/evidence/phase-4c.2c.1/PHASE_4C2C0_ERRATUM.md`
> **Target Document**: `research/evidence/phase-4c.2c.0/PHASE_REPORT.md` (Section 7)
> **Date of Erratum**: 2026-10-01
> **Verdict**: `INGEST_AUDIT_INTACT_DOCUMENTATION_ERRATUM_PUBLISHED`

---

## 1. Tóm tắt Sự cố (Executive Summary)

Trong quá trình kiểm toán đối soát giai đoạn **Phase 4C.2C.1**, một mâu thuẫn số liệu đã được phát hiện giữa bảng Markdown tại **Mục 7** của `research/evidence/phase-4c.2c.0/PHASE_REPORT.md` và các số liệu tính toán chuẩn tắc trong `research/evidence/phase-4c.2c/paired_run_metrics.csv`.

Kiểm toán nguồn gốc nhị phân (cryptographic byte-level audit) trên archive gốc `execution_9ee7fdb_complete_results.tar.gz` (SHA-256 `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`) và 15 thư mục run thô tại `execution_9ee7fdb` đã xác nhận dứt khoát:

1. **Các artifact thô hoàn toàn nguyên vẹn và nhất quán 100%**: Tất cả 15 runs có `run_receipt.json`, `metrics.json`, `predictions.json`, `epoch_history.json` và `best_checkpoint.pt` khớp nhau chính xác từng bit (sai số parity $< 10^{-6}$).
2. **Bản chất của mâu thuẫn**: Bảng Markdown Mục 7 trong báo cáo `phase-4c.2c.0/PHASE_REPORT.md` được soạn thảo thủ công trong phiên làm việc trước đó và đã ghi chép các con số nháp/placeholder không phản ánh các giá trị thực tế trong receipt hay predictions.
3. **Tính hợp lệ của quá trình Ingestion**: Quá trình kiểm toán nén TAR, bảo mật giải nén, kiểm tra rò rỉ locked-test (0 sample), và xác thực 91 inner-validation sources trong Phase 4C.2C.0 **hoàn toàn chính xác và giữ nguyên hiệu lực**.
4. **Nguyên tắc bảo toàn lịch sử**: Báo cáo gốc `phase-4c.2c.0/PHASE_REPORT.md` được **giữ nguyên trạng** không sửa chữa hậu nghiệm; văn bản đính chính này (Erratum) đóng vai trò là chứng thực chuẩn tắc.

---

## 2. Bảng Đối chiếu Chi tiết 15 Runs (Discrepancy vs Canonical Table)

| Cohort $N$ | Seed | Báo cáo C.0 (Mục 7) Epoch | Canonical Raw Best Epoch | Báo cáo C.0 Epochs Completed | Canonical Raw Epochs Completed | Báo cáo C.0 Val Macro-F1 | Canonical Raw / Phase 4C.2C Macro-F1 | Bản chất Chênh lệch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 50 | 42 | 16 | 7 | 20 | 12 | 0.5401 | 0.554932 | Sai lệch soạn thảo C.0 |
| 50 | 1337 | 8 | 12 | 13 | 17 | 0.5057 | 0.554609 | Sai lệch soạn thảo C.0 |
| 50 | 2025 | 1 | 4 | 6 | 9 | 0.5445 | 0.537959 | Sai lệch soạn thảo C.0 |
| 50 | 3407 | 1 | 1 | 6 | 6 | 0.5366 | 0.520806 | Sai lệch soạn thảo C.0 |
| 50 | 9001 | 2 | 5 | 7 | 10 | 0.5387 | 0.539816 | Sai lệch soạn thảo C.0 |
| 100 | 42 | 2 | 5 | 7 | 10 | 0.5739 | 0.565921 | Sai lệch soạn thảo C.0 |
| 100 | 1337 | 15 | 15 | 20 | 20 | 0.5694 | 0.575028 | Sai lệch soạn thảo C.0 |
| 100 | 2025 | 4 | 8 | 9 | 13 | 0.5518 | 0.609312 | Sai lệch soạn thảo C.0 |
| 100 | 3407 | 3 | 7 | 8 | 12 | 0.5471 | 0.560386 | Sai lệch soạn thảo C.0 |
| 100 | 9001 | 4 | 9 | 9 | 14 | 0.5692 | 0.572261 | Sai lệch soạn thảo C.0 |
| 250 | 42 | 4 | 9 | 9 | 14 | 0.5627 | 0.557823 | Sai lệch soạn thảo C.0 |
| 250 | 1337 | 3 | 3 | 8 | 8 | 0.5543 | 0.567195 | Sai lệch soạn thảo C.0 |
| 250 | 2025 | 7 | 12 | 12 | 17 | 0.5619 | 0.568787 | Sai lệch soạn thảo C.0 |
| 250 | 3407 | 2 | 2 | 7 | 7 | 0.5684 | 0.586902 | Sai lệch soạn thảo C.0 |
| 250 | 9001 | 3 | 6 | 8 | 11 | 0.5539 | 0.564870 | Sai lệch soạn thảo C.0 |

---

## 3. Xác minh Nguồn Chuẩn tắc (Proof of Invariance)

- **Archive gốc**: `execution_9ee7fdb_complete_results.tar.gz`
  - SHA-256: `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`
  - Size: `83,796,910 bytes`
- **Độ khớp nội tại giữa các artifact thô**:
  - `receipt.best_val_macro_f1 == receipt.final_metrics.macro_f1`: **15/15 MATCH**
  - `receipt.final_metrics.macro_f1 == metrics.json.macro_f1`: **15/15 MATCH**
  - `metrics.json.macro_f1 == recomputed_predictions.macro_f1`: **15/15 MATCH**
  - `metrics.json.macro_f1 == epoch_history_max_macro_f1`: **15/15 MATCH**
  - `metrics.json.macro_f1 == checkpoint.metrics.macro_f1`: **15/15 MATCH**
  - `receipt.best_epoch == checkpoint.epoch`: **15/15 MATCH**

## 4. Kết luận

Các phân tích đối chứng ghép cặp trong **Phase 4C.2C** và **Phase 4C.2C.1** đã sử dụng chính xác các artifact thô từ `execution_9ee7fdb`, đảm bảo tính trung thực khoa học tuyệt đối. Số liệu trong bảng Mục 7 của `phase-4c.2c.0/PHASE_REPORT.md` là lỗi soạn thảo văn bản và không làm ảnh hưởng đến tính toàn vẹn của kết quả nghiên cứu.
