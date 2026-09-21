# Biên bản Bàn giao Phiên làm việc (Session Handoff)

> **Cập nhật lúc**: 2026-09-21 (Phase 3.6)  
> **Tài liệu này được cập nhật ở cuối mỗi phiên làm việc để đảm bảo tính liên tục và lưu trữ ngữ cảnh trong Git.**

---

- **Branch**: `feat/production-ai-image-forensics`
- **Starting commit**: `cd59136`
- **Base main commit**: `460f6d5` (giữ nguyên tuyệt đối)
- **Completed (Phase 3.6 — Evidence Hardening and Claim Correction)**:
  - Kiểm toán và loại bỏ 100% đường dẫn tuyệt đối của máy cá nhân (`file:///`, `D:\`, `C:\`, `/Users/`, `/home/`) khỏi toàn bộ tài liệu và mã nguồn.
  - Sửa đổi toàn diện các tuyên bố phóng đại: đổi `CAND-C2-INHOUSE-MNV3` thành `ARCH-C2-INHOUSE-MNV3` (kiến trúc `architecture-only`, không phải checkpoint); kích thước INT8 ~2.6 MB phân loại là `estimated`; ONNX parity phân loại là `pipeline-only`; giấy phép trọng số là `not-applicable` do chưa tồn tại; chỉ số phát hiện/tổng quát hóa là `unverified` / `not measured`.
  - Thiết lập hệ thống quản lý minh chứng có thể kiểm tra bằng máy:
    - Sổ đăng ký: `docs/EVIDENCE_REGISTER.md`
    - Manifest máy đọc: `research/evidence/phase-3.6/evidence-manifest.json`
    - Schema JSON chuẩn hóa: `docs/schemas/evidence-manifest.v1.schema.json`
    - Tóm tắt môi trường & kết quả chạy thật: `environment.json`, `test-summary.json`, `build-summary.json`.
  - Nâng cấp `validateModelRegistry` trong `@forensics/shared` ép buộc đủ 7 tiêu chí bắt buộc trước khi chuyển sang `ready` (path, sha256, sizeBytes, task contract, runtime compatibility, license, evaluation status).
  - Triển khai `packages/shared/src/evidence-validator.ts` và test tự động đọc trực tiếp `evidence-manifest.json` từ đĩa, ngăn ngừa đường dẫn tuyệt đối và các kết quả giả định.
  - Mở rộng `docs/MODEL_ACQUISITION_GATE.md` thành 17 tiêu chí nghiêm ngặt trước khi một mô hình được nạp vào sản phẩm.
  - Soạn thảo `docs/PHASE_4A_DATA_FEASIBILITY.md` phân biệt rõ ràng pipeline smoke test (<50 MB) với huấn luyện khoa học, thiết lập ma trận quyết định dataset và cổng phê duyệt bắt buộc trước khi tải.
- **Verified**:
  - `pnpm test`: 40/40 tests passing trên 6 package (shared: 17, provenance: 3, report: 4, forensics: 5, inference: 8, web: 3).
  - `pnpm build`: Thành công 100%, mã thoát 0 trong 2.99s.
  - `ml/.venv/Scripts/python -m pytest ml/tests -v`: 5/5 tests passing trong 5.73s.
  - Không tải bất kỳ byte dataset hay model weights nào.
  - `main` không bị thay đổi (`460f6d5`), remote `origin` nguyên vẹn, không push, không deploy.
- **Not verified**:
  - Độ chính xác thực tế của mô hình phát hiện (do chưa có trọng số huấn luyện).
  - Độ trễ thực tế của suy luận trên thiết bị vật lý đa trình duyệt (Chrome, Safari, Edge, Firefox).
- **Model status**: `not trained` (chỉ có thiết kế kiến trúc `ARCH-C2-INHOUSE-MNV3`; `models/registry.json` ở trạng thái `status: not-trained`, `sizeBytes: 0`, `path: ""`).
- **Dataset status**: `none` (0 byte dữ liệu được tải về).
- **Known blockers**:
  - Cần người dùng phê duyệt đề xuất trong `docs/PHASE_4A_DATA_FEASIBILITY.md` trước khi tải bất kỳ dữ liệu nào.
- **Next exact task**:
  - Trình báo cáo nghiệm thu Phase 3.6 cho người dùng.
  - Dừng lại, KHÔNG tự ý triển khai Phase 4A cho đến khi nhận được chỉ thị cụ thể.
