# Biên bản Bàn giao Phiên làm việc (Session Handoff)

> **Cập nhật lúc**: 2026-09-21  
> **Tài liệu này được cập nhật ở cuối mỗi phiên làm việc để đảm bảo tính liên tục và lưu trữ ngữ cảnh trong Git.**

---

- **Branch**: `feat/production-ai-image-forensics`
- **Starting commit**: `6555b02`
- **Ending commit**: `e1379b4` (Commit 2: `fix: enforce honest no-model analysis state`)
- **Completed**:
  - Thiết lập trí nhớ dự án bền vững trong Git: `AGENTS.md`, `docs/PROJECT_STATE.md`, `docs/CODE_MAP.md`, `docs/SESSION_HANDOFF.md`.
  - Thực hiện kiểm chứng toàn diện 7 câu hỏi trong `docs/IMPLEMENTATION_TRUTH_AUDIT.md`.
  - Khảo sát các ứng viên mô hình tiền huấn luyện trong `docs/PRETRAINED_MODEL_CANDIDATES.md`.
  - Thiết lập cổng kiểm định và tiếp nhận mô hình nghiêm ngặt trong `docs/MODEL_ACQUISITION_GATE.md`.
  - Sửa đổi mã nguồn để áp đặt trạng thái "No-Model Honest State": không sinh xác suất 3 lớp, không gán nhãn `fully_generated`/`ai_edited`, verdict là `uncertain`, `confidence: null`, `probabilities: null`, hiển thị banner "Model not installed".
  - Thêm kiểm tra tự động tính toàn vẹn của Model Registry (`validateModelRegistry`).
  - Cập nhật tài liệu tiến độ `docs/BACKLOG.md`.
- **Verified**:
  - Baseline & Final unit test suite: `pnpm test` (all packages passing).
  - Production bundle build: `pnpm build` (TypeScript strict check + Vite bundle thành công).
  - Python ML test suite: `pytest ml/tests -v` (5/5 tests passing).
  - Không có file binary model hay dataset nào bị tải về trái phép.
  - Working tree được đối chiếu và không thay đổi remote/main.
- **Not verified**:
  - Độ trễ thực tế của ONNX inference trên model thật trong trình duyệt (do chưa nạp checkpoint).
  - Kiểm thử đa trình duyệt thực tế (Safari WebGPU, Firefox WASM) trên thiết bị vật lý.
- **Model status**: `not trained` (chưa có checkpoint; `models/registry.json` ở trạng thái `status: not-trained`, `sizeBytes: 0`, `path: ""`).
- **Dataset status**: `none` (chưa tải dataset).
- **Known blockers**:
  1. Cần người dùng phê duyệt phương án checkpoint/dataset trước khi tiến hành tải hoặc huấn luyện.
  2. Các checkpoint từ bên thứ ba (LAID, TruFor, CNNDetection) đang bị vướng giấy phép (`blocked-license` hoặc `rejected`).
- **Next exact task**:
  - Trình duyệt báo cáo `PHASE_3_5_MODEL_READINESS_REPORT` cho người dùng.
  - Chờ quyết định của người dùng về việc lựa chọn huấn luyện nội bộ checkpoint nhẹ (`CAND-C2-INHOUSE-MNV3`) hay cấp phép tải lát cắt dữ liệu nhỏ cho Phase 4.
