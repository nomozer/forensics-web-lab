# Biên bản Bàn giao Phiên làm việc (Session Handoff)

> **Cập nhật lúc**: 2026-09-21 (Phase 4A.0)  
> **Tài liệu này được cập nhật ở cuối mỗi phiên làm việc để đảm bảo tính liên tục và lưu trữ ngữ cảnh trong Git.**

---

- **Branch**: `feat/production-ai-image-forensics`
- **Starting commit**: `e36cdb6`
- **Base main commit**: `460f6d5` (giữ nguyên tuyệt đối)
- **Chiến lược áp dụng**: **DUAL-TRACK — Tách nghiên cứu và sản phẩm**
  - Luồng nghiên cứu (Research Track): dataset phi thương mại được dùng cho benchmark/thí nghiệm học thuật; trọng số huấn luyện tuyệt đối không đưa vào bản web sản phẩm.
  - Luồng sản phẩm (Product Track): chỉ huấn luyện từ dữ liệu có quyền sử dụng và phân phối trọng số rõ ràng (`product-eligible`).
  - Cách ly tuyệt đối: không làm nhiễm dữ liệu, checkpoint hoặc metric giữa hai luồng.
- **Completed (Phase 4A.0 — Data License Correction, Track Isolation and Acquisition Dry-Run)**:
  - Sửa lỗi giấy phép:
    - GenImage: `CC BY-NC-SA 4.0 with additional dataset terms` (cấm mục đích thương mại cho cả dataset và derivative works/trọng số phái sinh, xếp vào `research-only`).
    - RealHD: `availability: unavailable-or-pending` ("Coming soon" trên GitHub), `license: unverified`, `decision: blocked`.
    - Loại bỏ hoàn toàn tên gọi sai "GenImage Mini"; chuẩn hóa thành `Custom smoke subset sampled from GenImage` / `Project-defined GenImage smoke subset`.
    - Tách biệt hoàn toàn license của source code và license của dataset image.
  - Kiến trúc tách hai luồng (ADR 0006):
    - Ban hành `docs/adr/0006-research-product-data-isolation.md`.
    - Thiết lập cây thư mục: `data/research/`, `data/product/`, `artifacts/research/`, `artifacts/product/`, `models/research/`, `models/product/`.
    - Cập nhật `.gitignore` cách ly chặt chẽ dữ liệu nhị phân, ảnh, weights, archives khỏi Git; chỉ cho phép commit README, manifests, schemas, checksums, provenance.
  - Dataset Registry máy đọc & Schemas:
    - Tạo `datasets/registry.json`, `docs/schemas/dataset-registry.v1.schema.json`, `datasets/README.md`.
    - Tạo schema provenance `docs/schemas/dataset-manifest.v1.schema.json` với 14 trường định danh.
    - Triển khai validator `validateDatasetRegistry` và `validateProductionModelLineage` trong `@forensics/shared`.
  - Acquisition CLI Dry-Run:
    - Hoàn thiện `python -m ml.datasets.acquire` hỗ trợ `--dry-run`, `--dataset`, `--track`, `--validate-registry`.
    - Chặn `--execute`, từ chối dataset `blocked`, từ chối đưa `research-only` dataset vào `product` track. Không tải bất kỳ byte nào.
  - Bộ sinh Fixture giả lập mã nguồn thuần túy:
    - Tạo `ml/tests/fixtures/smoke_generator.py` sinh các ảnh hình học, gradient, noise và mask nhị phân phục vụ kiểm thử pipeline/loader.
    - Ghi chú rõ: không phải ảnh thật, không dùng đo accuracy, không sinh metric khoa học.
  - Contamination Guards & Automated Tests:
    - 10 unit tests trong `packages/shared/src/__tests__/dataset-registry.test.ts`.
    - 6 unit tests trong `ml/tests/test_contamination_guard.py`.
    - Kiểm tra fail-closed cho mọi trường hợp vi phạm bản quyền và phân luồng.
  - Hiệu chỉnh các tuyên bố kỹ thuật:
    - Tuyên bố Zero Server Egress: chuyển thành `architecture-supported`, `runtime-network-verification: unverified` (backlog `TASK-E2E-001`).
    - Ngân sách Web: WASM asset ~28.3 MB được ghi nhận là engine runtime binary (`ort-wasm-simd-threaded.jsep.wasm`), không phải model size (backlog `TASK-WEB-001`).
  - Evidence Register:
    - Bổ sung các evidence ID: `EV-LICENSE-GENIMAGE-001`, `EV-LICENSE-REALHD-001`, `EV-DATA-REGISTRY-001`, `EV-DATA-ISOLATION-001`, `EV-ACQUIRE-DRYRUN-001`, `EV-CONTAMINATION-001`, `EV-SMOKE-FIXTURE-001`, `EV-ZEROEGRESS-001`, `EV-WASM-BUDGET-001`, `EV-SCIENCE-002` trong `docs/EVIDENCE_REGISTER.md` và `research/evidence/phase-4a.0/evidence-manifest.json`.
- **Verified**:
  - `pnpm test`: 51/51 tests passing trên 6 package (shared: 28, provenance: 3, report: 4, forensics: 5, inference: 8, web: 3).
  - `pnpm build`: Thành công 100%, exit code 0.
  - `ml/.venv/Scripts/python -m pytest ml/tests -v`: 11/11 tests passing.
  - `python -m ml.datasets.acquire --dataset genimage --track research --dry-run`: Pass exit code 0.
  - Acquisition negative cases (`--track product`, `--dataset realhd`, `--execute`): Đều bị reject đúng luật.
  - Zero bytes downloaded (không tải ảnh, không tải archive, không tải checkpoint).
  - `main` không bị thay đổi (`460f6d5`), remote `origin` nguyên vẹn, không push, không deploy.
- **Model status**: `not trained` (kiến trúc `ARCH-C2-INHOUSE-MNV3`; `models/registry.json` ở trạng thái `status: not-trained`, `sizeBytes: 0`, `path: ""`).
- **Dataset status**: `none` (0 byte dữ liệu được tải về, dry-run only).
- **Known blockers**:
  - Chờ quyết định của người dùng về việc lựa chọn tập dữ liệu nghiên cứu, kích thước tải và phê duyệt tải dữ liệu cho Phase 4A.
- **Next exact task**:
  - Báo cáo kết quả Phase 4A.0 cho người dùng và dừng lại chờ chỉ thị tiếp theo.

