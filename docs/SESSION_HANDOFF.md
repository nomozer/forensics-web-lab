# Biên bản Bàn giao Phiên làm việc (Session Handoff)

> **Cập nhật lúc**: 2026-09-21 (Phase 4A.1)  
> **Tài liệu này được cập nhật ở cuối mỗi phiên làm việc để đảm bảo tính liên tục và lưu trữ ngữ cảnh trong Git.**

---

- **Branch**: `feat/production-ai-image-forensics`
- **Starting commit**: `a0e71d5`
- **Base main commit**: `460f6d5` (giữ nguyên tuyệt đối)
- **Chiến lược áp dụng**: **DUAL-TRACK — Tách nghiên cứu và sản phẩm**
  - Luồng nghiên cứu (Research Track): dataset phi thương mại được dùng cho benchmark/thí nghiệm học thuật; trọng số học từ GenImage bị cấm đưa vào sản phẩm theo chính sách bảo thủ của dự án.
  - Luồng sản phẩm (Product Track): chỉ huấn luyện từ dữ liệu có quyền sử dụng và phân phối trọng số thương mại rõ ràng (`product-eligible`).
  - Phân định dữ liệu thử nghiệm: `synthetic-smoke` được định danh là `fixture-only`, không tham gia sản phẩm, không sinh metric khoa học.
- **Completed (Phase 4A.1 — Acquisition Feasibility and Residual Evidence Correction)**:
  1. **Đính chính mức độ minh chứng và giấy phép tồn dư**:
     - GenImage derivative weights: tách bạch `source_terms` (prohibited for commercial use), `legal_interpretation` (`trained_weights_status: unclear`), `project_policy` (`production_use: prohibited`, `research_use: allowed_subject_to_terms`). Ghi nhận `derivativeWeights: unclear`, `productionPromotion: prohibited-by-project-policy`.
     - `synthetic-smoke`: chuyển sang track `fixture-only`, purpose `fixture`, `commercialUse: internal-testing-only`, `ownership: project-generated`, `licenseStatus: pending-project-license-decision`. Cập nhật validator cấm đi vào product model lineage.
     - Dataset chưa xác minh (`sagi-d`, `raid`, `realhd`): đặt `track: blocked`, `status: proposed/blocked`, `licenseStatus: unverified`, `acquisitionEnabled: false`.
     - Synthetic smoke fixture `--execute`: ghi nhận trong Evidence Register là local deterministic fixture generation, 0 network requests, 0 external bytes.
     - `.gitignore`: đính chính mức độ bảo vệ thành "verified against the tested extension and path matrix".
  2. **Chuẩn hóa mục đích dataset**:
     - Bổ sung enum `purpose`: `fixture`, `acquisition-smoke`, `exploratory-pilot`, `scientific-benchmark`, `product-training` vào JSON Schema và TypeScript validator.
  3. **Khảo sát Metadata Remote GenImage (Google Drive)**:
     - Truy cập thư mục chính thức `1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS` và thư mục con `BigGAN` `1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt`.
     - Xác lập 16 remote items trong `research/evidence/phase-4a.1/genimage-remote-inventory.json`.
     - Định dạng lưu trữ: multi-part zip split volumes (`.z01` .. `.z07`, `.zip`), không hỗ trợ tải lẻ từng ảnh.
  4. **Xác định tính khả thi của tập con (Subset Feasibility)**:
     - Kết luận: **Conclusion B** (phải tải archive chính thức rồi mới lấy mẫu; bác bỏ giả định `< 50 MB` trước đây).
     - Archive nhỏ nhất: BigGAN split archive gồm 8 files, tổng dung lượng nén ~24 GB (~23.5 GB), giải nén ~26 GB, yêu cầu tối thiểu 55 GB đĩa trống.
  5. **Xây dựng Đề xuất Tiếp nhận 3 Cấp (Acquisition Proposal)**:
     - Tạo `docs/GENIMAGE_ACQUISITION_PROPOSAL.md`:
       - Cấp A — Acquisition smoke (10–50 ảnh, `pipeline-only`).
       - Cấp B — Exploratory pilot (500–1,000 ảnh, `exploratory, not publication-grade`).
       - Cấp C — Scientific benchmark (toàn bộ split, pre-registered protocol).
  6. **Cập nhật Acquisition CLI & Khóa Tải**:
     - `python -m ml.datasets.acquire` hỗ trợ `--metadata-only`.
     - Cờ `--execute` với external dataset bị khóa an toàn trả về exit code 1 và thông báo yêu cầu người dùng phê duyệt archive/byte cụ thể.
  7. **Kiểm tra Liên kết Sạch (Link Invariance)**:
     - Không có machine-local links (`file:///`, `C:\`, `D:\`, `/Users/`, `/home/`) trong markdown link targets của toàn bộ documentation.
  8. **Evidence Register**:
     - Bổ sung: `EV-LICENSE-WEIGHTS-INTERPRETATION-001`, `EV-FIXTURE-TRACK-001`, `EV-GENIMAGE-REMOTE-METADATA-001`, `EV-GENIMAGE-SUBSET-FEASIBILITY-001`, `EV-EXTERNAL-DOWNLOAD-LOCK-001`, `EV-LOCAL-LINK-SCAN-001`.
- **Verified**:
  - `pnpm test`: 56/56 tests passing trên 6 package (shared: 33, provenance: 3, report: 4, forensics: 5, inference: 8, web: 3).
  - `pnpm build`: Exit code 0 thành công.
  - `ml/.venv/Scripts/python -m pytest ml/tests -v`: 15/15 tests passing.
  - `python -m ml.datasets.acquire --dataset genimage --track research --metadata-only`: Pass exit code 0.
  - Zero external dataset bytes downloaded (`external dataset bytes: 0`, `model bytes: 0`, `training runs: 0`).
  - `main` không bị thay đổi (`460f6d5`), remote `origin` nguyên vẹn, không push, không deploy.
- **Model status**: `not trained` (`ARCH-C2-INHOUSE-MNV3`; `models/registry.json` ở trạng thái `status: not-trained`, `sizeBytes: 0`).
- **Dataset status**: `none` (chưa tải external dataset; chỉ có fixture generator nội bộ).
- **Known blockers**:
  - Chờ người dùng phê duyệt đề xuất tại `docs/GENIMAGE_ACQUISITION_PROPOSAL.md`: Phê duyệt tải BigGAN archive (~24 GB) cho Research Track hoặc giữ nguyên `fixture-only`.
- **Next exact task**:
  - Trình báo cáo Phase 4A.1 cho người dùng và chờ quyết định phê duyệt.
