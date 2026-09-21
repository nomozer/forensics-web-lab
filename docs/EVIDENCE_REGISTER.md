# Sổ Quản lý Minh chứng Nghiên cứu và Kỹ thuật (Evidence Register)

> **Dự án**: `forensics-web-lab`  
> **Phiên bản Schema**: `docs/schemas/evidence-manifest.v1.schema.json`  
> **Manifest máy đọc**: `research/evidence/phase-3.6/evidence-manifest.json`  
> **Cập nhật lần cuối**: Phase 3.6 (Commit `cd59136`)

---

## 1. Nguyên tắc Quản lý Minh chứng & Phân loại Trạng thái

Hệ sinh thái nghiên cứu và sản phẩm của **Forensics Web Lab** tuân thủ nguyên tắc trung thực khoa học tuyệt đối (*Scientific Honesty*). Mọi tuyên bố kỹ thuật, hiệu năng, kích thước hoặc độ chính xác đều phải được phân loại theo tập 8 trạng thái minh chứng chuẩn:

| Trạng thái | Định nghĩa chuẩn | Điều kiện công nhận |
| :--- | :--- | :--- |
| `verified` | Đã có minh chứng trực tiếp, đo lường thật và có thể tái tạo 100%. | Phải có artifact cụ thể, lệnh chạy tái tạo, kết quả đo thật trong repository. |
| `reported` | Được báo cáo trong y văn/tài liệu tham khảo trước đây nhưng chưa kiểm chứng độc lập. | Có trích dẫn tài liệu gốc, ghi rõ chưa kiểm chứng trên codebase nội bộ. |
| `estimated` | Giá trị ước lượng lý thuyết (tính toán toán học), chưa đo trên artifact thật. | Công thức tính toán rõ ràng, chỉ rõ sai số biên khả dĩ. |
| `architecture-only` | Mới tồn tại ở tầng thiết kế kiến trúc hoặc mã nguồn PyTorch/TS, chưa huấn luyện. | Mã nguồn khởi tạo module có thể chạy, không gọi là checkpoint. |
| `pipeline-only` | Chỉ chứng minh pipeline kỹ thuật/dòng chảy dữ liệu hoạt động với dữ liệu giả định. | Unit test pipeline đạt; không suy diễn tính đúng đắn trên trọng số thật. |
| `unverified` | Chưa có minh chứng hoặc chưa thực hiện đo lường. | Ghi nhận trung thực `unverified` hoặc `not measured`. |
| `blocked` | Bị chặn do thiếu dữ liệu, trọng số, giấy phép bản quyền hoặc tài nguyên phần cứng. | Nêu rõ phụ thuộc đang chặn và điều kiện gỡ chặn. |
| `rejected` | Đã đánh giá toàn diện và kết luận không phù hợp với tiêu chuẩn dự án. | Nêu lý do loại trừ (kích thước quá lớn, giấy phép hạn chế, v.v.). |

---

## 2. Bảng Đăng ký Minh chứng Chính quy (Phase 3.6 Evidence Register)

| Evidence ID | Claim | Category | Status | Artifact | Reproduction command | Result | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-GIT-001** | Trạng thái Git nằm trên nhánh `feat/production-ai-image-forensics`, `main` không bị can thiệp tại `460f6d5`, remote nguyên vẹn. | `git-state` | `verified` | [environment.json](research/evidence/phase-3.6/environment.json) | `git status --short && git branch --show-current && git rev-parse HEAD && git rev-parse main && git remote -v` | Nhánh `feat/production-ai-image-forensics`, HEAD `cd59136`, main `460f6d5`, working tree sạch trước Phase 3.6, remote `origin` không đổi. | Chỉ kiểm toán workspace cục bộ; không xác minh đồng bộ upstream GitHub. |
| **EV-TEST-TS-001** | Bộ kiểm thử TypeScript đạt 100% trên toàn bộ 6 package monorepo. | `test-ts` | `verified` | [test-summary.json](research/evidence/phase-3.6/test-summary.json) | `pnpm test` | 40/40 tests vượt qua trên `@forensics/shared`, `@forensics/provenance`, `@forensics/report`, `@forensics/forensics`, `@forensics/inference`, và `web`. | Chạy trên Node.js/jsdom thông qua Vitest; chưa thay thế cho kiểm thử E2E đa trình duyệt vật lý. |
| **EV-TEST-PY-001** | Bộ kiểm thử Python ML đạt 100% trên kiến trúc mô hình, hàm mất mát, manifest dữ liệu, chia nhóm chống rò rỉ và cân chỉnh nhiệt độ. | `test-py` | `verified` | [test-summary.json](research/evidence/phase-3.6/test-summary.json) | `ml/.venv/Scripts/python -m pytest ml/tests -v` | 5/5 tests vượt qua trong 5.73s (`test_mobilenetv3_forward_shape`, `test_focal_loss_computation`, `test_manifest_csv_roundtrip`, `test_group_split_prevents_leakage`, `test_temperature_scaler_and_ece`). | Chạy trên fixture tensor giả lập/tổng hợp; chưa đánh giá độ hội tụ khi huấn luyện trên dữ liệu thật. |
| **EV-BUILD-001** | Bản dựng production web bundle thành công với TypeScript strict mode và Vite. | `build` | `verified` | [build-summary.json](research/evidence/phase-3.6/build-summary.json) | `pnpm build` | Thoát mã 0 trong 2.99s, tạo `dist/index.html` (1.08 kB), `forensics.worker.js` (437.93 kB), và tài nguyên ONNX WASM (28.3 MB). | Xác minh đóng gói tài nguyên tĩnh; chưa chứng minh độ mượt WebGPU/WASM khi tải thực tế. |
| **EV-NOMODEL-001** | Hệ thống trả về kết luận `uncertain`, `confidence: null`, `probabilities: null` khi chưa cài đặt mô hình; cấm sinh kết luận AI giả. | `no-model` | `verified` | [contracts.test.ts](packages/shared/src/__tests__/contracts.test.ts) | `pnpm test` | Contract tests xác minh trạng thái no-model trung thực và từ chối mọi đối tượng vi phạm tự ý gán nhãn AI. | Áp dụng ở tầng contract, worker fusion, UI và export report; không ngăn người dùng hiểu nhầm tín hiệu DSP khám phá. |
| **EV-REPORT-001** | Báo cáo xuất bản (JSON / HTML Print) công bố rõ ràng trạng thái "Model not installed" và không tạo điểm tin cậy AI giả khi thiếu mô hình. | `no-model` | `verified` | [report.test.ts](packages/report/src/__tests__/report.test.ts) | `pnpm test` | Module báo cáo sinh cấu trúc hợp lệ theo schema JSON v1.0 và giao diện in ấn công bố rõ ràng mô hình không khả dụng. | Kiểm thử tính logic và câu chữ hiển thị; chưa kiểm toán bản in trên máy in vật lý. |
| **EV-REGISTRY-001** | Bộ kiểm định Registry từ chối mô hình chưa xác minh/mô hình giả và cấm chuyển trạng thái `ready` nếu thiếu các tiêu chí bắt buộc. | `registry` | `verified` | [registry-validator.ts](packages/shared/src/registry-validator.ts) | `pnpm test` | Validator ép buộc đủ 7 điều kiện cho trạng thái `ready` (path, sha256, sizeBytes, task contract, runtime compatibility, license, evaluation status) và chặn mock/dummy. | Xác minh cấu trúc schema và sự tồn tại của file; chưa kiểm định tính đúng đắn toán học sâu của ma trận trọng số. |
| **EV-ONNX-001** | Pipeline xuất ONNX và kiểm thử sai số số học (parity) hoạt động bình thường trên kiến trúc mạng nơ-ron. | `onnx-pipeline` | `pipeline-only` | [export_onnx.py](ml/export/export_onnx.py) | `ml/.venv/Scripts/python -m pytest ml/tests/test_model_pipeline.py -v` | Mã pipeline tồn tại và chạy trơn tru trên tensor kích thước chuẩn; sai số trên checkpoint thật là `unverified`. | Chưa có checkpoint huấn luyện thật; sai số số học thật, suy giảm do lượng tử hóa INT8 và lỗi opset runtime chỉ đo được sau khi huấn luyện. |
| **EV-RUNTIME-001** | Tài nguyên ONNX Runtime Web WASM đã được đóng gói và kiểm chứng thực thi trên Chrome, Edge, Firefox, Safari. | `browser-runtime` | `unverified` | [ort-session.ts](packages/inference/src/ort-session.ts) | `pnpm build` | File WASM (28.3 MB) nằm trong bundle `apps/web/dist`; việc thực thi suy luận trực tiếp trên cả 4 trình duyệt lớn chưa được kiểm chứng vật lý. | Bị chặn do chưa có file mô hình ONNX đã huấn luyện và môi trường thử nghiệm đa trình duyệt thực tế. |
| **EV-SIZE-001** | Kích thước mô hình MobileNetV3 nội bộ sau lượng tử hóa INT8 đạt khoảng 2.6 MB. | `model-size` | `estimated` | [PRETRAINED_MODEL_CANDIDATES.md](docs/PRETRAINED_MODEL_CANDIDATES.md) | `ml/.venv/Scripts/python -c "from ml.training.mobilenetv3_forensics import MobileNetV3Forensics; m = MobileNetV3Forensics(); p = sum(p.numel() for p in m.parameters()); print(f'Params: {p}')"` | Kiến trúc có 2,544,115 tham số; ước lượng lý thuyết 1 byte/param cho ra ~2.43 - 2.6 MB; chưa có file ONNX INT8 thực tế để cân đo. | Hoàn toàn là ước lượng toán học; kích thước file thực tế phụ thuộc vào metadata đồ thị ONNX, schema lượng tử hóa và bảng trọng số. |
| **EV-LICENSE-001** | Repository và trọng số mô hình tương lai có giấy phép mở được xác minh (ví dụ: MIT / Apache-2.0). | `license` | `unverified` | [LICENSING.md](docs/LICENSING.md) | `git log -1 docs/LICENSING.md` | Giấy phép repository đang chờ quyết định của chủ dự án; giấy phép trọng số mô hình hiện tại là `not-applicable` do chưa có trọng số. | Quyền phân phối trọng số sau này phụ thuộc đồng thời vào giấy phép mã nguồn, giấy phép backbone tiền huấn luyện và giấy phép dữ liệu huấn luyện. |
| **EV-DATA-001** | Các bộ dữ liệu huấn luyện học thuật (GenImage, SAGI-D, RealHD, RAID) đã được tải xuống và thẩm định trong repository. | `dataset` | `blocked` | [DATASETS.md](docs/DATASETS.md) | `git status --short data/` | 0 byte dữ liệu được tải về; việc thu thập dữ liệu bị chặn nghiêm ngặt để chờ người dùng cấp phép rõ ràng. | Dữ liệu cục bộ lát cắt nhỏ (<50 MB) chỉ dùng cho smoke test pipeline; nghiên cứu khoa học bắt buộc phải có tập dữ liệu đầy đủ được duyệt. |
| **EV-SCIENCE-001** | Hệ thống sở hữu độ chính xác phát hiện, khả năng tổng quát hóa trên generator chưa thấy hoặc định vị inpainting đã đo lường. | `science` | `unverified` | [EVALUATION.md](docs/EVALUATION.md) | `git log -1 docs/EVALUATION.md` | Các chỉ số Macro F1, ECE, Brier score, và mIoU đều được ghi nhận trung thực là `not measured` / `not evaluated`; không có tuyên bố khoa học nào được đưa ra. | Không thể tuyên bố năng lực phát hiện khi chưa có trọng số mô hình thật được đánh giá trên tập chuẩn độc lập, không rò rỉ. |

---

## 3. Quy trình Tái tạo và Kiểm chứng bằng Máy

Mọi minh chứng trong bảng trên đều có thể được xác minh tự động thông qua công cụ kiểm thử hợp đồng trong `@forensics/shared`:

```bash
# Kiểm tra tính toàn vẹn của manifest máy đọc so với schema
pnpm --filter @forensics/shared test
```

Test tự động `contracts.test.ts` sẽ đọc trực tiếp tệp `research/evidence/phase-3.6/evidence-manifest.json`, xác minh schema, kiểm tra cấm đường dẫn tuyệt đối của máy cá nhân, cấm trùng lặp ID, và ngăn chặn việc gán trạng thái `verified` cho các kết quả chưa được đo lường thực tế.
