# Báo cáo Kiểm chứng Triển khai Thực tế (Implementation Truth Audit)

> **Ngày thực hiện**: 2026-09-21  
> **Commit kiểm toán**: `6555b02` (và các cập nhật của Phase 3.5)  
> **Nguyên tắc cốt lõi**: Đối chiếu trực tiếp mã nguồn và kết quả kiểm thử thực tế; không tin tưởng vào báo cáo mô tả; xác lập tính trung thực khoa học tuyệt đối.  
> **Các verdict hợp lệ**: `verified`, `partially-verified`, `not-implemented`, `misleading`.

---

## 1. Tổng quan Kết quả Đối chiếu

| STT | Câu hỏi Kiểm toán | Verdict | Tóm tắt Hiện trạng |
| :--- | :--- | :--- | :--- |
| 1 | Worker có thực sự gọi ONNX Runtime hay mới chỉ có contract? | `partially-verified` | Mã gọi ONNX thật tồn tại trong Worker nhưng đang ở trạng thái ngủ (dormant) vì chưa nạp checkpoint. |
| 2 | WASM assets có được tải và khởi tạo thật hay chỉ được copy vào bundle? | `partially-verified` | File nhị phân WASM được bundle thật bởi Vite, nhưng browser chưa fetch vào RAM khi chưa có model. |
| 3 | Khi không có model, hệ thống có tạo xác suất, verdict hoặc heatmap giả không? | `misleading` (xác suất/verdict) / `verified` (heatmap) | `fuseWithoutModel` từng tự tạo xác suất giả định và verdict `fully_generated`/`ai_edited`; Heatmap không bị làm giả. |
| 4 | Fusion calibrator có tham số calibration thật hay chỉ dùng giá trị mặc định? | `partially-verified` | Công thức toán học chuẩn xác nhưng các tham số ngưỡng và nhiệt độ $T$ là giá trị mặc định lý thuyết. |
| 5 | Heatmap hiện đến từ model prediction hay dữ liệu giả/heuristic? | `verified` | Heatmap đến từ model logits trên các patch; khi không có model thì `available: false`, không fake heatmap. |
| 6 | UI có phân biệt model-based evidence và heuristic evidence không? | `partially-verified` | Có phân tách component nhưng thẻ kết luận chưa làm rõ sự vắng mặt của model AI. |
| 7 | Report có ghi rõ model chưa được cài đặt hay không? | `partially-verified` | Ghi `modelId: none` nhưng thiếu cờ boolean `modelAvailable: false` cấp cao và chưa ẩn xác suất rỗng. |

---

## 2. Chi tiết Từng Tuyên bố Kiểm toán

### Tuyên bố 1: Worker thực hiện suy luận ONNX Runtime
- **Claim**: Web Worker (`forensics.worker.ts`) thực thi suy luận mô hình học sâu thông qua thư viện ONNX Runtime Web (`onnxruntime-web`).
- **Evidence file**:
  - `packages/inference/src/forensics.worker.ts` (dòng 12, 73–108)
  - `packages/inference/src/session-manager.ts` (dòng 34–97)
- **Evidence test**:
  - `packages/inference/src/__tests__/fusion.test.ts`
  - `apps/web/src/__tests__/app.test.tsx`
- **Verdict**: `partially-verified`
- **Phân tích chi tiết**: Mã gọi `ort.Tensor`, `session.run({ input: inputTensor })`, và vòng lặp trích xuất logits từng patch $224 \times 224$ là mã thực tế, đã biên dịch TypeScript nghiêm ngặt. Tuy nhiên, do `OnnxSessionManager.initializeSession()` chưa được cung cấp buffer trọng số nhị phân (trạng thái `models/registry.json` là `status: "not-trained"`), biến `hasModel` nhận giá trị `false`, khiến khối lệnh gọi `session.run` bị bỏ qua trong luồng chạy thực tế.
- **Required correction**: Giữ nguyên logic suy luận khi có model; bảo đảm khi `hasModel === false`, pipeline chuyển hướng rành mạch sang nhánh `fuseWithoutModel` minh bạch và an toàn.

---

### Tuyên bố 2: Khởi tạo và nạp tài nguyên WebAssembly (WASM)
- **Claim**: Các file nhị phân WASM của ONNX Runtime (`ort-wasm-simd-threaded.jsep-*.wasm`) được nạp và khởi tạo đa luồng CPU/SIMD trên trình duyệt.
- **Evidence file**:
  - `apps/web/vite.config.ts` (dòng 10–15: cấu hình `viteStaticCopy`)
  - `dist/assets/ort-wasm-simd-threaded.jsep-*.wasm` (được sinh ra sau khi chạy `pnpm build`)
  - `packages/inference/src/session-manager.ts` (dòng 71–74: cấu hình `ort.env.wasm`)
- **Evidence test**:
  - `pnpm build` xác nhận các file WASM (~23 MB) được sao chép chính xác vào thư mục `dist/assets/`.
- **Verdict**: `partially-verified`
- **Phân tích chi tiết**: Tài nguyên WASM được đóng gói hợp lệ trong bundle tĩnh của client. Tuy nhiên, cơ chế lazy loading của ONNX Runtime chỉ thực hiện HTTP GET tải file `.wasm` về trình duyệt khi phương thức `ort.InferenceSession.create()` được gọi với dữ liệu model hợp lệ. Khi chưa có file checkpoint, trình duyệt người dùng không tải file WASM này vào RAM.
- **Required correction**: Không cần sửa đổi cấu hình bundle; ghi nhận rõ trong tài liệu kiến trúc rằng WASM nạp theo cơ chế on-demand khi có checkpoint.

---

### Tuyên bố 3: Trạng thái trung thực khi không có mô hình (No-Model State)
- **Claim**: Hệ thống không sinh xác suất 3 lớp ngẫu nhiên, không tạo verdict giả định và không sinh heatmap giả khi chưa cài đặt checkpoint.
- **Evidence file**:
  - `packages/inference/src/fusion-calibrator.ts` (dòng 119–175)
- **Evidence test**:
  - `packages/inference/src/__tests__/fusion.test.ts` (test case `handles missing model gracefully`)
- **Verdict**: `misleading`
- **Phân tích chi tiết**:
  1. Trong `FusionCalibrator.fuseWithoutModel()`, mã nguồn cũ chứa logic gán xác suất giả định:
     - Nếu có AI metadata: tự ý gán `verdict = 'fully_generated'`, `confidence = 0.85`, và `probabilities: { no_ai_evidence: 0.1, fully_generated: 0.8, ai_edited: 0.1 }`.
     - Nếu phát hiện FFT và Noise cao: tự ý gán `verdict = 'ai_edited'`, `confidence = 0.68`, và `probabilities: { no_ai_evidence: 0.2, fully_generated: 0.2, ai_edited: 0.6 }`.
     - Trường hợp còn lại: gán `verdict = 'uncertain'`, `confidence = 0.5`, và `probabilities: { no_ai_evidence: 0.33, fully_generated: 0.33, ai_edited: 0.34 }`.
  2. Điều này vi phạm nghiêm trọng nguyên tắc trung thực khoa học: **Không có mô hình học sâu thì không thể đưa ra xác suất 3 lớp hay kết luận `fully_generated` / `ai_edited`**. Các tín hiệu DSP và metadata chỉ là dữ liệu khám phá.
  3. Về Heatmap: `HeatmapAccumulator.accumulate([])` xử lý đúng khi không có patch nào, trả về `available: false` và không sinh lưới nhiệt giả.
- **Required correction**: Sửa đổi toàn diện `FusionCalibrator.fuseWithoutModel()`:
  - Khi không có model (`hasModel === false`):
    - `label` bắt buộc là `'uncertain'`.
    - `confidence` bắt buộc là `null`.
    - `probabilities` bắt buộc là `null`.
    - `modelAvailable` là `false`.
    - `modelStatus` là `'not-installed'`.
    - Giải thích rõ: "Mô hình học sâu chưa được cài đặt (Model not installed). Đánh giá dựa trên tín hiệu phân tích DSP và siêu dữ liệu chỉ có tính chất khám phá sơ bộ, không cấu thành kết luận mô hình."

---

### Tuyên bố 4: Tham số hiệu chuẩn độ tin cậy (Calibration Parameters)
- **Claim**: Bộ trọng tài phân tích (`FusionCalibrator`) sử dụng các tham số hiệu chuẩn nhiệt độ và ngưỡng tin cậy đã được tối ưu hóa.
- **Evidence file**:
  - `packages/shared/src/constants.ts` (dòng 12–19)
  - `ml/evaluation/calibration.py` (dòng 1–65)
- **Evidence test**:
  - `packages/shared/src/__tests__/contracts.test.ts`
  - `ml/tests/test_model_pipeline.py`
- **Verdict**: `partially-verified`
- **Phân tích chi tiết**: Thuật toán Temperature Scaling và logic phân tách xác suất đã được cài đặt đầy đủ cả ở phía Python (`ml/`) và TypeScript (`packages/inference`). Tuy nhiên, do chưa tải dataset và chưa huấn luyện mô hình, các hằng số hiện thời (`UNCERTAIN_CONFIDENCE_THRESHOLD = 0.65`, `HIGH_CONFIDENCE_THRESHOLD = 0.85`, `MARGIN_CONFLICT_THRESHOLD = 0.15`, $T = 1.0$) là các tham số mặc định theo thiết kế kiến trúc lý thuyết (ADR-0004), chưa được tối ưu hóa bằng phương pháp cực tiểu hóa NLL trên tập kiểm định thực tế.
- **Required correction**: Giữ nguyên công thức và các ngưỡng mặc định; cập nhật tài liệu kỹ thuật ghi chú rõ đây là tham số lý thuyết ban đầu và sẽ được hiệu chuẩn thực nghiệm trong Phase 4.

---

### Tuyên bố 5: Nguồn gốc của bản đồ nhiệt (Heatmap Localization)
- **Claim**: Bản đồ nhiệt định vị vùng nghi vấn can thiệp phản ánh trực tiếp dự đoán của mô hình trên các patch không gian.
- **Evidence file**:
  - `packages/inference/src/heatmap-accumulator.ts` (dòng 23–103)
  - `packages/inference/src/forensics.worker.ts` (dòng 89–112)
- **Evidence test**:
  - `packages/inference/src/__tests__/fusion.test.ts` (test case `accumulates patch scores into a smooth heatmap grid`)
- **Verdict**: `verified`
- **Phân tích chi tiết**: Khi có mô hình, các patch $224 \times 224$ (stride 112) được đưa qua mạng để tính điểm nghi vấn can thiệp, sau đó được tích lũy vào ma trận $16 \times 16$ bằng kernel Gauss 2D. Khi không có mô hình, worker truyền mảng `scoredPatches` rỗng, `HeatmapAccumulator.accumulate([])` trả về `localization.available = false`, `regions = []`, không tạo bất kỳ dữ liệu nhiệt giả mạo nào.
- **Required correction**: Bổ sung thông điệp hướng dẫn người dùng trên giao diện `HeatmapViewer` giải thích rõ: bản đồ nhiệt tạm thời bị vô hiệu hóa do mô hình học sâu chưa được cài đặt.

---

### Tuyên bố 6: Phân định giữa Bằng chứng Mô hình và Tín hiệu Khám phá DSP
- **Claim**: Giao diện người dùng phân biệt rõ ràng giữa bằng chứng từ mô hình học sâu và tín hiệu phát hiện bất thường từ DSP/Metadata.
- **Evidence file**:
  - `apps/web/src/components/ResultVerdictCard.tsx`
  - `apps/web/src/components/ForensicsInspector.tsx`
  - `apps/web/src/components/ProvenanceInspector.tsx`
- **Evidence test**:
  - `apps/web/src/__tests__/app.test.tsx`
- **Verdict**: `partially-verified`
- **Phân tích chi tiết**: Giao diện hiển thị các phần tách biệt trong các thẻ kính (glass panels). Tuy nhiên, thẻ `ResultVerdictCard` chưa có trạng thái hiển thị rõ ràng khi hệ thống chạy ở chế độ "Chưa cài đặt mô hình", khiến người dùng có thể nhầm lẫn rằng các tín hiệu DSP đang đưa ra kết luận xác suất AI.
- **Required correction**:
  - Bổ sung banner cảnh báo `Model not installed (Mô hình chưa được cài đặt)` nổi bật trong `ResultVerdictCard`.
  - Thay thế các thanh phần trăm xác suất bằng thông báo giải thích rõ ràng khi không có model.
  - Trong `ForensicsInspector`, ghi chú rõ các chỉ số DSP là tín hiệu khám phá (exploratory heuristics), không cấu thành kết luận mô hình.

---

### Tuyên bố 7: Tính minh bạch trong Báo cáo Xuất bản (JSON & Print Report)
- **Claim**: Báo cáo JSON và bản in PDF ghi nhận trung thực trạng thái khả dụng của mô hình, không chứa dữ liệu giả mạo.
- **Evidence file**:
  - `packages/report/src/printable-report.ts`
  - `packages/report/src/json-exporter.ts`
  - `docs/schemas/analysis-output.v1.schema.json`
- **Evidence test**:
  - `packages/report/src/__tests__/report.test.ts`
- **Verdict**: `partially-verified`
- **Phân tích chi tiết**: Báo cáo HTML hiện tại ghi `Mô hình: none (none)`, nhưng mục 4 ô chỉ số thống kê trên cùng vẫn hiển thị tỷ lệ % dựa trên các trường `confidence` và `probabilities` giả định từ `fuseWithoutModel`. Báo cáo JSON schema v1 chưa có trường tường minh `modelAvailable` và `modelStatus`.
- **Required correction**:
  - Bổ sung trường `modelAvailable: false` và `modelStatus: "not-installed"` vào contract `AnalysisResult`.
  - Cho phép `confidence: null` và `probabilities: null`.
  - Cập nhật `PrintableReportGenerator` hiển thị `N/A` và banner thông báo khi không có mô hình.
