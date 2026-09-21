# Code Map: Forensics Web Lab

> **Ngày cập nhật**: 2026-09-21  
> **Commit phản ánh**: `6555b02` (và các cập nhật của Phase 3.5)  
> **Quy tắc trạng thái**:  
> - `implemented-and-tested`: Đã cài đặt code đầy đủ và có unit test / integration test kiểm chứng tự động.  
> - `implemented-unverified`: Đã cài đặt code nhưng chưa có test tự động trực tiếp hoặc chưa chạy thực tế đầy đủ.  
> - `interface-only`: Mới chỉ có định nghĩa type/interface/contract TypeScript, chưa có implementation.  
> - `placeholder`: Code tạm giữ chỗ, giá trị giả định hoặc trả về cố định.  
> - `blocked`: Bị chặn bởi phụ thuộc bên ngoài (ví dụ: thiếu checkpoint model hoặc dataset).

---

## 1. Frontend Web App (`apps/web`)

### App Component
- **File**: `apps/web/src/App.tsx`
- **Responsibility**: Quản lý state toàn cục của giao diện người dùng, điều phối luồng phân tích từ ImageDropzone tới WorkerController, hiển thị kết quả và xử lý hủy bỏ.
- **Public API**: `export const App: React.FC`
- **Inputs**: Tương tác người dùng (kéo thả file ảnh, hủy bỏ, tạo phiên mới).
- **Outputs**: Render cây component React hiển thị quy trình và kết quả.
- **Dependencies**: `@forensics/shared`, `@forensics/inference`, các components con trong `apps/web/src/components/`.
- **Tests**: `apps/web/src/__tests__/app.test.tsx`
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần đảm bảo UI hiển thị rõ ràng thông báo "Model not installed" khi chưa có model AI.

### ImageDropzone
- **File**: `apps/web/src/components/ImageDropzone.tsx`
- **Responsibility**: Nhận file ảnh từ người dùng, kiểm tra magic bytes nhị phân (JPEG, PNG, WebP), giải mã ảnh an toàn ra RGBA và tính SHA-256 client-side.
- **Public API**: `export const ImageDropzone: React.FC<ImageDropzoneProps>`
- **Inputs**: Event drop/drag/file input HTML5.
- **Outputs**: Đối tượng `ValidatedImageData` (buffer, rgba, width, height, sha256).
- **Dependencies**: Web APIs (`FileReader`, `Image`, `crypto.subtle`).
- **Tests**: Kiểm thử gián tiếp qua `app.test.tsx`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Giới hạn kích thước ảnh tối đa 32 MB để chống cạn kiệt bộ nhớ browser.

### ResultVerdictCard
- **File**: `apps/web/src/components/ResultVerdictCard.tsx`
- **Responsibility**: Hiển thị thẻ kết luận giám định chính theo 4 trạng thái chuẩn (`no_ai_evidence`, `fully_generated`, `ai_edited`, `uncertain`), độ tin cậy hiệu chuẩn và thanh xác suất 3 lớp.
- **Public API**: `export const ResultVerdictCard: React.FC<ResultVerdictCardProps>`
- **Inputs**: `verdict`, `confidence`, `probabilities`, `explanation`, `modelAvailable`, `modelStatus`.
- **Outputs**: Render thẻ verdict với màu sắc trực quan và thang đo xác suất.
- **Dependencies**: `@forensics/shared`.
- **Tests**: Kiểm thử gián tiếp qua `app.test.tsx`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần xử lý trường hợp `confidence: null` và `probabilities: null` khi không có model, hiển thị banner "Model not installed".

### HeatmapViewer
- **File**: `apps/web/src/components/HeatmapViewer.tsx`
- **Responsibility**: Hiển thị ảnh gốc kèm lớp phủ bản đồ nhiệt (Turbo colormap) render trên Canvas HTML5, cho phép chỉnh độ mờ (opacity) và bật/tắt bounding boxes của các cụm nghi vấn.
- **Public API**: `export const HeatmapViewer: React.FC<HeatmapViewerProps>`
- **Inputs**: `previewUrl`, `width`, `height`, `localization: LocalizationResult`.
- **Outputs**: Canvas overlay và danh sách bounding boxes.
- **Dependencies**: `@forensics/shared`, `@forensics/inference`.
- **Tests**: Kiểm thử gián tiếp qua `app.test.tsx`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Khi không có model (`localization.available = false`), hiển thị thông báo bản đồ nhiệt cần model học sâu để kích hoạt.

### ForensicsInspector
- **File**: `apps/web/src/components/ForensicsInspector.tsx`
- **Responsibility**: Hiển thị chi tiết 4 chỉ số phân tích tín hiệu số (FFT 2D, DCT 2D, Laplacian noise residual, JPEG BAG), cho phép mở rộng xem chi tiết kỹ thuật.
- **Public API**: `export const ForensicsInspector: React.FC<ForensicsInspectorProps>`
- **Inputs**: `signals: ForensicSignal[]`.
- **Outputs**: Danh sách thẻ tín hiệu DSP có điểm số và phân tích.
- **Dependencies**: `@forensics/shared`.
- **Tests**: Kiểm thử qua render UI test.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần ghi chú rõ các chỉ số DSP là tín hiệu khám phá sơ bộ (exploratory heuristics), không phải kết luận tuyệt đối của mô hình.

### ProvenanceInspector
- **File**: `apps/web/src/components/ProvenanceInspector.tsx`
- **Responsibility**: Hiển thị trạng thái C2PA Content Credentials và bảng trích xuất thẻ EXIF/XMP kèm mức độ nghi vấn đối với chữ ký phần mềm AI.
- **Public API**: `export const ProvenanceInspector: React.FC<ProvenanceInspectorProps>`
- **Inputs**: `provenance: ProvenanceResult`.
- **Outputs**: Bảng thẻ metadata và nhãn trạng thái C2PA.
- **Dependencies**: `@forensics/shared`.
- **Tests**: Kiểm thử qua render UI test.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Chưa có validator C2PA WASM chính thức (đang ở trạng thái an toàn `unsupported`).

### EvidenceLedger
- **File**: `apps/web/src/components/EvidenceLedger.tsx`
- **Responsibility**: Trình bày sổ cái bằng chứng minh bạch: danh sách bằng chứng ủng hộ, bằng chứng phản đối/mâu thuẫn, và các giới hạn khoa học của hệ thống.
- **Public API**: `export const EvidenceLedger: React.FC<EvidenceLedgerProps>`
- **Inputs**: `supportingEvidence`, `refutingEvidence`, `limitations`.
- **Outputs**: Danh sách đối chiếu hai cột minh bạch.
- **Dependencies**: `@forensics/shared`.
- **Tests**: Kiểm thử qua render UI test.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### ReportActions
- **File**: `apps/web/src/components/ReportActions.tsx`
- **Responsibility**: Cung cấp nút tải báo cáo JSON (tuân thủ schema v1) và nút mở hộp thoại in / xuất PDF (`PrintableReportGenerator`).
- **Public API**: `export const ReportActions: React.FC<ReportActionsProps>`
- **Inputs**: `result: AnalysisResult`, callbacks `onClearSession`, `onNewImage`.
- **Outputs**: Nút bấm thao tác xuất báo cáo.
- **Dependencies**: `@forensics/report`.
- **Tests**: Kiểm thử qua render UI test.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

---

## 2. Shared Domain Contracts (`packages/shared`)

### Types & Domain Definitions
- **File**: `packages/shared/src/types.ts`
- **Responsibility**: Định nghĩa toàn bộ kiểu dữ liệu cốt lõi: `AnalysisVerdict` (4 trạng thái), `AnalysisResult`, `ClassProbabilities`, `LocalizationResult`, `ForensicSignal`, `ProvenanceResult`, `ModelRegistryItem`.
- **Public API**: Export toàn bộ TypeScript interfaces/types.
- **Inputs**: Không có.
- **Outputs**: Type definitions.
- **Dependencies**: Không có (Zero external dependencies).
- **Tests**: `packages/shared/src/__tests__/contracts.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần cập nhật `AnalysisResult` hỗ trợ `confidence: number | null`, `probabilities: ClassProbabilities | null`, `modelAvailable: boolean`, `modelStatus: string`.

### JSON Schemas & Validation
- **File**: `packages/shared/src/schemas.ts`
- **Responsibility**: Hàm kiểm tra tính hợp lệ `validateAnalysisResult()` bảo đảm object tuân thủ schema v1.0.0.
- **Public API**: `export function validateAnalysisResult(obj: unknown): obj is AnalysisResult`
- **Inputs**: Object bất kỳ.
- **Outputs**: Boolean type-guard.
- **Dependencies**: `packages/shared/src/types.ts`.
- **Tests**: `packages/shared/src/__tests__/contracts.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần cho phép `confidence` và `probabilities` có giá trị `null` khi không có model.

### Registry Validator
- **File**: `packages/shared/src/registry-validator.ts`
- **Responsibility**: Kiểm tra tính toàn vẹn của Model Registry: model có status `ready` bắt buộc phải có sha256 hợp lệ và sizeBytes > 0; model `not-trained`/`not-installed` không được trỏ path giả.
- **Public API**: `export function validateModelRegistry(registry: ModelRegistry, fsExistsSync?: (p: string) => boolean): { valid: boolean; errors: string[] }`
- **Inputs**: `ModelRegistry` object.
- **Outputs**: Kết quả hợp lệ và danh sách lỗi.
- **Dependencies**: `packages/shared/src/types.ts`.
- **Tests**: `packages/shared/src/__tests__/contracts.test.ts`.
- **Actual status**: `implemented-and-tested` (bổ sung trong Phase 3.5)
- **Known gaps**: Không có.

### Constants
- **File**: `packages/shared/src/constants.ts`
- **Responsibility**: Lưu trữ tên sản phẩm, tên đề tài, các ngưỡng hiệu chuẩn (`CALIBRATION_THRESHOLDS`), giới hạn chuẩn (`STANDARD_LIMITATIONS`).
- **Public API**: `PRODUCT_NAME`, `RESEARCH_TITLE`, `CALIBRATION_THRESHOLDS`, `STANDARD_LIMITATIONS`.
- **Inputs**: Không có.
- **Outputs**: Constants.
- **Dependencies**: Không có.
- **Tests**: `packages/shared/src/__tests__/contracts.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Các ngưỡng hiệu chuẩn hiện là giá trị mặc định theo thiết kế lý thuyết ADR-0004.

---

## 3. Browser Forensics DSP (`packages/forensics`)

### 2D-FFT Radial Anomaly Analysis
- **File**: `packages/forensics/src/fft2d.ts`
- **Responsibility**: Tính biến đổi Fourier nhanh 2D trên ma trận xám $128 \times 128$, shift phổ trung tâm, tính profile tích phân bán kính để phát hiện đỉnh bất thường do kiến trúc tích chập tạo sinh.
- **Public API**: `export function analyzeFft2d(gray: Float32Array, width: number, height: number): { score: number; details: Record<string, unknown> }`
- **Inputs**: Mảng xám Float32, chiều rộng, chiều cao.
- **Outputs**: Điểm bất thường (0.0 đến 1.0) và chi tiết radial profile.
- **Dependencies**: Thuần toán TypeScript.
- **Tests**: `packages/forensics/src/__tests__/dsp.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Kích thước ảnh đầu vào cho FFT được resize/crop về $128 \times 128$ để tối ưu CPU trong Web Worker.

### 2D-DCT Block Variance Analysis
- **File**: `packages/forensics/src/dct2d.ts`
- **Responsibility**: Chia ảnh thành các khối $8 \times 8$, tính biến đổi Cosine rời rạc 2D, đo tỷ lệ năng lượng giữa các thành phần tần số cao và tần số thấp để phát hiện nén kép hoặc suy giảm tần số.
- **Public API**: `export function analyzeDct2d(gray: Float32Array, width: number, height: number): { score: number; details: Record<string, unknown> }`
- **Inputs**: Mảng xám Float32, kích thước.
- **Outputs**: Điểm số bất thường DCT và tỷ lệ năng lượng HF/LF.
- **Dependencies**: Thuần toán TypeScript.
- **Tests**: `packages/forensics/src/__tests__/dsp.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Laplacian Noise Residual Analysis
- **File**: `packages/forensics/src/noise-analysis.ts`
- **Responsibility**: Lọc ảnh bằng toán tử Laplacian $3 \times 3$ để tách phần dư nhiễu vi mô, chia khối $32 \times 32$, tính phương sai nhiễu từng khối để đo mức độ bất đồng nhất (inconsistency) không gian.
- **Public API**: `export function analyzeNoiseInconsistency(gray: Float32Array, width: number, height: number): { score: number; details: Record<string, unknown> }`
- **Inputs**: Mảng xám Float32, kích thước.
- **Outputs**: Điểm số bất đồng nhất phương sai nhiễu và chi tiết min/max/mean variance.
- **Dependencies**: Thuần toán TypeScript.
- **Tests**: `packages/forensics/src/__tests__/dsp.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### JPEG Block Artifact Grid (BAG)
- **File**: `packages/forensics/src/jpeg-block-grid.ts`
- **Responsibility**: Đo độ không liên tục tại ranh giới các khối $8 \times 8$ so với các điểm ảnh lân cận bên trong khối để xác định dấu vết cấu trúc lưới nén JPEG.
- **Public API**: `export function analyzeJpegBlockGrid(gray: Float32Array, width: number, height: number): { score: number; details: Record<string, unknown> }`
- **Inputs**: Mảng xám Float32, kích thước.
- **Outputs**: Điểm số chênh lệch ranh giới khối (boundary difference score).
- **Dependencies**: Thuần toán TypeScript.
- **Tests**: `packages/forensics/src/__tests__/dsp.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Error Level Analysis (ELA)
- **File**: `packages/forensics/src/ela.ts`
- **Responsibility**: So sánh mức chênh lệch sai số nén.
- **Public API**: `export function computeEla(rgba: Uint8ClampedArray, width: number, height: number): { score: number; details: Record<string, unknown> }`
- **Inputs**: Mảng RGBA, kích thước.
- **Outputs**: Điểm số sai số ELA.
- **Dependencies**: Thuần toán TypeScript.
- **Tests**: `packages/forensics/src/__tests__/dsp.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

---

## 4. Metadata & Provenance (`packages/provenance`)

### EXIF Parser
- **File**: `packages/provenance/src/exif-parser.ts`
- **Responsibility**: Phân tích nhị phân các cấu trúc TIFF/EXIF trong file JPEG/PNG, trích xuất Make, Model, Software, DateTime, ColorSpace.
- **Public API**: `export function parseExif(buffer: ArrayBuffer): MetadataSummaryItem[]`
- **Inputs**: Buffer nhị phân của file ảnh.
- **Outputs**: Mảng `MetadataSummaryItem`.
- **Dependencies**: Không có.
- **Tests**: `packages/provenance/src/__tests__/exif-parser.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Hỗ trợ chuẩn TIFF/EXIF thông dụng, không parse toàn bộ các tag vendor maker-note phức tạp.

### XMP Parser
- **File**: `packages/provenance/src/xmp-parser.ts`
- **Responsibility**: Quét chuỗi XML XMP trong header nhị phân ảnh, trích xuất siêu dữ liệu phần mềm, CreatorTool, DigitalSourceType.
- **Public API**: `export function parseXmp(buffer: ArrayBuffer): MetadataSummaryItem[]`
- **Inputs**: Buffer nhị phân của file ảnh.
- **Outputs**: Mảng `MetadataSummaryItem`.
- **Dependencies**: Không có.
- **Tests**: `packages/provenance/src/__tests__/exif-parser.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Parse an toàn bằng biểu thức chính quy chuỗi UTF-8 để tránh XML injection trong browser.

### AI Software Signatures
- **File**: `packages/provenance/src/software-signatures.ts`
- **Responsibility**: Bảng tra cứu chữ ký phần mềm và công cụ AI (Midjourney, DALL-E, Stable Diffusion, ComfyUI, Photoshop Generative Fill).
- **Public API**: `export function matchSoftwareSignature(text: string): { matches: boolean; suspicionScore: number; note: string }`
- **Inputs**: Chuỗi tên công cụ/phần mềm.
- **Outputs**: Điểm nghi vấn và ghi chú.
- **Dependencies**: Không có.
- **Tests**: `packages/provenance/src/__tests__/exif-parser.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### C2PA Adapter
- **File**: `packages/provenance/src/c2pa-adapter.ts`
- **Responsibility**: Kiểm tra sự hiện diện của C2PA box trong header JUMBF. Hiện đặt trạng thái an toàn là `unsupported` nếu runtime chưa có WASM validator tin cậy.
- **Public API**: `export async function verifyC2pa(buffer: ArrayBuffer): Promise<{ status: C2paStatus; details: Record<string, unknown> }>`
- **Inputs**: Buffer nhị phân của file ảnh.
- **Outputs**: Trạng thái C2PA (`unsupported`, `absent`, `present_valid`, `present_invalid`).
- **Dependencies**: Không có.
- **Tests**: `packages/provenance/src/__tests__/exif-parser.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần tích hợp c2pa-js WASM chính thức khi có thư viện tương thích web nhẹ.

---

## 5. Inference & Evidence Fusion (`packages/inference`)

### Web Worker (`forensics.worker.ts`)
- **File**: `packages/inference/src/forensics.worker.ts`
- **Responsibility**: Thực thi toàn bộ pipeline phân tích bất đồng bộ trong background thread: xác thực magic bytes -> trích xuất provenance -> tính tín hiệu DSP -> kiểm tra ONNX session -> gom cụm heatmap -> hợp nhất bằng chứng (`FusionCalibrator`) -> gửi kết quả về main thread.
- **Public API**: Web Worker message listener (`self.onmessage`).
- **Inputs**: `WorkerAnalyzePayload`.
- **Outputs**: `COMPLETE` với `AnalysisResult`, hoặc `PROGRESS`, hoặc `ERROR`.
- **Dependencies**: `@forensics/shared`, `@forensics/provenance`, `@forensics/forensics`, `onnxruntime-web`.
- **Tests**: Kiểm thử gián tiếp qua `WorkerController` và unit test các submodule.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Hiện tại `hasModel` là `false` do chưa có file checkpoint nhị phân; cần đảm bảo trạng thái trung thực `modelAvailable: false`, `modelStatus: 'not-installed'`.

### WorkerController
- **File**: `packages/inference/src/worker-controller.ts`
- **Responsibility**: Wrapper hướng đối tượng phía main thread để quản lý vòng đời của Web Worker, hỗ trợ báo cáo tiến độ và cơ chế hủy bỏ (`AbortSignal`).
- **Public API**: `class WorkerController { analyzeImage(...), terminate() }`
- **Inputs**: `WorkerAnalyzePayload`, `AnalysisOptions`.
- **Outputs**: Promise trả về `AnalysisResult`.
- **Dependencies**: `@forensics/shared`.
- **Tests**: `apps/web/src/__tests__/app.test.tsx`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Patch Extractor
- **File**: `packages/inference/src/patch-extractor.ts`
- **Responsibility**: Trích xuất tensor ảnh toàn phần $224 \times 224$ (bilinear resize, normalize NCHW) và lưới các patch chồng lấn $224 \times 224$ (stride 112, 50% overlap, tối đa 48 patches).
- **Public API**: `PatchExtractor.extractGlobalTensor(...)`, `PatchExtractor.extractPatches(...)`
- **Inputs**: Mảng RGBA, width, height.
- **Outputs**: Float32Array tensor format NCHW.
- **Dependencies**: Không có.
- **Tests**: `packages/inference/src/__tests__/fusion.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Heatmap Accumulator
- **File**: `packages/inference/src/heatmap-accumulator.ts`
- **Responsibility**: Tích lũy điểm số nghi vấn của các patch vào lưới không gian $16 \times 16$ bằng kernel Gauss 2D, phát hiện các cụm nghi vấn cục bộ (`regions`), và render bảng màu Turbo ra RGBA cho Canvas.
- **Public API**: `HeatmapAccumulator.accumulate(...)`, `HeatmapAccumulator.renderColormapRgba(...)`
- **Inputs**: Mảng `ScoredPatch[]`.
- **Outputs**: `LocalizationResult` và bảng màu RGBA.
- **Dependencies**: `@forensics/shared`.
- **Tests**: `packages/inference/src/__tests__/fusion.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Khi không có model (`scoredPatches` rỗng), trả về `available: false`, không sinh heatmap giả.

### Fusion Calibrator
- **File**: `packages/inference/src/fusion-calibrator.ts`
- **Responsibility**: Hợp nhất đa nguồn bằng chứng (logits mô hình học sâu, cụm định vị heatmap, chỉ số DSP, siêu dữ liệu provenance) qua Temperature Scaling và trọng tài 4 trạng thái (`no_ai_evidence`, `fully_generated`, `ai_edited`, `uncertain`).
- **Public API**: `FusionCalibrator.fuse(inputs: FusionInputs): FusionOutput`
- **Inputs**: `FusionInputs` (rawLogits, localization, forensics, provenance, hasModel).
- **Outputs**: `FusionOutput`.
- **Dependencies**: `@forensics/shared`.
- **Tests**: `packages/inference/src/__tests__/fusion.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần cập nhật `fuseWithoutModel()` để luôn trả về `label: 'uncertain'`, `confidence: null`, `probabilities: null`, `modelAvailable: false`, `modelStatus: 'not-installed'`.

### OnnxSessionManager
- **File**: `packages/inference/src/session-manager.ts`
- **Responsibility**: Quản lý phiên suy luận ONNX Runtime Web, kiểm tra WebGPU adapter và tự động fallback về WASM/CPU đa luồng khi cần.
- **Public API**: `OnnxSessionManager.initializeSession(...)`, `OnnxSessionManager.getSession()`, `OnnxSessionManager.getBackend()`
- **Inputs**: `ModelRegistryItem`, `ArrayBuffer` model weights.
- **Outputs**: `SessionInitResult`.
- **Dependencies**: `onnxruntime-web`, `@forensics/shared`.
- **Tests**: Kiểm thử hợp nhất trong test suite.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Hiện tại chưa có file nhị phân model để khởi tạo session thực tế trong runtime production.

---

## 6. Report Generation (`packages/report`)

### Printable Report Generator
- **File**: `packages/report/src/printable-report.ts`
- **Responsibility**: Sinh tài liệu HTML báo cáo giám định hoàn chỉnh, độc lập, có print stylesheet tối ưu cho khổ A4 để người dùng in trực tiếp hoặc lưu PDF.
- **Public API**: `PrintableReportGenerator.generateHtml(result)`, `PrintableReportGenerator.printReport(result)`
- **Inputs**: `AnalysisResult`.
- **Outputs**: HTML string hoặc mở cửa sổ in ấn trình duyệt.
- **Dependencies**: `@forensics/shared`.
- **Tests**: `packages/report/src/__tests__/report.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cần hiển thị rõ thông báo "Mô hình chưa cài đặt" và hiển thị `N/A` cho các xác suất mô hình khi không có model.

### JSON Exporter
- **File**: `packages/report/src/json-exporter.ts`
- **Responsibility**: Xuất báo cáo JSON tuân thủ đầy đủ schema v1.0.0 và kích hoạt tải file client-side.
- **Public API**: `JsonReportExporter.exportToJson(result)`, `JsonReportExporter.downloadJson(result)`
- **Inputs**: `AnalysisResult`.
- **Outputs**: Chuỗi JSON hoặc trigger browser download file.
- **Dependencies**: `@forensics/shared`.
- **Tests**: `packages/report/src/__tests__/report.test.ts`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

---

## 7. Python Machine Learning Pipeline (`ml/`)

### Dataset Adapters
- **File**: `ml/datasets/adapters.py`
- **Responsibility**: Chuẩn hóa cấu trúc thư mục của 4 nguồn dữ liệu nghiên cứu (GenImage, SAGI-D, RealHD, RAID) về định dạng thống nhất: path, label (0=authentic, 1=fully_generated, 2=ai_edited), generator_family, source_id, mask_path.
- **Public API**: `GenImageAdapter`, `SagiDAdapter`, `RealHdAdapter`, `RaidAdapter`, `get_adapter(...)`
- **Inputs**: Thư mục gốc dữ liệu.
- **Outputs**: Danh sách `RawSample`.
- **Dependencies**: Standard Python libraries.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Chờ tải dữ liệu thực tế để chạy trên quy mô toàn bộ dataset.

### Perceptual Deduplication
- **File**: `ml/datasets/dedup.py`
- **Responsibility**: Loại bỏ ảnh trùng lặp hoặc biến thể gần trùng lặp bằng mã băm SHA-256 (trùng lặp chính xác) và pHash 64-bit DCT (ngưỡng khoảng cách Hamming $\le 4$).
- **Public API**: `Deduplicator.compute_sha256(...)`, `Deduplicator.compute_phash(...)`, `Deduplicator.find_duplicates(...)`
- **Inputs**: Danh sách đường dẫn ảnh.
- **Outputs**: Tập các đường dẫn ảnh trùng lặp cần loại bỏ.
- **Dependencies**: `PIL`, `scipy.fftpack`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Group-Based Leakage-Free Split
- **File**: `ml/datasets/split.py`
- **Responsibility**: Phân chia tập dữ liệu thành Train (70%), Val (15%), Test (15%) theo nhóm `source_id` để ngăn ngừa rò rỉ dữ liệu (data leakage) giữa các tập.
- **Public API**: `GroupSplitter.split_manifest(...)`
- **Inputs**: Danh sách samples có `source_id`.
- **Outputs**: 3 danh sách samples phân bổ vào train/val/test.
- **Dependencies**: Standard Python.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Realistic Degradation Augmentation
- **File**: `ml/datasets/augmentations.py`
- **Responsibility**: Áp dụng các biến dạng thực tế thường gặp trên mạng xã hội: nén JPEG (quality 40-95), làm mờ Gaussian blur, resize nội suy, Gaussian noise, và Color jitter.
- **Public API**: `get_forensics_augmentation_pipeline(is_training=True)`
- **Inputs**: Tensor ảnh PyTorch hoặc PIL Image.
- **Outputs**: Tensor ảnh đã qua biến dạng tăng cường.
- **Dependencies**: `torchvision.transforms.v2`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Model Architecture
- **File**: `ml/training/mobilenetv3_forensics.py`
- **Responsibility**: Xây dựng kiến trúc mạng MobileNetV3-Small nhẹ (~2.54M tham số, mã định danh kiến trúc `ARCH-C2-INHOUSE-MNV3`), điều chỉnh classifier head cho 3 lớp bài toán (`authentic`, `fully_generated`, `ai_edited`).
- **Public API**: `MobileNetV3Forensics(num_classes=3, pretrained=True)`
- **Inputs**: Tensor ảnh $[B, 3, 224, 224]$.
- **Outputs**: Logits $[B, 3]$.
- **Dependencies**: `torch`, `torchvision.models`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested` (mã kiến trúc và forward pass đã test với dummy tensor)
- **Known gaps**: Trọng số chưa được huấn luyện trên dataset thật (`not trained`); chưa có file checkpoint thật; khả năng phân loại và tổng quát hóa thực tế là `unverified`.

### Focal Loss & Class Imbalance
- **File**: `ml/training/loss.py`
- **Responsibility**: Cài đặt Multi-Class Focal Loss để giải quyết vấn đề mất cân bằng mẫu nghiêm trọng giữa ảnh thật và ảnh tạo sinh/chỉnh sửa.
- **Public API**: `MultiClassFocalLoss(alpha=None, gamma=2.0)`
- **Inputs**: Logits và ground-truth targets.
- **Outputs**: Giá trị loss vô hướng.
- **Dependencies**: `torch`, `torch.nn.functional`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Calibration (Temperature Scaling)
- **File**: `ml/evaluation/calibration.py`
- **Responsibility**: Tối ưu hóa tham số nhiệt độ $T$ bằng cách tối thiểu hóa Negative Log-Likelihood trên tập validation để đưa ra xác suất tin cậy thực tế (calibrated confidence).
- **Public API**: `ModelWithTemperature(model)`
- **Inputs**: PyTorch model và validation dataloader.
- **Outputs**: Mô hình đã gắn tham số nhiệt độ $T$ tối ưu.
- **Dependencies**: `torch`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested` (thuật toán đã test trên tensor tổng hợp)
- **Known gaps**: Chưa có tập validation thật; việc hiệu chuẩn trên dữ liệu thực nghiệm sẽ thực hiện ở Phase 4.

### ONNX Export & Contract Validation
- **File**: `ml/export/export_onnx.py`, `ml/export/quantize.py`, `ml/export/validate_contract.py`
- **Responsibility**: Xuất mô hình PyTorch sang định dạng ONNX (opset 17), lượng tử hóa INT8 (dynamic quantization), và kiểm tra độ lệch kết quả giữa PyTorch và ONNX ($L_\infty < 10^{-4}$).
- **Public API**: `export_to_onnx(...)`, `quantize_onnx_model(...)`, `validate_onnx_contract(...)`
- **Inputs**: PyTorch checkpoint `.pt`.
- **Outputs**: File ONNX unquantized `.onnx` và quantized `.onnx`.
- **Dependencies**: `torch`, `onnx`, `onnxruntime`.
- **Tests**: `ml/tests/test_model_pipeline.py`.
- **Actual status**: `implemented-and-tested` (đường ống xuất ONNX và parity test đã kiểm thử thành công trên mô hình un-trained)
- **Known gaps**: Chờ checkpoint huấn luyện thật từ Phase 4 để xuất file production; parity trên checkpoint thật là `unverified`.

---

## 8. Model Registry & Metadata (`models/`)

### Model Registry
- **File**: `models/registry.json`
- **Responsibility**: Bản kê khai danh mục các mô hình được hỗ trợ bởi hệ thống, định dạng input, số lớp, trạng thái vòng đời (`status: not-trained`).
- **Actual status**: `implemented-and-tested` (kiểm thử tính toàn vẹn qua `validateModelRegistry` trong test suite)
- **Known gaps**: `status: "not-trained"`, `path: ""`, `sizeBytes: 0`, `sha256: ""` (không có file nhị phân checkpoint nào được cài đặt trong production).

### Model Card
- **File**: `models/MODEL_CARD.md`
- **Responsibility**: Báo cáo chuẩn mực về thông số mô hình, kiến trúc, phạm vi ứng dụng, rủi ro, và bảng chỉ số thực nghiệm (chưa điền kết quả giả mạo, ghi rõ chưa huấn luyện).
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Bảng chỉ số thực nghiệm khoa học đang ở trạng thái `not evaluated` chờ kết quả thật từ Phase 4. Kích thước INT8 ghi nhận là `estimated` (~2.6 MB), chưa đo trên artifact thật.

---

## 9. Dataset Registry, Dual-Track & Contamination Guards (`datasets/`, `ml/datasets/`, `packages/shared`)

### Dataset Registry & Schemas
- **File**: `datasets/registry.json`, `docs/schemas/dataset-registry.v1.schema.json`, `docs/schemas/dataset-manifest.v1.schema.json`
- **Responsibility**: Sổ bộ quản lý toàn diện các dataset nghiên cứu và sản phẩm, khai báo chính sách bản quyền (commercialUse, redistribution, derivativeWeights), nguồn chính thức, bằng chứng pháp lý và phân luồng (`track: "research-only" | "product-eligible" | "blocked"`).
- **Schema**: `dataset-registry.v1.schema.json` (JSON Schema Draft 2020-12), `dataset-manifest.v1.schema.json` (14 trường truy vết nguồn gốc chống data leakage).
- **Tests**: `packages/shared/src/__tests__/dataset-registry.test.ts`, `ml/tests/test_contamination_guard.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Hiện tại chưa có dataset bên ngoài nào được tải về máy; các dataset lớn đang ở trạng thái quy hoạch và dry-run.

### Dataset & Contamination Validator (TypeScript)
- **File**: `packages/shared/src/dataset-validator.ts`
- **Responsibility**: Cung cấp hàm `validateDatasetRegistry` (kiểm tra 13 quy tắc toàn vẹn pháp lý của registry) và `validateProductionModelLineage` (ngăn chặn tuyệt đối model sản phẩm tham chiếu dataset phi thương mại hoặc chưa rõ bản quyền).
- **Public API**: `validateDatasetRegistry(registry)`, `validateProductionModelLineage(modelEntry, datasetRegistry)`
- **Tests**: `packages/shared/src/__tests__/dataset-registry.test.ts` (10 unit tests).
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không có.

### Dataset Acquisition CLI (Dry-Run Only)
- **File**: `ml/datasets/acquire.py`
- **Responsibility**: CLI quản lý tiếp nhận dữ liệu với cờ `--dry-run`, xác thực tính hợp lệ của dataset từ `datasets/registry.json`, kiểm tra track isolation, chặn tuyệt đối việc tải file thật trong Phase 4A.0 (cờ `--execute` bị vô hiệu hóa), từ chối dataset bị block và dataset nghiên cứu đưa vào luồng sản phẩm.
- **Public API**: `python -m ml.datasets.acquire --dataset <id> --track <research|product> --dry-run`
- **Tests**: `ml/tests/test_contamination_guard.py` (6 unit tests).
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Cờ `--execute` và hàm download file thật được chủ động vô hiệu hóa cho đến khi có phê duyệt tải dữ liệu của người dùng.

### Synthetic Smoke Fixture Generator
- **File**: `ml/tests/fixtures/smoke_generator.py`
- **Responsibility**: Bộ sinh dữ liệu thử nghiệm nội bộ thuần túy bằng code (hình học, gradient, noise) kèm binary mask và manifest chuẩn 14 trường. Phục vụ kiểm thử unit test pipeline, parser, split và dataloader mà không phụ thuộc dữ liệu bên ngoài.
- **Public API**: `generate_smoke_dataset(output_dir, num_samples=8)`
- **Tests**: `ml/tests/test_contamination_guard.py`.
- **Actual status**: `implemented-and-tested`
- **Known gaps**: Không phải dữ liệu thật, không có giá trị đo độ chính xác hay benchmark khoa học.

