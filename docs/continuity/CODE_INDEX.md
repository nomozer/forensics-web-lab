# Code Index: Forensics Web Lab

> **Thư mục**: `docs/continuity/CODE_INDEX.md`  
> **Mục đích**: Bản đồ kiến trúc, mã nguồn thực tế và định vị trách nhiệm module cho các phiên làm việc của AI.  
> **Quy ước**: Toàn bộ đường dẫn trong tài liệu đều là đường dẫn tương đối từ gốc repository.

---

## 1. Mục tiêu Hệ thống & Nguyên tắc Kiến trúc

* **Tên hiển thị giao diện**: `Forensics Web Lab`
* **Tên đề tài nghiên cứu**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Nguyên tắc cốt lõi**:
  1. **Zero-Egress Client-Side**: Toàn bộ quá trình giải mã ảnh, phân tích DSP và suy luận ONNX diễn ra 100% trong trình duyệt (Web Worker), không gửi ảnh lên máy chủ.
  2. **WASM-First & WebGPU Optional**: Mặc định chạy CPU/WASM đa luồng SIMD; tự động tăng tốc WebGPU khi phần cứng hỗ trợ.
  3. **Dual-Track Data Isolation (ADR-0006)**: Tách biệt tuyệt đối Research Track (`data/research/`, `models/research/`) và Product Track (`data/product/`, `models/product/`).

---

## 2. Sơ đồ Luồng Xử lý Ảnh (Processing Flow Diagram)

```mermaid
graph TD
  User([User Image File]) --> Dropzone[apps/web: ImageDropzone.tsx]
  Dropzone --> Magic[Defensive Validation: Magic Bytes & Bomb Guard]
  Magic --> Controller[WorkerController.ts]
  Controller --> Worker[forensics.worker.ts]
  
  subgraph Worker_Thread [Web Worker Execution Pipeline]
    Worker --> Canvas[Canvas RGBA Decode & Letterbox]
    Canvas --> Meta[packages/provenance: EXIF / XMP / C2PA]
    Canvas --> DSP[packages/forensics: 2D FFT / DCT / Noise / ELA]
    Canvas --> Inf[packages/inference: ONNX Runtime Session]
    
    Inf --> ModelCheck{Model Available?}
    ModelCheck -->|No| NoModel[Honest No-Model: verdict=uncertain, confidence=null]
    ModelCheck -->|Yes| Inference[Patch Grid Inference & Global Classification]
    
    Meta --> Fusion[FusionCalibrator: Multimodal Evidence Fusion]
    DSP --> Fusion
    Inference --> Fusion
    NoModel --> Fusion
  end
  
  Fusion --> VerdictResult[AnalysisResult Object]
  VerdictResult --> UI[apps/web: Dashboard / ResultVerdictCard / HeatmapViewer]
  VerdictResult --> Report[packages/report: JSON Schema v1 & Printable PDF]
```

---

## 3. Cấu trúc Monorepo & Vai trò Package

| Workspace / Package | Thư mục | Vai trò và Trách nhiệm Kỹ thuật | Entry Point |
| :--- | :--- | :--- | :--- |
| `@forensics/shared` | `packages/shared/` | Định nghĩa contracts, TypeScript types, schemas, registry validators, evidence validators | `packages/shared/src/index.ts` |
| `@forensics/provenance` | `packages/provenance/` | Trích xuất siêu dữ liệu EXIF, XMP, IPTC, kiểm tra chữ ký AI software, C2PA adapter | `packages/provenance/src/index.ts` |
| `@forensics/forensics` | `packages/forensics/` | Xử lý tín hiệu số 2D (FFT spectrum, DCT frequency energy, noise variance, JPEG block ELA) | `packages/forensics/src/index.ts` |
| `@forensics/inference` | `packages/inference/` | ONNX Runtime Web session loader, sliding patch inference, fusion calibrator, worker driver | `packages/inference/src/index.ts` |
| `@forensics/report` | `packages/report/` | Tạo báo cáo JSON chuẩn schema và render HTML printable view cho xuất PDF | `packages/report/src/index.ts` |
| `apps/web` | `apps/web/` | Giao diện người dùng React 18, Vite, CSS design tokens, Canvas heatmap viewer | `apps/web/src/main.tsx` |
| `ml` | `ml/` | Workspace Python cho dataset adapters, group splitting, PyTorch training, ONNX export | `ml/training/train.py` |

---

## 4. Định vị File Chịu trách nhiệm Cốt lõi (Core Responsibility Map)

| Nhiệm vụ chức năng | File chịu trách nhiệm chính | Trạng thái kỹ thuật |
| :--- | :--- | :--- |
| **Validation & Magic Bytes** | `packages/shared/src/validation.ts` | `implemented-and-tested` |
| **Metadata & Provenance** | `packages/provenance/src/exif-parser.ts`, `c2pa-adapter.ts` | `implemented-and-tested` |
| **DSP Frequency (FFT/DCT)** | `packages/forensics/src/fft2d.ts`, `dct2d.ts` | `implemented-and-tested` |
| **DSP Noise & JPEG Artifacts**| `packages/forensics/src/noise-residual.ts`, `jpeg-artifacts.ts` | `implemented-and-tested` |
| **Model Runtime Inference** | `packages/inference/src/session-loader.ts`, `patch-infer.ts` | `implemented-and-tested` |
| **Evidence Fusion & Calibration**| `packages/inference/src/fusion-calibrator.ts` | `implemented-and-tested` |
| **Web Worker Coordinator** | `apps/web/src/workers/forensics.worker.ts`, `WorkerController.ts` | `implemented-and-tested` |
| **Heatmap Visualization** | `apps/web/src/components/HeatmapViewer.tsx` | `implemented-and-tested` |
| **Report Generation** | `packages/report/src/report-generator.ts`, `html-print.ts` | `implemented-and-tested` |
| **Dataset Manifest & Adapters** | `ml/datasets/manifest.py`, `ml/datasets/adapters/` | `implemented-and-tested` |
| **Deduplication & Group Split** | `ml/datasets/dedup.py`, `ml/datasets/split.py` | `implemented-and-tested` |
| **Model Architecture (PyTorch)**| `ml/training/mobilenetv3_forensics.py` | `implemented-and-tested` |
| **Loss & Training Engine** | `ml/training/losses.py`, `ml/training/train.py` | `implemented-and-tested` |
| **Temperature Scaling** | `ml/evaluation/calibration.py` | `implemented-and-tested` |
| **Evaluation Metrics Harness** | `ml/evaluation/metrics.py`, `evaluate.py` | `implemented-and-tested` |
| **ONNX Export & Parity Test** | `ml/export/export_onnx.py`, `validate_contract.py` | `implemented-and-tested` |
| **Acquisition Guard CLI** | `ml/datasets/acquire.py` | `implemented-and-tested` |
| **Acquisition Safety & Execution Gate** | `ml/datasets/acquire.py` (free-disk, .part, resume, checksum, safe-extract, staging, receipt) | `implemented-and-tested` |
| **Component-Scoped Acquisition** | `ml/datasets/acquire.py` (`--component`, `--max-download-bytes`, `HostnameRestrictedRedirectHandler`) | `implemented-and-tested` |
| **Mask Inventory & Audit Processor** | `ml/datasets/audit_masks.py` (PIL verification, pixel profile, dimension, source_id mapping) | `implemented-and-tested` |
| **Mask Data Manifest (Local)** | `data/research/tgif/manifests/masks-manifest.jsonl` (31,238 records, outside Git) | `generated-and-verified` |
| **Acquisition Receipt Location** | `data/research/tgif/acquisition-receipt.json` | `generated-and-verified` |
| **Acquisition Plan & Schema** | `docs/schemas/acquisition-plan.v1.schema.json`, `datasets/acquisition-plans/pilot-a-tgif.v1.json` | `implemented-and-tested` |
| **Paired Bootstrap Guard** | `ml/evaluation/bootstrap_guard.py` (stratified paired bootstrap 95% CI) | `implemented-and-tested` |
| **TGIF Cardinality Audit** | `research/evidence/phase-4a.4/tgif-cardinality-audit.json` | `audited-and-verified` |
| **Pilot Protocols & Configs** | `ml/configs/validator.py`, `pilot_tgif_edit.yaml`, `pilot_genimage_generated.yaml` | `implemented-and-tested` |
| **Source ID Auditor & Collision Engine** | `ml/datasets/audit_source_ids.py` (COCO source vs category-task disambiguation) | `implemented-and-tested` |
| **Remote Metadata Inventory** | `research/evidence/phase-4b.1/tgif-orig-remote-inventory.json`, `tgif-sd2-sp-remote-inventory.json` | `audited-and-verified` |
| **Small-Data Protocol & Options** | `docs/SMALL_DATA_PROTOCOL.md`, `research/evidence/phase-4b.1/small-data-options.json` | `protocol-established` |
| **Learning Curve Config & Validator** | `ml/configs/pilot_a_learning_curve.yaml`, `ml/configs/validator.py` (sample size, overflow guard) | `implemented-and-tested` |
| **Option P Acquisition Plan** | `datasets/acquisition-plans/pilot-a-tgif-option-p.v1.json` | `implemented-and-tested` |
| **Option P Downloader & Safe Extract** | `ml/datasets/acquire.py` (`safe_extract_tar`, HTTP Range resume, Windows file lock retry/fallback) | `implemented-and-tested` |
| **Option P Verification & Split Freeze** | `ml/datasets/verify_option_p.py` (Pillow decode audit, pairability matching, split freeze) | `implemented-and-tested` |
| **Option P Manifest (Local)** | `data/research/tgif/manifests/manifest_pilot_a_option_p.csv` (outside Git) | `generated-and-verified` |
| **Option P Split Lock Artifact** | `research/evidence/phase-4b.2/split-lock.json` (343 locked test sources, SHA-256 seal) | `frozen-and-sealed` |
| **Option P Test Suite** | `ml/tests/test_option_p_protocol.py` (11 targeted tests: Tar Slip, resume, drift, split lock, coverage) | `implemented-and-tested` |
| **Pair-Aware Sampler & Aggregator** | `ml/datasets/pair_aware_loader.py` (Strategy B: authentic native paired with edited native, 100% matched pixel geometry, balanced bbox/segm cycling, SHA-256 stable offset, source-level aggregation) | `implemented-and-tested` |
| **Locked-Test Role Access Guard** | `ml/evaluation/locked_test_guard.py` (role-based permissions, ExperimentLockBinding verification) | `implemented-and-tested` |
| **Training Manifest Schema** | `docs/schemas/training-manifest.v1.schema.json` (source-isolated manifest validation) | `schema-enforced` |
| **Experiment Lock Schema** | `docs/schemas/experiment-lock.v1.schema.json` (cryptographic evaluation binding schema) | `schema-enforced` |
| **Phase 4C Preregistered Config** | `ml/configs/pilot_a_binary_preregistered.yaml` (hash `727fc316...`, superseded `54140d42...`, `839531a7...`), `docs/PHASE_4C_PREREGISTRATION.md` | `preregistered-resealed` |
| **Phase 4C.1 Learning Curve Config** | `ml/configs/phase_4c1_learning_curve.yaml` (derived from `pilot_a_binary_preregistered.yaml`), `docs/PHASE_4C1_EXECUTION_PLAN.md` | `planned-awaiting-approval` |
| **Phase 4C.1 Colab Bundle Scripts** | `ml/datasets/export_phase_4c1_bundle.py`, `ml/datasets/validate_phase_4c1_bundle.py`, `ml/requirements-colab.txt`, `notebooks/phase_4c1_learning_curve_colab.ipynb`, `docs/PHASE_4C1_COLAB_GUIDE.md` | `created-awaiting-execution` |
| **Pixel-Reality Gate Auditor** | `scripts/audit_pixel_reality.py` (measure 6,156 images, confirm Case B native canvas, generate `pixel-geometry-audit.json`, `real-variant-map.json`) | `implemented-and-tested` |
| **Sampler Runtime Auditor** | `scripts/audit_sampler_runtime.py` (verify stable offset across PYTHONHASHSEED, generate `sampler-runtime-audit.json`) | `implemented-and-tested` |
| **Pretrained Weight Downloader** | `scripts/download_mobilenet_weights.py` (restricted domain, <=12 MiB ceiling, atomic rename, `pretrained-weight-receipt.json`) | `implemented-and-tested` |
| **Stage 0 Baselines Runner** | `scripts/run_stage0_baselines.py` (Dummy, Metadata-only, DSP-only on N=50,100,250 evaluated on inner_validation) | `implemented-and-tested` |
| **Smoke Training Runner** | `scripts/run_smoke_training.py` (single-seed N=50 seed 42, frozen backbone, resource profile, metrics, checkpoint) | `implemented-and-tested` |
| **Phase 4C.0 Gate Tests** | `ml/tests/test_phase4c0_gates.py` (9 targeted tests: Case B audit, real variant map, resolution match, stable sampler, leakage guard, frozen backbone, locked-test denial, network ceiling, checkpoint isolation) | `implemented-and-tested` |
| **Phase 4C.0a Reconciliation Script** | `scripts/reconcile_phase4c0.py` (checkpoint re-evaluation, Stage 0 prediction hashing, Option A vs B reconciliation, resource accounting) | `implemented-and-tested` |
| **Phase 4C.0a Reconciliation Tests** | `ml/tests/test_phase4c0a_reconciliation.py` (8 targeted tests: architecture binding, BCEWithLogitsLoss, metric tolerance, paired aggregation, bootstrap grouping, artifact hashes, heap vs RSS, locked-test zero) | `implemented-and-tested` |
| **Phase 4B.3/4B.4 Unit Tests** | `ml/tests/test_pair_aware_loader.py` (10 tests), `ml/tests/test_locked_test_guard.py` (7 tests), `ml/tests/test_evidence_reproducer.py` (7 tests) | `implemented-and-tested` |
| **Evidence Reproducer Script** | `scripts/reproduce-phase-4b3-evidence.py` (portable live measurement: Git, runtime, archives SHA-256/bytes, manifest invariants, clean-link audit; `--verify` and `--write-evidence` modes) | `implemented-and-tested` |
| **Continuity Checker CLI** | `scripts/continuity-check.mjs` | `implemented-and-tested` |
| **Continuity Checker Tests** | `scripts/__tests__/continuity-check.test.mjs` | `implemented-and-tested` |
| **Continuity Contract** | `AGENTS.md` (Mục 4: Continuity Contract) | `contract-enforced` |
| **CI Continuity Gate** | `.github/workflows/ci.yml` (Model-Agnostic Continuity Check) | `ci-enforced` |

---

## 5. Input / Output Contracts & Trạng thái Đầu ra

### 5.1. Bốn Trạng thái Kết luận Hệ thống (`AnalysisVerdict`)
1. `no_ai_evidence`: Không phát hiện dấu vết AI; siêu dữ liệu và tín hiệu số nhất quán với ảnh thông thường (tuyệt đối không khẳng định "ảnh thật 100%").
2. `fully_generated`: Phát hiện dấu vết nhất quán với ảnh do mô hình AI tạo sinh toàn phần (GAN/Diffusion).
3. `ai_edited`: Phát hiện vùng can thiệp cục bộ bằng AI (inpainting, thay thế đối tượng).
4. `uncertain`: Độ tin cậy dưới ngưỡng, hoặc tín hiệu mâu thuẫn, hoặc **chưa cài đặt model** (`confidence: null`).

### 5.2. AnalysisResult Schema Contract (`docs/schemas/analysis-output.v1.schema.json`)
```typescript
interface AnalysisResult {
  schemaVersion: "1.0.0";
  timestampUtc: string;
  image: { width: number; height: number; format: string; sha256: string };
  verdict: "no_ai_evidence" | "fully_generated" | "ai_edited" | "uncertain";
  confidence: number | null;                // null khi model chưa cài đặt
  probabilities: { authentic: number; fully_generated: number; ai_edited: number } | null;
  model: { available: boolean; modelId: string | null; version: string | null };
  forensicSignals: ForensicSignal[];        // DSP metrics: FFT, DCT, Noise, ELA
  provenance: ProvenanceResult;             // EXIF/XMP/C2PA tags
  localization: LocalizationResult;         // Heatmap matrix & bounding boxes
  evidence: { supporting: string[]; refuting: string[]; limitations: string[] };
}
```

---

## 6. Sổ Quản lý Metadata (Registries)

* **Model Registry**: `models/registry.json`  
  - Schema: `docs/schemas/model-registry.v1.schema.json`.  
  - Trạng thái hiện tại: `status: "not-trained"`, `path: ""`, `sizeBytes: 0`.  
  - Kiểm tra toàn vẹn: 7 tiêu chí bắt buộc qua `validateModelRegistry` trong `@forensics/shared`.
* **Dataset Registry**: `datasets/registry.json`  
  - Schema: `docs/schemas/dataset-registry.v1.schema.json`.  
  - 7 datasets đã đăng ký (`genimage`, `tgif`, `tgif2`, `synthetic-smoke`, `sagi-d`, `raid`, `realhd`).  
  - Phân luồng: `fixture-only`, `research-only`, `product-eligible`, `blocked`.

---

## 7. Liên hệ giữa Module Phần mềm và Câu hỏi Nghiên cứu (RQs)

| Câu hỏi Nghiên cứu | Module / Package chịu trách nhiệm | File kiểm chứng thực nghiệm |
| :--- | :--- | :--- |
| **RQ1 (3-Class Generalization)** | `ml/training/`, `ml/datasets/` | `ml/training/train.py`, `ml/evaluation/evaluate.py` |
| **RQ2 (Multimodal Evidence Fusion)** | `packages/inference/`, `packages/forensics/` | `packages/inference/src/fusion-calibrator.ts` |
| **RQ3 (ONNX INT8 Quantization)** | `ml/export/`, `packages/shared/` | `ml/export/export_onnx.py`, `quantize.py` |
| **RQ4 (Browser WASM Runtime)** | `packages/inference/`, `apps/web/` | `packages/inference/src/session-loader.ts`, `forensics.worker.ts` |
| **Auxiliary RQ5 (Localization)** | `apps/web/src/components/`, `packages/inference/` | `apps/web/src/components/HeatmapViewer.tsx`, `patch-infer.ts` |

---

## 8. Sổ tay Lệnh Thao tác Kỹ thuật (Key Commands)

```bash
# 1. Kiểm thử TypeScript toàn bộ 6 packages
pnpm test

# 2. Build production web bundle (Vite + TS strict)
pnpm build

# 3. Kiểm thử Python ML pipeline
ml/.venv/Scripts/python -m pytest ml/tests -v

# 4. Xác thực tính toàn vẹn của Dataset Registry
ml/.venv/Scripts/python -m ml.datasets.acquire --validate-registry

# 5. Kiểm tra metadata dataset GenImage (0 byte tải content)
ml/.venv/Scripts/python -m ml.datasets.acquire --dataset genimage --track research --metadata-only

# 6. Sinh fixture hình học nội bộ (0 network, 0 external bytes)
ml/.venv/Scripts/python -m ml.datasets.acquire --dataset synthetic-smoke --track fixture-only --execute

# 7. Kiểm tra tính toàn vẹn và hợp đồng continuity (cross-platform)
pnpm continuity:check
pnpm continuity:check -- --staged
pnpm continuity:check -- --base <PR_BASE_SHA> --head <PR_HEAD_SHA>
```
