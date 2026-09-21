# Work Item Backlog & Dependency Matrix: Forensics Web Lab

## 1. Project Phase Roadmap

```mermaid
graph TD
  P0[Phase 0: Audit & Architecture Specification] --> P1[Phase 1: Production Monorepo Foundation]
  P1 --> P2[Phase 2: Local Browser Forensic Pipeline]
  P1 --> P3[Phase 3: Dataset & Research Pipeline]
  P3 --> P35[Phase 3.5: Persistent Memory, Truth Audit & Model Readiness]
  P35 --> P36[Phase 3.6: Evidence Hardening & Claim Correction]
  P36 --> P4A0[Phase 4A.0: License Correction, Track Isolation & Acquisition Dry-Run]
  P4A0 --> P4A[Phase 4A: Data Preparation & Smoke Fixtures]
  P4A --> P4[Phase 4: Model Training & Evaluation]
  P4 --> P5[Phase 5: ONNX Browser Integration]
  P2 --> P5
  P5 --> P6[Phase 6: Complete Forensic UX & Reporting]
  P6 --> P7[Phase 7: Hardening & Release Verification]
```

---

## 2. Granular Task Breakdown

### Phase 0: Audit & Specification (Hoàn thành - Commit `625e584`)
* [x] **TASK-001**: Audit repository status, git branch, verify `forensics-web-lab` workspace.
* [x] **TASK-002**: Create and switch to development branch `feat/production-ai-image-forensics`.
* [x] **TASK-003**: Formulate comprehensive architecture specification (`docs/ARCHITECTURE.md`).
* [x] **TASK-004**: Formulate research plan, hypotheses, baseline, and evaluation protocols (`docs/RESEARCH_PLAN.md`).
* [x] **TASK-005**: Author dataset specifications and anti-leakage guidelines (`docs/DATASETS.md`).
* [x] **TASK-006**: Document academic dataset licensing and terms (`docs/DATA_LICENSES.md`).
* [x] **TASK-007**: Define training pipeline, loss, and augmentation specs (`docs/TRAINING.md`).
* [x] **TASK-008**: Define evaluation suites, calibration metrics, and benchmarks (`docs/EVALUATION.md`).
* [x] **TASK-009**: Document zero-egress privacy architecture (`docs/PRIVACY.md`).
* [x] **TASK-010**: Document threat modeling and defensive security rules (`docs/SECURITY.md`, `docs/THREAT_MODEL.md`).
* [x] **TASK-011**: Document physical limitations, scope boundaries, and disclaimers (`docs/LIMITATIONS.md`).
* [x] **TASK-012**: Author deployment configurations and COOP/COEP guides (`docs/DEPLOYMENT.md`).
* [x] **TASK-013**: Detail reproducibility instructions (`docs/REPRODUCIBILITY.md`).
* [x] **TASK-014**: Document repository licensing decision considerations (`docs/LICENSING.md`).
* [x] **TASK-015**: Create Model Card specification (`models/MODEL_CARD.md`).
* [x] **TASK-016**: Create Model Registry config and schema (`models/registry.json`, `docs/schemas/model-registry.v1.schema.json`).
* [x] **TASK-017**: Author Architectural Decision Records (ADRs 0001 through 0005).
* [x] **TASK-018**: Define Analysis Output JSON Schema (`docs/schemas/analysis-output.v1.schema.json`).
* [x] **TASK-019**: Commit Phase 0 with commit message: `docs: define product and research architecture`.

---

### Phase 1: Production Monorepo Foundation (Hoàn thành - Commit `f2a7fb4`)
* [x] **TASK-101**: Configure monorepo root: `pnpm-workspace.yaml`, `.npmrc`, root `package.json`, `.gitignore`.
* [x] **TASK-102**: Configure `packages/shared` with TypeScript types, contracts, constants, and schema exports.
* [x] **TASK-103**: Configure `packages/provenance` package skeleton for EXIF/XMP parsing and C2PA adapter.
* [x] **TASK-104**: Configure `packages/forensics` package skeleton for 2D DSP signal processing.
* [x] **TASK-105**: Configure `packages/inference` package skeleton for ONNX Runtime Web session and worker abstractions.
* [x] **TASK-106**: Configure `packages/report` package skeleton for report generation.
* [x] **TASK-107**: Set up `apps/web` with Vite, React, TypeScript strict mode, CSS design tokens, and layout.
* [x] **TASK-108**: Set up `ml/` Python workspace: `pyproject.toml`, `requirements.txt`, directory layout (`configs/`, `datasets/`, `training/`, `evaluation/`, `export/`, `tests/`).
* [x] **TASK-109**: Configure ESLint, Prettier, TypeScript build configs across all packages.
* [x] **TASK-110**: Configure GitHub Actions CI workflow (`.github/workflows/ci.yml`) for lint, typecheck, test, build.
* [x] **TASK-111**: Configure reproducible Docker development / build container (`Dockerfile`, `docker-compose.yml`).
* [x] **TASK-112**: Commit Phase 1 with message: `build: establish production monorepo foundation`.

---

### Phase 2: Local Browser Forensic Pipeline (Hoàn thành - Commit `a882eee`)
* [x] **TASK-201**: Implement defensive file validation (magic byte audit, dimension cap, decompression bomb guard).
* [x] **TASK-202**: Implement pure-TS EXIF, XMP, and IPTC metadata parser with software signature matching.
* [x] **TASK-203**: Implement C2PA Content Credentials adapter interface with graceful `unsupported` handling.
* [x] **TASK-204**: Implement canvas preprocessing (orientation normalization, tensor extraction, letterbox).
* [x] **TASK-205**: Implement 2D-FFT / 2D-DCT azimuthal frequency energy analyzer in TypeScript.
* [x] **TASK-206**: Implement spatial noise residual variance analyzer.
* [x] **TASK-207**: Implement JPEG 8x8 block grid artifact analyzer and Error Level Analysis (ELA) generator.
* [x] **TASK-208**: Implement dedicated Web Worker (`forensics.worker.ts`) with progress events and cancellation protocol.
* [x] **TASK-209**: Write comprehensive unit tests for validation, metadata, and forensic signal extractors.
* [x] **TASK-210**: Commit Phase 2 with message: `feat: add local browser forensic pipeline`.

---

### Phase 3: Dataset & Research Pipeline (Hoàn thành - Commit `6555b02`)
* [x] **TASK-301**: Implement dataset manifest generator (`ml/datasets/manifest.py`) adhering to schema.
* [x] **TASK-302**: Implement adapters for GenImage, SAGI-D, RealHD, RAID, and traditional edit datasets.
* [x] **TASK-303**: Implement perceptual hashing (pHash) and SHA-256 duplicate detection script.
* [x] **TASK-304**: Implement strict `source_id` group-based splitting and unseen-generator holdout splitters.
* [x] **TASK-305**: Implement multi-stage realistic degradation augmentation pipeline (Albumentations/torchvision).
* [x] **TASK-306**: Implement model training engine with class-weighted focal loss and checkpointing.
* [x] **TASK-307**: Implement evaluation harness computing Macro F1, ECE, Brier score, and localization mIoU.
* [x] **TASK-308**: Write unit tests for ML dataset adapters, splitting, and augmentation transforms.
* [x] **TASK-309**: Commit Phase 3 with message: `feat: add reproducible dataset and experiment pipeline`.

---

### Phase 3.5: Persistent Memory, Implementation Truth Audit & Model Readiness (Hoàn thành - Commit `cd59136`)
* [x] **TASK-351**: Create root `AGENTS.md` establishing persistent memory rules and scientific truthfulness.
* [x] **TASK-352**: Create `docs/PROJECT_STATE.md` with active project status, scope, and blockers.
* [x] **TASK-353**: Create `docs/CODE_MAP.md` mapping all modules, files, and actual verified statuses.
* [x] **TASK-354**: Conduct and document Implementation Truth Audit (`docs/IMPLEMENTATION_TRUTH_AUDIT.md`).
* [x] **TASK-355**: Audit pretrained candidate checkpoints in `docs/PRETRAINED_MODEL_CANDIDATES.md`.
* [x] **TASK-356**: Define Model Acquisition Gate in `docs/MODEL_ACQUISITION_GATE.md`.
* [x] **TASK-357**: Create `docs/SESSION_HANDOFF.md` and update `docs/BACKLOG.md`.
* [x] **TASK-358**: Refactor `FusionCalibrator` and worker to enforce honest no-model state (`uncertain`, `confidence: null`, `probabilities: null`, `modelAvailable: false`).
* [x] **TASK-359**: Update UI (`ResultVerdictCard`, `HeatmapViewer`, `ForensicsInspector`) to display "Model not installed" and classify heuristics as exploratory.
* [x] **TASK-360**: Implement automated registry integrity validation in `@forensics/shared` and add unit tests.
* [x] **TASK-361**: Verify all test suites (`pnpm test`, `pnpm build`, `pytest`) and create Phase 3.5 commits.

---

### Phase 3.6: Evidence Hardening & Claim Correction (Hoàn thành)
* [x] **TASK-362**: Conduct independent repository state audit and enforce clean Git baseline.
* [x] **TASK-363**: Audit and eliminate all absolute machine paths (`file:///`, `D:\`, `C:\`, `/Users/`, `/home/`) across documentation and code.
* [x] **TASK-364**: Correct overstated readiness claims: rename `CAND-C2-INHOUSE-MNV3` to `ARCH-C2-INHOUSE-MNV3`, mark size as `estimated`, parity as `pipeline-only`, weight license as `not-applicable`, and detection metrics as `unverified` / `not measured`.
* [x] **TASK-365**: Formalize 8 standardized evidence classifications across all project documentation.
* [x] **TASK-366**: Establish machine-readable Evidence Register (`docs/EVIDENCE_REGISTER.md`, `research/evidence/phase-3.6/evidence-manifest.json`) and JSON Schema (`docs/schemas/evidence-manifest.v1.schema.json`).
* [x] **TASK-367**: Implement automated manifest validator and expand registry validator to enforce 7 mandatory readiness gates with test coverage.
* [x] **TASK-368**: Expand `docs/MODEL_ACQUISITION_GATE.md` with 17 mandatory criteria before admitting any model to `ready` status.
* [x] **TASK-369**: Author `docs/PHASE_4A_DATA_FEASIBILITY.md` with dataset decision matrix and user authorization gate.
* [x] **TASK-370**: Execute genuine verification runs and capture execution summaries (`environment.json`, `test-summary.json`, `build-summary.json`).

---

### Phase 4A.0: Data License Correction, Track Isolation & Acquisition Dry-Run (Hoàn thành)
* [x] **TASK-4A01**: Re-audit dataset licenses for GenImage, RealHD, SAGI-D, and RAID against official author sources.
* [x] **TASK-4A02**: Correct GenImage terms (`CC BY-NC-SA 4.0 with additional dataset terms`, `research-only`, rename smoke samples to `Custom smoke subset sampled from GenImage`).
* [x] **TASK-4A03**: Set RealHD status to `unavailable-or-pending`, `license: unverified`, `decision: blocked`.
* [x] **TASK-4A04**: Establish formal Dual-Track Data & Model Isolation Policy in `docs/adr/0006-research-product-data-isolation.md`.
* [x] **TASK-4A05**: Establish directory conventions (`data/research/`, `data/product/`, `artifacts/research/`, `artifacts/product/`, `models/research/`, `models/product/`) and update `.gitignore`.
* [x] **TASK-4A06**: Create machine-checkable Dataset Registry (`datasets/registry.json`, `docs/schemas/dataset-registry.v1.schema.json`, `datasets/README.md`) with TypeScript and Python validators.
* [x] **TASK-4A07**: Create provenance tracking schema (`docs/schemas/dataset-manifest.v1.schema.json`).
* [x] **TASK-4A08**: Implement dataset acquisition CLI with dry-run mode (`ml/datasets/acquire.py`), rejecting blocked datasets and cross-track violations.
* [x] **TASK-4A09**: Implement synthetic geometric/noise smoke fixture generator (`ml/tests/fixtures/smoke_generator.py`) without external network downloads or copyright restrictions.
* [x] **TASK-4A10**: Implement automated contamination guards and unit tests in `@forensics/shared` and `ml/tests/test_contamination_guard.py`.
* [x] **TASK-4A11**: Correct zero-egress statement to `architecture-supported`, `runtime-network-verification: unverified` in `README.md`, `docs/PRIVACY.md`, `docs/PROJECT_STATE.md`.
* [x] **TASK-4A12**: Record 28.3 MB build WASM asset as engine runtime binary, not model weight.
* [ ] **TASK-WEB-001**: Measure compressed transfer size (gzip/brotli), lazy loading, cache behavior, WASM initialization time, inference latency and peak memory across browser targets.
* [ ] **TASK-E2E-001**: Implement browser E2E automated network logging to empirically confirm 0 outbound requests during image analysis.

---

### Phase 4A: Data Preparation & Smoke Fixtures (Chờ người dùng phê duyệt)
* [ ] **TASK-4A1**: User decision on Phase 4A dataset pathway (Option 1: Synthetic Smoke Fixture vs Option 2: Custom smoke subset sampled from GenImage).
* [ ] **TASK-4A2**: Prepare authorized dataset/fixture splits with group-based anti-leakage isolation.

---

### Phase 4: Model Training & Evaluation (Chờ phê duyệt)
* [ ] **TASK-401**: Check data/compute authorization (Stop and confirm if large data download required).
* [ ] **TASK-402**: Train and log baseline model with reproducible configs and deterministic seeds.
* [ ] **TASK-403**: Run full evaluation suite across in-domain, unseen generator, and degradation matrices.
* [ ] **TASK-404**: Fit temperature scaling calibration on held-out calibration split.
* [ ] **TASK-405**: Record genuine experimental metrics in `models/MODEL_CARD.md` and research logs (no fake numbers).
* [ ] **TASK-406**: Commit Phase 4 with message: `research: train and evaluate lightweight detector`.

---

### Phase 5: ONNX Browser Integration (Chờ Phase 4)
* [ ] **TASK-501**: Implement PyTorch -> ONNX export script with dynamic batching and Opset 17.
* [ ] **TASK-502**: Implement numerical contract tests verifying PyTorch vs ONNX Runtime CPU outputs.
* [ ] **TASK-503**: Implement INT8 post-training quantization and verify quantized accuracy and size.
* [ ] **TASK-504**: Compute SHA-256 checksum and register exported model in `models/registry.json`.
* [ ] **TASK-505**: Implement ONNX Runtime Web session loader in `packages/inference` with WASM/WebGPU orchestration.
* [ ] **TASK-506**: Implement overlapping patch batching and Gaussian-weighted heatmap accumulation in Web Worker.
* [ ] **TASK-507**: Implement browser benchmark runner measuring WASM vs WebGPU latencies.
* [ ] **TASK-508**: Commit Phase 5 with message: `feat: integrate calibrated ONNX inference`.

---

### Phase 6: Complete Forensic UX & Reporting
* [ ] **TASK-601**: Build responsive, professional forensic landing interface with drag-and-drop ingestion.
* [ ] **TASK-602**: Build real-time multi-stage analysis progress indicator with cancellation support.
* [ ] **TASK-603**: Build interactive results dashboard: 4-state verdict, calibrated confidence, probabilities.
* [ ] **TASK-604**: Build high-performance canvas heatmap viewer with opacity slider, colormap picker, and region boxes.
* [ ] **TASK-605**: Build forensic signal inspector (frequency spectrum, noise variance, JPEG block grid, ELA preview).
* [ ] **TASK-606**: Build metadata and C2PA provenance explorer.
* [ ] **TASK-607**: Build structured JSON report exporter conforming to JSON Schema v1.0.0.
* [ ] **TASK-608**: Build printable HTML / Save-as-PDF report view with dedicated `@media print` styles.
* [ ] **TASK-609**: Build privacy controls: "Clear Session" memory purge and cache manager.
* [ ] **TASK-610**: Commit Phase 6 with message: `feat: complete forensic analysis experience`.

---

### Phase 7: Hardening & Release Verification
* [ ] **TASK-701**: Conduct end-to-end security review (XSS escaping audit, decompression bomb testing, malformed file stress).
* [ ] **TASK-702**: Conduct cross-browser tests and fallback validation (WebGPU -> WASM fallback).
* [ ] **TASK-703**: Verify zero memory leaks across consecutive analysis cycles.
* [ ] **TASK-704**: Run Playwright end-to-end smoke test suite on production build.
* [ ] **TASK-705**: Verify documentation consistency, license notices, and reproducibility instructions.
* [ ] **TASK-706**: Commit Phase 7 with message: `chore: harden system for production release`.
