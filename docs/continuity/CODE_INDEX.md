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
| **Phase 4C.1 Colab Bundle Scripts** | `ml/datasets/export_phase_4c1_bundle.py` (reusable N250 bundle creation with frozen cohorts N50/N100/N250, package_tar_archive, audits), `ml/datasets/validate_phase_4c1_bundle.py` (fail-closed validator with 14 criteria: locked-test exclusion, dev/val isolation, nested cohorts, class balance, tarball verification), `ml/requirements-colab.txt`, `notebooks/phase_4c1_learning_curve_colab.ipynb` (canonical single Colab launcher with 5-cell structure, default EXECUTE=True, capability-based GPU detection, hardened error reporting, and persistent Drive input/output architecture for resumable 15-run execution; historical notebook preserved via git SHA `f26576cb...`), `docs/PHASE_4C1_COLAB_GUIDE.md` | `implemented-and-tested` |
| **Phase 4C.1 Training Runner** | `ml/training/run_phase_4c1.py` (canonical CLI for N=50, 100, 250 across 5 seeds, frozen cohorts, baselines, receipts, 9 required artifacts, checkpoints) | `implemented-and-tested` |
| **Phase 4C.1 Test Suite** | `ml/tests/test_phase_4c1_notebook.py` (17 canonical notebook tests: 5-cell structure, EXECUTE=True default, read-only fail-closed preflight, persistent Drive output staging, no mkdir on inputs, hardened error reporting, 15-run audit), `ml/tests/test_phase_4c1_operator.py` (48 tests: 9 static invariants, 10 GPU capability & policy, 9 scientific binding vs runtime observation, 3 dependency preflight policy, 17 fault injections & resume assertions), `test_phase_4c1_bundle.py` (unit & fault injection for reusable bundle & validator), `test_phase_4c1_runner.py` (frozen cohort invariance & multi-size loader tests) | `implemented-and-tested` |
| **Phase 4C.1B.4 Verification Scripts** | `run_smoke_local.py` (standalone local GPU smoke runner), `ml/evaluation/metrics.py` (added `compute_ece`), `ml/training/run_phase_4c1.py` (fixed config access) | `implemented-and-tested` |
| **Phase 4C.1B.4 Verification Evidence** | `research/evidence/phase-4c.1b.4/` (artifact-inventory, checksum-verification, partition-audit, metric-reproduction, baseline-reproduction, stage2-gate-audit, evidence-manifest, PHASE_REPORT) | `implemented-and-tested` |
| **Phase 4C.1B.5 Verification Evidence** | `research/evidence/phase-4c.1b.5/` (artifact-inventory, checksum-verification, partition-audit, metric-reproduction, baseline-reproduction, confidence-intervals, stage2-gate-audit, test-summary, evidence-manifest, PHASE_REPORT) | `implemented-and-tested` |
| **Phase 4C.1B.6 Verification Evidence** | `research/evidence/phase-4c.1b.6/` (environment, leakage-finding, validation-partition-audit, corrected-metric-reproduction, corrected-baseline-reproduction, confidence-intervals, stage2-gate-audit, test-summary, evidence-manifest, PHASE_REPORT) | `implemented-and-tested` |
| **Phase 4C.1B.6R Verification Evidence** | `research/evidence/phase-4c.1b.6r/` (environment, leakage-finding, validation-partition-audit, corrected-metric-reproduction, corrected-baseline-reproduction, confidence-intervals, stage2-gate-audit, test-summary, evidence-manifest, PHASE_REPORT) | `implemented-and-tested` |
| **Phase 4C.1B.6R.1 Verification Evidence** | `research/evidence/phase-4c.1b.6r.1/` (environment, source-repair-audit, fault-injection, predictions-dummy, dummy-baseline-metrics, dummy-baseline-receipt, dummy-baseline-contract, confidence-intervals, metadata-baseline-status, stage2-gate-status, regression-test-summary, test-summary, pytest-collection, pytest-full-output, cleanup-audit, PHASE_REPORT, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1B.6R.2 Verification Evidence** | `research/evidence/phase-4c.1b.6r.2/` (environment, stage1-prediction-receipt, predictions-stage1-validation-only, dummy-prediction-verification, predictions-dummy, dummy-baseline-metrics, paired-bootstrap-distribution-summary, confidence-intervals, fault-injection-results, regression-test-summary, metadata-baseline-status, stage2-gate-status, pytest-collection, pytest-full-output, test-summary, PHASE_REPORT, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1B.6R.3 Verification Evidence** | `research/evidence/phase-4c.1b.6r.3/` (environment, historical-root-cause-audit, fault-injection-results, prediction-key-audit, paired-bootstrap-verification, confidence-intervals, metadata-baseline-status, stage2-gate-status, regression-test-summary, pytest-collection, pytest-full-output, test-summary, notebook-hash-audit, PHASE_REPORT, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1B.6R.3.1 Verification Evidence** | `research/evidence/phase-4c.1b.6r.3.1/` (timestamp-audit, prior-manifest-correction, notebook-hash-audit, historical-root-cause-wording-audit, dependency-audit, clean-environment-verification, multiseed-execution-contract, test-summary, PHASE_REPORT, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1B.6R.3.2 Verification Evidence** | `research/evidence/phase-4c.1b.6r.3.2/` (environment, json-validation, dependency-declaration-audit, clean-venv-creation, clean-install-log, clean-pip-check, clean-import-check, clean-test-summary, colab-requirements-audit, prior-evidence-corrections, PHASE_REPORT, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1B.6R.3.2a Verification Evidence** | `research/evidence/phase-4c.1b.6r.3.2a/` (environment, dependency-amendment, preinstall-package-list, clean-install-log, clean-pip-check, clean-environment-lock, clean-import-check, clean-test-summary, lint-debt, t4-environment-policy, json-validation, evidence-manifest) | `implemented-and-tested` |
| **Phase 4C.1C.0 Operator Receipt** | `research/evidence/phase-4c.1c.0-preflight/t4-operator-script-receipt.json` (one-command operator receipt, 13/13 static safety validations, notebook superseded) | `generated-and-verified` |
| **Phase 4C.1C.0d Reusable Bundle Receipt** | `research/evidence/phase-4c.1c.0-preflight/reusable-bundle-receipt.json` (724,633,600 bytes, SHA-256 `d49a106f...`, 250 dev sources, 91 inner val sources, 0 locked test, nested cohorts N50/N100/N250) | `generated-and-verified` |
| **Phase 4C.1C.0d 15-Run T4 Operator Receipt** | `research/evidence/phase-4c.1c.0-preflight/t4-operator-all-stage1-receipt.json` (resumable 15-run operator script, 49,684 bytes, SHA-256 `bae476db...`, 91/91 assertions, ephemeral staging from verified code archive, config_hash & requirements_sha256 locked fail-closed, strict dictionary checksums, exact inner_validation cohort check, baseline pip-freeze preserved, completed run N50 seed42 verified COMPLETED_VALID, remote completed runs 1/15, 14 remaining) | `generated-and-verified` |
| **Phase 4C.1C.0d Execution Code Receipt** | `research/evidence/phase-4c.1c.0-preflight/execution-code-n250-receipt.json` (snapshot HEAD `79bb115...`, archive `phase_4c1_code_79bb115.tar.gz`, SHA-256 `5e775a77...`, operatorSha256 `bae476db...`, canonicalNotebookSha256 `d3b27ce0...`, bundleArchiveSha256 `d49a106f...`, bundleContentSha256 `c365c812...`) | `generated-and-verified` |
| **Phase 4C.1C.0d.1 Snapshot Binding Audit** | `research/evidence/phase-4c.1c.0-preflight/execution-snapshot-binding-audit.json` (verified diff `79bb115..dbb51d1` contains 0 executable code changes, code archive immutable, audited commit `dbb51d1...`) | `audited-and-verified` |
| **Phase 4C.1C.0d.1 Pre-Execution Verdict** | `research/evidence/phase-4c.1c.0-preflight/PRE_EXECUTION_GO_NO_GO.json` (verdict GO / RESUME: capability-based GPU detection, fail-closed scientific invariants, 80/80 operator checks, 19/19 notebook tests, canonical notebook compiled, remote completed runs=1/15, remaining=14, locked-test=0, stage 2=0) | `sealed-and-verified` |
| **Phase 4C.2 Stage 2 Configuration** | `ml/configs/phase_4c2_stage2_finetuning.yaml` (preregistered protocol, partial fine-tuning allowlist, differential LRs, 15-run matrix) | `implemented-and-tested` |
| **Phase 4C.2 Stage 2 Runner** | `ml/training/run_phase_4c2.py` (dedicated runner for Stage 2 partial fine-tuning with frozen BN policy and differential LRs) | `implemented-and-tested` |
| **Human Authorization Schemas** | `docs/schemas/human-unsealing-authorization.v1.schema.json` (legacy), `human-unsealing-authorization.v2.schema.json` (confirmatory structural/required-field contract; exact commit/archive/components supplied by external binding), and `human-manifest-custodian-authorization.v1.schema.json` (one data-integrity sealing session, no model/metrics/modification/disclosure) | `current-schemas-enforced-with-external-exact-bindings` |
| **Phase 4C.2G.0 Confirmatory Evaluator Preflight/Interface Engine** | `ml/evaluation/locked_test_evaluator.py`, `confirmatory_metrics.py`, `run_phase_4c2f_evaluator.py` (authorization/schema/checkpoint preflight, tamper-evident ledger, evaluation interface, metrics and synthetic dry-run are implemented; formal CLI does not call execute_checkpoint_evaluation, sealed package has no real inference function, locked-test loader, five-checkpoint orchestration, or Windows read-only/isolation enforcement compatible with the current controller receipt) | `preflight-implemented-real-execution-blocked` |
| **Phase 4C.2G.0.12 Complete Confirmatory Execution Surface** | `ml/evaluation/run_phase_4c2g_confirmatory.py`, `phase_4c2g_dataset.py`, `phase_4c2g_model.py`, `phase_4c2g_windows.py`, `phase_4c2g_io.py` (external binding verified before locked-test mount/read; exact 343-source/686-sample manifest contract; only exact regular root `custodian_inventory.json` excluded from sample inventory; nested metadata, arbitrary extras, traversal, symlink/reparse, duplicates and checksum failures rejected; canonical five-seed orchestration) | `sealed-exact-package-preflight-pass-pending-human-recovery-adjudication` |
| **Phase 4C.2G.0.10 Authorized-Session Orchestrator & Mixed-Commit Package Builder** | `scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1`, `scripts/research/build_phase4c2g_execution_package.py` (exact ifIndex + Name/InterfaceDescription/MacAddress resolution through `Get-NetAdapter -IncludeHidden`; InputObject pipeline isolation/recovery/restoration; self-testing contract mode with redirected child `-WhatIf`; bounded watchdog polling; deterministic package independently binds immutable evaluator, controller hotfix, and execution-package commits) | `implemented-tested-sealed-packaged-preflight-pass` |
| **Phase 4C.2G.0.10 Package/Binding/Authorization Evidence** | `research/evidence/phase-4c.2g.0.10/` (source/diff root cause, mixed-commit source binding, canonical execution binding, package receipt, exact packaged-controller preflight, preserved scientific invariants, supersession lineage, pending authorization request) | `ready-for-human-evaluator-authorization` |
| **Phase 4C.2G.0.5.1 Authorization Schema, External Binding & Evidence** | `docs/schemas/human-unsealing-authorization.v2.schema.json`, `research/evidence/phase-4c.2g.0.5/execution_authorization_binding.json`, `research/evidence/phase-4c.2g.0.5.1/` (schema validates structure; external file binds exact commit/archive/schema/components; missing independent locked manifest commitment yields `BLOCKED_MANIFEST_COMMITMENT_ABSENT`) | `generated-verified-blocked` |
| **Phase 4C.2G.0.6 Manifest Custodian Sealer** | `scripts/research/seal_locked_test_manifest.py` (evaluator/model/image-decoder/network independent; read-only/isolation proof before reservation; exact inventory/streaming SHA-256; NFC deterministic manifest; atomic outputs; interruption accounting and no retry) | `implemented-synthetic-tested-sealed` |
| **Phase 4C.2G.0.6 One-UAC Controller & Package Builder** | `scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1`, `build_phase4c2g_manifest_custodian_package.py` (exact authorization/package preflight, watchdog/isolation/read-only/sealer/restoration flow; zero evaluator calls; deterministic Git-blob archive) | `implemented-contract-tested-sealed` |
| **Phase 4C.2G.0.6 Custodian Authorization & Evidence** | `docs/schemas/human-manifest-custodian-authorization.v1.schema.json`, `research/evidence/phase-4c.2g.0.6/` (prospective addendum, canonicalization, exact source/archive/schema binding, PRE_CUSTODIAN gate; role-separated automated process, no independent human claimed) | `ready-for-human-manifest-custodian-approval` |
| **Phase 4C.2G.0.7 Custodian PowerShell 5.1 Hotfix** | `scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1`, `ml/tests/test_phase_4c2g_manifest_custodian.py`, `research/evidence/phase-4c.2g.0.7/` (scalarized equal-key component-count comparison at formal preflight; real failure regression; exact Git-object reseal at `a86888d`) | `resealed-awaiting-exact-reauthorization` |
| **Phase 4C.2G.0.8 Authorized Manifest Commitment** | `research/evidence/phase-4c.2g.0.8/` (redacted authorization lineage, exact commitment/hash counters, post-commit no-retry adjudication, verified network/watchdog/VHDX cleanup; manifest content remains external) | `commitment-sealed-evaluator-authorization-absent` |
| **Phase 4C.2G.0.9 Evaluator Authorization Preparation** | `research/evidence/phase-4c.2g.0.9/` (exact package/schema/component/checkpoint/manifest bindings; non-actionable pending request draft; fixture results; non-mutating Windows cmdlet reproduction; disabled post-approval command plan) | `blocked-sealed-orchestrator-adapter-parameter-binding` |
| **Locked-Test Manifest Schema** | `docs/schemas/locked-test-manifest.v1.schema.json` (exact 686 samples, binary labels and IDs, locked_test partition, per-file SHA-256 and strict additionalProperties=false) | `schema-enforced` |
| **Phase 4C.2G.0.4 Readiness & Execution-Surface Audit Evidence** | `research/evidence/phase-4c.2g.0.4/` (PHASE_REPORT.md, readiness_retry_receipt_binding.json, sealed_execution_surface_audit.json, environment.json, provenance_bindings.json) | `generated-and-verified` |
| **Phase 4C.2G.0.3.4 Egress/Reconciliation Controller** | `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` (PowerShell v1.4.0: ActiveStore IPv4/IPv6 route-owner isolation, protected internal adapters informational only when route-free, VPN and unidentified route-owner fail-closed checks, all-exit streaming verifier receipt SHA-256 binding with current-session timestamp validation, strict array-valued persistent_routes_ignored, exact adapter identity restoration polling up to 60 seconds, adapters initially Up require both AdminStatus=Up and Status=Up, NETWORK_RECOVERY_REQUIRED retains watchdog, non-mutating contract validation) | `implemented-and-tested` |
| **Phase 4C.2G.0.3.4 Standalone Offline Verifier & Wrapper** | `scripts/research/verify_phase_4c2g_offline_runtime.py`, `scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1` (shared route-owner egress semantics; output split into active_egress_adapters, protected_internal_adapters, connected_adapters_informational, persistent_routes_ignored; wrapper reports exact machine-readable reasons and never claims an active route when none exists) | `implemented-and-tested` |
| **Phase 4C.2G.0.3.4 Evidence Package** | `research/evidence/phase-4c.2g.0.3.4/` (PHASE_REPORT.md, runtime_attempt_2125_audit.json, hotfix_contract.json, offline_verifier_source_binding.json, environment.json, provenance_bindings.json) | `generated-and-verified` |
| **Phase 4C.2G.0.3.3 Pipeline Object Isolation Controller** | `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` (historical v1.3.2 controller; superseded by Phase 4C.2G.0.3.4) | `superseded` |
| **Phase 4C.2G.0.3.3 Evidence Package** | `research/evidence/phase-4c.2g.0.3.3/` (PHASE_REPORT.md, adapter_cmdlet_contract_audit.json, interrupted_attempt_parameter_binding_audit.json, stale_watchdog_cleanup_attempt_audit.json, passive_route_source_disagreement_attempt_audit.json, adapter_resolution_contract.json, controller_source_binding.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0.3.2 Hardened Recovery Script Controller** | `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` (PowerShell v1.2.0 hardened route-based network isolation controller: atomic `.part` staging with fileStream.Flush($true), AST syntax validation via [System.Management.Automation.Language.Parser]::ParseFile, semantic allowlist verification, automated quarantine of invalid recovery scripts to failed_recovery_scripts/, single-quote escaping, -ValidateRecoveryScriptOnly test mode, DryRun pre-flight AST validation, readiness status READY_FOR_SINGLE_UAC_READINESS_RETRY) | `implemented-and-tested` |
| **Phase 4C.2G.0.3.2 Evidence Package** | `research/evidence/phase-4c.2g.0.3.2/` (PHASE_REPORT.md, recovery_script_root_cause.json, generated_recovery_ast_test_receipt.json, interrupted_attempt_audit.json, timestamp_correction.json, controller_source_binding.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0.3.1 Minimal Network-Isolation Controller** | `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` (PowerShell v1.1.0 minimal route-based network isolation controller: minimal adapter selection based on InterfaceIndex and route ownership, active default route 0.0.0.0/0 and ::/0 ownership check, strict protection of internal adapters VMnet1/VMnet8/WSL/Hyper-V, 15-minute scheduled task recovery watchdog with AST syntax validation and schtasks /query read-back verification, iterative multi-round rescan with ambiguous route owner fail-closed restoration, 10-field DryRun table, 4 summary counters, single-UAC elevation workflow, DryRun inspection verdict DRY_RUN_INSPECTION_PASS, readiness status READY_FOR_SINGLE_UAC_READINESS_TEST) | `implemented-and-tested` |
| **Phase 4C.2G.0.3.1 Evidence Package** | `research/evidence/phase-4c.2g.0.3.1/` (PHASE_REPORT.md, minimal_adapter_selection_contract.json, route_to_adapter_dry_run_receipt.json, recovery_watchdog_contract.json, controller_security_audit.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0.3 Automated Network-Isolation Controller** | `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` (PowerShell automated network isolation controller: remote session fail-closed detection RDP/SSH/WinRM/CI, Administrator privilege check with detached UAC worker dispatch, snapshot of active egress adapters, 10-minute scheduled task recovery watchdog Phase4C2G_Emergency_Network_Recovery, try/finally exact allowlist adapter restoration, passive zero-probe inspection, offline verifier wrapper invocation, 12-point receipt validation, atomic receipt emission outside Git, DryRun inspection verdict DRY_RUN_INSPECTION_PASS, non-elevated verdict USER_UAC_CONFIRMATION_REQUIRED) | `implemented-and-tested` |
| **Phase 4C.2G.0.3 Evidence Package** | `research/evidence/phase-4c.2g.0.3/` (PHASE_REPORT.md, automated_isolation_controller_binding.json, controller_dry_run_receipt.json, controller_security_audit.json, recovery_watchdog_contract.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0.2B Offline Verifier Runner** | `scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1` (PowerShell wrapper resolving exact local host paths, 0 placeholders, prints current UTC/HEAD/receipt warning, exits 0/2/1, calls verify_phase_4c2g_offline_runtime.py only) | `implemented-and-tested` |
| **Phase 4C.2G.0.2B Evidence Package** | `research/evidence/phase-4c.2g.0.2b/` (PHASE_REPORT.md, OFFLINE_USER_RUNBOOK.md, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0.2A Standalone Offline Verifier** | `scripts/research/verify_phase_4c2g_offline_runtime.py` (fail-closed offline runtime verifier: Python stdlib only, 0 socket, 0 DNS, 0 HTTP, 0 torch/evaluator calls, atomic receipt generation via `.part` and `fsync`, passive network route/adapter inspection, worktree clean status, sealed components SHA-256, 5 canonical checkpoint digests, pip freeze checksum verification, unmounted locked-test confirmation) | `implemented-and-tested` |
| **Phase 4C.2G.0.2A Evidence Package** | `research/evidence/phase-4c.2g.0.2a/` (PHASE_REPORT.md, TIMESTAMP_CORRECTION_AUDIT.json, offline_verifier_source_binding.json, offline_verifier_test_receipt.json, PRE_PHYSICAL_DISCONNECTION_GATE.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G Preparation & Runtime Test Suite** | `ml/tests/test_phase_4c2g_preparation.py` (123 tests; Phase 4C.2G.0.3.4 adds g114-g123 for exact protected-internal-adapter fixture, VPN/proxy failures, persistent-route exclusion, accurate wrapper diagnostics, receipt hash/session binding, strict restoration polling/watchdog behavior, and locked-test zero access) | `implemented-and-tested` |
| **Phase 4C.2G.0.2 Evidence Package** | `research/evidence/phase-4c.2g.0.2/` (PHASE_REPORT.md, OFFLINE_HOST_PREPARATION.json, EXECUTION_WORKTREE_VERIFICATION.json, CHECKPOINT_STAGING_VERIFICATION.json, DEPENDENCY_ENVIRONMENT_LOCK.json, NETWORK_ISOLATION_INSPECTION.json, FILESYSTEM_PREPARATION.json, PRE_AUTHORIZATION_RUNTIME_GATE.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2G.0 Evidence Package** | `research/evidence/phase-4c.2g.0/` (PHASE_REPORT.md, EXECUTION_PACKAGE_RECEIPT.json, OFFLINE_RUNTIME_READINESS.json, HUMAN_AUTHORIZATION_REQUEST.json, PRE_AUTHORIZATION_GO_NO_GO.json, provenance_bindings.json, environment.json, HUMAN_APPROVAL_TEMPLATE.md) | `generated-and-verified` |
| **Phase 4C.2F.2 Evaluator Test Suite** | `ml/tests/test_phase_4c2f_evaluator.py` (45 unit, contract, exactness, and fault-injection tests: 30 base + 15 Phase 4C.2F.2 regression tests for schema const exactness, seed ordering, format checker, expiry oneOf, effective commit lock, pre-hotfix commit rejection, schema SHA self-verification, future timestamp rejection) | `implemented-and-tested` |
| **Phase 4C.2F.2 Evidence Package** | `research/evidence/phase-4c.2f/` (PHASE_REPORT.md, evaluator_contract.json, metric_implementation_contract.json, bootstrap_contract.json, evaluator_source_binding.json, checkpoint_resolution_audit.json, synthetic_dry_run_receipt.json, PRE_UNSEALING_GO_NO_GO.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2E Preregistration Test Suite** | `ml/tests/test_phase_4c2e_preregistration.py` (30 unit, contract, and fault-injection tests: candidate protocol locked as Stage 1 N=250, 5 seeds and 5 checkpoint hashes verified against Stage 1 lineage, no cherry-picking, no Stage 2 checkpoints, no retraining, no post-hoc ensemble, primary aggregate Macro-F1 across 5 checkpoints, source-cluster bootstrap 10,000 replicates with RNG seed 20261002, 0.5000 balanced binary uninformative reference, secondary descriptive metrics, evaluation-only calibration, zero accesses, clean paths, 10 fault-injections PASS) | `implemented-and-tested` |
| **Phase 4C.2E Evidence Package** | `research/evidence/phase-4c.2e/` (PHASE_REPORT.md, candidate_checkpoint_binding.json, confirmatory_metrics_plan.json, unsealing_protocol.json, LOCKED_TEST_PREREGISTRATION.json, PRE_EXECUTION_GO_NO_GO.json, provenance_bindings.json, environment.json) | `generated-and-verified` |
| **Phase 4C.2D Model Selection Tool** | `scripts/research/finalize_phase_4c2d_model_selection.py` (verifies 30-run cross-artifact numerical parity $\le 10^{-6}$, audits Stage 1 lineage and checkpoint reload contract, executes conservative model-selection decision, exact N=100 seed variability narrative with 3 exact SDs, two-step provenance model, verifies locked-test seal, generates CSV/JSON evidence and PHASE_REPORT.md) | `implemented-and-tested` |
| **Phase 4C.2D Test Suite** | `ml/tests/test_phase_4c2d_model_selection.py` (14 unit and regression tests: 10 base parity/lineage tests + 3 variability exact SD parity and fault-injection tests + 1 non-circular evidence seal test) | `implemented-and-tested` |
| **Phase 4C.2D Evidence Package** | `research/evidence/phase-4c.2d/` (PHASE_REPORT.md, FINAL_MODEL_SELECTION.json, final_metric_parity.json, stage1_metric_lineage.csv, stage1_metric_lineage.json, locked_test_readiness.json, environment.json, provenance_bindings.json) | `generated-and-verified` |
| **Phase 4C.2C.1 Lineage Reconciliation Tool** | `scripts/research/reconcile_phase_4c2c_lineage.py` (audits 7 Stage 2 raw sources, fail-closed cross-phase discrepancy gate, canonical endpoint decision generator, Erratum generator, deterministic machine-readable lineage CSV/JSON outputs) | `implemented-and-tested` |
| **Phase 4C.2C.1 Lineage & Paired Test Suite** | `ml/tests/test_phase_4c2c_paired_analysis.py` (40 unit and regression tests: 20 paired analysis tests + 20 lineage reconciliation & calibration semantics tests: C.0 discrepancy detection, receipt best/final extraction, metrics.json extraction, prediction recomputation, epoch_history computation, checkpoint metadata extraction, wrong root, duplicate candidates, cross-artifact checksum mismatch, lineage mismatch blocking, canonical endpoint determinism, markdown parity, calibration separation, overconfidence wording guard, legend verification, sparse bins, erratum verification, raw artifact byte-identity, locked-test denial) | `implemented-and-tested` |
| **Phase 4C.2C.1 Evidence Package** | `research/evidence/phase-4c.2c.1/` (PHASE_REPORT.md, stage2_metric_lineage.csv, stage2_metric_lineage.json, cross_phase_discrepancies.json, canonical_endpoint_decision.json, PHASE_4C2C0_ERRATUM.md) | `generated-and-verified` |
| **Phase 4C.2C Paired Analysis Tool** | `scripts/research/analyze_phase_4c2_paired.py` (canonical paired Stage 1 vs Stage 2 analysis script, CLI flags, 15-pair discovery, metric recomputation parity, exploratory paired t-test, exact 32 sign-flip permutation tests, Holm step-down correction, 95% t-interval df=4, multi-seed calibration reliability diagrams, publication SVG/PNG figure generator, machine-readable CSV/JSON evidence) | `implemented-and-tested` |
| **Phase 4C.2C Paired Analysis Test Suite** | `ml/tests/test_phase_4c2c_paired_analysis.py` (20 unit & fault-injection tests: 15-pair matching, missing S1/S2 fail-closed, duplicate N/seed fail-closed, receipt stage mismatch fail-closed, locked-test access fail-closed, stage 1 write fail-closed, source cohort mismatch fail-closed, target mismatch fail-closed, metric recomputation parity, delta direction identity, paired t-test analytical parity, exact 32 sign-flip permutations, minimum p-value 0.0625, Holm correction correctness, 95% t-interval correctness, clean paths, clean Git evidence, deterministic outputs, prohibited claims guard) | `implemented-and-tested` |
| **Phase 4C.2C Evidence Package** | `research/evidence/phase-4c.2c/` (PHASE_REPORT.md, paired_run_metrics.csv, primary_statistical_tests.csv/json, paired_summary.csv/json, calibration_summary.csv, analysis_environment.json, provenance_bindings.json, figures/paired_macro_f1_by_n, delta_macro_f1_by_seed, calibration_comparison in SVG and PNG) | `generated-and-verified` |
| **Phase 4C.2C.0 Ingestion & Audit Tool** | `scripts/research/ingest_phase_4c2_results.py` (portable streaming SHA-256, sidecar parser, strict TAR safety audit, atomic extraction, idempotent reuse, 15-run artifact verification, prediction cohort & leakage guards, provenance verifier) | `implemented-and-tested` |
| **Phase 4C.2C.0 Ingestion Test Suite** | `ml/tests/test_phase_4c2_results_ingest.py` (21 unit and fault-injection tests: valid archive, wrong hash/bytes, malformed sidecar, corrupt archive, absolute/traversal/symlink/special entries, duplicate entries, valid reuse, invalid fail-closed, missing run/artifact, checksum mismatch, wrong N/seed, locked-test access, Stage 1 writes, prediction cohort anomalies, trainable inventory mismatch, provenance checks) | `implemented-and-tested` |
| **Phase 4C.2C.0 Evidence Package** | `research/evidence/phase-4c.2c.0/` (import_audit_summary.json, archive_inventory.json, PHASE_REPORT.md) | `generated-and-verified` |
| **Phase 4C.2A Preregistration Evidence** | `research/evidence/phase-4c.2a/` (dataset_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, stage1_stage2_hyperparameter_diff.json, PHASE_REPORT.md) | `generated-and-verified` |
| **Phase 4C.2A Preregistration Test Suite** | `ml/tests/test_phase_4c2_preregistration.py` (9 unit tests: dataset binding read-only, namespace isolation, cohort/seed alignment, layer allowlist 204,674 params, differential LRs, capability-based GPU policy, locked-test block, 0-run invariants) | `implemented-and-tested` |
| **Phase 4C.2A.1 Contract Test Suite** | `ml/tests/test_phase_4c2_implementation_contract.py` (14 contract tests: actual named params, shapes/numel, init contract, hyperparameter diff, FocalLoss parity, frozen BN buffer invariance, gradient allowlist, optimizer groups, receipt schema, output rejection, locked test denial, capability GPU, 0 training runs) | `implemented-and-tested` |
| **Phase 4C.2B.3.1 Operator Script** | `scripts/phase_4c2_execute_all.sh` (canonical resumable 15-run operator script, 64,776 bytes, SHA-256 `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`, quoted heredocs `<<'PY'`, 10 TAR security fixtures, jsonschema preflight gate, 5 GiB disk capacity formula, safe deletion regex guards, failed publish quarantine, CRLF->LF normalization, exact 20-parameter scientific env lock with requirements hash binding, hardened artifact verifier, capability-based GPU detection, atomic publishing via `.part`, preflight retry failure archiving, resume fail-closed) | `implemented-and-tested` |
| **Phase 4C.2B Execution Snapshot Receipt** | `research/evidence/phase-4c.2b/execution_snapshot_receipt.json` (snapshot HEAD `9ee7fdb...`, archive `phase_4c2_code_9ee7fdb.tar.gz`, 10,478,136 bytes, SHA-256 `951e9089...`) | `generated-and-verified` |
| **Phase 4C.2B.3.1 Operator Receipt** | `research/evidence/phase-4c.2b/operator_receipt.json` (operator receipt, 64,776 bytes, SHA-256 `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`, status READY_FOR_USER_COLAB_PREFLIGHT) | `generated-and-verified` |
| **Phase 4C.2B.3.2 Colab Launcher Notebook** | `notebooks/phase_4c2_finetuning_colab.ipynb` (canonical Colab notebook, 15,668 bytes, SHA-256 `dee8f7c46879584c006419ace8a3e79153211011de304efd15dba00d70dc36e0`, 5 cells, default `EXECUTE = False`, exact code archive binding, exact execution_9ee7fdb audit without glob, Python >= 3.10 capability check, operator SHA/bytes verification, atomic restaging, 5 canonical REQUIRED_ARCHIVES, and full post-execution audit) | `implemented-and-tested` |
| **Phase 4C.2B Pretrained Weights Binding** | `research/evidence/phase-4c.2b/pretrained_weights_binding.json` (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`, file SHA-256 `047dcff4...`, backbone state fingerprint `d42bb32a...`) | `generated-and-verified` |
| **Phase 4C.2B.3.2 Operator & Notebook Tests** | `ml/tests/test_phase_4c2_operator.py` (89 tests including 10 TAR security behavioral fixtures, 18 Section J behavioral tests, 11 Section K lock mutation tests, path guards, component hash checks, verifiers), `ml/tests/test_phase_4c2_notebook.py` (16 tests including archive parity and 8 fault injections) | `implemented-and-tested` |
| **Phase 4C.1D Deterministic Analysis Pipeline** | `scripts/research/analyze_phase_4c1d_runs.py` (audits archives, verifies 15 runs, computes exact stats, permutation tests, Holm correction, per-seed baseline comparisons, draws figures, generates all CSV/JSON/MD) | `implemented-and-tested` |
| **Phase 4C.1D Evidence & Summary** | `research/evidence/phase-4c.1d/` (run_level_metrics.csv, learning_curve_summary.csv/json, paired_seed_deltas.csv, phase_4c1_results_audit.json, phase_4c1_learning_curve_report.md, PHASE_REPORT.md, figures/) | `generated-and-verified` |
| **Phase 4C.1D Test Suite** | `ml/tests/test_phase_4c1d_analysis.py` (9 unit tests: ingestion hashes, 15-run verification, 15 checkpoint prefixes match receipts, markdown table-CSV parity, summary recalculation, baseline per seed, figure canonical summary, fault-injection detection, and forbidden phrases assertion) | `implemented-and-tested` |
| **Phase 4C.1B.6 Eval Leakage Regression Tests** | `ml/tests/test_eval_leakage_regression.py` (10 tests: fault injection, validation-only, key matching, bootstrap integrity, delta identity, CI rejection, metadata NOT_EVALUATED, dummy reproduction) | `implemented-and-tested` |
| **Phase 4C.1B.6R.1 Reproducibility Scripts** | `scripts/research/create_dummy_baseline_artifact.py`, `scripts/research/reproduce_phase_4c1b6r1_statistics.py` (paired source bootstrap, dummy baseline artifact generation) | `implemented-and-tested` |
| **Phase 4C.1B.6R.2 Reproducibility Scripts** | `scripts/research/create_stage1_validation_artifact.py`, `scripts/research/reproduce_phase_4c1b6r2_statistics.py` (Stage 1 validation filter, paired cluster bootstrap) | `implemented-and-tested` |
| **Phase 4C.1B.6R.3 Reproducibility Scripts** | `scripts/research/reproduce_phase_4c1b6r2_statistics.py` (paired cluster bootstrap verification) | `implemented-and-tested` |
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
