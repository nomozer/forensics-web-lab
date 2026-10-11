# Implementation Plan: Document Pilot Reconciliation, Taxonomy/Compression Confounding Audit, and Paired Replay (C0 vs C1)

## Overview
This plan rectifies the epistemic overclaims of the document observation pilot ($N=40$), audits per-sample taxonomy, provenance, format, and compression confounding on `RealText-V2`, executes a controlled paired replay (C0 Original vs C1 Symmetric JPEG) across 3 geometries (P0 Center-Crop, P1 Letterbox, P2 Multi-Tile) on all 6 locked models, disentangles shared feature extraction from per-model latency, amends the localization preparation protocol to v1.1.0, and updates all continuity artifacts.

---

## Task List

### Task 1: Taxonomy & Provenance Audit ($N=40$)
- **File**: `research/evidence/document_pilot/document_sample_taxonomy_audit.json`
- **Action**: Create comprehensive per-sample taxonomy table for all 40 samples from `document_pilot_manifest.json`, RealText-V2 dataset card, and sample reports.
- **Fields**:
  - `sample_id`
  - `source_label` (`black` / `white`)
  - `document_domain_type` (e.g. corporate, legal, receipt, education)
  - `manipulation_type` (e.g. text redaction, date backdating, title alteration, pristine)
  - `ai_manipulation_evidence` (`none_verified` / `unverified_provenance`)
  - `editor_generator` (`unverified`)
  - `mask_semantics` (binary bounding pixel mask for tampered text)
  - `verified_source_group` (`unpaired_sample`)
  - `source_relation_status` (`unpaired`, 0 stem overlap)
  - `original_format` and `sha256`
- **Classification**:
  - Verified AI manipulation: 0 samples
  - Verified non-AI manipulation: 0 tools verified
  - Manipulation provenance unverified: 20 forged samples

### Task 2: Format & Compression History Audit
- **File**: `research/evidence/document_pilot/document_format_compression_audit.json`
- **Action**: Inspect magic bytes, dimensions, modes, orientation, JPEG quantization tables (DQT), chroma subsampling (SOF0/SOF2), and encoder markers.
- **Isolation Audit Rectification**:
  - Rectify historical claim of "100% cách ly tuyệt đối": The isolation audit only checked byte SHA-256 collisions against internal manifests (`three_class_pilot_manifest.csv`, `source_control_cohort_manifest.json`, `imagenet_authentic_train_manifest.json`, `so_fake_probe_manifest_v2.json`, `tgif_train_clean_subset_manifest_locked_n400.json`). Zero hash collision confirms no byte-identical reuse, but does not prove source independence or unconfoundedness.

### Task 3: Pre-Registered Paired Replay Runner (`run_document_replay_and_compression_audit.py`)
- **File**: `scripts/research/run_document_replay_and_compression_audit.py`
- **Protocol Details**:
  - **Models**: 6 locked frozen models from `locked_models_registry.json` (`A0-global`, `A0-local`, `A1-global`, `A1-local`, `A0-visual`, `A1-visual`).
  - **Geometries**:
    - P0: Canonical center-crop: $\min(W, H)$ from center, resize to $512 \times 512$ Lanczos (DSP) and $224 \times 224$ Bicubic from 512 (Visual).
    - P1: Orientation-correct letterbox: scale $\max(W, H)$ to 512, pad symmetrically to $512 \times 512$ with neutral gray $(128, 128, 128)$, mask padded with 0. Visual is $224 \times 224$ Bicubic from 512 canvas.
    - P2: Global letterbox view + 4 fixed quadrants ($256 \times 256$ crop of 512 canvas, each upscaled to $512 \times 512$ Lanczos for DSP and $224 \times 224$ Bicubic for Visual). Unweighted arithmetic mean across the 5 view probabilities.
  - **Conditions**:
    - C0: Original decoding (orientation-correct RGB).
    - C1: Symmetric JPEG: orientation-correct RGB is encoded to JPEG once with `quality=75`, `subsampling=2` (4:2:0), `optimize=False`, then decoded to RGB before geometry transforms. Applied identically to both forged and authentic samples.
  - **Decision Rule**: $s_{\text{AI}} = p(\text{ai\_edited}) + p(\text{fully\_generated}) \ge 0.5$.
  - **Outputs**:
    - `research/evidence/document_pilot/document_pilot_replay_predictions.json`
    - `research/evidence/document_pilot/document_pilot_replay_receipt.json`

### Task 4: Metrics, Statistical Bootstrap & Latency Disentanglement
- Recalculate historical metrics from `document_pilot_predictions.json` to verify baseline reproducibility.
- Compute replay metrics for C0 and C1: Macro-F1, Bal-Acc, TPR/TNR/FPR/FNR, AUROC, AP, Multiclass Brier, Binary Brier (with 0.25 uninformative baseline reference), 10-bin ECE.
- Compute label flips and paired sample bootstrap CIs (10,000 reps) comparing C1 vs C0 within each model/geometry, and P1/P2 vs P0.
- Rigorous latency measurement:
  - Benchmark on a deterministic subpanel (4 samples: 2 forged + 2 authentic).
  - Include warm-up runs.
  - Disentangle preprocessing, shared feature extraction, per-model scoring, and total pipeline.
  - Report median, mean, P95.

### Task 5: Epistemic Reconciliation Report & Localization Protocol Amendment
- **Files**:
  - `research/evidence/document_pilot/DOCUMENT_PILOT_RECONCILIATION_REPORT.md`
  - `research/evidence/three_class_preparation/PROTOCOL_LOCALIZATION_PREPARATION_AMENDMENT_V1.1.0.md`
- **Reconciliation Points**:
  - Address the 6 overclaims (P1 superiority, 100% forensic trace preservation, P2 spectral leakage hypothesis, A1 zero false alarm generalization, natural image backbone applicability, browser feasibility).
  - Explicitly document format confounding (100% JPEG forged vs 100% PNG authentic).
  - Document C0 vs C1 sensitivity: whether geometry differences persist or collapse under symmetric recompression.
- **Protocol Amendment v1.1.0**:
  - Restore official thesis title: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”
  - Mark localization as an exploratory candidate rather than confirmed requirement.
  - Recount actual edited images with masks ($N=250$ in train, $N=91$ in holdout).
  - Acknowledge development holdout is an inspected development set.
  - Mark resource/latency budgets as unmeasured targets.
  - Remove automated Colab UNet training commitment.

### Task 6: Unit Tests, Continuity Verification & Commit
- Add unit test suite `ml/tests/test_document_pilot_reconciliation.py`.
- Run full pytest and pnpm test suites.
- Update `docs/continuity/CURRENT_STATE.md`, `CODE_INDEX.md`, and prepend to `STATUS_LEDGER.md`.
- Run continuity check (`pnpm continuity:check -- --staged`).
- Git commit on `research/independent-cohort-acquisition`.
