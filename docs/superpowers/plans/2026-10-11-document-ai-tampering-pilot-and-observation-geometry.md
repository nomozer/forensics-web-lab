# Document AI-Tampering Data Assessment & Image-Level Detection Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the accessibility of AI-tampered document datasets and execute an empirical image-level detection pilot comparing 3 observation geometries (P0 Center-crop vs P1 Letterbox vs P2 Multi-tile 5 views) across the 6 frozen locked models.

**Architecture:** A standalone, reproducible evaluation pipeline on LOCAL CPU using MobileNetV3-small frozen feature extraction and canonical DSP, evaluating on a deterministic panel of 60 document images (30 authentic + 30 forged with pixel masks) from RealText-V2 (CC-BY-NC-4.0).

**Tech Stack:** Python 3.12, PyTorch, Pillow, NumPy, SciPy, scikit-learn.

**Spec:** `research/evidence/document_pilot/PROTOCOL_DOCUMENT_PILOT_V1.0.0.md`

## Global Constraints
- Run exclusively on LOCAL CPU. Zero retraining, zero refitting, zero recalibration, zero threshold tuning.
- All 6 locked models use `MobileNetV3-small` (576-D GAP) as the visual backbone (reconciling past textual errata calling A1 EfficientNet-B0).
- Thesis title: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.” Working definition: image documents or image-based PDF pages with AI-edited content (invoices, receipts, forms, contracts).
- Working definition is strictly a pilot working scope, not claimed as advisor approved; no expansion to LLM text generation.
- Strict isolation: ZERO access to sealed cohorts (Locked-Test, TGIF N=400).
- Network download cap: $\le 50$ MB total transfer, well within remaining 324 MB budget.
- Pre-registered binary decision threshold: $s_{\text{AI}} = p(\text{ai\_edited}) + p(\text{fully\_generated}) \ge 0.5$.

---

### Task 1: Audit Literature and Dataset Accessibility Receipts
**Files:**
- Create: `research/evidence/document_pilot/DOCUMENT_DATASET_ACCESSIBILITY_AUDIT.md`
- Test: Manual verification / receipts check

- [ ] **Step 1: Document accessibility status of 5 literature targets**
  - Record findings for DocTamper (Qu et al., CVPR 2023), OSTF (Wang et al., AAAI 2025), AIForge-Doc (arXiv:2602.20569), Fake or JPEG? (arXiv:2403.17608), GADNet (ESWA 2025).
  - Document access blockers (gating, password encryption, institutional requirements).
  - Document RealText-V2 (ACM MM 2026) characteristics, licensing, and format caveats.

### Task 2: Author Protocol Specification v1.0.0
**Files:**
- Create: `research/evidence/document_pilot/PROTOCOL_DOCUMENT_PILOT_V1.0.0.md`

- [ ] **Step 1: Write Protocol Specification**
  - Lock taxonomy: `authentic` (white) vs `ai_edited` (black).
  - Lock 3 observation conditions: P0 (center-crop min(w,h)), P1 (letterbox 512x512), P2 (multi-tile 5 views).
  - Lock binary score formula $s_{\text{AI}} \ge 0.5$.
  - Lock 6 models evaluated.

### Task 3: Ingestion & Deterministic Panel Manifesting
**Files:**
- Create: `scripts/research/acquire_document_pilot_panel.py`
- Create: `data/research/document_pilot/document_pilot_manifest.json`
- Create: `research/evidence/document_pilot/document_acquisition_receipt.json`
- Create: `research/evidence/document_pilot/document_pilot_isolation_audit.json`

- [ ] **Step 1: Implement acquisition script with strict network tracking**
  - Download 30 authentic + 30 forged + 30 masks from RealText-V2 (< 50 MB total).
  - Compute SHA-256 hashes, verify PIL decode.
  - Run isolation audit against all internal datasets (0 collisions).
  - Export manifest, acquisition receipt, and isolation audit.

### Task 4: Observation Geometry Runner & Preflight Smoke Tests
**Files:**
- Create: `scripts/research/run_document_observation_pilot.py`
- Create: `ml/tests/test_document_observation_pilot.py`

- [ ] **Step 1: Implement test suite `test_document_observation_pilot.py`**
  - Test P0, P1, P2 transform functions on dummy tensors and masks.
  - Test mask retention calculation.
  - Test probability aggregation across 5 views.
- [ ] **Step 2: Implement `run_document_observation_pilot.py`**
  - Load 6 locked models from `locked_models_registry.json`.
  - Implement P0, P1, P2 transforms.
  - Run preflight on 4 samples.

### Task 5: Execute Full Document Pilot & Export Receipts
**Files:**
- Generate: `research/evidence/document_pilot/document_pilot_predictions.json`
- Generate: `research/evidence/document_pilot/document_pilot_receipt.json`

- [ ] **Step 1: Run full pilot on 60 samples across 6 models and P0, P1, P2**
  - Compute AUROC, AP, Macro-F1, Bal-Acc, FPR, TNR, Brier, ECE.
  - Compute paired deltas vs P0 with 10,000 bootstrap replicates.
  - Record CPU execution latency.
  - Export predictions and receipt.

### Task 6: Phase Report, Continuity Updates & Verification
**Files:**
- Create: `research/evidence/document_pilot/PHASE_REPORT.md`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Write Phase Report with deep empirical synthesis**
- [ ] **Step 2: Update continuity documents**
- [ ] **Step 3: Run pytest, pnpm test, pnpm continuity:check, clean commit**
