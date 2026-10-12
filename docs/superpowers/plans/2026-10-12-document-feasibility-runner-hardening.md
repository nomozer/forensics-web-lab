# Document Feasibility Runner Hardening & Execution Preparation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all execution-blocking defects in the document feasibility study runner: reorder source preflight before model download, strictly verify source identity and quad envelope deltas, enforce checkpoint budget preflight, implement fresh run identification and fail-closed idempotency, regenerate the turnkey Colab notebook with verified hash parity, and prove all contracts with fresh tests on LOCAL.

**Tech Stack:** Python 3.12, PyArrow, Pillow, NumPy, Pytest, TypeScript / PNPM continuity.

**Spec / Directives:** User request in Phase 4C.8 trace, `AGENTS.md`, Protocol Amendment v1.2.0.

---

### Task 1: Re-order Runner Architecture — Source Preflight Before Model Download

**Files:**
- Modify: `scripts/research/run_document_feasibility_colab.py`

- [ ] **Step 1: Move CORD-v2 source preflight before checkpoint loading**
  Ensure Stage 2 downloads/reads CORD shard and verifies all 4 sources BEFORE Stage 3 checkpoint preflight and Stage 4 `from_pretrained()` call.
- [ ] **Step 2: Strict source identity & quad envelope verification**
  Assert `gt_data["meta"]["image_id"] == attempt["source_image_id"]` (fail immediately on mismatch; no silent continue). Verify native dimensions and EXIF=1. Compute quad envelope of matching words and record exact delta `[d_ymin, d_xmin, d_ymax, d_xmax]`. Specifically record Attempt 1 delta: `[1, 0, 0, -1]`.

---

### Task 2: Checkpoint Budget Preflight & Ungated Access Guard

**Files:**
- Modify: `scripts/research/run_document_feasibility_colab.py`

- [ ] **Step 1: Define FP16 pipeline file inventory and preflight function**
  Explicitly list the 18 required FP16 pipeline files with projected download size: 6,941,218,469 bytes (6.4645 GiB <= 12.0 GiB ceiling).
- [ ] **Step 2: Verify remote access and available disk space**
  Perform HTTP HEAD check on model repository to verify public/ungated access (no 401/403) before download. Check free disk space >= 10.0 GB. Emit `checkpoint_budget_preflight_receipt.json`.

---

### Task 3: Fresh Run Identification, Idempotency & Failure Isolation

**Files:**
- Modify: `scripts/research/run_document_feasibility_colab.py`

- [ ] **Step 1: Implement fresh run tagging and idempotency guard**
  Tag runs with `run_id` and short plan hash. If target run directory already contains completed run artifacts, fail-closed with a clear message to prevent overwriting or duplicate generation.
- [ ] **Step 2: Attempt-level and run-level failure receipts with partial archive**
  Distinguish statuses: `COMPLETED`, `FAILED`, `NOT_EXECUTED`. Save `attempt_failure_receipt.json` on attempt exception. Save `run_failure_receipt.json` on run-level error. Always package partial zip archive.

---

### Task 4: Regenerate Turnkey Colab Notebook with Hash Parity

**Files:**
- Modify: `scripts/research/build_document_feasibility_colab_notebook.py`
- Modify: `notebooks/independent_document_feasibility_colab.ipynb`

- [ ] **Step 1: Update notebook generator and compile all cells**
  Re-serialize notebook embedding updated runner and plan. Verify all code cells compile without errors via `compile()`.
- [ ] **Step 2: Verify bit-exact SHA-256 payload parity**
  Confirm staged payloads in Cell 2 match files on disk bit-for-bit.

---

### Task 5: Local Test Suite Expansion, Continuity Check & Handover

**Files:**
- Modify: `ml/tests/test_document_feasibility_geometry.py`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`
- Modify: `research/evidence/document_pilot/FEASIBILITY_STUDY_PREPARATION_REPORT.md`

- [ ] **Step 1: Add new unit tests for runner hardening**
  Test source-before-model order, strict mismatch rejection, quad envelope delta, checkpoint budget preflight, fresh run idempotency, and notebook hash parity.
- [ ] **Step 2: Run verification test suite**
  Execute `pytest ml/tests/test_document_feasibility_geometry.py -v`, `pnpm test`, and `pnpm typecheck`.
- [ ] **Step 3: Update continuity docs and verify continuity checker**
  Update continuity files, run `node scripts/continuity-check.mjs`, commit and push to branch.
