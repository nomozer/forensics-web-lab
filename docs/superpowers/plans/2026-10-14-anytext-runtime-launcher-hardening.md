# AnyText Runtime Launcher Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the AnyText Colab launcher and runner based on observed Colab runtime evidence: add `ujson==5.9.0` dependency, patch ModelScope download to ignore `README.md` and non-essential documentation, implement cache reuse and attempt protection, stream unbuffered logs with structured failure handling and partial archive packaging, and rebuild the canonical notebook with full local verification.

**Architecture:**
- Dependencies: Dedicated Python 3.10 environment installs pinned inference stack including `ujson==5.9.0`; smoke test asserts all versions and functional glyph guidance.
- Download: `snapshot_download` ignores `README.md`, `*.md`, and `nlp_csanmt_translation_zh2en/*`. Valid cached checkpoints and CLIP weights are detected and reused directly.
- Concurrency & Attempts: Single-instance lock file prevents concurrent runners. Completed attempt receipts are verified before inference; completed attempts are never re-run.
- Observability: Runner invoked with `python -u`, streaming stdout/stderr in real-time to notebook and log file. Granular stage failure reporting produces `failure_receipt.json` and `anytext_feasibility_partial_results.zip`.

**Tech Stack:** Python 3.10, PyTorch 2.0.1+cu118, ModelScope 1.10.0, Pillow 9.5.0, ujson 5.9.0, pytest.

---

### Task 1: Add `ujson==5.9.0` to Dependencies and Smoke Test

**Files:**
- Modify: `scripts/research/build_anytext_colab_notebook.py`

- [ ] **Step 1: Add `ujson==5.9.0` to dependencies list in Cell 2**
- [ ] **Step 2: Add `import ujson` and version assertion to smoke test code in Cell 2**

---

### Task 2: Harden Download Filtering, Missing Content-Length Handling, and Cache Reuse

**Files:**
- Modify: `scripts/research/run_anytext_colab.py`

- [ ] **Step 1: Update `ignore_file_pattern` to include `README.md` and `*.md` alongside translator**
- [ ] **Step 2: Implement pre-download cache verification to reuse valid existing checkpoint and CLIP weights**
- [ ] **Step 3: Handle download errors on non-essential metadata gracefully while strictly verifying required weights hash and size**

---

### Task 3: Concurrency Guard, Stage-Level Failure Handling, and Attempt Protection

**Files:**
- Modify: `scripts/research/run_anytext_colab.py`

- [ ] **Step 1: Add concurrency lock file guard (`.runner.lock`) to prevent concurrent runner runs**
- [ ] **Step 2: Add attempt completion check to prevent re-generating already completed attempts**
- [ ] **Step 3: Implement root-level failure handling that records stage, traceback, error and produces partial results archive**

---

### Task 4: Enhance Notebook Runner Execution Cell with Real-Time Streaming and Error Summary

**Files:**
- Modify: `scripts/research/build_anytext_colab_notebook.py`

- [ ] **Step 1: Update Cell 5 to invoke runner with `-u` and stream stdout/stderr line-by-line**
- [ ] **Step 2: Update Cell 5 to trap failures and display formatted failure summary block with receipt and log links**
- [ ] **Step 3: Update Cell 6 to handle downloading partial archive if main archive failed**

---

### Task 5: Rebuild Canonical Notebook, Update Tests, and Verify

**Files:**
- Modify: `ml/tests/test_anytext_feasibility.py`
- Rebuild: `notebooks/independent_anytext_feasibility_colab.ipynb`

- [ ] **Step 1: Run builder to regenerate canonical notebook**
- [ ] **Step 2: Add unit tests for `ujson`, README filter, cache reuse, attempt protection, and failure handling**
- [ ] **Step 3: Run pytest on `ml/tests/test_anytext_feasibility.py`**
- [ ] **Step 4: Run `pnpm test` and `pnpm continuity:check`**

---

### Task 6: Continuity Documentation, Commit & Push

**Files:**
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Update continuity documents with hardened launcher details**
- [ ] **Step 2: Verify git status, commit and push to `research/independent-cohort-acquisition`**
