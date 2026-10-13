# AnyText Launcher Fix & Colab Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining execution flaws in AnyText launcher, eliminate external CORD shard downloads in favor of the existing SDXL archive, isolate cache, ensure strict AnyText git revision checkout, enforce download inventory filtering, and produce a verified, runnable Colab notebook with technical preflight.

**Architecture:** 
- Input pipeline extracts authentic controls directly from the verified SDXL pilot zip archive (`document_feasibility_results.zip`), matching byte-exact SHA-256 without re-encoding.
- Model weights download leverages ModelScope's `snapshot_download` with `ignore_file_pattern=['nlp_csanmt_translation_zh2en/*']` to exclude 7.34 GiB translator weights and stay well under the 12 GiB budget.
- Cache directory is strictly isolated outside artifact directories, and archive packaging uses an explicit whitelist ensuring total zip size <= 100 MiB.
- Contact sheet incorporates zoomed ROIs for granular human assessment across Authentic, SDXL, AnyText Raw, and Final Composited images.

**Tech Stack:** Python 3, PyTorch, ModelScope, Pillow, NumPy, Git, Google Colab (Jupyter).

**Spec:** USER_REQUEST and Phase Continuity Protocols in `docs/continuity/CURRENT_STATE.md`.

## Global Constraints
- Target sources: exactly two CORD replacement sources (Attempt A: row 22 "280,000" -> "350,000", seed 20261012; Attempt B: row 61 "Populaire Chocolate" -> "Magnum Gold Almond", seed 20261013).
- No downloading heavy checkpoints or running full diffusion generation on LOCAL CPU.
- No modifying sealed TGIF, Locked-Test, or SDXL pilot archives.
- No Hugging Face tokens/login required.
- Outside-mask bitwise equality contract strictly preserved (`np.array_equal == True`, `max_diff == 0`).
- AnyText git commit locked to `3d4a260c67d92c0e27d6e18ed69990e19a40de6d`.
- ModelScope revision locked to `v1.1.3`.

---

### Task 1: Canonical Plan LF Normalization and Plan Builder Update

**Files:**
- Modify: `scripts/research/build_anytext_feasibility_plan.py`
- Modify: `research/evidence/document_pilot/anytext_feasibility_plan.json`

- [ ] **Step 1: Update `build_anytext_feasibility_plan.py` to write strict LF newlines and update plan fields if necessary**
- [ ] **Step 2: Run `build_anytext_feasibility_plan.py` to generate `anytext_feasibility_plan.json` with strict LF bytes**
- [ ] **Step 3: Record canonical SHA-256 hash of `anytext_feasibility_plan.json`**

---

### Task 2: Refactor Runner (`run_anytext_colab.py`) for Safe Archive Inputs, Filtered Downloads, Cache Isolation & Enhanced Reporting

**Files:**
- Modify: `scripts/research/run_anytext_colab.py`

- [ ] **Step 1: Implement safe zip extraction and hash verification from `document_feasibility_results.zip`**
- [ ] **Step 2: Isolate model cache (`--cache-dir` outside `run_dir` / `output_root`)**
- [ ] **Step 3: Implement filtered `snapshot_download` ignoring translator directory `nlp_csanmt_translation_zh2en/*`**
- [ ] **Step 4: Update inference to measure CUDA latency with `torch.cuda.synchronize()` and track VRAM**
- [ ] **Step 5: Enhance contact sheet to include zoomed ROI crops for all 5 columns**
- [ ] **Step 6: Update `human_review.csv` with dual criteria (`exact_text_success` and `visual_realism`)**
- [ ] **Step 7: Implement artifact whitelist packager enforcing <= 100 MiB ceiling**

---

### Task 3: Update Colab Notebook Generator and Rebuild Notebook

**Files:**
- Modify: `scripts/research/build_anytext_colab_notebook.py`
- Rebuild: `notebooks/independent_anytext_feasibility_colab.ipynb`

- [ ] **Step 1: Update Cell 2 to checkout `3d4a260c67d92c0e27d6e18ed69990e19a40de6d` and verify HEAD**
- [ ] **Step 2: Update Cell 3 to stage files with bit-exact LF bytes**
- [ ] **Step 3: Update Cell 4 to use archive upload and safe extraction instead of HF CORD parquet**
- [ ] **Step 4: Update Cell 5 to pass isolated cache directory `/content/modelscope_cache`**
- [ ] **Step 5: Run builder script to generate `notebooks/independent_anytext_feasibility_colab.ipynb`**
- [ ] **Step 6: Verify JSON syntax, cell AST compilation, and safety constraints**

---

### Task 4: Comprehensive Test Suite Update and Verification

**Files:**
- Modify: `ml/tests/test_anytext_feasibility.py`

- [ ] **Step 1: Add unit tests for safe zip extraction and authentic SHA verification**
- [ ] **Step 2: Add unit tests for LF line ending consistency and staging parity**
- [ ] **Step 3: Add unit tests for artifact packaging whitelist and cache exclusion**
- [ ] **Step 4: Run full test suite with virtual environment pytest**

---

### Task 5: Continuity Documentation and Git Synchronization

**Files:**
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Update CURRENT_STATE.md with launcher fix details and PREPARED_NOT_EXECUTED status**
- [ ] **Step 2: Update CODE_INDEX.md if schemas or CLI signatures were enhanced**
- [ ] **Step 3: Add summary entry to STATUS_LEDGER.md**
- [ ] **Step 4: Run `pnpm continuity:check`**
- [ ] **Step 5: Review git diff, commit and push to branch `research/independent-cohort-acquisition`**
