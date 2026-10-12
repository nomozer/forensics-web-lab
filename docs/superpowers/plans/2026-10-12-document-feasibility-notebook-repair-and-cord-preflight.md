# Document Feasibility Notebook Repair & CORD Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair all 8 identified defects in the document feasibility study Colab launcher and runner, establish Protocol Amendment v1.2.0 for the reconciled 260 MB CORD shard budget, perform true local CORD-v2 data preflight and annotation verification, programmatically generate a self-contained turnkey notebook, and verify all contracts with strict local tests.

**Architecture:** 
1. Reconcile budget and protocol via `INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.2.0.md` (raising original data ceiling from 200 MB to 260 MB to accommodate the 242.08 MB CORD-v2 validation shard).
2. Harden geometry and composite contract in `ml/evaluation/document_feasibility_geometry.py` with strict bounding box boundary validation.
3. Update canonical plan `document_feasibility_plan.json` with reconciled budget, seeds, and verified source metadata.
4. Overhaul `scripts/research/run_document_feasibility_colab.py` with proper `pyarrow` struct reading (`image.bytes`), `csv.DictWriter`, robust resume preserving human reviews, attempt failure handling, contact sheet generation, and strict budget limit assertions.
5. Create `scripts/research/build_document_feasibility_colab_notebook.py` to compile and serialize the notebook deterministically with SHA-256 integrity hashes, eliminating manual copy-paste errors.
6. Create `scripts/research/preflight_cord_sources.py` to download and cache the validation shard within budget, verify all 4 sources (bytes hash, RGB hash, EXIF, dimensions, annotations, bbox overlap), and emit `cord_source_preflight_receipt.json`.
7. Expand unit tests in `ml/tests/test_document_feasibility_geometry.py` to cover all 8 fixed defects.

**Tech Stack:** Python 3.12, PyArrow, Pillow, NumPy, PyTorch (metadata only on Local), Pytest, TypeScript / PNPM continuity.

**Spec / Directives:** User request in Phase 4C.8 trace, `AGENTS.md`, Protocol Amendment v1.1.0/v1.2.0.

## Global Constraints
- Target branch: `research/independent-cohort-acquisition`.
- Working tree must remain clean; zero leaks to `main`.
- 6 frozen models in `locked_models_registry.json` must remain 100% immutable (no refitting, no scoring).
- Sealed cohorts (TGIF N=400, Locked-Test 343 sources) must remain untouched (0 accesses).
- Zero Hugging Face tokens, zero logins, zero gated fallbacks.
- Zero checkpoints downloaded to Local CPU.
- Local CORD download capped at 260 MB (272,629,760 bytes), downloaded exactly once into local cache.

---

### Task 1: Protocol Amendment v1.2.0 (Reconcile Budget & Parquet Storage Root Cause)

**Files:**
- Create: `research/evidence/document_pilot/INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.2.0.md`

- [x] **Step 1: Write Protocol Amendment v1.2.0**
Document the root cause of the 242.08 MB validation shard (Snappy-compressed dictionary page covering all 100 images at offset 4). Formally raise the original data budget ceiling from 200 MB to 260 MB (272,629,760 bytes / 260 MiB) while keeping all other ceilings unchanged (checkpoint <= 12 GiB, dependencies <= 2 GiB, local transfer <= 100 MB, execution <= 60 mins). Reconcile seeds to `[20261012, 20261013, 20261014, 20261015]` across all artifacts.

- [x] **Step 2: Update canonical machine plan**
Modify `research/evidence/document_pilot/document_feasibility_plan.json` to reference protocol version `1.2.0`, update `original_data_colab_max_bytes` to `272629760`, and ensure seeds and bboxes match ground truth.

---

### Task 2: Fix Runner Implementation (`run_document_feasibility_colab.py`)

**Files:**
- Modify: `scripts/research/run_document_feasibility_colab.py`

- [x] **Step 1: Fix CORD Parquet loader for PyArrow struct schema**
Read `image` and `ground_truth` columns; access `table["image"][row_idx]["bytes"].as_py()`. Support local cached parquet path if available, or download with progress and single-file caching.
- [x] **Step 2: Fix CSV creation and resume logic**
Use `csv.DictWriter`. When resuming existing receipts, include them in the CSV. Read and preserve existing `human_verdict` and `human_notes` if `human_review.csv` exists.
- [x] **Step 3: Implement attempt failure handling and partial packaging**
Add `try...except` block per attempt. Write `attempt_failure_receipt.json` on error. Package partial zip even if an attempt fails.
- [x] **Step 4: Implement contact sheet generation**
Add `generate_contact_sheet()` creating `feasibility_contact_sheet.png` (5 columns: Authentic, ROI Before, ROI After, Final Edited, Intervention Mask).
- [x] **Step 5: Enforce budget limits fail-closed**
Assert shard size <= 272,629,760 bytes, checkpoint <= 12 GiB, GPU time <= 60 mins, archive <= 100 MB.

---

### Task 3: Turnkey Notebook Generator (`build_document_feasibility_colab_notebook.py`)

**Files:**
- Create: `scripts/research/build_document_feasibility_colab_notebook.py`
- Modify: `notebooks/independent_document_feasibility_colab.ipynb`

- [x] **Step 1: Write notebook generator script**
Read `document_feasibility_plan.json` and `run_document_feasibility_colab.py`. Validate their SHA-256. Construct 4-cell Jupyter notebook JSON structure. Verify all code cells compile cleanly via `compile()`.
- [x] **Step 2: Run generator to produce valid notebook**
Execute generator script, overwrite `notebooks/independent_document_feasibility_colab.ipynb`. Verify zero syntax or indentation errors.

---

### Task 4: Real CORD Data Preflight on LOCAL (`preflight_cord_sources.py`)

**Files:**
- Create: `scripts/research/preflight_cord_sources.py`
- Output: `research/evidence/document_pilot/cord_source_preflight_receipt.json`
- Output: `data/research/cord_v2/` (cached shard and verified authentic crops)

- [x] **Step 1: Implement preflight script**
Preflight shard size against 260 MB ceiling. Download shard once to `data/research/cord_v2/`. Extract rows 22, 61, 7, 93. Validate image dimensions, bytes SHA-256, RGB SHA-256, EXIF orientation, category, original text, and bounding box coverage. Generate visual ROI inspection sheets.
- [x] **Step 2: Run preflight script and verify output receipt**
Verify exit code 0 and receipt `cord_source_preflight_receipt.json`.

---

### Task 5: Unit Tests, Continuity Check & Final Verification

**Files:**
- Modify: `ml/tests/test_document_feasibility_geometry.py`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`
- Modify: `research/evidence/document_pilot/FEASIBILITY_STUDY_PREPARATION_REPORT.md`

- [x] **Step 1: Expand unit tests in `test_document_feasibility_geometry.py`**
Add tests for: (1) Notebook code cells compilation; (2) Notebook payload parity with canonical plan/runner; (3) CSV strict round-trip; (4) Resume behavior; (5) Failure handling and receipt generation; (6) Budget guard enforcement; (7) CORD source preflight receipt verification.
- [x] **Step 2: Run pytest and pnpm test**
Run pytest (`ml/.venv/Scripts/python.exe -m pytest ml/tests/test_document_feasibility_geometry.py -v`) and `pnpm test`.
- [x] **Step 3: Update continuity docs and verify `node scripts/continuity-check.mjs`**
Update continuity files and run continuity checker.
