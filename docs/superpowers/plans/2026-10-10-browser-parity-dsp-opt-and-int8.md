# Implementation Plan: Browser Numerical Parity, DSP Optimization & INT8 Quantization Trade-Offs

**Date:** 2026-10-10  
**Branch:** `research/independent-cohort-acquisition`  
**Research Questions Scope:** RQ4 (Browser runtime & numerical parity), RQ3 (INT8 quantization trade-offs)  
**Strict Invariants:**  
- No changes to research questions (RQ1 3-class remains uncompleted, RQ2 TGIF remains inconclusive, RQ5 localization remains auxiliary unverified).
- No access to or fitting on TGIF N=400 locked dataset.
- Dual-track isolation: Product Track strictly isolated from Research Track.
- No model parameter refitting, no cherry-picking, no threshold shifting.

---

## Task 1: Reconcile Prior Receipt Discrepancy & Export 6-Layer Python Reference

- **Objective:** Fix the 684.6 ms vs 683.3 ms mean latency discrepancy by ensuring automatic computation from raw timings; export full intermediate reference vectors from Python for all 16 development samples across 6 layers:
  1. Input tensor (mean, min, max, L1)
  2. Visual features (576-d)
  3. DSP features (16-d canonical)
  4. Standardized features (5 outer folds)
  5. Logits (raw, calibrated, fusion across 5 outer folds)
  6. Probabilities & predictions (across 5 outer folds)
- **Files Modified/Created:**
  - `scripts/research/export_fp32_layer_reference.py`
  - `research/evidence/onnx_fp32_parity/development_panel_reference_fp32.json`
- **Verification:** Run script, verify SHA-256 and shape/finiteness of all 16 samples.

---

## Task 2: Grounded DSP Profiling & Limited Optimization (Max 2 Changes)

- **Objective:** Profile individual DSP components and apply at most 2 grounded algorithmic optimizations preserving 100% mathematical parity.
- **Profile Baseline (Observed):**
  - FFT (128x128): 1.68 ms
  - DCT (8x8 AC energy): 203.69 ms (97.1% of DSP time, due to 11.2M redundant `Math.cos` calls per 512x512 image)
  - Noise Residual: 1.87 ms
  - JPEG Grid: 1.02 ms
  - Laplacian Variance: 1.52 ms
- **Optimization 1 (Grounded):** Pre-calculate static 8x8 DCT cosine projection tables (`COS_TABLE[u][x] * COS_TABLE[v][y]`) or separable 1D-then-1D DCT transform to eliminate millions of redundant `Math.cos` calls per image.
- **Optimization 2 (Grounded):** Reuse pre-allocated Float64Array / Float32Array buffers for image processing passes to eliminate heap GC pressure during repeated inferences.
- **Files Modified:**
  - `packages/forensics/src/canonical-dsp.ts`
  - `packages/forensics/src/__tests__/canonical-dsp.test.ts`
- **Verification:** Run vitest unit tests (assert parity within numerical epsilon <= 1e-5), measure latency before vs after.

---

## Task 3: 6-Layer Browser Numerical Parity Verification on Chromium

- **Objective:** Run browser verification comparing Web Worker outputs against `development_panel_reference_fp32.json` across all 6 layers on real Chromium browser, checking finite values, bounds, and maximum discrepancies.
- **Files Modified/Created:**
  - `apps/web/src/components/ResearchLabView.tsx` (add layer-by-layer parity audit logic)
  - `research/evidence/browser_fp32_parity/browser_fp32_layer_parity_receipt_<run_id>.json`
- **Verification:** Execute via automated browser subagent on `http://localhost:5173/`, verify 16/16 samples pass per-layer parity gates.

---

## Task 4: Multi-Session In-Browser FP32 Benchmark (RQ4)

- **Objective:** Execute 3 separate benchmark sessions with locked configuration (machine, OS, browser, warm-up, sample count), recording separated quantiles (P50, P95, mean) for Cold Start, Preprocessing, Backbone, DSP, Scoring, and End-to-End.
- **Target Evaluation:** Check whether P95 meets or misses the prospective 500 ms target; document honestly without moving the goalposts.
- **Memory & Network Measurement:** Document `performance.memory` heap, WASM linear memory, sandbox limits, and verify Zero-Egress via network inspection.
- **Files Created:**
  - `research/evidence/browser_fp32_parity/browser_benchmark_3runs_receipt.json`

---

## Task 5: ONNX INT8 Quantization & Trade-Off Evaluation (RQ3)

- **Objective:** Quantize the MobileNetV3-small FP32 backbone to INT8 using ONNX Runtime quantization tools with calibration data drawn strictly from `development_train` (zero access to TGIF N=400).
- **Files Created/Modified:**
  - `scripts/research/quantize_onnx_backbone_int8.py`
  - `models/research/onnx/mobilenet_v3_small_backbone_int8.onnx`
  - `scripts/research/evaluate_int8_tradeoffs.py`
  - `research/evidence/onnx_int8_tradeoff/int8_tradeoff_receipt.json`
- **Metrics Evaluated:**
  - File size reduction (FP32 vs INT8)
  - Feature discrepancy (max abs diff, cosine similarity, MSE)
  - Logit / probability discrepancy and decision flip rate across 5 outer folds
  - Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE on development cohort
  - Inference latency comparison on CPU/WASM

---

## Task 6: Build Isolation Audit, Continuity Check & Final Handover

- **Objective:**
  - Audit `apps/web` production build to ensure research weights/samples are not bundled into production assets.
  - Run all repository verification checks (`git diff --check`, `pnpm -r typecheck`, `pnpm test`, `pytest`, `pnpm continuity:check`).
  - Update `CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`.
  - Git commit and push to `research/independent-cohort-acquisition`.
