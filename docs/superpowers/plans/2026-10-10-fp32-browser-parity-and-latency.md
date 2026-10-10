# FP32 Browser Parity and Latency Re-benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the browser FP32 pipeline to achieve verified numerical parity PASS against the Python reference on the 16-sample development panel (all 160 decisions matching, within locked tolerances), re-benchmark latency on 3 isolated browser sessions, and reconcile INT8 trade-off conclusions for RQ3-RQ4.

**Architecture:** 
1. Port exact Pillow-matching 2-pass separable resampling (Bicubic a=-0.5 for 224x224 tensor; Bilinear for 128x128 FFT) into `@forensics/forensics` and `@forensics/inference`.
2. Update `packages/forensics/src/canonical-dsp.ts` to use exact uint8 Bilinear resampling with antialiasing for FFT and bit-identical centered coordinate mapping.
3. Update `packages/inference/src/research-pipeline.ts` to use exact Bicubic 2-pass resampling for ImageNet normalized tensor generation.
4. Verify end-to-end numerical parity on live Chromium browser across all 6 layers using real outputs, independent comparison script, and generate audited receipt.
5. Re-benchmark warm inference latency on 3 isolated browser context sessions with clean percentile calculations.
6. Audit and close INT8 evidence to bounded factual statements.

**Tech Stack:** TypeScript, React, Vite, ONNX Runtime Web 1.30.0 (WASM SIMD, 1 thread), Vitest, PyTorch 2.5.1, Pillow 12.3.0, Chromium Headless / Browser Automation.

**Spec:** User prompt requirements A-G, `docs/continuity/CURRENT_STATE.md`, `research/evidence/onnx_fp32_parity/development_panel_reference_fp32.json`.

## Global Constraints
- Backbone: MobileNetV3-small frozen (`047dcff4...`, ONNX sha256 `b4b35d3a...`).
- Models: Exactly 5 frozen outer-fold models and stackers.
- Panel: Exactly 16 development samples (8 pairs) from Option P `development_train`.
- Zero tuning/fitting, zero access to TGIF N=400.
- Strict tolerances: Decision agreement = 160/160 (100%), Layer 1 tensor mean diff <= 0.005, Layer 2 visual cosine >= 0.999, Layer 3 DSP max diff <= 0.005, Layer 5 logit max diff <= 0.005, Layer 6 prob max diff <= 0.005.

---

### Task 1: Update Canonical DSP FFT Resampling in TypeScript
**Files:**
- Modify: `packages/forensics/src/canonical-dsp.ts`
- Test: `packages/forensics/src/__tests__/canonical-dsp.test.ts`

- [ ] **Step 1: Write unit tests in canonical-dsp.test.ts for exact FFT parity**
- [ ] **Step 2: Run vitest to verify test failure against old implementation**
- [ ] **Step 3: Implement 2-pass Bilinear resampling with antialiasing in canonical-dsp.ts**
- [ ] **Step 4: Run vitest to verify all tests PASS**

---

### Task 2: Update Preprocessing Bicubic Resampling in TypeScript
**Files:**
- Modify: `packages/inference/src/research-pipeline.ts`
- Test: `packages/inference/src/__tests__/research-pipeline.test.ts`

- [ ] **Step 1: Write test for exact Bicubic 2-pass resampling matching Pillow**
- [ ] **Step 2: Run vitest to verify failure against old 4x4 tap bicubic**
- [ ] **Step 3: Implement 2-pass Bicubic resampling with antialiasing in research-pipeline.ts**
- [ ] **Step 4: Run vitest to verify all tests PASS**

---

### Task 3: Build Web Worker & Frontend Bundle
**Files:**
- Build: `packages/forensics`, `packages/inference`, `apps/web`

- [ ] **Step 1: Run pnpm typecheck across monorepo**
- [ ] **Step 2: Build apps/web production bundle and ensure wasm assets are properly mapped**
- [ ] **Step 3: Verify dev server / web bundle serves updated code**

---

### Task 4: Collect Raw Browser Outputs & Verify 6-Layer Parity PASS
**Files:**
- Modify: `apps/web/src/components/ResearchLabView.tsx` (if needed to export raw outputs)
- Create: `scripts/research/audit_browser_fp32_parity.py`
- Output: `research/evidence/browser_fp32_parity/browser_fp32_parity_audited_receipt.json`
- Output raw: `research/evidence/browser_fp32_parity/browser_raw_outputs.json`

- [ ] **Step 1: Run full parity suite on real Chromium browser via browser_subagent**
- [ ] **Step 2: Export raw browser outputs (tensors, 576-d visual, 16-d DSP, logits, probs, predictions)**
- [ ] **Step 3: Run independent Python audit script to compute layer-by-layer element-wise diffs**
- [ ] **Step 4: Verify 160/160 decisions match and all layers pass pre-registered tolerances**

---

### Task 5: Re-benchmark 3 Isolated Browser Sessions
**Files:**
- Output: `research/evidence/browser_fp32_parity/browser_benchmark_3isolated_sessions_receipt.json`

- [ ] **Step 1: Run 3 independent, isolated browser sessions on updated pipeline**
- [ ] **Step 2: Record cold initialization separately from warm inference**
- [ ] **Step 3: Compute mean, median P50, and P95 using standard consistent percentile formula**
- [ ] **Step 4: Save audited benchmark receipt**

---

### Task 6: Audit INT8 Section & Document Findings
**Files:**
- Review: `research/evidence/onnx_int8_tradeoff/int8_tradeoff_receipt.json`
- Modify: `docs/continuity/CURRENT_STATE.md`, `docs/continuity/STATUS_LEDGER.md`, `docs/continuity/CODE_INDEX.md`

- [ ] **Step 1: Audit INT8 trade-off statements and remove unproven assertions**
- [ ] **Step 2: Update continuity docs (CURRENT_STATE, STATUS_LEDGER, CODE_INDEX)**
- [ ] **Step 3: Run pnpm continuity:check and git diff --check**
- [ ] **Step 4: Formulate final commit and report answering RQ3-RQ4 and web packaging readiness**
