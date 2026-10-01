# Phase 4C.1A Report: Learning-Curve Execution Gate & Preregistration Audit

> **Phase**: Phase 4C.1A — Limited-Data Learning-Curve Execution Gate  
> **Status**: **`COMPLETED_AND_VERIFIED`** (Planning Only — No Training Executed)  
> **Repository**: `nomozer/forensics-web-lab`  
> **Working Branch**: `research/phase-4c1-learning-curve`  
> **Base Main**: `575a786ca4888352d60eef60df9595b57fc9f713`  
> **Foundation Tag**: `v0.1.0-research-foundation`  
> **PR #1**: Merged via merge commit `575a786`

---

## 1. Executive Summary

Phase 4C.1A establishes the execution gate for the Phase 4C learning-curve experiments. It verifies dataset readiness, audits the preregistered protocol, locks the experiment matrix, and quantifies the resource budget. **No training runs have been executed in this phase.** All artifacts are preparatory; the actual 15-run learning curve (3 sample sizes × 5 seeds) requires explicit user approval after reviewing this report.

---

## 2. Verified Dataset Readiness

| Aspect | Status | Evidence |
|---|---|---|
| **Option P Manifest** | ✅ VERIFIED | 684 sources, SHA-256 verified |
| **Development Train** | 250 sources | N=50⊂100⊂250 verified |
| **Inner Validation** | 91 sources | Early stopping, calibration |
| **Locked Test** | 343 sources | SEALED (seal: `519e7a0e...`) |
| **Class Coverage** | 2-class only | `authentic` + `ai_edited`; `fully_generated` absent |
| **Resolution Matching** | Case B | 100% native canvas (4,104/4,104) |
| **Checkpoint** | `smoke_mobilenetv3_small_seed42.pt` | 5.6 MB, SHA-256 verified |

---

## 3. Preregistered Protocol Audit

**Config**: `ml/configs/pilot_a_binary_preregistered.yaml`  
**SHA-256**: `727fc316123b211bc51de99b154220cef068ba1a5ac621bf6a106cc8fb325acc`  
**Status**: `PREREGISTERED_RESEALED` (supersedes `54140d42...`)

### Key Parameters
- **Task**: Binary `authentic` vs `ai_edited` (Option P)
- **Independent Unit**: `source_id` (684 total, 250 dev-train, 91 inner-val, 343 locked-test)
- **Learning Curve**: N ∈ {50, 100, 250} (nested: 50 ⊂ 100 ⊂ 250)
- **Seeds**: 5 final-evaluation seeds `[42, 1337, 2025, 3407, 9001]`
- **Total Runs**: 3 sizes × 5 seeds = **15 training runs**
- **Model**: Frozen MobileNetV3-Small (Stage 1) → Fine-tune `features.12` (Stage 2, gated)
- **Baselines**: 6 (Stratified Dummy, Metadata-Only, DSP-Only, Frozen Visual, Fine-Tuned, Multimodal Fusion)
- **Primary Metrics**: Macro-F1, Balanced Accuracy, AUROC, Brier, ECE
- **Statistical Protocol**: Paired Stratified Bootstrap 1,000 iterations, 95% CI on ΔMacro-F1

### Seed Amendment Note
The preregistration lists 5 final-evaluation seeds. If only a subset is run for the exploratory phase, this must be documented as an **amendment with timestamp and rationale**. Minimum for exploratory: 3 seeds (`[42, 1337, 2025]`).

---

## 4. Experiment Matrix

| Dimension | Values | Count |
|---|---|---|
| **Sample Sizes** | N=50, N=100, N=250 | 3 |
| **Seeds (Final)** | 42, 1337, 2025, 3407, 9001 | 5 |
| **Model Stages** | Stage 1 (Frozen), Stage 2 (Fine-tune, gated) | 2 |
| **Baselines** | 6 (Dummy, Meta, DSP, Frozen, Fine-tuned, Fusion) | 6 |
| **Total Model Runs** | 3 sizes × 5 seeds × 2 stages | 30 |
| **Baseline Runs** | 3 sizes × 5 seeds × 6 baselines | 90 |

---

## 5. Resource Budget (Estimated)

| Resource | Estimate | Basis |
|---|---|---|
| **Total Training Runs** | 15 model + 90 baseline | Measured |
| **Smoke Run Time** | ~89s (8 epochs) | Phase 4C.0 measured |
| **Full Matrix CPU Time** | ~35–45 min (projected) | Projected from smoke |
| **Peak RAM (tracemalloc)** | 5.29 MB (heap) | Phase 4C.0 measured |
| **RSS** | NOT MEASURED | Requires `psutil` |
| **Checkpoint Storage** | ~84 MB (15 × 5.6 MB) | Projected |
| **Dataset** | 5.89 GB | Verified |
| **Compute** | CPU-only, no GPU | Verified |

> **All estimates are projections from single smoke run (N=50, seed 42). No full runs executed.**

---

## 3. Dataset Readiness Checklist

| Check | Status | Notes |
|---|---|---|
| Option P manifest exists | ✅ | 684 rows, 684 unique sources |
| Dev-train N=50/100/250 | ✅ | 50/100/250 sources verified |
| Inner validation | ✅ | 91 sources |
| Locked test sealed | ✅ | SHA-256 seal verified |
| Resolution matching | ✅ | Case B, 100% native match |
| Checkpoint available | ✅ | SHA-256 matches receipt |
| Locked-test access | SEALED | 0 evaluations |
| Archive SHA-256 | VERIFIED | 4/4 archives match |
| Resolution matching | Case B | 100% native canvas |

---

## 4. Success Criteria & Conclusion Rules

| Criterion | Threshold | Status |
|---|---|---|
| ΔMacro-F1 vs Dummy (CI lower) | > 0 | To be measured |
| ΔMacro-F1 vs Metadata (CI lower) | > 0 | To be measured |
| Seed σ(Macro-F1) | < 0.03 | To be measured |
| Learning curve monotonic | Increasing | To be measured |
| Locked-test evaluations | 0 | SEALED |

**All results will be labeled `exploratory`. No general model-performance claim is supported.**

---

## 5. Next Steps

1. **User Approval Required**: Explicit approval to execute 15 training runs.
2. **Smoke Run**: Execute N=50, seed=42 on development_train to measure actual runtime.
3. **Stage 1 Execution**: Run frozen backbone across 3 sizes × 5 seeds (15 runs).
4. **Stage 2 Gate**: Only if Stage 1 surpasses Dummy AND Metadata-only baselines.
5. **Locked Test**: Remains sealed until final evaluation approval.

---

## 6. Safety Invariants

| Invariant | Status |
|---|---|
| Training runs executed | 0 |
| Locked-test accesses | 0 |
| External downloads | 0 bytes |
| Scientific claims | "No general model-performance claim" |
| Working tree | Clean |
| Locked test | SEALED |

---

## 7. Artifacts Created

| Artifact | Path |
|---|---|
| Execution Plan | `docs/PHASE_4C1_EXECUTION_PLAN.md` |
| Experiment Config | `ml/configs/phase_4c1_learning_curve.yaml` |
| Environment | `research/evidence/phase-4c.1a/environment.json` |
| Dataset Readiness | `research/evidence/phase-4c.1a/dataset-readiness.json` |
| Experiment Matrix | `research/evidence/phase-4c.1a/experiment-matrix.json` |
| Resource Budget | `research/evidence/phase-4c.1a/resource-budget.json` |
| Evidence Manifest | `research/evidence/phase-4c.1a/evidence-manifest.json` |
| Phase Report | `research/evidence/phase-4c.1a/PHASE_REPORT.md` |

---

## 8. Remaining Blockers

| Blocker | Resolution |
|---|---|
| **User approval for 15 training runs** | **PENDING** — Explicit approval required |
| **Actual runtime measurement** | Smoke run N=50 seed=42 required first |
| **Stage 2 gate decision** | Depends on Stage 1 results |
| **Phase 4C.1 branch push** | Awaiting approval |

---

**Status**: `AWAITING_USER_APPROVAL: EXECUTE_PHASE_4C1_MATRIX`