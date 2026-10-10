# Limited-Data Learning-Curve Execution Plan

> **Phase**: Phase 4C.1 — Binary Learning Curve on Option P (TGIF SD2-sp)  
> **Status**: **`PLANNING_COMPLETE — AWAITING_EXECUTION_APPROVAL`**  
> **Preregistration**: `ml/configs/pilot_a_binary_preregistered.yaml` (hash `727fc316...`)  
> **Foundation**: `v0.1.0-research-foundation` (merge commit `575a786`)  
> **Branch**: `research/phase-4c1-learning-curve`

---

## 1. Objective

Execute the preregistered learning-curve experiment for Pilot A (Binary: `authentic` vs `ai_edited`) on the frozen Option P dataset, evaluating sample efficiency across N ∈ {50, 100, 250} with 5 seeds, using Frozen MobileNetV3-Small (Stage 1) and gated Fine-Tuning (Stage 2).

---

## 2. Scope & Constraints

| Constraint | Value |
|---|---|
| **Dataset** | TGIF Option P (SD2-sp, 684 sources) |
| **Task** | Binary: `authentic` vs `ai_edited` |
| **Locked Test** | SEALED (343 sources, 0 evaluations) |
| **Evaluation Partition** | `inner_validation` (91 sources) |
| **Independent Unit** | `source_id` |
| **Model** | MobileNetV3-Small (ImageNet-1k pretrained) |
| **Training Runs** | 15 (3 sizes × 5 seeds) |
| **Baselines** | 6 (Dummy, Meta, DSP, Frozen, Fine-tuned, Fusion) |
| **Locked Test Access** | 0 (SEALED) |

---

## 3. Execution Sequence

### Step 0: Pre-Execution Verification
- [ ] Verify all 4 archives present and SHA-256 match
- [ ] Verify manifest: 684 sources, correct partitions
- [ ] Verify checkpoint `smoke_mobilenetv3_small_seed42.pt` SHA-256 matches receipt
- [ ] Verify locked-test seal unchanged
- [ ] Run hermetic CI gate: `pytest -m "not requires_research_artifact"`

### Step 1: Smoke Run (N=50, seed=42)
- **Purpose**: Measure actual epoch time, memory, throughput
- **Config**: Stage 1 (frozen backbone), N=50, seed 42
- **Output**: Actual epoch duration, peak RAM, DataLoader throughput
- **Gate**: Proceed only if smoke run completes without errors

### Step 2: Stage 1 — Frozen Backbone (15 runs)
For each N ∈ {50, 100, 250} and seed ∈ {42, 1337, 2025, 3407, 9001}:
- Train frozen backbone + linear head on `development_train[N]`
- Validate on `inner_validation`
- Select checkpoint by `inner_val_macro_f1`
- Log: epoch time, loss, metrics, checkpoint path

### Step 2b: Baseline Evaluation (90 runs)
For each N ∈ {50, 100, 250} and seed ∈ {42, 1337, 2025, 3407, 9001}:
- Run all 6 baselines on same `development_train[N]` and `inner_validation`
- Log: Macro-F1, Balanced Accuracy, AUROC, Brier, ECE

### Step 3: Stage 2 Gate (Optional)
For each N ∈ {50, 100, 250}:
- **Condition**: Stage 1 best Macro-F1 > Dummy AND > Metadata-Only on `inner_validation`
- If PASS: Run Stage 2 fine-tuning (unfreeze `features.12`, lr=5e-5, 15 epochs)
- If FAIL: Skip Stage 2, document rationale

### Step 4: Statistical Analysis
- Aggregate predictions by `source_id` (mean probability)
- Compute Macro-F1, Balanced Accuracy, AUROC, Brier, ECE per run
- Paired Stratified Bootstrap (1,000 iterations, resample by `source_id`)
- 95% CI for ΔMacro-F1 vs Dummy and vs Metadata-Only
- Report mean ± std across 5 seeds
- Check monotonic learning curve trend

### Step 5: Locked-Test Evaluation (Future, Separate Approval)
- Requires explicit approval
- Create `ExperimentLockBinding` for final evaluator role
- Run frozen checkpoints on `locked_test` (343 sources)
- Seal results

---

## 4. Resource Plan

| Resource | Plan |
|---|---|
| **Compute** | Local CPU only (no GPU) |
| **Smoke Run First** | Mandatory before full matrix |
| **Parallelism** | Sequential runs (single GPU/CPU) |
| **Disk** | ~7 GB total (dataset + checkpoints + logs) |
| **Memory** | Monitor RSS via `psutil` (add to training script) |

---

## 5. Gating Rules

| Gate | Condition | Action |
|---|---|---|
| **Smoke Run** | Completes without error | Proceed to full matrix |
| **Stage 2** | Stage 1 > Dummy AND > Metadata-Only on `inner_validation` | Run fine-tuning; else skip |
| **Locked Test** | Explicit user approval + `ExperimentLockBinding` | Execute final evaluation |

---

## 6. Scientific Reporting Requirements

All results must be reported as:
- **Exploratory** (no confirmatory claims)
- Per-seed results + mean ± std across 5 seeds
- 95% CI via Paired Stratified Bootstrap (1,000 iterations, `source_id` unit)
- ΔMacro-F1 vs Dummy: CI lower bound > 0 required for superiority claim
- ΔMacro-F1 vs Metadata: CI lower bound > 0 required
- Seed σ(Macro-F1) < 0.03
- Learning curve monotonic trend required
- All results labeled `exploratory` and `development-only`

---

## 7. Locked-Test Protection

| Rule | Enforcement |
|---|---|
| `LockedTestAccessGuard` | Blocks `TRAINING_LOADER` and `DEVELOPMENT_EVALUATOR` roles |
| `ExperimentLockBinding` | Required for `FINAL_EVALUATOR` role |
| Seal | `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9` |
| Evaluations | **0** (must remain 0 until final approval) |

---

## 8. Approval Checklist

Before execution, the following must be approved:

- [ ] **15 training runs** (Stage 1)
- [ ] **90 baseline runs**
- [ ] **Smoke run first** (N=50, seed=42)
- [ ] **Stage 2 conditional** (gated by Stage 1 results)
- [ ] **Locked test remains sealed** (0 evaluations)
- [ ] **No external downloads** during execution
- [ ] **Results labeled exploratory**

---

## 8. Config Files

| File | Purpose |
|---|---|
| `ml/configs/pilot_a_binary_preregistered.yaml` | Preregistered protocol (authoritative) |
| `ml/configs/phase_4c1_learning_curve.yaml` | Execution config (this phase) |
| `ml/configs/validator.py` | Validates config against schema |

---

## 9. Evidence Locations

| Artifact | Path |
|---|---|
| Phase 4C.1A Report | `research/evidence/phase-4c.1a/PHASE_REPORT.md` |
| Experiment Matrix | `research/evidence/phase-4c.1a/experiment-matrix.json` |
| Resource Budget | `research/evidence/phase-4c.1a/resource-budget.json` |
| Dataset Readiness | `research/evidence/phase-4c.1a/dataset-readiness.json` |
| Preregistration | `ml/configs/pilot_a_binary_preregistered.yaml` |
| Execution Config | `ml/configs/phase_4c1_learning_curve.yaml` |

---

## 10. Approval Required

> **This plan requires explicit user approval before any training run is executed.**

All artifacts are preparatory. The 15 training runs + 90 baselines will only begin after explicit confirmation.
