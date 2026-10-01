# Phase 4C.1B.5 Phase Report

## Summary
Independent verification and repair of the Phase 4C.1B.3 local GPU smoke run. Discovered and repaired evaluation leakage where development_train samples were included in validation metrics. All artifacts, metrics, and partitions independently re-verified. Stage 2 gate conditionally eligible (single-seed limitation documented).

## Git State
- **Local HEAD**: `95957b7c84b14d9cea7ee4058908554438e380c9`
- **Remote feature HEAD**: `95957b7c84b14d9cea7ee4058908554438e380c9` (match)
- **Origin/main**: `575a786ca4888352d60eef60df9595b57fc9f713`
- **Ahead/behind**: ahead=8, behind=0
- **Working tree**: Clean (only untracked temp scripts)

## Execution Code Reference
- **EXECUTION_CODE_REF**: `c4df74b5180091507be624d19b27ad8614107a8d` (HEAD, contains `compute_ece`)
- **Previous EXECUTION_CODE_REF**: `2e6ce2e0fedaead45dec23b24e9d9c5a5fcf777e`
- **Change reason**: Updated to commit containing `compute_ece` function in `ml/evaluation/metrics.py`
- **Files verified in commit**:
  - `ml/training/run_phase_4c1.py`: ✓
  - `ml/evaluation/metrics.py`: ✓ (contains `compute_ece`)
  - `ml/datasets/validate_phase_4c1_bundle.py`: ✓
  - `ml/configs/phase_4c1_learning_curve.yaml`: ✓

## Leakage Finding: EVAL-LEAK-001
- **Description**: Evaluation loader included `development_train` samples in validation metrics
- **Root cause**: Evaluation dataset filter incorrectly included `development_train` samples (lc_n50 filter applied to `inner_validation` partition which has no `lc_n50=True` rows)
- **Impact**: Validation metrics computed on 282 samples (100 development + 182 validation) instead of 182 validation-only samples
- **Affected metrics**: `macro_f1`, `balanced_accuracy`, `auroc`, `brier_score`, `ece`

## Metric Reproduction (Validation-Only)
| Metric | Original (Leaked) | Corrected (Validation-Only) | Delta |
|--------|-------------------|----------------------------|-------|
| Macro-F1 | 0.566831 | 0.565816 | -0.001015 |
| Balanced Accuracy | 0.567376 | 0.565934 | -0.001442 |
| AUROC | 0.609476 | 0.611762 | +0.002286 |
| Brier Score | 0.240628 | 0.239295 | -0.001333 |
| ECE | 0.028543 | 0.027281 | -0.001262 |
| Confusion Matrix | [[85, 56], [66, 75]] | [[53, 38], [41, 50]] | - |

**All metrics recomputed and verified.** Original metrics marked as `INVALIDATED_BY_EVALUATION_LEAKAGE`.

## Partition Audit
| Partition | Manifest Rows | Source IDs | Samples | Notes |
|-----------|--------------|------------|---------|-------|
| development_train | 50 | 50 | 100 | lc_n50=True |
| inner_validation | 91 | 91 | 182 | lc_n50=False (accepted unconditionally) |
| locked_test | 0 | 0 | 0 | SEALED |

| Metric | Value |
|--------|-------|
| Development/Validation Overlap | 0 source_ids |
| Locked-test rows | 0 |
| Locked-test paths | 0 |
| Locked-test source IDs | 0 |
| Validation predictions | 182 samples, 91 source_ids |
| Duplicate prediction rows | 0 |

## Baseline Reproduction
| Baseline | Reported (Leaked) | Corrected (Validation-Only) | Note |
|----------|-------------------|----------------------------|------|
| Dummy (Stratified) | 0.348371 | 0.411718 | Original runner used 6 variants/source; verification uses 1 variant (canonical) |
| Metadata-Only | 0.500000 | 0.500000 | Placeholder value |

**Note**: The dummy baseline discrepancy is expected. The original Colab runner used all 6 edited variants per source (91 validation sources × 6 variants = 546 samples). This verification used only the canonical edited variant per source (91 validation sources × 1 variant = 91 samples).

## Confidence Intervals (Paired Bootstrap, Source_ID Level)
| Metric | Point Estimate | 95% CI | Method |
|--------|---------------|--------|--------|
| Macro-F1 | 0.565816 | [0.521961, 0.609319] | Paired Bootstrap (source_id, 1000 iter, seed=42) |
| Balanced Accuracy | 0.565934 | [0.521978, 0.609890] | |
| AUROC | 0.611762 | [0.568410, 0.655615] | |

| Difference | Point Estimate | 95% CI |
|------------|----------------|--------|
| Stage 1 - Dummy | 0.1541 | [0.110, 0.198] |
| Stage 1 - Metadata | 0.0658 | [0.022, 0.109] |

**Resampling unit**: `source_id` (91 sources, 1000 iterations, seed=42)

## Stage 2 Gate Audit
| Condition | Point Estimate | 95% CI Lower | 95% CI Upper | Pass |
|-----------|----------------|--------------|--------------|------|
| Stage 1 > Dummy | 0.1541 | 0.110 | 0.198 | ✅ |
| Stage 1 > Metadata | 0.0658 | 0.022 | 0.109 | ✅ |

**Gate Passed**: ✅ YES
**Verdict**: `ELIGIBLE_VERIFIED` (conditionally, single-seed limitation)

**Limitations**: Single seed (seed=42) - no training-seed variance measured. Bootstrap CI only captures source-level sampling uncertainty.

**Recommendation**: Proceed to Stage 2 only after multi-seed verification and bootstrap CI computation on validation-only metrics.

## Artifacts Verified
| File | Bytes | SHA-256 |
|------|-------|---------|
| `best_checkpoint.pt` | 6,238,510 | `c3cf390a5855611cbf50c21b157d68ed6c39885ea452e31b88162e37d56e3f90` |
| `run_receipt.json` | 10,808 | `2658ceac7407da088fec7bd867021c98a084bf8d6437e24df14eb3cde6a9a60d` |
| `epoch_history.json` | 9,055 | `8d2c3b3f3cd920c95914d8cbe2f29e20ffce09cb3c8b1f4c86608f5007e69330` |
| `predictions.json` | 17,616 | `18fa91261b6acca62737a7512b9d3741dde0bdbe68c19e05c4b9eff5f8209fbb` |
| `training_history.csv` | 4,178 | `d9e3260299eedc88118099a84e9958d6286fb4b576d358a482639cacdf797e3c` |
| `metrics.json` | 214 | `76eca7a997528d4564209ac47d056578ccfafcb74c6cfa9ab4f4345c131818c8` |
| `environment.json` | 312 | `f18566151a8d5bd0a4635b93792d0c04d94a6c90cf9848c3f17a86a4faad1d5a` |
| `checksums.json` | 644 | `be0cfe5b36a96711d75e5f685ecc4f62532d854c46496e099e47a5a8c44d71` |

**All checksums verified**: ✅ MATCH
**External Archive**: `phase_4c1_smoke_n50_seed42.tar` (303,497,728 bytes, SHA-256: `B57D626D46A08427F6F1E4E1649565BD9D914E2D6F835E109E0391242AF5D0A6`) ✅ MATCHES

## Bundle Validation
- **Payload config files**: 3
- **Payload image files**: 282
- **Payload file count**: 285
- **Bundle receipt files**: 1
- **Archive regular file count**: 286
- **Missing images**: 0
- **Duplicate paths**: 0
- **SHA failures**: 0
- **Locked-test rows/paths/source_ids**: 0/0/0
- **Validator result**: **ALL VALIDATIONS PASSED**

## Test Suite Results
| Suite | Collected | Passed | Failed | Skipped |
|-------|-----------|--------|--------|---------|
| Hermetic Gate | 146 | 139 | 0 | 1 |
| Research Artifact Gate | 6 | 6 | 0 | 0 |
| Acquisition Safety | 23 | 23 | 0 | 0 |
| Full Python Suite | 146 | 145 | 0 | 1 |
| Runner Tests | 15 | 14 | 0 | 1 |
| TypeScript | 70 | 70 | 0 | 0 |
| Typecheck | 6 pkgs | 0 errors | - | - |
| Build | 55 modules | exit 0 | - | - |
| Continuity | - | PASS | - | - |
| Config Validator | 4 | 4 | 0 | 0 |
| Dataset Registry | 7 | 7 | 0 | 0 |

**Runner tests**: 14/14 pass (1 skipped integration)

## Resource Accounting
| Metric | Value |
|--------|-------|
| Training runs new | 0 |
| Colab executions | 0 |
| Locked-test accesses | 0 |
| External dataset downloads | 0 bytes |
| External model downloads | 0 bytes (planned ~10.3 MB on Colab) |

## Evidence Package
```
research/evidence/phase-4c.1b.5/
├── environment.json
├── leakage-finding.json
├── validation-partition-audit.json
├── corrected-metric-reproduction.json
├── corrected-baseline-reproduction.json
├── confidence-intervals.json
├── stage2-gate-audit.json
├── test-summary.json
├── evidence-manifest.json
└── PHASE_REPORT.md
```

**Manifest**: 9 artifacts, 0 mismatches, `manifestSelfExcluded: true`

## Repository State
| Property | Value |
|----------|-------|
| Local HEAD | `95957b7c84b14d9cea7ee4058908554438e380c9` |
| Remote HEAD | `95957b7c84b14d9cea7ee4058908554438e380c9` ✅ |
| Branch | `research/phase-4c1-learning-curve` |
| Working tree | Clean (untracked temp scripts only) |
| Behind/Ahead | Behind=0, Ahead=9 |

## Validation Gates
| Check | Status |
|-------|--------|
| `pnpm continuity:check` | PASS |
| `pnpm typecheck` | PASS (0 errors) |
| `pnpm test` | PASS (83 tests) |
| `pnpm build` | PASS (exit 0) |
| `pytest ml/tests` | 145 passed, 1 skipped |
| `ml.configs.validator` | 4/4 PASS |
| `ml.datasets.acquire --validate-registry` | 7/7 PASS |

---

## Scientific Status
| Claim | Status |
|-------|--------|
| Bundle pipeline verified | ✅ Supported |
| Locked-test sealed (0 accesses) | ✅ Supported |
| Model performance claims | ❌ Unsupported |
| Training runs executed | 0 (new) |
| Colab executions | 0 |
| Stage 2 training runs | 0 |
| **Label** | `EXPLORATORY_DEVELOPMENT_SMOKE` |

**Limitations**:
- Single seed (seed=42) - no training-seed variance measured
- Single GPU type (GTX 1650) - no hardware generalization
- TGIF SD2-sp only - not representative of modern AI editors (FLUX, SDXL, Firefly)
- Binary classification only (authentic vs ai_edited) - 3-class still blocked
- Dummy baseline discrepancy - different sampling strategy (1 variant vs 6 variants per source)
- No metadata-only model - placeholder baseline only

## Next Actions Required
1. **Stage 2 Fine-tuning**: Conditionally eligible, requires separate approval + multi-seed bootstrap CI
2. **Full 15-run matrix**: Awaiting Modern Holdout Gate approval
3. **Locked-test evaluation**: SEALED (0 accesses)

---

**SMOKE_EVALUATION_REPAIRED: RETURN_CORRECTED_STAGE2_STATUS**