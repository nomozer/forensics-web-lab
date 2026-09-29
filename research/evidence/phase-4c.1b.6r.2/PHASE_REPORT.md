# Phase 4C.1B.6R.2 - Measured Paired Bootstrap Closure

## Phase Summary
All placeholder statistical evidence replaced with measured results from Stage 1 and Dummy predictions on identical 182 validation samples. Paired source-cluster bootstrap (1000 iterations, seed=42) computed. Delta Macro-F1 = 0.07186, CI [-0.0115, 0.1508] includes 0 → condition NOT met. Metadata baseline NOT_MEASURED. Stage 2 gate: INSUFFICIENT_EVIDENCE.

## Starting Commit
1e71aa2 (Phase 4C.1B.6R.1 ending)

## Ending Commit
TBD (after validation)

## Branch
research/phase-4c1-learning-curve

## Key Changes
1. **Stage 1 Validation-Only Artifact**: Created from run predictions, filtered to 91 inner_validation source_ids, 182 samples (91 authentic, 91 ai_edited). Metrics verified: Macro-F1 0.565816095425034, Balanced Acc 0.5659340659340659, AUROC 0.6117618645091173.
2. **Dummy Prediction Verification**: 182 records, 91 source_ids, 100% sample_id/source_id/y_true match with Stage 1. Macro-F1 exactly 0.49395551257253384 (tolerance 1e-12).
3. **Paired Source-Cluster Bootstrap**: 91 clusters, 2 samples each, 1000 iterations, seed=42. Delta Macro-F1 point=0.07186058285250016, CI [-0.011494, 0.150825]. Delta Balanced Acc CI [-0.010989, 0.148489]. Delta AUROC CI [0.035849, 0.200495] (lower > 0).
4. **Stage 1 vs Dummy Condition**: CI lower bound NOT > 0 for Macro-F1 and Balanced Acc → superiority NOT statistically verified.
5. **Metadata Baseline**: NOT_MEASURED, gate_participation=false, label=HISTORICAL_PLACEHOLDER_EXCLUDED_FROM_GATE.
6. **Stage 2 Gate**: INSUFFICIENT_EVIDENCE (paired CI includes 0, metadata NOT_MEASURED, single-seed).
7. **Fault Injection**: EVAL-LEAK-001 reproduced (141 source_ids/282 samples buggy vs 91/182 correct).
8. **8 Regression Tests Added**: Fault injection, validation-only, key matching, bootstrap integrity, delta identity, CI rejection, metadata NOT_EVALUATED, dummy reproduction.
9. **All Gates Pass**: 153 collected, 152 passed, 1 skipped. TypeScript 70/70, typecheck 0, build OK, continuity PASS, configs 4/4, registry 7/7.

## Evidence Files
- environment.json
- stage1-prediction-receipt.json
- predictions-stage1-seed42-validation-only.json
- dummy-prediction-verification.json
- predictions-dummy-seed42.json (copied from 6r.1)
- dummy-baseline-metrics.json (copied from 6r.1)
- paired-bootstrap-distribution-summary.json
- confidence-intervals.json
- fault-injection-results.json
- regression-test-summary.json
- metadata-baseline-status.json
- stage2-gate-status.json
- pytest-collection.txt
- pytest-full-output.txt
- test-summary.json
- PHASE_REPORT.md
- evidence-manifest.json

## Test Results
- TypeScript: 70/70 tests passing
- Python: 153 collected, 152 passed, 1 skipped (8 new regression tests included)
- TypeScript typecheck: 0 errors
- Build: SUCCESS
- Continuity check: PASS
- Config validator: 4/4
- Dataset registry: 7/7

## Next Steps
Gate A.2 validation complete → Commit → Gate B: VS Code Colab T4 multi-seed execution (seeds 1337, 2025, 3407, 9001)