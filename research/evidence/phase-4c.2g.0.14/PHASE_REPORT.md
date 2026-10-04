# Phase 4C.2G.0.14 — Confirmatory Closure and Exploratory Error Analysis

## Closure verdict

`PHASE_4C2G_CLOSED_INSUFFICIENT_CONFIRMATORY_EVIDENCE`

The Phase 4C.2G experimental result is unchanged and final: arithmetic mean Macro-F1 `0.513835085980773`, preregistered source-cluster percentile 95% CI `[0.49976282750638307, 0.5273360991798881]`, and confirmatory verdict `INSUFFICIENT_CONFIRMATORY_EVIDENCE` because the lower bound is not strictly greater than `0.5000`.

This closure performed no inference, training, threshold tuning, calibration fitting, model selection, or bootstrap rerun. It read only the already-produced evaluator outputs.

## Artifact reconciliation

- Both checksum indices pass: 20/20 evaluator-output entries and 22/22 session entries.
- Five prediction files each contain exactly 686 records for the same ordered 343 paired sources.
- Macro-F1, balanced accuracy, AUROC, Brier, ECE, CITL, signed confidence gap, and confusion matrices were recomputed from the existing logits and match the published per-checkpoint artifacts.
- Mean Macro-F1 was recomputed exactly. The stored bootstrap summary and aggregate artifact agree; the bootstrap was not rerun.
- All input artifact byte counts and SHA-256 digests remained unchanged during analysis.

The outer wrapper exit `1` is confirmed as a false negative. The sealed controller exited `0`, the execution receipt is `COMPLETED_VALID`, and the complete scientific artifacts exist. The wrapper checked for a non-contract file named `confirmatory_result.json`; the controller correctly published `aggregate_confirmatory_metrics.json` and `confirmatory_decision.json`. No scientific quantity is affected and no retry occurred.

## Exploratory confusion analysis

Everything in this and the following sections is `EXPLORATORY_POST_HOC_LOCKED_TEST_ARTIFACT_ANALYSIS_NOT_CONFIRMATORY`.

| Seed | TN | FP | FN | TP | Authentic FPR | Edited FNR |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 151 | 192 | 132 | 211 | 0.5598 | 0.3848 |
| 1337 | 84 | 259 | 78 | 265 | 0.7551 | 0.2274 |
| 2025 | 185 | 158 | 164 | 179 | 0.4606 | 0.4781 |
| 3407 | 121 | 222 | 106 | 237 | 0.6472 | 0.3090 |
| 9001 | 191 | 152 | 167 | 176 | 0.4431 | 0.4869 |

Pooling the same fixed samples across checkpoints only as a non-independent description gives TN/FP/FN/TP = `732/983/647/1068`, authentic FPR `0.5732`, and edited FNR `0.3773`. The direction changes materially by seed: seed 1337 strongly favors `ai_edited`, whereas seeds 2025 and 9001 are closer to class balance. This is consistent with seed-sensitive decision bias rather than a stable operating characteristic.

## Exploratory probability analysis

- Mean `P(ai_edited)` pooled over the repeated sample-checkpoint rows is `0.506081` for authentic and `0.511111` for edited—a difference of only `0.005030`.
- Per-checkpoint mean separation ranges from `0.003566` to `0.006286`.
- `82.9738%` of probabilities lie in `[0.45, 0.55)` and `99.1545%` lie in `[0.40, 0.60)`.
- Mean prediction confidence is `0.530110` for correct decisions and `0.527958` for errors, so confidence weakly distinguishes correct from incorrect predictions.

The post-hoc pattern is extensive class-conditional overlap and score compression near 0.5. Threshold movement could exchange false positives for false negatives but cannot by itself create discrimination; this is a hypothesis for development experiments, not a confirmatory conclusion.

## Exploratory source-level errors

Source identifiers, sample identifiers, and paths are not published. Each source contributes its authentic/edited pair across five checkpoints, or ten decisions.

- `339/343` sources (`98.8338%`) have at least one error; only 4 are correct on all ten decisions.
- No source is wrong on all ten; the maximum is 9 errors, observed for 4 anonymous sources.
- `215/343` sources (`62.6822%`) have exactly 5/10 errors, consistent with many paired sources having one class repeatedly favored over the other.
- `189/343` sources (`55.1020%`) show at least one sample whose predicted class changes across seeds.
- The authentic member is falsely positive in all five checkpoints for 109 sources (`31.7784%`); the edited member is falsely negative in all five for 60 sources (`17.4927%`).

These source-level summaries are exploratory and must not be used to mine or tune against the completed locked test.

## Next phase boundary

The proposed next work is a separately preregistered development-only comparison using grouped nested cross-validation, followed—only if a prospectively defined development improvement gate is met—by a newly acquired, source-disjoint, independently sealed validation cohort. See `NEXT_EXPERIMENT_AND_INDEPENDENT_VALIDATION_PLAN.md`.

The completed Phase 4C.2G locked test is retired from model development and cannot be rerun or reinterpreted as confirmatory success.
