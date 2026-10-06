# Phase 4C.5A — Fusion Shift Diagnostics: Contract, Implementation and Synthetic Verification

> **Branch**: `claude/elegant-edison-uentky` (cloud-assigned; from `origin/main` `3a0292c`, CI run #30 success)<br>
> **Status**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`<br>
> **Real diagnostics / findings**: `0 / NOT_MEASURED`<br>
> **Fits, recalibration, tuning, threshold changes, feature re-extraction, backbone inference**: 0 by design<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access; the tool has no dataset or manifest input)<br>
> **Handoff**: `HANDOFF.md`

---

## 1. Question

Phase 4C.4B measured that `late_fusion_stacked` Macro-F1 falls from 0.6025974305 (original) to
0.4401056539 (JPEG q75) and 0.4026117354 (JPEG q50). Under JPEG its FNR rises sharply, and under
resize its FPR rises sharply. `visual_calibrated` barely moves. 4C.4B called the DSP shift a
hypothesis only. This phase builds a post-hoc diagnostic tool that tells technical defects (wrong
bindings, ordering, scaling, reconstruction) apart from model sensitivity to the transforms. It does
not assume DSP is the cause.

## 2. Contract read from code (not from file names)

| Item | Implementation fact (source) |
| :--- | :--- |
| 16 DSP features, order | `ml/training/dsp_features.py::DSP_FEATURE_NAMES` (FFT 4, DCT 3, noise residual 5, JPEG grid 3, Laplacian variance 1), computed on the BT.601 grey image in [0, 1]. `dct_mean_high_freq_energy`, `noise_global_level` and `noise_std_of_variances` are in grey-level units of that scale. `dct_ac_energy_variance` and `laplacian_variance` are in squared grey-level units. The rest are dimensionless ratios or scores, several of them clipped. Order cross-checked against the ablation `dsp_features_receipt.json` |
| Base scorers | per modality `StandardScaler` + `LogisticRegression`, fitted on outer-train images of both classes (`calibrated_late_fusion.run_outer_fold`); decision function in float32 |
| Temperatures | one per modality, fitted on inner-OOF logits; calibrated logit = `z / T` |
| Stacker inputs | `fold_model.json` `stacker.inputs = ["visual_calibrated_logit", "dsp_calibrated_logit"]`: **temperature-scaled logits**, not probabilities; float64 logistic regression |
| Fusion | `fusion_logit = b + a_v·z_v/T_v + a_d·z_d/T_d`, `p = sigmoid(fusion_logit)` |
| Class order, threshold, tie | `(authentic, ai_edited)`; `p ≥ 0.5 → ai_edited` (protocol `decision_threshold: 0.5`); selective τ_conf = 0.65 |
| Folds | the 4C.3B outer folds (outer-train = complement of outer-test), checked against the fold-lock commitments |

## 3. Tool (`scripts/research/diagnose_fusion_shift.py`)

- **Inputs**: the robustness `full/` scope, the 4C.3B late-fusion run and the ablation run. All are
  read-only, hash-verified, and reuse `load_full_scope`, `load_frozen_artifacts` and
  `score_with_frozen_models`. There is no dataset, image, weights or locked-test input.
- **Preflight (fail-closed)**:
  - protocol SHA, gate and receipt hashes, and binding of the frozen parameter digests to the run;
  - stacker inputs, DSP dimension, temperatures and DSP feature names;
  - identical sample order across conditions (authentic then ai_edited per source; 341/682);
  - `features.npz` keys, shapes, float32, finite values and commitments;
  - original DSP features equal to the 4C.3B cache commitment;
  - outer folds against the fold lock (test and train);
  - per-row outer fold and label equal 4C.3B;
  - stored predictions follow `p ≥ 0.5`;
  - **arithmetic reconstruction of all three recipes' stored logits and probabilities** from cached
    features (tolerance = protocol 1e-9);
  - exact decomposition (residual ≤ 1e-9);
  - per-DSP-feature split vs float64 recomputation (relative ≤ 1e-9; the float32 model-path gap is
    reported, not gated);
  - fitted DSP scaler equal to the recomputed outer-training original statistics (relative 1e-9),
    with StandardScaler's own constant-feature rule.
- **Analyses**: DSP feature shifts, component shifts, the fusion decomposition with per-feature
  DSP terms, decision transitions, confidence and high-confidence errors, and Macro-F1 per condition
  for the cross-check. Every table lists outer folds 0–4 before the pooled `all` row, by class.
- **Figures** (SVG, hand-rolled like the existing analyzers; matplotlib is not a dependency): DSP
  shift heatmap, stacker inputs/output shift, decomposition, decision transitions, coverage and
  errors. Each figure has axes with units, a legend, n, and a `SYNTHETIC — NOT REAL PERFORMANCE`
  banner for synthetic runs. The categorical palette passed the dataviz validator. Its contrast warning
  is relieved by in-bar labels, CSV tables and the auto-report. Rendered and inspected; no label
  collisions found.
- **Modes**: `preflight`, `analyze` (resume = hash-verified reuse, no rewrite; different bindings
  or tampering are refused), `verify` (byte-for-byte rebuild). Synthetic output under
  `research/evidence/` is refused; sample-level rows never go to the evidence copy.

## 4. Robustness notebook fix

`notebooks/development_robustness_colab.ipynb` pointed at the deleted branch `claude/keen-knuth-4esvaw`
and silently skipped the commit check when `EXPECTED_COMMIT` was empty. It now:
- requires a full 40-hex `EXPECTED_COMMIT` (pinned to `3a0292c840d44ca255acfd6588c3a16f6d61bdd2`,
  the main merge that contains the executed 4C.4B code);
- clones without a branch name and checks out that commit detached;
- refuses a checkout with local changes to tracked files;
- asserts `HEAD == EXPECTED_COMMIT`.

A test runs the cell against a local clone in four states: fresh clone, interrupted first run,
rerun, and a moved checkout. It also covers the empty SHA and local edits. Outputs are not stored,
and robustness was not re-run.

## 5. Verification (cloud, synthetic only)

See `test_summary.json`. 35 diagnostics tests cover:
- exact decomposition on hand-built models and on the synthetic 4C.3B world;
- DSP-only and visual-only input changes;
- the identity condition (Δ = 0);
- normalisation using only outer-training originals, with zero-scale handling;
- shuffled feature rows, DSP columns, sample order and fold assignment;
- tampered DSP feature names, NaN/Inf features, and tampered or missing artifacts;
- no `fit`/`fit_transform`/temperature fit/backbone forward/DSP extraction, and no file opened under
  the image bundle;
- resume without rewrite, verify mode, refusal of tampered or rebound outputs;
- refusal of synthetic output under research/evidence;
- axis ticks, and Macro-F1 parity with the robustness analyzer.

Each key guard was mutation-tested: disabling it makes its test fail.

Code review with the `code-review` skill gave 10 findings, and 9 were fixed with regression tests:
- notebook stuck after an interrupted clone;
- DSP attribution gate sensitive to float32 rounding;
- zero-scale rule different from StandardScaler;
- hard-coded 341/5/0.5 in text;
- NaN propagating into pooled standardized shifts;
- duplicated atomic writer;
- fold model re-read;
- repeated min/max computation;
- missing continuity docs.

One finding was declined: the second scoring pass is a deliberate bit-exact cross-check of the
components against the runner path, and it also re-verifies the early-fusion predictions.

## 6. Status of scientific claims

- No development_real diagnostic has been run. Every real finding is `NOT_MEASURED`.
- Synthetic fixture numbers are not research results and are not committed as evidence.
- Even after the real run, the decomposition is algebraic, not causal. Findings will be post-hoc
  exploratory on the same 341 development sources. They are not confirmatory and not a product
  claim. The web UI stays `uncertain` / `Model not installed`.

## 7. Next approved action

The local agent runs `HANDOFF.md` on this branch and records Phase 4C.5B with measured post-hoc
findings and gates. After branch CI passes, it merge-commits into `main`, verifies main CI and
ancestry, and deletes the fully merged branch.
