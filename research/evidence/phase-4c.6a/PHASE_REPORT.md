# Phase 4C.6A — Controlled DSP-Augmentation Experiment: Protocol Lock, Implementation and Synthetic Verification

> **Branch**: `claude/elegant-edison-uentky`. This branch name is assigned by the cloud session. The previous use of this name was merged into main at `9ed4ea6` and deleted; the branch was restarted from `origin/main` `9ed4ea6`, without reusing its history.<br>
> **Status**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`<br>
> **Real performance**: `NOT_MEASURED` (no development_real run in the cloud; artifacts are outside Git)<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access)<br>
> **Evidence class**: development / exploratory. The data have been studied many times; this is not confirmatory and not an independent validation.<br>
> **Handoff**: `HANDOFF.md`, `EXECUTION_GUIDE.md`

---

## 1. Question

Does training the DSP branch of the 4C.3B late fusion on JPEG/resize variants reduce the late-fusion
degradation measured in 4C.4B (Macro-F1 0.6026 → 0.4401 at JPEG q75)? And what does it cost on
original images? Only one thing changes: the DSP training policy.

## 2. Wording normalisation of Phase 4C.5B (numbers and raw artifacts unchanged)

`research/evidence/phase-4c.5b/PHASE_REPORT.md`, `CURRENT_STATE.md` and `STATUS_LEDGER.md` were changed as follows:

- **DSP share of the fusion-logit change.** ">99% on every condition" became the per-condition ratio of mean signed terms: 0.935 (q95), 0.992 (q75), 0.995 (q50), 1.001 (resize), 0.982 (resize→q75). Fold ranges are given, with fold 3 at q95 as low as 0.83. The ratio is labelled as a ratio of mean signed contributions, not causal attribution and not explained variance.
- **Implementation defects.** "No implementation defect exists" became "no deviation was found in the audited computation paths".
- **"Clean uncompressed training images" was removed.** Whether the originals were previously compressed is not established.
- **Coefficient roles.** The DSP logistic-regression feature coefficients $w_{d,j}$ are now separated from the two stacker coefficients $a_v, a_d$. The sign of a per-feature effect comes from $w_{d,j}$.
- **Residual types.** The two-term fusion decomposition residual ($4.44\times10^{-16}$) is now distinguished from the per-feature split, which has a float64 residual of $2.5\times10^{-16}$ and a float32/float64 gap of $1.4\times10^{-6}$ logit.
- **Visual term.** A small *mean* visual term no longer implies an invariant backbone. Per-image visual Δ reaches ±0.18 logit.
- **Figure descriptions** now match the actual marks: dot + IQR, stacked bars, dot plots.

## 3. Design (locked in `ml/configs/dsp_augmentation_protocol.yaml` before any real fit)

| Item | Locked choice |
| :--- | :--- |
| Scope | 341 development sources / 682 images; the 4C.3B 5 outer × 4 inner source-grouped folds (fold lock checked) |
| Recipes | A `visual_calibrated` (historical), B `late_fusion_original` (historical 4C.3B `late_fusion_stacked`), C `late_fusion_dsp_augmented` |
| Features | MobileNetV3-Small frozen 576-d + 16 DSP features, same definitions and order. **No new extraction.** Originals come from the 4C.3B ablation caches; transformed variants come from the 4C.4B robustness condition caches, which used the locked 4C.4 transforms with per-image extraction. All are bound by hash in the protocol |
| DSP augmentation | Every training image enters as original, jpeg_q95, jpeg_q75 and resize_0.5, each with sample weight 0.25 (total weight 1, as in the baseline). Variants are not treated as sources. jpeg_q50 and resize_0.5→jpeg_q75 are stress-only |
| Scaler | StandardScaler fitted on the **original** images of each fit's own training rows and applied to every variant. This matches the baseline policy |
| C selection | 4C.3B grid and procedure. Inner fits train on augmented inner-train; they are scored on **original** inner-validation images by mean Macro-F1, and the first strictly greater value wins |
| Calibration and stacker | DSP temperature fitted on original inner-OOF augmented-DSP logits. Two-input stacker on `[visual inner-OOF / T_v, augmented-DSP inner-OOF / T_d]`; inputs are calibrated logits, not probabilities |
| Visual branch | Unchanged. The 4C.3B visual scorer, visual temperature and visual inner-OOF scores are taken from the baseline reconstruction |
| Decisions | `p ≥ 0.5 → ai_edited`; τ_conf = 0.65 (descriptive only). No gating, pruning, fine-tuning, new classifier or threshold tuning |
| Prediction input | One image's own visual + DSP features under one condition. No paired image, source_id, filename, condition label or label |

**Implementation check against the historical code:**
- `calibrated_late_fusion.select_c_with_inner_scores` already selects C on inner-validation Macro-F1 (first strictly greater).
- Temperatures and the stacker are already fitted only on inner-OOF scores of outer-train.
- The baseline scaler is fitted on each fit's own (original) training rows.
- No difference was found, so no deviation from the requested design was needed.
- The 4C.3B artifacts do **not** store inner-OOF scores. They are therefore regenerated by refitting the 4C.3B baseline per fold, and those fits are counted in the budget below. Outer predictions are never used as a substitute.

## 4. Budget (derived by `dsp_augmentation.planned_budget`, locked in the protocol, tested against runner receipts)

| Item | Per outer fold | Full (5 folds) | Pilot (fold 0) |
| :--- | ---: | ---: | ---: |
| Baseline reconstruction (exact 4C.3B fold: 2 × 28 inner + 2 refits + 2 temperatures + 1 stacker) | 61 | 305 | 61 |
| Augmented DSP inner fits (4 inner × 7 C; each on 4 weighted variants) | 28 | 140 | 28 |
| Augmented DSP outer refit | 1 | 5 | 1 |
| Augmented DSP temperature optimisation | 1 | 5 | 1 |
| Augmented stacker fit | 1 | 5 | 1 |
| New visual-branch fits | 0 | 0 | 0 |
| **Total fits** | **92** | **460** | **92** |
| Feature extraction / image reads / backbone forward passes | 0 | 0 | 0 |
| Predictions (3 recipes × 6 conditions × images) | — | 12,276 | 2,484 (fold 0: 69 sources) |

## 5. Gates

1. **Preflight (0 fits)** checks:
   - the protocol and its predecessor SHAs;
   - the manifest SHA, development rows only, with no image reads;
   - the 4C.3B late-fusion and ablation fold artifacts, the fold lock, and the cache commitments;
   - the 4C.4B robustness full scope: run bindings, gate and the six condition-receipt SHAs bound in the protocol;
   - the condition features: order, shape, dtype, finite values, commitments, and original == cache.

   It then rebuilds A and B from the stored fold models and must reproduce the stored 4C.3B predictions (original) and the 4C.4B predictions (all 6 conditions) within 1e-9, with 0 decision mismatches. A missing artifact stops the run before any fit.
2. **Per-fold baseline gate**: the refitted 4C.3B fold must equal the stored `fold_model.json` exactly and reproduce the 4C.3B/4C.4B predictions (≤ 1e-9, 0 mismatches) before the augmented fold is persisted.
3. **Budget gate**: the run's fit count must equal the locked plan; the analyzer refuses a run that differs.
4. **Analysis verification**: `--verify` rebuilds every output and compares bytes.

## 6. Endpoints and reporting (locked)

- **Primary**: ΔMacro-F1 at jpeg_q75, `late_fusion_dsp_augmented − late_fusion_original`. Paired source-cluster bootstrap, 10,000 replicates, `numpy.random.Generator(PCG64(20261006))`, percentile 95%. The same source draw is used for both recipes, with both images of each drawn source.
- **Verdict vocabulary**:
  - `EXPLORATORY_JPEG75_IMPROVEMENT` (CI entirely > 0);
  - `EXPLORATORY_JPEG75_DEGRADATION` (entirely < 0);
  - `EXPLORATORY_JPEG75_UNRESOLVED` (contains 0);
  - `SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT` (synthetic runs).
- **Secondary (exploratory)**:
  - the same contrast at original and at every stress condition, plus augmented − visual;
  - Balanced Accuracy, AUROC, Brier, log-loss, ECE, FPR/FNR, coverage, selective accuracy and high-confidence error rate at τ = 0.65;
  - visual/DSP term shifts per fold and class;
  - the number of finite bootstrap replicates for every interval.
- **Reporting rules**:
  - original performance and every stress condition are reported whatever the direction;
  - a CI containing 0 is not equivalence, and a lower Brier is not proof of better calibration;
  - conditions and folds are not independent experiments;
  - the bootstrap holds models and folds fixed, so it does not cover training variability or development-reuse selection bias;
  - jpeg_q75 is itself an augmentation variant;
  - an improvement does not make the recipe production-eligible.

## 7. Implementation

| File | Role |
| :--- | :--- |
| `ml/configs/dsp_augmentation_protocol.yaml` | locked protocol (design, bindings, gates, endpoints, budget) |
| `ml/training/dsp_augmentation.py` | protocol validation and budget derivation; weighted augmented fit; augmented C selection; per-fold experiment; single-image scoring of the 3 recipes |
| `ml/training/run_dsp_augmentation.py` | CLI `--mode preflight\|pilot\|full`; gates; hash-bound resume (reuses `calibrated_late_fusion` run manifest, fold persistence and verification) |
| `scripts/research/analyze_dsp_augmentation.py` | metrics, paired bootstrap, verdict, contribution shifts, 2 SVG figures, report, `--verify` |
| `scripts/research/build_calibrated_late_fusion_package.py` | gains `--experiment controlled_dsp_augmentation` (19 members, no weights needed) |
| `ml/training/calibrated_late_fusion.py` | `run_outer_fold` also returns its in-memory fitted objects and inner-OOF scores; persisted artifacts unchanged |
| `ml/tests/test_dsp_augmentation.py` | synthetic tests (see `test_summary.json`) |

## 8. Verification (cloud, synthetic only)

See `test_summary.json`. The tests cover:
- variants of one source stay together, and every training image has total weight 1;
- with identical variants, the augmented fit and C selection reduce to the baseline;
- the scaler is fitted on originals only, and inner validation and inner-OOF use originals only;
- the baseline reproduces the 4C.3B code path exactly, and the visual branch is unchanged;
- outer-test mutation leaves every fitted parameter unchanged;
- training variants change only the augmented DSP branch, and stress-only conditions never affect training;
- predictions use a single image;
- preflight reproduces history with 0 fits;
- pilot and full spend exactly 92 and 460 fits, resume without refitting, and reject tampered artifacts, failed gates and wrong bindings;
- missing artifacts stop the run before any fit;
- there are no image reads, no backbone calls and no locked-test file access;
- outputs inside the repository or historical inputs are refused;
- analyzer: verdict vocabulary, paired bootstrap arithmetic, finite-replicate counts, budget refusal, byte-for-byte verify, and refusal of synthetic output under research/evidence;
- the package is deterministic, its members are closed under imports, and the staged package runs preflight and the analyzer without the repository.

Key guards were mutation-tested: disabling each one fails its test. Synthetic figures were rendered and inspected, and the palette is the dataviz-validated one already used in 4C.5A.

The `code-review` skill reported 9 findings. 8 were fixed with regression tests where behaviour changed:
- the analyzer budget gate;
- finite-replicate counts;
- the resume gate check;
- per-inner-fold matrices hoisted out of the C grid;
- removal of a no-op reassignment;
- the protocol now drives the variant list;
- named decomposition fields;
- the shared JSON writer.

The 9th finding (continuity docs) is addressed by this phase's continuity updates.

## 9. Status of claims

No development_real run has happened. Every real performance number is `NOT_MEASURED`. Synthetic numbers
are fixture values only and are not committed as evidence. The web UI stays `uncertain` / `Model not installed`.

## 10. Next approved action

The local agent runs `EXECUTION_GUIDE.md` on this branch (preflight → pilot → full → analyze → verify)
and records Phase 4C.6B with the measured results and gates. After branch CI passes, it merges into
`main` with a merge commit (no PR), pushes, verifies main CI and ancestry, then deletes the branch
locally (`git branch -d`) and remotely (`git push origin --delete`).
