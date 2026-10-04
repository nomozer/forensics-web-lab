# Next Experiment and Independent Validation Plan

## Status and boundary

This is a prospective proposal, not an authorization to train, evaluate, download data, or access the completed Phase 4C.2G locked test. Phase 4C.2G is closed with `INSUFFICIENT_CONFIRMATORY_EVIDENCE`. Its 343 sources, predictions, error ranks, and labels must never be used for threshold tuning, candidate selection, hard-example mining, training, or another confirmatory claim.

The Phase 4C.2G post-hoc findings are hypothesis-generating only. They suggest weak score separation, seed-dependent class bias, and source-pair failures; they do not establish a causal mechanism.

## Development-only experiment

### Objective

Improve source-general discrimination before any new external validation. Calibration or threshold movement alone is not an adequate objective because the existing class-conditional probability distributions nearly overlap.

### Data boundary

- Use only the existing 341 non-locked development sources: 250 `development_train` plus 91 historical `inner_validation` sources.
- Preserve authentic/edited pairs and group every split by `unique_source_id`.
- Do not load Phase 4C.2G predictions during model development. The aggregate closure report may be cited only as motivation.
- Keep all research weights and data in the Research Track under ADR-0006; no promotion to the product registry.

### Prospective design

1. Freeze a grouped nested cross-validation plan before execution: five source-group outer folds, inner folds for early stopping and any hyperparameter selection, and the same five training seeds `[42, 1337, 2025, 3407, 9001]` for every candidate.
2. Retain the exact Stage 1 frozen linear probe as the control.
3. Evaluate a small, preregistered candidate set rather than an open-ended search:
   - pair-aware objective adding a within-source ordering/ranking term so the edited member scores above its authentic pair;
   - mask-guided or edit-localized crop sampling on development data only, while retaining a global-image branch;
   - one lightweight spatial-plus-frequency candidate consistent with ADR-0003, bounded to the existing browser-oriented parameter budget.
4. Keep preprocessing, augmentations, optimizer ranges, stopping rules, and candidate count fixed in a new preregistration. Any threshold/calibration fitting occurs only inside inner folds and is frozen before outer-fold scoring.
5. Primary development endpoint: outer-fold source-grouped Macro-F1 averaged over the five fixed seeds. Required advancement gate: paired improvement over the Stage 1 control of at least `+0.03` Macro-F1 and a source-cluster 95% CI for the paired delta with lower bound above `0`.
6. Guard endpoints: balanced accuracy, AUROC, authentic false-positive rate, edited false-negative rate, within-source pair-ordering accuracy, Brier score, ECE, class-conditional score separation, and between-seed variability.
7. Select at most one final training recipe using only nested-development results. Retrain that fixed recipe on the full 341-source development pool under five fixed seeds. Do not inspect any independent-validation labels during this step.

The `+0.03` development gate is a prospective practical-effect requirement for the next phase; it does not alter or reinterpret Phase 4C.2G.

## New independent validation set

### Independence requirements

- Acquire a completely new source-disjoint cohort; do not reuse Option P development or the completed Phase 4C.2G locked-test sources.
- Prefer a forward-time and cross-domain cohort containing multiple authentic capture pipelines, editing generators, edit categories, compression levels, and image resolutions.
- Enforce perceptual and provenance deduplication against every prior research split before sealing.
- Record license and derivative-weight restrictions; keep the cohort research-only unless separate product eligibility is proven.
- Use an independent custodian to create a canonical per-sample manifest commitment before evaluator authorization.

### Size and composition

- Target at least 500 unique source pairs (1,000 balanced samples) as an initial floor.
- Determine and freeze final cardinality with a prospective source-cluster simulation before acquisition is unsealed, using a minimum relevant effect and desired power stated in the new preregistration—not the observed Phase 4C.2G errors.
- Predefine domain strata and minimum stratum counts. The pooled endpoint remains primary; strata are descriptive unless separately powered and multiplicity-controlled.

### Confirmatory protocol

1. Seal the chosen recipe, five seed-specific checkpoints, preprocessing, threshold, calibration policy, manifest, evaluator, and statistical plan before data access.
2. Primary endpoint: arithmetic mean of five per-checkpoint Macro-F1 values.
3. Uncertainty: 10,000-replicate source-cluster percentile bootstrap with a newly frozen RNG seed.
4. Recommended dual success gate, preregistered before access: 95% CI lower bound strictly above `0.5000` and point estimate at least `0.5500`.
5. Secondary endpoints: balanced accuracy, AUROC, class-specific sensitivity/specificity, Brier, ECE, within-source ordering accuracy, and descriptive domain-stratum results.
6. Run exactly one authorized evaluation session, publish success or failure, and prohibit tuning or automatic retry after reservation.

## Decision sequence

```text
Phase 4C.2G result frozen
  -> development-only preregistration
  -> grouped nested-CV candidate comparison
  -> no candidate passes: stop/revise on development
  -> one candidate passes: freeze recipe/checkpoints
  -> acquire and independently seal new source-disjoint validation set
  -> one new confirmatory evaluation
  -> publish regardless of outcome
```
