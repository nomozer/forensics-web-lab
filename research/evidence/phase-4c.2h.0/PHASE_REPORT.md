# Phase 4C.2H.0 — Development Diagnostics and Nested-CV Protocol Lock

## Verdict

`DEVELOPMENT_DIAGNOSTICS_COMPLETE_PROTOCOL_LOCKED`

This phase used only the 341-source development pool (`development_train` + historical `inner_validation`). It performed no locked-test read, inference, bootstrap, tuning, or result modification. The Phase 4C.2G result remains `INSUFFICIENT_CONFIRMATORY_EVIDENCE`.

## Diagnostic findings

| Question | Observed result | What is proved |
| --- | --- | --- |
| Label mapping | Exact `authentic -> 0`, `ai_edited -> 1`; 16/16 selected development hashes matched; unknown labels now fail closed | No mapping reversal was observed in the bounded smoke path |
| Preprocessing | Canonical transform versus sealed evaluator transform maximum absolute tensor difference `0.0` | No preprocessing drift exists between the compared implementations |
| Trainable parameters | Exactly four classifier tensors, 148,226 trainable parameters; 927,008 frozen parameters | The parameter allowlist is correct |
| Gradients | All four classifier tensors received gradients; no frozen feature parameter received a gradient | The backward path to the classifier is functional and parameter freezing works |
| Frozen buffers | Historical `model.train()` behavior changed 102 feature BatchNorm buffers; the corrected policy changed 0 | A real semantic defect in the historical meaning of “frozen” is reproduced; `requires_grad=False` did not freeze buffers |
| Tiny overfit | 8 sources / 16 images, 300 classifier-only AdamW steps on CPU: loss `0.1827092022 -> 0.05919013545`, accuracy `0.375 -> 1.0` | The fixed preprocessing, frozen representation, classifier, loss, gradient, and optimizer path can memorize this bounded development subset |
| Pair-ranking mechanics | Hinge loss `0.3000000119`; edited-score gradient is `+2.0` for the authentic row and `-2.0` for the edited row under gradient descent | The auxiliary loss pushes paired scores in the intended direction |
| Validation independence | Prediction API accepts only `(model, images)` | Labels, peer images, and source-pair information cannot enter prediction generation through this API |

The BatchNorm finding is a demonstrated implementation cause of historical frozen-buffer mutation. It is **not** demonstrated to be the cause of the Phase 4C.2G outcome. Label mapping, preprocessing drift, missing classifier gradients, and inability to fit this tiny subset were not supported by these probes. Generalization failure remains causally unresolved.

## Locked comparison protocol

- Data: 341 source-disjoint development sources, 682 canonical images; source membership commitment `025cbc4d7990a4ad0a3554507854e2a16d27e04cf2aeef1d8485a2c50e99e6af`.
- Recipes: exactly two — corrected Stage 1 frozen baseline and the same recipe plus pair-ranking loss.
- Shared classification loss: FocalLoss, gamma `2.0`, label smoothing `0.05`, weight `1.0`.
- Pair-ranking treatment only: weight `0.25`, margin `0.2`.
- Seeds: `[42, 1337, 2025]`.
- CV: 5 grouped outer folds and 4 grouped inner folds, source-level assignment only; seeds `20261004` and `20261005`; exact counts and membership commitments are in `development_diagnostics.json`.
- Epoch selection: median best epoch across the four inner folds, round-half-up; refit on outer-train and predict the held-out outer fold once.
- OOF aggregation: concatenate five disjoint outer-test folds per recipe and seed; calculate metrics per seed, then arithmetic mean and sample standard deviation across the three seeds. Recipe comparison is an exploratory paired per-seed OOF Macro-F1 delta.
- Each validation image is predicted independently from pixels only. Pair information and labels are used only after predictions for metrics; pair-ranking is training-only.

## Budget

The locked official matrix contains 120 inner fits plus 30 outer refits = 150 fits, with a ceiling of 3,750 fit-epochs and 4,092 OOF image predictions. These counts are exact planned workload, not completed training.

The measured local CPU smoke used the development-only reusable bundle (341 manifest rows, zero excluded non-development rows), 8 sources, 16 images, one frozen-backbone feature pass, and 300 classifier updates. Recorded training wall time was `2.24731539999993` seconds and in-process diagnostic wall time was `4.84161719999975` seconds on Python 3.12.10 / PyTorch 2.5.1+cu121. This measurement must not be extrapolated as a Colab runtime forecast.

## Artifacts and verification

- Protocol: `ml/configs/phase_4c2h_development_nested_cv.yaml`
- Contracts: `ml/training/phase_4c2h_development.py`
- Reproducible diagnostic: `scripts/research/run_phase_4c2h_development_diagnostics.py`
- Thin Colab launcher: `notebooks/phase_4c2h_development_colab.ipynb`
- Machine-readable receipt/fold lock: `research/evidence/phase-4c.2h.0/development_diagnostics.json`
- Targeted tests: 8/8 PASS.

The official 150-fit experiment was not run in this phase. No new independent validation cohort was collected. A future confirmatory claim still requires a new source-disjoint cohort and must not reuse the retired Phase 4C.2G locked test for selection.
