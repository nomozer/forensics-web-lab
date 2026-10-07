# Research workstream index

Use descriptive workstream names in current guides, notebook titles, runner help, and new storage paths. Phase codes remain historical trace identifiers in evidence metadata and `STATUS_LEDGER.md`; sealed artifacts and historical run paths are never renamed.

The complete old-path to current-path migration table and sealed exceptions are recorded in `docs/REPOSITORY_NAMING_MAP.md`.

| Phase trace | Workstream / job | Primary documentation | Runner / notebook |
|---|---|---|---|
| `4A` | Dataset feasibility | `docs/DATA_FEASIBILITY.md` | Planning document; no acquisition runner |
| `4C.0` | Binary protocol and baseline reconciliation | `docs/BINARY_CLASSIFICATION_PREREGISTRATION.md` | `scripts/reconcile_training_baseline.py` |
| `4C.1` | Learning curve | `docs/LEARNING_CURVE_EXECUTION_PLAN.md`, `docs/LEARNING_CURVE_COLAB_GUIDE.md` | `ml/training/run_learning_curve.py`, `notebooks/phase_4c1_learning_curve_colab.ipynb` |
| `4C.2A-C` | Partial fine-tuning | `ml/configs/phase_4c2_stage2_finetuning.yaml` | `ml/training/run_partial_finetuning.py`, `notebooks/phase_4c2_finetuning_colab.ipynb` |
| `4C.2D` | Candidate model selection | Historical evidence remains under `research/evidence/phase-4c.2d/` | `scripts/research/finalize_candidate_model_selection.py` |
| `4C.2E-G` | Locked-test confirmatory evaluation | Historical evidence remains under `research/evidence/phase-4c.2e/` through `phase-4c.2g.0.14/` | Sealed executor paths retain their historical names; see the exception table in `docs/REPOSITORY_NAMING_MAP.md` |
| `4C.2H` | Development-only nested CV | `ml/configs/phase_4c2h_development_nested_cv.yaml` | `ml/training/run_nested_cv_development.py`, `notebooks/phase_4c2h_development_colab.ipynb` |
| `4C.3` | Calibrated late fusion | `ml/configs/calibrated_late_fusion_protocol.yaml` | `ml/training/run_calibrated_late_fusion.py`, `notebooks/calibrated_late_fusion_colab.ipynb` |
| `4C.4` | Development robustness | `ml/configs/development_robustness_protocol.yaml` | `ml/training/run_development_robustness.py`, `notebooks/development_robustness_colab.ipynb` |
| `4C.5` | Fusion-shift diagnostics | Historical evidence remains under `research/evidence/phase-4c.5a/` and `phase-4c.5b/` | `scripts/research/diagnose_fusion_shift.py` |
| `4C.6` | Controlled DSP augmentation | `ml/configs/dsp_augmentation_protocol.yaml` | `ml/training/run_dsp_augmentation.py`, `notebooks/dsp_augmentation_colab.ipynb` |
| `4C.7A` | Independent validation preparation: endpoint lock, model bindings, bootstrap/evaluator preflight | `research/evidence/phase-4c.7a/PHASE_REPORT.md`, `ml/configs/independent_validation_protocol.yaml` | `ml/evaluation/independent_evaluator.py` (preflight only until a human-approved cohort exists) |
| `4C.7B` | Independent cohort acquisition: licensed source catalog, content-grounded edit plan, production inpainting, Technical QC, and Human Content QC handoff | `research/evidence/phase-4c.7b/PHASE_REPORT.md`, `research/evidence/phase-4c.7b/CONTENT_GROUNDED_EDITING_AMENDMENT.md` | `scripts/research/run_cohort_acquisition.py`, `notebooks/independent_cohort_acquisition_colab.ipynb` |

## Storage naming

New acquisition runs use:

`MyDrive/Colab Notebooks/forensics-web-lab/independent_cohort_acquisition/runs/<RUN_ID>`

Historical paths remain valid and must be supplied as their original `--output-root` when resumed or audited:

- `pilot-20261007T093824Z`: `MyDrive/forensics-web-lab/phase_4c7b_runs/`
- `pilot-20261007T132003Z`: `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c7b/runs/`

Do not move either historical run automatically. Its binding contains the original code, protocol, catalog, plan, and run identifiers.

## Terminology

- **Allocation plan**: the 440 licensed/disjoint source candidates plus locked source/tool/modification/mask quotas. It does not authorize generation by itself.
- **Content-grounded edit plan**: reviewed prompt, target/placement description, and normalized-canvas geometry for a bounded set of attempts.
- **Technical QC**: machine-checkable dimensions, modes, binary mask, mask-area class, non-blank image, masked change, and outside-mask invariance. It is not semantic approval.
- **Human Content QC**: explicit human review of prompt/operation match and visible edit quality. Agent observations never satisfy this gate.
