# Repository naming map

This document records the repository-wide functional naming normalization performed on branch `research/independent-cohort-acquisition`. Phase identifiers remain in protocol metadata, report prose, evidence metadata, and `docs/continuity/STATUS_LEDGER.md`; they are no longer the primary name of active, unsealed files where a functional name is safe.

## Naming rules

- Technical files use `snake_case`; executable scripts start with a verb.
- Colab notebooks use `<job>_colab.ipynb`.
- Current documentation uses the workstream or document purpose as its primary name.
- Historical evidence, raw artifacts, and sealed execution members retain their exact paths.
- A historical Git commit remains the authority for old path/hash bindings; this migration does not rewrite scientific results.

## Renamed files

| Previous path | Current path | Function |
|---|---|---|
| `docs/PHASE_4A_DATA_FEASIBILITY.md` | `docs/DATA_FEASIBILITY.md` | Dataset feasibility and preparation plan |
| `docs/PHASE_4C_PREREGISTRATION.md` | `docs/BINARY_CLASSIFICATION_PREREGISTRATION.md` | Binary authentic-versus-edited preregistration |
| `docs/PHASE_4C1_EXECUTION_PLAN.md` | `docs/LEARNING_CURVE_EXECUTION_PLAN.md` | Learning-curve execution plan |
| `docs/PHASE_4C1_COLAB_GUIDE.md` | `docs/LEARNING_CURVE_COLAB_GUIDE.md` | Learning-curve Colab guide |
| `ml/datasets/export_phase_4c1_bundle.py` | `ml/datasets/export_learning_curve_bundle.py` | Export a learning-curve dataset bundle |
| `ml/datasets/validate_phase_4c1_bundle.py` | `ml/datasets/validate_learning_curve_bundle.py` | Validate a learning-curve dataset bundle |
| `ml/training/run_phase_4c1.py` | `ml/training/run_learning_curve.py` | Run learning-curve training |
| `ml/training/run_phase_4c2.py` | `ml/training/run_partial_finetuning.py` | Run partial fine-tuning |
| `ml/training/run_phase_4c2h.py` | `ml/training/run_nested_cv_development.py` | Run development-only nested CV |
| `scripts/reconcile_phase4c0.py` | `scripts/reconcile_training_baseline.py` | Reconcile the smoke checkpoint and baselines |
| `scripts/reproduce-phase-4b3-evidence.py` | `scripts/reproduce_dataset_evidence.py` | Reproduce dataset and environment evidence |
| `scripts/research/analyze_phase_4c1d_runs.py` | `scripts/research/analyze_learning_curve_runs.py` | Analyze learning-curve runs |
| `scripts/research/analyze_phase_4c2_paired.py` | `scripts/research/analyze_partial_finetuning_pairs.py` | Compare frozen and partially fine-tuned runs |
| `scripts/research/analyze_phase_4c2g_closure.py` | `scripts/research/analyze_confirmatory_closure.py` | Audit confirmatory outputs and closure analyses |
| `scripts/research/analyze_phase_4c2h_results.py` | `scripts/research/analyze_nested_cv_results.py` | Audit nested-CV results |
| `scripts/research/build_phase_4c2h_snapshot.py` | `scripts/research/build_nested_cv_snapshot.py` | Build a deterministic nested-CV snapshot |
| `scripts/research/finalize_phase_4c2d_model_selection.py` | `scripts/research/finalize_candidate_model_selection.py` | Finalize candidate model selection |
| `scripts/research/ingest_phase_4c2_results.py` | `scripts/research/ingest_partial_finetuning_results.py` | Ingest and audit partial fine-tuning results |
| `scripts/research/reconcile_phase_4c2c_lineage.py` | `scripts/research/reconcile_partial_finetuning_lineage.py` | Reconcile partial fine-tuning metric lineage |
| `scripts/research/reproduce_phase_4c1b6r1_statistics.py` | `scripts/research/reproduce_paired_source_bootstrap.py` | Reproduce the paired source bootstrap |
| `scripts/research/reproduce_phase_4c1b6r2_statistics.py` | `scripts/research/reproduce_paired_cluster_bootstrap.py` | Reproduce the paired cluster bootstrap |
| `scripts/research/run_phase_4c2h_development_diagnostics.py` | `scripts/research/run_nested_cv_diagnostics.py` | Run bounded nested-CV diagnostics |
| `ml/tests/test_phase4c0_gates.py` | `ml/tests/test_binary_protocol_gates.py` | Binary protocol gate tests |
| `ml/tests/test_phase4c0a_reconciliation.py` | `ml/tests/test_training_baseline_reconciliation.py` | Baseline reconciliation tests |
| `ml/tests/test_phase_4c1_bundle.py` | `ml/tests/test_learning_curve_bundle.py` | Learning-curve bundle tests |
| `ml/tests/test_phase_4c1_notebook.py` | `ml/tests/test_learning_curve_notebook.py` | Learning-curve notebook tests |
| `ml/tests/test_phase_4c1_operator.py` | `ml/tests/test_learning_curve_operator.py` | Learning-curve operator tests |
| `ml/tests/test_phase_4c1_runner.py` | `ml/tests/test_learning_curve_runner.py` | Learning-curve runner tests |
| `ml/tests/test_phase_4c1d_analysis.py` | `ml/tests/test_learning_curve_analysis.py` | Learning-curve analysis tests |
| `ml/tests/test_phase_4c2_implementation_contract.py` | `ml/tests/test_partial_finetuning_contract.py` | Partial fine-tuning implementation-contract tests |
| `ml/tests/test_phase_4c2_notebook.py` | `ml/tests/test_partial_finetuning_notebook.py` | Partial fine-tuning notebook tests |
| `ml/tests/test_phase_4c2_operator.py` | `ml/tests/test_partial_finetuning_operator.py` | Partial fine-tuning operator tests |
| `ml/tests/test_phase_4c2_preregistration.py` | `ml/tests/test_partial_finetuning_protocol.py` | Partial fine-tuning protocol tests |
| `ml/tests/test_phase_4c2_results_ingest.py` | `ml/tests/test_partial_finetuning_results_ingest.py` | Partial fine-tuning ingestion tests |
| `ml/tests/test_phase_4c2c_paired_analysis.py` | `ml/tests/test_partial_finetuning_paired_analysis.py` | Partial fine-tuning paired-analysis and lineage tests |
| `ml/tests/test_phase_4c2d_model_selection.py` | `ml/tests/test_candidate_model_selection.py` | Candidate model-selection tests |
| `ml/tests/test_phase_4c2e_preregistration.py` | `ml/tests/test_confirmatory_protocol.py` | Confirmatory protocol tests |
| `ml/tests/test_phase_4c2f_evaluator.py` | `ml/tests/test_locked_test_evaluator.py` | Locked-test evaluator tests |
| `ml/tests/test_phase_4c2g_closure_analysis.py` | `ml/tests/test_confirmatory_closure_analysis.py` | Confirmatory closure-analysis tests |
| `ml/tests/test_phase_4c2g_execution.py` | `ml/tests/test_confirmatory_execution.py` | Confirmatory execution tests |
| `ml/tests/test_phase_4c2g_manifest_custodian.py` | `ml/tests/test_manifest_custodian.py` | Locked-test manifest-custodian tests |
| `ml/tests/test_phase_4c2g_preparation.py` | `ml/tests/test_confirmatory_preparation.py` | Confirmatory package/runtime preparation tests |
| `ml/tests/test_phase_4c2h_development.py` | `ml/tests/test_nested_cv_development.py` | Nested-CV contract tests |
| `ml/tests/test_phase_4c2h_notebook.py` | `ml/tests/test_nested_cv_notebook.py` | Nested-CV notebook tests |
| `ml/tests/test_phase_4c2h_results_analysis.py` | `ml/tests/test_nested_cv_results_analysis.py` | Nested-CV result-analysis tests |
| `ml/tests/test_phase_4c2h_runner.py` | `ml/tests/test_nested_cv_runner.py` | Nested-CV runner tests |

## Phase trace to workstream

| Phase trace | Workstream | Current primary files |
|---|---|---|
| `4A` | Dataset feasibility | `docs/DATA_FEASIBILITY.md` |
| `4C.0` | Binary protocol and baseline reconciliation | `docs/BINARY_CLASSIFICATION_PREREGISTRATION.md`, `scripts/reconcile_training_baseline.py` |
| `4C.1` | Learning curve | `docs/LEARNING_CURVE_EXECUTION_PLAN.md`, `ml/training/run_learning_curve.py`; sealed config/notebook paths listed below |
| `4C.2A-C` | Partial fine-tuning | `ml/training/run_partial_finetuning.py`; sealed protocol/operator/notebook paths listed below |
| `4C.2D` | Candidate model selection | `scripts/research/finalize_candidate_model_selection.py` |
| `4C.2E-G` | Locked-test confirmatory evaluation | Sealed execution names listed below; current analyses use `scripts/research/analyze_confirmatory_closure.py` |
| `4C.2H` | Development-only nested CV | `ml/training/run_nested_cv_development.py`; sealed protocol/notebook paths listed below |
| `4C.3` | Calibrated late fusion | `ml/configs/calibrated_late_fusion_protocol.yaml`, `ml/training/run_calibrated_late_fusion.py` |
| `4C.4` | Development robustness | `ml/configs/development_robustness_protocol.yaml`, `ml/training/run_development_robustness.py` |
| `4C.5` | Fusion-shift diagnostics | `scripts/research/diagnose_fusion_shift.py` |
| `4C.6` | Controlled DSP augmentation | `ml/configs/dsp_augmentation_protocol.yaml`, `ml/training/run_dsp_augmentation.py` |
| `4C.7A` | Independent validation preparation | `ml/configs/independent_validation_protocol.yaml`, `ml/evaluation/independent_evaluator.py` |
| `4C.7B` | Independent cohort acquisition | `scripts/research/run_cohort_acquisition.py`, `notebooks/independent_cohort_acquisition_colab.ipynb` |

## Exact-path exceptions

The following Git-tracked paths remain unchanged because sealed configs, notebooks, execution packages, source-binding manifests, component-hash keys, or authorization schemas refer to their exact names:

- `ml/configs/phase_4c1_learning_curve.yaml`
- `ml/configs/phase_4c2_stage2_finetuning.yaml`
- `ml/configs/phase_4c2h_development_nested_cv.yaml`
- `ml/training/phase_4c2h_development.py`
- `notebooks/phase_4c1_learning_curve_colab.ipynb`
- `notebooks/phase_4c2_finetuning_colab.ipynb`
- `notebooks/phase_4c2h_development_colab.ipynb`
- `scripts/phase_4c2_execute_all.sh`

- `ml/evaluation/phase_4c2g_dataset.py`
- `ml/evaluation/phase_4c2g_io.py`
- `ml/evaluation/phase_4c2g_model.py`
- `ml/evaluation/phase_4c2g_windows.py`
- `ml/evaluation/run_phase_4c2f_evaluator.py`
- `ml/evaluation/run_phase_4c2g_confirmatory.py`
- `scripts/research/build_phase4c2g_execution_package.py`
- `scripts/research/build_phase4c2g_manifest_custodian_package.py`
- `scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1`
- `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`
- `scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1`
- `scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1`
- `scripts/research/verify_phase_4c2g_offline_runtime.py`

All `research/evidence/phase-*` trees and their internal filenames are also immutable historical evidence. External ZIP files, extracted images/masks, receipts, ledgers, models, datasets, checkpoints, and historical run directories remain untouched. References in older `STATUS_LEDGER.md` entries intentionally preserve the paths that were true at the recorded commit.
