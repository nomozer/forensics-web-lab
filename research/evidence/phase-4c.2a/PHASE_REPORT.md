# Phase 4C.2A / 4C.2A.1 — Preregister Stage 2 Fine-Tuning Experiment & Implementation Contract Reconciliation

## Phase Summary
Formally designed and sealed the Stage 2 fine-tuning experimental protocol (Phase 4C.2A) and reconciled the execution contract directly against the codebase before building the training operator (Phase 4C.2A.1). Stage 2 reuses the canonical Phase 4C.1 reusable N=250 dataset bundle (`phase_4c1_binary_n250_reusable.tar`) in strictly read-only mode, with cryptographic verification of archive, content, and manifest hashes. Established isolated Stage 2 namespace (`phase_4c2/`) on Google Drive and local artifacts, preventing any writes to Stage 1 output paths.

In Phase 4C.2A.1, the implementation contract was verified against live Python reflection:
1. Model parameter naming was confirmed to follow Hệ A (`features.12.*` and `classifier.*`, with exactly 7 tensors and 204,674 trainable parameters).
2. Initialization contract was explicitly chosen and sealed to `from_pretrained_backbone_with_fresh_head` without loading Stage 1 checkpoints.
3. BatchNorm freeze policy was established with `apply_frozen_bn_policy()` ensuring `features.0-11` buffers remain byte-identical during training.
4. Differential optimizer parameter groups (5e-5 for backbone, 5e-4 for head) were verified disjoint and complete.
5. Machine-readable hyperparameter diff table was created (23 fields), establishing the formal treatment designation as `pre-registered partial fine-tuning protocol`.
6. Dedicated Stage 2 runner `ml/training/run_phase_4c2.py` and receipt schema `docs/schemas/stage2-receipt.v1.schema.json` were created, keeping Stage 1 snapshot 79bb115 immutable.
7. Readiness semantics were codified: `PREREGISTRATION_READY`, `IMPLEMENTATION_CONTRACT_VERIFIED`, with Colab execution `NOT_READY`.

Zero new training runs, zero locked-test accesses, and zero Stage 2 research invocations were executed in this phase.

## Starting Commit
f6eb57df121dbfc908ec1731d55bbe3c87dc5453 (Phase 4C.1D.2 canonical evidence base)

## Preregistration Commit
1d67007967dad18cf82c3104b6d6fbfc0b97e3f9 (Phase 4C.2A base)

## Branch
research/phase-4c1-learning-curve

## Key Implementation Contract Decisions

1. **Stage 1 Preservation**:
   - Phase 4C.1D.2 commit `f6eb57df121dbfc908ec1731d55bbe3c87dc5453` represents closed, immutable Stage 1 evidence.
   - All 15 Stage 1 run folders, checkpoints, receipts, metrics, and canonical learning curve reports remain intact.
   - Stage 1 runner snapshot `ml/training/run_phase_4c1.py` remains untouched.
   - Zero modifications to Stage 1 artifacts.

2. **Dataset Binding (Read-Only Reuse)**:
   - Canonical bundle: `phase_4c1_binary_n250_reusable.tar`
   - Archive SHA-256: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`
   - Content SHA-256: `c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`
   - Manifest SHA-256: `411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d`
   - Cohort preservation: N=50 (50 dev sources), N=100 (100 dev sources), N=250 (250 dev sources); strictly nested ($N=50 \subset N=100 \subset N=250$); identical 91 inner validation sources (182 samples, balanced 1:1).
   - Sealed in `research/evidence/phase-4c.2a/dataset_binding.json` with `read_only: true`. No new dataset generated.

3. **Namespace Isolation & Output Path Security**:
   - Google Drive output root: `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2/`
   - Local artifact root: external storage path under `phase_4c2/`
   - Forbidden targets: `phase_4c1/runs/execution_79bb115/`, `phase_4c1/runs/`
   - Security validator `validate_stage2_output_dir()` in `run_phase_4c2.py` fails closed if any output path targets or symlinks to Stage 1 output paths.

4. **Model Architecture & Trainable Parameter Allowlist**:
   - Class: `MobileNetV3Forensics(num_classes=2)` (total parameters: 1,075,234).
   - Unfrozen modules: `features.12` (`Conv2dNormActivation`) and `classifier` (`Sequential` head).
   - Frozen modules: `features.0` through `features.11` (870,560 parameters, `requires_grad = False`).
   - Confirmed Parameter Naming System: Hệ A (`features.12.*` and `classifier.*`).
   - Exact allowable trainable tensors (7 parameters, 204,674 weights):
     1. `features.12.0.weight`: `[576, 96, 1, 1]` (55,296)
     2. `features.12.1.weight`: `[576]` (576)
     3. `features.12.1.bias`: `[576]` (576)
     4. `classifier.0.weight`: `[256, 576]` (147,456)
     5. `classifier.0.bias`: `[256]` (256)
     6. `classifier.3.weight`: `[2, 256]` (512)
     7. `classifier.3.bias`: `[2]` (2)
   - Trainable parameters total: **204,674** (19.04%). Frozen parameters total: **870,560** (80.96%).

5. **Initialization Contract**:
   - Policy: `from_pretrained_backbone_with_fresh_head`.
   - Backbone initialized from `MobileNet_V3_Small_Weights.DEFAULT`.
   - Classifier initialized with seeded torch default initialization at model construction.
   - Stage 1 best checkpoints are strictly NOT loaded (`parent_checkpoint_path = null`, `parent_checkpoint_hash = null`).

6. **BatchNorm Freeze Policy**:
   - Modules in `features.0` through `features.11` are enforced in `eval()` mode.
   - Runner implements `apply_frozen_bn_policy(model)` called after every `model.train()` invocation.
   - Verified by behavioral test: `running_mean`, `running_var`, and `num_batches_tracked` of frozen blocks remain byte-identical during forward passes.

7. **Differential Optimizer Groups**:
   - Group 1: `features.12.*` at $\eta = 5 \times 10^{-5}$, weight decay $1 \times 10^{-4}$ (3 tensors, 56,448 parameters).
   - Group 2: `classifier.*` at $\eta = 5 \times 10^{-4}$, weight decay $1 \times 10^{-4}$ (4 tensors, 148,226 parameters).
   - Groups are mathematically disjoint and their union equals the exact 7 allowlisted tensors.

8. **Hyperparameter Diff & Treatment Designation**:
   - Documented in `research/evidence/phase-4c.2a/stage1_stage2_hyperparameter_diff.json` across 23 fields.
   - Treatment designation: `pre-registered partial fine-tuning protocol`.
   - Loss function: `FocalLoss(gamma=2.0, alpha=null, label_smoothing=0.05, reduction='mean')` (parity with Stage 1 verified).
   - Scheduler: `CosineAnnealingLR` ($T_{\text{max}} = 20, \eta_{\text{min}} = 1 \times 10^{-6}$). Max epochs: 20. Early stopping patience: 5.
   - Gradient clipping: `max_norm = 1.0`.

9. **Dedicated Runner & Receipt Schema**:
   - Dedicated Stage 2 runner: `ml/training/run_phase_4c2.py`.
   - Receipt schema: `docs/schemas/stage2-receipt.v1.schema.json`.
   - Stage identifier strictly locked to `partial_finetune` (stage `frozen` is rejected).

10. **Statistical Hypotheses & Evidence Semantics**:
    - Primary endpoint: Paired delta Macro-F1 ($\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{Stage 2}} - \text{Macro-F1}_{\text{Stage 1}}$).
    - Holm family of 3 hypotheses:
      - $H_{N50}$: paired $\Delta\text{Macro-F1}$ (Stage 2 - Stage 1) at $N=50$
      - $H_{N100}$: paired $\Delta\text{Macro-F1}$ (Stage 2 - Stage 1) at $N=100$
      - $H_{N250}$: paired $\Delta\text{Macro-F1}$ (Stage 2 - Stage 1) at $N=250$
    - Exact two-sided sign-flip permutation test: $p_{\min} = 2 / 2^5 = 0.0625$ with $n=5$.
    - Evidence semantics: `inner_validation` serves both checkpoint selection and development metrics reporting; Stage 2 provides exploratory/development evidence, not locked-test confirmatory evidence.

11. **Readiness Semantics**:
    - `preregistration_status`: `PREREGISTRATION_READY`
    - `implementation_contract_status`: `IMPLEMENTATION_CONTRACT_VERIFIED`
    - `colab_execution_status`: `NOT_READY`
    - `PRE_EXECUTION_GO_NO_GO.json` sealed with verdict `IMPLEMENTATION_CONTRACT_VERIFIED`.

## Evidence Files Created / Updated
- `dataset_binding.json`: Canonical dataset cryptographic binding and invariants.
- `PRE_EXECUTION_GO_NO_GO.json`: Pre-execution gate assessment and verdict (`IMPLEMENTATION_CONTRACT_VERIFIED`).
- `environment.json`: Execution environment configuration and provenance metadata.
- `stage1_stage2_hyperparameter_diff.json`: Machine-readable comparison table of 23 hyperparameter fields.
- `docs/schemas/stage2-receipt.v1.schema.json`: Formal JSON Schema for Stage 2 run receipts.
- `ml/configs/phase_4c2_stage2_finetuning.yaml`: Formal machine-readable preregistration config.
- `ml/training/run_phase_4c2.py`: Dedicated Stage 2 training runner.
- `ml/tests/test_phase_4c2_implementation_contract.py`: 14 contract unit and synthetic behavioral tests.

## Verification Invariants
- `training_runs_in_phase`: 0
- `locked_test_accesses`: 0
- `stage2_invocations`: 0
- `readiness_verdict`: IMPLEMENTATION_CONTRACT_VERIFIED
