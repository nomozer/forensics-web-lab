# Phase 4C.2A — Preregister Stage 2 Fine-Tuning Experiment

## Phase Summary
Formally designed and sealed the Stage 2 fine-tuning experimental protocol before executing any new training runs. Stage 2 reuses the canonical Phase 4C.1 reusable N=250 dataset bundle (`phase_4c1_binary_n250_reusable.tar`) in strictly read-only mode, with cryptographic verification of archive, content, and manifest hashes. Established isolated Stage 2 namespace (`phase_4c2/`) on Google Drive and local artifacts, preventing any writes to Stage 1 output paths. Defined the partial fine-tuning intervention targeting the classifier head and the final convolution block (`features.12` of MobileNetV3-Small), freezing all preceding backbone layers (`features.0` through `features.11`) to strictly bound trainable parameters to 204,674 (19.04% of total model parameters). Pre-registered differential learning rates (5e-5 for backbone, 5e-4 for head), AdamW optimizer, cosine annealing, 20 epochs, and 1-to-1 paired comparison across the exact 15 (N, seed) cells. Zero new training runs, zero locked-test accesses, and zero Stage 2 invocations were executed in this phase.

## Starting Commit
f6eb57df121dbfc908ec1731d55bbe3c87dc5453 (Phase 4C.1D.2 ending)

## Audited Through Commit
f6eb57df121dbfc908ec1731d55bbe3c87dc5453 (Phase 4C.1D.2 canonical evidence base)

## Functional Commit Scope
Phase 4C.2A preregistration and sealing of Stage 2 fine-tuning experiment protocol

## Branch
research/phase-4c1-learning-curve

## Key Preregistration Decisions

1. **Stage 1 Preservation**:
   - Phase 4C.1D.2 commit `f6eb57df121dbfc908ec1731d55bbe3c87dc5453` represents closed, immutable Stage 1 evidence.
   - All 15 Stage 1 run folders, checkpoints, receipts, metrics, and canonical learning curve reports remain intact.
   - Zero modifications to Stage 1 artifacts.

2. **Dataset Binding (Read-Only Reuse)**:
   - Canonical bundle: `phase_4c1_binary_n250_reusable.tar`
   - Archive SHA-256: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`
   - Content SHA-256: `c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`
   - Manifest SHA-256: `411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d`
   - Cohort preservation: N=50 (50 dev sources), N=100 (100 dev sources), N=250 (250 dev sources); strictly nested ($N=50 \subset N=100 \subset N=250$); identical 91 inner validation sources (182 samples, balanced 1:1).
   - Sealed in `research/evidence/phase-4c.2a/dataset_binding.json` with `read_only: true`. No new dataset generated.

3. **Namespace Isolation**:
   - Google Drive output root: `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2/`
   - Local artifact root: external storage path under `phase_4c2/`
   - Forbidden targets: `phase_4c1/runs/execution_79bb115/`, `phase_4c1/runs/`
   - Complete architectural independence across config, environment lock, receipts, checkpoints, logs, and downloads.

4. **Intervention Definition & Trainable Parameter Inventory**:
   - Model: `MobileNetV3Forensics(num_classes=2)` (total parameters: 1,075,234).
   - Unfrozen modules: `features.12` (`Conv2dNormActivation`) and `classifier` (`Sequential` head).
   - Frozen modules: `features.0` through `features.11` (870,560 parameters, `requires_grad = False`).
   - Exact allowable trainable tensors (7 parameters, 204,674 weights):
     1. `features.12.0.weight`: `[576, 96, 1, 1]` (55,296)
     2. `features.12.1.weight`: `[576]` (576)
     3. `features.12.1.bias`: `[576]` (576)
     4. `classifier.0.weight`: `[256, 576]` (147,456)
     5. `classifier.0.bias`: `[256]` (256)
     6. `classifier.3.weight`: `[2, 256]` (512)
     7. `classifier.3.bias`: `[2]` (2)
   - Trainable parameters total: **204,674**. Frozen parameters total: **870,560**.

5. **Sealed Hyperparameters**:
   - Optimizer: `AdamW` (weight decay: $1 \times 10^{-4}$)
   - Differential Learning Rates: $\eta_{\text{backbone}} = 5 \times 10^{-5}$ (`features.12`), $\eta_{\text{head}} = 5 \times 10^{-4}$ (`classifier`)
   - Scheduler: `CosineAnnealingLR` ($T_{\text{max}} = 20, \eta_{\text{min}} = 1 \times 10^{-6}$)
   - Batch size: 32; Max epochs: 20; Early-stopping patience: 5 epochs (checkpoint metric: `inner_val_macro_f1`)
   - Gradient clipping: `max_norm = 1.0`; Mixed precision: `fp32` deterministic
   - Loss function: `FocalLoss(gamma=2.0, label_smoothing=0.05, reduction='mean')`
   - Augmentation: None (canonical deterministic bicubic 224x224, standard ImageNet normalization)
   - GPU policy: Capability-based detection (T4, L4, V100, A100 or compatible CUDA GPU; min VRAM 8.0 GB; not hardcoded to T4 only)

6. **Experiment Matrix & Pairing**:
   - 15 runs: $N \in \{50, 100, 250\} \times \text{seeds } \{42, 1337, 2025, 3407, 9001\}$.
   - Paired 1-to-1 against Stage 1 corresponding cells on identical training cohorts and validation partitions.
   - Exploratory single-seed hyperparameter tuning prior to batch execution is strictly prohibited.

7. **Pre-Registered Analysis Plan**:
   - Primary endpoint: Paired delta Macro-F1 ($\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{Stage 2}} - \text{Macro-F1}_{\text{Stage 1}}$).
   - Secondary endpoints: Balanced accuracy, AUROC, runner-reported validation loss, Brier score, ECE, training time, peak VRAM.
   - Statistics: Mean ± Std and 95% CI (t-distribution df=4) per sample size; paired t-test exploratory ($n=5$); exact sign-flip permutation test resolution limit 0.0625; Holm-Bonferroni correction across 3 sample sizes.
   - Calibration protocol: Temperature scaling must not be fit on model selection data and reported on the same split; evaluation on independent calibration split or via nested cross-fitting required.

8. **Pre-Execution Gate**:
   - `PRE_EXECUTION_GO_NO_GO.json` sealed with verdict `READY`.
   - Execution wave remains pending explicit user authorization.

## Evidence Files Created
- `dataset_binding.json`: Canonical dataset cryptographic binding and invariants.
- `PRE_EXECUTION_GO_NO_GO.json`: Pre-execution gate assessment and verdict.
- `environment.json`: Execution environment configuration and provenance metadata.
- `ml/configs/phase_4c2_stage2_finetuning.yaml`: Formal machine-readable preregistration config.

## Verification Invariants
- `training_runs_in_phase`: 0
- `locked_test_accesses`: 0
- `stage2_invocations`: 0
- `readiness_verdict`: READY (preregistered, pending execution approval)
