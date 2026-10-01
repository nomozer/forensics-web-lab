# Phase 4C.1C.0 Preflight and Transfer Preparation - Complete

## Preflight Summary

### Local Git State ✅
- **Branch**: research/phase-4c1-learning-curve
- **Current HEAD**: 8d00f5a6064a70d93afc1468b300cd3b9234de2b
- **Remote HEAD**: 8d00f5a6064a70d93afc1468b300cd3b9234de2b
- **Working Tree**: Clean
- **Local = Remote**: Yes

### Bundle Verification ✅
- **Bundle Path**: D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\smoke_n50_seed42\phase_4c1_smoke_n50_seed42.tar
- **Bytes**: 303,497,728
- **SHA-256**: B57D626D46A08427F6F1E4E1649565BD9D914E2D6F835E109E0391242AF5D0A6 ✅ MATCHES
- **Bundle Validator**: ALL VALIDATIONS PASSED
- **Development Sources**: 50
- **Inner-Validation**: 91 source IDs, 182 samples
- **Locked-Test**: 0 rows, 0 source IDs, 0 files

### Code Archive ✅
- **Archive**: phase_4c1_code_8d00f5a.tar.gz
- **Bytes**: 745,321
- **SHA-256**: 5ED90DB104E9C0CE0FD220B52650EF936F298737530D3674403951748E2647EA
- **Created From**: HEAD 8d00f5a (short: 8d00f5a)

### Execution Receipts Created
1. **execution-head-receipt.json**: Current HEAD, bundle SHA, runner/config paths, historical commits
2. **execution-code-receipt.json**: Code archive SHA-256 (5ED90DB104E9C0CE0FD220B52650EF936F298737530D3674403951748E2647EA), 745,321 bytes

### Bundle Contents Verified ✅
- Development source IDs: 50 (N=50 cohort)
- Inner-validation source IDs: 91
- Validation samples: 182 (91 authentic + 91 ai_edited)
- Development/validation overlap: 0
- Locked-test: 0 rows, 0 source IDs, 0 files

---

## Next Steps: VS Code Colab Execution

### 1. Transfer to VS Code Colab
Upload to `/content/forensics-transfer/`:
1. Code archive: `phase_4c1_code_8d00f5a.tar.gz` (745,321 bytes, SHA-256: 5ED90DB104E9C0CE0FD220B52650EF936F298737530D3674403951748E2647EA)
2. Data bundle: `phase_4c1_smoke_n50_seed42.tar` (303,497,728 bytes, SHA-256: B57D626D46A08427F6F1E4E1649565BD9D914E2D6F835E109E0391242AF5D0A6)

### 2. T4 Preflight (Run in Colab Terminal)
```bash
pwd
ls -la /content/forensics-transfer/
nvidia-smi
python -c "
import sys, torch
print('python:', sys.version)
print('executable:', sys.executable)
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
"
# Verify GPU is T4
```

### 3. Extract and Verify
```bash
mkdir -p /content/forensics-web-lab /content/bundle /content/phase_4c1_outputs
tar -xzf /content/forensics-transfer/phase_4c1_code_8d00f5a.tar.gz -C /content/forensics-web-lab
tar -xf /content/forensics-transfer/phase_4c1_smoke_n50_seed42.tar -C /content/bundle

# Verify hashes
sha256sum /content/forensics-transfer/phase_4c1_code_8d00f5a.tar.gz
sha256sum /content/forensics-transfer/phase_4c1_smoke_n50_seed42.tar
```

### 3. Create Environment
```bash
python -m venv --system-site-packages /content/phase4c1-venv
/content/phase4c1-venv/bin/python -m pip install --upgrade pip
/content/phase4c1-venv/bin/python -m pip install -r /content/forensics-web-lab/ml/requirements-dev.txt
# If CUDA PyTorch already works, skip torch/torchvision install
```

### 4. Dry-run Runner
```bash
cd /content/forensics-web-lab
/content/phase4c1-venv/bin/python -m ml.training.run_phase_4c1 --help
```

---

## Seed Execution Plan

| Order | Seed | Label | Notes |
|-------|------|-------|-------|
| 1 | 42 | T4 | Re-run on T4 (local GTX 1650 was EXPLORATORY_LOCAL_SMOKE) |
| 2 | 1337 | T4 | |
| 3 | 2025 | T4 | |
| 4 | 3407 | T4 | |
| 5 | 9001 | T4 | |

All seeds use:
- Fixed N=50 cohort (same 50 source IDs)
- Same inner-validation (91 source IDs, 182 samples)
- Stage 1 frozen backbone
- Frozen MobileNetV3-Small + Linear Head
- Epoch budget: 25, Early stopping: 5
- Same environment lock

---

## Local Artifact Storage
Results downloaded to: `D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs\colab_t4\n50_seed_<SEED>\`

Each seed produces:
- best_checkpoint.pt
- run_receipt.json
- epoch_history.json
- predictions.json
- training_history.csv
- metrics.json
- environment.json
- checksums.json
- environment-binding.json
- phase4c1_environment_lock.txt

---

## Post-Execution: Local Verification
After all 5 seeds complete and artifacts downloaded:
1. Verify each archive SHA-256
2. Recompute Macro-F1, Balanced Accuracy, AUROC, ECE, Brier
3. Aggregate mean/std/95% CI across 5 seeds
4. Compare with dummy baseline (paired bootstrap)
5. Generate Phase 4C.1C.0 evidence

---

## Evidence to Generate
Phase 4C.1C.0 evidence at: `research/evidence/phase-4c.1c.0/`

Key artifacts:
- environment.json
- execution-head-receipt.json
- t4-environment-lock.txt / .sha256
- code-archive-receipt.json
- bundle-verification.json
- cohort-invariance.json
- seed-<SEED>-result.json (5 files)
- cross-seed-artifact-audit.json
- aggregate-metrics.json
- statistical-comparison.json
- stage2-gate-audit.json
- test-summary.json
- evidence-manifest.json
- PHASE_REPORT.md

---

## Gates
- **T4 Preflight**: GPU = T4, CUDA available
- **Bundle Validation**: PASS
- **Code Archive SHA**: Matches receipt
- **Environment Lock**: Created at seed 42, matched by seeds 1337, 2025, 3407, 9001
- **Each Seed**: 182 validation samples, 91 source IDs, locked-test = 0
- **Aggregate**: Mean/std/95% CI over 5 seeds
- **Stage 2**: INSUFFICIENT_EVIDENCE (metadata NOT_MEASURED)

---

## Ready for Execution
All local preflight complete. Ready for VS Code Colab T4 multi-seed execution.

**Execution Code HEAD**: `8d00f5a6064a70d93afc1468b300cd3b9234de2b`
**Bundle SHA-256**: B57D626D46A08427F6F1E4E1649565BD9D914E2D6F835E109E0391242AF5D0A6
**Code Archive SHA-256**: 5ED90DB104E9C0CE0FD220B52650EF936F298737530D3674403951748E2647EA
**Bundle SHA-256**: B57D626D46A08427F6F1E4E1649565BD9D914E2D6F835E109E0391242AF5D0A6