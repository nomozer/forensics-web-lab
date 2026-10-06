# Phase 4C.6 — Execution Guide (local, development_real)

The commands below run in the repository root (or in an extracted package directory) with the Python
3.12 environment from `ml/requirements.txt`. No GPU, no weights, no image decoding are needed: every
feature comes from the verified 4C.3B / 4C.4B caches. Nothing reads the locked test.

## 1. Inputs (all outside Git; all produced by earlier phases)

| Flag | Directory / file | Files read | Expected schema |
| :--- | :--- | :--- | :--- |
| `--manifest` | `manifest_pilot_a_option_p.csv` | the manifest only (development rows; image files must exist but are **not opened**) | SHA-256 `e1e6b6d2…` (robustness protocol binding) |
| `--data-root` | directory the manifest paths are relative to | existence checks only | 682 development image paths |
| `--ablation-dir` | visual/DSP ablation output | `shared/{visual,dsp}_features.pt` + receipts; `fits/visual_dsp_fusion/outer_{0..4}/{fit_receipt.json,model.pt,predictions.csv}` | caches `[341, 2, 576]` / `[341, 2, 16]` float32, commitments `351e4952…` / `582108a2…` |
| `--late-fusion-dir` | Phase 4C.3B late-fusion run | `run_manifest.json`, `fits/outer_{0..4}/{fold_receipt.json,fold_model.json,predictions.csv}` | run manifest SHA `99b95ff2…`; fold prediction SHAs as in the robustness protocol |
| `--robustness-dir` | Phase 4C.4B robustness runner output (contains `full/`) | `full/scope_manifest.json`, `full/original_gate.json`, `full/conditions/<6>/{condition_receipt.json,features.npz,images.json,predictions.csv}` | `features.npz` keys `visual` `[682,576]`, `dsp` `[682,16]` float32; receipt SHAs bound in the protocol (`predecessors.robustness_full_scope`) |
| `--output-dir` | **new** directory, outside Git or under `data/research/local-artifacts/` | — | refused inside the repository or inside any input directory |

## 2. Commands

```powershell
$Common = @(
  "--manifest",        "<path>\manifest_pilot_a_option_p.csv",
  "--data-root",       "<bundle dir>",
  "--ablation-dir",    "<visual_dsp_ablation output>",
  "--late-fusion-dir", "<calibrated_late_fusion output>",
  "--robustness-dir",  "data/research/local-artifacts/development_robustness",
  "--output-dir",      "data/research/local-artifacts/dsp_augmentation")

# 1) Preflight: 0 fits. Expect PREFLIGHT_PASS, data_origin development_real, every reconstruction difference 0.0 (≤ 1e-9).
python -m ml.training.run_dsp_augmentation --mode preflight @Common

# 2) Pilot: outer fold 0 only, exactly 92 fits; technical/runtime check only (do not change anything based on its numbers).
python -m ml.training.run_dsp_augmentation --mode pilot @Common

# 3) Full: resumes the verified pilot fold, fits folds 1-4 (368 fits); total 460; budget_check PASS.
python -m ml.training.run_dsp_augmentation --mode full @Common

# 4) Analyze into the evidence folder of the execution phase.
python scripts/research/analyze_dsp_augmentation.py `
  --output-dir data/research/local-artifacts/dsp_augmentation `
  --report-dir research/evidence/dsp_augmentation

# 5) Verify: rebuilds every analysis output and compares bytes (expect REPRODUCED, mismatched_files []).
python scripts/research/analyze_dsp_augmentation.py --verify `
  --output-dir data/research/local-artifacts/dsp_augmentation `
  --report-dir research/evidence/dsp_augmentation
```

Packaged alternative: `python scripts/research/build_calibrated_late_fusion_package.py --experiment controlled_dsp_augmentation --source-commit <functional commit> --output <file>`,
then verify `SNAPSHOT_MANIFEST.json` against `package_receipt.json` and run the same commands inside the extracted directory
(or use `notebooks/dsp_augmentation_colab.ipynb`).

## 3. Output directory, resume and history

- `$Out/preflight_receipt.json`, `run_manifest.json`, `pilot_summary.json`, `full_summary.json`,
  `fits/outer_k/{predictions.csv, fold_model.json, fold_receipt.json}`.
- A fold is resumed only when its receipt, predictions and model hashes verify, its bindings
  (protocol SHA, source membership, all-condition feature commitments, data origin) match, and its gate
  is PASS. Anything else is refused: move the directory aside, never edit it.
- `predictions.csv` holds sample-level rows (source IDs): it stays outside Git. Only the analyzer output
  (aggregate CSV/JSON/MD/SVG) goes to `research/evidence/dsp_augmentation/`.

## 4. Local gates (stop on the first failure; never loosen a tolerance)

1. Preflight `PREFLIGHT_PASS`; reconstruction 0 decision mismatches, max |Δp| ≤ 1e-9 for all 6 conditions and vs 4C.3B.
2. Pilot: fold 0 gate PASS, 92 fits, 0 unconverged expected (report if not).
3. Full: every fold gate PASS (`baseline_model_equal: true`), `budget_check: PASS`, total 460 fits.
4. Analyzer: verdict from the locked vocabulary; `--verify` → `REPRODUCED`.
5. `python -m pytest ml/tests -m "not requires_research_artifact"`, `pnpm continuity:check`,
   `node --test scripts/__tests__/continuity-check.test.mjs`, `git diff --check`, branch CI PASS.

If a gate fails because of an implementation defect: reproduce it with a regression test, record which
history it could affect, keep all historical artifacts unchanged, and stop.
