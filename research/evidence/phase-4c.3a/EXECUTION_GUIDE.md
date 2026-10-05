# Phase 4C.3A — Execution Guide: Calibrated Late Fusion (development only)

This guide covers the part that could not run in the cloud session: the dataset bundle, the
pretrained MobileNetV3-Small weights and the Visual/DSP ablation feature caches live outside Git.
Nothing here touches the locked test; the runner only loads the 341 development rows.

## 0. Inputs you need

| Input | Where it lives today | Binding |
| :--- | :--- | :--- |
| Development bundle with `manifest_pilot_a_option_p.csv` | local research storage or Drive `phase_4c1/inputs/phase_4c1_binary_n250_reusable.tar` | manifest SHA-256 `e1e6b6d2...b626` (checked by the runner) |
| Visual/DSP ablation output dir (`shared/visual_features.pt`, `shared/dsp_features.pt`, receipts, `fits/<recipe>/outer_k/`) | the ablation run directory (Drive `visual_dsp_ablation/runs/execution_ablation` on Colab) | cache SHA-256 in its own receipts (checked by the runner) |
| `mobilenet_v3_small-047dcff4.pth` | only needed if the ablation caches are unavailable | SHA-256 `047dcff4...4e1f` |

Reusing the ablation caches is the intended path: identical features make every late-fusion
prediction paired with the ablation predictions, and the analyzer then checks that `visual_raw` and
`dsp_raw` reproduce the ablation's `visual_control` and `dsp_only` exactly.

## 1. Local run (Windows PowerShell, from the repository root)

Set the three paths for your machine (placeholders below; do not commit machine paths):

```powershell
git fetch origin claude/keen-knuth-4esvaw
git checkout claude/keen-knuth-4esvaw
$Manifest    = "<path to manifest_pilot_a_option_p.csv>"
$DataRoot    = "<directory the manifest paths are relative to>"
$AblationRun = "<visual/dsp ablation output dir containing shared/ and fits/>"
$Out         = "data/research/local-artifacts/calibrated_late_fusion"

# 1) Preflight: verifies protocol, 682 image hashes, caches; fits nothing.
python -m ml.training.run_calibrated_late_fusion --mode preflight `
  --manifest $Manifest --data-root $DataRoot --feature-cache-dir $AblationRun --output-dir $Out

# 2) Pilot: outer fold 0 only (61 fits). Inspect fits/outer_0/fold_receipt.json.
python -m ml.training.run_calibrated_late_fusion --mode pilot `
  --manifest $Manifest --data-root $DataRoot --feature-cache-dir $AblationRun --output-dir $Out

# 3) Full: 5 outer folds (305 fits); fold 0 is resumed after hash verification.
python -m ml.training.run_calibrated_late_fusion --mode full `
  --manifest $Manifest --data-root $DataRoot --feature-cache-dir $AblationRun --output-dir $Out

# 4) Analysis into the evidence folder for the execution phase.
python scripts/research/analyze_calibrated_late_fusion.py --output-dir $Out `
  --ablation-output-dir $AblationRun --report-dir research/evidence/calibrated_late_fusion
```

Expected runtime is seconds for steps 2-4 on CPU once caches exist (the ablation's 435 fits took
13.27 s; this run has 305 fits). The 10,000-replicate bootstrap in step 4 is vectorised and took
about 7 s on synthetic data in the cloud session. These are expectations, not measurements on the
real cohort.

If the ablation caches are not available, drop `--feature-cache-dir` and pass
`--weights-path <mobilenet_v3_small-047dcff4.pth>`; caches are then rebuilt under `$Out/shared`.
Rebuilt visual features on a different device may differ in the last float bits, in which case
the reproduction check can report `DIVERGED` at sample level; report that as observed.

## 2. Colab alternative

1. Build the code package from the bound commit (`package_receipt.json` lists it):
   `python scripts/research/build_calibrated_late_fusion_package.py --source-commit <commit> --receipt <file>`
   Add `--weights-path <...pth>` to embed the weights if the ablation caches are not on Drive.
2. Upload it to Drive `calibrated_late_fusion/inputs/phase_4c3_calibrated_late_fusion_code.tar.gz`.
3. Open `notebooks/calibrated_late_fusion_colab.ipynb`, set `EXPECTED_SNAPSHOT_MANIFEST_SHA256`
   from the receipt, run with `MODE="preflight"`, then `"pilot"`, then `"full"` with `ALLOW_FULL=True`.
4. The last cell writes `calibrated_late_fusion/phase_4c3_results.tar.gz` (run receipts,
   predictions, report; no feature caches).

## 3. What to bring back for Phase 4C.3B

- `research/evidence/calibrated_late_fusion/` produced by step 4 (CSV/JSON/MD/figures; aggregate
  numbers only, no images).
- The pilot/full `*_summary.json` and `run_manifest.json` (contain hashes, no images).
- Whatever the reproduction check says. `DIVERGED` is a finding to report, not something to tune
  away.

## 4. Rules that still apply

- Development/exploratory only. The verdict vocabulary is fixed in the protocol; no confirmatory
  claim, no comparison against the retired Phase 4C.2G locked test.
- Do not edit the protocol after seeing results. A changed protocol changes its SHA-256 and the
  analyzer refuses runs produced under a different protocol file.
- Do not delete or overwrite fold directories that fail verification; move them aside and record why.
