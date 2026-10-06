# Phase 4C.4A → 4C.4B Handoff: Development Robustness (cloud → local agent)

The cloud session implemented and synthetically tested the robustness experiment. **No real
preflight, pilot, full run or analysis has been executed.** No robustness result exists yet.
The local agent continues on the same branch, runs the real data, completes the report and the
gates, and only then merges.

## 1. Source binding

| Item | Value |
| :--- | :--- |
| Branch | `claude/keen-knuth-4esvaw` (restarted from `origin/main`; previous use of this name was merged) |
| Base `origin/main` | `18e0b8f05a5c253728b1f6ccc8dadc852a439624` (CI run #28 `success`) |
| Implementation commit | `fffccce42ce6295d2b86e0a357179eb715f34be4` (code, protocol, tests; this HANDOFF line was pinned in the next commit) |
| Protocol | `ml/configs/development_robustness_protocol.yaml` |
| Protocol SHA-256 (CRLF→LF normalised) | `b711cdd9eeb56e9b8167daa508b5b85e400cdb8ec2006272fcc922f0061734d6` |
| Frozen 4C.3B bindings in the protocol | late-fusion `run_manifest.json` `99b95ff2…`, five fold `predictions.csv` hashes, 4C.3B `analysis_summary.json` (LF-normalised) `6a668afe…`, visual/DSP feature commitments `351e4952…` / `582108a2…`, dataset manifest `e1e6b6d2…`, weights `047dcff4…` |

Do not edit the protocol after seeing pilot or full results. Any edit changes its SHA-256; the
analyzer refuses runs produced under a different protocol file.

## 2. Dependencies

Python ≥ 3.12 environment from `ml/requirements.txt` (numpy, scipy, scikit-learn, torch,
torchvision, Pillow, PyYAML). Record versions; the runner writes them (plus Pillow, libjpeg and zlib
versions and the torch device) into `full/scope_manifest.json`. JPEG bytes depend on the libjpeg
build, so a rerun on another machine may give slightly different transformed images; their pixel
hashes are recorded per image in `conditions/<condition>/images.json`.

Use the same device type that built the ablation visual cache (see `device` in
`<ablation_dir>/shared/visual_features_receipt.json`) so the original condition reproduces 4C.3B.

## 3. Artifacts that must exist locally (all outside Git)

| Artifact | Passed as | Checked by |
| :--- | :--- | :--- |
| Development bundle with `manifest_pilot_a_option_p.csv` and the 682 images | `--manifest`, `--data-root` | manifest SHA + 682 image SHA-256 |
| Visual/DSP ablation output dir with `fits/visual_dsp_fusion/outer_{0..4}/{fit_receipt.json,model.pt,predictions.csv}` and `shared/{visual,dsp}_features.pt` + receipts | `--ablation-dir` | receipt hashes, cache SHA, feature commitments |
| Phase 4C.3B late-fusion output dir with `run_manifest.json` and `fits/outer_{0..4}/{fold_receipt.json,fold_model.json,predictions.csv}` | `--late-fusion-dir` | run-manifest SHA, fold hashes, protocol bindings |
| `mobilenet_v3_small-047dcff4.pth` | `--weights-path` | SHA-256 |

If anything is missing the runner raises `MissingArtifactsError` with the exact paths and stops.
It never refits a missing model.

## 4. Commands (repository root, paths only via CLI)

```powershell
$Manifest = "<path to manifest_pilot_a_option_p.csv>"
$DataRoot = "<directory the manifest paths are relative to>"
$Weights  = "<path to mobilenet_v3_small-047dcff4.pth>"
$Ablation = "<visual/DSP ablation output dir>"
$Late     = "<Phase 4C.3B calibrated_late_fusion output dir>"
$Out      = "data/research/local-artifacts/development_robustness"   # or any dir outside the repo
$Common   = @("--manifest",$Manifest,"--data-root",$DataRoot,"--weights-path",$Weights,
              "--late-fusion-dir",$Late,"--ablation-dir",$Ablation,"--output-dir",$Out,"--device","<cpu|cuda: match visual_features_receipt.json>")

# 1) Preflight: artifacts, hashes, sample alignment, fold bindings, frozen-model reproduction.
python -m ml.training.run_development_robustness --mode preflight @Common
#    expect status PREFLIGHT_PASS and max_abs_probability_difference 0.0 (tolerance 1e-9)

# 2) Technical pilot: outer fold 0 only, original + gate + five conditions. Technical check only.
python -m ml.training.run_development_robustness --mode pilot @Common

# 3) Full: original first; transformed conditions only after the full gate is PASS.
python -m ml.training.run_development_robustness --mode full @Common

# 4) Analysis into the evidence folder of the execution phase.
python scripts/research/analyze_development_robustness.py --output-dir $Out `
  --report-dir research/evidence/development_robustness
```

Pilot numbers must not be used to change conditions, endpoints, thresholds or anything else in
the protocol. If the gate fails, stop and report `original_gate.json`; do not loosen tolerances.

## 5. Output directory policy

- Runner output (`$Out`): transformed image copies, per-condition `features.npz`, predictions,
  receipts, gate files. Stays outside Git. The runner refuses an output dir inside the repository
  unless it is under the git-ignored `data/` tree.
- `--no-save-images` skips the transformed image copies (pixel hashes are still recorded).
- Only the analyzer output (aggregate CSV/JSON/MD/figures, no images, no raw features) goes into
  `research/evidence/development_robustness/`. Synthetic analyses are refused there.
- Expected size with image copies: on the order of the dataset size for 5 transformed conditions;
  not measured.

## 6. What ran in the cloud and what did not

Ran (synthetic only): 28 robustness tests (`ml/tests/test_development_robustness.py`), the
calibrated late fusion tests, the full hermetic ML suite, ruff on new files, continuity checker and
its unit tests. See `test_summary.json`.

Not run: real preflight, pilot, full run, analysis, original-reproduction gate on real images,
runtime measurement, CI on the pushed branch. Python 3.12 (CI version) was not used in the cloud
(3.11.15). No robustness metric exists yet; every robustness number is `not measured`.

## 7. Closing the phase (local agent)

1. Run section 4 steps; keep `preflight_receipt.json`, `pilot/scope_summary.json`,
   `full/scope_summary.json`, `full/original_gate.json` (copy the small JSON receipts into
   `research/evidence/development_robustness/`).
2. Write `research/evidence/phase-4c.4b/PHASE_REPORT.md` with the measured primary endpoint, the
   fixed verdict, the gate status, environment and limits (exploratory, source-only CI, not
   confirmatory, locked test not used).
3. Update `CURRENT_STATE.md`, `STATUS_LEDGER.md` (new top entry ≤ 20 lines), `CODE_INDEX.md` if
   structure changed; run `pytest ml/tests -m "not requires_research_artifact"`,
   `pnpm continuity:check`, `node --test scripts/__tests__/continuity-check.test.mjs`.
4. Commit and push the branch; then merge into `main` with a merge commit, verify CI and ancestry,
   and delete the local and remote branch.
