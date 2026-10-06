# Phase 4C.5A → 4C.5B Handoff: Fusion Shift Diagnostics (cloud → local agent)

The cloud session implemented the diagnostics tool and tested it on synthetic fixtures only.
**No real diagnostics were run. No real finding exists yet.** Every development_real number is
`NOT_MEASURED`. The local agent continues **on the same branch**, runs the tool on the real
Phase 4C.3B / 4C.4B artifacts, writes the report and runs the gates, and only then merges.

## 1. Source binding

| Item | Value |
| :--- | :--- |
| Branch | `claude/elegant-edison-uentky` (cloud-assigned; the requested name `claude/fusion-shift-diagnostics` was not used because the cloud session must push to its assigned branch) |
| Base `origin/main` | `3a0292c840d44ca255acfd6588c3a16f6d61bdd2` (CI run #30 `success`) |
| Implementation commit | pinned in the commit that follows it (see `git log` on this branch; the first commit of this branch) |
| Tool | `scripts/research/diagnose_fusion_shift.py` (`--mode preflight \| analyze \| verify`) |
| Protocol it checks against | `ml/configs/development_robustness_protocol.yaml` (unchanged, SHA-256 LF `b711cdd9…`) |

## 2. What is needed locally (all outside Git, all already produced in 4C.3B / 4C.4B)

| Artifact | Flag | Files read |
| :--- | :--- | :--- |
| Phase 4C.4B robustness runner output (the dir that contains `full/`) | `--robustness-dir` | `full/scope_manifest.json`, `full/original_gate.json`, `full/conditions/<6 conditions>/{condition_receipt.json, features.npz, predictions.csv, images.json}` |
| Phase 4C.3B late-fusion run dir | `--late-fusion-dir` | `run_manifest.json`, `fits/outer_{0..4}/{fold_receipt.json, fold_model.json, predictions.csv}` |
| Visual/DSP ablation run dir | `--ablation-dir` | `fits/visual_dsp_fusion/outer_{0..4}/{fit_receipt.json, model.pt, predictions.csv}`, `shared/dsp_features_receipt.json` |

Not needed and never opened: the image bundle and manifest, the MobileNetV3 weights, a GPU, any
locked-test file. The tool has no flag for them. Every hash and binding is checked first; if any file
is missing the tool raises `MissingArtifactsError` with the exact paths and stops. It never refits
anything and never re-extracts features.

## 3. Commands (repository root; Python 3.12 environment from `ml/requirements.txt`)

```powershell
$Robust   = "data/research/local-artifacts/development_robustness"   # 4C.4B runner --output-dir
$Late     = "<Phase 4C.3B calibrated_late_fusion output dir>"
$Ablation = "<visual/DSP ablation output dir>"
$Out      = "data/research/local-artifacts/fusion_shift_diagnostics"  # new dir; outside Git or under data/
$Common   = @("--robustness-dir",$Robust,"--late-fusion-dir",$Late,"--ablation-dir",$Ablation,"--output-dir",$Out)

# 1) Preflight: bindings, sample/source/fold/feature order, shapes, finite values,
#    reconstruction of every stored prediction, decomposition, scaler = outer-train stats.
python scripts/research/diagnose_fusion_shift.py --mode preflight @Common
#    expect PREFLIGHT_PASS, data_origin development_real, reconstruction max |Δlogit| and |Δp| <= 1e-9
#    (4C.4B reproduced 0.0), decision_mismatches 0, dsp_scaler_matches_outer_train PASS.

# 2) Analysis: writes CSV/JSON/MD/SVG into $Out and copies the aggregate files (not the
#    sample-level CSV) into the evidence folder.
python scripts/research/diagnose_fusion_shift.py --mode analyze @Common `
  --evidence-dir research/evidence/fusion_shift_diagnostics

# 3) Reproduction check: rebuilds every output in memory and compares bytes with $Out.
python scripts/research/diagnose_fusion_shift.py --mode verify @Common
#    expect REPRODUCED, mismatched_files [] (written to $Out/reproduction_check.json)
```

Provenance cross-check after step 2: `diagnostics_summary.json` → `macro_f1.late_fusion_stacked`
must equal the committed 4C.4B values: original `0.6025974305`, jpeg_q75 `0.4401056539`,
jpeg_q50 `0.4026117354` (and `research/evidence/development_robustness/condition_metrics.csv`).

If preflight fails, stop and report its exact message. Do not change tolerances, refit, re-extract,
or edit the robustness outputs. If the failure reveals an implementation defect, reproduce it with a
regression test, record which historical results it could affect, keep history unchanged, and do not
re-run any performance experiment in this phase.

## 4. Outputs

| File | Content | Goes to Git? |
| :--- | :--- | :--- |
| `preflight_receipt.json` | all checks, bindings, tolerances | yes (evidence copy) |
| `dsp_feature_shift.csv` | per condition × outer fold (0–4, then `all`) × class × 16 DSP features: original / transformed / paired-Δ mean, SD, median, q05/q25/q75/q95, IQR, min/max; Δ in outer-training SD units; fraction outside the outer-training original range | yes |
| `dsp_feature_contribution.csv` | per-feature split of Δ DSP contribution (logit units) and effective weight per raw unit | yes |
| `component_shift.csv` | paired Δ of raw and calibrated logits (stacker inputs), visual/DSP contributions, fusion logit/probability, calibrated-visual probability | yes |
| `decision_transitions.csv` | confusion counts, correct↔wrong transitions, label flips, coverage / selective accuracy / high-confidence errors at τ = 0.65 | yes |
| `confidence_histogram.csv` | confidence distribution by outcome | yes |
| `fold_parameters.csv` | temperatures, stacker coefficients, C per fold | yes |
| `diagnostics_summary.json`, `DIAGNOSTICS_REPORT.md`, `figures/*.svg` | summary, auto-report, 5 figures | yes |
| `sample_components.csv` | per-image components (source IDs) | **no** (stays in `$Out`) |
| `diagnostics_receipt.json` | bindings + SHA-256 of every output | stays in `$Out` |

Normalisation: `z = (x − mean_k) / sd_k`, where `mean_k` and `sd_k` (population SD) are computed from
the original DSP features of the outer-training images of fold k. They are never computed from
outer-test images. Preflight requires them to equal the fitted DSP StandardScaler (relative
tolerance 1e-9). A feature that StandardScaler treats as constant gets `NaN`, never a division by
~0. Pooled rows use finite values only (`std_delta_n`).

## 5. Resume and history

- Re-running `analyze` with the same `$Out` and unchanged inputs/tool checks the hashes of every
  output and returns `RESUMED` without rewriting anything.
- Different bindings (other inputs, another protocol, an edited tool) → refused; use a new `$Out`.
- Tampered outputs → refused. Output dirs inside the robustness / late-fusion / ablation dirs → refused.
- An existing evidence file with different content is never overwritten.

## 6. What ran in the cloud and what did not

Ran (synthetic only, Python 3.12.13 Linux venv): see `test_summary.json`. That covers the 35
diagnostics tests, the robustness tests (including the notebook checkout test), the calibrated
late-fusion tests, the full hermetic ML suite, ruff on changed files, and the continuity checker
with its unit tests. Synthetic figures were rendered and inspected (headless Chromium).

Not run: anything on development_real data; Windows; GitHub CI on the pushed branch (check it).

## 7. Closing the phase (local agent)

1. Run section 3; confirm the Macro-F1 cross-check.
2. Write `research/evidence/phase-4c.5b/PHASE_REPORT.md`: preflight status, environment, measured
   post-hoc findings (feature shifts, component shifts, decomposition, transitions, confidence), and
   limits. The decomposition is algebraic, not causal. Do not attribute the failure to DSP without
   evidence: report which term carries Δ fusion per condition and class, and whether the DSP values
   leave the outer-training range. Coverage rising is not quality rising.
3. Update `CURRENT_STATE.md`, add a new top `STATUS_LEDGER.md` entry (≤ 20 lines), and update
   `CODE_INDEX.md` if structure changed. Run
   `python -m pytest ml/tests -m "not requires_research_artifact"`, `pnpm continuity:check`, and
   `node --test scripts/__tests__/continuity-check.test.mjs`.
4. Commit and push this branch; wait for branch CI PASS; merge into `main` with a merge commit (no
   PR, squash, rebase, or force-push); push; verify main CI and ancestry; then delete the fully
   merged local and remote branch.

## 8. Robustness notebook

`notebooks/development_robustness_colab.ipynb` no longer refers to the deleted branch. It refuses
to run unless `EXPECTED_COMMIT` is a full 40-hex SHA (pinned to `3a0292c8…`, the main merge
commit containing the executed 4C.4B code), checks out that commit detached, and asserts
`HEAD == EXPECTED_COMMIT`. It does not need to be re-run.
