# Phase 4C.6A → 4C.6B Handoff: Controlled DSP Augmentation (cloud → local agent)

The cloud session locked the protocol, implemented and synthetically tested the experiment, and
built the package. **No real fit, prediction or analysis has been run. Every real performance number
is `NOT_MEASURED`.** The local agent continues on the same branch.

## 1. Source binding

| Item | Value |
| :--- | :--- |
| Branch | `claude/elegant-edison-uentky`. This name is cloud-assigned; `research/dsp-augmentation` could not be used. The earlier use of this name was merged at `9ed4ea6` and deleted; the branch was restarted from that commit, and no history was reused. |
| Base `origin/main` | `9ed4ea6424c32f802a6228d2631d33b49a4a125a` (CI run #34 `success`) |
| Functional commit | pinned with the package receipt in the follow-up commit (section 2) |
| Protocol | `ml/configs/dsp_augmentation_protocol.yaml` (SHA-256 LF-normalised in `package_receipt.json`) |
| Predecessors bound by the protocol | late-fusion protocol `85dcb242…`, robustness protocol `b711cdd9…`, robustness run bindings `b5b393b3…`, original gate `cc675f4c…`, six condition receipts (from the committed 4C.5B preflight receipt) |

## 2. Package and component bindings

`package_receipt.json` (same folder): source commit, archive SHA-256, `SNAPSHOT_MANIFEST.json`
SHA-256, and the SHA-256 of every member (19 files: 3 protocols, requirements, 11 `ml/training`
modules, 4 analysis scripts; no weights are needed). gzip bytes can differ between zlib builds; the
snapshot-manifest hash cannot. The Colab notebook pins that manifest hash and refuses an empty pin.

## 3. Execution

Follow `EXECUTION_GUIDE.md`: artifacts, schemas, the exact CLI for preflight → pilot → full →
analyze → verify, the output directory, the resume policy and the local gates.

Fit budget (locked, checked by runner and analyzer):

| | Pilot (fold 0) | Full |
| :--- | ---: | ---: |
| Baseline reconstruction | 61 | 305 |
| Augmented DSP inner / refit / temperature / stacker | 28 / 1 / 1 / 1 | 140 / 5 / 5 / 5 |
| **Total** | **92** | **460** |

Feature extractions, image reads and backbone forward passes are all 0.

## 4. What ran in the cloud and what did not

Ran (synthetic only, Python 3.12.13 Linux venv): see `test_summary.json`. That covers the targeted
tests, the hermetic ML suite, ruff, `git diff --check`, the continuity checker and its unit tests,
pnpm typecheck/test/build, the package build, the staged-package run, and figure rendering.

Not run: any development_real preflight, pilot, full run, analysis or verify; Windows execution.
The fit-time runtime of the real cohort was not measured.

## 5. Closing the phase (local agent)

1. Run the guide. Keep the small receipts (`preflight_receipt.json`, `pilot_summary.json`,
   `full_summary.json` without sample rows) next to the analyzer output in
   `research/evidence/dsp_augmentation/`.
2. Write `research/evidence/phase-4c.6b/PHASE_REPORT.md`. It must cover:
   - the primary result with its locked verdict, and the original-image trade-off;
   - every stress condition, reported whatever the direction;
   - the gates, the environment and the limits from the protocol.
3. Update `CURRENT_STATE.md`, add a new top entry (≤ 20 lines) to `STATUS_LEDGER.md`, and update
   `CODE_INDEX.md` if the structure changes.
4. Run the tests, `pnpm continuity:check` and `git diff --check`, then commit and push this branch.
   Wait for branch CI to PASS.
5. Merge into `main` with a **merge commit** (no PR, no squash, no rebase, no force-push, no
   protection bypass): `git checkout main && git merge --no-ff claude/elegant-edison-uentky && git push origin main`.
   Verify main CI PASS and `git merge-base --is-ancestor <branch head> origin/main`. Then delete the
   fully merged branch: `git branch -d claude/elegant-edison-uentky` and
   `git push origin --delete claude/elegant-edison-uentky`.
