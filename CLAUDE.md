# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` holds the binding rules for AI agents (Vietnamese). Read it first. Docs in this repo are mostly Vietnamese. Key points are summarized below.

## Session protocol (from AGENTS.md)

- Start every session with the git checks: `pwd`, `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git rev-parse main`, `git remote -v`, `git log --oneline -5`. Run them as separate commands.
- Read, in order: `docs/continuity/CURRENT_STATE.md`, `docs/continuity/CODE_INDEX.md`, `docs/continuity/STATUS_LEDGER.md`, and the latest `research/evidence/<phase>/PHASE_REPORT.md`. Also read any relevant ADR in `docs/adr/`. These files are very large (CODE_INDEX is ~376 KB), so use `offset`/`limit` or grep.
- Never commit to `main`, rewrite its history, force-push or change remotes. Work on a branch (currently `research/independent-cohort-acquisition`).
- Continuity update matrix: any change to model/dataset/metric/test/blocker/next action requires updating `CURRENT_STATE.md`. Adding, removing or renaming a module, schema, contract or CLI also requires `CODE_INDEX.md`. Closing a phase requires a new entry at the top of `STATUS_LEDGER.md`, `research/evidence/<phase>/PHASE_REPORT.md`, and a passing `pnpm continuity:check` before committing.

## Scientific honesty rules (these constrain code, not just docs)

- Never fabricate metrics (accuracy, F1, ECE). Write `not measured` / `not evaluated`.
- Don't auto-download datasets or checkpoints over 50 MB without user confirmation. Don't simulate training without data/GPU.
- No-model state: the system must NOT emit random or simulated 3-class probabilities and must NOT emit `fully_generated` or `ai_edited`. It returns `uncertain` and shows "Model not installed".
- Do not describe interfaces or placeholders as working features.
- Verdict vocabulary is fixed: `no_ai_evidence`, `fully_generated`, `ai_edited`, `uncertain`. Never claim an image is "real".
- Dual-track isolation (ADR-0006): research data/models (`data/research/`, `models/research/`) are strictly separate from product (`data/product/`, `models/product/`). Large data, locked-test sets and checkpoints live outside Git.
- Sealed/preregistered artifacts (configs, split locks, notebooks pinned by hash or SHA) must not be edited in place. See `docs/REPOSITORY_NAMING_MAP.md` for which paths are sealed.

## Commands

Node (pnpm workspace, root `package.json`):

```bash
pnpm install --frozen-lockfile
pnpm dev                       # vite dev server (apps/web)
pnpm typecheck                 # tsc --noEmit in every package
pnpm test                      # all vitest suites + continuity checker tests
pnpm build                     # all packages (web build runs prepare-assets first)
pnpm --filter @forensics/forensics test               # one package
pnpm --filter web exec vitest run path/to/file.test.ts  # one test file
pnpm continuity:check          # validate continuity docs (CI runs it with --base/--head)
pnpm test:continuity           # node --test scripts/__tests__/continuity-check.test.mjs
VITE_BASE_PATH=/forensics-web-lab/ pnpm --filter web build   # GitHub Pages build
```

Python (`ml/`, tested from the repo root, py312, ruff line-length 100):

```bash
pip install -r ml/requirements.txt          # also requirements-dev.txt / requirements-colab.txt
python -m pytest ml/tests -v -m "not requires_research_artifact"   # CI gate
python -m pytest ml/tests/test_option_p_protocol.py -v             # one file
python -m pytest ml/tests -k name                                  # one test
ruff check ml
```

Tests marked `requires_research_artifact` need quarantined local data or checkpoints that are not in Git. CI only collects them and does not run them.

## Architecture

pnpm monorepo (`packages/*`, `apps/*`) plus a Python `ml/` workspace. Everything is client-side by design (zero image egress).

Image flow: `apps/web` `ImageDropzone` → magic-byte/decompression-bomb validation (`packages/shared/src/validation.ts`) → `WorkerController.ts` → `forensics.worker.ts` (Web Worker). In the worker:
- `@forensics/provenance`: EXIF/XMP/IPTC/C2PA.
- `@forensics/forensics`: pure-TS DSP (2D FFT, DCT, noise residual, JPEG/ELA).
- `@forensics/inference`: ONNX Runtime Web session, patch inference, and `fusion-calibrator.ts` for evidence fusion.
- The results are fused into an `AnalysisResult`, rendered by the web UI (heatmap, verdict card) and serialized by `@forensics/report` (JSON schema v1 + printable HTML).

`@forensics/shared` holds the contracts and types that all packages depend on. Schemas are in `docs/schemas/`.

The current shipped demo is a binary (authentic vs AI-edited) FP32 MobileNetV3-small backbone plus a calibrated probe. `apps/web/public/models/mobilenet_v3_small_backbone_fp32.onnx` is tracked in Git. The ONNX Runtime WASM assets are copied into `public/wasm/` by `apps/web/scripts/prepare-assets.mjs` and are not committed. The worker infers its base path from `import.meta.env.BASE_URL` so it works under a GitHub Pages subpath. Browser/ONNX numerical parity was verified against the Python reference with locked per-layer tolerances. The tensor preprocessing (2-pass bicubic, fixed point) in TS must stay bit-compatible with the Python side, so don't "simplify" it.

`ml/` (Python): `datasets/` (manifests, adapters, dedup/group split, `acquire.py` guarded downloader with checksums and safe tar extraction, Option P verification and split freeze, pair-aware loader), `training/` (MobileNetV3 forensics model, `run_learning_curve.py`, losses), `evaluation/` (metrics, calibration, paired source-cluster bootstrap, `locked_test_guard.py`), `export/` (ONNX export and contract validation), `configs/` (YAML plus `validator.py`). Experiments are preregistered. Configs are hash-sealed, and the locked test set is access-controlled through experiment-lock bindings.

`research/evidence/<phase>/` stores per-phase receipts, audits and `PHASE_REPORT.md`. `scripts/` holds the continuity checker, audit and baseline scripts. `notebooks/` holds Colab launchers pinned to commit SHAs.

CI (`.github/workflows/ci.yml`): frozen install, continuity check, typecheck, test and build. A hermetic pytest gate on `ml/tests`. `deploy-pages.yml` publishes the web app to GitHub Pages.
