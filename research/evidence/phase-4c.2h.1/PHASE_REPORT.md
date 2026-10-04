# Phase 4C.2H.1 — Official Runner and Colab Pilot Readiness

## Verdict

`READY_FOR_COLAB_TWO_FIT_PILOT`

The official development runner, deterministic self-contained code snapshot, and thin Colab launcher are complete and tested. No official pilot fit or full nested-CV fit was executed in this phase.

## Locked execution design

- Data scope remains 341 development sources / 682 images; locked-test access count is zero.
- Recipes remain exactly `stage1_frozen` and `stage1_frozen_pair_ranking`.
- Seeds are exactly `[42, 1337, 2025]`. This is the prospectively locked reduction from the historical five seeds `[42, 1337, 2025, 3407, 9001]`; it bounds the new nested-CV budget and does not alter prior results.
- The 5 outer x 4 inner grouped-source design yields 120 inner fits and 30 outer refits (150 total).
- Both recipes use the same frozen pretrained backbone and the same BatchNorm policy: features are evaluated once in `eval()` mode and cached per independent image; only the classifier is trainable.
- Each inner fit may select its best epoch only from that inner validation fold's Macro-F1. Each outer refit uses the round-half-up median of its four inner best epochs, trains on all outer-train sources for that fixed epoch count, then predicts the untouched outer fold once.
- Validation and outer predictions consume one image feature at a time. Pair peers, source peer labels, and ground-truth labels are absent from the prediction API.

## Artifacts and resume behavior

Every fit publishes its history, classifier checkpoint, metrics, and receipt atomically; outer refits additionally publish predictions. Receipts bind protocol, code archive, source commit, dataset/manifest/weights, fold membership commitments, recipe, seed, folds, BatchNorm policy, and selected epoch count. Resume accepts only completed fits whose full binding and every artifact byte count/hash validate; invalid completed evidence fails closed.

The pilot consists of two official matrix members at the same fold and seed:

1. `inner__recipe-stage1_frozen__seed-42__outer-0__inner-0`
2. `inner__recipe-stage1_frozen_pair_ranking__seed-42__outer-0__inner-0`

When their exact receipts remain valid, they count as two of the 120 inner fits in a later explicit `--mode all` run.

## Exact code snapshot

- Source commit: `75568d1c03d89e02ad654df75c969924b857ee78`
- Archive: `phase_4c2h_code_75568d1.tar.gz`
- Bytes: `9,509,956`
- SHA-256: `2bd393873be3291084f86a329a34566bd9aa46c0efd8aa9f155e47571b132c18`
- Snapshot manifest SHA-256: `3800d87871c4121ab2368fa330331b53739f632f4ba4fd8257ef5904ec423711`
- Members: 11 excluding the manifest, including exact pretrained weights
- Pretrained weights SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- Backbone state fingerprint: `d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5`

A preliminary package was rejected before pilot because `ml/evaluation/calibration.py` was absent while imported by the package initializer. The corrected archive imports independently and reproduced the same bytes/SHA across two deterministic builds.

## Verification

- Targeted Phase H tests: 12/12 PASS.
- The wider Windows ML suite returned 724 PASS / 1 skipped / 19 failures. Eighteen failures are confined to legacy Bash/Linux permission fixtures or absent external Phase 4C.1 artifacts in this Windows workspace; one historical Stage 2 test uses an over-broad `phase_4c2*.ipynb` glob and misclassifies the already-existing, intentionally separate Phase H notebook as a duplicate. No Phase H test failed, and no full-suite PASS is claimed.
- Production-path fixture: both recipes trained through the real classifier path with canonical weights; shared feature-cache reuse, identical frozen feature state, inner artifacts, post-refit outer predictions, receipt resume, and tamper rejection PASS.
- Exact packaged import audit: protocol matrix, all manifest member hashes, pretrained file binding, and backbone fingerprint PASS from extracted archive rather than working tree.
- Exact packaged development preflight: PASS on all 341 sources / 682 samples. The CPU feature-cache build measured `51.0809s`, produced 1,579,411 bytes with SHA-256 `cb950e86...`, and a second invocation reused the same validated cache/receipt. This is preflight feature extraction, not a classifier fit or pilot runtime.
- Completed official fits: 0 inner, 0 outer. Official pilot runtime: `not measured` pending Colab.
- New inference/training against Phase 4C.2G locked test: 0/0. Phase 4C.2G evidence and `main` remain unchanged.

## Colab handoff

The notebook verifies archive/dataset/manifest/weights, Python/dependency minima, CUDA availability, and disk capacity; stages code and data safely; writes persistent histories/checkpoints/metrics/predictions/receipts to Drive; and defaults to `pilot`. A full 150-fit launch requires both `MODE = 'all'` and an explicit boolean opt-in.

See `PILOT_COLAB_INSTRUCTIONS.md` for exact upload paths, bindings, expected receipts, runtime fields, and resume procedure.
