# Phase 4B.4 Report: Executable Evidence Repair and Shortcut-Control Gate

> **Phase**: Phase 4B.4 — Executable Evidence Repair and Shortcut-Control Gate  
> **Status**: **`COMPLETED_AND_VERIFIED`**  
> **Execution Mode**: Strict Offline (0 model downloads, 0 training runs, 0 locked-test evaluations)  
> **Repository**: `nomozer/forensics-web-lab`  
> **Working Branch**: `feat/production-ai-image-forensics`  
> **Starting Git HEAD**: `054b97f` (live measured)  
> **Main Branch**: `460f6d5` (preserved, untouched)  
> **Verified Invariant**: Zero machine-local absolute links (`D:\`, `C:\`, `file:///`) in committed repository documents.  

---

## 1. Executive Summary

Phase 4B.4 transitions the research evidence framework from static documentation assertions to an executable, live-measured verification pipeline. It formally eliminates potential resolution shortcuts in the TGIF dataset, seals a cross-process deterministic variant sampler, and recalibrates scientific claims regarding archive proof levels, server ETags, and pseudoreplication.

Key accomplishments:
1. **Live Evidence Reproducer**: Created `scripts/reproduce-phase-4b3-evidence.py` and accompanying unit tests (`ml/tests/test_evidence_reproducer.py`). Measures Git state, environment versions, archive byte counts, SHA-256 digests, and manifest invariants directly from the filesystem.
2. **Three-Tier Archive Integrity**: Formally classified archive verification into (A) `local_artifact_integrity`, (B) `transport_completeness`, and (C) `upstream_identity`. Calibrated claim from absolute mathematical proof to rigorous empirical verification.
3. **ETag Discrepancy Calibration**: Classified differing WebDAV PROPFIND and HTTP GET ETags as `inferred-server-behavior` (`remote_metadata_differs` rather than `remote_content_changed`), avoiding unverified assertions about server-side Apache directives.
4. **Historical SHA-256 Audit**: Verified the provenance of `sd2-sp_validation.tar.gz`, establishing the verdict `documentation-typo` based on receipt records in commit `7d33072`.
5. **Deterministic Variant Offset**: Replaced Python's `hash()` with a 64-bit integer derived from `SHA-256(source_id)`, verified invariant across differing `PYTHONHASHSEED` values.
6. **Resolution Shortcut Removal**: Implemented resolution-matched sampling (`native ↔ native`, `512 ↔ 512`, `1024 ↔ 1024`), balanced bbox/segm cycling, and enforced an identical preprocessing pipeline for authentic and edited classes.
7. **Preregistration Reseal**: Resealed Phase 4C preregistration (`docs/PHASE_4C_PREREGISTRATION.md`, `ml/configs/pilot_a_binary_preregistered.yaml`) with new config hash `54140d42...`, formally marking Phase 4B.3 hash `839531a7...` as superseded.

---

## 2. Archive Proof Classification & Server Behavior

### 2.1 Three-Tier Proof Hierarchy
To uphold scientific honesty, archive claims are classified into three distinct evidentiary tiers:

* **Tier A — `local_artifact_integrity` (Status: `verified`)**:
  Live SHA-256 and byte count measurements over all four downloaded tarballs match `download-receipt.json` byte-for-byte. Confirms zero bit rot or post-acquisition alterations.
* **Tier B — `transport_completeness` (Status: `verified`)**:
  Transfers matched HTTP `Content-Length`. Safe extraction completed without archive errors or path traversal. 100% of images (6,156 uncompressed PNGs) decoded successfully via PIL. Cardinality and pairs match expected structures.
* **Tier C — `upstream_identity` (Status: `independent-verification-unavailable`)**:
  Upstream TGIF authors published files on Nextcloud without an official cryptographic ledger (SHA-256 / MD5). Independent proof that the archives match the author's release at creation is unavailable until upstream publishes authoritative hashes.

> **Calibrated Statement**:  
> *"Local artifact integrity và transport completeness đã được xác minh; upstream identity chưa thể kiểm chứng độc lập do nguồn không công bố checksum chuẩn."*

### 2.2 ETag Interpretation Correction
* **Measured ETags**:
  * `orig_validation.tar.gz`: WebDAV `"cf53e5e4922ca59f812d4d5462cfb9b8"` vs HTTP GET `"33420b-62589574d6089"`
  * `sd2-sp_validation.tar.gz`: WebDAV `"44b9e62f224dd312f59d9a5bd79d1171"` vs HTTP GET `"69420b-62589574d6089"`
* **Verdict**: `inferred-server-behavior` (`remote_metadata_differs: true`, `remote_content_changed: false`). Consistent with differing endpoint-specific ETag generation algorithms between WebDAV PROPFIND and the Apache HTTP static file module.

### 2.3 Checksum History Audit (`sd2-sp_validation.tar.gz`)
* **Verdict**: `documentation-typo`
* **Evidence**: Both `download-receipt.json` and `acquisition-binding.json` generated at acquisition time in commit `7d33072` recorded the exact current hash (`bd9eb4399f60166a...`). The differing hash in Phase 4B.2 `PHASE_REPORT.md` (`3b3ba9f...`) was an errant copy-paste error during report composition.

---

## 3. Manifest Audit & Image Cardinality

All metrics measured live via `scripts/reproduce-phase-4b3-evidence.py`:

| Metric / Invariant | Live Measured Value | Expected Assertion | Status |
| :--- | :---: | :---: | :---: |
| Base Manifest Rows (`option_p`) | 684 | 684 | **PASS** |
| Unique `source_id` | 684 | 684 | **PASS** |
| Unique `instance_id` | 684 | 684 | **PASS** |
| Partition: `development_train` | 250 | 250 | **PASS** |
| Partition: `inner_validation` | 91 | 91 | **PASS** |
| Partition: `locked_test` | 343 | 343 | **PASS** |
| Learning Curve: $N=50$ | 50 | 50 | **PASS** |
| Learning Curve: $N=100$ | 100 | 100 | **PASS** |
| Learning Curve: $N=250$ | 250 | 250 | **PASS** |
| Nested Invariance ($N=50 \subset N=100 \subset N=250$) | True | True | **PASS** |
| Cross-Partition Overlap (Dev/Val/Test) | 0 / 0 / 0 | 0 / 0 / 0 | **PASS** |
| Locked Split Seal | `519e7a0e6815e781...` | Match | **PASS** |
| Variant Pairs Manifest Rows | 4,104 | 4,104 | **PASS** |
| Authentic PNG Images on Disk | 2,052 | 2,052 | **PASS** |
| Edited PNG Images on Disk | 4,104 | 4,104 | **PASS** |
| Total Uncompressed PNG Images | 6,156 | 6,156 | **PASS** |

---

## 4. Sampler Determinism & Shortcut Elimination

### 4.1 Cross-Process Deterministic Variant Offset
* Replaced Python's randomized `hash()` with SHA-256 derived 64-bit integer:
  ```python
  stable_source_offset = int.from_bytes(
      hashlib.sha256(source_id.encode("utf-8")).digest()[:8],
      "big"
  )
  ```
* Verified via `test_stable_offset_across_pythonhashseeds`: Subprocesses spawned under `PYTHONHASHSEED=0` and `PYTHONHASHSEED=999999` yielded identical sample sequences across 4 epochs.

### 4.2 Resolution-Matched Sampling & Balanced Cycling
Physical inspection of disk assets confirmed authentic images exist at native, 512, and 1024 resolutions, while edited inpainting variants exist on native canvas:
* **Resolution Matching**: Enforced `pair[0].resolution_bucket == pair[1].resolution_bucket` for 100% of sampled pairs.
* **Balanced Formula**:
  $$\text{resolution\_idx} = (\text{epoch} + \text{stable\_source\_offset}) \pmod 3$$
  $$\text{edit\_type\_idx} = \left(\left\lfloor \frac{\text{epoch}}{3} \right\rfloor + \text{stable\_source\_offset}\right) \pmod 2$$
  $$\text{edited\_variant\_idx} = \text{edit\_type\_idx} \times 3 + \text{resolution\_idx}$$
  $$\text{authentic\_variant\_idx} = \text{resolution\_idx}$$
* **Symmetric Preprocessing**: Both classes pass through identical $224 \times 224$ bicubic resize and ImageNet tensor normalization with no conditional branching.

### 4.3 Calibrated Pseudoreplication Claim
> *"Pair-aware sampling kiểm soát đóng góp gradient theo source và giảm pseudoreplication; suy luận thống kê tiếp tục sử dụng `source_id` làm đơn vị độc lập. Các variant cùng source vẫn là dữ liệu tương quan và không làm tăng cỡ mẫu khoa học. Số lượng mẫu độc lập $N$ luôn là số unique `source_id`."*

---

## 5. Resealed Phase 4C Preregistration

* **Resealed Document**: `docs/PHASE_4C_PREREGISTRATION.md`
* **Resealed Config**: `ml/configs/pilot_a_binary_preregistered.yaml`
* **New Config SHA-256**: `54140d42485384436052befe58ac46775123e1cb009ea3a11d7389470359fd3b`
* **Superseded Hash**: `839531a700b211e5adc4a145e811c4533dbb45a044b63e816c474a462e88ddd1` (`superseded-by-phase-4b.4`)
* **Model & Licensing Fields**:
  * `codeLicense`: `BSD-3-Clause` (verified PyTorch / torchvision)
  * `pretrainedWeightSource`: Official torchvision CDN
  * `pretrainedWeightTermsStatus`: `unverified` (no standalone weights license separate from ImageNet non-commercial research use)
  * `trainingDatasetProvenance`: ImageNet-1K (ILSVRC 2012)
  * `researchUseDecision`: Approved for non-commercial academic research only
  * `compute_budget`: Status `estimated-not-measured`, `benchmarkRequired: true`

---

## 6. Verification Suite Results

| Test / Gate Suite | Command | Exit Code | Result | Details |
| :--- | :--- | :---: | :---: | :--- |
| Vitest Workspace | `pnpm -r run test` | 0 | **PASS** | 57 passed across 6 packages |
| Continuity Unit Suite | `node --test scripts/__tests__/continuity-check.test.mjs` | 0 | **PASS** | 13 passed |
| Python ML Suite | `ml/.venv/Scripts/python -m pytest ml/tests -v` | 0 | **PASS** | 90 passed in 10.35s |
| Config Validator | `ml/.venv/Scripts/python -m ml.configs.validator --validate-all` | 0 | **PASS** | 4 configs validated |
| Dataset Registry | `ml/.venv/Scripts/python -m ml.datasets.acquire --validate-registry` | 0 | **PASS** | 7 datasets verified |
| Evidence Reproducer | `ml/.venv/Scripts/python scripts/reproduce-phase-4b3-evidence.py --verify` | 0 | **PASS** | All live invariants verified |
| Production Build | `pnpm build` | 0 | **PASS** | TypeScript + Vite bundle exit 0 |

---

## 7. Strict Resource Accounting

| Resource Category | Quantity | Status | Verification |
| :--- | :---: | :---: | :--- |
| Model Weights Downloaded | **0 bytes** | Zero Network | Verified via network logs |
| Training Runs Executed | **0** | No Training | Verified via process audit |
| Locked-Test Evaluations | **0** | Sealed | `LockedTestAccessGuard` unbroken |
| Main Branch Modifications | **0 commits** | Untouched | Main at `460f6d5` |
| Remote Push Operations | **0 pushes** | Local Only | Branch remains unpushed |

---

## 8. Proposed Phase 4C.0 Transition

Phase 4C.0 is structured as a gated preliminary smoke verification before any multi-seed training runs:

1. **Pretrained Weights Download Gate**: Request user approval to download official torchvision MobileNetV3-Small weights (~10.3 MB).
2. **Benchmark Smoke Run**: Execute a single smoke run ($N=50$, seed 42) on `development_train` (50 authentic : 50 edited) to empirically measure epoch duration, memory consumption, loss convergence, and DataLoader throughput.
3. **Stage 0 Baselines**: Evaluate Dummy, Metadata-only, and DSP-only classifiers on `development_train` and `inner_validation`.
4. **Learning Curve Execution**: Proceed with full 3-size, 5-seed runs ($N \in \{50, 100, 250\}$) only after the smoke run passes all integrity gates.
