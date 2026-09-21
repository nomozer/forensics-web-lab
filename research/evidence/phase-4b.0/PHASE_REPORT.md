# Phase 4B.0 Report: TGIF Masks Live Acquisition Smoke and Inventory Audit

## 1. Executive Summary

| Attribute | Value |
| :--- | :--- |
| **Phase** | **4B.0 — TGIF Masks Live Acquisition Smoke** |
| **Starting Commit** | `8de2151` |
| **Implementation Snapshot Commit** | `84aa127` |
| **Main Branch Ref** | `460f6d5` (Strictly untouched) |
| **User Approval Scope** | `tgif-masks` ONLY |
| **Requested URL** | `https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?path=%2F&files=masks` |
| **Final URL (after redirect)** | `https://cloud.ilabt.imec.be/public.php/dav/files/xEeAzrY7ES9KA8o/masks?accept=zip` |
| **Redirect Count** | 1 (strictly confined to `cloud.ilabt.imec.be`) |
| **Expected Bytes** | 42,362,470 bytes |
| **Downloaded Content Bytes** | **42,327,429 bytes** (~40.37 MiB) |
| **Hard Network Ceiling** | **67,108,864 bytes** (64 MiB, ceiling respected) |
| **Archive SHA-256** | `62c89a65441a35e6bd10d49abec8f2cad9ab028e91edf3d72b344ec29bfa0fe9` |
| **Uncompressed Extracted Bytes** | **141,559,934 bytes** (~135.0 MiB) |
| **Exact Mask File Count** | **31,238 PNG files** |
| **Valid Files** | **31,238** (100%) |
| **Invalid / Corrupted Files** | **0** (0%) |
| **Unique Source IDs** | **2,242** MS-COCO image sources |
| **Model Weights Downloaded** | **0 bytes** |
| **Training Runs Executed** | **0** |
| **Verdict** | **PASS** |

---

## 2. Strict Safety & Boundary Adherence

1. **User Approval Enforcement**:
   User approval was granted strictly for `tgif-masks`. Live acquisition of `tgif-orig`, `tgif-sd2-sp`, and `GenImage` remained strictly locked. Downloader CLI actively blocks any unapproved component.
2. **Ceiling Verification**:
   Network traffic was monitored before and during streaming. The downloaded payload of `42,327,429` bytes remained well beneath the hard network ceiling of `67,108,864` bytes (64 MiB).
3. **Hostname Restriction**:
   Initial and redirect hosts were restricted strictly to `cloud.ilabt.imec.be`. Single redirect was accepted within domain.
4. **Staging & Zip Slip Protection**:
   Archive was downloaded to `.part`, atomically staged, tested via `zipfile.ZipFile.testzip()`, and extracted into `.staging` with explicit path traversal, absolute path, and symlink checks before moving to `data/research/tgif/masks/`.
5. **Git Binary Isolation**:
   All masks, archives, `.part`, `.staging`, and JSONL manifest files reside under `data/research/tgif/` which is ignored by `.gitignore`. Verified zero binary files present in `git status`.

---

## 3. Mask Component Inventory & Cardinality

Upstream Nextcloud dynamic zip packages three split tar archives:
* `masks_testing.tar.gz` (5,272,350 bytes, SHA-256 `420f010647cd670545db894b02f17deaa1e1fc4ec1a802b5cd9e7a530fc89ce8`)
* `masks_training.tar.gz` (31,758,802 bytes, SHA-256 `2bf8e2cbfa824b79f5c25eab1acbbae3d8a54c8e35dc2a42d655afd873fb12c7`)
* `masks_validation.tar.gz` (5,295,371 bytes, SHA-256 `dcf94bbf535e5c13a01dbf61cd5551070d23ffa43e175cfb0b7b8dbf9646b438`)

### Exact File Counts by Split
* `train`: **24,400** files across 54 categories
* `val`: **3,410** files across 13 categories
* `test`: **3,428** files across 14 categories
* **Total**: **31,238** mask PNG files

### Mask Types Breakdown
* `bbox`: **12,495** bounding box masks
* `segm`: **12,495** segmentation masks
* `generic_mask`: **6,248** masks (resolution variants)

### Image Properties
* Color Modes: `L` (grayscale single-channel): 24,990 files; `RGB` (3-channel): 6,248 files.
* Pixel Values: Predominantly `{0, 255}` binary masks.
* Resolution Variants: 9,372 files at `1024` width, 9,372 files at `512` width, 12,494 native resolution files.

### Filename & Source ID Mapping
* Filename convention follows `^<source_id>_mask_<type>[_<resolution>][.png_ps_mask].png$`.
* `source_id` is successfully and losslessly extractable via leading digits, zero-padded to 12 digits matching standard MS-COCO naming.
* `mask_count`: Transitioned from `estimated` to `verified` (**31,238**).
* `filename_convention`: Transitioned to `verified`.
* `source_id_extraction_method`: Transitioned to `verified`.

---

## 4. Manifest & Lineage

* Local manifest: `data/research/tgif/manifests/masks-manifest.jsonl` (31,238 records, kept outside Git).
* Evidence summary: `research/evidence/phase-4b.0/mask-manifest-summary.json`.
* Acquisition receipt: `data/research/tgif/acquisition-receipt.json`.

---

## 5. Scientific Claims Status

* **Detection Metrics**: `not evaluated` (no model trained, no evaluation performed).
* **Scope**: Phase 4B.0 proves only the live data acquisition pipeline, safety ceiling enforcement, and mask inventory audit. It does NOT make any claim regarding detection accuracy or localization performance.
