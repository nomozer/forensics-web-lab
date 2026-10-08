# Phase 4C.7B pilot audit summaries

## 1. Follow-up pilot audit summary: `pilot-20261008T113700Z` (2026-10-08)

> Run: `pilot-20261008T113700Z`<br>
> Binding commit: `d9d99b678053972436032a828056a76a6392fbb5`<br>
> Approved plan SHA-256: `eeace0e57a34f9f3824a3ce1a52bfa374670dca3cc19c668554d62257141632d`<br>
> Audit status: `TECHNICAL_PASS_PENDING_HUMAN_CONTENT_QC`<br>
> This summary is not a human Content QC approval.

- Intake ZIP SHA-256: `3bdb1890d2aa6d34bb2829e159408ef44ab30d630c6f85792fce5fb7ade9e2b4` (6,129,778 bytes). All 34 archive members passed safe anti-path-traversal verification and were extracted into `pilot-20261008T113700Z/`.
- The production `--audit-run` CLI returned `PASS`: 8 pairs, manifest SHA-256 `cf0e4300d1e6f760e82ab76fdd29ffd534b35577003ae8ccf4ef9a9c25799704`, matching receipt and binding.
- Direct ledger count: 8 attempts, 8 accepted, 0 rejected/failed; exactly 2 per stratum, 1 attempt/candidate, `automatic_replacement=false`.
- All 16 images are 512x512 RGB. All 8 masks are 512x512 `L`, binary `{0,255}`, with pixel counts matching coordinate rectangles down to the exact pixel.
- Outside-mask mean L1: `0.000000` across all 8 pairs (enforced by construction via mask compositing; not evidence of raw diffusion preserving outside regions). Inside-mask mean L1: 11.08–76.14; difference std dev: 3.62–59.68.
- Agent content observations (advisory only; not replacing human decision):
  - `IND_COCO_SD2_002` (cabinets remodel): plausible navy cabinetry with stainless hood; recommendation `NEEDS_REVIEW`.
  - `IND_COMMONS_SD2_001` (Option A headland removal): 100% headland/cliff removed, smooth sea/horizon; coastline town infilled as natural cliff consistent with accepted risk; recommendation `NEEDS_REVIEW`.
  - `IND_COCO_SDXL_041` (chandelier): high-detail chandelier rendered, valance contact subtle; horizontal crystal/vault cut-off across ceiling at y=155; recommendation `NEEDS_REVIEW`.
  - `IND_COCO_SDXL_002` (bread tomato), `IND_COMMONS_SD2_002` (suitcase), `IND_COMMONS_SDXL_001` (bird): complete object omission (0 requested items generated); recommendation `REJECT`.
  - `IND_COCO_SD2_001` (pan to clock): murky metallic blob with square seam; `IND_COMMONS_SDXL_003` (column): psychedelic soda-bottle pillar with neon reflections and severed railing; recommendation `REJECT`.
- Review dossier: self-contained HTML contact sheet with embedded images, SVGs, difference maps, and per-pair review cards at `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/content_qc_contact_sheet.html`.
- Gate: Human Content QC remains strictly `PENDING_CONTENT_QC` until user decides per-pair.

---

## 2. Historical pilot audit summary: `pilot-20261007T132003Z` (2026-10-07)

> Run: `pilot-20261007T132003Z`  
> Binding: `7d2eea4027e2a17b51e6665ff81d481e4e333d48`  
> Audit status: `TECHNICAL_PASS_PENDING_HUMAN_CONTENT_QC`  
> This summary is not a human Content QC approval.

- Intake ZIP SHA-256: `1fcd1de57373c7250583d87a4a4050a91d918b8bd62ec105d4aab737d340053d` (6,112,677 bytes). Its 34-member inventory passed safe-path checks and was extracted without overwriting an existing directory.
- The production `--audit-run` command from a clean detached LF checkout at the exact binding commit returned `PASS`: 8 pairs and manifest SHA-256 `caa0766addff4d87d9d11206f1e398a29ca04c58f0013866fac749fac09bc416`.
- Direct ledger count: 8 attempts, 8 accepted, 0 rejected/failed, with no rejection reasons; exactly 2 accepted pairs in each of `coco_sd2`, `coco_sdxl`, `commons_sd2`, and `commons_sdxl`.
- All 16 images are 512x512 RGB. All 8 masks are 512x512 `L`, binary `{0,255}`, and within their declared area bracket. File inventories and all recorded SHA-256 bindings match receipts, manifest, and ledgers.
- Detector calls remain 0; independent performance remains `NOT_MEASURED`. Full 400-pair acquisition, training, and evaluator execution were not run.
- Agent-only visual screening found material concerns on all 8 pairs: 7 likely-reject recommendations for prompt/operation mismatch or severe synthesis artifacts, and 1 inconclusive removal sample whose intended removed object is not evident. Canonical Content QC remains `PENDING_CONTENT_QC` for every pair.
- Follow-up diagnosis measured mean outside-mask L1 of 3.808–9.311 on all 8 pairs, above the locked maximum 0.5. It found no model-parameter mismatch and no evidence of crop/resize coordinate drift; the demonstrated content cause is missing target registration plus generic prompts/random normalized-canvas masks. See `PILOT_CONTENT_DIAGNOSIS.md`.
- Gate: technical intake is complete, but the pilot is **not ready to unlock the full acquisition**. A human must review the external contact sheet and record final per-pair decisions.

Detailed images, masks, ZIP, logs, and receipts remain outside Git under `data/research/local-artifacts/phase-4c.7b/pilot-20261007T132003Z/`; the agent's detailed per-pair report is the sibling file `pilot-20261007T132003Z_AGENT_PILOT_AUDIT_REPORT.md`.
