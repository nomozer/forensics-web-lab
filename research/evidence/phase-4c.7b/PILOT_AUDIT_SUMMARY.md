# Phase 4C.7B real pilot audit summary

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
