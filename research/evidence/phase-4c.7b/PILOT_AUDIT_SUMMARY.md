# Phase 4C.7B pilot audit summaries

## 1. Follow-up pilot audit summary: `pilot-20261008T113700Z` (2026-10-08)

> Run: `pilot-20261008T113700Z`<br>
> Binding commit: `d9d99b678053972436032a828056a76a6392fbb5`<br>
> Approved plan SHA-256: `eeace0e57a34f9f3824a3ce1a52bfa374670dca3cc19c668554d62257141632d`<br>
> Audit status: `TECHNICAL_PASS_PENDING_HUMAN_CONTENT_QC`<br>
> This summary is not a human Content QC approval.

- **Intake & Package Verification**: ZIP SHA-256 `3bdb1890d2aa6d34bb2829e159408ef44ab30d630c6f85792fce5fb7ade9e2b4` (6,129,778 bytes). All 34 archive members passed safe anti-path-traversal verification and were extracted into `pilot-20261008T113700Z/`. Original ZIP preserved untouched.
- **Production CLI Audit**: Returned `PASS` (`python scripts/research/run_cohort_acquisition.py --audit-run ...`), manifest SHA-256 `cf0e4300d1e6f760e82ab76fdd29ffd534b35577003ae8ccf4ef9a9c25799704` matching receipt and binding.
- **Ledger Semantics Clarification**: In `attempt_ledger.jsonl`, status `ACCEPTED` represents strictly **Technical QC Acceptance** (Step D/E in pipeline: 512x512 RGB canvas, binary L mask, non-blank, outside L1=0, inside L1 >= 3.0, non-blank image std >= 5.0, disjointness verified). Canonical **Human Content QC** is tracked separately in `provenance_ledger.jsonl` and `run_receipt.json` as `PENDING_CONTENT_QC`.
- **Exact Pixel Metrics (Canonical Pipeline Formula)**:
  - Outside-mask mean L1: exactly `0.000000` across all 8 pairs (enforced by construction via binary mask compositing `Image.composite(gen, auth, mask)`; not evidence of raw diffusion preserving outside regions).
  - Inside-mask mean L1: verified across ZIP, extracted files, and HTML Base64 embeds down to 1e-6:
    - `IND_COCO_SD2_001`: `37.228664`
    - `IND_COCO_SD2_002`: `44.094597`
    - `IND_COCO_SDXL_041`: `30.664345`
    - `IND_COCO_SDXL_002`: `16.786127`
    - `IND_COMMONS_SD2_001`: `54.337215`
    - `IND_COMMONS_SD2_002`: `30.735847`
    - `IND_COMMONS_SDXL_001`: `11.076676`
    - `IND_COMMONS_SDXL_003`: `76.143341`
- **Coordinate-Verified Visual Observations (Advisory Agent Screening)**:
  - `IND_COCO_SD2_001` (clock replaces pan): Murky metallic/glass blob without formed clock face; harsh square boundary seam at $y=95, x=280$ with cooler wall tone; recommendation `REJECT`.
  - `IND_COCO_SD2_002` (cabinet remodel): **Only upper cabinets and range hood** ($y \in [75, 323], x \in [95, 415]$) remodeled to matte navy blue; **lower cabinets and counter** ($y > 323$) remain 100% authentic honey-oak wood; minor synthetic gloss on right cabinet; recommendation `NEEDS_REVIEW`.
  - `IND_COCO_SDXL_041` (chandelier): Crystal pendants terminate neatly above $y=140$ ($y \in [138, 140], x \in [180, 420]$) and are **not cut off**; seam at $y=155$ is the newly painted smooth white ceiling plaster/soffit cutting horizontally into darker authentic textured drywall; valance contact at $y=155, x \in [12, 25]$ is subtle; recommendation `NEEDS_REVIEW`.
  - `IND_COMMONS_SD2_001` (headland removal): Original pine headland removed, but **not converted 100% to sea**; SD2 generated a newly synthesized coastal landscape: steep green grassy hillside ($x \in [380, 512], y \in [305, 480]$), sea stacks ($x \in [380, 430], y \in [340, 440]$), low reefs ($x \in [340, 390], y \in [450, 480]$), and foreground rocky mound ($x \in [200, 310], y \in [470, 512]$); distant town buildings in mask replaced by dark rock; recommendation `NEEDS_REVIEW`.
  - `IND_COCO_SDXL_002` (bread tomato), `IND_COMMONS_SD2_002` (suitcase), `IND_COMMONS_SDXL_001` (bird): Complete object omission (0 requested items generated; infilled with bread crumbs, cobblestones, and cloudy sky); recommendation `REJECT`.
  - `IND_COMMONS_SDXL_003` (column): Psychedelic soda-bottle pillar with neon reflections and severed park railing at $x=170, 380$; recommendation `REJECT`.
- **Technical Diagnosis (Code & Model Behavior)**:
  - *Proven Cause (Boundary Seams)*: Zero-feathering 1-bit compositing (`Image.composite`) with axis-aligned boxes cutting continuous scene geometry causes immediate 1-pixel color/texture steps.
  - *Proven Cause (Contact Sheet Bug)*: Fixed previous script trying to read non-existent `att['qc_details']`. Contact sheet now displays exact computed inside L1 values.
  - *High-Probability Hypotheses (Small Insertion Deficits)*: Latent space downsampling (8x) gives tiny spatial capacity (e.g., $13 \times 11$ latent px for tomato, $18 \times 14$ for bird); combined with strong background prompt conditioning ("bread", "cobblestones", "cloudy sky"), UNet inpainting strongly prioritizes context infilling over object synthesis.
- **Review Dossier**: Self-contained HTML contact sheet at `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/content_qc_contact_sheet.html`.
- **Gate**: Human Content QC remains strictly `PENDING_CONTENT_QC` until user decides per-pair.

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
