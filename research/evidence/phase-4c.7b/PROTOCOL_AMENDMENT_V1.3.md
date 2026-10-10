# Protocol Amendment v1.3: Wikimedia Commons Replacement for Unsplash Lite & Stratum Realignment

> **Status**: LOCKED (Pre-Acquisition Amendment)<br>
> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Effective Date**: 2026-10-07<br>
> **Supersedes**: Protocol Amendment v1.2 (`PROTOCOL_AMENDMENT_V1.2.md`) specifically regarding the Unsplash Lite authentic source channel.<br>
> **Preserves**: All scientific invariants from Phase 4C.7A / Amendment v1.2 ($N_{\text{target}}=400$ pairs, 440 buffer pool, binary classification, 2 candidate recipes, 5 outer-fold models, 6 transform conditions, arithmetic mean Macro-F1, paired cluster bootstrap, detector isolation invariant).

---

## 1. Rationale for Source Replacement

Under Protocol Amendment v1.2, the 200 non-COCO authentic candidate pairs were planned to be drawn from the Unsplash Lite dataset (via a Hugging Face mirror).
A legal and licensing review of the Unsplash Dataset Terms (§2.A, §3.A–B) revealed:
1. **Unsplash Dataset Terms Scope**: The terms explicitly grant rights only to download, store, and utilize the dataset for *internal machine learning training*.
2. **Derivative Redistribution Prohibition**: The dataset terms expressly prohibit disseminating, redistributing, or publishing dataset images or derivative works without written permission. Creating, editing, and sharing an independent evaluation cohort for peer review and replication is therefore not covered by the Unsplash Dataset Terms.
3. **Attribution & Provenance**: The Unsplash Lite mirror lacked structured proof of individual photographer permission for third-party edited publication.

To maintain strict scientific and legal integrity, Protocol Amendment v1.3 replaces the Unsplash Lite source channel with **Wikimedia Commons** photographic works.

---

## 2. Source Channel Specification

### 2.1. Source 1: COCO 2017 Clean Subset (200 pairs / 220 buffer candidates)
- **Origin**: COCO 2017 validation split (`captions_val2017.json` / `instances_val2017.json`).
- **Disjointness**: 100% disjoint from all 684 historical Option P source IDs (4,316 candidate images available).
- **Provenance Verification**: Every candidate is cross-verified via Flickr oEmbed API against its official Flickr photo ID (`https://www.flickr.com/services/oembed/?url=https://flic.kr/p/...`).
- **Licensing**: Must carry an active Creative Commons Attribution license (CC BY 2.0, CC BY-SA 2.0, CC BY 3.0, CC BY-SA 3.0, CC BY 4.0, CC BY-SA 4.0) confirmed by Flickr oEmbed. NoDerivs (`by-nd`, `by-nc-nd`) is strictly excluded.

### 2.2. Source 2: Wikimedia Commons Quality Photographic Works (200 pairs / 220 buffer candidates)
- **Origin**: Wikimedia Commons `Category:Quality_images` and `Category:Featured_pictures_on_Wikimedia_Commons` via the official MediaWiki Action API (`action=query&prop=imageinfo&iiprop=url|size|extmetadata|mime|sha1`).
- **Content Type**: Photographic bitmap images (`image/jpeg`, `image/png`), minimum resolution $512 \times 512$.
- **Provenance Verification**: Creator attribution (`Artist`), source page URL (`https://commons.wikimedia.org/wiki/File:...`), origin ID (`commons:<pageid>`), original timestamp (`DateTimeOriginal`), and publication timestamp (`DateTime`) extracted directly from `extmetadata`.
- **Licensing**: Verified Creative Commons license permitting derivative works and redistribution (CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA versions 2.0, 2.5, 3.0, 4.0). ND licenses strictly excluded. Usage policy requiring attribution and ShareAlike (where applicable) is enforced.

---

## 3. Realignment of the 2x2 Orthogonal Strata

The four strata are formally renamed to reflect the verified authentic sources:
1. `coco_sd2`: COCO 2017 Clean × Stable Diffusion 2 Inpainting (100 target pairs, 110 buffer pool)
2. `coco_sdxl`: COCO 2017 Clean × SDXL Inpainting 1.0 (100 target pairs, 110 buffer pool)
3. `commons_sd2`: Wikimedia Commons × Stable Diffusion 2 Inpainting (100 target pairs, 110 buffer pool)
4. `commons_sdxl`: Wikimedia Commons × SDXL Inpainting 1.0 (100 target pairs, 110 buffer pool)

**Total Target**: 400 source pairs (800 images + 400 masks).
**Total Buffer Pool**: 440 candidates (110 per stratum).

---

## 4. Quota and Replacement Policy Invariants

The multi-dimensional quota layout within each stratum remains strictly locked:
- **Core 100 slots**:
  * Modification types: 40 replacement, 30 removal & infill, 30 insertion
  * Mask area classes: 30 small (<10%), 40 medium (10-30%), 30 large (>30%)
- **Buffer 10 slots**:
  * One instance of each 9 (modification, mask) combinations + 1 extra (replacement, medium)
  * Marginals: 4/3/3 and 3/4/3
- **Replacement Rule**:
  * Failed candidates are replaced strictly in order within the same stratum.
  * If replacement would distort the required modification type or mask area margins, the candidate buffer provides cell-covering replacements.
