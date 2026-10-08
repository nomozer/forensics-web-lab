# Content-grounded editing amendment

Metadata: workstream `independent_cohort_acquisition`; historical phase trace `4C.7B`; amendment version `1.6.0`; status `PROPOSED_FOR_HUMAN_REVIEW`.

This amendment changes only future runs. It does not rename, move, modify, or reinterpret any sealed artifact or checksum from earlier runs.

## Evidence requiring the amendment

The real eight-pair pilot `pilot-20261007T132003Z` proved that the allocation runner assigned prompts from a generic list and placed ellipse masks from a seeded random draw. Neither operation inspected image content or registered a target object/placement. The eight masks were created on the normalized 512×512 canvas, so there is no evidence of a crop/resize coordinate shift; the proven defect is the absence of source-content annotation.

All eight edited images also exceed the locked `max_unmasked_pixel_delta_l1 = 0.5` threshold (observed range 3.808–9.311). The production Technical QC implementation did not evaluate that threshold. The implementation also used `2.0` rather than the locked `non_blank_std_threshold = 5.0`.

These findings explain invalid Technical QC acceptance and make content-grounded regions necessary. They do not prove that every visual defect has one model-side cause, and they do not convert agent observations into Human Content QC decisions.

## Policy change for future generation

Before any production generation attempt, each candidate must have a reviewed edit instruction bound to the run plan:

- prompt written for the actual normalized authentic image;
- target/inserted object description;
- placement rationale;
- target bounding box and mask bounding box in half-open `xyxy` coordinates on the normalized 512×512 canvas;
- unchanged source, tool, modification-type, and mask-class allocations.

Removal and replacement require a visible existing target identified before generation. Insertion requires a scene-compatible support/placement region identified before generation. The runner must fail closed when any field is missing, the target lies outside the mask, or the mask violates its registered area class.

Registered rectangular plans are checked against the protocol classes themselves: small is 1–10%, medium is 10–30%, and large is 30–50% of the normalized canvas. The broader Technical QC raster tolerance is not permission to register a bbox in the wrong named class. Earlier distinction corrected `IND_COCO_SD2_002`: `[95,75,415,315]` was only 29.296875%; the proposed `[95,75,415,323]` region is 30.273438% and extends only within the same upper-fixture/backsplash context.

Generated output is composited with the authentic image using the binary request mask before Technical QC. This makes pixels outside the requested region invariant. Technical QC enforces the registered `5.0`, `3.0`, and `0.5` thresholds for non-blank standard deviation, masked L1 change, and unmasked L1 change respectively.

An outside-mask mean L1 value of exactly `0` after this compositing step is true **by construction**: those pixels are copied from the authentic image. It does not demonstrate that the raw diffusion output independently preserved pixels outside the mask.

## Human Content QC criteria

A reviewer may accept a future pair only when the intended operation is visibly achieved in the registered target region, the prompt is semantically compatible with the authentic scene, target geometry is coherent, and there are no severe seams or unrelated scene rewrites. For removal, the registered existing target must be absent and the infill plausible; for replacement, the registered existing target must be replaced by the prompted object; for insertion, the prompted object must appear at the registered scene-compatible placement. Technical invariance outside the mask remains a machine gate, not a substitute for this review. The edit need not be aesthetically perfect when these criteria are satisfied.

## Bounded follow-up pilot

The next proposed pilot has a maximum budget of eight attempts: two registered candidates per stratum, with the same tools, modification-type distribution (3 replacement, 1 removal, 4 insertion), and mask-class distribution (3 small, 2 medium, 3 large) as the audited pilot. The registered instructions are in `content_grounded_pilot_plan.json`.

Version 1.6.0 resolves the three open follow-up instructions for human review:

1. **`IND_COMMONS_SD2_001` geometric correction**: The previous mask `[220, 285, 512, 512]` (27.88%) was geometrically deficient on the authentic photograph: at $x=220$, it sliced through the prominent rocky cliff base on the lower left (which extends to $x \approx 205$ with submerged sea stacks to $x \approx 190$), leaving unmasked rocks in the sea; at $y=285$, it crossed into the background bay and sliced through the distant hotel and town. The revised mask `[190, 305, 512, 512]` ($322 \times 207 = 66,654$ px = 25.426483%, strictly within `medium_10_to_30pct` 10%–30%) completely envelops 100% of the registered foreground headland, cliff base, and pine tree canopy ($y \ge 308$), while preserving the background hotel, beach, and sunset horizon outside the mask.

2. **`IND_COCO_SDXL_001` candidate substitution**: The cat-in-sink scene (`coco:501523`) cannot accommodate a $\ge 30\%$ insertion on the upper wall without severe occlusion of the cat and basin. Amendment v1.6.0 proposes substituting candidate `IND_COCO_SDXL_041` (origin `coco:189310`, author `an iconoclast`, CC BY 2.0; pool index 40) within `coco_sdxl`, preserving `sdxl_inpainting`, `object_insertion`, and `large_over_30pct`. The target is a crystal chandelier on the open living room ceiling (`target_bbox: [180, 15, 332, 145]`, `mask_bbox: [0, 0, 512, 160]`, area $512 \times 160 = 81,920$ px = 31.250000%), with zero furniture or subject occlusion. The strict pool-order alternative `IND_COCO_SDXL_005` (cargo airplane on tarmac at night, pool index 4; mask `[0, 260, 320, 512]`, area 30.7617%) is documented as an option, but partially overlaps background baggage carts and workers.

3. **`IND_COMMONS_SDXL_002` candidate substitution**: On the "Tower of Freedom" statue, the child figure is held tightly across the adult figure's chest with adult hands and fingers wrapped across the torso and legs. Any mask covering the child statue includes the adult's hands and arms, creating an unavoidable anatomical/identity confound inside the mask. Amendment v1.6.0 proposes substituting candidate `IND_COMMONS_SDXL_003` (origin `commons:166529058`, author `Crisco 1492`, CC BY-SA 4.0; pool index 2) within `commons_sdxl`, preserving `sdxl_inpainting`, `object_replacement`, and `large_over_30pct`. The candidate depicts "Tower Song" by Ted Bieler, a single, completely isolated vertical abstract aluminium sculpture column on an open park lawn (`target_bbox: [190, 0, 360, 512]`, `mask_bbox: [170, 0, 380, 512]`, area $210 \times 512 = 107,520$ px = 41.015625%), replacing it with a classical fluted marble column. This candidate is the immediate next eligible record in pool order and completely eliminates the intertwining limbs confound. Retaining `IND_COMMONS_SDXL_002` with explicit human acceptance of the arm confound remains documented as Option 2B.

No automatic replacement candidate is authorized. Each registered candidate may be generated once. Any generation error or Technical QC rejection is recorded and leaves a quota deficit; a new reviewed plan is required before another attempt. The committed plan remains `human_review_status: PENDING`, and the production CLI refuses generation until a human records approval. Human Content QC may later accept or reject generated pairs; this document does not pre-approve any result.

Full acquisition remains blocked until the user reviews the plan/overlays, a future eight-pair pilot completes, and Human Content QC records an explicit decision.
