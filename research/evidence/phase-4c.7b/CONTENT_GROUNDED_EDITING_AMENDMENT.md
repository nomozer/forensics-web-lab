# Content-grounded editing amendment

Metadata: workstream `independent_cohort_acquisition`; historical phase trace `4C.7B`; amendment version `1.4.0`; status `PROPOSED_FOR_HUMAN_REVIEW`.

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

Generated output is composited with the authentic image using the binary request mask before Technical QC. This makes pixels outside the requested region invariant. Technical QC enforces the registered `5.0`, `3.0`, and `0.5` thresholds for non-blank standard deviation, masked L1 change, and unmasked L1 change respectively.

## Human Content QC criteria

A reviewer may accept a future pair only when the intended operation is visibly achieved in the registered target region, the prompt is semantically compatible with the authentic scene, target geometry is coherent, and there are no severe seams or unrelated scene rewrites. For removal, the registered existing target must be absent and the infill plausible; for replacement, the registered existing target must be replaced by the prompted object; for insertion, the prompted object must appear at the registered scene-compatible placement. Technical invariance outside the mask remains a machine gate, not a substitute for this review. The edit need not be aesthetically perfect when these criteria are satisfied.

## Bounded follow-up pilot

The next proposed pilot has a maximum budget of eight attempts: two registered candidates per stratum, with the same eight source IDs, tools, modification-type distribution (3 replacement, 1 removal, 4 insertion), and mask-class distribution (3 small, 2 medium, 3 large) as the audited pilot. The registered instructions are in `content_grounded_pilot_plan.json`.

No automatic replacement candidate is authorized. Each registered candidate may be generated once. Any generation error or Technical QC rejection is recorded and leaves a quota deficit; a new reviewed plan is required before another attempt. The committed plan remains `human_review_status: PENDING`, and the production CLI refuses generation until a human records approval. Human Content QC may later accept or reject generated pairs; this document does not pre-approve any result.

Full acquisition remains blocked until the user reviews the plan/overlays, a future eight-pair pilot completes, and Human Content QC records an explicit decision.
