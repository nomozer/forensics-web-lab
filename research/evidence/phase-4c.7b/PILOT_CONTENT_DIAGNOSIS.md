# Real pilot content diagnosis

> Run: `pilot-20261007T132003Z`  
> Scope: the eight accepted authentic–mask–edited tuples only  
> Status: `AGENT_DIAGNOSIS_COMPLETE_HUMAN_CONTENT_QC_PENDING`  
> This document is not Human Content QC approval.

## Reproducible facts

- The original attempt ledger has 8 records: 8 `ACCEPTED`, 0 `QC_FAILED`, 0 errors, exactly 2 per stratum. This is not inferred from the number of images.
- Every mask was synthesized directly on the normalized 512×512 canvas. The source images were center-cover resized and cropped before generation. Because the run contains no source-content target coordinates, no source-to-canvas mapping was attempted; there is no evidence of a shifted coordinate transform. The demonstrated defect is that no content target was registered at all.
- All model bindings match the run protocol. SD2 used revision `5f74973cbb64c8568780732c17f43eb269d63a0d`, `DDIMScheduler`, 50 steps, guidance 7.5, fp16, 512×512. SDXL used revision `115134f363124c53c7d878647567d04daf26e41e`, `EulerDiscreteScheduler`, 30 steps, guidance 7.5, fp16, 512×512. No parameter mismatch was found.
- All eight outputs violate the locked outside-mask limit (`max_unmasked_pixel_delta_l1 = 0.5`). The production code at the bound commit did not run this check.

| Pair | Mask bbox on normalized canvas | Raw → resized; center crop offset | Mean L1 inside / outside mask | Content observation and agent disposition |
|---|---:|---:|---:|---|
| `IND_COCO_SD2_001` | `[148,159,339,296]` | 640×427 → 767×512; `(127,0)` | 44.960 / 8.564 | Mask covers central range/pan area. Prompt requests a bronze statue, but no coherent statue appears; the cooking area is replaced by a severe structural discontinuity. Agent: likely reject. |
| `IND_COCO_SD2_002` | `[64,63,483,402]` | 352×230 → 784×512; `(136,0)` | 50.110 / 3.808 | Large mask covers most cabinets and stove. Guitar-like fragments are not a coherent replacement; the kitchen geometry is broadly rewritten. Agent: likely reject. |
| `IND_COCO_SDXL_001` | `[122,23,467,456]` | 640×480 → 683×512; `(85,0)` | 52.626 / 5.840 | Large mask covers the sink and existing black cat. A ginger kitten is partly visible, but a large glossy black form and blue textures dominate; this behaves as destructive replacement rather than a clean insertion. Agent: likely reject. |
| `IND_COCO_SDXL_002` | `[356,228,455,335]` | 288×160 → 922×512; `(205,0)` | 16.051 / 4.667 | Small mask lies on the bread. The requested brass lamp is absent; only a local bread deformation is visible. Agent: likely reject. |
| `IND_COMMONS_SD2_001` | `[180,78,473,353]` | 4543×3176 → 732×512; `(110,0)` | 13.657 / 4.693 | Mask spans sky, sea, and coastline. The edited coastline/sky remains plausible at a glance, but the run never identified an object to remove, so operation success cannot be determined from the artifact. Agent: inconclusive; recommend reject pending a defined target. |
| `IND_COMMONS_SD2_002` | `[254,168,489,491]` | 5278×3410 → 792×512; `(140,0)` | 46.466 / 6.690 | Mask covers the car's front wheel/body area, not an empty lamp placement. No brass lamp appears; wheel and front geometry are heavily altered. Agent: likely reject. |
| `IND_COMMONS_SDXL_001` | `[164,20,295,201]` | 1744×3121 → 512×916; `(0,202)` | 19.821 / 9.311 | Mask covers the monument top and inscription. No kitten appears; inscription and relief are corrupted. Agent: likely reject. |
| `IND_COMMONS_SDXL_002` | `[82,190,453,481]` | 4000×6000 → 512×768; `(0,128)` | 46.340 / 7.181 | Mask covers the child statue and arms. A guitar-like/abstract form appears, but it is severely malformed and occludes the sculpture unnaturally. Agent: likely reject. |

Canonical `content_qc_status` remains `PENDING_CONTENT_QC` for all eight. These dispositions are agent screening notes only.

## Cause classification

### Demonstrated causes

1. The allocation plan selected generic prompts by index without reading the authentic image.
2. The mask generator selected a seeded random ellipse on the normalized canvas without a target object or placement annotation.
3. Technical QC omitted the locked outside-mask L1 check and used a non-blank standard-deviation threshold of 2.0 instead of 5.0.
4. The execution log mislabeled candidate-level Technical QC rejection as “content QC.”

### Not demonstrated

- Crop/resize coordinate drift: no source coordinates existed to drift, and the masks were generated after normalization.
- Model revision, scheduler, step, guidance, or output-size mismatch: the audited parameters match the locked model-specific protocol.
- A single model defect as the cause of every visual failure: poor appearance alone does not establish this.
- Absence of AI editing when the prompt is not matched: every pair has measured pixel changes; semantic mismatch is a Content QC issue, not proof of no AI operation.

## Corrective boundary

The implementation now requires a content-grounded edit plan before production generation, composites generated pixels only inside the registered binary mask, enforces the locked Technical QC thresholds, and uses Technical QC terminology. The proposed eight-attempt plan and Human Content QC criteria are defined in `CONTENT_GROUNDED_EDITING_AMENDMENT.md` and `content_grounded_pilot_plan.json`. No follow-up generation was run.

## Historical incident count reconciliation

The older incident `pilot-20261007T093824Z` is not present in the current local artifact directory or any mounted drive, so its ledger cannot be re-opened in this session. The preserved incident transcription states 222 total attempt records and 2 accepted `coco_sd2` records. Those figures imply 220 `coco_sdxl` failure records, not 110 attempt records; the pool contained 110 unique SDXL candidates. The phase report is corrected to say “220 SDXL attempt records / 110 unique candidates.” The reason each candidate produced two records cannot be independently established without the original ledger and is therefore not asserted.
