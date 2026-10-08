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

---

## 2. Follow-up pilot technical diagnosis & remediation plan (`pilot-20261008T113700Z`)

> Run: `pilot-20261008T113700Z`<br>
> Binding commit: `d9d99b678053972436032a828056a76a6392fbb5`<br>
> Status: `REMEDIATION_PLAN_PREPARED_PENDING_HUMAN_REVIEW`<br>
> Note: Zero generation executed; Human Content QC for current run remains strictly `PENDING`.

### 2.1. Technical QC Threshold Reconciliation

Cross-checking code at commit `d9d99b678053972436032a828056a76a6392fbb5` (`ml/evaluation/independent_cohort_acquisition.py:evaluate_technical_qc`) against `CONTENT_GROUNDED_EDITING_AMENDMENT.md` establishes the exact thresholds:

1. **`NON_BLANK_STD_THRESHOLD = 5.0`**: Standard deviation of pixel arrays for both authentic and edited images must be $\ge 5.0$, protecting against solid-color or degenerate near-blank images.
2. **`MIN_MASKED_PIXEL_DELTA_L1 = 3.0`**: Mean absolute difference L1 between authentic and edited images within the mask region (`mask == 255`) must be $\ge 3.0$, ensuring measurable intervention took place.
3. **`MAX_UNMASKED_PIXEL_DELTA_L1 = 0.5`**: Mean absolute difference L1 outside the mask region (`mask == 0`) must be $\le 0.5$, ensuring strict background invariance.
4. **`diff_std` (Difference Image Standard Deviation)**: Computed purely as a descriptive statistic ($\text{std}(I_{\text{edit}} - I_{\text{auth}})$); **no threshold exists** in either code or protocol.
5. **Threshold Correction**: Earlier notes citing `inside L1 >= 5.0` conflated `NON_BLANK_STD_THRESHOLD = 5.0` with `MIN_MASKED_PIXEL_DELTA_L1 = 3.0`. The code-enforced inside-mask delta requirement is strictly `3.0`. All 8 candidates passed this threshold (observed range: 11.08 to 76.14).

### 2.2. Pipeline Transmission Audit

Inspection of code, configuration, and Colab logs reveals the exact runtime configuration:

| Component | Current Implementation | Status & Characteristic |
| :--- | :--- | :--- |
| **Prompt** | `spec.prompt` passed directly to `pipeline(prompt=...)` | Raw text; no template, no positive modifiers, no trigger prefix. |
| **Negative Prompt** | Not passed (`None`) | Default unconditional embedding `""`; no penalty on blur/empty infill. |
| **Mask Polarity** | Mode `L`, `{0, 255}` | Standard Diffusers convention: `255` = inpaint region, `0` = preserve. Correct. |
| **Preprocessing** | `normalize_image_to_canvas` | Lanczos aspect-preserving scale + center crop to 512×512. Direct coordinate mapping. |
| **Strength** | Not passed | Default inpainting behavior (pure random noise initialization in mask at $t=T$). |
| **Guidance Scale** | Fixed at `7.5` | Standard classifier-free guidance for both SD2 and SDXL. |
| **Scheduler & Steps** | SD2: DDIM (50 steps); SDXL: EulerDiscrete (30 steps) | Aligned with locked model registry. |
| **Inference Canvas** | Fixed `512×512` for both tools | SD2 is at native training resolution; SDXL (trained at 1024×1024) is compressed to 512×512. |

### 2.3. Cause Classification: Proven vs Unverified Hypotheses

#### A. Proven Software & Physical Causes
1. **Hard Binary Compositing (Boundary Seams)**: `composite_generated_region` executes `Image.composite(gen, auth, mask)` on a binary 1-bit mask with zero feathering. When axis-aligned rectangular boxes cut through continuous physical geometry (horizontal railing at $x=170, 380$; ceiling plaster at $y=155$; kitchen wall tone at $y=95, x=280$), an immediate 1-pixel color/texture discontinuity is physically inevitable.
2. **Contact Sheet Metric Display Bug**: Script referenced non-existent `att['qc_details']` key in `attempt_ledger.jsonl`, defaulting to `in L1=0.0`. Fixed by computing L1 directly from RGB arrays.
3. **Ledger Semantics**: `attempt_ledger.jsonl` status `ACCEPTED` represents Technical QC passage at Step D/E, whereas Human Content QC is tracked separately in `provenance_ledger.jsonl` and `run_receipt.json` as `PENDING_CONTENT_QC`.

#### B. Unverified Hypotheses (Requiring Controlled Empirical Testing)
1. **Latent Space Resolution Bottleneck**: In SDXL and SD2 ($8\times$ downsampling VAE), a 3.78% mask ($110 \times 90$ px) maps to only $\approx 13 \times 11$ latent pixels; a 6.58% mask ($150 \times 115$ px) maps to $\approx 18 \times 14$ latent pixels. It is hypothesized that at 512×512 canvas resolution, tiny latent patches lack sufficient spatial capacity to synthesize distinct multi-part foreground objects. *(Unverified hypothesis)*.
2. **Infill Conditioning Bias from Surrounding Textures**: Text prompts describe both the target object and strong background context (e.g., "resting on slice of bread", "cobblestones beside car", "cloudy sky"). Surrounded by unmasked bread, pavement, or sky latents, UNet cross-attention may favor continuing background texture over synthesizing an isolated object. *(Unverified hypothesis; note: inference reverse sampling does NOT perform gradient optimization or loss minimization at test time)*.

### 2.4. Concrete Remediation Proposals

#### Proposal A: Object Insertion Enhancement
1. **Option A.1: Prompt Refinement & Negative Prompting (Methodological Test)**:
   - Isolate salient object tokens from background description: e.g., `"a ripe red cherry tomato with shiny skin, food photography, sharp focus"` instead of repeating `"resting on slice of bread, matching lunchbox lighting"`.
   - Add negative prompt: `"empty, blurry, smooth texture, missing object, background only"`.
   - Test increasing `guidance_scale` (e.g., from 7.5 to 9.5–11.0) for insertion tasks to strengthen text attention relative to image conditioning.
2. **Option A.2: Local Crop Inference with Padding (Methodological Test)**:
   - Extract a square local bounding crop around the mask with padding margin $P$ (e.g., context-preserving crop of $256 \times 256$ or $512 \times 512$).
   - Upscale the local crop to model-native resolution ($1024 \times 1024$ for SDXL, $512 \times 512$ for SD2) where the mask occupies a substantially larger latent area.
   - Run inpainting, downscale result with Lanczos, and composite back to the 512×512 canvas strictly within the registered mask bbox.
   - *Geometric & Optical Risks*:
     - **Perspective Drift**: Local crop loses global horizon line and room vanishing points, risking incorrect camera tilt.
     - **Illumination Misalignment**: Shadow direction may decouple from scene key light outside the crop.
     - **Scale Ambiguity**: Model lacks reference objects to calibrate physical size (e.g., tomato may appear disproportionately large).
     - **Resampling Artifacts**: Round-trip Lanczos resampling may induce subtle boundary blurring or ringing.

#### Proposal B: Seam Reduction Preserving Outside-Mask Invariance
1. **Inward-Only Edge Feathering (Inward Blending)**:
   - **Invariance Rule**: Outside-mask pixels must remain strictly untouched ($\text{outside\_L1} = 0.000000$).
   - **Mechanism**: Blending is applied **strictly inside** the registered mask region ($mask == 255$) across an inward margin of $k$ pixels ($k = 2$ or $3$ px):
     $$I_{\text{composite}}(p) = \alpha(p) \cdot I_{\text{gen}}(p) + (1 - \alpha(p)) \cdot I_{\text{auth}}(p)$$
     where $\alpha(p) = 0$ for $p$ outside mask, $\alpha(p)$ ramps smoothly (linear or cosine) from $0 \to 1$ over $k$ pixels inside the mask edge, and $\alpha(p) = 1$ in the mask core ($d > k$).
   - **Scope & Limitations**: Inward feathering eliminates 1-pixel high-frequency step discontinuities; it **cannot** fix macroscopic geometric severance (such as severed railings or altered architectural planes).
2. **Geometric Natural-Boundary Alignment**:
   - For scenes with strong architectural lines or linear structures (ceilings, railings), masks should be registered along natural scene seams (e.g., architectural moldings, trim edges) rather than arbitrary axis-aligned boxes cutting mid-structure, or candidates with uncuttable linear foregrounds should be avoided.

---

### 2.5. Controlled Next-Pilot Experimental Design

To isolate causal factors without confounding multiple simultaneous changes, the next pilot should be structured into distinct experimental arms:

| Test Arm | Focus Hypothesis | Variables Tested | Controlled Invariants |
| :--- | :--- | :--- | :--- |
| **Arm 1: Prompt & Negative Prompt** | Infill Conditioning Bias | Prompt token isolation + Negative prompt | Standard full-canvas 512×512; hard composite; guidance 7.5 |
| **Arm 2: Inward Feathering** | Boundary Seams | Hard binary composite vs Inward 2px cosine feathering | Identical generation output; outside L1 = 0.000000 preserved |
| **Arm 3: Guidance & Local Crop** | Latent Resolution Deficit | Local crop inference ($1024 \times 1024$) vs 512×512 canvas | Same candidate & prompt; evaluate scale/perspective consistency |
| **Arm 4: Candidate / Natural Boundary** | Macroscopic Discontinuity | Natural-boundary mask alignment vs axis-aligned cut | Tool and quota bracket preserved |

#### Candidate Accounting & Scientific Governance
- **Historical Integrity**: Run `pilot-20261008T113700Z` consumed its 8 registered attempts under `content_grounded_pilot_plan.json`. All artifacts and ledgers remain sealed.
- **Candidate Options for Next Plan**:
  - *Option 1 (Progressive Buffer Pool)*: Select 8 fresh candidates from the 440-pool buffer (pool index order) applying refined instructions.
  - *Option 2 (Transparent Diagnostic Calibration)*: Register a dedicated non-production diagnostic run ID (`pilot-202610...-diag`) with explicit one-shot re-attempts on failed candidates (e.g., bread tomato, suitcase, sky bird) to directly test Arm 1 and Arm 3 under identical authentic images.
- **Protocol Gate**: No parameter or method changes will be merged into production code without an approved amendment. The next pilot plan remains **UNAPPROVED** until human review. Zero GPU generation executed in this session. Human Content QC for `pilot-20261008T113700Z` remains **`PENDING`**.

---

### 2.6. Empirical Inward Feathering (k=2 px) Assessment Results

Using the authentic, edited, and mask files from `pilot-20261008T113700Z`, derived images with inward feathering ($k=2$ px, cosine transition) were computed across all 8 candidates and saved to `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/derived_feathered_k2/`. All 8 derived images satisfy the strict invariant `outside_mask_mean_l1 = 0.000000` (max outside delta = 0).

A self-contained comparison contact sheet was generated at `data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z/feathering_comparison_contact_sheet.html`.

#### Detailed Visual Case Findings
1. **Wall Clock (`IND_COCO_SD2_001`)**:
   - *Seam Softening*: The 1-pixel hard step at $y=95$ and $x \in \{195, 280\}$ is smoothed over a 2-pixel cosine gradient. Tile color transitions are less abruptly stepped.
   - *Ghosting & Detail Re-emergence*: Low ghosting. A 1-2 pixel sliver of authentic grout line slightly bleeds inward.
   - *Geometric Severance*: Does NOT fix the primary defect: the inpaint content inside remains a murky metallic blob rather than a recognizable clock.
2. **Ceiling Chandelier (`IND_COCO_SDXL_041`)**:
   - *Seam Softening*: The sharp line at $y=155$ is visually softened from $y=153$ to $155$.
   - *Ghosting & Detail Re-emergence*: At $x \in [12, 25]$ (where mask contacts curtain valance apex), blending dark fabric with white plaster yields a minor faint blur band. Authentic curtain texture re-appears in the 2px strip at $y=153..154$.
   - *Geometric Severance*: The macroscopic mismatch between the bright white modern ceiling plane (generated) and the textured off-white ceiling (authentic) remains starkly apparent. Feathering smooths the step but leaves the planar tonal discontinuity intact.
3. **Park Column (`IND_COMMONS_SDXL_003`)**:
   - *Seam Softening*: Converts the vertical 1-pixel edges at $x=170$ and $x=380$ into a 2-pixel transition.
   - *Ghosting & Detail Re-emergence*: At the railing intersection ($y \approx 340..375$), blending the severed railing end with park foliage creates a semi-transparent, blurred railing stub.
   - *Geometric Severance*: The continuous metal railing remains physically severed across a 210-pixel gap ($x=170..380$). A 2-pixel blend cannot bridge or reconnect missing geometry.
- **Empirical Conclusion**: Inward feathering effectively eliminates high-frequency 1-pixel edge steps without touching outside pixels, but **cannot resolve macroscopic structural severance, planar tonal mismatches, or internal semantic generation omissions**.

---

### 2.7. Proposed 6-Attempt Diagnostic Plan Specification & Implementation (`content_grounded_diagnostic_plan.json`)

To directly evaluate full-canvas inference vs local-crop padded inference across insertion omissions, a dedicated diagnostic plan has been registered at `research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json` (SHA-256: `8f2d980965a56cf8939d774e36321e583a99ba89faba4a502d59c0a25a9630e3`).

- **Status**: Strictly `PENDING` (human_reviewer=null, human_reviewed_at_utc=null). Zero GPU runs executed.
- **Objective Refinement**: Evaluates methodological differences between full-canvas and local-crop execution (including context window alteration, token scale shift, and geometric area scaling) and does NOT claim to isolate latent downsampling alone. Identical PRNG seeds across different tensor resolutions do not guarantee identical noise fields due to dimension-dependent sampling order.
- **Candidate Scope & Input Bindings**: Exactly 3 candidates with insertion omissions bound to sealed authentic and mask PNG files from `pilot-20261008T113700Z`:
  - `IND_COCO_SDXL_002` (bread tomato): auth SHA-256 `7aefc1d1...` (`images/IND_COCO_SDXL_002_auth.png`), mask SHA-256 `2ff6b165...` (`masks/IND_COCO_SDXL_002_mask.png`), seed 20272319, steps 30, EulerDiscreteScheduler.
  - `IND_COMMONS_SD2_002` (suitcase): auth SHA-256 `98004bf7...` (`images/IND_COMMONS_SD2_002_auth.png`), mask SHA-256 `fee9ac7e...` (`masks/IND_COMMONS_SD2_002_mask.png`), seed 20283429, steps 50, DDIMScheduler.
  - `IND_COMMONS_SDXL_001` (sky bird): auth SHA-256 `7680e4ce...` (`images/IND_COMMONS_SDXL_001_auth.png`), mask SHA-256 `4981cedc...` (`masks/IND_COMMONS_SDXL_001_mask.png`), seed 20294438, steps 30, EulerDiscreteScheduler.
  - Input policy: Must use sealed normalized PNGs from `pilot-20261008T113700Z`; remote redownloading from the web is strictly prohibited. Runner resolves inputs from both flat root (`base_dir / filename`) and standard package paths (`base_dir / images/` and `base_dir / masks/`). Prioritize reusing the existing run directory on Google Drive (`RUNS_ROOT / "pilot-20261008T113700Z"`).
- **Controlled Arms**:
  - **Arm A (Full Canvas)**: Standard $512 \times 512$ inference under locked parameters (identical authentic, mask, prompt, seed from ledger, guidance 7.5, strength 1.0, scheduler, steps). Geometric area factor = $1.0\times$. Direct composite onto canvas; outside L1 = 0.000000.
  - **Arm B (Local Crop with Padding)**: Local square crop ($256 \times 256$ for SDXL, $320 \times 320$ for SD2) upscaled via Lanczos (RGB) and Nearest (mask) to model native resolution ($1024 \times 1024$ for SDXL, $512 \times 512$ for SD2) $\to$ inpainting $\to$ downscaled via Lanczos $\to$ remapped to canvas and composited strictly within registered binary mask. Outside pixels invariant ($0.000000$).
  - **Geometric Area Factors & Coordinate Rounding Rules**:
    - SDXL candidates (tomato, bird): $(1024/256)^2 = 16.0\times$ geometric area ratio. Integer scaling factor = 4. Resized masks verified {0, 255} binary; tomato bbox [356, 332, 796, 692] (158,400 px); bird bbox [376, 180, 976, 640] (276,000 px).
    - SD2 suitcase candidate: $(512/320)^2 = 2.56\times$ geometric area ratio. Linear scale factor = 1.6. Continuous mapping: $x \in [0.0, 360.0], y \in [236.8, 512.0]$. Nearest-neighbor sampling maps destination row $y_{dst}=236 \to$ source row 147 (value 0), $y_{dst}=237 \to$ source row 148 (value 255). Resized mask raster bbox is exactly `[0, 237, 360, 512]`, containing exactly 99,000 pixels, values strictly $\{0, 255\}$.
    - Latent pixel counts: methodologically estimated assuming 8x VAE downsampling ($13 \times 11 = 143$ vs $55 \times 45 = 2475$ for tomato; $28 \times 21 = 588$ vs $45 \times 34 = 1530$ for suitcase; $18 \times 14 = 252$ vs $75 \times 57 = 4275$ for bird), not directly measured tensor tokens.
- **Experimental Invariant**: Negative prompt and feathering are strictly excluded from this comparison.
- **Attempt Budget**: Exactly 6 attempts (3 candidates $\times$ 2 arms). One attempt per branch, no retries, no automatic replacement. Started attempts are recorded in durable ledger even if failed and count toward budget.
- **Diagnostic Runner Implementation**: Implemented `ml/evaluation/independent_cohort_diagnostic.py` and CLI `--mode diagnostic` in `scripts/research/run_cohort_acquisition.py`. Enforces fail-closed approval gate, verifies input hashes, re-seeds RNG independently per arm with candidate registered seed, bars mock engines (`DiagnosticEngineError` when allow_mock=False), preserves raw pre-composited images, enforces outside L1 = 0, supports resume without re-running attempts, and generates self-contained HTML contact sheet with PENDING review status. Tested via `ml/tests/test_independent_cohort_diagnostic.py` (11/11 PASS).
- **Colab Handover Architecture**:
  - `notebooks/independent_cohort_acquisition_colab.ipynb` supports explicit selection `EXECUTION_MODE = "diagnostic"`.
  - Removed all fallback/guessing on `EXECUTION_MODE`; stops immediately before CLI if mode is missing or invalid.
  - Records successful execution context (`RUN_CONTEXT`) in Cell 3 for Cell 4 audit and packaging.
  - Preflight `--check-diagnostic-plan` validates plan hash and Drive inputs (`RUNS_ROOT / "pilot-20261008T113700Z"`).
  - PENDING plan status causes execution to fail-closed before consuming GPU attempts.
  - Diagnostic results are isolated in dedicated run directory and do not enter the official independent cohort or alter historical pilot records.
- **Governance & Approval Protocol**:
  1. Human Content QC determinations per candidate for `pilot-20261008T113700Z` remains PENDING. This is a separate review decision and is NOT a prerequisite for diagnostic execution.
  2. Human decision to approve or revise the proposed 6-attempt diagnostic plan (`content_grounded_diagnostic_plan.json`) remains PENDING.
  3. Approval workflow: When the user approves the diagnostic plan, commit the plan with `human_review_status = "APPROVED"` and reviewer metadata in a functional commit, then pin `EXPECTED_COMMIT` in the notebook to that new commit and commit/push together.
