# Real pilot content diagnosis

> Run: `pilot-20261007T132003Z`
> Scope: the eight accepted authentic–mask–edited tuples only
> Status: `AGENT_DIAGNOSIS_COMPLETE_HUMAN_CONTENT_QC_PENDING`
> This document is not Human Content QC approval.
> Evidence classification: run observations are `internal-empirical`; causal explanations under “Unverified Hypotheses” are `unverified-hypothesis`. External citation keys resolve through `docs/references.bib`.

## Reproducible facts

- The original attempt ledger has 8 records: 8 `ACCEPTED`, 0 `QC_FAILED`, 0 errors, exactly 2 per stratum. This is not inferred from the number of images.
- Every mask was synthesized directly on the normalized 512×512 canvas. The source images were center-cover resized and cropped before generation. Because the run contains no source-content target coordinates, no source-to-canvas mapping was attempted; there is no evidence of a shifted coordinate transform. The demonstrated defect is that no content target was registered at all.
- All model bindings match the run protocol. SD2 used the explicitly classified **community mirror** revision `5f74973cbb64c8568780732c17f43eb269d63a0d` `[@sd2CommunityInpaintingModelCard]`, `DDIMScheduler`, 50 steps, guidance 7.5, fp16, 512×512. SDXL used the official Diffusers model-card revision `115134f363124c53c7d878647567d04daf26e41e` `[@sdxlInpaintingModelCard]`, `EulerDiscreteScheduler`, 30 steps, guidance 7.5, fp16, 512×512. No parameter mismatch was found in the internal run binding; source verification does not assert semantic success.
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
1. **Latent Space Resolution Bottleneck**: In SDXL and SD2 (assuming $8\times$ VAE downsampling), a 3.78% mask ($110 \times 90$ px) maps to only $\approx 13 \times 11$ latent pixels; a 6.58% mask ($150 \times 115$ px) maps to $\approx 18 \times 14$ latent pixels. It is hypothesized that at 512×512 canvas resolution, tiny latent patches lack sufficient spatial capacity to synthesize distinct multi-part foreground objects. *(Source classification: `unverified-hypothesis`; latent tensor counts were estimated, not directly instrumented.)*.
2. **Infill Conditioning Bias from Surrounding Textures**: Text prompts describe both the target object and strong background context (e.g., "resting on slice of bread", "cobblestones beside car", "cloudy sky"). Surrounded by unmasked bread, pavement, or sky latents, UNet cross-attention may favor continuing background texture over synthesizing an isolated object. *(Source classification: `unverified-hypothesis`; inference reverse sampling does NOT perform gradient optimization or loss minimization at test time.)*.

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

To directly evaluate full-canvas inference vs local-crop padded inference across insertion omissions, a dedicated diagnostic plan has been registered at `research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json`. Prior to approval, the plan file was verified to match pre-approval hash SHA-256 `8f2d980965a56cf8939d774e36321e583a99ba89faba4a502d59c0a25a9630e3`. Upon human approval, the plan is registered under approved SHA-256 `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157`.

- **Status**: `APPROVED` by human reviewer Dũng Phạm `<valdung04@gmail.com>` at `2026-10-08T15:17:27Z`. Zero GPU runs executed locally.
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
- **Diagnostic Runner Implementation**: Implemented `ml/evaluation/independent_cohort_diagnostic.py` and CLI `--mode diagnostic` in `scripts/research/run_cohort_acquisition.py`. Enforces fail-closed approval gate, verifies input hashes, re-seeds RNG independently per arm with candidate registered seed, bars mock engines (`DiagnosticEngineError` when allow_mock=False), preserves raw pre-composited images, enforces outside L1 = 0, supports resume without re-running attempts, and generates self-contained HTML contact sheet with PENDING review status. Tested via `ml/tests/test_independent_cohort_diagnostic.py` (12/12 PASS).
- **Colab Handover Architecture**:
  - `notebooks/independent_cohort_acquisition_colab.ipynb` supports explicit selection `EXECUTION_MODE = "diagnostic"`.
  - Removed all fallback/guessing on `EXECUTION_MODE`; stops immediately before CLI if mode is missing or invalid.
  - Records successful execution context (`RUN_CONTEXT`) in Cell 3 for Cell 4 audit and packaging.
  - Preflight `--check-diagnostic-plan` validates plan hash and Drive inputs (`RUNS_ROOT / "pilot-20261008T113700Z"`).
  - Diagnostic results are isolated in dedicated run directory and do not enter the official independent cohort or alter historical pilot records.
- **Governance & Approval Protocol**:
  1. Human Content QC determinations per candidate for `pilot-20261008T113700Z` remains PENDING. This is a separate review decision and is NOT a prerequisite for diagnostic execution.
  2. Diagnostic plan (`content_grounded_diagnostic_plan.json`) is APPROVED by human reviewer Dũng Phạm `<valdung04@gmail.com>` at `2026-10-08T15:17:27Z`. Approved plan SHA-256 is `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157`.
  3. Notebook pin workflow: Approved functional commit is registered and pushed, with `EXPECTED_COMMIT` in `notebooks/independent_cohort_acquisition_colab.ipynb` pinned directly to the functional commit.

---

### 2.8. Diagnostic Run Execution, Audit & Empirical A/B Evaluation (`diag-20261008T154628Z`)

> Run: `diag-20261008T154628Z`<br>
> Source commit: `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`<br>
> Approved plan SHA-256: `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157`<br>
> Status: `DIAGNOSTIC_INTAKE_AND_AUDIT_PASS_PENDING_HUMAN_CONTENT_QC`<br>
> Package ZIP SHA-256: `f13d7baf458e9db86809bced5a20c27f37adb6f698353a5678b200262a481d20` (15,988,679 bytes)<br>
> Isolation notice: Exploratory diagnostic run; strictly excluded from official cohort. Zero detector models evaluated. Full cohort remains locked.

#### 2.8.1. Archive Intake and Production Audit

- **Archive Safety Verification**: ZIP archive `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z_package.zip` inspected prior to extraction. Contains exactly 30 entries (zero directory traversal `..`, zero absolute paths, zero leading slashes). Extracted safely into dedicated run folder `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/`. All original ZIP, image, mask, receipt, and ledger files preserved intact without overwriting historical pilot runs.
- **Binding & Production CLI Audit**:
  - Production CLI audit executed at bound commit `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`: `python scripts/research/run_cohort_acquisition.py --audit-run diag-20261008T154628Z` $\to$ **`PASS`**.
  - Run ID verified: `diag-20261008T154628Z`.
  - Source commit verified: `0f99897c8ee003dc8ecbb55b09c4aaeb2994c2bc`.
  - Approved plan hash verified: `e352504016960a24c3c539b5d33a2876c222d1975426a7911de0ec35fe9df157`.
  - Attempt count verified: exactly 6 attempts across 3 candidates $\times$ 2 arms (`IND_COCO_SDXL_002`, `IND_COMMONS_SD2_002`, `IND_COMMONS_SDXL_001`). Note: `STARTED` and `ACCEPTED` in `attempt_ledger.jsonl` represent two lifecycle events within each attempt, not 12 attempts.
  - Input integrity verified: all authentic and mask PNG hashes match the approved plan JSON and match bit-identically to sealed normalized inputs from `pilot-20261008T113700Z` (`max_auth_diff == 0`, `max_mask_diff == 0`).

#### 2.8.2. Exact Recalculated Pixel-Level Metrics

Recalculated directly from on-disk RGB arrays (`auth.png`, `edit.png`, and `mask.png` in `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/`):

| Candidate ID | Target Object | Model | Arm | Linear Scale | Geometric Area Factor | Raw Gen Shape | Inside Mean L1 (std) | Inside Max Delta | Outside Mean L1 | Outside Max Delta |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `IND_COCO_SDXL_002` | Cherry tomato on bread | SDXL | **Arm A** | 1.0× | 1.0× | (512, 512, 3) | 16.7861 (13.73) | 81.0 | 0.000000 | 0.0 |
| `IND_COCO_SDXL_002` | Cherry tomato on bread | SDXL | **Arm B** | 4.0× | 16.0× | (1024, 1024, 3) | 28.2929 (32.96) | 176.0 | 0.000000 | 0.0 |
| `IND_COMMONS_SD2_002` | Wheeled travel suitcase | SD2 | **Arm A** | 1.0× | 1.0× | (512, 512, 3) | 30.7358 (24.12) | 146.0 | 0.000000 | 0.0 |
| `IND_COMMONS_SD2_002` | Wheeled travel suitcase | SD2 | **Arm B** | 1.6× | 2.56× | (512, 512, 3) | 43.3932 (40.49) | 226.0 | 0.000000 | 0.0 |
| `IND_COMMONS_SDXL_001` | Flying bird in cloudy sky | SDXL | **Arm A** | 1.0× | 1.0× | (512, 512, 3) | 11.0767 (9.00) | 43.0 | 0.000000 | 0.0 |
| `IND_COMMONS_SDXL_001` | Flying bird in cloudy sky | SDXL | **Arm B** | 4.0× | 16.0× | (1024, 1024, 3) | 13.0674 (32.45) | 214.0 | 0.000000 | 0.0 |

*Strict Invariance*: Across all 6 attempts, `outside_mean_l1 == 0.000000` and `outside_max_delta == 0.0` hold bit-identically by virtue of registered canvas mask compositing.

##### Tọa độ đối chiếu Kế hoạch (Plan) vs Thực thi (Execution)

| Candidate ID | Target Object | Target Bbox | Registered Mask Bbox (px & %) | Local Crop Bbox ($W \times H$) | Resized-Mask in Inference | Linear Scale | Area Factor | Plan vs Execution Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `IND_COCO_SDXL_002` | Cherry tomato on bread | `[375, 265, 430, 320]` | `[345, 245, 455, 335]` (9,900 px, 3.776550%) | `[256, 162, 512, 418]` ($256 \times 256$) | `[356, 332, 796, 692]` ($1024 \times 1024$) | 4.0× | 16.0× | **MATCH** (Bit-exact) |
| `IND_COMMONS_SD2_002` | Travel suitcase on street | `[45, 355, 190, 495]` | `[0, 340, 225, 512]` (38,700 px, 14.762878%) | `[0, 192, 320, 512]` ($320 \times 320$) | `[0, 237, 360, 512]` ($512 \times 512$) | 1.6× | 2.56× | **MATCH** (Bit-exact) |
| `IND_COMMONS_SDXL_001` | Bird flying in sky | `[395, 75, 455, 125]` | `[350, 45, 500, 160]` (17,250 px, 6.580353%) | `[256, 0, 512, 256]` ($256 \times 256$) | `[376, 180, 976, 640]` ($1024 \times 1024$) | 4.0× | 16.0× | **MATCH** (Bit-exact) |

*Lưu ý phân biệt tọa độ*:
- Với vali (`IND_COMMONS_SD2_002`): Crop bbox là `[0, 192, 320, 512]`. Tọa độ `[0, 237, 360, 512]` là bbox của mask sau khi scale 1.6× bên trong không gian suy luận $512 \times 512$, không phải crop bbox.
- Với chim (`IND_COMMONS_SDXL_001`): Crop bbox trong kế hoạch và thực thi là `[256, 0, 512, 256]`. Không dùng `[187, 0, 443, 256]` (vốn là một ghi chú tính toán sơ thảo không có trong plan).

#### 2.8.3. Visual Evaluation and Metric Dissociation

Visual inspection on image artifacts in `data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/images/` reveals critical qualitative distinctions across four explicit criteria: (1) hiện diện đúng đối tượng, (2) vị trí và tỷ lệ, (3) độ chân thực, (4) chất lượng biên:

1. **`IND_COCO_SDXL_002` (Cherry Tomato)**:
   - *Arm A (Full Canvas)*: Complete semantic omission. Cả ba ca Arm A đều không tạo đối tượng yêu cầu; vùng inpaint chỉ gồm texture ruột bánh mì lấp đầy phẳng.
   - *Arm B (Local Crop)*: Tạo được đối tượng yêu cầu. Xuất hiện quả cà chua cherry đỏ trong phạm vi target bbox `[375, 265, 430, 320]` với thể tích tròn 3D, cuống đài xanh 5 cánh.
   - *Đánh giá Độ sắc, Độ bóng và Kết cấu Nền*: Quả cà chua có độ sắc nét tốt và có đốm bóng sáng (specular highlight tại ~`[394, 276, 403, 286]`); tuy nhiên, kết cấu lát bánh mì bên trong mask `[345, 245, 455, 335]` do SDXL sinh lại có độ mịn cao hơn, thiếu các lỗ rỗ tự nhiên so với crumb bánh mì authentic xung quanh.
   - *Chất lượng Biên*: Crop bbox là `[256, 162, 512, 418]`, registered mask bbox là `[345, 245, 455, 335]`. Hard binary compositing bảo toàn bit-exact ngoài mask (`outside_L1 = 0.0`), nhưng tạo ra một bước chuyển tiếp vi mô (subtle rectangular step) trong kết cấu ruột bánh mì dọc theo **biên mask đã đăng ký `[345, 245, 455, 335]`** do ghép nhị phân với bánh mì authentic.
2. **`IND_COMMONS_SD2_002` (Suitcase)**:
   - *Arm A (Full Canvas)*: Complete semantic omission. Không tạo vật thể; chỉ phủ vân đá cuội phẳng.
   - *Arm B (Local Crop)*: Semantic hallucination. Mô hình sinh ra một chiếc xe ô tô đồ chơi cổ điển có mui và bánh xe thay vì vali du lịch!
   - *Crucial Finding (Metric Dissociation)*: Inside mean L1 tăng mạnh từ 30.74 lên 43.39 (max delta 226). **"L1 tăng không bảo đảm thành công ngữ nghĩa."** Không rút ra kết luận khái quát hóa về mối tương quan tổng quát từ chỉ ba ca thực nghiệm. Chiếc xe đồ chơi ảo giác tạo ra độ lệch pixel lớn nhất trong khi hoàn toàn thất bại về mặt ngữ nghĩa.
   - *Boundary Observation*: Mask bbox trên canvas là `[0, 340, 225, 512]`. Phân tích biên xác nhận bóng xe đồ chơi là một quầng oval tập trung dưới gầm xe, không chạm tới mép canvas ($x=0, y=512$). Tại các mép mask bên trong ($y=340$ và $x=225$), đá cuội hòa nhập tự nhiên không có gờ bậc. Khuyết tật là ảo giác ngữ nghĩa hoàn toàn, không phải lỗi ghép biên.
3. **`IND_COMMONS_SDXL_001` (Sky Bird)**:
   - *Arm A (Full Canvas)*: Complete semantic omission. Không tạo đối tượng; chỉ phủ mảng mây xám phẳng.
   - *Arm B (Local Crop)*: Tạo được đối tượng chim (hình bóng chim dang cánh bay ngược sáng).
   - *Đặc Biệt Kiểm Tra Vị Trí Chim (Placement Deficit)*:
     * Target đăng ký: `[395, 75, 455, 125]` (tọa độ canvas `[xmin, ymin, xmax, ymax]`).
     * Silhouette tối quan sát được: khoảng `[375, 126, 414, 156]` (hoặc $y \in [125, 156]$ ở các pixel viền khử răng cưa).
     * Độ lệch tâm hình học: $dx = -30.5\text{ px}$ (lệch sang trái), $dy = +41.0\text{ px}$ (lệch xuống dưới).
     * **Độ trùng lặp theo trục đứng: 0 px** (Target kết thúc ở $y=125$, silhouette bắt đầu ở $y=126$; chim hoàn toàn nằm dưới target box).
     * *Phương pháp và giới hạn phép đo*: Đo lại trên ảnh PNG bằng phân tích ngưỡng cường độ sáng trung bình RGB trên ảnh xám kết hợp kiểm chứng trực quan trên từng pixel. Giới hạn: ngưỡng màu không phải là segmentation ground truth tuyệt đối do hiệu ứng khử răng cưa (anti-aliasing) và tán xạ quang học tại viền lông; tuy nhiên trên toàn bộ dải ngưỡng từ 40 đến 160, silhouette tối luôn nằm trọn trong khoảng `[375, 126, 414, 156]`, hoàn toàn lệch khỏi target box `[395, 75, 455, 125]`.
     * *Phân biệt rõ ràng*: Chim **nằm hoàn toàn trong mask** `[350, 45, 500, 160]` (do đó compositing bảo toàn `outside_mean_l1 = 0.000000`), nhưng **KHÔNG đúng target placement**.
     * *Nguyên tắc nghiên cứu bất biến*: **Không dịch chuyển target hoặc mask sau khi xem kết quả để hợp thức hóa ảnh đầu ra.**
   - *Chất Lượng Biên & Vùng Trời Lệch Tông*: **Không mô tả là hòa nhập hoàn hảo.** Vùng trời chữ nhật bên trong mask `[350, 45, 500, 160]` bị lệch tông màu trung bình ($\Delta \text{RGB} \approx [-3.07, -3.79, -3.66]$) so với bầu trời authentic xung quanh, tạo ra một ranh giới chữ nhật mờ nhìn thấy được trên nền mây (bước nhảy tại mép phải đạt $-4.82$ R, $-4.57$ G, $-3.09$ B).

#### 2.8.4. Execution Evidence, Timing Scope, and Runtime Profiling

- **Crop and Resolution Verification**: Verified from raw generated images (`images/<id>_raw_gen.png`): SDXL Arm B generated at native $1024 \times 1024$ (linear 4.0×, area 16.0×); SD2 Arm B generated at native $512 \times 512$ with 1.6× padding (linear 1.6×, area 2.56×). Resized mask raster bboxes match rounding rules.
- **Timing Measurement Scope & Evidence**:
  - Measurement source: `diagnostic_receipt.json` và `attempt_ledger.jsonl` ghi nhận thời gian hoàn toàn khớp nhau cho cả 6 attempts:
    * `IND_COCO_SDXL_002`: Arm A `114.319s`, Arm B `24.684s`
    * `IND_COMMONS_SD2_002`: Arm A `71.542s`, Arm B `7.461s`
    * `IND_COMMONS_SDXL_001`: Arm A `37.614s`, Arm B `26.883s`
  - Scope in code (`ml/evaluation/independent_cohort_diagnostic.py`): `start_time = time.perf_counter()` được đo **trước** lệnh `get_engine(tool_key)`.
  - Đối với các attempt Arm A (1, 3, 5), `get_engine` khởi tạo `DiagnosticDiffusersEngine`, tải trọng số mô hình từ ổ đĩa/cache vào CUDA memory (`pipeline_cls.from_pretrained`) và nạp sang GPU (`pipeline.to("cuda")`).
  - Đối với các attempt Arm B (2, 4, 6), pipeline đã thường trú sẵn trong bộ nhớ GPU (`active_engine is not None and current_tool_key == tool_key`).
  - **Không kết luận tốc độ suy luận A/B từ thời gian toàn attempt**: Thời gian ghi nhận phản ánh tổng thời gian của attempt (gồm nạp mô hình khi khởi tạo tool key), không phản ánh tốc độ khử nhiễu UNet thuần túy. Chưa xác định tốc độ suy luận riêng của A/B từ thời gian toàn attempt; thực tế với SDXL, Arm B phải xử lý tensor lớn gấp 4 lần ($128 \times 128$ vs $64 \times 64$), thể hiện qua lượng VRAM đỉnh cao hơn (8.96 GiB vs 7.17 GiB).
- **Runtime Environment & Primary Sources**:
  - Inpainting models pinned in `ml/evaluation/independent_cohort_diagnostic.py`:
    * SDXL: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (official Diffusers model card, pinned revision `115134f363124c53c7d878647567d04daf26e41e`) `[@sdxlInpaintingModelCard]`
    * SD2: `sd2-community/stable-diffusion-2-inpainting` (community mirror, pinned revision `5f74973cbb64c8568780732c17f43eb269d63a0d`; not an official Stability AI source) `[@sd2CommunityInpaintingModelCard]`
  - Runtime environment recorded in `diagnostic_receipt.json`: OS `Linux 6.6.122+-x86_64-with-glibc2.39`, Python `3.13.15`, NumPy `2.1.3`, Pillow `11.3.0`.
  - Unrecorded fields: PyTorch version, Diffusers version, CUDA runtime version, và model GPU cụ thể không có trong receipt (được ghi nhận trung thực là chưa ghi nhận, không suy đoán).

#### 2.8.5. Methodological Scope & Unverified Hypotheses

1. **A/B Observational Finding**: Local crop at native resolution produced target objects for two SDXL insertion candidates (tomato, bird), whereas full canvas infilled background textures across all three Arm A attempts. For SD2, local crop produced a toy car hallucination. Báo cáo chính xác rằng **"hai ca Arm B tạo được đối tượng (cà chua, chim)"**, không suy diễn thành **"hai ca đạt đầy đủ Content QC"**.
2. **Causal Hypotheses Remain Unverified**:
   - Giả thuyết về việc suy giảm dung lượng latent token (8× downsampling) không thể được cô lập là nguyên nhân duy nhất, vì thao tác crop đồng thời thay đổi cửa sổ bối cảnh thị giác (thu hẹp trường nhìn) và tỷ lệ tương đối của token.
   - Giả thuyết về sự thiên lệch chú ý vào bối cảnh ô tô xung quanh gây ra ảo giác xe đồ chơi là một phỏng đoán hợp lý nhưng chưa được cô lập thực nghiệm.
   - Trên các lưới tensor có kích thước khác nhau ($64 \times 64$ vs $128 \times 128$), chuỗi số ngẫu nhiên PRNG của PyTorch khác biệt ngay cả khi dùng cùng seed nguyên.
3. **Phân biệt Quan sát Hình ảnh và Nhận định Nguyên nhân**: Cần phân biệt rạch ròi giữa quan sát trực quan (có/không có vật thể, vị trí silhouette, bậc tương phản tại biên mask, hình thái bóng đổ) với nhận định nguyên nhân cơ chế (vốn vẫn là giả thuyết).
4. **No Generalization Claims**: Với đúng 3 candidates và $n=1$ seed/candidate, đây là các quan sát phương pháp luận chẩn đoán. Không đề xuất loại mask < 3% như một quy tắc đã được chứng minh, không chốt giải pháp hybrid pipeline từ 3 ca chẩn đoán.

#### 2.8.6. Contact Sheet & Bảng Quyết định Thẩm định Con người (6 Attempts)

- **Contact Sheet Hoàn thiện**: Tệp HTML tự chứa hoàn toàn tại [`diagnostic_contact_sheet.html`](../../../data/research/local-artifacts/phase-4c.7b/diag-20261008T154628Z/diagnostic_contact_sheet.html) (11,222,571 bytes). Chứa ảnh authentic, overlay ground-truth, Arm A, Arm B, và các khung phóng đại (zoom) chi tiết với các hộp bounding box hiển thị target, mask và silhouette quan sát được, phân tích 4 tiêu chí rõ ràng.
- **Khóa Quản trị**: Full cohort generation tiếp tục **BỊ KHÓA HOÀN TOÀN**. Chưa đăng ký hoặc phê duyệt ngân sách generation mới. Kết quả chẩn đoán được cách ly trong `diag-20261008T154628Z` và không được đưa vào cohort chính thức.
- **Bảng Quyết định Duyệt Thẩm định Con người (6 Attempts)**:

| Attempt ID | Candidate / Prompt | Arm / Scale | Technical QC | Inside L1 | Quan sát Thực nghiệm A/B (4 Tiêu chí) | Khuyến nghị của Agent | Quyết định Thẩm định Con người |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `IND_COCO_SDXL_002_ARM_A` | Cà chua trên bánh mì | Arm A (1.0×) | PASS | 16.79 | Omission (phủ vân ruột bánh mì; không tạo vật thể) | Khuyến nghị **REJECT** (Bỏ sót vật thể) | **REJECT** (Omission) |
| `IND_COCO_SDXL_002_ARM_B` | Cà chua trên bánh mì | Arm B (4.0×) | PASS | 28.29 | Tạo đúng quả cà chua, có cuống đài xanh, độ bóng specular highlight; bậc tương phản vi mô tại biên mask `[345, 245, 455, 335]` | Trình Người dùng xem ảnh và đánh giá chất lượng | **REJECT** (Chưa đạt tiêu chí chất lượng biên vi mô & kết cấu ruột bánh mì) |
| `IND_COMMONS_SD2_002_ARM_A` | Vali du lịch trên phố | Arm A (1.0×) | PASS | 30.74 | Omission (phủ vân đá cuội; không tạo vật thể) | Khuyến nghị **REJECT** (Bỏ sót vật thể) | **REJECT** (Omission) |
| `IND_COMMONS_SD2_002_ARM_B` | Vali du lịch trên phố | Arm B (1.6×) | PASS | 43.39 | Hallucination (xe ô tô đồ chơi thay vì vali du lịch; bóng oval dưới gầm xe) | Khuyến nghị **REJECT** (Ảo giác ngữ nghĩa) | **REJECT** (Ảo giác xe đồ chơi) |
| `IND_COMMONS_SDXL_001_ARM_A` | Chim bay trên bầu trời | Arm A (1.0×) | PASS | 11.08 | Omission (phủ mảng mây xám phẳng; không tạo vật thể) | Khuyến nghị **REJECT** (Bỏ sót vật thể) | **REJECT** (Omission) |
| `IND_COMMONS_SDXL_001_ARM_B` | Chim bay trên bầu trời | Arm B (4.0×) | PASS | 13.07 | Tạo được đối tượng chim nhưng lệch vị trí target (`[395, 75, 455, 125]` vs `[375, 126, 414, 156]`) và vùng trời chữ nhật mask lệch tông | Khuyến nghị **REJECT** (Theo yêu cầu placement hiện tại) | **REJECT** (Lệch vị trí placement & lệch tông trời) |

*Ghi chú*: Người dùng đã thẩm định và chính thức quyết định **REJECT toàn bộ 6 diagnostic attempts (6/6 REJECT)**. Tám cặp ảnh pilot lịch sử (`pilot-20261008T113700Z`) tiếp tục giữ trạng thái **`PENDING_CONTENT_QC`**; full cohort tiếp tục **BỊ KHÓA HOÀN TOÀN**. Zero detector calls, independent performance `NOT_MEASURED`, full cohort `NOT_RUN`.

---

### 2.9. Kế hoạch Hiệu chuẩn Tiếp theo: Thử nghiệm So sánh Ảnh hưởng Guidance Scale 7.5 và 9.5 trong Cấu hình Prompt/Negative Prompt Cố định Mới (APPROVED by Human Reviewer)

> **Kế hoạch máy đọc**: `research/evidence/phase-4c.7b/content_grounded_calibration_proposal.json`<br>
> **Mã băm SHA-256 trước phê duyệt**: `03a811efa3c503c270d8827af20e25bc5720bfa8caa78b228013988315b5cadd`<br>
> **Mã băm SHA-256 sau phê duyệt**: `2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c`<br>
> **Trạng thái Quản trị**: `APPROVED` bởi người dùng Dũng Phạm `<valdung04@gmail.com>` tại `2026-10-08T19:34:30Z` (Phê duyệt đúng 2 attempts trên `IND_COCO_SDXL_002`, canvas 512×512, seed 20272319, zero retries, zero automatic replacement; chỉ chạy trên GPU Colab khi người dùng kích hoạt, chưa thực thi trong phiên làm việc local này).<br>
> **Ngân sách đã duyệt**: Đúng **2 attempts** một lần (zero retries, zero automatic replacement).

#### 2.9.1. Lỗi đã Quan sát Thực nghiệm vs Giả thuyết Còn Cần Kiểm chứng
1. **Lỗi Đã Quan Sát Thực Nghiệm (Empirically Observed Defects)**:
   - *Omission (Bỏ sót vật thể)*: Cả ba ca Arm A đều không tạo đối tượng yêu cầu trên canvas 512×512 (`IND_COCO_SDXL_002`, `IND_COMMONS_SD2_002`, `IND_COMMONS_SDXL_001` ở cả pilot và diagnostic Arm A). Mô hình sinh texture lấp đầy phẳng (infill smoothing) thay vì tạo vật thể được yêu cầu trong prompt.
   - *Semantic Hallucination (Ảo giác ngữ nghĩa)*: Xuất hiện ở `IND_COMMONS_SD2_002_ARM_B` (SD2 sinh xe ô tô đồ chơi thay vì vali du lịch), và `IND_COMMONS_SDXL_003` (SDXL sinh cột chai thủy tinh kỳ dị phản quang neon thay vì cột đá cẩm thạch La Mã).
   - *Placement Deficit & Tonal Boundary Step*: Xuất hiện ở `IND_COMMONS_SDXL_001_ARM_B` (chim tạo ra dạt xuống dưới-trái [375, 126, 414, 156], hoàn toàn lệch khỏi target bbox [395, 75, 455, 125]; vùng trời chữ nhật mask lệch tông so với nền).
   - *Macroscopic Structural Severance (Đứt gãy hình học vĩ mô)*: Bounding box chữ nhật cắt ngang qua các cấu trúc vật lý liên tục (lan can kim loại ở `IND_COMMONS_SDXL_003` bị cắt đứt 210 px; trần thạch cao ở `IND_COCO_SDXL_041` bị lệch tông màu và độ nhám tại $y=155$; tường bếp ở `IND_COCO_SD2_001` tại $y=95$).
   - *Microscopic Seams (Bậc tương phản vi mô tại biên mask)*: Tạo ra bởi phép ghép nhị phân 1-bit (`Image.composite`) cắt ngang cấu trúc hạt/vân (như ruột bánh mì ở `IND_COCO_SDXL_002_ARM_B`).
2. **Nhận Định Cơ Chế Tiếp Tục Là Giả Thuyết (Unverified Hypotheses)**:
   - *Giả thuyết VAE downsampling / Latent Token Capacity*: Chưa được đo lường trực tiếp trên tensor latent (chỉ suy luận lý thuyết từ kiến trúc $8\times$ downsampling).
   - *Giả thuyết Cross-Attention Infill Bias*: Giả định rằng embedding của bối cảnh không masked lấn át token vật thể trong cross-attention; chưa được kiểm chứng qua attention map trích xuất từ UNet.
   - *Giả thuyết Ngữ cảnh ô tô gây ảo giác xe đồ chơi*: Có thể do bối cảnh ô tô gần đó, nhưng cũng có thể do biến đổi PRNG noise field trên kích thước crop hoặc bias của SD2 checkpoint.

#### 2.9.2. Phân biệt Thành công Tạo đúng Đối tượng vs Các Tiêu chí Content QC
- **Hiện diện đúng Đối tượng (Semantic Object Presence)**: Yêu cầu tiên quyết về mặt nội dung (có hay không có quả cà chua, con chim).
- **Vị trí và Tỷ lệ (Placement & Scale)**: Đối tượng tạo ra phải nằm trong vùng target bounding box đã đăng ký, với tỷ lệ hợp lý.
- **Độ Chân thực (Realism & Lighting)**: Hình thái, độ sắc nét, độ bóng specular highlight và kết cấu nền so với ảnh authentic.
- **Chất lượng Biên (Boundary Integrity)**: Ranh giới ghép nhị phân không tạo bậc tương phản vi mô, lệch tông màu hay đường viền chữ nhật lộ liễu.
- **Báo Cáo Chính Xác**: "Hai ca Arm B tạo được đối tượng (cà chua, chim)" không có nghĩa là "hai ca đạt đầy đủ Content QC". Chim vi phạm vị trí placement và lệch tông trời; cà chua cần người dùng đánh giá độ sắc, độ bóng và kết cấu nền.

#### 2.9.3. Nhận Định Về Shortcut Detector (Spurious Feature Shortcut)
- Trong nghiên cứu phát hiện ảnh giả mạo (image forensics), mục tiêu là huấn luyện và đánh giá các detector phân biệt ảnh do AI tạo sinh (`ai_edited`) dựa trên các đặc trưng nội tại của quá trình sinh ảnh (diffusion artifacts, tần số bất thường, thống kê residual nhiễu).
- **Nguy Cơ / Giả Thuyết Shortcut (Spurious Feature Risk)**:
  - Nếu ảnh trong cohort độc lập mang các dấu vết biên nhân tạo quá rõ nét (như viền cắt 1-pixel sắc nhọn từ phép ghép nhị phân, hoặc vùng làm mịn nhân tạo do feathering cố định $k=2$ px, hoặc ranh giới chữ nhật hoàn hảo cắt qua kết cấu ảnh), các mô hình detector (đặc biệt là các nhánh DSP FFT/DCT và noise residual) có nguy cơ học được **shortcut phân biệt dựa vào viền ghép hình học** thay vì học các đặc trưng sinh ảnh của mô hình diffusion.
  - **Giữ là Nguy cơ / Giả thuyết**: Nhận định về shortcut detector tiếp tục được giữ là **nguy cơ / giả thuyết** cần kiểm soát khi chưa có bằng chứng thực nghiệm đối chứng; **tuyệt đối không tuyên bố hiệu năng của detector đã bị thổi phồng**.
- **Nguyên tắc Quản trị**: Cần giữ nguyên tắc thận trọng tối đa, không tùy tiện áp dụng feathering cố định vào production khi chưa có thẩm định thực nghiệm đối chứng trên detector.

#### 2.9.4. Không Chốt Giải pháp Hybrid hay Ngưỡng Loại Mask từ Ba Ca Diagnostic
- Ba ca chẩn đoán ($n=3$ candidates, 1 seed/candidate) chỉ là khảo sát phương pháp luận. Không đủ độ khái quát thống kê để thiết lập quy tắc loại mask < 3% hoặc cam kết kiến trúc pipeline hybrid.
- Mọi quyết định thay đổi pipeline cohort chính thức tiếp tục bị đóng băng cho đến khi hoàn tất thẩm định con người.

#### 2.9.5. Đề xuất Một Thử nghiệm Hiệu chuẩn Đơn biến Cụ thể (PENDING Review)
Nhằm so sánh ảnh hưởng của guidance scale 7.5 và 9.5 trong một cấu hình prompt/negative prompt cố định mới, trên một candidate (`IND_COCO_SDXL_002`) và một seed (`20272319`) mà **không cần thay đổi trường nhìn (crop) hay độ phân giải canvas**, đề xuất một thử nghiệm nhỏ:

1. **Mục tiêu Đã Điều chỉnh**: So sánh ảnh hưởng của guidance scale 7.5 và 9.5 trong một cấu hình prompt/negative prompt cố định mới, trên một candidate và một seed.
   - *Giới hạn phương pháp luận*: Phép thử **không tuyên bố tách riêng ảnh hưởng của prompt hoặc negative prompt**, **không tuyên bố chứng minh infill bias**, và **không phải là ngân sách tối thiểu để xác định nguyên nhân omission**.
   - *Tính chất đối chiếu*: Việc so sánh với kết quả lịch sử (pilot hoặc diagnostic) chỉ mang tính **tham khảo** vì nhiều thành phần cấu hình văn bản (cả prompt và negative prompt) đã thay đổi đồng thời.
2. **Biến Thay đổi (Independent Variables)**:
   - *Attempt 1 (`CALIB_COCO_SDXL_002_PROMPT_G75`)*:
     * Prompt mới: `"a ripe red cherry tomato with shiny skin, distinct green calyx stem, sharp focus, natural daylight photography"`
     * Negative prompt mới: `"empty, blurry, smooth texture, missing object, bread crumb only, background infill"`
     * Guidance scale: `7.5`
   - *Attempt 2 (`CALIB_COCO_SDXL_002_PROMPT_G95`)*:
     * Cùng prompt mới và negative prompt mới như Attempt 1.
     * Guidance scale: `9.5`
3. **Biến Kiểm soát Cố định (Controlled Invariants)**:
   - Ứng viên cố định: `IND_COCO_SDXL_002` (ảnh authentic hash `7aefc1d1...` và mask hash `2ff6b165...` niêm phong từ `pilot-20261008T113700Z`).
   - Seed cố định: `20272319` (cùng seed với pilot và diagnostic).
   - Canvas cố định: Full canvas $512 \times 512$ (không crop, không rescale).
   - Mô hình cố định: SDXL Inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, pinned revision `115134f363124c53c7d878647567d04daf26e41e`) `[@sdxlInpaintingModelCard]`.
   - Scheduler & Steps: EulerDiscrete, 30 steps, strength 1.0.
   - Ghép ảnh: Phép ghép nhị phân bảo toàn bit-exact ngoài mask (`outside_mean_l1 = 0.000000`).
4. **Tiêu chí Đánh giá (Evaluation Criteria)**:
   - *Hiện diện đúng đối tượng (Object presence)*: Có xuất hiện quả cà chua cherry hay không.
   - *Vị trí và tỷ lệ (Position & scale)*: Có nằm trong target bounding box [375, 265, 430, 320] hay không.
   - *Độ chân thực (Realism)*: Hình thái quả, cuống đài, độ sắc nét, độ bóng specular highlight so với lát bánh mì.
   - *Chất lượng biên (Boundary quality)*: Bước chuyển tiếp vi mô dọc theo biên mask [345, 245, 455, 335].
   - *Inside Mean L1 Delta*: Đo lường mức độ biến đổi pixel trong mask.
5. **Ngân sách Đề xuất**:
   - Ngân sách đề xuất: **Đúng 2 attempts** ($N=2$).
6. **Ràng buộc Quản trị**:
   - Trạng thái kế hoạch: **`APPROVED`** bởi người dùng Dũng Phạm `<valdung04@gmail.com>` tại `2026-10-08T19:34:30Z` (đúng 2 attempts trên `IND_COCO_SDXL_002`, canvas 512×512, seed 20272319).
   - Tuyệt đối **KHÔNG tự động thực thi generation** trong phiên làm việc local này.
   - Không đưa ảnh thử nghiệm vào cohort chính thức. Full cohort tiếp tục **BỊ KHÓA HOÀN TOÀN**.
