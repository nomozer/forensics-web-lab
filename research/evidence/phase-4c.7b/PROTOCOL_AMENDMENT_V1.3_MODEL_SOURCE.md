# Protocol Amendment v1.3.1: Stable Diffusion 2 Inpainting Model Source Amendment

> **Status**: LOCKED (Pre-Acquisition Amendment)<br>
> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Effective Date**: 2026-10-07<br>
> **Supersedes**: Inpainting model source reference in Protocol Amendment v1.2 (`PROTOCOL_AMENDMENT_V1.2.md`) and Cohort Specification v1.3.0 (`cohort_specification.json`) specifically regarding the Stable Diffusion 2 Inpainting checkpoint repository.<br>
> **Preserves**: All scientific invariants from Phase 4C.7A / Amendment v1.2 / Amendment v1.3:
> - Target sample size $N_{\text{target}}=400$ pairs (440 buffer pool across 4 strata: `coco_sd2`, `coco_sdxl`, `commons_sd2`, `commons_sdxl`, 100 target / 110 buffer per stratum).
> - Binary classification contract (`authentic` vs `ai_edited`).
> - Model architecture: Stable Diffusion 2 Inpainting (9-channel UNet input, 512x512 resolution).
> - Generation parameters: DDIM scheduler, 50 inference steps, guidance scale 7.5, fixed PRNG seeds, exact candidate prompts.
> - SDXL Inpainting 1.0 parameters: EulerDiscreteScheduler, 30 inference steps, guidance scale 7.5.
> - Quotas: 40% replacement / 30% removal / 30% insertion; mask area 30% small / 40% medium / 30% large.
> - Prohibition on automated fallback: zero fallback between models.
> - Zero Detector Scoring invariant: zero detector inference during cohort acquisition.

---

## 1. Incident Report: Run `pilot-20261007T082516Z`

- **Execution Context**: Google Colab T4 GPU runtime, commit `3710762868a007af4fe79798bad79d086abcd5c8`.
- **Preflight Outcome**: GPU policy check PASSED (`Tesla T4`, 15.0 GiB VRAM).
- **Failure Point**: Inpainting runner halted at `HfApi().model_info("stabilityai/stable-diffusion-2-inpainting")` with `RepositoryNotFoundError` / HTTP 401 Unauthorized (`{"error":"Invalid username or password."}`).
- **Receipt & Ledger Status**: 0 completed receipts written; attempt ledger was observed empty.
- **Root Cause Analysis**:
  Hugging Face returns HTTP 401 for unauthenticated REST requests when a repository is deleted, gated, private, or deprecated by the organization (to prevent unauthorized enumeration). Stability AI has deprecated/restricted `stabilityai/stable-diffusion-2-inpainting`.
  The error does **not** indicate a missing user login token; the system must operate on open, public weights without requiring or exposing private authentication tokens.

---

## 2. Alternative Source Appraisal: `sd2-community/stable-diffusion-2-inpainting`

An empirical audit of public Hugging Face repositories identified the community mirror `sd2-community/stable-diffusion-2-inpainting`.

### 2.1. Model Card & Affiliation Disclaimer
The model card explicitly states:
> *"⚠️ This repository is a mirror of the now deprecated `stabilityai/stable-diffusion-2-inpainting`, this repository and organization are not affiliated in any way with Stability AI."*

**Scientific Honesty Standard**:
- This repository is classified as a **community mirror** (`community_mirror`), **not** an official Stability AI publication.
- We make **no claim of bit-exact weight identity** to the original Stability AI checkpoint without direct comparative proof.
- It provides open, unauthenticated access to the standard Diffusers Stable Diffusion 2 Inpainting pipeline weights.

### 2.2. License & Legal Scope
- **License**: CreativeML OpenRAIL++ (`openrail++`).
- **Permitted Uses**: Research evaluation, benchmarking, derivative generation with standard OpenRAIL behavioral restrictions.

### 2.3. Architecture & Configurations Audit
Fetched and audited via live REST query at revision `5f74973cbb64c8568780732c17f43eb269d63a0d`:
- `model_index.json`: Pipeline class `StableDiffusionInpaintPipeline`, diffusers 0.8.0.
- `unet/config.json`: `UNet2DConditionModel`, `in_channels: 9`, `out_channels: 4`, `sample_size: 64`.
- `scheduler/scheduler_config.json`: Compatible with `DDIMScheduler`.
- `vae/config.json`: `AutoencoderKL`.
- `text_encoder/config.json`: `CLIPTextModel`.

### 2.4. Component Hashes & File Sizes (LFS OID Verification)
- **UNet fp16 weights**: `unet/diffusion_pytorch_model.fp16.safetensors`
  - LFS OID: `29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8`
  - Size: 1,731,933,536 bytes
- **UNet fp32 weights**: `unet/diffusion_pytorch_model.safetensors`
  - LFS OID: `9bcbb17f54b039f58bf78677fab8cd8a35dd686f6c9dd553e3646a8b0aaff41a`
  - Size: 3,463,784,116 bytes
- **Checkpoint bundle**: `512-inpainting-ema.safetensors`
  - LFS OID: `b29e2ed9a8fe58e76f7e801bda091d23738bd74c1da3f339bcbe2d40922fcb60`
  - Size: 5,214,662,094 bytes
- **VAE fp16 weights**: `vae/diffusion_pytorch_model.fp16.safetensors`
  - LFS OID: `3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a`
  - Size: 167,335,342 bytes
- **Text Encoder fp16 weights**: `text_encoder/model.fp16.safetensors`
  - LFS OID: `681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668`
  - Size: 680,821,096 bytes

---

## 3. SDXL Inpainting 1.0 Source Audit

- **Repository**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- **Classification**: Official Diffusers release (`official_diffusers`).
- **Pinned Revision SHA**: `115134f363124c53c7d878647567d04daf26e41e`
- **License**: CreativeML OpenRAIL++ (`openrail++`).
- **UNet fp16 weights**: `unet/diffusion_pytorch_model.fp16.safetensors`
  - LFS OID: `6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f`
  - Size: 5,135,178,560 bytes

---

## 4. Fail-Closed Model Preflight Contract

Before creating any run directory on disk, the acquisition CLI must execute a fail-closed preflight check:
1. **Metadata API check**:
   - Query `https://huggingface.co/api/models/{repo_id}`.
   - Distinguish HTTP 401 (unavailable / deprecated / gated / private), HTTP 404 (not found), HTTP 403 (forbidden), and network errors.
   - Fail closed if repository is gated or private.
   - Resolve and record the commit SHA.
2. **Config file check**:
   - Query raw endpoint for `model_index.json`, `unet/config.json`, and `scheduler/scheduler_config.json`.
   - Parse and validate JSON structure.
3. **Weight file HEAD check**:
   - Perform HTTP HEAD request against safetensors weight URL to verify accessibility and content-length without downloading full multi-gigabyte weights.
4. **Zero Dirty Run Directories**:
   - Preflight execution occurs strictly **prior** to `run_dir` creation. Any failure stops the process immediately, leaving no empty or dirty run directories.
