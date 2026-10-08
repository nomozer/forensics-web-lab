#!/usr/bin/env python3
"""Generate the 6-attempt diagnostic calibration plan and coordinate overlay visualizations.

Candidates:
- IND_COCO_SDXL_002 (bread tomato)
- IND_COMMONS_SD2_002 (suitcase)
- IND_COMMONS_SDXL_001 (sky bird)

Arms per candidate:
- Arm A: Full-canvas 512x512 inference under current locked pipeline configuration
- Arm B: Local crop with padding at model native resolution, remapped and composited

All seeds, models, schedulers, steps, prompts, and parameters are locked to match ledger values.
Status remains strictly PENDING.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"
IMAGES_DIR = RUN_DIR / "images"
MASKS_DIR = RUN_DIR / "masks"
PLAN_OUTPUT_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json"
OVERLAYS_DIR = RUN_DIR / "diagnostic_overlays"


DIAGNOSTIC_PLAN = {
    "schema_version": "1.0.0",
    "workstream": "independent_cohort_acquisition_diagnostic",
    "phase_trace": "4C.7B",
    "purpose": (
        "Proposed 6-attempt diagnostic calibration plan comparing full-canvas inference (Arm A) "
        "vs local-crop padded inference (Arm B) across 3 object insertion omission candidates. "
        "This comparison evaluates methodological differences between full-canvas and local-crop execution "
        "(including context window alteration, token scale shift, and geometric area scaling) and does NOT "
        "claim to isolate latent downsampling alone. Identical seeds across different tensor resolutions "
        "do not guarantee identical noise fields due to dimension-dependent sampling order."
    ),
    "human_review_status": "PENDING",
    "human_reviewer": None,
    "human_reviewed_at_utc": None,
    "coordinate_space": "normalized_512x512_canvas_xyxy_half_open",
    "attempt_budget": 6,
    "automatic_replacement": False,
    "bound_pilot_run_id": "pilot-20261008T113700Z",
    "bound_pilot_zip_sha256": "3bdb1890d2aa6d34bb2829e159408ef44ab30d630c6f85792fce5fb7ade9e2b4",
    "bound_pilot_manifest_sha256": "cf0e4300d1e6f760e82ab76fdd29ffd534b35577003ae8ccf4ef9a9c25799704",
    "input_source_policy": (
        "Must use sealed normalized PNGs from pilot-20261008T113700Z as inputs; "
        "remote redownload from web is strictly prohibited."
    ),
    "review_criteria": [
        "object_semantic_presence: Was the requested object generated inside the masked region?",
        "position_and_scale: Is the generated object properly positioned within the target bbox with plausible relative scale?",
        "lighting_and_perspective: Does lighting, shading, color temperature, and perspective match the authentic background?",
        "boundary_integrity: Are edge transitions clean without visible boundary step seams or halos?",
        "outside_mask_invariance: Every pixel outside the registered mask must remain bit-exact identical to the authentic input (outside_mean_L1 == 0.000000)."
    ],
    "governance_notes": [
        "Diagnostic calibration experiment: 3 candidates x 2 arms (Arm A: Full Canvas vs Arm B: Local Crop with Padding) = exactly 6 attempts.",
        "This diagnostic budget is distinct from and does not alter the 8 production pilot attempts already consumed in pilot-20261008T113700Z.",
        "Results of this diagnostic will NOT automatically enter the official independent cohort or overwrite previous pilot records.",
        "With n=3 cases and 1 seed per case, observations are exploratory and cannot be generalized without broader evaluation.",
        "Negative prompt and feathering are intentionally excluded from this comparison.",
        "Geometric area ratio is 16.0 for SDXL cases and 2.56 for SD2 suitcase. Latent pixel numbers are methodological estimates assuming 8x VAE downsampling, not directly measured latent tensor tokens.",
        "Production cohort specification, approved plans, and full cohort lock remain strictly UNMODIFIED and LOCKED."
    ],
    "resizing_rules": {
        "rgb_resampling": "LANCZOS",
        "mask_resampling": "NEAREST",
        "mask_dtype_invariance": "Mask after resize must strictly contain only {0, 255} binary values."
    },
    "diagnostic_candidates": [
        {
            "candidate_id": "IND_COCO_SDXL_002",
            "common_name": "bread_tomato",
            "stratum_id": "coco_sdxl",
            "tool_key": "sdxl_inpainting",
            "model_checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "model_revision": "115134f363124c53c7d878647567d04daf26e41e",
            "origin_id": "coco:293044",
            "authentic_input_filename": "IND_COCO_SDXL_002_auth.png",
            "authentic_sha256": "7aefc1d1dff39ac5527ce47734421e34094af42d9763fe35ab3b94fee62e4571",
            "mask_input_filename": "IND_COCO_SDXL_002_mask.png",
            "mask_sha256": "2ff6b16571048015060095e9a240b28ed1073e9ea70a9d7870bfaf3b49fb9bee",
            "generation_seed": 20272319,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "scheduler": "EulerDiscreteScheduler",
            "strength": 1.0,
            "negative_prompt": None,
            "prompt": "a small red cherry tomato resting on the slice of bread, matching the lunchbox lighting and camera angle",
            "target_description": "a small red cherry tomato to insert on the bread slice",
            "target_bbox_xyxy": [375, 265, 430, 320],
            "mask_bbox_xyxy": [345, 245, 455, 335],
            "mask_area_class": "small_under_10pct",
            "mask_area_pct": 3.776550,
            "mask_pixel_count": 9900,
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COCO_SDXL_002_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "geometric_area_scale_factor": 1.0,
                    "local_mask_xyxy": [345, 245, 455, 335],
                    "estimated_latent_mask_dimensions": [13, 11],
                    "estimated_latent_mask_pixels": 143,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [345, 245, 455, 335]. Outside pixels remain exactly 0.000000."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COCO_SDXL_002_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [256, 162, 512, 418],
                    "crop_dimensions": [256, 256],
                    "padding_margins": {"left": 89, "right": 57, "top": 83, "bottom": 83},
                    "local_mask_xyxy": [89, 83, 199, 173],
                    "inference_resolution": [1024, 1024],
                    "linear_scale_factor": 4.0,
                    "geometric_area_scale_factor": 16.0,
                    "coordinate_mapping_rule": "Crop 256x256 scaled 4.0x to 1024x1024. Local mask [89, 83, 199, 173] maps to [356, 332, 796, 692] via integer scale 4.",
                    "scaled_mask_in_inference": [356, 332, 796, 692],
                    "scaled_mask_raster_bbox": [356, 332, 796, 692],
                    "scaled_mask_pixel_count": 158400,
                    "scaled_mask_unique_values": [0, 255],
                    "estimated_latent_mask_dimensions": [55, 45],
                    "estimated_latent_mask_pixels": 2475,
                    "mapping_pipeline": "Crop 256x256 from canvas -> Upscale crop and local mask to 1024x1024 (RGB Lanczos, Mask Nearest) -> SDXL inpainting at 1024x1024 -> Downscale generated crop to 256x256 via Lanczos -> Remap crop to canvas [256, 162, 512, 418] -> Composite into 512x512 canvas strictly inside original mask [345, 245, 455, 335]. Outside pixels remain exactly 0.000000."
                }
            }
        },
        {
            "candidate_id": "IND_COMMONS_SD2_002",
            "common_name": "travel_suitcase",
            "stratum_id": "commons_sd2",
            "tool_key": "stable_diffusion_2_inpainting",
            "model_checkpoint": "sd2-community/stable-diffusion-2-inpainting",
            "model_revision": "5f74973cbb64c8568780732c17f43eb269d63a0d",
            "origin_id": "commons:81567907",
            "authentic_input_filename": "IND_COMMONS_SD2_002_auth.png",
            "authentic_sha256": "98004bf740d5ec04e2144a6a436c1926a2a6b208983de3a375bd74a04bef206b",
            "mask_input_filename": "IND_COMMONS_SD2_002_mask.png",
            "mask_sha256": "fee9ac7ec5f6c14b0692290b2f68daebc6bdd2f5b38ecfeae85e4fc45d145a94",
            "generation_seed": 20283429,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "scheduler": "DDIMScheduler",
            "strength": 1.0,
            "negative_prompt": None,
            "prompt": "a brown leather travel suitcase standing on the cobblestones beside the vintage car, realistic scale and daylight shadows",
            "target_description": "a brown leather travel suitcase to insert on the cobblestones beside the car",
            "target_bbox_xyxy": [45, 355, 190, 495],
            "mask_bbox_xyxy": [0, 340, 225, 512],
            "mask_area_class": "medium_10_to_30pct",
            "mask_area_pct": 14.762878,
            "mask_pixel_count": 38700,
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COMMONS_SD2_002_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "geometric_area_scale_factor": 1.0,
                    "local_mask_xyxy": [0, 340, 225, 512],
                    "estimated_latent_mask_dimensions": [28, 21],
                    "estimated_latent_mask_pixels": 588,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [0, 340, 225, 512]. Outside pixels remain exactly 0.000000."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COMMONS_SD2_002_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [0, 192, 320, 512],
                    "crop_dimensions": [320, 320],
                    "padding_margins": {"left": 0, "right": 95, "top": 148, "bottom": 0},
                    "local_mask_xyxy": [0, 148, 225, 320],
                    "inference_resolution": [512, 512],
                    "linear_scale_factor": 1.6,
                    "geometric_area_scale_factor": 2.56,
                    "coordinate_mapping_rule": (
                        "Local crop (320x320) scaled by factor 1.6 (512/320) to 512x512. "
                        "Continuous mapping: x in [0.0, 360.0], y in [236.8, 512.0]. "
                        "Nearest-neighbor resampling samples source y=floor(y_dst / 1.6); "
                        "row 236 samples source row 147 (value 0), row 237 samples source row 148 (value 255), "
                        "yielding raster bbox [0, 237, 360, 512]."
                    ),
                    "scaled_mask_in_inference": [0, 237, 360, 512],
                    "scaled_mask_raster_bbox": [0, 237, 360, 512],
                    "scaled_mask_pixel_count": 99000,
                    "scaled_mask_unique_values": [0, 255],
                    "estimated_latent_mask_dimensions": [45, 34],
                    "estimated_latent_mask_pixels": 1530,
                    "mapping_pipeline": "Crop 320x320 from canvas -> Upscale crop and local mask to 512x512 (RGB Lanczos, Mask Nearest) -> SD2 inpainting at 512x512 -> Downscale generated crop to 320x320 via Lanczos -> Remap crop to canvas [0, 192, 320, 512] -> Composite into 512x512 canvas strictly inside original mask [0, 340, 225, 512]. Outside pixels remain exactly 0.000000."
                }
            }
        },
        {
            "candidate_id": "IND_COMMONS_SDXL_001",
            "common_name": "sky_bird",
            "stratum_id": "commons_sdxl",
            "tool_key": "sdxl_inpainting",
            "model_checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "model_revision": "115134f363124c53c7d878647567d04daf26e41e",
            "origin_id": "commons:166503140",
            "authentic_input_filename": "IND_COMMONS_SDXL_001_auth.png",
            "authentic_sha256": "7680e4ced18e54a188202847955a17add141c879df4278c1824de52690ae90cc",
            "mask_input_filename": "IND_COMMONS_SDXL_001_mask.png",
            "mask_sha256": "4981cedcdeff5cc4ca9d785e7debc6cd93a09f8f40d5295ba1cb8516844ab084",
            "generation_seed": 20294438,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "scheduler": "EulerDiscreteScheduler",
            "strength": 1.0,
            "negative_prompt": None,
            "prompt": "a small dark bird flying in the cloudy sky, distant scale and natural daylight",
            "target_description": "a small dark bird to insert in the open cloudy sky at upper-right",
            "target_bbox_xyxy": [395, 75, 455, 125],
            "mask_bbox_xyxy": [350, 45, 500, 160],
            "mask_area_class": "small_under_10pct",
            "mask_area_pct": 6.580353,
            "mask_pixel_count": 17250,
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COMMONS_SDXL_001_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "geometric_area_scale_factor": 1.0,
                    "local_mask_xyxy": [350, 45, 500, 160],
                    "estimated_latent_mask_dimensions": [18, 14],
                    "estimated_latent_mask_pixels": 252,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [350, 45, 500, 160]. Outside pixels remain exactly 0.000000."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COMMONS_SDXL_001_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [256, 0, 512, 256],
                    "crop_dimensions": [256, 256],
                    "padding_margins": {"left": 94, "right": 12, "top": 45, "bottom": 96},
                    "local_mask_xyxy": [94, 45, 244, 160],
                    "inference_resolution": [1024, 1024],
                    "linear_scale_factor": 4.0,
                    "geometric_area_scale_factor": 16.0,
                    "coordinate_mapping_rule": "Crop 256x256 scaled 4.0x to 1024x1024. Local mask [94, 45, 244, 160] maps to [376, 180, 976, 640] via integer scale 4.",
                    "scaled_mask_in_inference": [376, 180, 976, 640],
                    "scaled_mask_raster_bbox": [376, 180, 976, 640],
                    "scaled_mask_pixel_count": 276000,
                    "scaled_mask_unique_values": [0, 255],
                    "estimated_latent_mask_dimensions": [75, 57],
                    "estimated_latent_mask_pixels": 4275,
                    "mapping_pipeline": "Crop 256x256 from canvas -> Upscale crop and local mask to 1024x1024 (RGB Lanczos, Mask Nearest) -> SDXL inpainting at 1024x1024 -> Downscale generated crop to 256x256 via Lanczos -> Remap crop to canvas [256, 0, 512, 256] -> Composite into 512x512 canvas strictly inside original mask [350, 45, 500, 160]. Outside pixels remain exactly 0.000000."
                }
            }
        }
    ]
}


def render_coordinate_overlays():
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    for cand in DIAGNOSTIC_PLAN["diagnostic_candidates"]:
        cid = cand["candidate_id"]
        auth_path = IMAGES_DIR / f"{cid}_auth.png"
        if not auth_path.is_file():
            print(f"Skipping overlay for {cid}: authentic image not found at {auth_path}")
            continue

        im = Image.open(auth_path).convert("RGBA")
        overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Arm B crop bbox (Green dashed / transparent fill)
        crop_bbox = cand["arms"]["arm_b_local_crop"]["crop_bbox_xyxy"]
        draw.rectangle(crop_bbox, outline=(34, 197, 94, 255), width=3)
        draw.rectangle(crop_bbox, fill=(34, 197, 94, 40))

        # 2. Mask bbox (Red)
        mask_bbox = cand["mask_bbox_xyxy"]
        draw.rectangle(mask_bbox, outline=(239, 68, 68, 255), width=2)
        draw.rectangle(mask_bbox, fill=(239, 68, 68, 60))

        # 3. Target bbox (Cyan)
        target_bbox = cand["target_bbox_xyxy"]
        draw.rectangle(target_bbox, outline=(6, 182, 212, 255), width=2)
        draw.rectangle(target_bbox, fill=(6, 182, 212, 50))

        # Labels
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        draw.text((crop_bbox[0] + 6, crop_bbox[1] + 6), "Arm B Crop (Green)", fill=(34, 197, 94, 255), font=font)
        draw.text((mask_bbox[0] + 4, mask_bbox[1] + 4), "Mask (Red)", fill=(239, 68, 68, 255), font=font)
        draw.text((target_bbox[0] + 4, target_bbox[1] + 4), "Target (Cyan)", fill=(6, 182, 212, 255), font=font)

        composited = Image.alpha_composite(im, overlay).convert("RGB")
        out_path = OVERLAYS_DIR / f"{cid}_diagnostic_overlay.png"
        composited.save(out_path)
        print(f"Rendered overlay: {out_path}")


def main():
    PLAN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan_bytes = json.dumps(DIAGNOSTIC_PLAN, indent=2).encode("utf-8")
    PLAN_OUTPUT_PATH.write_bytes(plan_bytes)
    sha256 = hashlib.sha256(plan_bytes).hexdigest()
    print(f"Wrote diagnostic plan to: {PLAN_OUTPUT_PATH}")
    print(f"Plan SHA-256: {sha256}")

    render_coordinate_overlays()


if __name__ == "__main__":
    main()
