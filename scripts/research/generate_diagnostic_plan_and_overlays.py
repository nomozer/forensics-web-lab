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

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"
IMAGES_DIR = RUN_DIR / "images"
PLAN_OUTPUT_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_diagnostic_plan.json"
OVERLAYS_DIR = RUN_DIR / "diagnostic_overlays"


DIAGNOSTIC_PLAN = {
    "schema_version": "1.0.0",
    "workstream": "independent_cohort_acquisition_diagnostic",
    "phase_trace": "4C.7B",
    "purpose": "Proposed 6-attempt diagnostic calibration plan comparing full-canvas inference (Arm A) vs local-crop inference with padding (Arm B) across 3 object insertion omission candidates.",
    "human_review_status": "PENDING",
    "human_reviewer": None,
    "human_reviewed_at_utc": None,
    "coordinate_space": "normalized_512x512_canvas_xyxy_half_open",
    "attempt_budget": 6,
    "automatic_replacement": False,
    "governance_notes": [
        "Diagnostic calibration experiment: 3 candidates x 2 arms (Arm A: Full Canvas vs Arm B: Local Crop with Padding) = exactly 6 attempts.",
        "This diagnostic budget is distinct from and does not alter the 8 production pilot attempts already consumed in pilot-20261008T113700Z.",
        "Results of this diagnostic will NOT automatically enter the official independent cohort or overwrite previous pilot records.",
        "With n=3 cases and 1 seed per case, observations are exploratory and cannot be generalized without broader evaluation.",
        "Negative prompt and feathering are intentionally excluded from this comparison to strictly isolate the latent spatial capacity factor.",
        "Production cohort specification, approved plans, and full cohort lock remain strictly UNMODIFIED and LOCKED."
    ],
    "diagnostic_candidates": [
        {
            "candidate_id": "IND_COCO_SDXL_002",
            "common_name": "bread_tomato",
            "stratum_id": "coco_sdxl",
            "tool_key": "sdxl_inpainting",
            "model_checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "model_revision": "115134f363124c53c7d878647567d04daf26e41e",
            "origin_id": "coco:293044",
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
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COCO_SDXL_002_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "upscale_factor": 1.0,
                    "local_mask_xyxy": [345, 245, 455, 335],
                    "latent_mask_dimensions_approx": [13, 11],
                    "latent_mask_pixels_approx": 143,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [345, 245, 455, 335]."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COCO_SDXL_002_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [256, 162, 512, 418],
                    "crop_dimensions": [256, 256],
                    "padding_margins": {"left": 89, "right": 57, "top": 83, "bottom": 83},
                    "local_mask_xyxy": [89, 83, 199, 173],
                    "inference_resolution": [1024, 1024],
                    "upscale_factor": 4.0,
                    "scaled_mask_in_inference": [356, 332, 796, 692],
                    "latent_mask_dimensions_approx": [55, 45],
                    "latent_mask_pixels_approx": 2475,
                    "latent_capacity_gain_factor": 17.3,
                    "mapping_pipeline": "Crop 256x256 from canvas -> Upscale crop and local mask to 1024x1024 via Lanczos -> SDXL inpainting at 1024x1024 -> Downscale generated crop to 256x256 via Lanczos -> Remap crop to canvas [256, 162, 512, 418] -> Composite into 512x512 canvas strictly inside [345, 245, 455, 335]. Outside pixels remain exactly 0.000000."
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
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COMMONS_SD2_002_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "upscale_factor": 1.0,
                    "local_mask_xyxy": [0, 340, 225, 512],
                    "latent_mask_dimensions_approx": [28, 21],
                    "latent_mask_pixels_approx": 588,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [0, 340, 225, 512]."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COMMONS_SD2_002_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [0, 192, 320, 512],
                    "crop_dimensions": [320, 320],
                    "padding_margins": {"left": 0, "right": 95, "top": 148, "bottom": 0},
                    "local_mask_xyxy": [0, 148, 225, 320],
                    "inference_resolution": [512, 512],
                    "upscale_factor": 1.6,
                    "scaled_mask_in_inference": [0, 237, 360, 512],
                    "latent_mask_dimensions_approx": [45, 34],
                    "latent_mask_pixels_approx": 1530,
                    "latent_capacity_gain_factor": 2.6,
                    "mapping_pipeline": "Crop 320x320 from canvas -> Upscale crop and local mask to 512x512 via Lanczos -> SD2 inpainting at 512x512 -> Downscale generated crop to 320x320 via Lanczos -> Remap crop to canvas [0, 192, 320, 512] -> Composite into 512x512 canvas strictly inside [0, 340, 225, 512]. Outside pixels remain exactly 0.000000."
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
            "arms": {
                "arm_a_full_canvas": {
                    "arm_id": "IND_COMMONS_SDXL_001_ARM_A",
                    "method": "full_canvas_inference",
                    "crop_bbox_xyxy": [0, 0, 512, 512],
                    "padding_margins": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                    "inference_resolution": [512, 512],
                    "upscale_factor": 1.0,
                    "local_mask_xyxy": [350, 45, 500, 160],
                    "latent_mask_dimensions_approx": [18, 14],
                    "latent_mask_pixels_approx": 252,
                    "mapping_pipeline": "Direct inference at 512x512; composite output with authentic image strictly within mask [350, 45, 500, 160]."
                },
                "arm_b_local_crop": {
                    "arm_id": "IND_COMMONS_SDXL_001_ARM_B",
                    "method": "local_crop_padded_inference",
                    "crop_bbox_xyxy": [256, 0, 512, 256],
                    "crop_dimensions": [256, 256],
                    "padding_margins": {"left": 94, "right": 12, "top": 45, "bottom": 96},
                    "local_mask_xyxy": [94, 45, 244, 160],
                    "inference_resolution": [1024, 1024],
                    "upscale_factor": 4.0,
                    "scaled_mask_in_inference": [376, 180, 976, 640],
                    "latent_mask_dimensions_approx": [75, 57],
                    "latent_mask_pixels_approx": 4275,
                    "latent_capacity_gain_factor": 17.0,
                    "mapping_pipeline": "Crop 256x256 from canvas -> Upscale crop and local mask to 1024x1024 via Lanczos -> SDXL inpainting at 1024x1024 -> Downscale generated crop to 256x256 via Lanczos -> Remap crop to canvas [256, 0, 512, 256] -> Composite into 512x512 canvas strictly inside [350, 45, 500, 160]. Outside pixels remain exactly 0.000000."
                }
            }
        }
    ]
}


def draw_overlay(auth_img: Image.Image, target_box: list[int], mask_box: list[int], crop_box: list[int]) -> Image.Image:
    """Create coordinate overlay showing Target (blue), Mask (red), and Local Crop (green)."""
    vis = auth_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Local Crop (Green translucent fill + solid outline)
    cx1, cy1, cx2, cy2 = crop_box
    draw.rectangle([cx1, cy1, cx2, cy2], fill=(34, 197, 94, 45), outline=(34, 197, 94, 255), width=3)

    # Mask Box (Red translucent fill + dashed outline)
    mx1, my1, mx2, my2 = mask_box
    draw.rectangle([mx1, my1, mx2, my2], fill=(239, 68, 68, 60), outline=(239, 68, 68, 255), width=2)

    # Target Box (Cyan outline)
    tx1, ty1, tx2, ty2 = target_box
    draw.rectangle([tx1, ty1, tx2, ty2], outline=(6, 182, 212, 255), width=2)

    # Text annotations
    draw.text((cx1 + 6, cy1 + 6), f"Local Crop: {crop_box}", fill=(34, 197, 94, 255))
    draw.text((mx1 + 4, my1 + 4), f"Mask: {mask_box}", fill=(239, 68, 68, 255))
    draw.text((tx1 + 4, ty1 + 4), f"Target: {target_box}", fill=(6, 182, 212, 255))

    combined = Image.alpha_composite(vis, overlay)
    return combined.convert("RGB")


def main():
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON plan
    with open(PLAN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(DIAGNOSTIC_PLAN, f, indent=2, ensure_ascii=False)
    print(f"Diagnostic plan saved at: {PLAN_OUTPUT_PATH}")

    # Generate overlay images
    for item in DIAGNOSTIC_PLAN["diagnostic_candidates"]:
        cid = item["candidate_id"]
        auth_path = IMAGES_DIR / f"{cid}_auth.png"
        auth_img = Image.open(auth_path).convert("RGB")

        target_box = item["target_bbox_xyxy"]
        mask_box = item["mask_bbox_xyxy"]
        crop_box = item["arms"]["arm_b_local_crop"]["crop_bbox_xyxy"]

        overlay_img = draw_overlay(auth_img, target_box, mask_box, crop_box)
        out_overlay_path = OVERLAYS_DIR / f"{cid}_crop_overlay.png"
        overlay_img.save(out_overlay_path, format="PNG")
        print(f"Saved overlay for {cid} -> {out_overlay_path.name}")


if __name__ == "__main__":
    main()
