#!/usr/bin/env python3
"""Build deterministic pre-registered generation plan for 4 CORD-v2 document feasibility attempts.

Locks source IDs, bounding boxes, original texts, target texts, prompts, seeds,
and generator configuration prior to execution on Google Colab GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PLAN_PATH = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"

PLAN_METADATA: Dict[str, Any] = {
    "study_name": "Controlled AI-Document Generation Feasibility Study",
    "protocol_version": "1.1.0",
    "official_project_title": "Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.",
    "execution_target": "GOOGLE_COLAB_GPU_EPHEMERAL",
    "local_role": "PLAN_PREPARATION_AND_INDEPENDENT_VERIFICATION",
    "dataset": {
        "repository": "naver-clova-ix/cord-v2",
        "split": "validation",
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98",
        "license": "CC-BY 4.0",
        "parquet_file": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    },
    "model": {
        "repository": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "revision": "115134f363124c53c7d878647567d04daf26e41e",
        "license": "OpenRAIL++",
        "pipeline_class": "StableDiffusionXLInpaintPipeline",
        "torch_dtype": "float16",
        "variant": "fp16",
        "canvas_size": 1024,
        "scheduler": "EulerDiscreteScheduler",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "strength": 0.99,
    },
    "quarantine_policy": "DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION",
    "budget_limits": {
        "checkpoint_colab_max_bytes": 12884901888,  # 12 GiB
        "original_data_colab_max_bytes": 209715200,  # 200 MB
        "artifacts_transfer_max_bytes": 104857600,   # 100 MB
        "max_gpu_minutes": 60,
    },
}

ATTEMPTS: List[Dict[str, Any]] = [
    {
        "attempt_index": 0,
        "attempt_id": "attempt_0_replacement_price",
        "source_row_index": 22,
        "source_image_id": 22,
        "source_doc_identifier": "cord_v2_valid_row22_img22",
        "native_size": {"width": 576, "height": 864},
        "operation": "replacement",
        "field_category": "menu.price",
        "original_text": "280,000",
        "target_text": "350,000",
        "target_bbox": [442, 406, 468, 488],  # [ymin, xmin, ymax, xmax]
        "context_margin_ratio": 0.5,
        "prompt": "a photo of a retail receipt, crisp clear printed text displaying '350,000', matching authentic thermal receipt paper texture, sharp dark printed typography, clean alignment",
        "negative_prompt": "blurry, low quality, distorted text, illegible fonts, messy background, watermarks, artifacts, colored stains, handwritten",
        "seed": 20261012,
        "rationale": "Evaluate numerical glyph replacement on thermal paper texture without disturbing neighboring line items.",
    },
    {
        "attempt_index": 1,
        "attempt_id": "attempt_1_replacement_item_name",
        "source_row_index": 61,
        "source_image_id": 61,
        "source_doc_identifier": "cord_v2_valid_row61_img61",
        "native_size": {"width": 1356, "height": 2315},
        "operation": "replacement",
        "field_category": "menu.nm",
        "original_text": "Populaire Chocolate",
        "target_text": "Magnum Gold Almond",
        "target_bbox": [1085, 128, 1156, 749],  # [ymin, xmin, ymax, xmax]
        "context_margin_ratio": 0.4,
        "prompt": "a high resolution photo of a store receipt, sharp printed text displaying 'Magnum Gold Almond', matching receipt typography and texture, uniform lighting, crisp black font",
        "negative_prompt": "blurry, low quality, distorted text, illegible fonts, crooked baseline, stains, artifacts",
        "seed": 20261013,
        "rationale": "Evaluate long alphanumeric product name substitution on high-resolution receipt.",
    },
    {
        "attempt_index": 2,
        "attempt_id": "attempt_2_removal_discount_line",
        "source_row_index": 7,
        "source_image_id": 7,
        "source_doc_identifier": "cord_v2_valid_row7_img7",
        "native_size": {"width": 576, "height": 864},
        "operation": "removal",
        "field_category": "sub_total.discount_price",
        "original_text": "Discount: -9,545",
        "target_text": "",
        "target_bbox": [524, 208, 556, 492],  # [ymin, xmin, ymax, xmax]
        "context_margin_ratio": 0.5,
        "prompt": "a photo of a retail receipt, clean blank receipt paper texture, subtle paper grain, uniform off-white surface, no text, completely erased",
        "negative_prompt": "text, numbers, letters, blurry stains, dark spots, ghosting, visible outlines, artifacts",
        "seed": 20261014,
        "rationale": "Evaluate complete semantic erasure of discount subtotal line with seamless paper background restoration.",
    },
    {
        "attempt_index": 3,
        "attempt_id": "attempt_3_removal_tax_line",
        "source_row_index": 93,
        "source_image_id": 93,
        "source_doc_identifier": "cord_v2_valid_row93_img93",
        "native_size": {"width": 864, "height": 1296},
        "operation": "removal",
        "field_category": "sub_total.tax_price",
        "original_text": "TAX 15,636",
        "target_text": "",
        "target_bbox": [939, 383, 975, 605],  # [ymin, xmin, ymax, xmax]
        "context_margin_ratio": 0.5,
        "prompt": "a photo of a retail receipt, clean blank receipt paper texture, subtle vertical paper fiber, uniform flat lighting, seamless empty paper, no text",
        "negative_prompt": "text, numbers, letters, ghost glyphs, smudges, stains, artifacts, border lines",
        "seed": 20261015,
        "rationale": "Evaluate erasure of tax line between subtotal and grand total while preserving surrounding table layout.",
    },
]


def build_plan() -> Dict[str, Any]:
    """Construct full plan dictionary and validate constraints."""
    # Integrity assertions
    assert len(ATTEMPTS) == 4, "Must have exactly 4 attempts."
    source_ids = [a["source_image_id"] for a in ATTEMPTS]
    assert len(set(source_ids)) == 4, "All 4 source documents must be distinct."

    ops = [a["operation"] for a in ATTEMPTS]
    assert ops.count("replacement") == 2, "Must have exactly 2 replacements."
    assert ops.count("removal") == 2, "Must have exactly 2 removals."

    for a in ATTEMPTS:
        ymin, xmin, ymax, xmax = a["target_bbox"]
        assert 0 <= ymin < ymax <= a["native_size"]["height"]
        assert 0 <= xmin < xmax <= a["native_size"]["width"]
        assert a["prompt"] and a["negative_prompt"]
        assert a["seed"] > 0

    plan_doc = {
        "metadata": PLAN_METADATA,
        "attempts": ATTEMPTS,
    }
    return plan_doc


def main() -> None:
    plan = build_plan()
    OUTPUT_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated pre-registered plan: {OUTPUT_PLAN_PATH}")
    print(f"Attempts: {len(plan['attempts'])} ({[a['attempt_id'] for a in plan['attempts']]})")


if __name__ == "__main__":
    main()
