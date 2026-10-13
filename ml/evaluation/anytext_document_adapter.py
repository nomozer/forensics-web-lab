"""AnyText Document Adapter and Geometry Contract.

Provides preprocessing, position mask conditioning, inverse mapping,
and strict outside-mask compositing for AnyText text-editing on document images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

from ml.evaluation.document_feasibility_geometry import (
    extract_roi_and_build_canvas,
    composite_inpainted_patch,
    verify_outside_equality,
    compute_pixel_change_statistics,
)


def prepare_anytext_editing_inputs(
    image: Image.Image,
    target_bbox: List[int] | Tuple[int, int, int, int],
    canvas_size: int = 512,
    context_margin_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Prepares AnyText ori_image and draw_pos inputs from native document and target bbox.

    Args:
        image: Original PIL Image in RGB format (native dimensions).
        target_bbox: [ymin, xmin, ymax, xmax] in native coordinates.
        canvas_size: Square canvas resolution for AnyText (512x512).
        context_margin_ratio: Margin expansion around target bbox (default: 0.5).

    Returns:
        Dict with:
            ori_image_canvas: PIL Image (RGB, 512x512) for AnyText ori_image.
            draw_pos_canvas: PIL Image (L, 512x512, {0, 255}) for AnyText draw_pos.
            geometry_receipt: Dict tracking transformations for inverse mapping.
            native_mask: PIL Image (L, orig_w x orig_h, {0, 255}).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    ymin, xmin, ymax, xmax = [int(v) for v in target_bbox]
    if ymin < 0 or xmin < 0 or ymax > orig_h or xmax > orig_w or ymin >= ymax or xmin >= xmax:
        raise ValueError(f"Invalid bounding box {target_bbox} for image size ({orig_w}, {orig_h})")

    # Use verified geometry builder with 512x512 canvas size
    roi_data = extract_roi_and_build_canvas(
        image=image,
        target_bbox=target_bbox,
        canvas_size=canvas_size,
        context_margin_ratio=context_margin_ratio,
    )

    orig_w, orig_h = image.size
    ymin, xmin, ymax, xmax = [int(v) for v in target_bbox]

    # Native intervention mask
    native_mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
    native_mask_arr[ymin:ymax, xmin:xmax] = 255
    native_mask = Image.fromarray(native_mask_arr, mode="L")

    return {
        "ori_image_canvas": roi_data["canvas_image"],
        "draw_pos_canvas": roi_data["canvas_mask"],
        "context_crop": roi_data["context_crop"],
        "geometry_receipt": roi_data["geometry_receipt"],
        "native_mask": native_mask,
    }


def composite_anytext_editing_output(
    original_image: Image.Image,
    raw_anytext_output: Image.Image | np.ndarray,
    geometry_receipt: Dict[str, Any],
    native_mask: Image.Image,
) -> Dict[str, Any]:
    """Composites raw AnyText canvas back into native document strictly inside mask.

    Args:
        original_image: Original PIL Image in RGB format.
        raw_anytext_output: 512x512 RGB raw output from AnyText (PIL or numpy array).
        geometry_receipt: Receipt from prepare_anytext_editing_inputs.
        native_mask: Native binary mask (L mode, {0, 255}).

    Returns:
        Dict with:
            final_edited_document: PIL Image in RGB format (native dimensions).
            is_outside_equal: bool (True if 100% bitwise identical outside mask).
            max_outside_diff: int (must be 0).
            raw_model_canvas: PIL Image (512x512).
    """
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    if isinstance(raw_anytext_output, np.ndarray):
        raw_canvas = Image.fromarray(raw_anytext_output.astype(np.uint8), mode="RGB")
    else:
        raw_canvas = raw_anytext_output.convert("RGB")

    # Composite using verified contract
    comp_data = composite_inpainted_patch(
        original_image=original_image,
        model_output_canvas=raw_canvas,
        geometry_receipt=geometry_receipt,
    )
    final_img = comp_data["final_edited_image"]
    final_mask = comp_data["final_intervention_mask"]

    # Strict outside-mask equality check
    is_outside_equal, max_outside_diff = verify_outside_equality(
        authentic_image=original_image,
        edited_image=final_img,
        intervention_mask=final_mask,
    )

    stats = compute_pixel_change_statistics(
        authentic_image=original_image,
        edited_image=final_img,
        intervention_mask=final_mask,
    )

    return {
        "final_edited_document": final_img,
        "final_composited_document": final_img,
        "final_intervention_mask": final_mask,
        "is_outside_equal": is_outside_equal,
        "max_outside_diff": max_outside_diff,
        "raw_model_canvas": raw_canvas,
        "composite_receipt": comp_data["composite_receipt"],
        "statistics": stats,
        "verification": {
            "outside_equal": is_outside_equal,
            "max_outside_diff": max_outside_diff,
            "modified_pixel_count": stats["changed_pixels_count"],
        },
    }
