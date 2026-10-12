"""Document feasibility geometry transformations and composite contract.

Implements aspect-ratio preserving context cropping, canvas letterboxing,
binary mask transforms, inverse mapping, and strict bitwise outside-mask
equality verification for AI document manipulation experiments.
"""

from __future__ import annotations

import json
from typing import Any, Tuple, Dict, List
import numpy as np
from PIL import Image


def extract_roi_and_build_canvas(
    image: Image.Image,
    target_bbox: List[int] | Tuple[int, int, int, int],
    canvas_size: int = 1024,
    context_margin_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Extract context crop around target bounding box and letterbox onto square canvas.

    Args:
        image: PIL Image in RGB format (native dimensions).
        target_bbox: [ymin, xmin, ymax, xmax] in original pixel coordinates.
        canvas_size: Square canvas resolution for diffusion model (e.g. 1024).
        context_margin_ratio: Margin to expand around target bbox (e.g. 0.5 = 50%).

    Returns:
        Dict containing:
            canvas_image: PIL Image (RGB, canvas_size x canvas_size)
            canvas_mask: PIL Image (L, canvas_size x canvas_size, strictly {0, 255})
            context_crop: PIL Image (RGB, crop_w x crop_h)
            geometry_receipt: Dict detailing all transformations for inverse mapping.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    ymin, xmin, ymax, xmax = [int(v) for v in target_bbox]

    # Validate bounding box strictly: silent clamping is forbidden
    if ymin < 0 or xmin < 0 or ymax > orig_h or xmax > orig_w:
        raise ValueError(
            f"Bounding box {target_bbox} is out of bounds for image dimensions ({orig_w}, {orig_h}). "
            f"Silent clamping is forbidden by protocol."
        )
    if ymin >= ymax or xmin >= xmax:
        raise ValueError(
            f"Degenerate bounding box {target_bbox}: ymin ({ymin}) >= ymax ({ymax}) "
            f"or xmin ({xmin}) >= xmax ({xmax})."
        )

    bbox_w = xmax - xmin
    bbox_h = ymax - ymin

    # Expand margin
    margin_x = int(round(bbox_w * context_margin_ratio))
    margin_y = int(round(bbox_h * context_margin_ratio))

    crop_ymin = max(0, ymin - margin_y)
    crop_ymax = min(orig_h, ymax + margin_y)
    crop_xmin = max(0, xmin - margin_x)
    crop_xmax = min(orig_w, xmax + margin_x)

    crop_w = crop_xmax - crop_xmin
    crop_h = crop_ymax - crop_ymin

    # Crop image
    context_crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    # Build mask in crop coordinate space
    # Target bbox relative to crop origin
    target_rel_ymin = ymin - crop_ymin
    target_rel_ymax = ymax - crop_ymin
    target_rel_xmin = xmin - crop_xmin
    target_rel_xmax = xmax - crop_xmin

    mask_crop_arr = np.zeros((crop_h, crop_w), dtype=np.uint8)
    mask_crop_arr[target_rel_ymin:target_rel_ymax, target_rel_xmin:target_rel_xmax] = 255
    mask_crop = Image.fromarray(mask_crop_arr, mode="L")

    # Aspect-ratio preserving scaling
    scale = min(canvas_size / crop_w, canvas_size / crop_h)
    scaled_w = max(1, min(canvas_size, int(round(crop_w * scale))))
    scaled_h = max(1, min(canvas_size, int(round(crop_h * scale))))

    scaled_image = context_crop.resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
    # Mask resize MUST use NEAREST neighbor to avoid intermediate gray values
    scaled_mask = mask_crop.resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST)

    # Re-binarize to ensure exact {0, 255}
    scaled_mask_arr = (np.array(scaled_mask) > 127).astype(np.uint8) * 255
    scaled_mask = Image.fromarray(scaled_mask_arr, mode="L")

    # Letterbox onto square canvas
    pad_x = (canvas_size - scaled_w) // 2
    pad_y = (canvas_size - scaled_h) // 2

    canvas_image = Image.new("RGB", (canvas_size, canvas_size), color=(128, 128, 128))
    canvas_image.paste(scaled_image, (pad_x, pad_y))

    canvas_mask = Image.new("L", (canvas_size, canvas_size), color=0)
    canvas_mask.paste(scaled_mask, (pad_x, pad_y))

    geometry_receipt = {
        "original_size": [orig_w, orig_h],
        "target_bbox": [ymin, xmin, ymax, xmax],
        "crop_bbox": [crop_ymin, crop_xmin, crop_ymax, crop_xmax],
        "crop_size": [crop_w, crop_h],
        "scale": float(scale),
        "scaled_size": [scaled_w, scaled_h],
        "padding": [pad_y, pad_x],
        "canvas_size": canvas_size,
        "context_margin_ratio": float(context_margin_ratio),
    }

    return {
        "canvas_image": canvas_image,
        "canvas_mask": canvas_mask,
        "context_crop": context_crop,
        "geometry_receipt": geometry_receipt,
    }


def composite_inpainted_patch(
    original_image: Image.Image,
    model_output_canvas: Image.Image,
    geometry_receipt: Dict[str, Any],
) -> Dict[str, Any]:
    """Inverse map model canvas output back to original coordinates and composite strictly inside mask.

    Guarantees that pixels outside the intervention mask remain 100% bitwise identical
    to the authentic source image.

    Args:
        original_image: Authentic PIL Image (native dimensions).
        model_output_canvas: Raw output from inpainting model (canvas_size x canvas_size).
        geometry_receipt: Transformation parameters from extract_roi_and_build_canvas.

    Returns:
        Dict containing:
            final_edited_image: PIL Image (RGB, native dimensions)
            final_intervention_mask: PIL Image (L, native dimensions, strictly {0, 255})
            composite_receipt: Summary metadata
    """
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")
    if model_output_canvas.mode != "RGB":
        model_output_canvas = model_output_canvas.convert("RGB")

    orig_w, orig_h = geometry_receipt["original_size"]
    crop_ymin, crop_xmin, crop_ymax, crop_xmax = geometry_receipt["crop_bbox"]
    crop_w, crop_h = geometry_receipt["crop_size"]
    scaled_w, scaled_h = geometry_receipt["scaled_size"]
    pad_y, pad_x = geometry_receipt["padding"]
    target_ymin, target_xmin, target_ymax, target_xmax = geometry_receipt["target_bbox"]

    # 1. Unpad canvas
    unpadded_patch = model_output_canvas.crop((pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h))

    # 2. Resize unpadded patch back to crop size
    restored_patch = unpadded_patch.resize((crop_w, crop_h), resample=Image.Resampling.BICUBIC)
    restored_patch_arr = np.array(restored_patch)

    # 3. Create full-frame candidate edited image
    orig_arr = np.array(original_image)
    candidate_full_arr = orig_arr.copy()
    candidate_full_arr[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = restored_patch_arr

    # 4. Create final intervention mask in original coordinates
    final_mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
    final_mask_arr[target_ymin:target_ymax, target_xmin:target_xmax] = 255

    # 5. Composite STRICTLY inside intervention mask:
    # pixels where mask == 255 take candidate_full_arr, everywhere else takes orig_arr bit-for-bit
    final_edited_arr = np.where(final_mask_arr[:, :, None] == 255, candidate_full_arr, orig_arr)

    final_edited_image = Image.fromarray(final_edited_arr, mode="RGB")
    final_intervention_mask = Image.fromarray(final_mask_arr, mode="L")

    composite_receipt = {
        "status": "COMPOSITED",
        "original_size": [orig_w, orig_h],
        "crop_bbox": [crop_ymin, crop_xmin, crop_ymax, crop_xmax],
        "target_bbox": [target_ymin, target_xmin, target_ymax, target_xmax],
        "unpadded_size": [scaled_w, scaled_h],
        "restored_crop_size": [crop_w, crop_h],
    }

    return {
        "final_edited_image": final_edited_image,
        "final_intervention_mask": final_intervention_mask,
        "composite_receipt": composite_receipt,
    }


def verify_outside_equality(
    authentic_image: Image.Image | np.ndarray,
    edited_image: Image.Image | np.ndarray,
    intervention_mask: Image.Image | np.ndarray,
) -> Tuple[bool, int]:
    """Verify that all pixels outside the intervention mask are strictly bitwise identical.

    Returns:
        (is_equal, max_absolute_diff)
    """
    auth_arr = np.array(authentic_image, dtype=np.uint8)
    edit_arr = np.array(edited_image, dtype=np.uint8)
    mask_arr = np.array(intervention_mask, dtype=np.uint8)

    if auth_arr.shape[:2] != mask_arr.shape[:2] or edit_arr.shape[:2] != mask_arr.shape[:2]:
        raise ValueError("Image dimensions and mask dimensions do not match.")

    outside_indices = mask_arr == 0
    if not np.any(outside_indices):
        # Entire image is masked (should not happen in document pilot)
        return True, 0

    auth_outside = auth_arr[outside_indices]
    edit_outside = edit_arr[outside_indices]

    is_equal = bool(np.array_equal(auth_outside, edit_outside))
    if is_equal:
        return True, 0

    diff = np.abs(auth_outside.astype(np.int32) - edit_outside.astype(np.int32))
    max_diff = int(np.max(diff))
    return False, max_diff


def compute_pixel_change_statistics(
    authentic_image: Image.Image | np.ndarray,
    edited_image: Image.Image | np.ndarray,
    intervention_mask: Image.Image | np.ndarray,
) -> Dict[str, Any]:
    """Compute detailed pixel difference statistics inside and outside the intervention mask."""
    auth_arr = np.array(authentic_image, dtype=np.int32)
    edit_arr = np.array(edited_image, dtype=np.int32)
    mask_arr = np.array(intervention_mask, dtype=np.uint8)

    inside_mask = mask_arr == 255
    outside_mask = mask_arr == 0

    inside_pixels_count = int(np.sum(inside_mask))
    outside_pixels_count = int(np.sum(outside_mask))

    # Inside differences
    if inside_pixels_count > 0:
        inside_diff = np.abs(auth_arr[inside_mask] - edit_arr[inside_mask])
        inside_changed = np.any(inside_diff > 0, axis=-1)
        changed_pixels_count = int(np.sum(inside_changed))
        changed_pixel_ratio = float(changed_pixels_count / inside_pixels_count)
        inside_mean_l1 = float(np.mean(inside_diff))
        inside_max_l1 = int(np.max(inside_diff))
    else:
        changed_pixels_count = 0
        changed_pixel_ratio = 0.0
        inside_mean_l1 = 0.0
        inside_max_l1 = 0

    # Outside differences
    if outside_pixels_count > 0:
        outside_diff = np.abs(auth_arr[outside_mask] - edit_arr[outside_mask])
        outside_changed = np.any(outside_diff > 0, axis=-1)
        outside_changed_count = int(np.sum(outside_changed))
        outside_mean_l1 = float(np.mean(outside_diff))
        outside_max_l1 = int(np.max(outside_diff))
    else:
        outside_changed_count = 0
        outside_mean_l1 = 0.0
        outside_max_l1 = 0

    return {
        "inside_mask_pixel_count": inside_pixels_count,
        "changed_pixels_count": changed_pixels_count,
        "changed_pixel_ratio": changed_pixel_ratio,
        "inside_mean_l1": inside_mean_l1,
        "inside_max_l1": inside_max_l1,
        "outside_mask_pixel_count": outside_pixels_count,
        "outside_changed_count": outside_changed_count,
        "outside_mean_l1": outside_mean_l1,
        "outside_max_l1": outside_max_l1,
        "is_outside_bit_identical": (outside_changed_count == 0),
    }
