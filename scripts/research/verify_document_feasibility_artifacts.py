#!/usr/bin/env python3
"""Local Artifact Verification Harness & Contact Sheet Generator for Document Feasibility Study.

Performs fail-closed technical QC on downloaded Colab artifacts:
- Bitwise array equality on outside-mask pixels (np.array_equal).
- Geometry contract compliance and native dimension symmetry.
- Inside-mask pixel change statistics and seam boundary diagnostics.
- Generation of visual 5-column multi-row contact sheet.
- Preservation of PENDING_HUMAN_REVIEW status in human_review.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_attempt_artifacts(attempt_dir: Path) -> Dict[str, Any]:
    """Verify technical QC invariants on a single attempt folder."""
    required_files = [
        "authentic_control.png",
        "generation_context.png",
        "raw_model_output.png",
        "final_edited_document.png",
        "final_intervention_mask.png",
        "geometry_receipt.json",
        "attempt_receipt.json",
    ]
    missing = [f for f in required_files if not (attempt_dir / f).exists()]
    if missing:
        return {
            "attempt_id": attempt_dir.name,
            "status": "FAIL_MISSING_FILES",
            "missing_files": missing,
        }

    # Load receipts
    with open(attempt_dir / "geometry_receipt.json", "r", encoding="utf-8") as f:
        geo_receipt = json.load(f)
    with open(attempt_dir / "attempt_receipt.json", "r", encoding="utf-8") as f:
        att_receipt = json.load(f)

    # Load images
    auth_img = Image.open(attempt_dir / "authentic_control.png").convert("RGB")
    edit_img = Image.open(attempt_dir / "final_edited_document.png").convert("RGB")
    mask_img = Image.open(attempt_dir / "final_intervention_mask.png").convert("L")

    # 1. Geometry & dimensions symmetry
    orig_w, orig_h = geo_receipt["original_size"]
    if auth_img.size != (orig_w, orig_h) or edit_img.size != (orig_w, orig_h) or mask_img.size != (orig_w, orig_h):
        return {
            "attempt_id": attempt_dir.name,
            "status": "FAIL_DIMENSION_MISMATCH",
            "expected_size": [orig_w, orig_h],
            "auth_size": list(auth_img.size),
            "edit_size": list(edit_img.size),
            "mask_size": list(mask_img.size),
        }

    # 2. Binary mask validity
    mask_arr = np.array(mask_img, dtype=np.uint8)
    unique_vals = np.unique(mask_arr)
    if not set(unique_vals).issubset({0, 255}):
        return {
            "attempt_id": attempt_dir.name,
            "status": "FAIL_NON_BINARY_MASK",
            "unique_values": [int(v) for v in unique_vals],
        }
    if 255 not in unique_vals:
        return {
            "attempt_id": attempt_dir.name,
            "status": "FAIL_EMPTY_MASK",
        }

    # 3. Outside-mask bitwise equality check
    auth_arr = np.array(auth_img, dtype=np.uint8)
    edit_arr = np.array(edit_img, dtype=np.uint8)
    outside = mask_arr == 0
    auth_outside = auth_arr[outside]
    edit_outside = edit_arr[outside]

    is_outside_equal = bool(np.array_equal(auth_outside, edit_outside))
    diff_outside = np.abs(auth_outside.astype(np.int32) - edit_outside.astype(np.int32))
    max_outside_diff = int(np.max(diff_outside)) if np.any(outside) else 0

    if not is_outside_equal:
        return {
            "attempt_id": attempt_dir.name,
            "status": "FAIL_OUTSIDE_MASK_LEAKAGE",
            "is_outside_equal": False,
            "max_outside_diff": max_outside_diff,
        }

    # 4. Inside-mask statistics
    inside = mask_arr == 255
    auth_inside = auth_arr[inside].astype(np.int32)
    edit_inside = edit_arr[inside].astype(np.int32)
    diff_inside = np.abs(auth_inside - edit_inside)

    in_mean_l1 = float(np.mean(diff_inside))
    in_max_l1 = int(np.max(diff_inside))
    changed_pixels = int(np.sum(np.any(diff_inside > 0, axis=-1)))
    total_inside_pixels = int(np.sum(inside))
    changed_ratio = float(changed_pixels / total_inside_pixels) if total_inside_pixels > 0 else 0.0

    return {
        "attempt_id": attempt_dir.name,
        "status": "TECHNICAL_QC_PASS",
        "operation": att_receipt.get("operation", "unknown"),
        "original_text": att_receipt.get("original_text", ""),
        "target_text": att_receipt.get("target_text", ""),
        "dimensions": [orig_w, orig_h],
        "is_outside_bitwise_equal": True,
        "max_outside_diff": 0,
        "inside_mask_pixels": total_inside_pixels,
        "inside_mean_l1": in_mean_l1,
        "inside_max_l1": in_max_l1,
        "changed_pixels": changed_pixels,
        "changed_pixel_ratio": changed_ratio,
        "preliminary_assessment": att_receipt.get("preliminary_assessment", "UNKNOWN"),
    }


def generate_contact_sheet(
    attempts_dirs: List[Path],
    output_path: Path,
) -> None:
    """Render a comprehensive 5-column contact sheet comparison across all attempts."""
    panel_w = 280
    panel_h = 280
    num_cols = 5
    num_rows = len(attempts_dirs)
    header_h = 50
    row_header_w = 0
    margin = 15

    total_w = margin + num_cols * (panel_w + margin)
    total_h = header_h + num_rows * (panel_h + margin + 30) + margin

    sheet = Image.new("RGB", (total_w, total_h), color=(245, 245, 248))
    draw = ImageDraw.Draw(sheet)

    col_titles = [
        "1. Authentic Document",
        "2. ROI Before Edit",
        "3. Intervention Mask",
        "4. ROI After Edit",
        "5. Pixel Diff (5x)",
    ]

    for col_idx, title in enumerate(col_titles):
        x = margin + col_idx * (panel_w + margin)
        draw.text((x + 10, 15), title, fill=(30, 30, 30))

    for row_idx, att_dir in enumerate(attempts_dirs):
        y_top = header_h + row_idx * (panel_h + margin + 30)

        # Load images
        auth_p = att_dir / "authentic_control.png"
        edit_p = att_dir / "final_edited_document.png"
        mask_p = att_dir / "final_intervention_mask.png"
        geo_p = att_dir / "geometry_receipt.json"

        if not (auth_p.exists() and edit_p.exists() and mask_p.exists() and geo_p.exists()):
            continue

        auth_img = Image.open(auth_p).convert("RGB")
        edit_img = Image.open(edit_p).convert("RGB")
        mask_img = Image.open(mask_p).convert("L")
        with open(geo_p, "r", encoding="utf-8") as f:
            geo = json.load(f)

        crop_ymin, crop_xmin, crop_ymax, crop_xmax = geo["crop_bbox"]
        target_ymin, target_xmin, target_ymax, target_xmax = geo["target_bbox"]

        # Text banner
        banner_text = f"Attempt {row_idx}: {att_dir.name}"
        draw.text((margin + 10, y_top), banner_text, fill=(70, 70, 80))
        img_y = y_top + 25

        # Col 1: Authentic document overview thumbnail with bounding box
        auth_thumb = auth_img.copy()
        draw_thumb = ImageDraw.Draw(auth_thumb)
        draw_thumb.rectangle([crop_xmin, crop_ymin, crop_xmax, crop_ymax], outline=(255, 0, 0), width=4)
        auth_thumb_fitted = auth_thumb.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
        sheet.paste(auth_thumb_fitted, (margin, img_y))

        # Col 2: ROI before edit
        roi_before = auth_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        roi_before_fitted = roi_before.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
        sheet.paste(roi_before_fitted, (margin + (panel_w + margin), img_y))

        # Col 3: Intervention Mask
        mask_roi = mask_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        mask_roi_fitted = mask_roi.resize((panel_w, panel_h), Image.Resampling.NEAREST)
        sheet.paste(mask_roi_fitted.convert("RGB"), (margin + 2 * (panel_w + margin), img_y))

        # Col 4: ROI after edit
        roi_after = edit_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        roi_after_fitted = roi_after.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
        sheet.paste(roi_after_fitted, (margin + 3 * (panel_w + margin), img_y))

        # Col 5: Pixel difference heatmap
        before_arr = np.array(roi_before, dtype=np.int32)
        after_arr = np.array(roi_after, dtype=np.int32)
        diff_arr = np.clip(np.abs(before_arr - after_arr) * 5, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_arr, mode="RGB")
        diff_fitted = diff_img.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
        sheet.paste(diff_fitted, (margin + 4 * (panel_w + margin), img_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    print(f"Contact sheet saved: {output_path}")


def verify_feasibility_results(
    input_path: Path,
    output_receipt_path: Path,
) -> Dict[str, Any]:
    """Execute full verification pipeline on folder or zip archive."""
    temp_dir: Optional[Path] = None
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_dir = Path(tempfile.mkdtemp(prefix="feasibility_verify_"))
        print(f"Extracting zip archive: {input_path} -> {temp_dir}")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_dir)
        work_dir = temp_dir
    else:
        work_dir = input_path

    # Discover attempt folders
    attempt_dirs = sorted([d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_")])
    if not attempt_dirs:
        raise ValueError(f"No attempt directories found in {work_dir}")

    results = []
    all_passed = True
    for att_dir in attempt_dirs:
        res = verify_attempt_artifacts(att_dir)
        results.append(res)
        if res.get("status") != "TECHNICAL_QC_PASS":
            all_passed = False
        print(f"[{res.get('status')}] {att_dir.name}: bitwise_equal={res.get('is_outside_bitwise_equal')}, in_mean_l1={res.get('inside_mean_l1')}")

    # Generate contact sheet
    contact_sheet_path = output_receipt_path.parent / "feasibility_contact_sheet.png"
    generate_contact_sheet(attempt_dirs, contact_sheet_path)

    # Check human review CSV
    csv_path = work_dir / "human_review.csv"
    human_qc_status = "PENDING_HUMAN_REVIEW"
    if csv_path.exists():
        csv_content = csv_path.read_text(encoding="utf-8")
        if "HUMAN_PASS" in csv_content:
            human_qc_status = "EXECUTED_HUMAN_QC_COMPLETED"

    summary_receipt = {
        "status": "TECHNICAL_QC_PASS" if all_passed else "TECHNICAL_QC_FAIL",
        "official_project_title": "Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.",
        "protocol_version": "1.1.0",
        "verified_attempts_count": len(results),
        "all_outside_bitwise_equal": all_passed,
        "human_qc_status": human_qc_status,
        "contact_sheet_path": str(contact_sheet_path),
        "attempts": results,
    }

    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_receipt_path, "w", encoding="utf-8") as f:
        json.dump(summary_receipt, f, indent=2)
    print(f"Verification receipt saved: {output_receipt_path}")

    # Cleanup temp extraction directory if created
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return summary_receipt


def main():
    parser = argparse.ArgumentParser(description="Verify Document Feasibility Artifacts.")
    parser.add_argument("--input-dir", type=Path, help="Path to unzipped feasibility results folder.")
    parser.add_argument("--zip", type=Path, help="Path to feasibility results zip archive.")
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_verification_receipt.json",
    )
    args = parser.parse_args()

    in_path = args.zip if args.zip else args.input_dir
    if not in_path:
        # Default fallback to default results folder if exists
        default_dir = REPO_ROOT / "research" / "evidence" / "document_pilot" / "feasibility_results"
        if default_dir.exists():
            in_path = default_dir
        else:
            parser.error("Must provide either --zip or --input-dir.")

    verify_feasibility_results(in_path, args.output_receipt)


if __name__ == "__main__":
    main()
