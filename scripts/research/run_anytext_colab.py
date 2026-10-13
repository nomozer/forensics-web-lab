#!/usr/bin/env python3
"""AnyText Document Generation Feasibility Colab Runner.

Runs AnyText text-editing on two locked replacement sources (Attempt A & B)
on Google Colab GPU without Hugging Face token or paid dependencies.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_gpu_and_resources(min_vram_gb: float = 7.0) -> Dict[str, Any]:
    """Inspects CUDA device, VRAM, and disk storage."""
    res: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "None",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "disk_free_gb": 0.0,
    }
    # Disk check
    disk = shutil.disk_usage("/")
    res["disk_free_gb"] = round(disk.free / (1024**3), 2)

    try:
        import torch
        if torch.cuda.is_available():
            res["cuda_available"] = True
            res["device_name"] = torch.cuda.get_device_name(0)
            res["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            res["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    except Exception as e:
        res["error"] = str(e)
    return res


def resolve_font_path(custom_path: Optional[str] = None) -> str:
    """Finds an available open-source TrueType font for glyph rendering."""
    candidates = [
        custom_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "font/Arial_Unicode.ttf",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())

    # Fallback to downloading or apt installing
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-dejavu-core"], check=False)
        if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").is_file():
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    except Exception:
        pass

    raise FileNotFoundError("Could not find a valid open-source TTF font for AnyText glyph rendering.")


def run_anytext_inference(
    pipe: Any,
    ori_image_canvas: Image.Image,
    draw_pos_canvas: Image.Image,
    prompt: str,
    target_text: str,
    seed: int,
    sampling_params: Dict[str, Any],
    a_prompt: str,
    n_prompt: str,
) -> Tuple[Image.Image, float, Dict[str, Any]]:
    """Invokes AnyText pipeline in text-editing mode."""
    t0 = time.time()

    # AnyText expectations:
    # ori_image: numpy array (H, W, 3) RGB
    # draw_pos: numpy array (H, W, 1) where 255 is the text edit region
    ori_np = np.array(ori_image_canvas.convert("RGB"))
    pos_np = np.array(draw_pos_canvas.convert("L"))[..., np.newaxis]

    input_data = {
        "prompt": prompt,
        "seed": seed,
        "draw_pos": pos_np,
        "ori_image": ori_np,
    }

    forward_params = {
        "mode": "text-editing",
        "image_count": 1,
        "ddim_steps": sampling_params.get("ddim_steps", 20),
        "cfg_scale": sampling_params.get("cfg_scale", 9.0),
        "strength": sampling_params.get("strength", 1.0),
        "eta": sampling_params.get("eta", 0.0),
        "a_prompt": a_prompt,
        "n_prompt": n_prompt,
        "show_debug": False,
        "revise_pos": False,
    }

    results, rtn_code, rtn_warning, debug_info = pipe(input_data, **forward_params)
    latency_sec = time.time() - t0

    if rtn_code < 0 or not results:
        raise RuntimeError(f"AnyText inference failed with code {rtn_code}: {rtn_warning}")

    raw_output_np = results[0]
    raw_output_img = Image.fromarray(raw_output_np.astype(np.uint8), mode="RGB")
    meta = {
        "latency_sec": round(latency_sec, 2),
        "rtn_code": rtn_code,
        "rtn_warning": rtn_warning,
    }
    return raw_output_img, latency_sec, meta


def generate_anytext_contact_sheet(
    attempts_data: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates a high-res comparison contact sheet for the AnyText attempts."""
    thumb_w, thumb_h = 320, 320
    header_h = 60
    label_h = 30
    cols = 5  # Authentic, SDXL Output, AnyText Raw, AnyText Final, Mask Inspection
    rows = len(attempts_data)
    margin = 15

    total_w = cols * thumb_w + (cols + 1) * margin
    total_h = header_h + rows * (thumb_h + label_h) + (rows + 1) * margin

    sheet = Image.new("RGB", (total_w, total_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.text(
        (margin, 15),
        "AnyText Feasibility Study: 2 Replacement Sources vs Authentic & SDXL",
        fill=(20, 20, 20),
    )

    col_titles = [
        "1. Authentic (I0)",
        "2. SDXL Pilot (I2)",
        "3. AnyText Raw (512x512)",
        "4. AnyText Final Composited",
        "5. Intervention Mask",
    ]
    for c_idx, title in enumerate(col_titles):
        x = margin + c_idx * (thumb_w + margin)
        draw.text((x, 40), title, fill=(80, 80, 80))

    for r_idx, att in enumerate(attempts_data):
        y_top = header_h + margin + r_idx * (thumb_h + label_h + margin)
        draw.text(
            (margin, y_top),
            f"Attempt {att['id']}: {att['target_text']} (Seed {att['seed']})",
            fill=(10, 50, 120),
        )

        images = [
            att.get("authentic_img"),
            att.get("sdxl_img"),
            att.get("anytext_raw_img"),
            att.get("anytext_final_img"),
            att.get("mask_img"),
        ]

        for c_idx, img in enumerate(images):
            x = margin + c_idx * (thumb_w + margin)
            y = y_top + label_h
            if img:
                thumb = img.copy()
                thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.BICUBIC)
                # Center thumbnail
                offset_x = x + (thumb_w - thumb.width) // 2
                offset_y = y + (thumb_h - thumb.height) // 2
                sheet.paste(thumb, (offset_x, offset_y))
                # Border
                draw.rectangle([offset_x, offset_y, offset_x + thumb.width, offset_y + thumb.height], outline=(200, 200, 200), width=1)
            else:
                draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill=(230, 230, 230))
                draw.text((x + 20, y + thumb_h // 2), "N/A", fill=(120, 120, 120))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")
    print(f"[OK] Contact sheet written to {output_path}")


def resolve_input_file(inputs_dir: Path, att: Dict[str, Any]) -> Optional[Path]:
    """Resolves authentic control image path from inputs_dir."""
    att_id = att["attempt_id"]
    row_idx = att.get("source_row_index")
    # Potential directories and filenames
    candidates = [
        inputs_dir / att_id / "authentic_control.png",
        inputs_dir / f"{att_id}.png",
        inputs_dir / f"attempt_{0 if 'price' in att_id else 1}_{att_id.split('_', 2)[-1]}" / "authentic_control.png",
    ]
    if row_idx == 22:
        candidates.append(inputs_dir / "attempt_0_replacement_price" / "authentic_control.png")
    elif row_idx == 61:
        candidates.append(inputs_dir / "attempt_1_replacement_item_name" / "authentic_control.png")

    # Also search by glob inside inputs_dir
    for c in candidates:
        if c.is_file():
            return c
    for p in inputs_dir.glob("**/authentic_control.png"):
        if row_idx == 22 and "price" in str(p):
            return p
        if row_idx == 61 and "item_name" in str(p):
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AnyText Feasibility Runner")
    parser.add_argument("--plan-path", type=str, required=True, help="Path to anytext_feasibility_plan.json")
    parser.add_argument("--inputs-dir", type=str, required=True, help="Path to directory containing input authentic controls")
    parser.add_argument("--output-root", type=str, required=True, help="Path to output directory")
    parser.add_argument("--custom-font", type=str, default=None, help="Custom TTF font path")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only")
    args = parser.parse_args()

    plan_path = Path(args.plan_path).resolve()
    inputs_dir = Path(args.inputs_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not plan_path.is_file():
        print(f"[FAIL] Plan not found: {plan_path}")
        return 1

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_hash_short = hashlib.sha256(plan_path.read_bytes()).hexdigest()[:8]
    run_id = f"run_anytext_{run_timestamp}_{plan_hash_short}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ANYTEXT FEASIBILITY RUNNER: {run_id} ===")
    hw_info = check_gpu_and_resources()
    print(f"CUDA Available: {hw_info['cuda_available']} ({hw_info['device_name']}, VRAM: {hw_info['vram_total_gb']} GB)")
    print(f"Disk Free: {hw_info['disk_free_gb']} GB")

    font_path = resolve_font_path(args.custom_font)
    print(f"Resolved Font: {font_path}")

    # Check input files
    print("\n--- Verifying Input Sources ---")
    attempts = plan["attempts"]
    for att in attempts:
        att_file = resolve_input_file(inputs_dir, att)
        if not att_file or not att_file.is_file():
            print(f"[FAIL] Missing input file for {att['attempt_id']} in {inputs_dir}")
            return 1
        observed_sha = compute_sha256(att_file)
        if observed_sha != att["authentic_sha256"]:
            print(f"[FAIL] SHA mismatch for {att['attempt_id']}: expected {att['authentic_sha256']}, got {observed_sha}")
            return 1
        print(f"[PASS] Verified input {att['attempt_id']}: {observed_sha[:16]}...")

    if args.preflight_only:
        print("[OK] Preflight completed successfully.")
        return 0

    if not hw_info["cuda_available"]:
        print("[FAIL] CUDA GPU is required for AnyText generation. Running on CPU is not supported.")
        return 1

    # Staging AnyText pipeline
    print("--- Loading AnyText Model from ModelScope ---")
    os.environ["MODELSCOPE_CACHE"] = str(run_dir / ".cache_modelscope")
    from modelscope.pipelines import pipeline

    pipe = pipeline(
        "my-anytext-task",
        model=plan["model_specification"]["modelscope_repo_id"],
        model_revision=plan["model_specification"]["model_revision"],
        use_fp16=True,
        use_translator=False,  # Exclude Chinese NLP translation model to respect 12 GiB budget
        font_path=font_path,
    )
    print("[OK] AnyText pipeline initialized successfully.")

    # Import geometry adapter
    from ml.evaluation.anytext_document_adapter import (
        prepare_anytext_editing_inputs,
        composite_anytext_editing_output,
    )

    contact_sheet_items = []
    review_records = []

    for att in attempts:
        att_id = att["attempt_id"]
        att_dir = run_dir / att_id
        att_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Executing {att_id} ---")

        # Load authentic
        att_input_file = resolve_input_file(inputs_dir, att)
        if not att_input_file or not att_input_file.is_file():
            print(f"[FAIL] Missing input file for {att['attempt_id']}")
            return 1
        auth_img = Image.open(att_input_file).convert("RGB")

        # Prepare inputs
        prep_res = prepare_anytext_editing_inputs(
            image=auth_img,
            target_bbox=att["target_bbox"],
            canvas_size=att["sampling_parameters"].get("canvas_width", 512),
            context_margin_ratio=att["sampling_parameters"].get("context_margin_ratio", 0.5),
        )

        # Run AnyText
        raw_canvas, latency, inf_meta = run_anytext_inference(
            pipe=pipe,
            ori_image_canvas=prep_res["ori_image_canvas"],
            draw_pos_canvas=prep_res["draw_pos_canvas"],
            prompt=att["prompt"],
            target_text=att["target_text"],
            seed=att["seed"],
            sampling_params=att["sampling_parameters"],
            a_prompt=att["a_prompt"],
            n_prompt=att["n_prompt"],
        )

        # Composite output
        comp_res = composite_anytext_editing_output(
            original_image=auth_img,
            raw_anytext_output=raw_canvas,
            geometry_receipt=prep_res["geometry_receipt"],
            native_mask=prep_res["native_mask"],
        )

        final_img = comp_res["final_edited_document"]
        if not comp_res["is_outside_equal"]:
            print(f"[FAIL] Outside-mask equality violated for {att_id} (max diff: {comp_res['max_outside_diff']})")
            return 1
        print(f"[PASS] Outside-mask equality verified (diff = 0)")

        # Save artifacts
        auth_save = att_dir / "authentic_control.png"
        raw_save = att_dir / "anytext_raw_canvas.png"
        final_save = att_dir / "anytext_final_edited.png"
        mask_save = att_dir / "native_mask.png"
        draw_pos_save = att_dir / "draw_pos_canvas.png"

        auth_img.save(auth_save, "PNG")
        raw_canvas.save(raw_save, "PNG")
        final_img.save(final_save, "PNG")
        prep_res["native_mask"].save(mask_save, "PNG")
        prep_res["draw_pos_canvas"].save(draw_pos_save, "PNG")

        # Save attempt receipt
        att_receipt = {
            "attempt_id": att_id,
            "status": "TECHNICAL_QC_PASS",
            "operation": att["operation"],
            "target_text": att["target_text"],
            "seed": att["seed"],
            "latency_sec": latency,
            "outside_mask_bitwise_equal": comp_res["is_outside_equal"],
            "max_outside_diff": comp_res["max_outside_diff"],
            "hashes": {
                "authentic_sha256": compute_sha256(auth_save),
                "raw_canvas_sha256": compute_sha256(raw_save),
                "final_edited_sha256": compute_sha256(final_save),
                "native_mask_sha256": compute_sha256(mask_save),
            },
            "geometry_receipt": prep_res["geometry_receipt"],
        }
        (att_dir / "attempt_receipt.json").write_text(json.dumps(att_receipt, indent=2), encoding="utf-8")

        # Collect for contact sheet
        sdxl_img_path = inputs_dir / att["attempt_id"] / "final_edited_document.png"
        sdxl_img = Image.open(sdxl_img_path).convert("RGB") if sdxl_img_path.is_file() else None

        contact_sheet_items.append({
            "id": att_id,
            "target_text": att["target_text"],
            "seed": att["seed"],
            "authentic_img": auth_img,
            "sdxl_img": sdxl_img,
            "anytext_raw_img": raw_canvas,
            "anytext_final_img": final_img,
            "mask_img": prep_res["native_mask"],
        })

        review_records.append({
            "attempt_id": att_id,
            "target_text": att["target_text"],
            "human_verdict": "PENDING_HUMAN_REVIEW",
            "text_legibility_score": "",
            "background_realism_score": "",
            "notes": "",
        })

    # Contact sheet
    cs_path = run_dir / "anytext_contact_sheet.png"
    generate_anytext_contact_sheet(contact_sheet_items, cs_path)

    # Review CSV
    csv_path = run_dir / "human_review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_records[0].keys()))
        writer.writeheader()
        writer.writerows(review_records)

    # Package ZIP archive
    zip_path = output_root / "anytext_feasibility_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_p, _, files in os.walk(run_dir):
            for file in files:
                p = Path(root_p) / file
                rel = p.relative_to(run_dir)
                zf.write(p, arcname=str(Path(run_id) / rel))

    print(f"\n[DONE] AnyText results archived to: {zip_path}")
    print(f"       SHA-256: {compute_sha256(zip_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
