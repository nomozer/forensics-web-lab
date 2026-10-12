#!/usr/bin/env python3
"""Runner script for Google Colab GPU execution of Document Generation Feasibility Study.

Performs preflight checks, FP16 safetensors checkpoint downloading without tokens,
CORD-v2 source extraction, inpainting inference, inverse geometry compositing,
receipt generation, and packaging of document_feasibility_results.zip.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Apply Windows socket port exclusion safety patch if running on Windows
if sys.platform == "win32":
    _orig_create_conn = socket.create_connection

    def _safe_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        if source_address is None:
            for p in range(51000, 55000):
                try:
                    return _orig_create_conn(address, timeout=timeout, source_address=("0.0.0.0", p))
                except (PermissionError, OSError):
                    continue
        return _orig_create_conn(address, timeout=timeout, source_address=source_address)

    socket.create_connection = _safe_create_connection


# Standalone Geometry & Composite functions for complete portability on Colab
def extract_roi_and_build_canvas(
    image: Image.Image,
    target_bbox: List[int] | Tuple[int, int, int, int],
    canvas_size: int = 1024,
    context_margin_ratio: float = 0.5,
) -> Dict[str, Any]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    orig_w, orig_h = image.size
    ymin, xmin, ymax, xmax = [int(v) for v in target_bbox]
    ymin = max(0, min(orig_h - 1, ymin))
    ymax = max(ymin + 1, min(orig_h, ymax))
    xmin = max(0, min(orig_w - 1, xmin))
    xmax = max(xmin + 1, min(orig_w, xmax))

    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    margin_x = int(round(bbox_w * context_margin_ratio))
    margin_y = int(round(bbox_h * context_margin_ratio))

    crop_ymin = max(0, ymin - margin_y)
    crop_ymax = min(orig_h, ymax + margin_y)
    crop_xmin = max(0, xmin - margin_x)
    crop_xmax = min(orig_w, xmax + margin_x)

    crop_w = crop_xmax - crop_xmin
    crop_h = crop_ymax - crop_ymin

    context_crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    target_rel_ymin = ymin - crop_ymin
    target_rel_ymax = ymax - crop_ymin
    target_rel_xmin = xmin - crop_xmin
    target_rel_xmax = xmax - crop_xmin

    mask_crop_arr = np.zeros((crop_h, crop_w), dtype=np.uint8)
    mask_crop_arr[target_rel_ymin:target_rel_ymax, target_rel_xmin:target_rel_xmax] = 255
    mask_crop = Image.fromarray(mask_crop_arr, mode="L")

    scale = min(canvas_size / crop_w, canvas_size / crop_h)
    scaled_w = max(1, min(canvas_size, int(round(crop_w * scale))))
    scaled_h = max(1, min(canvas_size, int(round(crop_h * scale))))

    scaled_image = context_crop.resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
    scaled_mask = mask_crop.resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST)
    scaled_mask_arr = (np.array(scaled_mask) > 127).astype(np.uint8) * 255
    scaled_mask = Image.fromarray(scaled_mask_arr, mode="L")

    pad_x = (canvas_size - scaled_w) // 2
    pad_y = (canvas_size - scaled_h) // 2

    canvas_image = Image.new("RGB", (canvas_size, canvas_size), color=(128, 128, 128))
    canvas_image.paste(scaled_image, (pad_x, pad_y))

    canvas_mask = Image.new("L", (canvas_size, canvas_size), color=0)
    canvas_mask.paste(scaled_mask, (pad_x, pad_y))

    receipt = {
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
        "geometry_receipt": receipt,
    }


def composite_inpainted_patch(
    original_image: Image.Image,
    model_output_canvas: Image.Image,
    geometry_receipt: Dict[str, Any],
) -> Dict[str, Any]:
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

    unpadded_patch = model_output_canvas.crop((pad_x, pad_y, pad_x + scaled_w, pad_y + scaled_h))
    restored_patch = unpadded_patch.resize((crop_w, crop_h), resample=Image.Resampling.BICUBIC)
    restored_patch_arr = np.array(restored_patch)

    orig_arr = np.array(original_image)
    candidate_full_arr = orig_arr.copy()
    candidate_full_arr[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = restored_patch_arr

    final_mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
    final_mask_arr[target_ymin:target_ymax, target_xmin:target_xmax] = 255

    final_edited_arr = np.where(final_mask_arr[:, :, None] == 255, candidate_full_arr, orig_arr)

    return {
        "final_edited_image": Image.fromarray(final_edited_arr, mode="RGB"),
        "final_intervention_mask": Image.fromarray(final_mask_arr, mode="L"),
        "composite_receipt": {
            "status": "COMPOSITED",
            "original_size": [orig_w, orig_h],
            "crop_bbox": [crop_ymin, crop_xmin, crop_ymax, crop_xmax],
            "target_bbox": [target_ymin, target_xmin, target_ymax, target_xmax],
        },
    }


def verify_outside_equality(
    authentic_image: Image.Image | np.ndarray,
    edited_image: Image.Image | np.ndarray,
    intervention_mask: Image.Image | np.ndarray,
) -> Tuple[bool, int]:
    auth_arr = np.array(authentic_image, dtype=np.uint8)
    edit_arr = np.array(edited_image, dtype=np.uint8)
    mask_arr = np.array(intervention_mask, dtype=np.uint8)
    outside = mask_arr == 0
    if not np.any(outside):
        return True, 0
    auth_out = auth_arr[outside]
    edit_out = edit_arr[outside]
    is_eq = bool(np.array_equal(auth_out, edit_out))
    if is_eq:
        return True, 0
    max_d = int(np.max(np.abs(auth_out.astype(np.int32) - edit_out.astype(np.int32))))
    return False, max_d


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fetch_cord_source_image(row_idx: int, parquet_url: str) -> Image.Image:
    """Fetch single source receipt image from CORD-v2 parquet row."""
    import urllib.request
    import pyarrow.parquet as pq

    class RangeFile(io.BufferedIOBase):
        def __init__(self, url: str):
            self.url = url
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                self.total_size = int(resp.headers.get("Content-Length", 0))
            self.pos = 0

        def readable(self): return True
        def seekable(self): return True
        def seek(self, offset, whence=io.SEEK_SET):
            if whence == io.SEEK_SET: self.pos = offset
            elif whence == io.SEEK_CUR: self.pos += offset
            elif whence == io.SEEK_END: self.pos = self.total_size + offset
            return self.pos
        def tell(self): return self.pos
        def read(self, size=-1):
            if size == -1 or self.pos + size > self.total_size: size = self.total_size - self.pos
            if size <= 0: return b""
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + size - 1}", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp: chunk = resp.read()
            self.pos += len(chunk)
            return chunk

    rf = RangeFile(parquet_url)
    pf = pq.ParquetFile(rf)
    # Read specific row group or slice
    # In CORD validation parquet, there is 1 row group with 100 rows
    table = pf.read(columns=["bytes"])
    raw_png_bytes = table["bytes"][row_idx].as_py()
    rf.close()

    img = Image.open(io.BytesIO(raw_png_bytes)).convert("RGB")
    return img


def run_feasibility_pipeline(
    plan_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute pre-registered feasibility plan."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    meta = plan["metadata"]
    attempts = plan["attempts"]
    print(f"=== Starting Document Generation Feasibility Study ===")
    print(f"Study: {meta['study_name']} (Protocol v{meta['protocol_version']})")
    print(f"Generator: {meta['model']['repository']} (Rev: {meta['model']['revision'][:8]})")
    print(f"Target Output Directory: {output_dir.resolve()}")

    # 1. Hardware Preflight Check
    preflight: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": sys.platform,
        "device_requested": device,
    }

    disk_stat = shutil.disk_usage(output_dir)
    preflight["disk_free_gb"] = disk_stat.free / (1024**3)
    print(f"Disk free space: {preflight['disk_free_gb']:.2f} GB")

    has_cuda = False
    try:
        import torch
        preflight["torch_version"] = torch.__version__
        has_cuda = torch.cuda.is_available()
        preflight["cuda_available"] = has_cuda
        if has_cuda:
            preflight["cuda_device_name"] = torch.cuda.get_device_name(0)
            preflight["cuda_device_count"] = torch.cuda.device_count()
            preflight["total_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {preflight['cuda_device_name']} ({preflight['total_vram_gb']:.2f} GB VRAM)")
        else:
            print("WARNING: CUDA is not available! Running on CPU.")
    except ImportError:
        preflight["torch_version"] = "NOT_INSTALLED"
        preflight["cuda_available"] = False
        print("WARNING: PyTorch not installed.")

    preflight_path = output_dir / "preflight_receipt.json"
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2)

    if dry_run:
        print("[DRY-RUN] Preflight completed successfully. Skipping model inference.")
        return {"status": "DRY_RUN_COMPLETED", "preflight": preflight}

    if not has_cuda and device == "cuda":
        raise RuntimeError("CUDA requested but not available. Aborting GPU execution.")

    # 2. Model Loading (SDXL Inpainting FP16 Safetensors)
    from diffusers import StableDiffusionXLInpaintPipeline
    import torch

    repo_id = meta["model"]["repository"]
    revision = meta["model"]["revision"]

    print(f"\n[Stage 1] Loading SDXL Inpainting Pipeline ({repo_id} @ {revision[:8]})...")
    load_start = time.time()
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    # Enable VRAM memory optimizations for T4 compatibility
    pipe.enable_attention_slicing()
    load_duration = time.time() - load_start
    print(f"Model successfully loaded in {load_duration:.2f}s.")

    # Parquet URL
    parquet_url = (
        f"https://huggingface.co/datasets/{meta['dataset']['repository']}/resolve/"
        f"{meta['dataset']['revision']}/{meta['dataset']['parquet_file']}"
    )

    # 3. Execute Attempts
    attempt_results: List[Dict[str, Any]] = []
    csv_rows: List[str] = [
        "attempt_id,source_image_id,operation,original_text,target_text,outside_bitwise_equal,inside_mean_l1,changed_pixel_ratio,preliminary_assessment,human_verdict,human_notes"
    ]

    for attempt in attempts:
        att_id = attempt["attempt_id"]
        att_dir = output_dir / att_id
        att_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Running Attempt: {att_id} ({attempt['operation'].upper()}) ---")

        # Resume check
        receipt_path = att_dir / "attempt_receipt.json"
        if receipt_path.exists():
            print(f"Receipt already exists for {att_id}. Skipping execution (resumed).")
            with open(receipt_path, "r", encoding="utf-8") as rf:
                att_res = json.load(rf)
            attempt_results.append(att_res)
            continue

        # A. Fetch authentic source
        print(f"Fetching CORD-v2 source row {attempt['source_row_index']}...")
        auth_img = fetch_cord_source_image(attempt["source_row_index"], parquet_url)
        auth_path = att_dir / "authentic_control.png"
        auth_img.save(auth_path, format="PNG")

        # B. Geometry ROI & Canvas Build
        canvas_dict = extract_roi_and_build_canvas(
            image=auth_img,
            target_bbox=attempt["target_bbox"],
            canvas_size=meta["model"]["canvas_size"],
            context_margin_ratio=attempt["context_margin_ratio"],
        )
        context_path = att_dir / "generation_context.png"
        canvas_dict["context_crop"].save(context_path, format="PNG")

        geo_receipt_path = att_dir / "geometry_receipt.json"
        with open(geo_receipt_path, "w", encoding="utf-8") as gf:
            json.dump(canvas_dict["geometry_receipt"], gf, indent=2)

        # C. Inference
        print(f"Executing SDXL inpainting (Seed={attempt['seed']}, Steps={meta['model']['num_inference_steps']})...")
        generator = torch.Generator(device="cuda").manual_seed(attempt["seed"])
        inf_start = time.time()
        with torch.inference_mode():
            out = pipe(
                prompt=attempt["prompt"],
                negative_prompt=attempt["negative_prompt"],
                image=canvas_dict["canvas_image"],
                mask_image=canvas_dict["canvas_mask"],
                num_inference_steps=meta["model"]["num_inference_steps"],
                guidance_scale=meta["model"]["guidance_scale"],
                strength=meta["model"]["strength"],
                generator=generator,
            )
        inf_duration = time.time() - inf_start
        raw_output_img = out.images[0]
        raw_output_path = att_dir / "raw_model_output.png"
        raw_output_img.save(raw_output_path, format="PNG")

        # D. Inverse Mapping & Composite
        print("Compositing into original document coordinates...")
        comp_res = composite_inpainted_patch(
            original_image=auth_img,
            model_output_canvas=raw_output_img,
            geometry_receipt=canvas_dict["geometry_receipt"],
        )
        final_edited = comp_res["final_edited_image"]
        final_mask = comp_res["final_intervention_mask"]

        edited_path = att_dir / "final_edited_document.png"
        mask_path = att_dir / "final_intervention_mask.png"
        final_edited.save(edited_path, format="PNG")
        final_mask.save(mask_path, format="PNG")

        # E. Technical Verification
        is_outside_equal, max_outside_diff = verify_outside_equality(auth_img, final_edited, final_mask)
        print(f"Outside-mask bitwise equal: {is_outside_equal} (Max diff: {max_outside_diff})")

        # Pixel change statistics
        auth_arr = np.array(auth_img, dtype=np.int32)
        edit_arr = np.array(final_edited, dtype=np.int32)
        mask_arr = np.array(final_mask, dtype=np.uint8) == 255

        in_diff = np.abs(auth_arr[mask_arr] - edit_arr[mask_arr])
        in_mean_l1 = float(np.mean(in_diff)) if np.any(mask_arr) else 0.0
        changed_count = int(np.sum(np.any(in_diff > 0, axis=-1)))
        total_in = int(np.sum(mask_arr))
        changed_ratio = float(changed_count / total_in) if total_in > 0 else 0.0

        print(f"Inside-mask mean L1: {in_mean_l1:.2f}, Changed pixel ratio: {changed_ratio:.2%}")

        # Preliminary Assessment (Heuristic description)
        preliminary = "TECHNICAL_VALID"
        if not is_outside_equal:
            preliminary = "TECHNICAL_FAIL_OUTSIDE_LEAKAGE"
        elif in_mean_l1 < 1.0 or changed_ratio < 0.1:
            preliminary = "POTENTIAL_UNMODIFIED"
        else:
            preliminary = f"FEASIBLE_{attempt['operation'].upper()}"

        receipt_data = {
            "attempt_id": att_id,
            "source_doc_identifier": attempt["source_doc_identifier"],
            "operation": attempt["operation"],
            "original_text": attempt["original_text"],
            "target_text": attempt["target_text"],
            "inference_duration_sec": inf_duration,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if has_cuda else 0.0,
            "technical_qc": {
                "is_outside_bitwise_equal": is_outside_equal,
                "max_outside_diff": max_outside_diff,
                "inside_mean_l1": in_mean_l1,
                "changed_pixel_count": changed_count,
                "changed_pixel_ratio": changed_ratio,
            },
            "file_hashes": {
                "authentic_control_sha256": compute_sha256(auth_path),
                "generation_context_sha256": compute_sha256(context_path),
                "raw_model_output_sha256": compute_sha256(raw_output_path),
                "final_edited_document_sha256": compute_sha256(edited_path),
                "final_intervention_mask_sha256": compute_sha256(mask_path),
            },
            "preliminary_assessment": preliminary,
        }

        with open(receipt_path, "w", encoding="utf-8") as rf:
            json.dump(receipt_data, rf, indent=2)
        attempt_results.append(receipt_data)

        csv_rows.append(
            f"{att_id},{attempt['source_image_id']},{attempt['operation']},"
            f"\"{attempt['original_text']}\",\"{attempt['target_text']}\","
            f"{is_outside_equal},{in_mean_l1:.2f},{changed_ratio:.4f},{preliminary},"
            f"PENDING_HUMAN_REVIEW,\"\")"
        )

    # 4. Write Human Review CSV
    csv_path = output_dir / "human_review.csv"
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("\n".join(csv_rows) + "\n")
    print(f"\nWritten {csv_path}")

    # 5. Build Final Manifest & Zip Archive
    manifest = {
        "metadata": meta,
        "preflight": preflight,
        "model_load_duration_sec": load_duration,
        "attempts_summary": attempt_results,
    }
    manifest_path = output_dir / "feasibility_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    zip_path = output_dir / "document_feasibility_results.zip"
    print(f"Packaging archive to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                p = Path(root) / file
                if p == zip_path:
                    continue
                rel_p = p.relative_to(output_dir)
                zf.write(p, rel_p)

    archive_size_mb = zip_path.stat().st_size / (1024**2)
    print(f"Archive created successfully: {zip_path} ({archive_size_mb:.2f} MB)")
    print("=== Generation Feasibility Study Execution Complete ===")

    return {
        "status": "EXECUTED_PENDING_HUMAN_QC",
        "archive_path": str(zip_path),
        "archive_size_mb": archive_size_mb,
        "attempts": attempt_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Document Feasibility Study on Colab GPU.")
    parser.add_argument("--plan", type=Path, default=Path("research/evidence/document_pilot/document_feasibility_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/evidence/document_pilot/feasibility_results"))
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight checks without downloading model or inferencing.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    run_feasibility_pipeline(
        plan_path=args.plan,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        device=args.device,
    )


if __name__ == "__main__":
    main()
