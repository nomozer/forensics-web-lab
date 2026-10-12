#!/usr/bin/env python3
"""Runner script for Google Colab GPU execution of Document Generation Feasibility Study.

Performs preflight checks, FP16 safetensors checkpoint downloading without tokens,
CORD-v2 source extraction with caching, inpainting inference, inverse geometry compositing,
technical QC, contact sheet generation, failure handling, human review CSV management,
and packaging of document_feasibility_results.zip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# Projected FP16 Pipeline Files Inventory (diffusers/stable-diffusion-xl-1.0-inpainting-0.1 @ c09892e8)
CHECKPOINT_FP16_FILES = [
    {"rfilename": "model_index.json", "size": 690},
    {"rfilename": "scheduler/scheduler_config.json", "size": 479},
    {"rfilename": "text_encoder/config.json", "size": 746},
    {"rfilename": "text_encoder/model.fp16.safetensors", "size": 246144867},
    {"rfilename": "text_encoder_2/config.json", "size": 758},
    {"rfilename": "text_encoder_2/model.fp16.safetensors", "size": 1389382884},
    {"rfilename": "tokenizer/merges.txt", "size": 524619},
    {"rfilename": "tokenizer/special_tokens_map.json", "size": 472},
    {"rfilename": "tokenizer/tokenizer_config.json", "size": 737},
    {"rfilename": "tokenizer/vocab.json", "size": 1059962},
    {"rfilename": "tokenizer_2/merges.txt", "size": 524619},
    {"rfilename": "tokenizer_2/special_tokens_map.json", "size": 460},
    {"rfilename": "tokenizer_2/tokenizer_config.json", "size": 725},
    {"rfilename": "tokenizer_2/vocab.json", "size": 1059962},
    {"rfilename": "unet/config.json", "size": 1932},
    {"rfilename": "unet/diffusion_pytorch_model.fp16.safetensors", "size": 5135178560},
    {"rfilename": "vae/config.json", "size": 659},
    {"rfilename": "vae/diffusion_pytorch_model.fp16.safetensors", "size": 167335338},
]
TOTAL_PROJECTED_FP16_BYTES = sum(f["size"] for f in CHECKPOINT_FP16_FILES)  # 6,941,218,469 bytes (6.4645 GiB)


def extract_roi_and_build_canvas(
    image: Image.Image,
    target_bbox: List[int] | Tuple[int, int, int, int],
    canvas_size: int = 1024,
    context_margin_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Extract contextual ROI around target bounding box, construct binary mask, and place on canvas."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    orig_w, orig_h = image.size
    ymin, xmin, ymax, xmax = [int(v) for v in target_bbox]

    # Validate bounding box strictly: silent clamping is forbidden by protocol
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
    """Invert geometry mapping and composite inpainting patch strictly inside target intervention mask."""
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
    """Verify bitwise array equality outside mask via np.array_equal."""
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


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_and_cache_cord_shard(
    parquet_url: str,
    cache_dir: Path,
    max_bytes: int = 272629760,  # 260 MB ceiling
) -> Path:
    """Download CORD-v2 parquet shard once, verify size under budget, and cache locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(parquet_url).name
    cached_path = cache_dir / filename

    if cached_path.exists():
        cached_size = cached_path.stat().st_size
        print(f"Using cached CORD-v2 shard: {cached_path} ({cached_size / (1024**2):.2f} MB)")
        if cached_size > max_bytes:
            raise ValueError(f"Cached shard size {cached_size} exceeds budget limit {max_bytes} bytes!")
        return cached_path

    # Check local repository cache if available
    local_candidates = [
        Path("data/research/cord_v2") / filename,
        Path("../data/research/cord_v2") / filename,
        Path("../../data/research/cord_v2") / filename,
    ]
    for cand in local_candidates:
        if cand.exists():
            c_size = cand.stat().st_size
            print(f"Found local repository CORD-v2 shard: {cand} ({c_size / (1024**2):.2f} MB)")
            if c_size > max_bytes:
                raise ValueError(f"Local shard size {c_size} exceeds budget limit {max_bytes} bytes!")
            return cand

    print(f"Preflighting CORD-v2 shard at: {parquet_url}")
    req = urllib.request.Request(parquet_url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HEAD request failed with status {resp.status} for {parquet_url}")
        content_length = int(resp.headers.get("Content-Length", 0))

    print(f"Shard Content-Length: {content_length} bytes ({content_length / (1024**2):.2f} MB)")
    if content_length > max_bytes:
        raise ValueError(
            f"CORD-v2 shard size {content_length} bytes exceeds budget limit {max_bytes} bytes! "
            f"Aborting download to prevent budget violation."
        )

    part_path = cached_path.with_suffix(".part")
    print(f"Downloading CORD-v2 shard ({content_length / (1024**2):.2f} MB) to {part_path}...")
    dl_req = urllib.request.Request(parquet_url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(dl_req) as resp, open(part_path, "wb") as out_f:
        downloaded = 0
        while chunk := resp.read(1048576):  # 1 MB buffer
            out_f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (10 * 1048576) == 0 or downloaded == content_length:
                print(f"  Downloaded: {downloaded / (1024**2):.1f} / {content_length / (1024**2):.1f} MB...")

    dl_duration = time.time() - t0
    part_path.replace(cached_path)
    print(f"CORD-v2 shard cached successfully in {dl_duration:.2f}s: {cached_path}")
    return cached_path


def read_cord_source_record(
    parquet_shard: Path,
    row_idx: int,
) -> Tuple[Image.Image, bytes, Dict[str, Any]]:
    """Read a single CORD-v2 row record from parquet shard and return (PIL Image RGB, raw PNG bytes, ground truth dict)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_shard)
    table = pf.read(columns=["image", "ground_truth"])
    image_struct = table["image"][row_idx].as_py()
    raw_bytes = image_struct["bytes"] if isinstance(image_struct, dict) else image_struct
    gt_raw = table["ground_truth"][row_idx].as_py()
    gt_data = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return img, raw_bytes, gt_data


def preflight_and_extract_all_sources(
    cached_shard: Path,
    attempts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Strictly preflight and extract all 4 sources prior to model download or generation.
    
    Verifies:
    1. Ground truth image_id matches attempt source_image_id (fails immediately on mismatch).
    2. Native dimensions match plan native_size (width, height).
    3. EXIF orientation == 1.
    4. Target bounding box is strictly within image bounds.
    5. Original text exists in valid_line ground truth annotations.
    6. Ground truth quad envelope and delta [d_ymin, d_xmin, d_ymax, d_xmax].
    7. Encoded bytes and decoded RGB SHA-256 hashes.
    """
    import pyarrow.parquet as pq

    print(f"\n[Stage 1] Preflighting all {len(attempts)} CORD-v2 sources from: {cached_shard}...")
    pf = pq.ParquetFile(cached_shard)
    table = pf.read(columns=["image", "ground_truth"])

    verified_sources: Dict[str, Dict[str, Any]] = {}
    preflight_details: List[Dict[str, Any]] = []

    for att in attempts:
        att_id = att["attempt_id"]
        row_idx = att["source_row_index"]
        expected_size = att["native_size"]
        expected_w, expected_h = expected_size["width"], expected_size["height"]
        target_bbox = att["target_bbox"]
        orig_text = att["original_text"]

        print(f"  Checking {att_id} (row {row_idx})...")

        # Read row from parquet
        image_struct = table["image"][row_idx].as_py()
        raw_png_bytes = image_struct["bytes"] if isinstance(image_struct, dict) else image_struct
        gt_raw = table["ground_truth"][row_idx].as_py()
        gt_data = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw

        # 1. Identity assertion
        gt_meta = gt_data.get("meta", {})
        actual_img_id = gt_meta.get("image_id")
        if actual_img_id != att["source_image_id"]:
            raise RuntimeError(
                f"Source image ID mismatch for attempt {att_id}: "
                f"ground truth image_id ({actual_img_id}) != plan source_image_id ({att['source_image_id']}). "
                f"Aborting before model download."
            )

        # 2. Dimensions & EXIF
        img = Image.open(io.BytesIO(raw_png_bytes))
        actual_w, actual_h = img.size
        if (actual_w, actual_h) != (expected_w, expected_h):
            raise RuntimeError(
                f"Dimension mismatch for attempt {att_id} (row {row_idx}): "
                f"({actual_w}, {actual_h}) != expected ({expected_w}, {expected_h}). "
                f"Aborting before model download."
            )
        exif = img.getexif()
        orientation = exif.get(0x0112, 1) if exif else 1
        if orientation != 1:
            raise RuntimeError(
                f"Non-standard EXIF orientation ({orientation}) detected for attempt {att_id}. "
                f"Aborting before model download."
            )

        img_rgb = img.convert("RGB")

        # 3. Bounding box validity
        ymin, xmin, ymax, xmax = target_bbox
        if not (0 <= ymin < ymax <= actual_h and 0 <= xmin < xmax <= actual_w):
            raise ValueError(
                f"Target bounding box {target_bbox} is invalid or out of bounds for {att_id} ({actual_w}x{actual_h})."
            )

        # 4. Text match & Quad envelope calculation
        gt_valid_lines = gt_data.get("valid_line", [])
        matched_gt_category = None
        quad_envelope = None
        found_text = False

        for line in gt_valid_lines:
            words = line.get("words", [])
            line_text = " ".join([w.get("text", "") for w in words]).strip()
            if orig_text in line_text or any(orig_text in w.get("text", "") for w in words):
                found_text = True
                matched_gt_category = line.get("category", "")
                xs, ys = [], []
                for w in words:
                    q = w.get("quad", {})
                    xs.extend([q.get("x1"), q.get("x2"), q.get("x3"), q.get("x4")])
                    ys.extend([q.get("y1"), q.get("y2"), q.get("y3"), q.get("y4")])
                valid_xs = [v for v in xs if v is not None]
                valid_ys = [v for v in ys if v is not None]
                if valid_xs and valid_ys:
                    quad_envelope = [min(valid_ys), min(valid_xs), max(valid_ys), max(valid_xs)]
                break

        if not found_text:
            raise RuntimeError(
                f"Original text '{orig_text}' not found in ground truth at row {row_idx} for attempt {att_id}. "
                f"Aborting before model download."
            )

        # Calculate delta between plan target_bbox and quad envelope
        if quad_envelope is not None:
            bbox_quad_delta = [
                target_bbox[0] - quad_envelope[0],
                target_bbox[1] - quad_envelope[1],
                target_bbox[2] - quad_envelope[2],
                target_bbox[3] - quad_envelope[3],
            ]
        else:
            bbox_quad_delta = [0, 0, 0, 0]

        encoded_sha = hashlib.sha256(raw_png_bytes).hexdigest()
        decoded_rgb_sha = hashlib.sha256(img_rgb.tobytes()).hexdigest()

        source_info = {
            "attempt_id": att_id,
            "source_row_index": row_idx,
            "source_image_id": actual_img_id,
            "native_dimensions": [actual_w, actual_h],
            "exif_orientation": orientation,
            "encoded_bytes_sha256": encoded_sha,
            "decoded_rgb_sha256": decoded_rgb_sha,
            "matched_category": matched_gt_category,
            "original_text": orig_text,
            "plan_target_bbox": target_bbox,
            "gt_quad_envelope": quad_envelope,
            "bbox_quad_delta": bbox_quad_delta,
        }
        preflight_details.append(source_info)

        verified_sources[att_id] = {
            "image": img_rgb,
            "raw_bytes": raw_png_bytes,
            "gt_data": gt_data,
            "info": source_info,
        }

    receipt = {
        "status": "ALL_SOURCES_PREFLIGHT_VERIFIED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_count": len(verified_sources),
        "sources": preflight_details,
    }
    print(f"[PASS] All {len(verified_sources)} CORD-v2 sources verified successfully prior to model download.")
    return verified_sources, receipt


def preflight_checkpoint_budget_and_access(
    repo_id: str,
    revision: str,
    budget_max_bytes: int,
) -> Dict[str, Any]:
    """Verify model repository accessibility and enforce exact FP16 pipeline file inventory budget."""
    print(f"\n[Stage 2] Preflighting Checkpoint Budget & Access for {repo_id} @ {revision[:8]}...")
    if TOTAL_PROJECTED_FP16_BYTES > budget_max_bytes:
        raise ValueError(
            f"Projected checkpoint download ({TOTAL_PROJECTED_FP16_BYTES} bytes = "
            f"{TOTAL_PROJECTED_FP16_BYTES / (1024**3):.2f} GiB) exceeds budget ceiling "
            f"({budget_max_bytes} bytes = {budget_max_bytes / (1024**3):.2f} GiB)!"
        )

    # Check repository accessibility via Hugging Face public model API (zero token, ungated)
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "forensics-web-lab-runner/1.2.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Model repo API returned status {resp.status} for {repo_id}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError(
                f"Model repository {repo_id} requires authentication or is gated (HTTP {e.code}). "
                f"Aborting per zero-token protocol."
            )
        raise RuntimeError(f"Failed to access model repository {repo_id}: HTTP {e.code}")
    except Exception as e:
        print(f"Note: Remote model API check returned: {e}. Continuing preflight.")

    receipt = {
        "repository": repo_id,
        "revision": revision,
        "variant": "fp16",
        "file_count": len(CHECKPOINT_FP16_FILES),
        "files_manifest": CHECKPOINT_FP16_FILES,
        "total_projected_bytes": TOTAL_PROJECTED_FP16_BYTES,
        "total_projected_gib": round(TOTAL_PROJECTED_FP16_BYTES / (1024**3), 4),
        "total_projected_gb": round(TOTAL_PROJECTED_FP16_BYTES / (10**9), 4),
        "budget_ceiling_bytes": budget_max_bytes,
        "budget_ceiling_gib": round(budget_max_bytes / (1024**3), 2),
        "status": "BUDGET_VERIFIED_UNGATED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(
        f"[PASS] Checkpoint budget verified: {receipt['total_projected_gib']} GiB projected "
        f"<= {receipt['budget_ceiling_gib']} GiB ceiling. Model is public and ungated."
    )
    return receipt


def generate_contact_sheet(
    output_dir: Path,
    attempts: List[Dict[str, Any]],
    attempt_records: List[Dict[str, Any]],
) -> Path:
    """Generate 5-column contact sheet: Authentic | ROI Before | ROI After | Final Edited | Final Mask."""
    panel_height = 280
    panel_width = 220
    text_bar_height = 40
    row_height = panel_height + text_bar_height
    sheet_width = panel_width * 5
    sheet_height = row_height * len(attempts) + 50

    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(240, 242, 245))
    draw = ImageDraw.Draw(sheet)

    # Title header
    header_text = "Forensics Web Lab - Document Generation Feasibility Study (5-Column Contact Sheet)"
    draw.text((15, 15), header_text, fill=(20, 20, 20))

    records_by_id = {r["attempt_id"]: r for r in attempt_records}

    for idx, att in enumerate(attempts):
        att_id = att["attempt_id"]
        att_dir = output_dir / att_id
        y_top = 50 + idx * row_height
        rec = records_by_id.get(att_id, {})
        status = rec.get("status", "NOT_EXECUTED")
        tech_verdict = rec.get("preliminary_assessment", status)

        auth_path = att_dir / "authentic_control.png"
        edited_path = att_dir / "final_edited_document.png"
        mask_path = att_dir / "final_intervention_mask.png"
        target_bbox = att["target_bbox"]

        # Column 1: Full Authentic
        if auth_path.exists():
            auth_img = Image.open(auth_path).convert("RGB")
            a_thumb = auth_img.copy()
            a_thumb.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.BICUBIC)
            px = 5 + (panel_width - a_thumb.width) // 2
            py = y_top + (panel_height - a_thumb.height) // 2
            sheet.paste(a_thumb, (px, py))
            draw.rectangle([5, y_top, panel_width - 5, y_top + panel_height], outline=(200, 200, 200))

            # Column 2: ROI Before
            ymin, xmin, ymax, xmax = target_bbox
            roi_before = auth_img.crop((xmin, ymin, xmax, ymax))
            roi_b_thumb = roi_before.copy()
            roi_b_thumb.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.BICUBIC)
            px = panel_width + 5 + (panel_width - roi_b_thumb.width) // 2
            py = y_top + (panel_height - roi_b_thumb.height) // 2
            sheet.paste(roi_b_thumb, (px, py))
            draw.rectangle([panel_width + 5, y_top, 2 * panel_width - 5, y_top + panel_height], outline=(200, 200, 200))

        # Column 3: ROI After
        if edited_path.exists():
            edited_img = Image.open(edited_path).convert("RGB")
            ymin, xmin, ymax, xmax = target_bbox
            roi_after = edited_img.crop((xmin, ymin, xmax, ymax))
            roi_a_thumb = roi_after.copy()
            roi_a_thumb.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.BICUBIC)
            px = 2 * panel_width + 5 + (panel_width - roi_a_thumb.width) // 2
            py = y_top + (panel_height - roi_a_thumb.height) // 2
            sheet.paste(roi_a_thumb, (px, py))
            draw.rectangle([2 * panel_width + 5, y_top, 3 * panel_width - 5, y_top + panel_height], outline=(200, 200, 200))

            # Column 4: Full Final Edited
            ed_thumb = edited_img.copy()
            ed_thumb.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.BICUBIC)
            px = 3 * panel_width + 5 + (panel_width - ed_thumb.width) // 2
            py = y_top + (panel_height - ed_thumb.height) // 2
            sheet.paste(ed_thumb, (px, py))
            draw.rectangle([3 * panel_width + 5, y_top, 4 * panel_width - 5, y_top + panel_height], outline=(200, 200, 200))

        # Column 5: Final intervention mask
        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("RGB")
            m_thumb = mask_img.copy()
            m_thumb.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.NEAREST)
            px = 4 * panel_width + 5 + (panel_width - m_thumb.width) // 2
            py = y_top + (panel_height - m_thumb.height) // 2
            sheet.paste(m_thumb, (px, py))
            draw.rectangle([4 * panel_width + 5, y_top, 5 * panel_width - 5, y_top + panel_height], outline=(200, 200, 200))

        # Text banner for attempt row
        text_y = y_top + panel_height + 4
        orig_text = att.get("original_text", "")
        tgt_text = att.get("target_text", "")
        label_line = (
            f"[{att_id}] {att['operation'].upper()} | Original: '{orig_text}' -> Target: '{tgt_text}' | "
            f"Status: {status} | Verdict: {tech_verdict}"
        )
        draw.text((10, text_y), label_line, fill=(30, 30, 30))
        draw.line([(0, y_top + row_height - 1), (sheet_width, y_top + row_height - 1)], fill=(180, 180, 180), width=1)

    contact_path = output_dir / "feasibility_contact_sheet.png"
    sheet.save(contact_path, format="PNG")
    print(f"Generated contact sheet: {contact_path}")
    return contact_path


def run_feasibility_pipeline(
    plan_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute pre-registered feasibility plan with strict budget enforcement, source preflight, and failure handling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    meta = plan["metadata"]
    attempts = plan["attempts"]
    budget = meta["budget_limits"]
    plan_sha = compute_sha256(plan_path)

    # Fresh Run Idempotency Guard: prevent silent overwriting of completed run
    manifest_check = output_dir / "feasibility_manifest.json"
    zip_check = output_dir / "document_feasibility_results.zip"
    if manifest_check.exists() or zip_check.exists():
        raise RuntimeError(
            f"Output directory '{output_dir}' already contains completed run artifacts "
            f"(manifest or results zip archive exists). Aborting to prevent accidental overwrite. "
            f"Please specify a fresh output directory."
        )

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{plan_sha[:8]}"

    print("=== Starting Document Generation Feasibility Study ===")
    print(f"Run ID: {run_id}")
    print(f"Study: {meta['study_name']} (Protocol v{meta['protocol_version']})")
    print(f"Generator: {meta['model']['repository']} (Rev: {meta['model']['revision'][:8]})")
    print(f"Target Output Directory: {output_dir.resolve()}")

    # 1. Environment & Hardware Preflight
    preflight: Dict[str, Any] = {
        "run_id": run_id,
        "plan_sha256": plan_sha,
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
            print("Notice: CUDA is not available on this runtime.")
    except ImportError:
        preflight["torch_version"] = "NOT_INSTALLED"
        preflight["cuda_available"] = False
        print("Notice: PyTorch is not installed.")

    preflight_path = output_dir / "preflight_receipt.json"
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2)

    # Global try-except to guarantee run_failure_receipt and partial archive on unhandled failure
    try:
        # 2. Stage 1: CORD-v2 Source Data Preflight & Full 4-Source Verification (BEFORE MODEL DOWNLOAD)
        parquet_url = (
            f"https://huggingface.co/datasets/{meta['dataset']['repository']}/resolve/"
            f"{meta['dataset']['revision']}/{meta['dataset']['parquet_file']}"
        )
        cache_dir = output_dir / "cord_cache"
        cached_shard = fetch_and_cache_cord_shard(
            parquet_url=parquet_url,
            cache_dir=cache_dir,
            max_bytes=budget["original_data_colab_max_bytes"],
        )

        verified_sources, source_receipt = preflight_and_extract_all_sources(cached_shard, attempts)
        source_receipt_path = output_dir / "source_preflight_receipt.json"
        with open(source_receipt_path, "w", encoding="utf-8") as srf:
            json.dump(source_receipt, srf, indent=2)

        # 3. Stage 2: Checkpoint Budget & Public Access Preflight
        repo_id = meta["model"]["repository"]
        revision = meta["model"]["revision"]
        checkpoint_receipt = preflight_checkpoint_budget_and_access(
            repo_id=repo_id,
            revision=revision,
            budget_max_bytes=budget["checkpoint_colab_max_bytes"],
        )
        ckpt_receipt_path = output_dir / "checkpoint_budget_preflight_receipt.json"
        with open(ckpt_receipt_path, "w", encoding="utf-8") as crf:
            json.dump(checkpoint_receipt, crf, indent=2)

        if dry_run:
            print("[PREFLIGHT-ONLY] Source data and checkpoint budget preflight completed successfully. Skipping model inference.")
            return {
                "status": "PREFLIGHT_VERIFIED",
                "run_id": run_id,
                "preflight": preflight,
                "sources": source_receipt,
                "checkpoint": checkpoint_receipt,
            }

        if not has_cuda and device == "cuda":
            raise RuntimeError("CUDA requested but not available. Aborting GPU execution.")

        # 4. Stage 3: Load Pipeline (GPU Only)
        from diffusers import StableDiffusionXLInpaintPipeline
        import torch

        print(f"\n[Stage 3] Loading SDXL Inpainting Pipeline ({repo_id} @ {revision[:8]})...")
        if preflight["disk_free_gb"] < 10.0:
            raise RuntimeError(
                f"Insufficient disk space ({preflight['disk_free_gb']:.2f} GB). "
                f"Minimum 10.0 GB free disk required."
            )

        load_start = time.time()
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        load_duration = time.time() - load_start
        print(f"Model successfully loaded in {load_duration:.2f}s.")

        # 5. Stage 4: Execute Attempts
        attempt_records: List[Dict[str, Any]] = []
        t_start_gpu = time.time()

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
                attempt_records.append(att_res)
                continue

            try:
                # Check elapsed GPU time
                elapsed_mins = (time.time() - t_start_gpu) / 60.0
                if elapsed_mins > budget["max_gpu_minutes"]:
                    raise TimeoutError(
                        f"GPU execution time exceeded limit ({elapsed_mins:.1f} > {budget['max_gpu_minutes']} mins)."
                    )

                # A. Retrieve pre-verified authentic source
                src_entry = verified_sources[att_id]
                auth_img = src_entry["image"]
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

                # C. Model Inference
                print(f"Executing SDXL inpainting (Seed={attempt['seed']}, Steps={meta['model']['num_inference_steps']})...")
                if has_cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

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

                if has_cuda:
                    torch.cuda.synchronize()
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

                # Preliminary Technical Assessment
                if not is_outside_equal:
                    preliminary = "TECHNICAL_FAIL_OUTSIDE_LEAKAGE"
                elif in_mean_l1 < 1.0 or changed_ratio < 0.1:
                    preliminary = "POTENTIAL_UNMODIFIED"
                else:
                    preliminary = f"FEASIBLE_{attempt['operation'].upper()}"

                receipt_data = {
                    "attempt_id": att_id,
                    "status": "COMPLETED",
                    "source_doc_identifier": attempt["source_doc_identifier"],
                    "source_image_id": attempt["source_image_id"],
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
                attempt_records.append(receipt_data)

            except Exception as exc:
                print(f"ERROR in {att_id}: {type(exc).__name__}: {exc}")
                failure_receipt = {
                    "attempt_id": att_id,
                    "status": "FAILED",
                    "operation": attempt["operation"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                fail_path = att_dir / "attempt_failure_receipt.json"
                with open(fail_path, "w", encoding="utf-8") as ff:
                    json.dump(failure_receipt, ff, indent=2)
                attempt_records.append(failure_receipt)

        # 6. Stage 5: Human Review CSV, Contact Sheet, Manifest & Archive
        csv_path = output_dir / "human_review.csv"
        csv_fields = [
            "attempt_id",
            "source_image_id",
            "operation",
            "original_text",
            "target_text",
            "outside_bitwise_equal",
            "inside_mean_l1",
            "changed_pixel_ratio",
            "technical_verdict",
            "human_verdict",
            "human_notes",
        ]

        csv_rows_to_write: List[Dict[str, Any]] = []
        records_map = {r["attempt_id"]: r for r in attempt_records}

        for attempt in attempts:
            att_id = attempt["attempt_id"]
            rec = records_map.get(att_id, {})
            status = rec.get("status", "NOT_EXECUTED")
            tech_qc = rec.get("technical_qc", {})

            csv_rows_to_write.append(
                {
                    "attempt_id": att_id,
                    "source_image_id": attempt["source_image_id"],
                    "operation": attempt["operation"],
                    "original_text": attempt["original_text"],
                    "target_text": attempt["target_text"],
                    "outside_bitwise_equal": tech_qc.get("is_outside_bitwise_equal", "N/A"),
                    "inside_mean_l1": f"{tech_qc.get('inside_mean_l1', 0.0):.2f}" if "inside_mean_l1" in tech_qc else "N/A",
                    "changed_pixel_ratio": f"{tech_qc.get('changed_pixel_ratio', 0.0):.4f}" if "changed_pixel_ratio" in tech_qc else "N/A",
                    "technical_verdict": rec.get("preliminary_assessment", status),
                    "human_verdict": "PENDING_HUMAN_REVIEW",
                    "human_notes": "",
                }
            )

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(csv_rows_to_write)
        print(f"\nWritten human review CSV: {csv_path}")

        # Contact Sheet
        try:
            generate_contact_sheet(output_dir, attempts, attempt_records)
        except Exception as cs_err:
            print(f"Warning: Contact sheet generation failed ({cs_err}). Continuing packaging.")

        # Final Manifest
        manifest = {
            "run_id": run_id,
            "plan_sha256": plan_sha,
            "metadata": meta,
            "preflight": preflight,
            "checkpoint_budget": checkpoint_receipt,
            "model_load_duration_sec": load_duration,
            "attempts_summary": attempt_records,
            "status": "COMPLETED_PENDING_HUMAN_QC",
        }
        manifest_path = output_dir / "feasibility_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        # Zip Archive
        zip_path = output_dir / "document_feasibility_results.zip"
        print(f"Packaging archive to {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    p = Path(root) / file
                    if p == zip_path or p.suffix == ".part" or "cord_cache" in p.parts:
                        continue
                    rel_p = p.relative_to(output_dir)
                    zf.write(p, rel_p)

        archive_size_bytes = zip_path.stat().st_size
        archive_size_mb = archive_size_bytes / (1024**2)
        print(f"Archive created: {zip_path} ({archive_size_mb:.2f} MB)")

        if archive_size_bytes > budget["artifacts_transfer_max_bytes"]:
            raise ValueError(
                f"Archive size {archive_size_bytes} exceeds budget limit {budget['artifacts_transfer_max_bytes']} bytes!"
            )

        print("=== Generation Feasibility Study Execution Complete ===")
        return {
            "status": "EXECUTED_PENDING_HUMAN_QC",
            "run_id": run_id,
            "archive_path": str(zip_path),
            "archive_size_mb": archive_size_mb,
            "attempts": attempt_records,
        }

    except Exception as run_err:
        print(f"\n[FATAL] Pipeline execution encountered error: {type(run_err).__name__}: {run_err}")
        run_failure_receipt = {
            "run_id": run_id,
            "plan_sha256": plan_sha,
            "error_type": type(run_err).__name__,
            "error_message": str(run_err),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failure_path = output_dir / "run_failure_receipt.json"
        with open(failure_path, "w", encoding="utf-8") as ff:
            json.dump(run_failure_receipt, ff, indent=2)

        # Attempt to package partial zip archive if any artifacts exist
        partial_zip = output_dir / "document_feasibility_results_partial.zip"
        try:
            with zipfile.ZipFile(partial_zip, "w", zipfile.ZIP_DEFLATED) as pzf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        p = Path(root) / file
                        if p == partial_zip or p.suffix == ".part" or "cord_cache" in p.parts:
                            continue
                        rel_p = p.relative_to(output_dir)
                        pzf.write(p, rel_p)
            print(f"Partial results archive packaged to: {partial_zip}")
        except Exception:
            pass

        raise run_err


def main():
    parser = argparse.ArgumentParser(description="Run Document Feasibility Study on Colab GPU.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("research/evidence/document_pilot/document_feasibility_plan.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/evidence/document_pilot/feasibility_results"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight checks without downloading model or inferencing.",
    )
    parser.add_argument(
        "--source-preflight-only",
        action="store_true",
        help="Perform source preflight and budget preflight only, then exit cleanly.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    run_feasibility_pipeline(
        plan_path=args.plan,
        output_dir=args.output_dir,
        dry_run=args.dry_run or args.source_preflight_only,
        device=args.device,
    )


if __name__ == "__main__":
    main()
