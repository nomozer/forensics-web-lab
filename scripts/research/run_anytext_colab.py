#!/usr/bin/env python3
"""AnyText Document Generation Feasibility Colab Runner.

Runs AnyText text-editing on two locked replacement sources (Attempt A & B)
on Google Colab GPU without Hugging Face token or paid dependencies.

Inputs are extracted directly from the verified SDXL pilot archive:
`data/research/local-artifacts/document-feasibility/document_feasibility_results.zip`
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
import traceback
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


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def check_gpu_and_resources(min_vram_gb: float = 7.0) -> Dict[str, Any]:
    """Inspects CUDA device, VRAM, and disk storage."""
    res: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "None",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "disk_free_gb": 0.0,
    }
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


def resolve_font_path(custom_path: Optional[str] = None) -> Tuple[str, str]:
    """Finds an available open-source TrueType font for glyph rendering and returns (path, sha256)."""
    candidates = [
        custom_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "font/Arial_Unicode.ttf",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            p = Path(c).resolve()
            return str(p), compute_sha256(p)

    # Fallback to apt install if on Debian/Ubuntu
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-dejavu-core"], check=False)
        target = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if target.is_file():
            return str(target), compute_sha256(target)
    except Exception:
        pass

    raise FileNotFoundError("Could not find a valid TTF font for AnyText glyph rendering.")


def safe_extract_archive_inputs(
    archive_path: Path,
    target_extract_dir: Path,
    plan: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Extracts authentic controls and SDXL outputs from document_feasibility_results.zip.

    Verifies archive SHA-256 and byte-exact authentic control hashes.
    Rejects directory traversal and absolute paths.
    """
    input_spec = plan.get("input_source_specification", {})
    expected_archive_sha = input_spec.get("archive_sha256")

    observed_archive_sha = compute_sha256(archive_path)
    if expected_archive_sha and observed_archive_sha != expected_archive_sha:
        raise ValueError(
            f"Input archive SHA-256 mismatch! Expected {expected_archive_sha}, got {observed_archive_sha}"
        )
    print(f"[PASS] Archive SHA-256 verified: {observed_archive_sha[:16]}...")

    target_extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_inputs: Dict[str, Dict[str, Any]] = {}

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = set(zf.namelist())

        # Path traversal security check
        for member_name in namelist:
            if member_name.startswith("/") or member_name.startswith("\\") or ".." in Path(member_name).parts:
                raise SecurityError(f"Malicious member path detected in archive: {member_name}")

        # Check manifest inside zip
        manifest_data = {}
        if "feasibility_manifest.json" in namelist:
            manifest_data = json.loads(zf.read("feasibility_manifest.json").decode("utf-8"))

        attempts = plan["attempts"]
        for att in attempts:
            att_id = att["attempt_id"]
            row_idx = att.get("source_row_index")
            expected_auth_sha = att["authentic_sha256"]

            # Match attempt in archive
            target_auth_member = None
            target_sdxl_member = None

            # 1. Try finding via manifest
            for summ in manifest_data.get("attempts_summary", []):
                if summ.get("source_image_id") == row_idx or summ.get("target_text") == att["target_text"]:
                    m_auth_sha = summ.get("file_hashes", {}).get("authentic_control_sha256")
                    if m_auth_sha == expected_auth_sha:
                        folder = summ["attempt_id"]
                        target_auth_member = f"{folder}/authentic_control.png"
                        target_sdxl_member = f"{folder}/final_edited_document.png"
                        break

            # 2. Fallback to direct path conventions
            if not target_auth_member:
                if row_idx == 22:
                    target_auth_member = "attempt_0_replacement_price/authentic_control.png"
                    target_sdxl_member = "attempt_0_replacement_price/final_edited_document.png"
                elif row_idx == 61:
                    target_auth_member = "attempt_1_replacement_item_name/authentic_control.png"
                    target_sdxl_member = "attempt_1_replacement_item_name/final_edited_document.png"

            if target_auth_member not in namelist:
                raise FileNotFoundError(f"Missing authentic member '{target_auth_member}' in archive for {att_id}")

            # Extract raw authentic bytes directly without re-encoding
            raw_auth_bytes = zf.read(target_auth_member)
            obs_auth_sha = compute_bytes_sha256(raw_auth_bytes)
            if obs_auth_sha != expected_auth_sha:
                raise ValueError(
                    f"Authentic image SHA-256 mismatch for {att_id}! Expected {expected_auth_sha}, got {obs_auth_sha}"
                )

            dest_att_dir = target_extract_dir / att_id
            dest_att_dir.mkdir(parents=True, exist_ok=True)
            dest_auth_path = dest_att_dir / "authentic_control.png"
            dest_auth_path.write_bytes(raw_auth_bytes)

            dest_sdxl_path = None
            if target_sdxl_member and target_sdxl_member in namelist:
                dest_sdxl_path = dest_att_dir / "sdxl_pilot_output.png"
                dest_sdxl_path.write_bytes(zf.read(target_sdxl_member))

            resolved_inputs[att_id] = {
                "authentic_path": dest_auth_path,
                "authentic_sha256": obs_auth_sha,
                "sdxl_path": dest_sdxl_path,
            }
            print(f"[PASS] Extracted and verified authentic input for {att_id}: {obs_auth_sha[:16]}...")

    return resolved_inputs


def resolve_inputs_from_directory(
    inputs_dir: Path,
    plan: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Resolves authentic inputs from an existing directory (fallback)."""
    resolved_inputs: Dict[str, Dict[str, Any]] = {}
    attempts = plan["attempts"]

    for att in attempts:
        att_id = att["attempt_id"]
        row_idx = att.get("source_row_index")
        expected_sha = att["authentic_sha256"]

        candidates = [
            inputs_dir / att_id / "authentic_control.png",
            inputs_dir / f"{att_id}.png",
            inputs_dir / ("attempt_0_replacement_price" if row_idx == 22 else "attempt_1_replacement_item_name") / "authentic_control.png",
        ]

        found_path: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                found_path = c
                break

        if not found_path:
            # Glob search
            for p in inputs_dir.glob("**/authentic_control.png"):
                if compute_sha256(p) == expected_sha:
                    found_path = p
                    break

        if not found_path:
            raise FileNotFoundError(f"Could not locate authentic control for {att_id} in {inputs_dir}")

        obs_sha = compute_sha256(found_path)
        if obs_sha != expected_sha:
            raise ValueError(f"Authentic SHA mismatch for {att_id}: expected {expected_sha}, got {obs_sha}")

        sdxl_candidates = [
            found_path.parent / "final_edited_document.png",
            found_path.parent / "sdxl_pilot_output.png",
        ]
        sdxl_path = next((p for p in sdxl_candidates if p.is_file()), None)

        resolved_inputs[att_id] = {
            "authentic_path": found_path,
            "authentic_sha256": obs_sha,
            "sdxl_path": sdxl_path,
        }
        print(f"[PASS] Verified input {att_id}: {obs_sha[:16]}...")

    return resolved_inputs


def download_and_verify_modelscope_anytext(
    plan: Dict[str, Any],
    cache_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Downloads AnyText weights via ModelScope snapshot_download ignoring translator.

    Verifies checkpoint SHA-256 and records actual downloaded bytes.
    """
    model_spec = plan["model_specification"]
    repo_id = model_spec["modelscope_repo_id"]
    revision = model_spec["model_revision"]
    checkpoint_file = model_spec["checkpoint_file"]
    expected_ckpt_sha = model_spec["checkpoint_sha256"]
    planned_bytes = model_spec["total_required_weights_bytes"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir.resolve())

    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[INFO] Initiating ModelScope snapshot download for {repo_id} ({revision})...")
    print("[INFO] Excluding 'nlp_csanmt_translation_zh2en/*' to preserve budget...")

    t0 = time.time()
    model_dir = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=str(cache_dir.resolve()),
        ignore_file_pattern=["nlp_csanmt_translation_zh2en/*"],
    )
    download_duration = time.time() - t0

    model_dir_path = Path(model_dir)

    # Check translator directory was excluded
    translator_dir = model_dir_path / "nlp_csanmt_translation_zh2en"
    has_translator = translator_dir.is_dir() and any(translator_dir.iterdir())
    if has_translator:
        raise RuntimeError("Translator directory was unexpectedly downloaded! Exceeds budget constraints.")

    # Calculate actual downloaded bytes in model_dir
    actual_downloaded_bytes = sum(f.stat().st_size for f in model_dir_path.rglob("*") if f.is_file())
    actual_downloaded_gib = round(actual_downloaded_bytes / (1024**3), 4)

    # Checkpoint hash verification
    ckpt_path = model_dir_path / checkpoint_file
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    observed_ckpt_sha = compute_sha256(ckpt_path)
    if observed_ckpt_sha != expected_ckpt_sha:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch! Expected {expected_ckpt_sha}, got {observed_ckpt_sha}"
        )

    receipt = {
        "model_dir": str(model_dir_path),
        "repo_id": repo_id,
        "revision": revision,
        "download_duration_sec": round(download_duration, 2),
        "planned_weights_bytes": planned_bytes,
        "actual_downloaded_bytes": actual_downloaded_bytes,
        "actual_downloaded_gib": actual_downloaded_gib,
        "translator_excluded": True,
        "checkpoint_sha256": observed_ckpt_sha,
        "checkpoint_status": "VERIFIED_MATCH",
    }
    print(f"[PASS] Model downloaded ({actual_downloaded_gib} GiB) and verified: {observed_ckpt_sha[:16]}...")
    return model_dir, receipt


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
    """Invokes AnyText pipeline in text-editing mode with CUDA synchronization."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_sec = time.time() - t0

    if rtn_code < 0 or not results:
        raise RuntimeError(f"AnyText inference failed with code {rtn_code}: {rtn_warning}")

    raw_output_np = results[0]
    raw_output_img = Image.fromarray(raw_output_np.astype(np.uint8), mode="RGB")

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 2)
        if torch.cuda.is_available()
        else "NOT_MEASURED"
    )

    meta = {
        "latency_sec": round(latency_sec, 2),
        "rtn_code": rtn_code,
        "rtn_warning": rtn_warning,
        "peak_vram_mb": peak_vram_mb,
    }
    return raw_output_img, latency_sec, meta


def crop_zoomed_roi(
    image: Image.Image,
    target_bbox: List[int],
    margin_ratio_y: float = 0.8,
    margin_ratio_x: float = 0.4,
    min_pad_y: int = 30,
    min_pad_x: int = 60,
) -> Image.Image:
    """Crops a zoomed region of interest around target_bbox [ymin, xmin, ymax, xmax]."""
    doc_w, doc_h = image.size
    ymin, xmin, ymax, xmax = target_bbox

    h_box = ymax - ymin
    w_box = xmax - xmin

    pad_y = max(min_pad_y, int(h_box * margin_ratio_y))
    pad_x = max(min_pad_x, int(w_box * margin_ratio_x))

    crop_ymin = max(0, ymin - pad_y)
    crop_ymax = min(doc_h, ymax + pad_y)
    crop_xmin = max(0, xmin - pad_x)
    crop_xmax = min(doc_w, xmax + pad_x)

    return image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))


def generate_anytext_contact_sheet(
    attempts_data: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates an enhanced comparison contact sheet with zoomed ROIs."""
    cell_w, cell_h = 360, 360
    header_h = 75
    label_h = 35
    cols = 6  # 1. Auth ROI, 2. SDXL ROI, 3. AnyText Raw ROI, 4. AnyText Final ROI, 5. AnyText Final Full, 6. Mask ROI
    rows = len(attempts_data)
    margin = 15

    total_w = cols * cell_w + (cols + 1) * margin
    total_h = header_h + rows * (cell_h + label_h) + (rows + 1) * margin

    sheet = Image.new("RGB", (total_w, total_h), (250, 250, 252))
    draw = ImageDraw.Draw(sheet)

    # Header
    draw.text(
        (margin, 12),
        "AnyText Document Feasibility Study: Zoomed ROI Comparison (Attempt A & B)",
        fill=(15, 23, 42),
    )
    draw.text(
        (margin, 38),
        "Descriptive comparison between SDXL (1024x1024) and AnyText (512x512 canvas). Not an isolated architecture study.",
        fill=(71, 85, 105),
    )

    col_titles = [
        "1. Authentic ROI (Zoomed)",
        "2. SDXL Pilot ROI (Zoomed)",
        "3. AnyText Raw Canvas ROI",
        "4. AnyText Final ROI (Zoomed)",
        "5. AnyText Final Full Document",
        "6. Intervention Mask (Zoomed)",
    ]
    for c_idx, title in enumerate(col_titles):
        x = margin + c_idx * (cell_w + margin)
        draw.text((x, 56), title, fill=(30, 41, 59))

    for r_idx, att in enumerate(attempts_data):
        y_top = header_h + margin + r_idx * (cell_h + label_h + margin)
        draw.text(
            (margin, y_top),
            f"Attempt {att['id']} | Target: '{att['target_text']}' | Seed: {att['seed']} | Latency: {att.get('latency', 'N/A')}s",
            fill=(2, 132, 199),
        )

        cells = [
            att.get("authentic_roi"),
            att.get("sdxl_roi"),
            att.get("anytext_raw_roi"),
            att.get("anytext_final_roi"),
            att.get("anytext_final_full"),
            att.get("mask_roi"),
        ]

        for c_idx, img in enumerate(cells):
            x = margin + c_idx * (cell_w + margin)
            y = y_top + label_h
            if img:
                thumb = img.copy()
                thumb.thumbnail((cell_w, cell_h), Image.Resampling.BICUBIC)
                offset_x = x + (cell_w - thumb.width) // 2
                offset_y = y + (cell_h - thumb.height) // 2
                sheet.paste(thumb, (offset_x, offset_y))
                draw.rectangle([offset_x, offset_y, offset_x + thumb.width, offset_y + thumb.height], outline=(203, 213, 225), width=1)
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(241, 245, 249))
                draw.text((x + 20, y + cell_h // 2 - 10), "N/A (Not Provided)", fill=(148, 163, 184))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")
    print(f"[OK] Enhanced contact sheet written to: {output_path}")


def package_results_zip(
    run_dir: Path,
    output_zip_path: Path,
    max_size_bytes: int = 100 * 1024 * 1024,
) -> Path:
    """Packages run artifacts into zip enforcing strict whitelist and <= 100 MiB limit."""
    allowed_extensions = {".png", ".json", ".csv"}
    disallowed_keywords = {".ckpt", ".pt", ".bin", ".cache", ".py", ".pth", ".safetensors"}

    if output_zip_path.exists():
        output_zip_path.unlink()

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_p, _, files in os.walk(run_dir):
            for file in files:
                p = Path(root_p) / file
                ext = p.suffix.lower()

                # Filter by whitelist
                if ext not in allowed_extensions:
                    continue
                if any(kw in file.lower() for kw in disallowed_keywords):
                    continue

                rel_name = p.relative_to(run_dir)
                zf.write(p, arcname=str(Path(run_dir.name) / rel_name))

    zip_size = output_zip_path.stat().st_size
    if zip_size > max_size_bytes:
        output_zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"Result archive exceeded 100 MiB limit! ({zip_size / (1024**2):.2f} MiB). Archive deleted."
        )

    print(f"[OK] Results safely archived to: {output_zip_path}")
    print(f"     Archive Size: {zip_size / (1024**2):.2f} MiB (Ceiling: 100 MiB)")
    print(f"     SHA-256: {compute_sha256(output_zip_path)}")
    return output_zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AnyText Feasibility Runner")
    parser.add_argument("--plan-path", type=str, required=True, help="Path to anytext_feasibility_plan.json")
    parser.add_argument("--inputs-zip", type=str, default=None, help="Path to document_feasibility_results.zip")
    parser.add_argument("--inputs-dir", type=str, default=None, help="Path to inputs directory")
    parser.add_argument("--output-root", type=str, required=True, help="Path to outputs root directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to ModelScope cache directory")
    parser.add_argument("--custom-font", type=str, default=None, help="Custom TTF font path")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only")
    args = parser.parse_args()

    plan_path = Path(args.plan_path).resolve()
    output_root = Path(args.output_root).resolve()

    if not plan_path.is_file():
        print(f"[FAIL] Plan not found: {plan_path}")
        return 1

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_bytes_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_anytext_{run_timestamp}_{plan_bytes_sha[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cache directory isolation: strictly outside run_dir and output_root
    if args.cache_dir:
        model_cache_dir = Path(args.cache_dir).resolve()
    else:
        # Default Colab location /content/modelscope_cache, or sibling on local
        model_cache_dir = Path("/content/modelscope_cache") if Path("/content").is_dir() else output_root.parent / ".cache_modelscope"

    print(f"=== ANYTEXT FEASIBILITY RUNNER: {run_id} ===")
    print(f"Plan Path: {plan_path} (SHA-256: {plan_bytes_sha[:16]}...)")
    print(f"Model Cache Dir: {model_cache_dir} (ISOLATED)")

    hw_info = check_gpu_and_resources()
    print(f"CUDA Available: {hw_info['cuda_available']} ({hw_info['device_name']}, VRAM: {hw_info['vram_total_gb']} GB)")
    print(f"Disk Free: {hw_info['disk_free_gb']} GB")

    font_path, font_sha256 = resolve_font_path(args.custom_font)
    print(f"Resolved Font: {font_path} (SHA-256: {font_sha256[:16]}...)")

    # Resolve inputs: prefer inputs-zip if provided, else inputs-dir
    print("\n--- Resolving Authentic Inputs ---")
    extracted_inputs: Dict[str, Dict[str, Any]] = {}
    if args.inputs_zip and Path(args.inputs_zip).is_file():
        staging_dir = run_dir / ".inputs_staging"
        extracted_inputs = safe_extract_archive_inputs(
            archive_path=Path(args.inputs_zip).resolve(),
            target_extract_dir=staging_dir,
            plan=plan,
        )
    elif args.inputs_dir and Path(args.inputs_dir).is_dir():
        extracted_inputs = resolve_inputs_from_directory(
            inputs_dir=Path(args.inputs_dir).resolve(),
            plan=plan,
        )
    else:
        # Check default archive path from plan
        default_archive = Path(plan.get("input_source_specification", {}).get("archive_path", ""))
        if not default_archive.is_file():
            default_archive = Path(__file__).resolve().parent.parent.parent / default_archive
        if default_archive.is_file():
            staging_dir = run_dir / ".inputs_staging"
            extracted_inputs = safe_extract_archive_inputs(
                archive_path=default_archive,
                target_extract_dir=staging_dir,
                plan=plan,
            )
        else:
            print("[FAIL] Neither valid --inputs-zip nor --inputs-dir provided, and default archive not found.")
            return 1

    preflight_receipt = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "plan_sha256": plan_bytes_sha,
        "hardware": hw_info,
        "font": {"path": font_path, "sha256": font_sha256},
        "inputs_verified": {k: v["authentic_sha256"] for k, v in extracted_inputs.items()},
        "cache_dir": str(model_cache_dir),
        "status": "PREFLIGHT_PASS",
    }
    (run_dir / "preflight_receipt.json").write_text(json.dumps(preflight_receipt, indent=2), encoding="utf-8")

    if args.preflight_only:
        print("[OK] Preflight completed successfully.")
        return 0

    if not hw_info["cuda_available"]:
        print("[FAIL] CUDA GPU is required for AnyText generation. Running on CPU is not supported.")
        return 1

    # Model download & verification
    print("\n--- Downloading AnyText Model (Excluding Translator) ---")
    model_dir, model_receipt = download_and_verify_modelscope_anytext(
        plan=plan,
        cache_dir=model_cache_dir,
    )
    (run_dir / "model_download_receipt.json").write_text(json.dumps(model_receipt, indent=2), encoding="utf-8")

    # Initialize AnyText Pipeline
    print("\n--- Initializing AnyText Pipeline from Local Weights ---")
    from modelscope.pipelines import pipeline

    pipe = pipeline(
        "my-anytext-task",
        model=model_dir,
        use_fp16=True,
        use_translator=False,
        font_path=font_path,
    )
    print("[OK] AnyText pipeline initialized successfully.")

    # Import geometry adapter
    from ml.evaluation.anytext_document_adapter import (
        prepare_anytext_editing_inputs,
        composite_anytext_editing_output,
    )

    contact_sheet_items: List[Dict[str, Any]] = []
    review_records: List[Dict[str, Any]] = []
    attempts = plan["attempts"]

    has_failure = False

    for att in attempts:
        att_id = att["attempt_id"]
        att_dir = run_dir / att_id
        att_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Executing {att_id} ---")

        auth_info = extracted_inputs.get(att_id)
        if not auth_info:
            print(f"[FAIL] Missing authentic input data for {att_id}")
            has_failure = True
            continue

        auth_img = Image.open(auth_info["authentic_path"]).convert("RGB")
        sdxl_img = Image.open(auth_info["sdxl_path"]).convert("RGB") if auth_info.get("sdxl_path") and auth_info["sdxl_path"].is_file() else None

        try:
            # 1. Geometry preparation
            prep_res = prepare_anytext_editing_inputs(
                image=auth_img,
                target_bbox=att["target_bbox"],
                canvas_size=att["sampling_parameters"].get("canvas_width", 512),
                context_margin_ratio=att["sampling_parameters"].get("context_margin_ratio", 0.5),
            )

            # 2. Run AnyText inference
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

            # 3. Composite output strictly inside target mask
            comp_res = composite_anytext_editing_output(
                original_image=auth_img,
                raw_anytext_output=raw_canvas,
                geometry_receipt=prep_res["geometry_receipt"],
                native_mask=prep_res["native_mask"],
            )

            final_img = comp_res["final_edited_document"]
            is_outside_equal = comp_res["is_outside_equal"]
            max_outside_diff = comp_res["max_outside_diff"]

            if not is_outside_equal:
                raise ValueError(f"Outside-mask equality contract violated! Max outside diff: {max_outside_diff}")
            print(f"[PASS] Outside-mask equality verified (diff = 0).")

            # 4. Save artifacts
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

            # Compute ZOOMED ROIs for contact sheet
            target_bbox = att["target_bbox"]
            auth_roi = crop_zoomed_roi(auth_img, target_bbox)
            sdxl_roi = crop_zoomed_roi(sdxl_img, target_bbox) if sdxl_img else None
            anytext_final_roi = crop_zoomed_roi(final_img, target_bbox)
            mask_roi = crop_zoomed_roi(prep_res["native_mask"].convert("RGB"), target_bbox)

            # Crop ROI from raw canvas around target box in canvas
            t_box_canvas = prep_res["geometry_receipt"]["target_bbox_in_canvas"]
            raw_roi = crop_zoomed_roi(raw_canvas, t_box_canvas)

            # Save ROI images
            auth_roi.save(att_dir / "authentic_roi_zoomed.png", "PNG")
            anytext_final_roi.save(att_dir / "anytext_final_roi_zoomed.png", "PNG")
            raw_roi.save(att_dir / "anytext_raw_roi_zoomed.png", "PNG")
            mask_roi.save(att_dir / "mask_roi_zoomed.png", "PNG")
            if sdxl_roi:
                sdxl_roi.save(att_dir / "sdxl_roi_zoomed.png", "PNG")

            att_receipt = {
                "attempt_id": att_id,
                "status": "TECHNICAL_QC_PASS",
                "operation": att["operation"],
                "target_text": att["target_text"],
                "seed": att["seed"],
                "latency_sec": latency,
                "peak_vram_mb": inf_meta["peak_vram_mb"],
                "outside_mask_bitwise_equal": is_outside_equal,
                "max_outside_diff": max_outside_diff,
                "hashes": {
                    "authentic_sha256": compute_sha256(auth_save),
                    "raw_canvas_sha256": compute_sha256(raw_save),
                    "final_edited_sha256": compute_sha256(final_save),
                    "native_mask_sha256": compute_sha256(mask_save),
                },
                "geometry_receipt": prep_res["geometry_receipt"],
            }
            (att_dir / "attempt_receipt.json").write_text(json.dumps(att_receipt, indent=2), encoding="utf-8")

            contact_sheet_items.append({
                "id": att_id,
                "target_text": att["target_text"],
                "seed": att["seed"],
                "latency": latency,
                "authentic_roi": auth_roi,
                "sdxl_roi": sdxl_roi,
                "anytext_raw_roi": raw_roi,
                "anytext_final_roi": anytext_final_roi,
                "anytext_final_full": final_img,
                "mask_roi": mask_roi,
            })

            review_records.append({
                "attempt_id": att_id,
                "source_row_index": att["source_row_index"],
                "original_text": att["original_text"],
                "target_text": att["target_text"],
                "exact_text_success": "PENDING_HUMAN_REVIEW",
                "visual_realism": "PENDING_HUMAN_REVIEW",
                "human_verdict": "PENDING_HUMAN_REVIEW",
                "notes": "Awaiting human reviewer assessment.",
            })

        except Exception as e:
            has_failure = True
            err_trace = traceback.format_exc()
            print(f"[FAIL] Error in {att_id}: {e}")
            fail_receipt = {
                "attempt_id": att_id,
                "status": "FAILED",
                "error": str(e),
                "traceback": err_trace,
            }
            (att_dir / "failure_receipt.json").write_text(json.dumps(fail_receipt, indent=2), encoding="utf-8")

    # Generate contact sheet if any item succeeded
    if contact_sheet_items:
        cs_path = run_dir / "anytext_contact_sheet.png"
        generate_anytext_contact_sheet(contact_sheet_items, cs_path)

    # Write human review CSV
    if review_records:
        csv_path = run_dir / "human_review.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "attempt_id",
                "source_row_index",
                "original_text",
                "target_text",
                "exact_text_success",
                "visual_realism",
                "human_verdict",
                "notes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_records)

    # Package results into zip archive
    zip_path = output_root / "anytext_feasibility_results.zip"
    package_results_zip(run_dir=run_dir, output_zip_path=zip_path)

    if has_failure:
        print("[WARNING] One or more attempts encountered errors. Review failure receipts.")
        return 1

    print("\n[DONE] AnyText Feasibility Run executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
