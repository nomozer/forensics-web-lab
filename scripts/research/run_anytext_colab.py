#!/usr/bin/env python3
"""AnyText Document Generation Feasibility Colab Runner.

Runs AnyText text-editing on two locked replacement sources (Attempt A & B)
on Google Colab GPU without Hugging Face token or paid dependencies.

Inputs are extracted directly from the verified SDXL pilot archive:
`data/research/local-artifacts/document-feasibility/document_feasibility_results.zip`
"""

from __future__ import annotations

import argparse
import atexit
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_SHA256_MEMO: Dict[Tuple[str, int, int], str] = {}


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file with stat-validated memoization for large files."""
    resolved = path.resolve()
    cache_key: Optional[Tuple[str, int, int]] = None
    try:
        st = resolved.stat()
        cache_key = (str(resolved), st.st_size, st.st_mtime_ns)
        if st.st_size >= 10 * 1024 * 1024 and cache_key in _SHA256_MEMO:
            return _SHA256_MEMO[cache_key]
    except OSError:
        cache_key = None

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    digest = h.hexdigest()
    if cache_key is not None and cache_key[1] >= 10 * 1024 * 1024:
        _SHA256_MEMO[cache_key] = digest
    return digest


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


def acquire_runner_lock(lock_file: Path) -> None:
    """Acquires a single-instance lock file with PID checking to prevent concurrent execution."""
    if lock_file.is_file():
        try:
            lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
            pid = lock_data.get("pid")
            is_alive = False
            if pid and isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    is_alive = True
                except (OSError, ProcessLookupError, PermissionError):
                    is_alive = False
            if is_alive:
                raise RuntimeError(
                    f"Another AnyText runner process (PID {pid}) is currently running! "
                    f"Concurrent executions are prohibited to preserve experimental integrity."
                )
        except RuntimeError:
            raise
        except Exception:
            pass
    lock_file.write_text(
        json.dumps({"pid": os.getpid(), "timestamp": datetime.datetime.now().isoformat()}),
        encoding="utf-8",
    )


def release_runner_lock(lock_file: Path) -> None:
    """Releases the lock file safely."""
    try:
        if lock_file.is_file():
            lock_file.unlink(missing_ok=True)
    except Exception:
        pass


# Mandatory CLIP files required for AnyText inference (clip-vit-large-patch14).
# CLIPTokenizer in transformers==4.30.2 requires both vocab.json and merges.txt.
# A non-empty CLIP directory or mere file existence is NOT proof that weights are complete.
CLIP_MANDATORY_FILES = [
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
]
# At least one of these weight files must exist:
CLIP_WEIGHT_FILE_CANDIDATES = [
    "pytorch_model.bin",
    "model.safetensors",
]


def verify_clip_weights_complete(
    clip_dir: Path,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, List[str]]:
    """Verifies that all mandatory CLIP inference files are present and valid.

    Never treats file existence alone as proof of integrity: rejects 0-byte files
    and validates size + SHA-256 against metadata_map when provided.

    Returns:
        Tuple of (is_complete, list_of_missing_or_corrupt_files).
    """
    if not clip_dir.is_dir():
        return False, ["directory_not_found"]

    missing: List[str] = []
    for f in CLIP_MANDATORY_FILES:
        fp = clip_dir / f
        rel_key = f"clip-vit-large-patch14/{f}"
        if not fp.is_file():
            missing.append(f)
        elif fp.stat().st_size == 0:
            missing.append(f"{f}(empty)")
        elif metadata_map and rel_key in metadata_map:
            exp_meta = metadata_map[rel_key]
            if fp.stat().st_size != exp_meta["size"]:
                missing.append(f"{f}(size_mismatch)")
            elif compute_sha256(fp) != exp_meta["sha256"]:
                missing.append(f"{f}(sha256_mismatch)")

    has_weights = False
    for w in CLIP_WEIGHT_FILE_CANDIDATES:
        wp = clip_dir / w
        rel_key = f"clip-vit-large-patch14/{w}"
        if wp.is_file() and wp.stat().st_size > 0:
            if metadata_map and rel_key in metadata_map:
                exp_meta = metadata_map[rel_key]
                if wp.stat().st_size == exp_meta["size"] and compute_sha256(wp) == exp_meta["sha256"]:
                    has_weights = True
                    break
                else:
                    missing.append(f"{w}(integrity_mismatch)")
            else:
                has_weights = True
                break

    if not has_weights:
        missing.append(f"one_of({', '.join(CLIP_WEIGHT_FILE_CANDIDATES)})")

    return len(missing) == 0, missing


PRECISION_POLICY_ID = "fp16-weights-xformers-0.0.20-no-autocast-v1"


def assert_fp16_attention_path() -> Dict[str, Any]:
    """Fails closed unless AnyText's xformers MemoryEfficientCrossAttention is active.

    Without xformers, upstream CrossAttention upcasts q/k to fp32 (ATTN_PRECISION defaults to "fp32")
    while v stays fp16 under model.half(), raising "expected scalar type Half but found Float".
    We do not silently fall back to an environment override: that would change the precision policy.
    """
    import torch
    import ldm.modules.attention as attn

    info: Dict[str, Any] = {
        "policy_id": PRECISION_POLICY_ID,
        "torch": torch.__version__,
        "xformers_available": bool(attn.XFORMERS_IS_AVAILBLE),
        "attn_precision_env": os.environ.get("ATTN_PRECISION"),
        "attn_precision_effective": getattr(attn, "_ATTN_PRECISION", None),
        "autocast": "not used",
    }
    try:
        import xformers

        info["xformers"] = xformers.__version__
    except ImportError:
        info["xformers"] = None
    if not info["xformers_available"]:
        raise RuntimeError(
            "AnyText xformers attention is not active (ldm.modules.attention.XFORMERS_IS_AVAILBLE is False). "
            "The fp16 path requires xformers==0.0.20 from scripts/research/anytext_requirements.txt; "
            "refusing to run with the fp32-upcast CrossAttention that fails with 'expected scalar type Half but found Float'."
        )
    return info


def describe_failure_site(exc: BaseException, max_tensors: int = 24) -> Dict[str, Any]:
    """Describes the failing frames: location, tensor dtype/device/shape and module parameter dtypes."""
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    site: Dict[str, Any] = {"exception_type": type(exc).__name__, "frames": []}
    try:
        import torch
    except ImportError:
        return site

    def _frame_info(frame: Any, lineno: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "file": frame.f_code.co_filename,
            "line": lineno,
            "function": frame.f_code.co_name,
            "tensors": {},
            "module_parameter_dtypes": {},
        }
        for name, val in list(frame.f_locals.items()):
            if len(info["tensors"]) >= max_tensors:
                break
            if isinstance(val, torch.Tensor):
                info["tensors"][name] = {"dtype": str(val.dtype), "device": str(val.device), "shape": list(val.shape)}
            elif isinstance(val, (tuple, list)) and val and all(isinstance(v, torch.Tensor) for v in val):
                info["tensors"][name] = [{"dtype": str(v.dtype), "device": str(v.device), "shape": list(v.shape)} for v in val]
            elif isinstance(val, torch.nn.Module):
                counts: Dict[str, int] = {}
                devices = set()
                for p in val.parameters():
                    counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + 1
                    devices.add(str(p.device))
                info["module_parameter_dtypes"][f"{name}:{type(val).__name__}"] = {"dtypes": counts, "devices": sorted(devices)}
        return info

    try:
        if frames:
            innermost = _frame_info(*frames[-1])
            site["innermost_frame"] = innermost
            # first non-torch, non-stdlib frame from the end = the module/operation that called into torch
            for frame, lineno in reversed(frames):
                fname = frame.f_code.co_filename.replace("\\", "/")
                if "/site-packages/torch/" not in fname and "/lib/python" not in fname:
                    site["innermost_project_frame"] = _frame_info(frame, lineno)
                    break
    except Exception as diag_err:  # diagnostics must never mask the original failure
        site["diagnostic_error"] = repr(diag_err)
    return site


def build_failure_receipt(
    stage: str,
    exc: BaseException,
    attempt_id: Optional[str] = None,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured failure receipt keeping stage, attempt id, original traceback and dtype/device evidence."""
    receipt: Dict[str, Any] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "status": "FAILED",
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "failure_site": describe_failure_site(exc),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if extra:
        receipt.update(extra)
    return receipt


def compute_attempt_fingerprint(plan: Dict[str, Any], att: Dict[str, Any], authentic_sha256: str) -> str:
    """Binds a result to model revision, source bytes, text, seed, sampling and precision policy."""
    spec = plan.get("model_specification", {})
    payload = {
        "code_commit_sha": spec.get("code_commit_sha"),
        "model_revision": spec.get("model_revision"),
        "checkpoint_sha256": spec.get("checkpoint_sha256"),
        "geometry_contract": plan.get("geometry_contract"),
        "attempt": {
            k: att.get(k)
            for k in (
                "attempt_id", "source_row_index", "original_text", "target_text", "target_bbox",
                "seed", "prompt", "a_prompt", "n_prompt", "sampling_parameters",
            )
        },
        "authentic_sha256": authentic_sha256,
        "precision_policy_id": PRECISION_POLICY_ID,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def find_prior_completed_attempt(
    output_root: Path,
    attempt_id: str,
    expected_fingerprint: str,
    exclude_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Finds a prior TECHNICAL_QC_PASS attempt whose fingerprint and artifact hashes are verified."""
    if not output_root.is_dir():
        return None
    for receipt_path in output_root.glob(f"*/{attempt_id}/attempt_receipt.json"):
        try:
            att_folder = receipt_path.parent
            if exclude_dir and (
                att_folder.resolve() == exclude_dir.resolve()
                or exclude_dir.resolve() in att_folder.resolve().parents
            ):
                continue
            rcpt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if rcpt.get("status") != "TECHNICAL_QC_PASS" or rcpt.get("outside_mask_bitwise_equal") is not True:
                continue
            if rcpt.get("config_fingerprint") != expected_fingerprint:
                continue
            raw_path = att_folder / "anytext_raw_canvas.png"
            final_path = att_folder / "anytext_final_edited.png"
            hashes = rcpt.get("hashes", {})
            if not raw_path.is_file() or not final_path.is_file():
                continue
            if compute_sha256(raw_path) != hashes.get("raw_canvas_sha256"):
                continue
            if compute_sha256(final_path) != hashes.get("final_edited_sha256"):
                continue
            return att_folder
        except Exception:
            continue
    return None


# Official ModelScope repository file inventory for damo/cv_anytext_text_generation_editing at revision v1.1.3
# Total kept files: 11 blobs, 7,444,695,175 bytes (~6.9334 GiB <= 12.0 GiB budget ceiling).
OFFICIAL_ANYTEXT_FILES_METADATA: Dict[str, Dict[str, Any]] = {
    "anytext_v1.1.ckpt": {
        "size": 5730281759,
        "sha256": "09c88d145aa08f067641dc1d10c8890855939fa756d4cdc991e781a5f3918f59",
    },
    "configuration.json": {
        "size": 140,
        "sha256": "b72f68146709c90b52bd9728b356760b7069ec1276149f223bfc7445576ea1e5",
    },
    "ms_wrapper.py": {
        "size": 25783,
        "sha256": "ed6a60595e8cf263b1fd68bc4e7b41064e05855dda9a53de18e6b04e51eb7899",
    },
    "clip-vit-large-patch14/config.json": {
        "size": 4519,
        "sha256": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
    },
    "clip-vit-large-patch14/merges.txt": {
        "size": 524619,
        "sha256": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    },
    "clip-vit-large-patch14/preprocessor_config.json": {
        "size": 316,
        "sha256": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    },
    "clip-vit-large-patch14/pytorch_model.bin": {
        "size": 1710671599,
        "sha256": "f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3",
    },
    "clip-vit-large-patch14/special_tokens_map.json": {
        "size": 389,
        "sha256": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    },
    "clip-vit-large-patch14/tokenizer.json": {
        "size": 2224003,
        "sha256": "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4",
    },
    "clip-vit-large-patch14/tokenizer_config.json": {
        "size": 905,
        "sha256": "deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129",
    },
    "clip-vit-large-patch14/vocab.json": {
        "size": 961143,
        "sha256": "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69",
    },
}

OFFICIAL_EXCLUDE_PATH_PATTERNS: List[str] = [
    r"^nlp_csanmt_translation_zh2en/.*",
    r"^description/.*",
    r"^README(\..*)?$",
    r".*\.md$",
    r"^\..*",
    r".*/\..*",
]

SDK_PATCH_VERSION = "modelscope-1.10.0-http-get-and-cache-patch-v1.0.0"


def filter_repo_files(
    files: List[Dict[str, Any]],
    exclude_patterns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filters ModelScope repository files by matching regex patterns against full Path and Name.

    ModelScope 1.10.0 snapshot_download only matches ignore_file_pattern against model_file['Name']
    (basename), which fails to exclude subfolder paths like 'nlp_csanmt_translation_zh2en/config.json'
    unless filtered by full 'Path'. This function checks both full 'Path' and 'Name', preserving all
    required inference files (including clip-vit-large-patch14/merges.txt) while excluding the
    translator subfolder, documentation, and hidden files.
    """
    patterns = exclude_patterns if exclude_patterns is not None else OFFICIAL_EXCLUDE_PATH_PATTERNS
    compiled_patterns = [re.compile(p) for p in patterns]

    kept: List[Dict[str, Any]] = []
    for f in files:
        if f.get("Type") == "tree":
            continue
        path = f.get("Path", f.get("Name", ""))
        name = f.get("Name", "")
        matched = False
        for pat in compiled_patterns:
            if pat.search(path) is not None or pat.search(name) is not None:
                matched = True
                break
        if not matched:
            kept.append(f)
    return kept


def inspect_catalog_and_cache_integrity(
    model_dir: Path,
    metadata_map: Dict[str, Dict[str, Any]],
    download_ceiling_bytes: int,
) -> Dict[str, Any]:
    """Builds pre-download inventory from official metadata and verifies on-disk cache integrity.

    Never treats file existence as proof of integrity: every cached file in the catalog
    must match both its official metadata Size and SHA-256. Invalid/corrupted files are
    flagged for re-download. Also verifies catalog total and required download delta
    against the locked download budget ceiling.
    """
    if not metadata_map:
        raise ValueError("metadata_map cannot be empty; fail-closed.")

    catalog_total_bytes = 0
    valid_cached_files: List[str] = []
    missing_files: List[str] = []
    corrupted_files: List[Dict[str, Any]] = []
    files_to_download: Dict[str, Dict[str, Any]] = {}
    valid_cached_bytes = 0
    bytes_to_download = 0

    for rel_path, meta in metadata_map.items():
        exp_size = meta.get("size")
        exp_sha = meta.get("sha256")
        if not isinstance(exp_size, int) or exp_size <= 0 or not exp_sha or not isinstance(exp_sha, str):
            raise ValueError(f"Invalid metadata entry for '{rel_path}': {meta}")

        catalog_total_bytes += exp_size
        dest_path = model_dir / rel_path
        if not dest_path.is_file():
            missing_files.append(rel_path)
            files_to_download[rel_path] = meta
            bytes_to_download += exp_size
            continue

        try:
            obs_size = dest_path.stat().st_size
            if obs_size != exp_size:
                corrupted_files.append({
                    "rel_path": rel_path,
                    "reason": "size_mismatch",
                    "expected_size": exp_size,
                    "observed_size": obs_size,
                })
                files_to_download[rel_path] = meta
                bytes_to_download += exp_size
                continue

            obs_sha = compute_sha256(dest_path)
            if obs_sha != exp_sha:
                corrupted_files.append({
                    "rel_path": rel_path,
                    "reason": "sha256_mismatch",
                    "expected_sha256": exp_sha,
                    "observed_sha256": obs_sha,
                })
                files_to_download[rel_path] = meta
                bytes_to_download += exp_size
                continue

            valid_cached_files.append(rel_path)
            valid_cached_bytes += exp_size
        except OSError as exc:
            corrupted_files.append({
                "rel_path": rel_path,
                "reason": f"os_error: {exc}",
            })
            files_to_download[rel_path] = meta
            bytes_to_download += exp_size

    within_budget = (
        catalog_total_bytes <= download_ceiling_bytes
        and bytes_to_download <= download_ceiling_bytes
    )

    return {
        "catalog_file_count": len(metadata_map),
        "catalog_total_bytes": catalog_total_bytes,
        "catalog_total_gib": round(catalog_total_bytes / (1024**3), 4),
        "valid_cached_files": valid_cached_files,
        "valid_cached_bytes": valid_cached_bytes,
        "valid_cached_gib": round(valid_cached_bytes / (1024**3), 4),
        "missing_files": missing_files,
        "corrupted_files": corrupted_files,
        "files_to_download": files_to_download,
        "bytes_to_download": bytes_to_download,
        "gib_to_download": round(bytes_to_download / (1024**3), 4),
        "download_ceiling_bytes": download_ceiling_bytes,
        "within_budget": within_budget,
        "all_valid": len(files_to_download) == 0,
    }


def robust_download_file(
    repo_id: str,
    file_rel_path: str,
    revision: str,
    destination: Path,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    max_retries: int = 3,
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
    custom_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Downloads a file from ModelScope API with strict size and hash validation.

    Tolerates missing Content-Length header IF downloaded length and SHA-256
    match official metadata. Fails closed on truncated file, hash mismatch,
    Content-Length header mismatch, or missing metadata.
    """
    meta_dict = metadata_map if metadata_map is not None else OFFICIAL_ANYTEXT_FILES_METADATA
    if file_rel_path in meta_dict:
        if expected_size is None:
            expected_size = meta_dict[file_rel_path].get("size")
        if expected_sha256 is None:
            expected_sha256 = meta_dict[file_rel_path].get("sha256")

    if expected_size is None or expected_sha256 is None:
        raise ValueError(
            f"Missing official metadata (both size and sha256 required) for file '{file_rel_path}'. "
            f"Failing closed to prevent unverified downloads."
        )

    # 1. Cache integrity check (never trust file existence alone)
    if destination.is_file():
        try:
            cur_size = destination.stat().st_size
            if cur_size == expected_size:
                cur_sha = compute_sha256(destination)
                if cur_sha == expected_sha256:
                    return {
                        "path": str(destination),
                        "rel_path": file_rel_path,
                        "status": "reused",
                        "bytes": expected_size,
                        "sha256": expected_sha256,
                    }
            # Existing file is corrupted/incomplete; remove before downloading
            destination.unlink(missing_ok=True)
        except OSError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    encoded_path = urllib.parse.quote(file_rel_path, safe="")
    url = f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo?Revision={revision}&FilePath={encoded_path}"

    last_err: Optional[Exception] = None
    for attempt_idx in range(1, max_retries + 1):
        try:
            part_path.unlink(missing_ok=True)
            hasher = hashlib.sha256()
            downloaded_bytes = 0

            # Prefer requests if available, fallback to urllib.request
            use_requests = "requests" in sys.modules
            if not use_requests:
                try:
                    import requests  # noqa: F401
                    use_requests = True
                except ImportError:
                    use_requests = False

            req_headers = {"User-Agent": "ModelScope/1.10.0"}
            if custom_headers:
                req_headers.update(custom_headers)

            if use_requests:
                import requests
                with requests.get(url, stream=True, headers=req_headers, timeout=timeout) as resp:
                    resp.raise_for_status()
                    content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                    header_len_str = resp.headers.get("Content-Length")
                    header_len = int(header_len_str) if header_len_str is not None else None
                    if header_len is not None and content_encoding not in ("gzip", "deflate", "br") and header_len != expected_size:
                        raise ValueError(
                            f"Content-Length header mismatch for '{file_rel_path}': "
                            f"header specifies {header_len} bytes, metadata requires {expected_size} bytes."
                        )
                    with part_path.open("wb") as pf:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                pf.write(chunk)
                                hasher.update(chunk)
                                downloaded_bytes += len(chunk)
            else:
                import zlib
                req = urllib.request.Request(url, headers=req_headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status} downloading '{file_rel_path}'")
                    content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                    header_len_str = resp.headers.get("Content-Length")
                    header_len = int(header_len_str) if header_len_str is not None else None
                    if header_len is not None and content_encoding not in ("gzip", "deflate", "br") and header_len != expected_size:
                        raise ValueError(
                            f"Content-Length header mismatch for '{file_rel_path}': "
                            f"header specifies {header_len} bytes, metadata requires {expected_size} bytes."
                        )
                    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if content_encoding == "gzip" else None
                    with part_path.open("wb") as pf:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            data_chunk = decompressor.decompress(chunk) if decompressor is not None else chunk
                            if data_chunk:
                                pf.write(data_chunk)
                                hasher.update(data_chunk)
                                downloaded_bytes += len(data_chunk)
                        if decompressor is not None:
                            tail = decompressor.flush()
                            if tail:
                                pf.write(tail)
                                hasher.update(tail)
                                downloaded_bytes += len(tail)

            # Strict validations against official metadata
            if downloaded_bytes != expected_size:
                part_path.unlink(missing_ok=True)
                raise ValueError(
                    f"File '{file_rel_path}' download incomplete: expected {expected_size} bytes, "
                    f"got {downloaded_bytes} bytes."
                )

            obs_sha = hasher.hexdigest()
            if obs_sha != expected_sha256:
                part_path.unlink(missing_ok=True)
                raise ValueError(
                    f"File '{file_rel_path}' SHA-256 hash mismatch: expected {expected_sha256}, "
                    f"got {obs_sha}."
                )

            # Atomic rename
            part_path.replace(destination)
            return {
                "path": str(destination),
                "rel_path": file_rel_path,
                "status": "downloaded",
                "bytes": downloaded_bytes,
                "sha256": obs_sha,
            }
        except Exception as e:
            part_path.unlink(missing_ok=True)
            last_err = e
            if attempt_idx < max_retries:
                time.sleep(1.0 * attempt_idx)

    assert last_err is not None
    raise last_err


def get_sdk_patch_metadata() -> Dict[str, str]:
    """Computes deterministic version and SHA-256 hash of the isolated ModelScope 1.10.0 patch."""
    import inspect

    parts = [
        SDK_PATCH_VERSION,
        json.dumps(OFFICIAL_ANYTEXT_FILES_METADATA, sort_keys=True),
        json.dumps(OFFICIAL_EXCLUDE_PATH_PATTERNS),
    ]
    for fn in (filter_repo_files, inspect_catalog_and_cache_integrity, robust_download_file, patch_modelscope):
        try:
            parts.append(inspect.getsource(fn))
        except OSError:
            parts.append(fn.__code__.co_code.hex())

    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return {
        "sdk_patch_version": SDK_PATCH_VERSION,
        "sdk_patch_sha256": digest,
        "target_sdk": "modelscope==1.10.0",
    }


def patch_modelscope(metadata_map: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    """Applies an in-process compatibility patch to ModelScope 1.10.0 in the isolated AnyText runtime.

    1. Replaces `http_get_file(url, local_dir, file_name, cookies, headers=None)` in both
       `modelscope.hub.file_download` and `modelscope.hub.snapshot_download` so that responses
       omitting `Content-Length` (due to chunked gzip transfer of JSON/text files) succeed iff
       the downloaded payload matches both official metadata `Size` and `Sha256`. Fails closed
       with `FileDownloadError` on missing metadata, header mismatch, truncated length, or bad hash.
    2. Patches `HubApi.get_model_files` using `filter_repo_files` so regex filtering is evaluated
       against the full repository `Path` (excluding `nlp_csanmt_translation_zh2en/*`, `description/*`,
       `README.md`, and hidden files while retaining `clip-vit-large-patch14/merges.txt` and all
       inference files).
    3. Patches `ModelFileSystemCache.exists` so cached files are verified by on-disk size and
       SHA-256 against official metadata rather than trusting `.msc` entries or mere file existence.
    """
    try:
        import modelscope.hub.file_download as ms_fd
    except (ImportError, Exception):
        return False

    meta = metadata_map if metadata_map is not None else OFFICIAL_ANYTEXT_FILES_METADATA

    def _resolve_file_download_error() -> type:
        try:
            from modelscope.hub.errors import FileDownloadError
            return FileDownloadError
        except ImportError:
            try:
                from modelscope.utils.constant import FileDownloadError
                return FileDownloadError
            except ImportError:
                return RuntimeError

    # 1. Patch http_get_file with exact ModelScope 1.10.0 signature:
    #    http_get_file(url: str, local_dir: str, file_name: str, cookies: CookieJar, headers: Optional[Dict[str, str]] = None)
    orig_http_get_file = getattr(ms_fd, "_orig_http_get_file", None)
    if orig_http_get_file is None:
        orig_http_get_file = ms_fd.http_get_file
        ms_fd._orig_http_get_file = orig_http_get_file

    def patched_http_get_file(
        url: str,
        local_dir: str,
        file_name: str,
        cookies: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        err_cls = _resolve_file_download_error()
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        file_path_list = params.get("FilePath", [])
        rel_path = file_path_list[0] if file_path_list else file_name

        expected_meta = meta.get(rel_path)
        if (
            expected_meta is None
            or expected_meta.get("size") is None
            or expected_meta.get("sha256") is None
        ):
            raise err_cls(
                f"File '{rel_path}' has no official metadata (size and sha256 required) in registry; "
                f"failing closed to prevent unverified downloads"
            )

        expected_size = int(expected_meta["size"])
        expected_sha256 = str(expected_meta["sha256"])

        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)
        dest_path = local_dir_path / file_name
        temp_path = local_dir_path / f".{file_name}.part.{os.getpid()}"

        import requests
        req_headers = dict(headers) if headers else {}
        if "user-agent" not in {k.lower(): v for k, v in req_headers.items()}:
            req_headers["User-Agent"] = "ModelScope/1.10.0"

        try:
            temp_path.unlink(missing_ok=True)
            with requests.get(
                url,
                stream=True,
                cookies=cookies,
                headers=req_headers,
                timeout=(30, 300),
            ) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                header_len = int(content_length) if content_length is not None else None
                if header_len is not None and header_len != expected_size:
                    raise err_cls(
                        f"File '{rel_path}' Content-Length header ({header_len}) does not match "
                        f"official metadata size ({expected_size})"
                    )

                hasher = hashlib.sha256()
                downloaded_length = 0
                with temp_path.open("wb") as tf:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tf.write(chunk)
                            hasher.update(chunk)
                            downloaded_length += len(chunk)

                if expected_size != downloaded_length:
                    raise err_cls(
                        f"File '{rel_path}' download incomplete: expected {expected_size} bytes "
                        f"but got {downloaded_length} (Content-Length={content_length})"
                    )
                obs_sha = hasher.hexdigest()
                if obs_sha != expected_sha256:
                    raise err_cls(
                        f"File '{rel_path}' SHA-256 mismatch: expected {expected_sha256} "
                        f"but got {obs_sha}"
                    )

            os.replace(str(temp_path), str(dest_path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    ms_fd.http_get_file = patched_http_get_file
    ms_fd._anytext_sdk_patch_info = get_sdk_patch_metadata()

    # Also patch snapshot_download.http_get_file because snapshot_download imports it directly
    try:
        import modelscope.hub.snapshot_download as ms_sd
        if getattr(ms_sd, "_orig_http_get_file", None) is None and hasattr(ms_sd, "http_get_file"):
            ms_sd._orig_http_get_file = ms_sd.http_get_file
        ms_sd.http_get_file = patched_http_get_file
    except (ImportError, AttributeError):
        pass

    # 2. Patch HubApi.get_model_files to filter by full Path regex
    try:
        from modelscope.hub.api import HubApi
        orig_get_model_files = getattr(HubApi, "_orig_get_model_files", None)
        if orig_get_model_files is None:
            orig_get_model_files = HubApi.get_model_files
            HubApi._orig_get_model_files = orig_get_model_files

        def patched_get_model_files(self, *args, **kwargs):
            raw_files = orig_get_model_files(self, *args, **kwargs)
            return filter_repo_files(raw_files, OFFICIAL_EXCLUDE_PATH_PATTERNS)

        HubApi.get_model_files = patched_get_model_files
    except (ImportError, AttributeError):
        pass

    # 3. Patch ModelFileSystemCache.exists to enforce size + SHA-256 verification on cached files
    try:
        from modelscope.hub.utils.caching import ModelFileSystemCache
        orig_cache_exists = getattr(ModelFileSystemCache, "_orig_exists", None)
        if orig_cache_exists is None:
            orig_cache_exists = ModelFileSystemCache.exists
            ModelFileSystemCache._orig_exists = orig_cache_exists

        def patched_cache_exists(self, model_file: Dict[str, Any]) -> bool:
            rel_path = model_file.get("Path") or model_file.get("Name") or ""
            cache_root = getattr(self, "cache_root_location", None)
            if not cache_root or not rel_path:
                return False
            file_path = Path(cache_root) / rel_path
            if not file_path.is_file():
                return False
            exp_meta = meta.get(rel_path)
            exp_size = exp_meta.get("size") if exp_meta else model_file.get("Size")
            exp_sha = exp_meta.get("sha256") if exp_meta else model_file.get("Sha256")
            if exp_size is None or exp_sha is None:
                return False
            try:
                if file_path.stat().st_size != int(exp_size):
                    file_path.unlink(missing_ok=True)
                    return False
                if compute_sha256(file_path) != str(exp_sha):
                    file_path.unlink(missing_ok=True)
                    return False
                return True
            except OSError:
                return False

        ModelFileSystemCache.exists = patched_cache_exists
    except (ImportError, AttributeError):
        pass

    return True


def download_and_verify_modelscope_anytext(
    plan: Dict[str, Any],
    cache_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Downloads AnyText weights via ModelScope with hardened verification and cache reuse.

    Verifies checkpoint and catalog files by Size and SHA-256 against official metadata.
    Reuses valid cached files and downloads only missing or corrupted files.
    """
    model_spec = plan["model_specification"]
    repo_id = model_spec["modelscope_repo_id"]
    revision = model_spec["model_revision"]
    checkpoint_file = model_spec["checkpoint_file"]
    expected_ckpt_sha = model_spec["checkpoint_sha256"]
    planned_bytes = model_spec["total_required_weights_bytes"]

    # Read download ceiling from resource budget
    resource_budget = plan.get("resource_budget", {})
    max_download_gib = float(resource_budget.get("max_colab_download_gib", 12.0))
    download_ceiling_bytes = int(max_download_gib * (1024**3))

    planned_required_bytes = model_spec.get("total_required_weights_bytes", planned_bytes)
    planned_required_gib = round(planned_required_bytes / (1024**3), 4)

    # 1. Pre-download inventory verification against locked budget ceiling
    print(f"[INFO] Pre-download inventory verification: planned required weights {planned_required_gib} GiB ({planned_required_bytes} bytes).")
    print(f"[INFO] Locked budget ceiling: {max_download_gib} GiB ({download_ceiling_bytes} bytes).")

    if planned_required_bytes > download_ceiling_bytes:
        raise RuntimeError(
            f"Pre-download inventory check failed: planned required weights ({planned_required_gib} GiB) "
            f"exceed locked budget ceiling ({max_download_gib} GiB)."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir.resolve())

    # Pre-inventory of cache_dir before network download
    pre_inventory: Dict[str, int] = {}
    for f in cache_dir.rglob("*"):
        if f.is_file():
            try:
                pre_inventory[str(f.resolve())] = f.stat().st_size
            except OSError:
                pass
    pre_existing_cache_bytes = sum(pre_inventory.values())
    print(f"[INFO] Pre-download cache inventory: {len(pre_inventory)} files, {pre_existing_cache_bytes / (1024**3):.4f} GiB already present.")

    # 2. Locate model directory in cache
    existing_ckpt_path: Optional[Path] = None
    for p in cache_dir.rglob(checkpoint_file):
        if p.is_file():
            existing_ckpt_path = p
            break

    if existing_ckpt_path:
        model_dir_path = existing_ckpt_path.parent
    else:
        model_dir_path = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    cache_reused = False
    download_duration = 0.0
    repaired_files_count = 0
    downloaded_files_count = 0
    reused_valid_files_count = 0

    metadata_map = model_spec.get("files_metadata")
    if metadata_map is None and expected_ckpt_sha == OFFICIAL_ANYTEXT_FILES_METADATA["anytext_v1.1.ckpt"]["sha256"]:
        metadata_map = OFFICIAL_ANYTEXT_FILES_METADATA

    sdk_patch_info = get_sdk_patch_metadata()
    catalog_summary: Optional[Dict[str, Any]] = None

    if metadata_map is not None:
        catalog_pre = inspect_catalog_and_cache_integrity(
            model_dir=model_dir_path,
            metadata_map=metadata_map,
            download_ceiling_bytes=download_ceiling_bytes,
        )
        print(
            f"[INFO] Official catalog inventory ({revision}): {catalog_pre['catalog_file_count']} files, "
            f"{catalog_pre['catalog_total_gib']} GiB total; {len(catalog_pre['valid_cached_files'])} valid in cache, "
            f"{len(catalog_pre['missing_files'])} missing, {len(catalog_pre['corrupted_files'])} corrupted."
        )
        if not catalog_pre["within_budget"]:
            raise RuntimeError(
                f"Pre-download inventory check failed: catalog total ({catalog_pre['catalog_total_gib']} GiB) "
                f"or required download ({catalog_pre['gib_to_download']} GiB) exceeds ceiling ({max_download_gib} GiB)."
            )

        reused_valid_files_count = len(catalog_pre["valid_cached_files"])
        repaired_files_count = len(catalog_pre["corrupted_files"])

        # Remove any corrupted files immediately so they are never trusted
        for bad_item in catalog_pre["corrupted_files"]:
            bad_path = model_dir_path / bad_item["rel_path"]
            print(f"[WARN] Removing corrupted cached file '{bad_item['rel_path']}' ({bad_item['reason']}).")
            bad_path.unlink(missing_ok=True)

        if catalog_pre["all_valid"]:
            print("[PASS] All 11 official catalog files verified by Size and SHA-256 in cache! Reusing valid cache.")
            patch_modelscope(metadata_map)
            cache_reused = True
        elif checkpoint_file in catalog_pre["valid_cached_files"]:
            missing_items = list(catalog_pre["files_to_download"].items())
            print(
                f"[INFO] Main checkpoint '{checkpoint_file}' is verified intact in cache! "
                f"Downloading only {len(missing_items)} missing/invalid files ({catalog_pre['gib_to_download']} GiB)..."
            )
            patch_modelscope(metadata_map)
            t0 = time.time()
            for rel_p, m in missing_items:
                dest_f = model_dir_path / rel_p
                print(f"[INFO] Downloading {rel_p} ({m['size']} bytes)...")
                robust_download_file(
                    repo_id=repo_id,
                    file_rel_path=rel_p,
                    revision=revision,
                    destination=dest_f,
                    expected_size=m["size"],
                    expected_sha256=m["sha256"],
                    metadata_map=metadata_map,
                )
                downloaded_files_count += 1
            download_duration = time.time() - t0
            print(f"[PASS] Successfully downloaded and verified {downloaded_files_count} missing/repaired files in {download_duration:.2f}s.")
            cache_reused = True

        catalog_summary = {
            "catalog_file_count": catalog_pre["catalog_file_count"],
            "catalog_total_bytes": catalog_pre["catalog_total_bytes"],
            "reused_valid_files_count": reused_valid_files_count,
            "corrupted_files_repaired_count": repaired_files_count,
            "downloaded_files_count": downloaded_files_count,
        }
    elif existing_ckpt_path and existing_ckpt_path.is_file():
        print(f"[INFO] Inspecting existing checkpoint in cache: {existing_ckpt_path}")
        try:
            obs_sha = compute_sha256(existing_ckpt_path)
            if obs_sha == expected_ckpt_sha:
                clip_sub = existing_ckpt_path.parent / "clip-vit-large-patch14"
                clip_ok, clip_missing = verify_clip_weights_complete(clip_sub)
                if clip_ok:
                    print("[PASS] Verified existing checkpoint SHA-256 and CLIP weights in cache! Reusing valid cache.")
                    cache_reused = True
                else:
                    print(f"[INFO] Checkpoint matched but CLIP weights incomplete in cache (missing: {clip_missing}). Will proceed with download.")
            else:
                print(f"[INFO] Checkpoint SHA mismatch in cache ({obs_sha[:8]} != {expected_ckpt_sha[:8]}). Will re-download.")
        except Exception as e:
            print(f"[INFO] Cache inspection check error: {e}")

    if not cache_reused:
        # Patch modelscope before initiating snapshot_download
        patch_modelscope(metadata_map)

        from modelscope.hub.snapshot_download import snapshot_download

        print(f"[INFO] Initiating ModelScope snapshot download for {repo_id} ({revision})...")
        print(f"[INFO] Active SDK compatibility patch: {sdk_patch_info['sdk_patch_version']} (sha256={sdk_patch_info['sdk_patch_sha256'][:16]}...)")
        print(f"[INFO] Excluding 'nlp_csanmt_translation_zh2en/*', 'description/*', 'README.md', and '*.md' via full-path regex filter...")

        t0 = time.time()
        try:
            model_dir = snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=str(cache_dir.resolve()),
                ignore_file_pattern=OFFICIAL_EXCLUDE_PATH_PATTERNS,
            )
            download_duration = time.time() - t0
            model_dir_path = Path(model_dir)
        except Exception as dl_err:
            download_duration = time.time() - t0
            # If download threw an error, check if checkpoint exists and matches SHA
            found_ckpt: Optional[Path] = None
            for p in cache_dir.rglob(checkpoint_file):
                if p.is_file():
                    found_ckpt = p
                    break
            if found_ckpt and compute_sha256(found_ckpt) == expected_ckpt_sha:
                # Checkpoint is intact — also verify CLIP weights are complete
                clip_sub_err = found_ckpt.parent / "clip-vit-large-patch14"
                clip_ok_err, clip_missing_err = verify_clip_weights_complete(clip_sub_err, metadata_map=metadata_map)
                if not clip_ok_err:
                    failure_receipt = {
                        "status": "FAILED",
                        "stage": "snapshot_download_error_handler",
                        "error": str(dl_err),
                        "checkpoint_sha256_matched": True,
                        "clip_complete": False,
                        "clip_missing_files": clip_missing_err,
                        "sdk_patch_version": sdk_patch_info["sdk_patch_version"],
                        "sdk_patch_sha256": sdk_patch_info["sdk_patch_sha256"],
                        "message": "Checkpoint intact but CLIP weights incomplete after download error. Cannot continue.",
                    }
                    receipt_path = found_ckpt.parent / "failure_receipt.json"
                    receipt_path.write_text(json.dumps(failure_receipt, indent=2), encoding="utf-8")
                    print(f"[FAIL] Checkpoint intact but CLIP incomplete (missing: {clip_missing_err}). Failure receipt: {receipt_path}")
                    raise RuntimeError(
                        f"CLIP weights incomplete after snapshot_download error (missing: {clip_missing_err}). "
                        f"Failure receipt written to {receipt_path}"
                    ) from dl_err
                print(f"[WARN] snapshot_download encountered non-fatal error: {dl_err}, but required checkpoint and CLIP weights are intact. Reusing cache.")
                model_dir_path = found_ckpt.parent
            else:
                raise dl_err

    assert model_dir_path is not None, "Model directory path resolution failed."

    # Check translator directory was excluded
    translator_dir = model_dir_path / "nlp_csanmt_translation_zh2en"
    has_translator = translator_dir.is_dir() and any(translator_dir.iterdir())
    if has_translator:
        raise RuntimeError("Translator directory was unexpectedly downloaded! Exceeds budget constraints.")

    # Post-inventory of cache_dir to measure on-disk cache growth
    post_inventory: Dict[str, int] = {}
    for f in cache_dir.rglob("*"):
        if f.is_file():
            try:
                post_inventory[str(f.resolve())] = f.stat().st_size
            except OSError:
                pass

    cache_total_files_bytes = sum(f.stat().st_size for f in model_dir_path.rglob("*") if f.is_file())
    cache_total_files_gib = round(cache_total_files_bytes / (1024**3), 4)

    # Observed cache growth on disk during download (filesystem delta, not direct network traffic)
    growth_bytes = 0
    for p, sz in post_inventory.items():
        if p not in pre_inventory:
            growth_bytes += sz
        elif sz > pre_inventory[p]:
            growth_bytes += (sz - pre_inventory[p])
    observed_cache_growth_bytes = growth_bytes
    observed_cache_growth_gib = round(observed_cache_growth_bytes / (1024**3), 4)

    # Supplementary post-download check
    if observed_cache_growth_bytes > download_ceiling_bytes:
        raise RuntimeError(
            f"Supplementary post-download check failed: observed cache growth {observed_cache_growth_gib} GiB "
            f"exceeded budget ceiling {max_download_gib} GiB."
        )

    # Checkpoint hash verification
    ckpt_path = model_dir_path / checkpoint_file
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    observed_ckpt_sha = compute_sha256(ckpt_path)
    if observed_ckpt_sha != expected_ckpt_sha:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch! Expected {expected_ckpt_sha}, got {observed_ckpt_sha}"
        )

    # CLIP weights completeness verification (fail-closed)
    clip_dir_post = model_dir_path / "clip-vit-large-patch14"
    clip_complete_post, clip_missing_post = verify_clip_weights_complete(clip_dir_post, metadata_map=metadata_map)
    if not clip_complete_post:
        failure_receipt = {
            "status": "FAILED",
            "stage": "post_download_clip_verification",
            "checkpoint_sha256_matched": True,
            "clip_complete": False,
            "clip_missing_files": clip_missing_post,
            "sdk_patch_version": sdk_patch_info["sdk_patch_version"],
            "sdk_patch_sha256": sdk_patch_info["sdk_patch_sha256"],
            "message": "Checkpoint verified but CLIP weights incomplete. Cannot proceed to inference.",
        }
        failure_path = model_dir_path / "failure_receipt.json"
        failure_path.write_text(json.dumps(failure_receipt, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"CLIP weights incomplete after download (missing: {clip_missing_post}). "
            f"Failure receipt: {failure_path}"
        )

    # If official metadata_map is active, verify all catalog files are intact post-download
    if metadata_map is not None:
        catalog_post = inspect_catalog_and_cache_integrity(
            model_dir=model_dir_path,
            metadata_map=metadata_map,
            download_ceiling_bytes=download_ceiling_bytes,
        )
        if not catalog_post["all_valid"]:
            raise RuntimeError(
                f"Post-download catalog integrity verification failed! "
                f"Missing: {catalog_post['missing_files']}, Corrupted: {catalog_post['corrupted_files']}"
            )
        if catalog_summary is not None:
            catalog_summary["post_download_all_valid"] = True

    receipt = {
        "model_dir": str(model_dir_path),
        "repo_id": repo_id,
        "revision": revision,
        "download_duration_sec": round(download_duration, 2),
        "planned_weights_bytes": planned_bytes,
        "planned_weights_gib": planned_required_gib,
        "download_ceiling_gib": max_download_gib,
        "download_ceiling_bytes": download_ceiling_bytes,
        "pre_download_inventory_verified": True,
        "cache_reused": cache_reused,
        "cache_pre_existing_bytes": pre_existing_cache_bytes,
        "cache_total_files_bytes": cache_total_files_bytes,
        "cache_total_files_gib": cache_total_files_gib,
        "observed_cache_growth_bytes": observed_cache_growth_bytes,
        "observed_cache_growth_gib": observed_cache_growth_gib,
        "network_traffic_bytes": "NOT_MEASURED",
        "network_accounting_status": "NOT_MEASURED",
        "network_accounting_description": "Cache growth observed via pre/post directory inventory diff, not direct socket network measurement.",
        "within_budget_ceiling": observed_cache_growth_bytes <= download_ceiling_bytes,
        "translator_excluded": True,
        "readme_excluded": True,
        "checkpoint_sha256": observed_ckpt_sha,
        "checkpoint_status": "VERIFIED_MATCH",
        "clip_complete": True,
        "clip_missing_files": [],
        "sdk_patch_version": sdk_patch_info["sdk_patch_version"],
        "sdk_patch_sha256": sdk_patch_info["sdk_patch_sha256"],
        "sdk_patch_target": sdk_patch_info["target_sdk"],
        "catalog_verification": catalog_summary,
    }
    print(f"[PASS] Model cache verified: {cache_total_files_gib} GiB on disk, observed cache growth: {observed_cache_growth_gib} GiB (within {max_download_gib} GiB ceiling).")
    print(f"[PASS] Network traffic note: Direct network socket traffic NOT_MEASURED (cache growth is reported as on-disk filesystem delta).")
    print(f"[PASS] Checkpoint verified: {observed_ckpt_sha[:16]}...")
    print(f"[PASS] CLIP weights verified: all mandatory files (including merges.txt) present and valid.")
    return str(model_dir_path), receipt


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


def target_bbox_in_canvas(geometry_receipt: Dict[str, Any]) -> List[int]:
    """Maps the native target bbox [ymin, xmin, ymax, xmax] into 512x512 model-canvas coordinates."""
    crop_ymin, crop_xmin, _, _ = geometry_receipt["crop_bbox"]
    ymin, xmin, ymax, xmax = geometry_receipt["target_bbox"]
    pad_y, pad_x = geometry_receipt["padding"]
    scale = geometry_receipt["scale"]
    return [
        int(round(pad_y + (ymin - crop_ymin) * scale)),
        int(round(pad_x + (xmin - crop_xmin) * scale)),
        int(round(pad_y + (ymax - crop_ymin) * scale)),
        int(round(pad_x + (xmax - crop_xmin) * scale)),
    ]


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


def finalize_run(
    output_root: Path,
    run_dir: Path,
    run_id: str,
    status: str,
    plan_sha256: str,
    stage: str,
    attempt_statuses: Dict[str, Dict[str, Any]],
    precision_info: Optional[Dict[str, Any]],
    run_label: str,
) -> Path:
    """Writes run_manifest.json, archives the run under a run-specific name and updates latest_run.json."""
    manifest = {
        "run_id": run_id,
        "run_label": run_label,
        "status": status,
        "stage_reached": stage,
        "plan_sha256": plan_sha256,
        "runner_sha256": compute_sha256(Path(__file__).resolve()),
        "precision_policy": precision_info,
        "attempts": attempt_statuses,
        "human_content_qc": "PENDING_HUMAN_REVIEW",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    kind = "results" if status == "COMPLETE" else "partial_results"
    zip_path = output_root / f"anytext_feasibility_{kind}_{run_id}.zip"
    package_results_zip(run_dir=run_dir, output_zip_path=zip_path)
    pointer = {
        "run_id": run_id,
        "status": status,
        "run_dir": run_dir.name,
        "archive": zip_path.name,
        "archive_sha256": compute_sha256(zip_path),
        "timestamp": manifest["timestamp"],
    }
    (output_root / "latest_run.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return zip_path


def main() -> int:
    # Ensure script directory and current working directory are in sys.path
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    parser = argparse.ArgumentParser(description="Run AnyText Feasibility Runner")
    parser.add_argument("--plan-path", type=str, required=True, help="Path to anytext_feasibility_plan.json")
    parser.add_argument("--inputs-zip", type=str, default=None, help="Path to document_feasibility_results.zip")
    parser.add_argument("--inputs-dir", type=str, default=None, help="Path to inputs directory")
    parser.add_argument("--output-root", type=str, required=True, help="Path to outputs root directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to ModelScope cache directory")
    parser.add_argument("--custom-font", type=str, default=None, help="Custom TTF font path")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only")
    parser.add_argument("--run-label", type=str, default="", help="Short label embedded in the run id (e.g. repair_fp16_xformers)")
    args = parser.parse_args()

    plan_path = Path(args.plan_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lock_file = output_root / ".runner.lock"
    try:
        acquire_runner_lock(lock_file)
    except RuntimeError as lock_err:
        print(f"[FAIL] {lock_err}")
        return 1

    current_stage = "INITIALIZATION"
    run_dir: Optional[Path] = None
    plan_bytes_sha = "UNKNOWN"
    run_id = "UNKNOWN"
    precision_info: Optional[Dict[str, Any]] = None
    attempt_statuses: Dict[str, Dict[str, Any]] = {}

    try:
        if not plan_path.is_file():
            raise FileNotFoundError(f"Plan not found: {plan_path}")

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_bytes_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = re.sub(r"[^A-Za-z0-9_-]", "", args.run_label)
        run_id = f"run_anytext_{label + '_' if label else ''}{run_timestamp}_{plan_bytes_sha[:8]}"
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

        current_stage = "INPUT_EXTRACTION"
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
                raise FileNotFoundError("Neither valid --inputs-zip nor --inputs-dir provided, and default archive not found.")

        current_stage = "PREFLIGHT"
        hw_info = check_gpu_and_resources()
        print(f"CUDA Available: {hw_info['cuda_available']} ({hw_info['device_name']}, VRAM: {hw_info['vram_total_gb']} GB)")
        print(f"Disk Free: {hw_info['disk_free_gb']} GB")

        font_path, font_sha256 = resolve_font_path(args.custom_font)
        print(f"Resolved Font: {font_path} (SHA-256: {font_sha256[:16]}...)")

        preflight_receipt = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "plan_sha256": plan_bytes_sha,
            "hardware": hw_info,
            "font": {"path": font_path, "sha256": font_sha256},
            "inputs_verified": {k: v["authentic_sha256"] for k, v in extracted_inputs.items()},
            "cache_dir": str(model_cache_dir),
            "sdk_patch": get_sdk_patch_metadata(),
            "status": "PREFLIGHT_PASS",
        }
        (run_dir / "preflight_receipt.json").write_text(json.dumps(preflight_receipt, indent=2), encoding="utf-8")

        if args.preflight_only:
            print("[OK] Preflight completed successfully.")
            return 0

        if not hw_info["cuda_available"]:
            raise RuntimeError("CUDA GPU is required for AnyText generation. Running on CPU is not supported.")

        current_stage = "MODEL_DOWNLOAD"
        print("\n--- Downloading AnyText Model (Excluding Translator & Docs) ---")
        model_dir, model_receipt = download_and_verify_modelscope_anytext(
            plan=plan,
            cache_dir=model_cache_dir,
        )
        (run_dir / "model_download_receipt.json").write_text(json.dumps(model_receipt, indent=2), encoding="utf-8")

        current_stage = "PIPELINE_INIT"
        print("\n--- Initializing AnyText Pipeline from Local Weights ---")
        precision_info = assert_fp16_attention_path()
        print(f"[PASS] fp16 attention path: xformers {precision_info['xformers']} active (policy {PRECISION_POLICY_ID}).")
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

        current_stage = "INFERENCE"
        contact_sheet_items: List[Dict[str, Any]] = []
        review_records: List[Dict[str, Any]] = []
        attempts = plan["attempts"]
        has_failure = False

        for att in attempts:
            att_id = att["attempt_id"]
            att_dir = run_dir / att_id
            att_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- Checking / Executing {att_id} ---")

            auth_info = extracted_inputs.get(att_id)
            if not auth_info:
                print(f"[FAIL] Missing authentic input data for {att_id}")
                has_failure = True
                fail_receipt = {
                    "attempt_id": att_id,
                    "stage": "INPUT_EXTRACTION",
                    "status": "FAILED",
                    "error": f"Missing authentic input data for {att_id}",
                    "traceback": "",
                }
                attempt_statuses[att_id] = {"status": "FAILED", "stage": "INPUT_EXTRACTION"}
                (att_dir / "failure_receipt.json").write_text(json.dumps(fail_receipt, indent=2), encoding="utf-8")
                continue

            att_fingerprint = compute_attempt_fingerprint(plan, att, auth_info["authentic_sha256"])
            # Check if prior completed attempt exists
            prior_att_folder = find_prior_completed_attempt(output_root, att_id, att_fingerprint, exclude_dir=run_dir)
            if prior_att_folder:
                print(f"[PASS] Prior completed attempt found at: {prior_att_folder}")
                # Copy artifacts to att_dir if different
                if prior_att_folder.resolve() != att_dir.resolve():
                    for src_item in prior_att_folder.iterdir():
                        if src_item.is_file():
                            shutil.copy2(src_item, att_dir / src_item.name)

                prior_rcpt_file = att_dir / "attempt_receipt.json"
                if prior_rcpt_file.is_file():
                    att_receipt = json.loads(prior_rcpt_file.read_text(encoding="utf-8"))
                    print(f"[INFO] Reusing verified result for {att_id} (outside_mask_bitwise_equal={att_receipt.get('outside_mask_bitwise_equal')}).")
                    attempt_statuses[att_id] = {"status": "REUSED_VERIFIED", "reused_from": prior_att_folder.parent.name, "config_fingerprint": att_fingerprint}

                    auth_roi_path = att_dir / "authentic_roi_zoomed.png"
                    sdxl_roi_path = att_dir / "sdxl_roi_zoomed.png"
                    raw_roi_path = att_dir / "anytext_raw_roi_zoomed.png"
                    final_roi_path = att_dir / "anytext_final_roi_zoomed.png"
                    final_img_path = att_dir / "anytext_final_edited.png"
                    if not final_img_path.is_file():
                        final_img_path = att_dir / "final_edited_document.png"
                    mask_roi_path = att_dir / "mask_roi_zoomed.png"

                    contact_sheet_items.append({
                        "id": att_id,
                        "target_text": att["target_text"],
                        "seed": att["seed"],
                        "latency": att_receipt.get("latency_sec", 0.0),
                        "authentic_roi": Image.open(auth_roi_path).convert("RGB") if auth_roi_path.is_file() else None,
                        "sdxl_roi": Image.open(sdxl_roi_path).convert("RGB") if sdxl_roi_path.is_file() else None,
                        "anytext_raw_roi": Image.open(raw_roi_path).convert("RGB") if raw_roi_path.is_file() else None,
                        "anytext_final_roi": Image.open(final_roi_path).convert("RGB") if final_roi_path.is_file() else None,
                        "anytext_final_full": Image.open(final_img_path).convert("RGB") if final_img_path.is_file() else None,
                        "mask_roi": Image.open(mask_roi_path).convert("RGB") if mask_roi_path.is_file() else None,
                    })

                    review_records.append({
                        "attempt_id": att_id,
                        "source_row_index": att["source_row_index"],
                        "original_text": att["original_text"],
                        "target_text": att["target_text"],
                        "exact_text_success": "PENDING_HUMAN_REVIEW",
                        "visual_realism": "PENDING_HUMAN_REVIEW",
                        "human_verdict": "PENDING_HUMAN_REVIEW",
                        "notes": "Reused prior verified attempt; awaiting human reviewer assessment.",
                    })
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
                t_box_canvas = target_bbox_in_canvas(prep_res["geometry_receipt"])
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
                    "run_id": run_id,
                    "config_fingerprint": att_fingerprint,
                    "precision_policy": precision_info,
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
                attempt_statuses[att_id] = {"status": "TECHNICAL_QC_PASS", "config_fingerprint": att_fingerprint}

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
                print(f"[FAIL] Error in {att_id}: {type(e).__name__}: {e}")
                fail_receipt = build_failure_receipt(
                    "INFERENCE", e, attempt_id=att_id, run_id=run_id,
                    extra={"precision_policy": precision_info, "config_fingerprint": att_fingerprint},
                )
                print(f"[FAIL] Failing frame: {fail_receipt['failure_site'].get('innermost_project_frame', {}).get('file')}:"
                      f"{fail_receipt['failure_site'].get('innermost_project_frame', {}).get('line')}")
                attempt_statuses[att_id] = {"status": "FAILED", "stage": "INFERENCE", "error_type": type(e).__name__}
                (att_dir / "failure_receipt.json").write_text(json.dumps(fail_receipt, indent=2), encoding="utf-8")

        current_stage = "PACKAGING"
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

        if has_failure:
            fail_rcpt = {
                "run_id": run_id,
                "stage": "INFERENCE",
                "status": "PARTIAL_FAILURE",
                "error": "One or more generation attempts failed.",
                "traceback": "",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            fail_rcpt["attempts"] = attempt_statuses
            (run_dir / "failure_receipt.json").write_text(json.dumps(fail_rcpt, indent=2), encoding="utf-8")

            partial_zip = finalize_run(output_root, run_dir, run_id, "PARTIAL_FAILURE", plan_bytes_sha, current_stage, attempt_statuses, precision_info, args.run_label)
            print(f"\n[WARNING] Runner completed with partial failure. Partial archive created at: {partial_zip}")
            return 1

        finalize_run(output_root, run_dir, run_id, "COMPLETE", plan_bytes_sha, current_stage, attempt_statuses, precision_info, args.run_label)

        print("\n[DONE] AnyText Feasibility Run executed successfully.")
        return 0

    except Exception as err:
        err_tb = traceback.format_exc()
        print("\n" + "=" * 80)
        print(f"[RUNNER FAILURE] Stage: {current_stage}")
        print(f"Error: {err}")
        print(f"Exit Code: 1")
        print(f"Traceback:\n{err_tb}")
        print("=" * 80)

        fail_info = build_failure_receipt(
            current_stage, err, run_id=run_id,
            extra={"plan_sha256": plan_bytes_sha, "precision_policy": precision_info, "attempts": attempt_statuses},
        )
        if run_dir and run_dir.is_dir():
            (run_dir / "failure_receipt.json").write_text(json.dumps(fail_info, indent=2), encoding="utf-8")
            try:
                partial_zip = finalize_run(output_root, run_dir, run_id, "FAILED", plan_bytes_sha, current_stage, attempt_statuses, precision_info, args.run_label)
                print(f"[INFO] Partial results archive created at: {partial_zip}")
            except Exception as zip_e:
                print(f"[WARN] Failed to package partial archive: {zip_e}")
        else:
            # No run directory yet: unique name so earlier receipts are never overwritten.
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            (output_root / f"failure_receipt_init_{stamp}.json").write_text(json.dumps(fail_info, indent=2), encoding="utf-8")

        return 1

    finally:
        release_runner_lock(lock_file)



if __name__ == "__main__":
    sys.exit(main())
