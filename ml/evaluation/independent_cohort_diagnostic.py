"""Diagnostic evaluation harness for Phase 4C.7B independent cohort acquisition.

Implements the explicit, fail-closed diagnostic execution pipeline:
- Compares full-canvas inference (Arm A) vs local-crop padded inference (Arm B) across
  the 3 omission candidates (tomato, suitcase, bird).
- Maximum 6 attempts hard budget (1 attempt per arm).
- Fail-closed approval guard: refuses generation unless plan status is 'APPROVED'.
- Durable attempt ledger and run binding with tamper detection.
- Re-seeds PRNG with registered seed before each attempt (no shared RNG state).
- Preserves raw pre-compositing generated images alongside final 512x512 composites.
- Guarantees exact bitwise invariance of outside-mask pixels (outside_mean_L1 == 0.000000).
- Fully isolated from the official cohort (dedicated run directory, receipts, and contact sheet).
"""

from __future__ import annotations

import base64
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort_acquisition import (
    INPAINTING_MODEL_REGISTRY,
    NON_BLANK_STD_THRESHOLD,
    MIN_MASKED_PIXEL_DELTA_L1,
    MAX_UNMASKED_PIXEL_DELTA_L1,
    TARGET_CANVAS_SIZE,
    GenerationContractError,
    assert_detector_isolation,
    check_gpu_policy,
)

DIAGNOSTIC_BUDGET_LIMIT = 6


class DiagnosticPlanNotApprovedError(RuntimeError):
    """Raised when attempting to execute a diagnostic plan whose status is not APPROVED."""


class DiagnosticPlanIntegrityError(ValueError):
    """Raised when diagnostic plan hash or structure fails validation."""


class DiagnosticInputMissingError(FileNotFoundError):
    """Raised when an input authentic or mask image is missing."""


class DiagnosticInputHashMismatchError(ValueError):
    """Raised when an input image hash does not match the plan's registered hash."""


class DiagnosticBudgetExceededError(RuntimeError):
    """Raised when attempts exceed the 6-attempt budget limit."""


class DiagnosticGeometryError(ValueError):
    """Raised when crop or mask coordinates violate canvas bounds."""


class DiagnosticRunAuditError(RuntimeError):
    """Raised when an existing diagnostic run fails audit."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_and_validate_diagnostic_plan(
    plan_path: Path | str,
    expected_plan_sha256: str | None = None,
    require_approval: bool = True,
) -> dict[str, Any]:
    """Load and validate the diagnostic plan fail-closed."""
    p_path = Path(plan_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Diagnostic plan not found at {p_path}")

    content = p_path.read_bytes()
    actual_hash = _sha256_bytes(content)
    if expected_plan_sha256 and actual_hash != expected_plan_sha256:
        raise DiagnosticPlanIntegrityError(
            f"Diagnostic plan SHA-256 mismatch: expected {expected_plan_sha256}, got {actual_hash}"
        )

    data = json.loads(content.decode("utf-8"))

    if require_approval:
        status = data.get("human_review_status")
        if status != "APPROVED":
            raise DiagnosticPlanNotApprovedError(
                f"Diagnostic plan human_review_status is '{status}' (expected 'APPROVED'). "
                "Execution is refused until explicitly approved by human reviewer."
            )

    budget = data.get("attempt_budget", 0)
    if budget != DIAGNOSTIC_BUDGET_LIMIT:
        raise DiagnosticBudgetExceededError(
            f"Diagnostic plan attempt_budget {budget} != expected {DIAGNOSTIC_BUDGET_LIMIT}"
        )

    candidates = data.get("diagnostic_candidates", [])
    if len(candidates) != 3:
        raise DiagnosticPlanIntegrityError(
            f"Expected exactly 3 diagnostic candidates, found {len(candidates)}"
        )

    for cand in candidates:
        arms = cand.get("arms", {})
        if "arm_a_full_canvas" not in arms or "arm_b_local_crop" not in arms:
            raise DiagnosticPlanIntegrityError(
                f"Candidate {cand.get('candidate_id')} must define both arm_a_full_canvas and arm_b_local_crop"
            )

    return data


def verify_diagnostic_inputs(
    plan_data: dict[str, Any],
    inputs_dir: Path | str,
) -> dict[str, dict[str, Path]]:
    """Verify that all authentic and mask images exist and match their registered SHA-256 hashes."""
    base_dir = Path(inputs_dir)
    verified: dict[str, dict[str, Path]] = {}

    for cand in plan_data.get("diagnostic_candidates", []):
        cid = cand["candidate_id"]
        auth_fname = cand["authentic_input_filename"]
        auth_hash = cand["authentic_sha256"]
        mask_fname = cand["mask_input_filename"]
        mask_hash = cand["mask_sha256"]

        # Search in base_dir, base_dir/images, base_dir/masks
        auth_path = None
        for cand_path in (base_dir / auth_fname, base_dir / "images" / auth_fname):
            if cand_path.is_file():
                auth_path = cand_path
                break
        if not auth_path:
            raise DiagnosticInputMissingError(
                f"Authentic input image for {cid} not found (looked for {auth_fname} in {base_dir})"
            )

        actual_auth_hash = _sha256_file(auth_path)
        if actual_auth_hash != auth_hash:
            raise DiagnosticInputHashMismatchError(
                f"Authentic image {auth_path.name} SHA-256 {actual_auth_hash} != registered {auth_hash}"
            )

        mask_path = None
        for cand_path in (base_dir / mask_fname, base_dir / "masks" / mask_fname):
            if cand_path.is_file():
                mask_path = cand_path
                break
        if not mask_path:
            raise DiagnosticInputMissingError(
                f"Mask input image for {cid} not found (looked for {mask_fname} in {base_dir})"
            )

        actual_mask_hash = _sha256_file(mask_path)
        if actual_mask_hash != mask_hash:
            raise DiagnosticInputHashMismatchError(
                f"Mask image {mask_path.name} SHA-256 {actual_mask_hash} != registered {mask_hash}"
            )

        # Verify raster dimensions and mask binary values
        with Image.open(auth_path) as im:
            if im.size != TARGET_CANVAS_SIZE or im.mode != "RGB":
                raise DiagnosticGeometryError(
                    f"Authentic image {auth_path} must be 512x512 RGB, got {im.size} {im.mode}"
                )
        with Image.open(mask_path) as mk:
            if mk.size != TARGET_CANVAS_SIZE:
                raise DiagnosticGeometryError(
                    f"Mask image {mask_path} must be 512x512, got {mk.size}"
                )
            arr = np.array(mk)
            uniques = np.unique(arr)
            if not set(uniques.tolist()).issubset({0, 255}):
                raise DiagnosticGeometryError(
                    f"Mask {mask_path} must be strictly binary {{0, 255}}, got {uniques}"
                )

        verified[cid] = {"authentic": auth_path, "mask": mask_path}

    return verified


class DiagnosticMockEngine:
    """Fast deterministic mock engine for CPU unit tests. Completely isolated from detector models."""

    def __init__(self, tool_key: str = "mock"):
        self.tool_key = tool_key

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        seed: int,
        height: int,
        width: int,
    ) -> Image.Image:
        rng = np.random.default_rng(seed)
        auth_arr = np.array(authentic_img, dtype=np.uint8).copy()
        mask_arr = np.array(mask_img, dtype=np.uint8)

        # Generate deterministic synthetic patch
        color_base = rng.integers(50, 210, size=3, dtype=np.uint8)
        noise = rng.integers(-25, 25, size=(height, width, 3), dtype=np.int16)
        synthetic_patch = np.clip(color_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        in_mask = mask_arr == 255
        auth_arr[in_mask] = synthetic_patch[in_mask]
        return Image.fromarray(auth_arr, mode="RGB")

    def describe(self) -> dict[str, Any]:
        return {"engine": "DiagnosticMockEngine", "is_synthetic": True}

    def peak_vram_bytes(self) -> int | str:
        return "NOT_MEASURED"

    def unload(self) -> None:
        pass


class DiagnosticDiffusersEngine:
    """Production GPU inpainting engine using HuggingFace Diffusers."""

    def __init__(
        self,
        tool_key: str,
        device: str = "cuda",
        torch_dtype: Any = None,
        resolved_revision: str | None = None,
    ) -> None:
        self.tool_key = tool_key
        self.device = device
        self.pipeline: Any = None

        if tool_key not in INPAINTING_MODEL_REGISTRY:
            raise ValueError(f"Unknown inpainting tool key: {tool_key}")
        reg = INPAINTING_MODEL_REGISTRY[tool_key]
        checkpoint = reg["checkpoint"]

        try:
            import diffusers
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Diffusers or PyTorch not available for GPU execution: {e}."
            ) from e

        self.torch = torch
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        pipeline_cls = getattr(diffusers, reg["pipeline_class_name"])
        scheduler_cls = getattr(diffusers, reg["scheduler_class_name"])
        extra = dict(reg.get("extra_pipeline_kwargs", {}))
        self.num_steps = reg["num_inference_steps"]
        self.guidance_scale = reg["guidance_scale"]

        self.revision = resolved_revision or reg["pinned_revision"]
        self.pipeline = pipeline_cls.from_pretrained(
            checkpoint, revision=self.revision, torch_dtype=self.dtype, **extra
        )
        self.pipeline.scheduler = scheduler_cls.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

        self.config = {
            "engine": "DiagnosticDiffusersEngine",
            "tool_key": tool_key,
            "checkpoint": checkpoint,
            "revision": self.revision,
            "pipeline": pipeline_cls.__name__,
            "scheduler": scheduler_cls.__name__,
            "dtype": str(self.dtype),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
            "diffusers_version": diffusers.__version__,
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        seed: int,
        height: int,
        width: int,
    ) -> Image.Image:
        if authentic_img.size != (width, height) or authentic_img.mode != "RGB":
            raise GenerationContractError(
                f"Input authentic image size/mode invalid: {authentic_img.size}, {authentic_img.mode}; expected ({width}, {height}), RGB"
            )
        if mask_img.size != (width, height) or mask_img.mode not in ("L", "1"):
            raise GenerationContractError(
                f"Input mask image size/mode invalid: {mask_img.size}, {mask_img.mode}; expected ({width}, {height}), L"
            )

        # Fresh isolated generator per attempt seeded with registered seed
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            image=authentic_img,
            mask_image=mask_img,
            height=height,
            width=width,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        if result.size != (width, height) or result.mode != "RGB":
            raise GenerationContractError(
                f"Diffusers inpainting produced invalid output {result.size}, {result.mode}; expected ({width}, {height}), RGB"
            )
        return result

    def describe(self) -> dict[str, Any]:
        return dict(self.config)

    def peak_vram_bytes(self) -> int | str:
        if self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated())
        return "NOT_MEASURED"

    def unload(self) -> None:
        del self.pipeline
        self.pipeline = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def execute_diagnostic_run(
    output_dir: Path | str,
    plan_path: Path | str,
    inputs_dir: Path | str,
    expected_commit: str,
    engine_provider: Callable[[str], Any] | None = None,
    resume: bool = False,
    require_approval: bool = True,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Execute the 6-attempt diagnostic calibration run fail-closed."""
    assert_detector_isolation()
    out_dir = Path(output_dir)

    plan_p = Path(plan_path)
    plan_data = load_and_validate_diagnostic_plan(
        plan_p, require_approval=require_approval
    )
    plan_sha256 = _sha256_file(plan_p)

    verified_inputs = verify_diagnostic_inputs(plan_data, inputs_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = out_dir / "attempt_ledger.jsonl"
    binding_path = out_dir / "run_binding.json"

    run_binding = {
        "run_id": out_dir.name,
        "mode": "diagnostic",
        "git_commit": expected_commit,
        "diagnostic_plan_sha256": plan_sha256,
        "total_budget": DIAGNOSTIC_BUDGET_LIMIT,
        "bound_pilot_run_id": plan_data["bound_pilot_run_id"],
        "bound_pilot_zip_sha256": plan_data["bound_pilot_zip_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if binding_path.exists():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding.get("diagnostic_plan_sha256") != plan_sha256:
            raise DiagnosticRunAuditError(
                f"Existing run binding plan hash {existing_binding.get('diagnostic_plan_sha256')} != current {plan_sha256}"
            )
        if existing_binding.get("git_commit") != expected_commit:
            raise DiagnosticRunAuditError(
                f"Existing run binding git commit {existing_binding.get('git_commit')} != current {expected_commit}"
            )
    else:
        binding_path.write_text(json.dumps(run_binding, indent=2), encoding="utf-8")

    # Load existing ledger attempts if resume
    completed_arms: set[str] = set()
    total_attempts_recorded = 0
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                arm_id = record.get("arm_id")
                if arm_id:
                    completed_arms.add(arm_id)
                total_attempts_recorded += 1

    if not resume and completed_arms:
        raise DiagnosticRunAuditError(
            f"Run directory {out_dir} already contains {len(completed_arms)} attempts. Use --resume to continue."
        )

    # Flatten all 6 planned arms
    planned_arms: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for cand in plan_data["diagnostic_candidates"]:
        arms = cand["arms"]
        planned_arms.append((cand, "arm_a_full_canvas", arms["arm_a_full_canvas"]))
        planned_arms.append((cand, "arm_b_local_crop", arms["arm_b_local_crop"]))

    if len(planned_arms) != DIAGNOSTIC_BUDGET_LIMIT:
        raise DiagnosticBudgetExceededError(
            f"Planned arms count {len(planned_arms)} != {DIAGNOSTIC_BUDGET_LIMIT}"
        )

    # Prepare engine loader
    active_engine: Any = None
    current_tool_key: str | None = None

    def get_engine(tool_key: str) -> Any:
        nonlocal active_engine, current_tool_key
        if active_engine is not None and current_tool_key == tool_key:
            return active_engine
        if active_engine is not None and hasattr(active_engine, "unload"):
            active_engine.unload()

        if engine_provider is not None:
            active_engine = engine_provider(tool_key)
        elif allow_mock:
            active_engine = DiagnosticMockEngine(tool_key)
        else:
            active_engine = DiagnosticDiffusersEngine(tool_key)
        current_tool_key = tool_key
        return active_engine

    results: list[dict[str, Any]] = []

    for cand, arm_key, arm_cfg in planned_arms:
        arm_id = arm_cfg["arm_id"]
        cid = cand["candidate_id"]

        if arm_id in completed_arms:
            print(f"[Diagnostic Resume] Arm {arm_id} already consumed; skipping.")
            continue

        if total_attempts_recorded >= DIAGNOSTIC_BUDGET_LIMIT:
            raise DiagnosticBudgetExceededError(
                f"Attempt budget of {DIAGNOSTIC_BUDGET_LIMIT} has been exhausted; cannot execute {arm_id}."
            )

        print(f"\n>>> [Diagnostic Attempt {total_attempts_recorded + 1}/{DIAGNOSTIC_BUDGET_LIMIT}] Executing {arm_id} ({arm_cfg['method']})...")
        start_time = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        total_attempts_recorded += 1

        # Record attempt start immediately to durable ledger
        start_record = {
            "attempt_index": total_attempts_recorded,
            "attempt_id": f"ATTEMPT_{total_attempts_recorded:02d}_{arm_id}",
            "arm_id": arm_id,
            "candidate_id": cid,
            "method": arm_cfg["method"],
            "status": "STARTED",
            "started_at_utc": started_utc,
        }
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(start_record) + "\n")

        auth_path = verified_inputs[cid]["authentic"]
        mask_path = verified_inputs[cid]["mask"]
        auth_img = Image.open(auth_path).convert("RGB")
        canvas_mask = Image.open(mask_path).convert("L")

        prompt = cand["prompt"]
        seed = cand["generation_seed"]
        tool_key = cand["tool_key"]
        engine = get_engine(tool_key)

        status = "FAILED"
        err_msg: str | None = None
        inside_l1 = 0.0
        outside_l1 = 0.0
        raw_gen_img: Image.Image | None = None
        composite_img: Image.Image | None = None

        try:
            if arm_cfg["method"] == "full_canvas_inference":
                # Arm A: Full Canvas 512x512
                raw_gen_img = engine.inpaint(
                    authentic_img=auth_img,
                    mask_img=canvas_mask,
                    prompt=prompt,
                    seed=seed,
                    height=512,
                    width=512,
                )
                composite_img = Image.composite(raw_gen_img, auth_img, canvas_mask)

            elif arm_cfg["method"] == "local_crop_padded_inference":
                # Arm B: Local Crop with padding, resized to model native resolution
                crop_bbox = arm_cfg["crop_bbox_xyxy"]
                crop_auth = auth_img.crop(crop_bbox)
                crop_mask = canvas_mask.crop(crop_bbox)

                inf_w, inf_h = arm_cfg["inference_resolution"]
                # Resize RGB via Lanczos, mask via Nearest
                scaled_auth = crop_auth.resize((inf_w, inf_h), resample=Image.Resampling.LANCZOS)
                scaled_mask = crop_mask.resize((inf_w, inf_h), resample=Image.Resampling.NEAREST)

                # Generate at native resolution
                raw_gen_img = engine.inpaint(
                    authentic_img=scaled_auth,
                    mask_img=scaled_mask,
                    prompt=prompt,
                    seed=seed,
                    height=inf_h,
                    width=inf_w,
                )

                # Downscale back to crop dimensions via Lanczos
                crop_w = crop_bbox[2] - crop_bbox[0]
                crop_h = crop_bbox[3] - crop_bbox[1]
                downscaled_gen = raw_gen_img.resize((crop_w, crop_h), resample=Image.Resampling.LANCZOS)

                # Remap crop onto full canvas buffer
                canvas_buf = auth_img.copy()
                canvas_buf.paste(downscaled_gen, (crop_bbox[0], crop_bbox[1]))

                # Strictly composite with original registered 512x512 canvas mask
                composite_img = Image.composite(canvas_buf, auth_img, canvas_mask)
            else:
                raise ValueError(f"Unknown diagnostic arm method: {arm_cfg['method']}")

            # Assert bitwise invariance of outside pixels
            auth_arr = np.array(auth_img, dtype=np.float32)
            comp_arr = np.array(composite_img, dtype=np.float32)
            mask_arr = np.array(canvas_mask, dtype=np.uint8)

            diff = np.abs(comp_arr - auth_arr)
            outside_pixels = mask_arr == 0
            inside_pixels = mask_arr == 255

            max_outside_delta = float(np.max(diff[outside_pixels])) if np.any(outside_pixels) else 0.0
            outside_l1 = float(np.mean(diff[outside_pixels])) if np.any(outside_pixels) else 0.0
            inside_l1 = float(np.mean(diff[inside_pixels])) if np.any(inside_pixels) else 0.0

            if max_outside_delta > 0.0 or outside_l1 > 0.0:
                raise GenerationContractError(
                    f"Arm {arm_id} violated outside pixel invariance: max_delta={max_outside_delta}, mean_L1={outside_l1}"
                )

            # Save artifacts
            auth_out = images_dir / f"{arm_id}_auth.png"
            raw_gen_out = images_dir / f"{arm_id}_raw_gen.png"
            edit_out = images_dir / f"{arm_id}_edit.png"
            mask_out = masks_dir / f"{arm_id}_mask.png"

            auth_img.save(auth_out)
            raw_gen_img.save(raw_gen_out)
            composite_img.save(edit_out)
            canvas_mask.save(mask_out)

            status = "ACCEPTED"

        except Exception as e:
            err_msg = str(e)
            print(f"ERROR in {arm_id}: {e}")

        elapsed_sec = round(time.perf_counter() - start_time, 3)
        peak_vram = engine.peak_vram_bytes() if hasattr(engine, "peak_vram_bytes") else "NOT_MEASURED"

        finish_record = {
            "attempt_index": total_attempts_recorded,
            "attempt_id": f"ATTEMPT_{total_attempts_recorded:02d}_{arm_id}",
            "arm_id": arm_id,
            "candidate_id": cid,
            "method": arm_cfg["method"],
            "status": status,
            "started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_sec,
            "peak_vram_bytes": peak_vram,
            "inside_mean_l1": round(inside_l1, 4),
            "outside_mean_l1": round(outside_l1, 6),
            "error_message": err_msg,
            "human_content_qc_status": "PENDING",
            "runtime_versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pillow": Image.__version__,
            },
        }

        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(finish_record) + "\n")

        results.append(finish_record)
        print(f"Finished {arm_id}: status={status}, inside_L1={inside_l1:.2f}, outside_L1={outside_l1:.6f}, elapsed={elapsed_sec}s")

    if active_engine is not None and hasattr(active_engine, "unload"):
        active_engine.unload()

    # Write summary receipt
    receipt = {
        "run_id": out_dir.name,
        "mode": "diagnostic",
        "diagnostic_plan_sha256": plan_sha256,
        "git_commit": expected_commit,
        "total_attempts_executed": total_attempts_recorded,
        "total_budget_limit": DIAGNOSTIC_BUDGET_LIMIT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": results,
    }
    (out_dir / "diagnostic_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # Generate visual inspection contact sheet
    contact_sheet_path = out_dir / "diagnostic_contact_sheet.html"
    generate_diagnostic_contact_sheet(out_dir, plan_data, contact_sheet_path)

    return receipt


def _image_to_base64(path: Path) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def generate_diagnostic_contact_sheet(
    run_dir: Path | str,
    plan_data: dict[str, Any],
    output_html_path: Path | str,
) -> None:
    """Generate self-contained HTML contact sheet for human review of diagnostic outputs."""
    r_dir = Path(run_dir)
    images_dir = r_dir / "images"
    masks_dir = r_dir / "masks"
    ledger_path = r_dir / "attempt_ledger.jsonl"

    attempts: dict[str, dict[str, Any]] = {}
    if ledger_path.is_file():
        with ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") in ("ACCEPTED", "FAILED"):
                    attempts[rec["arm_id"]] = rec

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Phase 4C.7B Diagnostic Calibration Contact Sheet (6 Attempts)</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 24px; margin: 0; }",
        "h1 { color: #38bdf8; font-size: 24px; margin-bottom: 8px; }",
        ".banner { background: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; margin-bottom: 24px; font-size: 13px; line-height: 1.5; }",
        ".cand-section { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 20px; margin-bottom: 32px; }",
        ".cand-title { font-size: 18px; color: #60a5fa; font-weight: 600; margin-bottom: 6px; }",
        ".cand-desc { font-size: 13px; color: #94a3b8; margin-bottom: 16px; }",
        ".arms-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
        ".arm-card { background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 14px; }",
        ".arm-header { font-size: 14px; font-weight: 600; color: #f3f4f6; margin-bottom: 8px; display: flex; justify-content: space-between; }",
        ".badge-arm-a { background: #0369a1; color: #e0f2fe; padding: 2px 8px; border-radius: 4px; font-size: 11px; }",
        ".badge-arm-b { background: #15803d; color: #dcfce7; padding: 2px 8px; border-radius: 4px; font-size: 11px; }",
        ".images-row { display: flex; gap: 10px; margin-bottom: 12px; }",
        ".img-box { text-align: center; flex: 1; }",
        ".img-box img { width: 100%; max-width: 150px; height: auto; aspect-ratio: 1; border-radius: 4px; border: 1px solid #4b5563; object-fit: contain; background: #000; }",
        ".img-box span { display: block; font-size: 11px; color: #9ca3af; margin-top: 4px; }",
        ".metrics-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }",
        ".metrics-table td { padding: 4px 6px; border-bottom: 1px solid #374151; }",
        ".metrics-table td:first-child { color: #9ca3af; }",
        ".qc-pending { color: #f59e0b; font-weight: 600; }",
        "</style></head><body>",
        "<h1>Phase 4C.7B Diagnostic Calibration Contact Sheet</h1>",
        "<div class='banner'>",
        "<strong>Diagnostic Isolation Notice:</strong> Exactly 6 attempts (3 candidates &times; 2 arms). "
        "Arm A evaluates full-canvas 512&times;512 inference; Arm B evaluates local-crop padded inference. "
        "All candidates use locked seeds, prompts, and guidance from the pilot ledger. "
        "Human Content QC status is <strong>strictly PENDING</strong> for all arms. "
        "Zero detector models evaluated. Results do NOT enter official cohort.",
        "</div>",
    ]

    for cand in plan_data.get("diagnostic_candidates", []):
        cid = cand["candidate_id"]
        cname = cand["common_name"]
        prompt = cand["prompt"]
        seed = cand["generation_seed"]

        html.append(f"<div class='cand-section'>")
        html.append(f"<div class='cand-title'>{cid} &mdash; {cname}</div>")
        html.append(f"<div class='cand-desc'>Prompt: <em>\"{prompt}\"</em> | Seed: <code>{seed}</code></div>")
        html.append("<div class='arms-grid'>")

        for arm_key, badge_cls, label in (
            ("arm_a_full_canvas", "badge-arm-a", "Arm A (Full Canvas 512x512)"),
            ("arm_b_local_crop", "badge-arm-b", "Arm B (Local Crop with Padding)"),
        ):
            arm_cfg = cand["arms"][arm_key]
            arm_id = arm_cfg["arm_id"]
            rec = attempts.get(arm_id, {})

            auth_b64 = _image_to_base64(images_dir / f"{arm_id}_auth.png")
            mask_b64 = _image_to_base64(masks_dir / f"{arm_id}_mask.png")
            raw_b64 = _image_to_base64(images_dir / f"{arm_id}_raw_gen.png")
            edit_b64 = _image_to_base64(images_dir / f"{arm_id}_edit.png")

            html.append("<div class='arm-card'>")
            html.append(f"<div class='arm-header'><span>{arm_id}</span><span class='{badge_cls}'>{label}</span></div>")

            html.append("<div class='images-row'>")
            if auth_b64:
                html.append(f"<div class='img-box'><img src='{auth_b64}' alt='Auth'><span>Authentic</span></div>")
            if mask_b64:
                html.append(f"<div class='img-box'><img src='{mask_b64}' alt='Mask'><span>Mask</span></div>")
            if raw_b64:
                html.append(f"<div class='img-box'><img src='{raw_b64}' alt='Raw'><span>Raw Generated</span></div>")
            if edit_b64:
                html.append(f"<div class='img-box'><img src='{edit_b64}' alt='Edit'><span>Final Composite</span></div>")
            html.append("</div>")

            # Metrics table
            status = rec.get("status", "NOT_RUN")
            inside_l1 = rec.get("inside_mean_l1", "N/A")
            outside_l1 = rec.get("outside_mean_l1", "N/A")
            elapsed = rec.get("elapsed_seconds", "N/A")
            peak_vram = rec.get("peak_vram_bytes", "N/A")
            if isinstance(peak_vram, int):
                peak_vram = f"{peak_vram / 1024**3:.2f} GiB"

            scale_factor = arm_cfg.get("geometric_area_scale_factor", 1.0)
            latent_px = arm_cfg.get("estimated_latent_mask_pixels", "N/A")

            html.append("<table class='metrics-table'>")
            html.append(f"<tr><td>Execution Status</td><td><strong>{status}</strong></td></tr>")
            html.append(f"<tr><td>Inside Mean L1</td><td>{inside_l1}</td></tr>")
            html.append(f"<tr><td>Outside Mean L1 (Invariance)</td><td><code>{outside_l1}</code></td></tr>")
            html.append(f"<tr><td>Area Scale Factor</td><td>{scale_factor}&times;</td></tr>")
            html.append(f"<tr><td>Estimated Latent Mask Px</td><td>~{latent_px}</td></tr>")
            html.append(f"<tr><td>Elapsed Time</td><td>{elapsed}s</td></tr>")
            html.append(f"<tr><td>Peak VRAM</td><td>{peak_vram}</td></tr>")
            html.append("<tr><td>Human Content QC</td><td><span class='qc-pending'>PENDING</span></td></tr>")
            html.append("</table>")

            html.append("</div>")  # arm-card

        html.append("</div>")  # arms-grid
        html.append("</div>")  # cand-section

    html.append("</body></html>")
    out_p = Path(output_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(html), encoding="utf-8")
    print(f"Generated diagnostic contact sheet at: {out_p}")


def audit_diagnostic_run(
    run_dir: Path | str,
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    """Audit a completed or in-progress diagnostic run directory."""
    r_dir = Path(run_dir)
    binding_path = r_dir / "run_binding.json"
    ledger_path = r_dir / "attempt_ledger.jsonl"
    receipt_path = r_dir / "diagnostic_receipt.json"

    if not binding_path.is_file():
        raise DiagnosticRunAuditError(f"Missing run_binding.json in {r_dir}")
    if not ledger_path.is_file():
        raise DiagnosticRunAuditError(f"Missing attempt_ledger.jsonl in {r_dir}")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    for k in ("run_id", "mode", "git_commit", "diagnostic_plan_sha256"):
        if k in expected_binding and binding.get(k) != expected_binding[k]:
            raise DiagnosticRunAuditError(
                f"Run binding mismatch on key '{k}': found {binding.get(k)}, expected {expected_binding[k]}"
            )

    records: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)

    completed = [r for r in records if r.get("status") in ("ACCEPTED", "FAILED")]
    accepted = [r for r in completed if r.get("status") == "ACCEPTED"]

    # Verify outside L1 is exactly 0.000000 on all accepted
    for a in accepted:
        if a.get("outside_mean_l1", 0.0) != 0.0:
            raise DiagnosticRunAuditError(
                f"Attempt {a.get('arm_id')} has non-zero outside L1: {a.get('outside_mean_l1')}"
            )

    return {
        "status": "PASS",
        "run_id": binding["run_id"],
        "git_commit": binding["git_commit"],
        "plan_sha256": binding["diagnostic_plan_sha256"],
        "total_attempts": len(completed),
        "accepted_attempts": len(accepted),
        "records": records,
    }
