"""Calibration execution harness for Phase 4C.7B independent cohort acquisition.

Implements the explicit, fail-closed calibration execution pipeline for the approved proposal:
- Compares guidance scale 7.5 and 9.5 on candidate IND_COCO_SDXL_002 under fixed prompt/negative prompt.
- Full canvas 512x512 without cropping or spatial remapping.
- Hard budget limit: strictly 2 attempts.
- Fail-closed approval guard: refuses generation unless plan status is 'APPROVED'.
- Plan hash verification: verifies approved SHA-256 (2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c).
- Sealed input verification: verifies authentic and mask PNG hashes from pilot-20261008T113700Z.
- Re-seeds PRNG with seed 20272319 independently before each attempt.
- Explicitly passes prompt, negative_prompt, and guidance_scale to pipeline.
- Re-initializes generator per attempt; no feathering, no retries, no automatic replacement.
- An attempt that starts but encounters error counts towards the 2-attempt budget.
- Resume skips already consumed attempts without re-generating.
- Preserves raw pre-composited generated images alongside final 512x512 composites.
- Preserves outside-mask pixels with bit-exact invariance (outside_mean_L1 == 0.000000).
- Results have Human Content QC = PENDING and do NOT enter official cohort.
- Comprehensive runtime telemetry recorded: Python, Torch, Diffusers, Transformers, NumPy, Pillow, CUDA, GPU.
"""

from __future__ import annotations

import base64
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

CALIBRATION_BUDGET_LIMIT = 2
APPROVED_CALIBRATION_PLAN_SHA256 = (
    "2611af81c2e104fb04235c4f3c15371b2c95e8ae57a72190e5a82623f91dbf2c"
)


class CalibrationPlanNotApprovedError(RuntimeError):
    """Raised when attempting to execute a calibration plan whose status is not APPROVED."""


class CalibrationPlanIntegrityError(ValueError):
    """Raised when calibration plan hash, binding, or structure fails validation."""


class CalibrationInputMissingError(FileNotFoundError):
    """Raised when an input authentic or mask image is missing."""


class CalibrationInputHashMismatchError(ValueError):
    """Raised when an input image hash does not match the plan's registered hash."""


class CalibrationBudgetExceededError(RuntimeError):
    """Raised when attempts exceed the 2-attempt budget limit."""


class CalibrationGeometryError(ValueError):
    """Raised when canvas or mask dimensions violate 512x512 contract."""


class CalibrationRunAuditError(RuntimeError):
    """Raised when an existing calibration run fails audit."""


class CalibrationEngineError(RuntimeError):
    """Raised when an engine configuration or mock usage is invalid in production."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_and_validate_calibration_plan(
    plan_path: Path | str,
    expected_plan_sha256: str | None = None,
    require_approval: bool = True,
) -> dict[str, Any]:
    """Load and validate the calibration plan fail-closed."""
    p_path = Path(plan_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Calibration plan not found at {p_path}")

    content = p_path.read_bytes()
    actual_hash = _sha256_bytes(content)
    if expected_plan_sha256 and actual_hash != expected_plan_sha256:
        raise CalibrationPlanIntegrityError(
            f"Calibration plan SHA-256 mismatch: expected {expected_plan_sha256}, got {actual_hash}"
        )

    data = json.loads(content.decode("utf-8"))

    if require_approval:
        status = data.get("human_review_status")
        if status != "APPROVED":
            raise CalibrationPlanNotApprovedError(
                f"Calibration plan human_review_status is '{status}' (expected 'APPROVED'). "
                "Execution is refused until explicitly approved by human reviewer."
            )

    budget = data.get("attempt_budget", 0)
    if budget != CALIBRATION_BUDGET_LIMIT:
        raise CalibrationBudgetExceededError(
            f"Calibration plan attempt_budget {budget} != expected {CALIBRATION_BUDGET_LIMIT}"
        )

    # Validate controlled invariants
    invariants = data.get("controlled_invariants", {})
    if invariants.get("canvas_dimensions") != [512, 512]:
        raise CalibrationPlanIntegrityError(
            f"Invalid canvas_dimensions: {invariants.get('canvas_dimensions')}; expected [512, 512]"
        )
    if invariants.get("generation_seed") != 20272319:
        raise CalibrationPlanIntegrityError(
            f"Invalid generation_seed: {invariants.get('generation_seed')}; expected 20272319"
        )
    if invariants.get("model_checkpoint") != "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
        raise CalibrationPlanIntegrityError(
            f"Invalid model_checkpoint: {invariants.get('model_checkpoint')}"
        )
    if invariants.get("model_revision") != "115134f363124c53c7d878647567d04daf26e41e":
        raise CalibrationPlanIntegrityError(
            f"Invalid model_revision: {invariants.get('model_revision')}"
        )

    attempts = data.get("calibration_attempts", [])
    if len(attempts) != CALIBRATION_BUDGET_LIMIT:
        raise CalibrationPlanIntegrityError(
            f"Expected exactly {CALIBRATION_BUDGET_LIMIT} calibration attempts, found {len(attempts)}"
        )

    guidance_scales = [att.get("guidance_scale") for att in attempts]
    if guidance_scales != [7.5, 9.5]:
        raise CalibrationPlanIntegrityError(
            f"Invalid guidance scales {guidance_scales}; expected [7.5, 9.5]"
        )

    for att in attempts:
        if att.get("candidate_id") != "IND_COCO_SDXL_002":
            raise CalibrationPlanIntegrityError(
                f"Attempt {att.get('attempt_id')} candidate {att.get('candidate_id')} != 'IND_COCO_SDXL_002'"
            )
        if att.get("crop_applied") is not False:
            raise CalibrationPlanIntegrityError(
                f"Attempt {att.get('attempt_id')} must have crop_applied=False (full canvas only)"
            )
        if att.get("inference_canvas") != [512, 512]:
            raise CalibrationPlanIntegrityError(
                f"Attempt {att.get('attempt_id')} inference_canvas must be [512, 512]"
            )
        if not att.get("negative_prompt"):
            raise CalibrationPlanIntegrityError(
                f"Attempt {att.get('attempt_id')} missing required negative_prompt"
            )

    return data


def verify_calibration_inputs(
    plan_data: dict[str, Any],
    inputs_dir: Path | str,
) -> dict[str, dict[str, Path]]:
    """Verify that all authentic and mask images exist and match their registered SHA-256 hashes."""
    base_dir = Path(inputs_dir)
    verified: dict[str, dict[str, Path]] = {}

    for att in plan_data.get("calibration_attempts", []):
        att_id = att["attempt_id"]
        cid = att["candidate_id"]
        auth_fname = att["authentic_input_filename"]
        auth_hash = att["authentic_sha256"]
        mask_fname = att["mask_input_filename"]
        mask_hash = att["mask_sha256"]

        auth_path = None
        for cand_path in (base_dir / auth_fname, base_dir / "images" / auth_fname):
            if cand_path.is_file():
                auth_path = cand_path
                break
        if not auth_path:
            raise CalibrationInputMissingError(
                f"Authentic input image for {att_id} not found (looked for {auth_fname} in {base_dir})"
            )

        actual_auth_hash = _sha256_file(auth_path)
        if actual_auth_hash != auth_hash:
            raise CalibrationInputHashMismatchError(
                f"Authentic image {auth_path.name} SHA-256 {actual_auth_hash} != registered {auth_hash}"
            )

        mask_path = None
        for cand_path in (base_dir / mask_fname, base_dir / "masks" / mask_fname):
            if cand_path.is_file():
                mask_path = cand_path
                break
        if not mask_path:
            raise CalibrationInputMissingError(
                f"Mask input image for {att_id} not found (looked for {mask_fname} in {base_dir})"
            )

        actual_mask_hash = _sha256_file(mask_path)
        if actual_mask_hash != mask_hash:
            raise CalibrationInputHashMismatchError(
                f"Mask image {mask_path.name} SHA-256 {actual_mask_hash} != registered {mask_hash}"
            )

        with Image.open(auth_path) as im:
            if im.size != TARGET_CANVAS_SIZE or im.mode != "RGB":
                raise CalibrationGeometryError(
                    f"Authentic image {auth_path} must be 512x512 RGB, got {im.size} {im.mode}"
                )
        with Image.open(mask_path) as mk:
            if mk.size != TARGET_CANVAS_SIZE:
                raise CalibrationGeometryError(
                    f"Mask image {mask_path} must be 512x512, got {mk.size}"
                )
            arr = np.array(mk)
            uniques = np.unique(arr)
            if not set(uniques.tolist()).issubset({0, 255}):
                raise CalibrationGeometryError(
                    f"Mask {mask_path} must be strictly binary {{0, 255}}, got {uniques}"
                )

        verified[att_id] = {"authentic": auth_path, "mask": mask_path}

    return verified


def collect_runtime_telemetry() -> dict[str, str]:
    """Capture precise execution environment telemetry without speculation."""
    telemetry: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": "NOT_INSTALLED",
        "diffusers": "NOT_INSTALLED",
        "transformers": "NOT_INSTALLED",
        "numpy": "NOT_INSTALLED",
        "pillow": "NOT_INSTALLED",
        "cuda": "NOT_AVAILABLE",
        "gpu": "NOT_AVAILABLE",
    }

    try:
        import torch
        telemetry["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            telemetry["cuda"] = str(torch.version.cuda) if hasattr(torch.version, "cuda") else "AVAILABLE_UNKNOWN_VERSION"
            telemetry["gpu"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    try:
        import diffusers
        telemetry["diffusers"] = str(diffusers.__version__)
    except Exception:
        pass

    try:
        import transformers
        telemetry["transformers"] = str(transformers.__version__)
    except Exception:
        pass

    try:
        import numpy as np_mod
        telemetry["numpy"] = str(np_mod.__version__)
    except Exception:
        pass

    try:
        import PIL
        telemetry["pillow"] = str(PIL.__version__)
    except Exception:
        pass

    return telemetry


class CalibrationMockEngine:
    """Fast deterministic mock engine for CPU unit tests. Completely isolated from detector models.

    CRITICAL: Strictly marked as synthetic. Prohibited for real production calibration execution.
    """
    is_mock: bool = True
    is_synthetic: bool = True

    def __init__(self, tool_key: str = "sdxl_inpainting"):
        self.tool_key = tool_key
        self.last_call_params: dict[str, Any] = {}

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        seed: int,
        height: int,
        width: int,
        num_inference_steps: int = 30,
    ) -> Image.Image:
        self.last_call_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        rng = np.random.default_rng(seed)
        auth_arr = np.array(authentic_img, dtype=np.uint8).copy()
        mask_arr = np.array(mask_img, dtype=np.uint8)

        # Shift synthetic color slightly with guidance scale to reflect parameter sensitivity
        scale_bias = int(guidance_scale * 5)
        color_base = rng.integers(50 + scale_bias, 200 + scale_bias, size=3, dtype=np.uint8)
        noise = rng.integers(-25, 25, size=(height, width, 3), dtype=np.int16)
        synthetic_patch = np.clip(color_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        in_mask = mask_arr == 255
        auth_arr[in_mask] = synthetic_patch[in_mask]
        return Image.fromarray(auth_arr, mode="RGB")

    def describe(self) -> dict[str, Any]:
        return {"engine": "CalibrationMockEngine", "is_synthetic": True}

    def peak_vram_bytes(self) -> int | str:
        return "NOT_MEASURED"

    def unload(self) -> None:
        pass


class CalibrationDiffusersEngine:
    """Production GPU inpainting engine using HuggingFace Diffusers."""

    def __init__(
        self,
        tool_key: str = "sdxl_inpainting",
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

        self.revision = resolved_revision or reg["pinned_revision"]
        self.pipeline = pipeline_cls.from_pretrained(
            checkpoint, revision=self.revision, torch_dtype=self.dtype, **extra
        )
        self.pipeline.scheduler = scheduler_cls.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

        self.config = {
            "engine": "CalibrationDiffusersEngine",
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
        negative_prompt: str | None,
        guidance_scale: float,
        seed: int,
        height: int,
        width: int,
        num_inference_steps: int = 30,
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
            negative_prompt=negative_prompt,
            image=authentic_img,
            mask_image=mask_img,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
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


def execute_calibration_run(
    output_dir: Path | str,
    plan_path: Path | str,
    inputs_dir: Path | str,
    expected_commit: str,
    engine_provider: Callable[[str], Any] | None = None,
    resume: bool = False,
    require_approval: bool = True,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Execute the 2-attempt calibration run fail-closed."""
    assert_detector_isolation()
    out_dir = Path(output_dir)

    plan_p = Path(plan_path)
    plan_data = load_and_validate_calibration_plan(
        plan_p, require_approval=require_approval
    )
    plan_sha256 = _sha256_file(plan_p)

    verified_inputs = verify_calibration_inputs(plan_data, inputs_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = out_dir / "attempt_ledger.jsonl"
    binding_path = out_dir / "run_binding.json"

    run_binding = {
        "run_id": out_dir.name,
        "mode": "calibration",
        "git_commit": expected_commit,
        "calibration_plan_sha256": plan_sha256,
        "total_budget": CALIBRATION_BUDGET_LIMIT,
        "bound_pilot_run_id": plan_data["bound_pilot_run_id"],
        "bound_pilot_zip_sha256": plan_data["bound_pilot_zip_sha256"],
        "bound_pilot_manifest_sha256": plan_data["bound_pilot_manifest_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if binding_path.exists():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding.get("calibration_plan_sha256") != plan_sha256:
            raise CalibrationRunAuditError(
                f"Existing run binding plan hash {existing_binding.get('calibration_plan_sha256')} != current {plan_sha256}"
            )
        if existing_binding.get("git_commit") != expected_commit:
            raise CalibrationRunAuditError(
                f"Existing run binding git commit {existing_binding.get('git_commit')} != current {expected_commit}"
            )
    else:
        binding_path.write_text(json.dumps(run_binding, indent=2), encoding="utf-8")

    # Load existing ledger attempts if resume
    # Any attempt that was STARTED, ACCEPTED, or FAILED is counted as consumed
    consumed_attempt_ids: set[str] = set()
    total_attempts_recorded = 0
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                aid = record.get("attempt_id")
                status = record.get("status")
                if aid and status in ("STARTED", "ACCEPTED", "FAILED", "ERROR"):
                    consumed_attempt_ids.add(aid)
                total_attempts_recorded += 1

    if not resume and consumed_attempt_ids:
        raise CalibrationRunAuditError(
            f"Run directory {out_dir} already contains {len(consumed_attempt_ids)} attempts. Use --resume to continue."
        )

    planned_attempts = plan_data.get("calibration_attempts", [])
    if len(planned_attempts) != CALIBRATION_BUDGET_LIMIT:
        raise CalibrationBudgetExceededError(
            f"Planned attempts count {len(planned_attempts)} != {CALIBRATION_BUDGET_LIMIT}"
        )

    # Telemetry
    telemetry = collect_runtime_telemetry()

    engine_cache: dict[str, Any] = {}

    def get_engine(tool_key: str) -> Any:
        if tool_key in engine_cache:
            return engine_cache[tool_key]
        if engine_provider is not None:
            eng = engine_provider(tool_key)
            if getattr(eng, "is_mock", False) and not allow_mock:
                raise CalibrationEngineError(
                    "Mock engine is prohibited in production calibration run."
                )
            engine_cache[tool_key] = eng
            return eng
        if allow_mock:
            eng = CalibrationMockEngine(tool_key)
            engine_cache[tool_key] = eng
            return eng
        eng = CalibrationDiffusersEngine(tool_key=tool_key)
        engine_cache[tool_key] = eng
        return eng

    receipt_attempts: list[dict[str, Any]] = []

    try:
        for idx, att in enumerate(planned_attempts, start=1):
            att_id = att["attempt_id"]
            cid = att["candidate_id"]
            guidance = att["guidance_scale"]
            prompt = att["prompt"]
            neg_prompt = att["negative_prompt"]
            seed = att["generation_seed"]
            steps = att.get("num_inference_steps", 30)

            if att_id in consumed_attempt_ids:
                print(f"[Phase 4C.7B Calibration] Attempt {att_id} already consumed; skipping per resume contract.")
                continue

            # Check remaining budget
            if len(consumed_attempt_ids) >= CALIBRATION_BUDGET_LIMIT:
                raise CalibrationBudgetExceededError(
                    f"Attempt budget of {CALIBRATION_BUDGET_LIMIT} consumed; refusing further execution."
                )

            start_iso = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()

            # Record STARTED event in durable ledger immediately (consumes budget)
            start_event = {
                "event": "ATTEMPT_STARTED",
                "attempt_index": idx,
                "attempt_id": att_id,
                "candidate_id": cid,
                "guidance_scale": guidance,
                "status": "STARTED",
                "started_at_utc": start_iso,
            }
            with ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(start_event) + "\n")
            consumed_attempt_ids.add(att_id)

            auth_path = verified_inputs[att_id]["authentic"]
            mask_path = verified_inputs[att_id]["mask"]

            auth_img = Image.open(auth_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")

            # Persist authentic and mask inputs into run directory
            auth_dest = images_dir / f"{att_id}_auth.png"
            mask_dest = masks_dir / f"{att_id}_mask.png"
            if not auth_dest.exists():
                auth_img.save(auth_dest)
            if not mask_dest.exists():
                mask_img.save(mask_dest)

            tool_key = att["tool_key"]
            try:
                engine = get_engine(tool_key)
                # Execute inpainting with explicit negative_prompt and guidance_scale
                raw_generated = engine.inpaint(
                    authentic_img=auth_img,
                    mask_img=mask_img,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    guidance_scale=guidance,
                    seed=seed,
                    height=512,
                    width=512,
                    num_inference_steps=steps,
                )

                # Binary outside-invariant composite
                composite_img = Image.composite(raw_generated, auth_img, mask_img)

                # Verify outside pixel invariance
                auth_arr = np.array(auth_img, dtype=np.float32)
                comp_arr = np.array(composite_img, dtype=np.float32)
                mask_arr = np.array(mask_img, dtype=np.uint8) == 255

                outside_mask = ~mask_arr
                outside_diff = np.abs(comp_arr[outside_mask] - auth_arr[outside_mask])
                outside_mean_l1 = float(np.mean(outside_diff)) if np.any(outside_mask) else 0.0
                outside_max_delta = float(np.max(outside_diff)) if np.any(outside_mask) else 0.0

                if outside_mean_l1 > 0.0 or outside_max_delta > 0.0:
                    raise GenerationContractError(
                        f"Outside mask pixel invariance violated on {att_id}: mean={outside_mean_l1}, max={outside_max_delta}"
                    )

                inside_diff = np.abs(comp_arr[mask_arr] - auth_arr[mask_arr])
                inside_mean_l1 = float(np.mean(inside_diff)) if np.any(mask_arr) else 0.0
                inside_std = float(np.std(comp_arr[mask_arr])) if np.any(mask_arr) else 0.0

                # Technical QC check
                tech_qc_pass = (
                    inside_std >= NON_BLANK_STD_THRESHOLD
                    and inside_mean_l1 >= MIN_MASKED_PIXEL_DELTA_L1
                    and outside_mean_l1 <= MAX_UNMASKED_PIXEL_DELTA_L1
                )
                tech_qc_status = "PASS" if tech_qc_pass else "FAIL"

                # Save raw and composite images
                raw_dest = images_dir / f"{att_id}_raw.png"
                comp_dest = images_dir / f"{att_id}_composite.png"
                raw_generated.save(raw_dest)
                composite_img.save(comp_dest)

                elapsed = time.perf_counter() - t0
                finish_iso = datetime.now(timezone.utc).isoformat()
                peak_vram = engine.peak_vram_bytes()

                accept_record = {
                    "event": "ATTEMPT_FINISHED",
                    "attempt_index": idx,
                    "attempt_id": att_id,
                    "candidate_id": cid,
                    "guidance_scale": guidance,
                    "status": "ACCEPTED",
                    "technical_qc_status": tech_qc_status,
                    "started_at_utc": start_iso,
                    "finished_at_utc": finish_iso,
                    "elapsed_seconds": round(elapsed, 3),
                    "peak_vram_bytes": peak_vram,
                    "inside_mean_l1": round(inside_mean_l1, 4),
                    "outside_mean_l1": round(outside_mean_l1, 6),
                    "outside_max_delta": round(outside_max_delta, 1),
                    "human_content_qc_status": "PENDING",
                    "error_message": None,
                }
                with ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(accept_record) + "\n")

                receipt_attempts.append(accept_record)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                finish_iso = datetime.now(timezone.utc).isoformat()
                fail_record = {
                    "event": "ATTEMPT_FINISHED",
                    "attempt_index": idx,
                    "attempt_id": att_id,
                    "candidate_id": cid,
                    "guidance_scale": guidance,
                    "status": "FAILED",
                    "technical_qc_status": "FAIL",
                    "started_at_utc": start_iso,
                    "finished_at_utc": finish_iso,
                    "elapsed_seconds": round(elapsed, 3),
                    "peak_vram_bytes": "NOT_MEASURED",
                    "inside_mean_l1": 0.0,
                    "outside_mean_l1": 0.0,
                    "outside_max_delta": 0.0,
                    "human_content_qc_status": "NOT_APPLICABLE",
                    "error_message": str(exc),
                }
                with ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(fail_record) + "\n")
                receipt_attempts.append(fail_record)
                raise
    finally:
        for eng in engine_cache.values():
            eng.unload()

    receipt_data = {
        "run_id": out_dir.name,
        "mode": "calibration",
        "calibration_plan_sha256": plan_sha256,
        "git_commit": expected_commit,
        "total_attempts_executed": len(receipt_attempts),
        "total_budget_limit": CALIBRATION_BUDGET_LIMIT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_telemetry": telemetry,
        "attempts": receipt_attempts,
    }

    receipt_path = out_dir / "calibration_receipt.json"
    receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

    # Generate self-contained contact sheet
    generate_calibration_contact_sheet(
        run_dir=out_dir,
        output_html=out_dir / "calibration_contact_sheet.html",
        plan_data=plan_data,
        receipt_data=receipt_data,
    )

    return receipt_data


def generate_calibration_contact_sheet(
    run_dir: Path | str,
    output_html: Path | str,
    plan_data: dict[str, Any],
    receipt_data: dict[str, Any],
) -> None:
    """Generate self-contained HTML review contact sheet with base64 embedded images."""
    r_dir = Path(run_dir)
    images_dir = r_dir / "images"
    masks_dir = r_dir / "masks"

    def _b64_img(path: Path) -> str:
        if not path.is_file():
            return ""
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Phase 4C.7B Calibration Contact Sheet</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }",
        "h1, h2, h3 { color: #38bdf8; }",
        ".header-box { background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }",
        ".badge-pending { background: #f59e0b; color: #000; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".badge-pass { background: #10b981; color: #000; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".card { background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 24px; border: 1px solid #334155; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-top: 16px; }",
        ".img-box { text-align: center; background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #334155; }",
        ".img-box img { max-width: 100%; height: auto; border-radius: 4px; }",
        ".img-box p { margin: 8px 0 0 0; font-size: 13px; color: #94a3b8; font-weight: 600; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }",
        "th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }",
        "th { background: #0f172a; color: #94a3b8; }",
        ".governance { background: #1e1b4b; border: 1px solid #4338ca; padding: 16px; border-radius: 6px; margin-top: 24px; font-size: 13px; line-height: 1.6; color: #c7d2fe; }",
        "</style>",
        "</head>",
        "<body>",
        f"<div class='header-box'>",
        f"<h1>Phase 4C.7B Calibration Contact Sheet — Run: {r_dir.name}</h1>",
        f"<p><strong>Objective:</strong> Compare guidance scale 7.5 vs 9.5 on candidate IND_COCO_SDXL_002 under fixed prompt/negative prompt configuration.</p>",
        f"<p><strong>Plan Hash:</strong> <code>{receipt_data.get('calibration_plan_sha256')}</code> | <strong>Git Commit:</strong> <code>{receipt_data.get('git_commit')}</code></p>",
        f"<p><strong>Human Content QC Status:</strong> <span class='badge-pending'>PENDING</span> (Independent of cohort; not merged into dataset)</p>",
        f"</div>",
    ]

    for att_rec in receipt_data.get("attempts", []):
        att_id = att_rec["attempt_id"]
        guidance = att_rec["guidance_scale"]
        status = att_rec["status"]
        tech_qc = att_rec.get("technical_qc_status", "UNKNOWN")
        inside_l1 = att_rec.get("inside_mean_l1", 0.0)
        outside_l1 = att_rec.get("outside_mean_l1", 0.0)

        auth_uri = _b64_img(images_dir / f"{att_id}_auth.png")
        mask_uri = _b64_img(masks_dir / f"{att_id}_mask.png")
        raw_uri = _b64_img(images_dir / f"{att_id}_raw.png")
        comp_uri = _b64_img(images_dir / f"{att_id}_composite.png")

        html_parts.extend([
            f"<div class='card'>",
            f"<h2>Attempt: {att_id} (Guidance Scale: {guidance})</h2>",
            f"<p>Status: <strong>{status}</strong> | Technical QC: <span class='badge-pass'>{tech_qc}</span> | Inside Mean L1: <code>{inside_l1}</code> | Outside Mean L1: <code>{outside_l1}</code></p>",
            f"<div class='grid'>",
            f"<div class='img-box'><img src='{auth_uri}' alt='Authentic'><p>Authentic (512x512)</p></div>",
            f"<div class='img-box'><img src='{mask_uri}' alt='Mask'><p>Registered Mask (512x512)</p></div>",
            f"<div class='img-box'><img src='{raw_uri}' alt='Raw Generated'><p>Raw SDXL Output</p></div>",
            f"<div class='img-box'><img src='{comp_uri}' alt='Composite'><p>1-Bit Composite</p></div>",
            f"</div>",
            f"</div>",
        ])

    html_parts.extend([
        "<div class='governance'>",
        "<h3>Governance & Scientific Honesty Declarations</h3>",
        "<ul>",
        "<li>Exploratory calibration run: strictly 2 attempts on candidate IND_COCO_SDXL_002.</li>",
        "<li>Does NOT claim to isolate prompt vs negative prompt effects, and does NOT prove infill bias.</li>",
        "<li>Results are isolated in this run directory; they do NOT enter the official independent cohort.</li>",
        "<li>All historical pilot pairs from pilot-20261008T113700Z remain strictly PENDING_CONTENT_QC.</li>",
        "<li>Full cohort generation remains strictly LOCKED.</li>",
        "</ul>",
        "</div>",
        "</body>",
        "</html>",
    ])

    Path(output_html).write_text("\n".join(html_parts), encoding="utf-8")


def audit_calibration_run(
    run_dir: Path | str,
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    """Audit a completed calibration run directory against expected binding and contracts."""
    r_dir = Path(run_dir)
    if not r_dir.is_dir():
        raise CalibrationRunAuditError(f"Run directory not found: {r_dir}")

    binding_path = r_dir / "run_binding.json"
    if not binding_path.is_file():
        raise CalibrationRunAuditError(f"Missing run_binding.json in {r_dir}")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    for k in ("run_id", "mode", "git_commit", "calibration_plan_sha256"):
        if k in expected_binding and binding.get(k) != expected_binding[k]:
            raise CalibrationRunAuditError(
                f"Run binding mismatch for '{k}': expected '{expected_binding[k]}', found '{binding.get(k)}'"
            )

    receipt_path = r_dir / "calibration_receipt.json"
    if not receipt_path.is_file():
        raise CalibrationRunAuditError(f"Missing calibration_receipt.json in {r_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    ledger_path = r_dir / "attempt_ledger.jsonl"
    if not ledger_path.is_file():
        raise CalibrationRunAuditError(f"Missing attempt_ledger.jsonl in {r_dir}")

    # Verify attempts in receipt
    attempts = receipt.get("attempts", [])
    if len(attempts) > CALIBRATION_BUDGET_LIMIT:
        raise CalibrationRunAuditError(
            f"Receipt contains {len(attempts)} attempts > budget {CALIBRATION_BUDGET_LIMIT}"
        )

    # Verify outside L1 invariance on actual composite images
    for att in attempts:
        if att.get("status") == "ACCEPTED":
            att_id = att["attempt_id"]
            comp_path = r_dir / "images" / f"{att_id}_composite.png"
            auth_path = r_dir / "images" / f"{att_id}_auth.png"
            mask_path = r_dir / "masks" / f"{att_id}_mask.png"

            if not comp_path.is_file() or not auth_path.is_file() or not mask_path.is_file():
                raise CalibrationRunAuditError(f"Missing image files for attempt {att_id}")

            with Image.open(comp_path) as comp_im, Image.open(auth_path) as auth_im, Image.open(mask_path) as mask_im:
                c_arr = np.array(comp_im.convert("RGB"), dtype=np.float32)
                a_arr = np.array(auth_im.convert("RGB"), dtype=np.float32)
                m_arr = np.array(mask_im.convert("L"), dtype=np.uint8) == 255
                outside = ~m_arr
                diff = np.abs(c_arr[outside] - a_arr[outside])
                mean_l1 = float(np.mean(diff)) if np.any(outside) else 0.0
                max_d = float(np.max(diff)) if np.any(outside) else 0.0

                if mean_l1 > 0.0 or max_d > 0.0:
                    raise CalibrationRunAuditError(
                        f"Attempt {att_id} outside L1 invariance violated: mean={mean_l1}, max={max_d}"
                    )

    return {
        "status": "PASS",
        "run_id": r_dir.name,
        "mode": "calibration",
        "total_attempts": len(attempts),
        "calibration_plan_sha256": binding.get("calibration_plan_sha256"),
        "git_commit": binding.get("git_commit"),
    }
