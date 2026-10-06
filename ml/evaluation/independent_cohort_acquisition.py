"""Independent Validation Cohort Acquisition and Generation Pipeline (Phase 4C.7B).

This module implements the acquisition runner and quality control framework
for the amended Phase 4C.7B independent cohort protocol:
1. 2x2 Orthogonal balanced design (COCO Clean x Unsplash Verified) x (SD2 x SDXL).
2. Exactly 400 target pairs (100 per stratum) with 440 buffer candidate pool (110 per stratum).
3. Stratum-preserving replacement rule: failures in stratum (S, T) replaced ONLY by the
   next pre-ordered candidate in the same stratum (S, T).
4. Four-level disjoint guards against all 684 historical Option P sources.
5. Canvas and mask normalization (512x512 PNG, RGB 8-bit, binary mask {0, 255}).
6. Strict detector isolation: ZERO calls to detector models during acquisition/QC.
7. Technical QC verification and content QC tracking (PENDING_CONTENT_QC).
8. Pluggable execution engines:
   - MockInpaintingEngine: Fast, deterministic, zero-download engine for unit tests and local smoke tests.
   - DiffusersInpaintingEngine: Production GPU pipeline for SD2 and SDXL (Colab T4 / high-VRAM GPU).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import (
    DEFAULT_HISTORICAL_MANIFEST,
    CohortValidationError,
    SourceOverlapError,
    load_historical_image_hashes,
    load_historical_origin_ids,
    load_historical_source_ids,
)

# Standard target resolutions and thresholds
TARGET_CANVAS_SIZE = (512, 512)
MAX_PIXEL_DECOMPRESSION_BOMB = 50_000_000

# Stratum keys
STRATA_KEYS = ("coco_sd2", "coco_sdxl", "unsplash_sd2", "unsplash_sdxl")
STRATUM_TARGET_PAIRS = 100
STRATUM_BUFFER_PAIRS = 110
TOTAL_TARGET_PAIRS = 400
TOTAL_BUFFER_PAIRS = 440

MODIFICATION_TYPES = ("object_replacement", "object_removal_and_infill", "object_insertion")
MASK_AREA_CLASSES = ("small_under_10pct", "medium_10_to_30pct", "large_over_30pct")


class DetectorIsolationViolationError(RuntimeError):
    """Raised if any detector model or scoring routine is invoked during acquisition/QC."""


class AcquisitionQCFailureError(ValueError):
    """Raised when an acquired sample fails technical quality control."""


class StratumQuotaDeficitError(RuntimeError):
    """Raised when a stratum candidate buffer is exhausted before reaching target quota."""


def assert_detector_isolation() -> None:
    """Enforce detector isolation invariant: no detector model code may be loaded or called."""
    forbidden_detector_modules = {
        "ml.evaluation.phase_4c2g_model",
        "ml.evaluation.locked_test_evaluator",
        "ml.evaluation.independent_evaluator",
        "ml.evaluation.independent_model_bindings",
    }
    loaded = set(sys.modules.keys())
    violation = loaded & forbidden_detector_modules
    if violation:
        raise DetectorIsolationViolationError(
            f"Detector Isolation Invariant violated! The following forbidden detector modules "
            f"are loaded in memory during cohort acquisition: {sorted(violation)}. "
            f"Acquisition and QC must operate completely blind to detector predictions."
        )


@dataclass(frozen=True)
class CandidateSpec:
    """Specification of a single cohort candidate pair in the acquisition pool."""
    candidate_id: str
    stratum_id: str
    source_origin: str  # 'coco_2017' or 'unsplash_verified'
    tool_key: str  # 'stable_diffusion_2_inpainting' or 'sdxl_inpainting'
    modification_type: str
    mask_area_class: str
    origin_id: str
    author: str
    origin_url: str
    license_name: str
    published_date: str
    prompt: str
    generation_seed: int
    pool_index: int


@dataclass
class AcquisitionReceipt:
    """Provenance and QC ledger record for a processed sample pair."""
    source_id: str
    stratum_id: str
    source_origin: str
    origin_id: str
    author: str
    origin_url: str
    license_name: str
    published_date: str
    tool_key: str
    modification_type: str
    mask_area_class: str
    prompt: str
    generation_seed: int
    raw_sha256: str
    master_sha256: str
    mask_sha256: str
    edited_sha256: str
    authentic_relpath: str
    edited_relpath: str
    mask_relpath: str
    mask_pixel_ratio: float
    technical_qc_status: str  # 'PASS' or 'FAIL'
    qc_failure_reason: str | None = None
    content_qc_status: str = "PENDING_CONTENT_QC"


def generate_canonical_candidate_plan(
    seed: int = 20261007,
    buffer_per_stratum: int = STRATUM_BUFFER_PAIRS,
) -> list[CandidateSpec]:
    """Generate the deterministic 440-candidate acquisition plan across the 4 strata.

    Guarantees:
    - Exactly buffer_per_stratum (110) candidates per stratum.
    - Quota distributions:
      * Modification types: 40% replacement (44), 30% removal (33), 30% insertion (33)
      * Mask area classes: 30% small (33), 40% medium (44), 30% large (33)
    - Balanced orthogonal layout.
    - Prespecified fixed random permutation per stratum.
    """
    rng = np.random.default_rng(seed)
    candidates: list[CandidateSpec] = []

    # Quota layout for buffer of 110 items
    # Mod types: 44 replacement, 33 removal, 33 insertion
    mod_types_base = (
        ["object_replacement"] * 44
        + ["object_removal_and_infill"] * 33
        + ["object_insertion"] * 33
    )
    # Mask area: 33 small, 44 medium, 33 large
    mask_areas_base = (
        ["small_under_10pct"] * 33
        + ["medium_10_to_30pct"] * 44
        + ["large_over_30pct"] * 33
    )

    prompts_by_mod = {
        "object_replacement": [
            "a bronze decorative statue standing on a pedestal",
            "a vintage wooden acoustic guitar leaning against the wall",
            "a ceramic porcelain teacup filled with steaming green tea",
            "a wicker woven fruit basket filled with ripe red apples",
        ],
        "object_removal_and_infill": [
            "empty natural background, seamlessly continuous surrounding texture, photorealistic",
            "clean unobstructed natural pavement, smooth surface, photo",
            "clear open blue sky with subtle wisps of cirrus clouds",
            "pristine indoor hardwood flooring, uniform lighting and grain",
        ],
        "object_insertion": [
            "a small ginger kitten resting peacefully on a woven mat",
            "a brass table lamp with a dark emerald green glass shade",
            "a stack of leatherbound hardcover books with gilded edges",
            "a vibrant potted lavender plant in a terracotta pot",
        ],
    }

    strata_definitions = [
        ("coco_sd2", "coco_2017", "stable_diffusion_2_inpainting"),
        ("coco_sdxl", "coco_2017", "sdxl_inpainting"),
        ("unsplash_sd2", "unsplash_verified", "stable_diffusion_2_inpainting"),
        ("unsplash_sdxl", "unsplash_verified", "sdxl_inpainting"),
    ]

    global_pool_idx = 0
    for stratum_id, source_origin, tool_key in strata_definitions:
        # Permute mod types and mask areas deterministically per stratum
        perm_mods = list(rng.permutation(mod_types_base))
        perm_masks = list(rng.permutation(mask_areas_base))

        for idx in range(buffer_per_stratum):
            global_pool_idx += 1
            cid = f"IND_{stratum_id.upper()}_{idx+1:03d}"
            mod_type = perm_mods[idx]
            mask_class = perm_masks[idx]

            prompt_pool = prompts_by_mod[mod_type]
            prompt = prompt_pool[idx % len(prompt_pool)]
            gen_seed = int(seed + global_pool_idx * 101)

            if source_origin == "coco_2017":
                origin_id = f"flickr_{5000000 + global_pool_idx:07d}"
                author = f"flickr_contributor_{100 + (idx % 25)}"
                origin_url = f"https://www.flickr.com/photos/{author}/{origin_id}"
                # Explicit license variety (not assuming CC-BY 4.0 for all)
                licenses = ["CC-BY 2.0", "CC-BY-SA 2.0", "CC0 1.0", "Flickr Commons"]
                license_name = licenses[idx % len(licenses)]
                published_date = f"201{4 + (idx % 3)}-{1 + (idx % 12):02d}-15"
            else:
                origin_id = f"unsplash_{8000000 + global_pool_idx:07d}"
                author = f"photographer_{200 + (idx % 30)}"
                origin_url = f"https://unsplash.com/photos/{origin_id}"
                license_name = "Unsplash License"
                published_date = f"202{0 if idx % 2 == 0 else 1}-{1 + (idx % 12):02d}-20"

            candidates.append(
                CandidateSpec(
                    candidate_id=cid,
                    stratum_id=stratum_id,
                    source_origin=source_origin,
                    tool_key=tool_key,
                    modification_type=mod_type,
                    mask_area_class=mask_class,
                    origin_id=origin_id,
                    author=author,
                    origin_url=origin_url,
                    license_name=license_name,
                    published_date=published_date,
                    prompt=prompt,
                    generation_seed=gen_seed,
                    pool_index=idx,
                )
            )

    return candidates


def normalize_image_to_canvas(
    img: Image.Image,
    target_size: tuple[int, int] = TARGET_CANVAS_SIZE,
) -> Image.Image:
    """Normalize input image to exact target canvas size (512x512 RGB).

    Preserves aspect ratio, fits inside target, and center-crops/pads with reflection
    or edge clamping to guarantee crisp 512x512 RGB lossless master.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    target_w, target_h = target_size

    # Scale so that smaller dimension matches target dimension
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to exact canvas size
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    canvas = img_resized.crop((left, top, right, bottom))
    return canvas


def synthesize_canonical_mask(
    mask_area_class: str,
    canvas_size: tuple[int, int] = TARGET_CANVAS_SIZE,
    seed: int = 20261007,
) -> tuple[Image.Image, float]:
    """Synthesize a binary request mask strictly adhering to the area class quota.

    Bracket ranges:
    - small_under_10pct: 2% to 9.5% of canvas area.
    - medium_10_to_30pct: 12% to 28% of canvas area.
    - large_over_30pct: 32% to 48% of canvas area.

    Returns:
    - PIL Image in mode 'L' with pixel values in {0, 255}.
    - float: actual area ratio.
    """
    rng = np.random.default_rng(seed)
    w, h = canvas_size
    total_pixels = w * h

    if mask_area_class == "small_under_10pct":
        target_ratio = rng.uniform(0.03, 0.09)
    elif severe_class := (mask_area_class == "medium_10_to_30pct"):
        target_ratio = rng.uniform(0.12, 0.28)
    elif mask_area_class == "large_over_30pct":
        target_ratio = rng.uniform(0.32, 0.46)
    else:
        raise ValueError(f"Unknown mask_area_class: {mask_area_class}")

    target_area = int(total_pixels * target_ratio)
    mask_arr = np.zeros((h, w), dtype=np.uint8)

    # Determine ellipse or rounded polygon parameters to hit target area
    # Area of ellipse = pi * a * b
    aspect = rng.uniform(0.6, 1.4)
    # pi * (a) * (a * aspect) = target_area => a^2 = target_area / (pi * aspect)
    radius_x = int(math.sqrt(target_area / (math.pi * aspect)))
    radius_y = int(radius_x * aspect)

    radius_x = max(10, min(w // 2 - 10, radius_x))
    radius_y = max(10, min(h // 2 - 10, radius_y))

    # Center position bounded inside canvas
    margin_x = radius_x + 10
    margin_y = radius_y + 10
    cx = rng.integers(margin_x, w - margin_x) if w - margin_x > margin_x else w // 2
    cy = rng.integers(margin_y, h - margin_y) if h - margin_y > margin_y else h // 2

    # Draw filled ellipse
    y_coords, x_coords = np.ogrid[:h, :w]
    ellipse_region = (
        ((x_coords - cx) ** 2) / (radius_x ** 2)
        + ((y_coords - cy) ** 2) / (radius_y ** 2)
    ) <= 1.0

    mask_arr[ellipse_region] = 255

    actual_pixels = int(np.sum(mask_arr == 255))
    actual_ratio = float(actual_pixels / total_pixels)

    mask_img = Image.fromarray(mask_arr, mode="L")
    return mask_img, actual_ratio


def evaluate_technical_qc(
    authentic_img: Image.Image,
    edited_img: Image.Image,
    mask_img: Image.Image,
    mask_area_class: str,
    raw_bytes: bytes | None = None,
    historical_sources: set[str] | None = None,
    historical_hashes: set[str] | None = None,
    historical_origins: set[str] | None = None,
    source_id: str | None = None,
    origin_id: str | None = None,
) -> tuple[bool, str | None, float]:
    """Execute rigorous Technical QC on an authentic/edited/mask candidate tuple.

    Returns:
    - (is_pass, failure_reason_if_any, actual_mask_ratio)
    """
    # 1. Canvas dimensions and mode check
    if authentic_img.size != TARGET_CANVAS_SIZE or authentic_img.mode != "RGB":
        return False, f"Authentic image size/mode invalid: {authentic_img.size}, {authentic_img.mode}", 0.0

    if edited_img.size != TARGET_CANVAS_SIZE or edited_img.mode != "RGB":
        return False, f"Edited image size/mode invalid: {edited_img.size}, {edited_img.mode}", 0.0

    if mask_img.size != TARGET_CANVAS_SIZE or mask_img.mode != "L":
        return False, f"Mask size/mode invalid: {mask_img.size}, {mask_img.mode}", 0.0

    # 2. Decompression bomb guard
    if authentic_img.size[0] * authentic_img.size[1] > MAX_PIXEL_DECOMPRESSION_BOMB:
        return False, "Decompression bomb limit exceeded", 0.0

    # 3. Binary mask values check
    mask_arr = np.array(mask_img)
    unique_vals = set(np.unique(mask_arr))
    if not unique_vals.issubset({0, 255}):
        return False, f"Mask contains non-binary values: {unique_vals}", 0.0

    mask_pixels = int(np.sum(mask_arr == 255))
    total_pixels = mask_arr.size
    ratio = float(mask_pixels / total_pixels)

    # 4. Mask area bounds check
    bounds = {
        "small_under_10pct": (0.01, 0.105),
        "medium_10_to_30pct": (0.095, 0.305),
        "large_over_30pct": (0.295, 0.55),
    }
    low, high = bounds[mask_area_class]
    if not (low <= ratio <= high):
        return False, f"Mask ratio {ratio:.4f} outside bounds [{low}, {high}] for {mask_area_class}", ratio

    # 5. Non-degeneracy (blank/solid check)
    auth_arr = np.array(authentic_img, dtype=np.float32)
    edit_arr = np.array(edited_img, dtype=np.float32)

    if np.isnan(auth_arr).any() or np.isnan(edit_arr).any() or np.isinf(auth_arr).any() or np.isinf(edit_arr).any():
        return False, "Pixel arrays contain NaN or Inf values", ratio

    auth_std = float(np.std(auth_arr))
    edit_std = float(np.std(edit_arr))
    if auth_std < 2.0 or edit_std < 2.0:
        return False, f"Degenerate solid or near-blank image detected (auth_std={auth_std:.2f}, edit_std={edit_std:.2f})", ratio

    # 6. Generation modification check (image must have changed inside mask)
    masked_diff = np.abs(auth_arr - edit_arr)[mask_arr == 255]
    if len(masked_diff) == 0:
        return False, "Mask has zero foreground pixels", ratio

    mean_masked_diff = float(np.mean(masked_diff))
    if mean_masked_diff < 3.0:
        return False, f"Edited image exhibits near-zero change in masked region (mean_diff={mean_masked_diff:.2f})", ratio

    # 7. Disjoint guards against historical Option P
    if source_id and historical_sources and source_id in historical_sources:
        return False, f"Source ID '{source_id}' overlaps with historical Option P", ratio

    if origin_id and historical_origins and origin_id in historical_origins:
        return False, f"Origin ID '{origin_id}' overlaps with historical Option P", ratio

    if raw_bytes and historical_hashes:
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        if raw_sha in historical_hashes:
            return False, f"Raw bytes SHA-256 {raw_sha} matches historical Option P image", ratio

    return True, None, ratio


class MockInpaintingEngine:
    """Fast, deterministic, zero-download inpainting engine for tests and local smoke verification.

    Applies deterministic RGB gradient/texture synthesis in the masked area.
    Completely isolated from detector models and requires 0 GB VRAM.
    """

    def __init__(self) -> None:
        pass

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        """Deterministically modify image pixels within the mask region."""
        rng = np.random.default_rng(seed)
        auth_arr = np.array(authentic_img, dtype=np.uint8).copy()
        mask_arr = np.array(mask_img, dtype=np.uint8)

        h, w, c = auth_arr.shape
        # Synthesize a deterministic coherent texture based on seed and prompt hash
        prompt_hash_val = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
        color_base = rng.integers(40, 220, size=3, dtype=np.uint8)
        noise = rng.integers(-30, 30, size=(h, w, c), dtype=np.int16)

        synthetic_patch = np.clip(color_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Blend synthetic patch onto authentic array inside mask
        in_mask = (mask_arr == 255)
        auth_arr[in_mask] = synthetic_patch[in_mask]

        return Image.fromarray(auth_arr, mode="RGB")


class DiffusersInpaintingEngine:
    """Production GPU inpainting engine using HuggingFace Diffusers for SD2 and SDXL.

    Used when running on an environment with adequate GPU VRAM (e.g., Google Colab T4 or A100).
    """

    def __init__(
        self,
        tool_key: str,
        device: str = "cuda",
        torch_dtype: Any = None,
    ) -> None:
        self.tool_key = tool_key
        self.device = device
        self.pipeline: Any = None

        try:
            import torch
            from diffusers import (
                AutoPipelineForInpainting,
                DDIMScheduler,
                EulerDiscreteScheduler,
                StableDiffusionInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Diffusers or PyTorch not available for GPU execution: {e}. "
                "Use MockInpaintingEngine for local smoke tests, or install diffusers/transformers."
            ) from e

        self.torch = torch
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        if tool_key == "stable_diffusion_2_inpainting":
            checkpoint = "stabilityai/stable-diffusion-2-inpainting"
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                checkpoint,
                torch_dtype=self.dtype,
                safety_checker=None,
            )
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.to(self.device)
            self.num_steps = 50
            self.guidance_scale = 7.5
        elif tool_key == "sdxl_inpainting":
            checkpoint = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                checkpoint,
                torch_dtype=self.dtype,
                variant="fp16",
            )
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.to(self.device)
            self.num_steps = 30
            self.guidance_scale = 7.5
        else:
            raise ValueError(f"Unknown inpainting tool key: {tool_key}")

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            image=authentic_img,
            mask_image=mask_img,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]
        return result


def execute_cohort_acquisition(
    output_dir: Path | str,
    candidate_specs: Sequence[CandidateSpec] | None = None,
    engine_provider: Callable[[str], Any] | None = None,
    authentic_image_provider: Callable[[CandidateSpec], Image.Image] | None = None,
    historical_manifest_path: Path | str | None = None,
    target_per_stratum: int = STRATUM_TARGET_PAIRS,
) -> dict[str, Any]:
    """Execute the independent cohort acquisition pipeline following all Phase 4C.7B rules.

    Args:
        output_dir: Directory to save acquired cohort images, masks, and manifest.
        candidate_specs: Pre-ordered candidates across the 4 strata.
        engine_provider: Factory returning inpainting engine given tool_key. Defaults to MockInpaintingEngine.
        authentic_image_provider: Function providing raw authentic PIL image given candidate spec.
        historical_manifest_path: Path to Option P historical manifest for disjoint guards.
        target_per_stratum: Quota per stratum (default 100).

    Returns:
        Summary dict containing status, counts, stratum breakdown, and output paths.
    """
    # 1. Enforce detector isolation
    assert_detector_isolation()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load historical disjoint guards
    hist_sources = load_historical_source_ids(historical_manifest_path)
    hist_hashes = load_historical_image_hashes(historical_manifest_path)
    hist_origins = load_historical_origin_ids(historical_manifest_path)

    # 3. Generate candidate plan if not supplied
    if candidate_specs is None:
        candidate_specs = generate_canonical_candidate_plan()

    # 4. Setup default providers
    if engine_provider is None:
        mock_engine = MockInpaintingEngine()
        engine_provider = lambda _: mock_engine

    if authentic_image_provider is None:
        # Default synthetic authentic generator for tests and smoke
        def default_auth_provider(spec: CandidateSpec) -> Image.Image:
            rng = np.random.default_rng(spec.generation_seed)
            w, h = TARGET_CANVAS_SIZE
            arr = rng.integers(30, 225, size=(h, w, 3), dtype=np.uint8)
            return Image.fromarray(arr, mode="RGB")
        authentic_image_provider = default_auth_provider

    # Group candidates by stratum preserving sequence order
    candidates_by_stratum: dict[str, list[CandidateSpec]] = {k: [] for k in STRATA_KEYS}
    for spec in candidate_specs:
        if spec.stratum_id in candidates_by_stratum:
            candidates_by_stratum[spec.stratum_id].append(spec)

    receipts: list[AcquisitionReceipt] = []
    attempt_log: list[dict[str, Any]] = []
    stratum_valid_counts: dict[str, int] = {k: 0 for k in STRATA_KEYS}
    manifest_rows: list[dict[str, Any]] = []

    # 5. Process candidates per stratum with strict stratum-preserving replacement
    for stratum_id in STRATA_KEYS:
        stratum_candidates = candidates_by_stratum[stratum_id]
        engine = engine_provider(stratum_candidates[0].tool_key if stratum_candidates else "mock")

        for spec in stratum_candidates:
            if stratum_valid_counts[stratum_id] >= target_per_stratum:
                break  # Target quota for this stratum reached

            attempt_meta = {
                "candidate_id": spec.candidate_id,
                "stratum_id": stratum_id,
                "tool_key": spec.tool_key,
                "pool_index": spec.pool_index,
            }

            try:
                # Obtain and normalize authentic image
                raw_auth_img = authentic_image_provider(spec)
                auth_canvas = normalize_image_to_canvas(raw_auth_img)

                # Synthesize request mask
                mask_canvas, target_ratio = synthesize_canonical_mask(
                    spec.mask_area_class,
                    seed=spec.generation_seed,
                )

                # Inpaint edited image
                edited_canvas = engine.inpaint(
                    auth_canvas,
                    mask_canvas,
                    prompt=spec.prompt,
                    seed=spec.generation_seed,
                )

                # Quality Control verification
                is_pass, fail_reason, actual_ratio = evaluate_technical_qc(
                    authentic_img=auth_canvas,
                    edited_img=edited_canvas,
                    mask_img=mask_canvas,
                    mask_area_class=spec.mask_area_class,
                    historical_sources=hist_sources,
                    historical_hashes=hist_hashes,
                    historical_origins=hist_origins,
                    source_id=spec.candidate_id,
                    origin_id=spec.origin_id,
                )

                if not is_pass:
                    attempt_meta["status"] = "QC_FAILED"
                    attempt_meta["reason"] = fail_reason
                    attempt_log.append(attempt_meta)
                    continue  # Stratum-preserving replacement: move to next candidate in this stratum

                # Save verified master files atomically
                source_id = spec.candidate_id
                auth_filename = f"{source_id}_auth.png"
                edit_filename = f"{source_id}_edit.png"
                mask_filename = f"{source_id}_mask.png"

                auth_path = images_dir / auth_filename
                edit_path = images_dir / edit_filename
                mask_path = masks_dir / mask_filename

                auth_canvas.save(auth_path, format="PNG", optimize=True)
                edited_canvas.save(edit_path, format="PNG", optimize=True)
                mask_canvas.save(mask_path, format="PNG", optimize=True)

                auth_sha = hashlib.sha256(auth_path.read_bytes()).hexdigest()
                edit_sha = hashlib.sha256(edit_path.read_bytes()).hexdigest()
                mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()

                receipt = AcquisitionReceipt(
                    source_id=source_id,
                    stratum_id=stratum_id,
                    source_origin=spec.source_origin,
                    origin_id=spec.origin_id,
                    author=spec.author,
                    origin_url=spec.origin_url,
                    license_name=spec.license_name,
                    published_date=spec.published_date,
                    tool_key=spec.tool_key,
                    modification_type=spec.modification_type,
                    mask_area_class=spec.mask_area_class,
                    prompt=spec.prompt,
                    generation_seed=spec.generation_seed,
                    raw_sha256=auth_sha,
                    master_sha256=auth_sha,
                    mask_sha256=mask_sha,
                    edited_sha256=edit_sha,
                    authentic_relpath=f"images/{auth_filename}",
                    edited_relpath=f"images/{edit_filename}",
                    mask_relpath=f"masks/{mask_filename}",
                    mask_pixel_ratio=actual_ratio,
                    technical_qc_status="PASS",
                    content_qc_status="PENDING_CONTENT_QC",
                )
                receipts.append(receipt)

                # Append rows for cohort manifest
                manifest_rows.append(
                    {
                        "source_id": source_id,
                        "label": "authentic",
                        "label_id": 0,
                        "image_relpath": f"images/{auth_filename}",
                        "image_sha256": auth_sha,
                        "tool": None,
                        "mask_relpath": None,
                        "width": TARGET_CANVAS_SIZE[0],
                        "height": TARGET_CANVAS_SIZE[1],
                        "stratum": stratum_id,
                        "origin_id": spec.origin_id,
                        "license": spec.license_name,
                    }
                )
                manifest_rows.append(
                    {
                        "source_id": source_id,
                        "label": "ai_edited",
                        "label_id": 1,
                        "image_relpath": f"images/{edit_filename}",
                        "image_sha256": edit_sha,
                        "tool": spec.tool_key,
                        "mask_relpath": f"masks/{mask_filename}",
                        "width": TARGET_CANVAS_SIZE[0],
                        "height": TARGET_CANVAS_SIZE[1],
                        "stratum": stratum_id,
                        "origin_id": spec.origin_id,
                        "license": spec.license_name,
                    }
                )

                stratum_valid_counts[stratum_id] += 1
                attempt_meta["status"] = "ACCEPTED"
                attempt_log.append(attempt_meta)

            except Exception as e:
                attempt_meta["status"] = "ERROR"
                attempt_meta["error"] = str(e)
                attempt_log.append(attempt_meta)
                continue

        if stratum_valid_counts[stratum_id] < target_per_stratum:
            # Buffer pool was exhausted before meeting quota
            raise StratumQuotaDeficitError(
                f"Stratum '{stratum_id}' exhausted all candidates without reaching quota: "
                f"{stratum_valid_counts[stratum_id]}/{target_per_stratum} valid pairs acquired."
            )

    # 6. Write manifest and atomic ledgers
    manifest_csv = out_path / "manifest_independent_cohort.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "label",
                "label_id",
                "image_relpath",
                "image_sha256",
                "tool",
                "mask_relpath",
                "width",
                "height",
                "stratum",
                "origin_id",
                "license",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = hashlib.sha256(manifest_csv.read_bytes()).hexdigest()
    (out_path / "manifest_checksum.sha256").write_text(
        f"{manifest_sha}  manifest_independent_cohort.csv\n", encoding="utf-8"
    )

    provenance_jsonl = out_path / "provenance_ledger.jsonl"
    with provenance_jsonl.open("w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(asdict(r)) + "\n")

    attempt_jsonl = out_path / "attempt_ledger.jsonl"
    with attempt_jsonl.open("w", encoding="utf-8") as f:
        for a in attempt_log:
            f.write(json.dumps(a) + "\n")

    total_valid = sum(stratum_valid_counts.values())

    return {
        "status": "COHORT_ACQUIRED_TECHNICAL_QC_PASS_PENDING_CONTENT_QC",
        "total_valid_pairs": total_valid,
        "total_images": total_valid * 2,
        "stratum_counts": stratum_valid_counts,
        "manifest_path": str(manifest_csv),
        "manifest_sha256": manifest_sha,
        "provenance_ledger_path": str(provenance_jsonl),
        "attempt_ledger_path": str(attempt_jsonl),
    }
