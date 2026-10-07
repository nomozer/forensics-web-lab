"""Independent Validation Cohort Acquisition and Generation Pipeline (Phase 4C.7B).

This module implements the production acquisition runner, real image downloader,
and quality control framework for the amended Phase 4C.7B independent cohort protocol:
1. 2x2 Orthogonal balanced design (COCO 2017 Clean x Unsplash Verified) x (SD2 x SDXL).
2. Exactly 400 target pairs (100 per stratum) with 440 buffer candidate pool (110 per stratum).
3. Real authentic photographs with verified provenance from official sources (no synthetic noise).
4. Fail-closed production pipeline: zero fallback to random or placeholder pixels on download failure.
5. Multi-layer disjoint guards against all 684 historical Option P sources.
6. Canvas and mask normalization (512x512 PNG, RGB 8-bit, binary mask {0, 255}).
7. Strict detector isolation: ZERO calls to detector models during acquisition/QC.
8. Durable on-disk persistence (immediate attempt logging, resume with SHA-256 tamper detection).
9. Technical QC verification and content QC tracking (PENDING_CONTENT_QC).
10. Pluggable execution engines:
    - MockInpaintingEngine: Fast deterministic engine for unit tests (requires allow_synthetic=True).
    - DiffusersInpaintingEngine: Production GPU pipeline for SD2 and SDXL (Colab T4 / A100 GPU).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence
import urllib.request

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

DEFAULT_CATALOG_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog.json"


class DetectorIsolationViolationError(RuntimeError):
    """Raised if any detector model or scoring routine is invoked during acquisition/QC."""


class AcquisitionQCFailureError(ValueError):
    """Raised when an acquired sample fails technical quality control."""


class StratumQuotaDeficitError(RuntimeError):
    """Raised when a stratum candidate buffer is exhausted before reaching target quota."""


class SyntheticSourceProhibitedError(ValueError):
    """Raised when synthetic sources or noise generators are supplied to production cohort acquisition."""


class ImageDownloadError(RuntimeError):
    """Raised when downloading an authentic photographic image fails."""


class MissingProvenanceError(ValueError):
    """Raised when required provenance metadata (download URL, license, author) is missing."""


class TamperDetectedError(RuntimeError):
    """Raised when an existing artifact checksum mismatches recorded ledger upon resume."""


class QuotaDeficitError(RuntimeError):
    """Raised when final cohort fails to meet the exact required quotas."""


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
    download_url: str
    license_name: str
    license_evidence_source: str
    published_date: str
    prompt: str
    generation_seed: int
    pool_index: int
    is_synthetic: bool = False
    evidence_class: str = "verified_real_catalog"
    eligible_for_independent_cohort: bool = True


@dataclass
class AcquisitionReceipt:
    """Provenance and QC ledger record for a processed sample pair."""
    source_id: str
    stratum_id: str
    source_origin: str
    origin_id: str
    author: str
    origin_url: str
    download_url: str
    license_name: str
    license_evidence_source: str
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
    is_synthetic: bool = False


def download_authentic_image(
    spec: CandidateSpec,
    timeout: float = 25.0,
    max_retries: int = 2,
    hist_sources: set[str] | None = None,
    hist_hashes: set[str] | None = None,
    hist_origins: set[str] | None = None,
    allow_synthetic: bool = False,
) -> tuple[Image.Image, bytes, str]:
    """Download authentic photographic image with fail-closed integrity and provenance guards.

    Rules:
    - If spec.is_synthetic is True and allow_synthetic is False: FAIL-CLOSED.
    - If required provenance (download_url, license_name, author) is missing: FAIL-CLOSED.
    - If download encounters HTTP 404, network error, or timeout: FAIL-CLOSED (no synthetic fallback).
    - Checks raw byte hash against historical Option P: FAIL-CLOSED if collision.
    - Decodes with PIL: FAIL-CLOSED if corrupt.

    Returns:
    - (PIL Image, raw_bytes, raw_sha256)
    """
    if spec.is_synthetic and not allow_synthetic:
        raise SyntheticSourceProhibitedError(
            f"Candidate {spec.candidate_id} is marked synthetic, which is strictly prohibited "
            "for production independent cohort acquisition!"
        )

    if not spec.download_url:
        raise MissingProvenanceError(f"Candidate {spec.candidate_id} missing download_url.")
    if not spec.license_name:
        raise MissingProvenanceError(f"Candidate {spec.candidate_id} missing license_name.")
    if not spec.license_evidence_source:
        raise MissingProvenanceError(f"Candidate {spec.candidate_id} missing license_evidence_source.")
    if not spec.author:
        raise MissingProvenanceError(f"Candidate {spec.candidate_id} missing author.")

    # Disjoint checks on metadata before network
    if hist_sources and spec.candidate_id in hist_sources:
        raise SourceOverlapError(f"Candidate ID {spec.candidate_id} collides with historical Option P source.")
    if hist_origins and spec.origin_id in hist_origins:
        raise SourceOverlapError(f"Origin ID {spec.origin_id} collides with historical Option P instance.")

    # Support local file path or file:// URL for testing/offline mirrors
    data: bytes | None = None
    url = spec.download_url

    if url.startswith("file://") or Path(url).is_file():
        local_p = Path(url[7:] if url.startswith("file://") else url)
        if not local_p.is_file():
            raise ImageDownloadError(f"Local file not found for candidate {spec.candidate_id}: {local_p}")
        data = local_p.read_bytes()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ForensicsWebLab-IndependentAcquisition/1.2)"})
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise ImageDownloadError(f"HTTP status {resp.status} fetching {url}")
                    data = resp.read()
                    break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise ImageDownloadError(
                        f"Failed to download authentic image for candidate {spec.candidate_id} "
                        f"from {url} after {max_retries+1} attempts: {last_err}"
                    ) from last_err

    if not data or len(data) == 0:
        raise ImageDownloadError(f"Zero bytes received for candidate {spec.candidate_id} from {url}")

    raw_sha256 = hashlib.sha256(data).hexdigest()

    # Disjoint check on raw downloaded bytes
    if hist_hashes and raw_sha256 in hist_hashes:
        raise SourceOverlapError(
            f"Downloaded image SHA-256 {raw_sha256} for candidate {spec.candidate_id} "
            "matches an image from historical Option P! Disjoint guard triggered."
        )

    # Decode image with Pillow
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ImageDownloadError(
            f"Downloaded byte stream for candidate {spec.candidate_id} failed Pillow decode: {e}"
        ) from e

    return img, data, raw_sha256


def generate_canonical_candidate_plan(
    catalog_path: Path | str | None = None,
    seed: int = 20261007,
    buffer_per_stratum: int = STRATUM_BUFFER_PAIRS,
) -> list[CandidateSpec]:
    """Generate the canonical candidate acquisition plan using verified real sources from catalog.

    Guarantees:
    - Real authentic photographs with verifiable provenance from official sources.
    - Zero synthetic images or mock IDs.
    - Exactly buffer_per_stratum (110) candidates per stratum.
    - Quota distributions:
      * Modification types: 40% replacement (44), 30% removal (33), 30% insertion (33)
      * Mask area classes: 30% small (33), 40% medium (44), 30% large (33)
    - Balanced orthogonal layout across 4 strata.
    """
    cat_path = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
    if not cat_path.is_file():
        raise FileNotFoundError(
            f"Verified candidate catalog not found at {cat_path}. "
            "Run 'python scripts/research/build_verified_candidate_catalog.py' to generate it."
        )

    catalog_data = json.loads(cat_path.read_text(encoding="utf-8"))
    coco_catalog = catalog_data.get("coco_candidates", [])
    unsplash_catalog = catalog_data.get("unsplash_candidates", [])

    if len(coco_catalog) < buffer_per_stratum * 2:
        raise ValueError(f"Catalog has only {len(coco_catalog)} COCO candidates, need {buffer_per_stratum * 2}")
    if len(unsplash_catalog) < buffer_per_stratum * 2:
        raise ValueError(f"Catalog has only {len(unsplash_catalog)} Unsplash candidates, need {buffer_per_stratum * 2}")

    rng = np.random.default_rng(seed)
    candidates: list[CandidateSpec] = []

    # Quota layout for buffer of 110 items per stratum
    mod_types_base = (
        ["object_replacement"] * 44
        + ["object_removal_and_infill"] * 33
        + ["object_insertion"] * 33
    )
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

    strata_configs = [
        ("coco_sd2", "coco_2017", "stable_diffusion_2_inpainting", coco_catalog[:buffer_per_stratum]),
        ("coco_sdxl", "coco_2017", "sdxl_inpainting", coco_catalog[buffer_per_stratum : buffer_per_stratum * 2]),
        ("unsplash_sd2", "unsplash_verified", "stable_diffusion_2_inpainting", unsplash_catalog[:buffer_per_stratum]),
        ("unsplash_sdxl", "unsplash_verified", "sdxl_inpainting", unsplash_catalog[buffer_per_stratum : buffer_per_stratum * 2]),
    ]

    global_pool_idx = 0
    for stratum_id, source_origin, tool_key, raw_pool in strata_configs:
        perm_mods = list(rng.permutation(mod_types_base))
        perm_masks = list(rng.permutation(mask_areas_base))

        for idx, item in enumerate(raw_pool):
            global_pool_idx += 1
            cid = f"IND_{stratum_id.upper()}_{idx+1:03d}"
            mod_type = perm_mods[idx]
            mask_class = perm_masks[idx]

            prompt_pool = prompts_by_mod[mod_type]
            prompt = prompt_pool[idx % len(prompt_pool)]
            gen_seed = int(seed + global_pool_idx * 101)

            candidates.append(
                CandidateSpec(
                    candidate_id=cid,
                    stratum_id=stratum_id,
                    source_origin=source_origin,
                    tool_key=tool_key,
                    modification_type=mod_type,
                    mask_area_class=mask_class,
                    origin_id=item["origin_id"],
                    author=item["author"],
                    origin_url=item["origin_url"],
                    download_url=item["download_url"],
                    license_name=item["license_name"],
                    license_evidence_source=item["license_evidence_source"],
                    published_date=item["published_date"],
                    prompt=prompt,
                    generation_seed=gen_seed,
                    pool_index=idx,
                    is_synthetic=False,
                    evidence_class="verified_real_catalog",
                    eligible_for_independent_cohort=True,
                )
            )

    return candidates


def generate_synthetic_fixture_plan(
    seed: int = 20261007,
    buffer_per_stratum: int = STRATUM_BUFFER_PAIRS,
) -> list[CandidateSpec]:
    """Generate synthetic candidate plan explicitly marked as test fixture.

    Strictly ineligible for official independent cohort evaluation.
    Used only in hermetic offline tests and unit test suites.
    """
    rng = np.random.default_rng(seed)
    candidates: list[CandidateSpec] = []

    mod_types_base = ["object_replacement"] * 44 + ["object_removal_and_infill"] * 33 + ["object_insertion"] * 33
    mask_areas_base = ["small_under_10pct"] * 33 + ["medium_10_to_30pct"] * 44 + ["large_over_30pct"] * 33

    strata_definitions = [
        ("coco_sd2", "coco_2017", "stable_diffusion_2_inpainting"),
        ("coco_sdxl", "coco_2017", "sdxl_inpainting"),
        ("unsplash_sd2", "unsplash_verified", "stable_diffusion_2_inpainting"),
        ("unsplash_sdxl", "unsplash_verified", "sdxl_inpainting"),
    ]

    global_pool_idx = 0
    for stratum_id, source_origin, tool_key in strata_definitions:
        perm_mods = list(rng.permutation(mod_types_base))
        perm_masks = list(rng.permutation(mask_areas_base))

        for idx in range(buffer_per_stratum):
            global_pool_idx += 1
            cid = f"SYNTH_{stratum_id.upper()}_{idx+1:03d}"
            candidates.append(
                CandidateSpec(
                    candidate_id=cid,
                    stratum_id=stratum_id,
                    source_origin=source_origin,
                    tool_key=tool_key,
                    modification_type=perm_mods[idx],
                    mask_area_class=perm_masks[idx],
                    origin_id=f"synthetic_origin_{global_pool_idx:06d}",
                    author="synthetic_author",
                    origin_url="internal://synthetic",
                    download_url="internal://synthetic_noise",
                    license_name="SYNTHETIC_FIXTURE_NON_OFFICIAL",
                    license_evidence_source="internal://fixture",
                    published_date="2026-10-07",
                    prompt="synthetic test prompt",
                    generation_seed=int(seed + global_pool_idx * 101),
                    pool_index=idx,
                    is_synthetic=True,
                    evidence_class="synthetic",
                    eligible_for_independent_cohort=False,
                )
            )

    return candidates


def normalize_image_to_canvas(
    img: Image.Image,
    target_size: tuple[int, int] = TARGET_CANVAS_SIZE,
) -> Image.Image:
    """Normalize input image to exact target canvas size (512x512 RGB lossless)."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
    """
    rng = np.random.default_rng(seed)
    w, h = canvas_size
    total_pixels = w * h

    if mask_area_class == "small_under_10pct":
        target_ratio = rng.uniform(0.03, 0.09)
    elif mask_area_class == "medium_10_to_30pct":
        target_ratio = rng.uniform(0.12, 0.28)
    elif mask_area_class == "large_over_30pct":
        target_ratio = rng.uniform(0.32, 0.46)
    else:
        raise ValueError(f"Unknown mask_area_class: {mask_area_class}")

    target_area = int(total_pixels * target_ratio)
    mask_arr = np.zeros((h, w), dtype=np.uint8)

    aspect = rng.uniform(0.6, 1.4)
    radius_x = int(math.sqrt(target_area / (math.pi * aspect)))
    radius_y = int(radius_x * aspect)

    radius_x = max(10, min(w // 2 - 10, radius_x))
    radius_y = max(10, min(h // 2 - 10, radius_y))

    margin_x = radius_x + 10
    margin_y = radius_y + 10
    cx = rng.integers(margin_x, w - margin_x) if w - margin_x > margin_x else w // 2
    cy = rng.integers(margin_y, h - margin_y) if h - margin_y > margin_y else h // 2

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
    """Execute rigorous Technical QC on an authentic/edited/mask candidate tuple."""
    if authentic_img.size != TARGET_CANVAS_SIZE or authentic_img.mode != "RGB":
        return False, f"Authentic image size/mode invalid: {authentic_img.size}, {authentic_img.mode}", 0.0

    if edited_img.size != TARGET_CANVAS_SIZE or edited_img.mode != "RGB":
        return False, f"Edited image size/mode invalid: {edited_img.size}, {edited_img.mode}", 0.0

    if mask_img.size != TARGET_CANVAS_SIZE or mask_img.mode != "L":
        return False, f"Mask size/mode invalid: {mask_img.size}, {mask_img.mode}", 0.0

    if authentic_img.size[0] * authentic_img.size[1] > MAX_PIXEL_DECOMPRESSION_BOMB:
        return False, "Decompression bomb limit exceeded", 0.0

    mask_arr = np.array(mask_img)
    unique_vals = set(np.unique(mask_arr))
    if not unique_vals.issubset({0, 255}):
        return False, f"Mask contains non-binary values: {unique_vals}", 0.0

    mask_pixels = int(np.sum(mask_arr == 255))
    total_pixels = mask_arr.size
    ratio = float(mask_pixels / total_pixels)

    bounds = {
        "small_under_10pct": (0.01, 0.105),
        "medium_10_to_30pct": (0.095, 0.305),
        "large_over_30pct": (0.295, 0.55),
    }
    low, high = bounds[mask_area_class]
    if not (low <= ratio <= high):
        return False, f"Mask ratio {ratio:.4f} outside bounds [{low}, {high}] for {mask_area_class}", ratio

    auth_arr = np.array(authentic_img, dtype=np.float32)
    edit_arr = np.array(edited_img, dtype=np.float32)

    if np.isnan(auth_arr).any() or np.isnan(edit_arr).any() or np.isinf(auth_arr).any() or np.isinf(edit_arr).any():
        return False, "Pixel arrays contain NaN or Inf values", ratio

    auth_std = float(np.std(auth_arr))
    edit_std = float(np.std(edit_arr))
    if auth_std < 2.0 or edit_std < 2.0:
        return False, f"Degenerate solid or near-blank image detected (auth_std={auth_std:.2f}, edit_std={edit_std:.2f})", ratio

    masked_diff = np.abs(auth_arr - edit_arr)[mask_arr == 255]
    if len(masked_diff) == 0:
        return False, "Mask has zero foreground pixels", ratio

    mean_masked_diff = float(np.mean(masked_diff))
    if mean_masked_diff < 3.0:
        return False, f"Edited image exhibits near-zero change in masked region (mean_diff={mean_masked_diff:.2f})", ratio

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
    """Deterministic inpainting engine for unit tests and local technical smoke tests.

    Requires explicit allow_synthetic=True permission. Completely isolated from detector models.
    """

    def inpaint(
        self,
        authentic_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        rng = np.random.default_rng(seed)
        auth_arr = np.array(authentic_img, dtype=np.uint8).copy()
        mask_arr = np.array(mask_img, dtype=np.uint8)

        h, w, c = auth_arr.shape
        color_base = rng.integers(40, 220, size=3, dtype=np.uint8)
        noise = rng.integers(-30, 30, size=(h, w, c), dtype=np.int16)

        synthetic_patch = np.clip(color_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        in_mask = mask_arr == 255
        auth_arr[in_mask] = synthetic_patch[in_mask]

        return Image.fromarray(auth_arr, mode="RGB")


class DiffusersInpaintingEngine:
    """Production GPU inpainting engine using HuggingFace Diffusers for SD2 and SDXL."""

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
                DDIMScheduler,
                EulerDiscreteScheduler,
                StableDiffusionInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Diffusers or PyTorch not available for GPU execution: {e}. "
                "Install diffusers, transformers, accelerate."
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

    def unload(self) -> None:
        """Release GPU memory cleanly."""
        del self.pipeline
        self.pipeline = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def execute_cohort_acquisition(
    output_dir: Path | str,
    candidate_specs: Sequence[CandidateSpec] | None = None,
    engine_provider: Callable[[str], Any] | None = None,
    authentic_image_provider: Callable[[CandidateSpec], tuple[Image.Image, bytes, str]] | None = None,
    historical_manifest_path: Path | str | None = None,
    target_per_stratum: int = STRATUM_TARGET_PAIRS,
    mode: str = "production",  # "production" or "fixture_test"
    allow_synthetic: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute independent cohort acquisition pipeline with fail-closed integrity.

    Modes:
    - 'production': Real photographic images only. Fails closed on download errors or synthetic sources.
    - 'fixture_test': Allows synthetic fixtures only if allow_synthetic=True. Emits provisional synthetic metadata.
    """
    assert_detector_isolation()

    if mode == "production" and allow_synthetic:
        raise SyntheticSourceProhibitedError(
            "allow_synthetic=True is forbidden when mode='production'! "
            "Production independent cohort must contain 100% verified real photographs."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Historical disjoint guards
    hist_sources = load_historical_source_ids(historical_manifest_path)
    hist_hashes = load_historical_image_hashes(historical_manifest_path)
    hist_origins = load_historical_origin_ids(historical_manifest_path)

    # 2. Candidate plan
    if candidate_specs is None:
        if mode == "production":
            candidate_specs = generate_canonical_candidate_plan()
        else:
            candidate_specs = generate_synthetic_fixture_plan()

    if not allow_synthetic:
        for spec in candidate_specs:
            if spec.is_synthetic:
                raise SyntheticSourceProhibitedError(
                    f"Candidate {spec.candidate_id} is marked synthetic, which is strictly prohibited "
                    "for production independent cohort acquisition!"
                )

    # 3. Setup default authentic provider
    if authentic_image_provider is None:
        if mode == "production":
            def real_downloader_provider(spec: CandidateSpec) -> tuple[Image.Image, bytes, str]:
                return download_authentic_image(
                    spec,
                    hist_sources=hist_sources,
                    hist_hashes=hist_hashes,
                    hist_origins=hist_origins,
                    allow_synthetic=False,
                )
            authentic_image_provider = real_downloader_provider
        else:
            if not allow_synthetic:
                raise SyntheticSourceProhibitedError("Fixture test mode requires allow_synthetic=True.")
            def mock_fixture_provider(spec: CandidateSpec) -> tuple[Image.Image, bytes, str]:
                rng = np.random.default_rng(spec.generation_seed)
                arr = rng.integers(30, 225, size=(TARGET_CANVAS_SIZE[1], TARGET_CANVAS_SIZE[0], 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw_b = buf.getvalue()
                return img, raw_b, hashlib.sha256(raw_b).hexdigest()
            authentic_image_provider = mock_fixture_provider

    # 4. Group candidates by stratum preserving sequence order
    candidates_by_stratum: dict[str, list[CandidateSpec]] = {k: [] for k in STRATA_KEYS}
    for spec in candidate_specs:
        if spec.stratum_id in candidates_by_stratum:
            candidates_by_stratum[spec.stratum_id].append(spec)

    # 5. Durable Ledgers
    provenance_jsonl = out_path / "provenance_ledger.jsonl"
    attempt_jsonl = out_path / "attempt_ledger.jsonl"

    receipts: list[AcquisitionReceipt] = []
    completed_sources: set[str] = set()
    stratum_valid_counts: dict[str, int] = {k: 0 for k in STRATA_KEYS}
    mod_counts: dict[str, int] = {m: 0 for m in MODIFICATION_TYPES}
    mask_counts: dict[str, int] = {m: 0 for m in MASK_AREA_CLASSES}
    stratum_mod_counts: dict[str, dict[str, int]] = {k: {m: 0 for m in MODIFICATION_TYPES} for k in STRATA_KEYS}
    stratum_mask_counts: dict[str, dict[str, int]] = {k: {m: 0 for m in MASK_AREA_CLASSES} for k in STRATA_KEYS}
    manifest_rows: list[dict[str, Any]] = []

    # Resume check
    if resume and provenance_jsonl.is_file():
        print(f"Resuming from existing provenance ledger: {provenance_jsonl} ...")
        with provenance_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r_dict = json.loads(line)
                receipt = AcquisitionReceipt(**r_dict)

                # Verify files on disk and checksums
                auth_p = out_path / receipt.authentic_relpath
                edit_p = out_path / receipt.edited_relpath
                mask_p = out_path / receipt.mask_relpath

                if not (auth_p.is_file() and edit_p.is_file() and mask_p.is_file()):
                    raise TamperDetectedError(f"Missing file on disk for completed source: {receipt.source_id}")

                if hashlib.sha256(auth_p.read_bytes()).hexdigest() != receipt.master_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on authentic master: {auth_p}")
                if hashlib.sha256(edit_p.read_bytes()).hexdigest() != receipt.edited_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on edited master: {edit_p}")
                if hashlib.sha256(mask_p.read_bytes()).hexdigest() != receipt.mask_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on mask: {mask_p}")

                receipts.append(receipt)
                completed_sources.add(receipt.source_id)
                stratum_valid_counts[receipt.stratum_id] += 1
                mod_counts[receipt.modification_type] += 1
                mask_counts[receipt.mask_area_class] += 1
                stratum_mod_counts[receipt.stratum_id][receipt.modification_type] += 1
                stratum_mask_counts[receipt.stratum_id][receipt.mask_area_class] += 1

                manifest_rows.append(
                    {
                        "source_id": receipt.source_id,
                        "label": "authentic",
                        "label_id": 0,
                        "image_relpath": receipt.authentic_relpath,
                        "image_sha256": receipt.master_sha256,
                        "tool": None,
                        "mask_relpath": None,
                        "width": TARGET_CANVAS_SIZE[0],
                        "height": TARGET_CANVAS_SIZE[1],
                        "stratum": receipt.stratum_id,
                        "origin_id": receipt.origin_id,
                        "license": receipt.license_name,
                    }
                )
                manifest_rows.append(
                    {
                        "source_id": receipt.source_id,
                        "label": "ai_edited",
                        "label_id": 1,
                        "image_relpath": receipt.edited_relpath,
                        "image_sha256": receipt.edited_sha256,
                        "tool": receipt.tool_key,
                        "mask_relpath": receipt.mask_relpath,
                        "width": TARGET_CANVAS_SIZE[0],
                        "height": TARGET_CANVAS_SIZE[1],
                        "stratum": receipt.stratum_id,
                        "origin_id": receipt.origin_id,
                        "license": receipt.license_name,
                    }
                )

    prov_writer_handle = provenance_jsonl.open("a", encoding="utf-8")
    attempt_writer_handle = attempt_jsonl.open("a", encoding="utf-8")

    def log_attempt(record: dict[str, Any]) -> None:
        attempt_writer_handle.write(json.dumps(record) + "\n")
        attempt_writer_handle.flush()

    def log_receipt(receipt: AcquisitionReceipt) -> None:
        prov_writer_handle.write(json.dumps(asdict(receipt)) + "\n")
        prov_writer_handle.flush()

    try:
        # Target quota breakdown per stratum (for 100 pairs target)
        target_mod_stratum = {
            "object_replacement": 40,
            "object_removal_and_infill": 30,
            "object_insertion": 30,
        }
        target_mask_stratum = {
            "small_under_10pct": 30,
            "medium_10_to_30pct": 40,
            "large_over_30pct": 30,
        }

        # 6. Execute by stratum sequentially (unload GPU models between tool strata)
        for stratum_id in STRATA_KEYS:
            stratum_candidates = candidates_by_stratum[stratum_id]
            if not stratum_candidates:
                continue

            tool_key = stratum_candidates[0].tool_key
            if engine_provider is not None:
                engine = engine_provider(tool_key)
            elif mode == "fixture_test" or allow_synthetic:
                engine = MockInpaintingEngine()
            elif mode == "production":
                engine = DiffusersInpaintingEngine(tool_key=tool_key)
            else:
                engine = MockInpaintingEngine()

            for spec in stratum_candidates:
                if stratum_valid_counts[stratum_id] >= target_per_stratum:
                    break

                if spec.candidate_id in completed_sources:
                    continue

                if target_per_stratum == STRATUM_TARGET_PAIRS:
                    # Enforce multi-dimensional quota preservation during replacement
                    if stratum_mod_counts[stratum_id][spec.modification_type] >= target_mod_stratum[spec.modification_type]:
                        continue
                    if stratum_mask_counts[stratum_id][spec.mask_area_class] >= target_mask_stratum[spec.mask_area_class]:
                        continue

                attempt_meta: dict[str, Any] = {
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "candidate_id": spec.candidate_id,
                    "stratum_id": stratum_id,
                    "tool_key": spec.tool_key,
                    "modification_type": spec.modification_type,
                    "mask_area_class": spec.mask_area_class,
                    "origin_id": spec.origin_id,
                    "pool_index": spec.pool_index,
                }

                try:
                    # Step A: Download authentic image (fail-closed)
                    auth_res = authentic_image_provider(spec)
                    if isinstance(auth_res, tuple) and len(auth_res) == 3:
                        raw_auth_img, raw_bytes, raw_sha256 = auth_res
                    elif isinstance(auth_res, Image.Image):
                        raw_auth_img = auth_res
                        buf = io.BytesIO()
                        auth_res.save(buf, format="PNG")
                        raw_bytes = buf.getvalue()
                        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                    else:
                        raise ValueError(f"Unexpected return type from authentic provider: {type(auth_res)}")

                    auth_canvas = normalize_image_to_canvas(raw_auth_img)

                    # Step B: Synthesize request mask
                    mask_canvas, target_ratio = synthesize_canonical_mask(
                        spec.mask_area_class,
                        seed=spec.generation_seed,
                    )

                    # Step C: Generate inpainting
                    edited_canvas = engine.inpaint(
                        auth_canvas,
                        mask_canvas,
                        prompt=spec.prompt,
                        seed=spec.generation_seed,
                    )

                    # Step D: Technical Quality Control
                    is_pass, fail_reason, actual_ratio = evaluate_technical_qc(
                        authentic_img=auth_canvas,
                        edited_img=edited_canvas,
                        mask_img=mask_canvas,
                        mask_area_class=spec.mask_area_class,
                        raw_bytes=raw_bytes,
                        historical_sources=hist_sources,
                        historical_hashes=hist_hashes,
                        historical_origins=hist_origins,
                        source_id=spec.candidate_id,
                        origin_id=spec.origin_id,
                    )

                    if not is_pass:
                        attempt_meta["status"] = "QC_FAILED"
                        attempt_meta["reason"] = fail_reason
                        log_attempt(attempt_meta)
                        continue  # Stratum-preserving replacement: move to next candidate

                    # Step E: Save verified master files atomically
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

                    master_sha = hashlib.sha256(auth_path.read_bytes()).hexdigest()
                    edit_sha = hashlib.sha256(edit_path.read_bytes()).hexdigest()
                    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()

                    receipt = AcquisitionReceipt(
                        source_id=source_id,
                        stratum_id=stratum_id,
                        source_origin=spec.source_origin,
                        origin_id=spec.origin_id,
                        author=spec.author,
                        origin_url=spec.origin_url,
                        download_url=spec.download_url,
                        license_name=spec.license_name,
                        license_evidence_source=spec.license_evidence_source,
                        published_date=spec.published_date,
                        tool_key=spec.tool_key,
                        modification_type=spec.modification_type,
                        mask_area_class=spec.mask_area_class,
                        prompt=spec.prompt,
                        generation_seed=spec.generation_seed,
                        raw_sha256=raw_sha256,
                        master_sha256=master_sha,
                        mask_sha256=mask_sha,
                        edited_sha256=edit_sha,
                        authentic_relpath=f"images/{auth_filename}",
                        edited_relpath=f"images/{edit_filename}",
                        mask_relpath=f"masks/{mask_filename}",
                        mask_pixel_ratio=actual_ratio,
                        technical_qc_status="PASS",
                        content_qc_status="PENDING_CONTENT_QC",
                        is_synthetic=spec.is_synthetic,
                    )

                    log_receipt(receipt)
                    receipts.append(receipt)
                    completed_sources.add(source_id)

                    manifest_rows.append(
                        {
                            "source_id": source_id,
                            "label": "authentic",
                            "label_id": 0,
                            "image_relpath": f"images/{auth_filename}",
                            "image_sha256": master_sha,
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
                    mod_counts[spec.modification_type] += 1
                    mask_counts[spec.mask_area_class] += 1

                    attempt_meta["status"] = "ACCEPTED"
                    log_attempt(attempt_meta)

                except Exception as e:
                    attempt_meta["status"] = "ERROR"
                    attempt_meta["error"] = str(e)
                    log_attempt(attempt_meta)
                    continue

            # Unload engine after stratum
            if hasattr(engine, "unload"):
                engine.unload()

            if stratum_valid_counts[stratum_id] < target_per_stratum:
                raise StratumQuotaDeficitError(
                    f"Stratum '{stratum_id}' exhausted all candidates without reaching quota: "
                    f"{stratum_valid_counts[stratum_id]}/{target_per_stratum} valid pairs acquired."
                )

    finally:
        prov_writer_handle.close()
        attempt_writer_handle.close()

    # 7. Write manifest and atomic checksum
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

    total_valid = sum(stratum_valid_counts.values())
    is_full_target_met = (total_valid == TOTAL_TARGET_PAIRS) and (target_per_stratum == STRATUM_TARGET_PAIRS)

    if target_per_stratum == STRATUM_TARGET_PAIRS:
        if total_valid != TOTAL_TARGET_PAIRS:
            raise QuotaDeficitError(
                f"Cohort acquisition failed to meet target quota: {total_valid}/{TOTAL_TARGET_PAIRS} valid pairs."
            )
        for s in STRATA_KEYS:
            if stratum_valid_counts[s] != STRATUM_TARGET_PAIRS:
                raise QuotaDeficitError(
                    f"Stratum '{s}' quota not met: {stratum_valid_counts[s]}/{STRATUM_TARGET_PAIRS}."
                )
        if (
            mod_counts["object_replacement"] != 160
            or mod_counts["object_removal_and_infill"] != 120
            or mod_counts["object_insertion"] != 120
        ):
            raise QuotaDeficitError(f"Modification type quotas skewed: {mod_counts}")
        if (
            mask_counts["small_under_10pct"] != 120
            or mask_counts["medium_10_to_30pct"] != 160
            or mask_counts["large_over_30pct"] != 120
        ):
            raise QuotaDeficitError(f"Mask area quotas skewed: {mask_counts}")
        if len(completed_sources) != TOTAL_TARGET_PAIRS:
            raise QuotaDeficitError(f"Unique source count mismatch: {len(completed_sources)} vs {TOTAL_TARGET_PAIRS}")

    status = (
        "PROVISIONAL_TECHNICAL_QC_PASS_PENDING_CONTENT_QC"
        if is_full_target_met
        else "PARTIAL_PILOT_TECHNICAL_PASS"
    )

    return {
        "status": status,
        "is_full_cohort_final": False,  # Strict: requires content QC
        "total_valid_pairs": total_valid,
        "total_images": total_valid * 2,
        "stratum_counts": stratum_valid_counts,
        "modification_type_counts": mod_counts,
        "mask_area_counts": mask_counts,
        "manifest_path": str(manifest_csv),
        "manifest_sha256": manifest_sha,
        "provenance_ledger_path": str(provenance_jsonl),
        "attempt_ledger_path": str(attempt_jsonl),
    }
