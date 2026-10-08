"""Independent cohort acquisition and generation pipeline.

This module implements the production acquisition runner, real image downloader,
and quality control framework for the amended Phase 4C.7B independent cohort protocol:
1. 2x2 orthogonal design (COCO 2017 x Wikimedia Commons) x (SD2 x SDXL).
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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from typing import Any, Callable, Sequence
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import (
    DEFAULT_HISTORICAL_MANIFEST,
    SourceOverlapError,
    load_historical_image_hashes,
)

# Standard target resolutions and thresholds
TARGET_CANVAS_SIZE = (512, 512)
MAX_PIXEL_DECOMPRESSION_BOMB = 50_000_000
NON_BLANK_STD_THRESHOLD = 5.0
MIN_MASKED_PIXEL_DELTA_L1 = 3.0
MAX_UNMASKED_PIXEL_DELTA_L1 = 0.5
MASK_AREA_BOUNDS = {
    "small_under_10pct": (0.0, 0.12),
    "medium_10_to_30pct": (0.08, 0.32),
    "large_over_30pct": (0.28, 0.52),
}

# Stratum keys
STRATA_KEYS = ("coco_sd2", "coco_sdxl", "commons_sd2", "commons_sdxl")
STRATUM_TARGET_PAIRS = 100
STRATUM_BUFFER_PAIRS = 110
TOTAL_TARGET_PAIRS = 400
TOTAL_BUFFER_PAIRS = 440

MODIFICATION_TYPES = ("object_replacement", "object_removal_and_infill", "object_insertion")
MASK_AREA_CLASSES = ("small_under_10pct", "medium_10_to_30pct", "large_over_30pct")
TARGET_MOD_PER_STRATUM = {"object_replacement": 40, "object_removal_and_infill": 30, "object_insertion": 30}
TARGET_MASK_PER_STRATUM = {"small_under_10pct": 30, "medium_10_to_30pct": 40, "large_over_30pct": 30}

# v1 catalog (verified_candidate_catalog.json) is SUPERSEDED: its 440 entries fail the
# eligibility rules below (see catalog_eligibility_audit.json). v2 must be built with
# scripts/research/build_verified_candidate_catalog.py on a host that can reach the sources.
DEFAULT_CATALOG_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json"

# Committed map of the 684 historical Option P sources (all COCO 2017 image IDs, incl. the
# 343 retired locked-test sources). Only IDs are read; no locked-test image or feature.
HISTORICAL_SOURCE_MAP = REPO_ROOT / "research/evidence/phase-4c.0/real-variant-map.json"

# Locked execution policy (Protocol v1.2: Colab T4 / A100; SDXL fp16 needs a >=12 GB GPU).
MIN_GPU_VRAM_BYTES = 12 * 1024**3

# ---- Pre-registered candidate eligibility rules (independent of any detector output) ----
# Use = create AI-edited derivatives and export/share the edited cohort for research.
CREATOR_PLACEHOLDER_RE = re.compile(r"^(flickr_contributor_|photographer_|synthetic_author|unknown|anonymous)", re.I)
CC_LICENSE_RE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)/?$")
CC_USAGE_POLICY = {
    "by": ("ELIGIBLE", "Attribution required (creator, source, license link); indicate changes."),
    "by-sa": ("ELIGIBLE_WITH_POLICY", "Attribution required; indicate changes; edited derivatives shared only under CC BY-SA (ShareAlike)."),
    "by-nc": ("ELIGIBLE_WITH_POLICY", "Attribution required; indicate changes; NonCommercial research use only."),
    "by-nc-sa": ("ELIGIBLE_WITH_POLICY", "Attribution required; indicate changes; NonCommercial research use only; edited derivatives shared only under CC BY-NC-SA (ShareAlike)."),
}
CC_NO_DERIVATIVES = {"by-nd", "by-nc-nd"}
# Acquisition channels and whether their own terms cover creating + sharing edited derivatives.
CHANNEL_TERMS = {
    "coco_2017_image_info": True,  # per-image Flickr CC license governs; COCO terms defer to Flickr
    "wikimedia_commons_api": True,  # MediaWiki / Wikimedia Commons photographic works under CC
    # Unsplash Dataset Terms s.2A/s.3A-B: Lite = download/store + internal ML training only;
    # disseminating/redistributing Licensed Data is prohibited without written permission.
    "unsplash_lite_dataset": False,
    "unsplash_lite_dataset_hf_mirror": False,
}
REQUIRED_PROVENANCE_FIELDS = (
    "download_url", "source_page_url", "license_name", "license_url", "license_version",
    "license_evidence_source", "provenance_checked_at_utc", "download_rendition",
)
SOURCE_KEY_NAMESPACES = ("coco", "flickr", "commons", "wikimedia", "unsplash")


class DetectorIsolationViolationError(RuntimeError):
    """Raised if any detector model or scoring routine is invoked during acquisition/QC."""


class AcquisitionQCFailureError(ValueError):
    """Raised when an acquired sample fails technical quality control."""


class GenerationContractError(RuntimeError):
    """Raised when an inpainting engine or generation step violates the protocol contract (e.g. invalid dimensions or mode)."""


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


class CandidateEligibilityError(ValueError):
    """Raised when a catalog cannot supply the pre-registered quota of eligible candidates."""


class ContentGroundingError(ValueError):
    """Raised when production generation lacks a reviewed, content-grounded edit instruction."""


class RunBindingMismatchError(RuntimeError):
    """Raised when existing run artifacts belong to a different run/commit/catalog/plan."""


class RunAuditError(RuntimeError):
    """Raised when a run directory does not prove a complete, correctly bound acquisition."""


class ModelPreflightError(RuntimeError):
    """Raised when inpainting model metadata, configuration, or weight access check fails."""

    def __init__(
        self,
        message: str,
        repo_id: str = "",
        step: str = "",
        error_category: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.repo_id = repo_id
        self.step = step
        self.error_category = error_category
        self.status_code = status_code


def load_historical_source_keys(source_map_path: Path | str = HISTORICAL_SOURCE_MAP) -> set[str]:
    """Namespaced identifiers ('coco:<image_id>') of all 684 historical Option P sources."""
    data = json.loads(Path(source_map_path).read_text(encoding="utf-8"))
    return {f"coco:{int(sid)}" for sid in data["source_mapping"]}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_candidate_eligibility(entry: dict[str, Any], historical_keys: set[str]) -> dict[str, Any]:
    """Apply the pre-registered eligibility rules to one catalog entry.

    A field counts as verified only when the entry carries the verification marker and an
    evidence URL from the source; a non-empty value alone is never enough.
    """
    reasons: list[str] = []

    channel = entry.get("acquisition_channel")
    if channel not in CHANNEL_TERMS:
        reasons.append("ACQUISITION_CHANNEL_UNDECLARED")
    elif not CHANNEL_TERMS[channel]:
        reasons.append("CHANNEL_TERMS_DO_NOT_COVER_EDITED_EXPORT")

    raw_keys = entry.get("source_keys")
    keys = set(raw_keys) if isinstance(raw_keys, list) else set()
    if not keys or any(
        not isinstance(k, str) or k.partition(":")[0] not in SOURCE_KEY_NAMESPACES or not k.partition(":")[2]
        for k in keys
    ):
        reasons.append("SOURCE_KEYS_MISSING")
    if keys & historical_keys:
        reasons.append("HISTORICAL_OPTION_P_OVERLAP")

    creator = entry.get("creator_name") or ""
    if not creator:
        reasons.append("CREATOR_MISSING")
    if CREATOR_PLACEHOLDER_RE.match(creator) or CREATOR_PLACEHOLDER_RE.match(entry.get("author") or ""):
        reasons.append("CREATOR_PLACEHOLDER")
    if entry.get("creator_verification") != "VERIFIED_FROM_SOURCE" or not str(
        entry.get("creator_evidence_url") or ""
    ).startswith(("https://", "http://")):
        reasons.append("CREATOR_UNVERIFIED")

    for f in REQUIRED_PROVENANCE_FIELDS:
        if not entry.get(f):
            reasons.append(f"PROVENANCE_FIELD_MISSING:{f}")
    if entry.get("published_date") and not entry.get("published_date_source"):
        reasons.append("DATE_PROVENANCE_UNLABELED")
    if entry.get("date_captured") and not entry.get("date_captured_source"):
        reasons.append("DATE_PROVENANCE_UNLABELED")

    usage_policy = ""
    status = "EXCLUDED"
    m = CC_LICENSE_RE.search(str(entry.get("license_url") or ""))
    if not m:
        reasons.append("LICENSE_NOT_IN_POLICY")
    else:
        code, version = m.groups()
        if entry.get("license_version") and entry["license_version"] != version:
            reasons.append("LICENSE_VERSION_MISMATCH")
        if code in CC_NO_DERIVATIVES:
            if entry.get("special_permission_evidence"):
                status, usage_policy = "ELIGIBLE_WITH_POLICY", (
                    f"Separate permission recorded ({entry['special_permission_evidence']}); attribution required."
                )
            else:
                reasons.append("LICENSE_ND_NO_EDITED_EXPORT")
        elif code in CC_USAGE_POLICY:
            status, usage_policy = CC_USAGE_POLICY[code]
        else:
            reasons.append("LICENSE_NOT_IN_POLICY")

    if reasons:
        return {"status": "EXCLUDED", "reasons": sorted(set(reasons)), "usage_policy": ""}
    return {"status": status, "reasons": [], "usage_policy": usage_policy}


def check_gpu_policy(cuda_available: bool, total_vram_bytes: int, device_name: str) -> None:
    """Fail closed unless the runtime matches the locked GPU execution policy."""
    if not cuda_available:
        raise RuntimeError("Execution policy requires a CUDA GPU (Colab T4/A100); none detected.")
    if total_vram_bytes < MIN_GPU_VRAM_BYTES:
        raise RuntimeError(
            f"Execution policy requires >= {MIN_GPU_VRAM_BYTES / 1024**3:.0f} GiB VRAM; "
            f"{device_name} has {total_vram_bytes / 1024**3:.2f} GiB."
        )


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
    source_keys: tuple[str, ...] = ()
    creator_url: str = ""
    license_url: str = ""
    usage_policy: str = ""
    date_captured: str = ""
    provenance_checked_at_utc: str = ""
    download_rendition: str = ""
    target_description: str = ""
    placement_rationale: str = ""
    target_bbox_xyxy: tuple[int, int, int, int] | None = None
    mask_bbox_xyxy: tuple[int, int, int, int] | None = None


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
    run_id: str = ""
    source_keys: tuple[str, ...] | list[str] = ()
    creator_url: str = ""
    license_url: str = ""
    usage_policy: str = ""
    date_captured: str = ""
    download_rendition: str = ""
    downloaded_at_utc: str = ""
    raw_width: int = 0
    raw_height: int = 0
    engine_config: dict[str, Any] | None = None
    target_description: str = ""
    placement_rationale: str = ""
    target_bbox_xyxy: tuple[int, int, int, int] | None = None
    mask_bbox_xyxy: tuple[int, int, int, int] | None = None


def download_authentic_image(
    spec: CandidateSpec,
    timeout: float = 25.0,
    max_retries: int = 2,
    hist_keys: set[str] | None = None,
    hist_hashes: set[str] | None = None,
    allow_synthetic: bool = False,
) -> tuple[Image.Image, bytes, str]:
    """Download authentic photographic image with fail-closed integrity and provenance guards.

    Rules:
    - If spec.is_synthetic is True and allow_synthetic is False: FAIL-CLOSED.
    - If required provenance (download_url, license_name, author) is missing: FAIL-CLOSED.
    - If download encounters HTTP 404, network error, or timeout: FAIL-CLOSED (no synthetic fallback).
    - Checks namespaced source keys (coco:/flickr:/unsplash:) against historical Option P.
    - Checks raw byte hash against historical hashes (byte-identical files only).
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

    # Disjoint check on namespaced source identifiers before network
    overlap = set(spec.source_keys) & (hist_keys or set())
    if overlap:
        raise SourceOverlapError(f"Candidate {spec.candidate_id} source keys {sorted(overlap)} overlap historical Option P.")

    # Support local file path or file:// URL for testing/offline mirrors
    data: bytes | None = None
    url = spec.download_url

    if url.startswith("file://") or Path(url).is_file():
        local_p = Path(url[7:] if url.startswith("file://") else url)
        if not local_p.is_file():
            raise ImageDownloadError(f"Local file not found for candidate {spec.candidate_id}: {local_p}")
        data = local_p.read_bytes()
    else:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "ForensicsWebLabResearch/2.0 "
                    "(https://github.com/nomozer/forensics-web-lab; contact: nomozer on GitHub) Python/3.12"
                )
            },
        )
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


def quota_layout(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Per-stratum (modification, mask) labels for 110 slots.

    Slots 0-99 carry exactly the target quota (40/30/30 x 30/40/30, each dimension permuted).
    Slots 100-109 hold every (modification, mask) cell once plus one extra
    (object_replacement, medium) - marginals 4/3/3 and 3/4/3 - so a failed core slot of any
    cell can be replaced without breaking either quota. (Permuting all 110 slots jointly left
    the first 100 off-quota and made full runs unsatisfiable by in-order replacement.)
    """
    core_mods = list(rng.permutation([k for k, n in TARGET_MOD_PER_STRATUM.items() for _ in range(n)]))
    core_masks = list(rng.permutation([k for k, n in TARGET_MASK_PER_STRATUM.items() for _ in range(n)]))
    cells = [(m, a) for m in MODIFICATION_TYPES for a in MASK_AREA_CLASSES] + [("object_replacement", "medium_10_to_30pct")]
    buffer = [cells[i] for i in rng.permutation(len(cells))]
    return core_mods + [m for m, _ in buffer], core_masks + [a for _, a in buffer]


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
    catalog_synthetic = bool(catalog_data.get("is_synthetic", False))
    historical_keys = load_historical_source_keys()

    # Eligible entries in frozen catalog order; excluded entries are skipped, never edited.
    need = buffer_per_stratum * 2
    second_key = "commons_candidates" if "commons_candidates" in catalog_data else "unsplash_candidates"
    source_origin_name = "wikimedia_commons" if second_key == "commons_candidates" else "unsplash_verified"
    stratum_prefix = "commons" if second_key == "commons_candidates" else "unsplash"

    pools: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key in ("coco_candidates", second_key):
        evaluated = [(e, evaluate_candidate_eligibility(e, historical_keys)) for e in catalog_data.get(key, [])]
        eligible = [(e, r) for e, r in evaluated if r["status"] != "EXCLUDED"]
        if len(eligible) < need:
            reasons: dict[str, int] = {}
            for _, r in evaluated:
                for reason in r["reasons"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
            raise CandidateEligibilityError(
                f"{cat_path.name}: {key} has {len(eligible)}/{need} eligible candidates "
                f"({len(evaluated)} evaluated); exclusion reasons: {reasons}"
            )
        pools[key] = eligible[:need]
    coco_catalog = pools["coco_candidates"]
    second_catalog = pools[second_key]

    rng = np.random.default_rng(seed)
    candidates: list[CandidateSpec] = []


    strata_configs = [
        ("coco_sd2", "coco_2017", "stable_diffusion_2_inpainting", coco_catalog[:buffer_per_stratum]),
        ("coco_sdxl", "coco_2017", "sdxl_inpainting", coco_catalog[buffer_per_stratum : buffer_per_stratum * 2]),
        (f"{stratum_prefix}_sd2", source_origin_name, "stable_diffusion_2_inpainting", second_catalog[:buffer_per_stratum]),
        (f"{stratum_prefix}_sdxl", source_origin_name, "sdxl_inpainting", second_catalog[buffer_per_stratum : buffer_per_stratum * 2]),
    ]

    global_pool_idx = 0
    for stratum_id, source_origin, tool_key, raw_pool in strata_configs:
        perm_mods, perm_masks = quota_layout(rng)

        for idx, (item, eligibility) in enumerate(raw_pool):
            global_pool_idx += 1
            cid = f"IND_{stratum_id.upper()}_{idx+1:03d}"
            mod_type = perm_mods[idx]
            mask_class = perm_masks[idx]

            gen_seed = int(seed + global_pool_idx * 101)

            candidates.append(
                CandidateSpec(
                    candidate_id=cid,
                    stratum_id=stratum_id,
                    source_origin=source_origin,
                    tool_key=tool_key,
                    modification_type=mod_type,
                    mask_area_class=mask_class,
                    origin_id=item.get("origin_id") or min(item["source_keys"]),
                    author=item["creator_name"],
                    origin_url=item["source_page_url"],
                    download_url=item["download_url"],
                    license_name=item["license_name"],
                    license_evidence_source=item["license_evidence_source"],
                    published_date=item.get("published_date") or "",
                    # The catalog plan allocates quotas only. Production prompts and
                    # regions are supplied later by a content-grounded edit plan.
                    prompt="",
                    generation_seed=gen_seed,
                    pool_index=idx,
                    is_synthetic=catalog_synthetic,
                    evidence_class="synthetic" if catalog_synthetic else "verified_real_catalog",
                    eligible_for_independent_cohort=not catalog_synthetic,
                    source_keys=tuple(sorted(item["source_keys"])),
                    creator_url=item.get("creator_url", ""),
                    license_url=item["license_url"],
                    usage_policy=eligibility["usage_policy"],
                    date_captured=item.get("date_captured") or "",
                    provenance_checked_at_utc=item["provenance_checked_at_utc"],
                    download_rendition=item["download_rendition"],
                )
            )

    return candidates


def _validated_bbox(value: Any, field_name: str, candidate_id: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContentGroundingError(f"{candidate_id}: {field_name} must contain four integer coordinates")
    try:
        x0, y0, x1, y1 = (int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ContentGroundingError(f"{candidate_id}: {field_name} must contain integers") from exc
    width, height = TARGET_CANVAS_SIZE
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ContentGroundingError(
            f"{candidate_id}: {field_name} {(x0, y0, x1, y1)} is outside the normalized 512x512 canvas"
        )
    return x0, y0, x1, y1


def _validate_content_grounded_candidate(spec: CandidateSpec) -> None:
    if not spec.prompt.strip():
        raise ContentGroundingError(f"{spec.candidate_id}: prompt is required")
    if not spec.target_description.strip():
        raise ContentGroundingError(f"{spec.candidate_id}: target_description is required")
    if not spec.placement_rationale.strip():
        raise ContentGroundingError(f"{spec.candidate_id}: placement_rationale is required")
    target = _validated_bbox(spec.target_bbox_xyxy, "target_bbox_xyxy", spec.candidate_id)
    mask = _validated_bbox(spec.mask_bbox_xyxy, "mask_bbox_xyxy", spec.candidate_id)
    tx0, ty0, tx1, ty1 = target
    mx0, my0, mx1, my1 = mask
    if not (mx0 <= tx0 < tx1 <= mx1 and my0 <= ty0 < ty1 <= my1):
        raise ContentGroundingError(f"{spec.candidate_id}: target_bbox_xyxy must be contained by mask_bbox_xyxy")
    ratio = ((mx1 - mx0) * (my1 - my0)) / (TARGET_CANVAS_SIZE[0] * TARGET_CANVAS_SIZE[1])
    low, high = MASK_AREA_BOUNDS[spec.mask_area_class]
    if not low <= ratio <= high:
        raise ContentGroundingError(
            f"{spec.candidate_id}: planned mask ratio {ratio:.4f} outside [{low}, {high}] "
            f"for {spec.mask_area_class}"
        )


def load_content_grounded_edit_plan(
    candidates: Sequence[CandidateSpec],
    edit_plan_path: Path | str,
    target_per_stratum: int,
    require_human_approval: bool = False,
) -> list[CandidateSpec]:
    """Bind reviewed prompts and normalized-canvas regions to an allocation plan.

    The edit-plan file selects the only candidates that may be attempted. This
    deliberately disables unreviewed automatic replacements: a rejected pilot
    candidate produces a quota deficit instead of silently trying a random region.
    """
    data = json.loads(Path(edit_plan_path).read_text(encoding="utf-8"))
    if require_human_approval and data.get("human_review_status") != "APPROVED":
        raise ContentGroundingError(
            "content-grounded edit plan is not human-approved; generation remains blocked"
        )
    entries = data.get("candidates")
    if not isinstance(entries, list):
        raise ContentGroundingError("edit plan must contain a candidates list")
    if require_human_approval:
        unresolved = [
            str(entry.get("candidate_id", ""))
            for entry in entries
            if entry.get("instruction_review", {}).get("agent_status") == "NEEDS_USER_DECISION"
        ]
        if unresolved:
            raise ContentGroundingError(
                f"content-grounded edit plan has unresolved instruction decisions: {unresolved}"
            )
    expected_attempts = target_per_stratum * len(STRATA_KEYS)
    if data.get("attempt_budget") != expected_attempts or len(entries) != expected_attempts:
        raise ContentGroundingError(
            f"edit plan must bind exactly {expected_attempts} candidates/attempts; "
            f"attempt_budget={data.get('attempt_budget')!r}, candidates={len(entries)}"
        )
    if data.get("automatic_replacement") is not False:
        raise ContentGroundingError("edit plan must explicitly set automatic_replacement=false")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[CandidateSpec] = []
    seen: set[str] = set()
    for entry in entries:
        candidate_id = str(entry.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ContentGroundingError(f"duplicate or empty candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        if candidate_id not in by_id:
            raise ContentGroundingError(f"{candidate_id}: not present in the verified allocation plan")
        base = by_id[candidate_id]
        for locked_field in ("stratum_id", "tool_key", "modification_type", "mask_area_class"):
            if entry.get(locked_field) != getattr(base, locked_field):
                raise ContentGroundingError(f"{candidate_id}: edit plan changes locked field {locked_field}")
        grounded = replace(
            base,
            prompt=str(entry.get("prompt", "")),
            target_description=str(entry.get("target_description", "")),
            placement_rationale=str(entry.get("placement_rationale", "")),
            target_bbox_xyxy=_validated_bbox(entry.get("target_bbox_xyxy"), "target_bbox_xyxy", candidate_id),
            mask_bbox_xyxy=_validated_bbox(entry.get("mask_bbox_xyxy"), "mask_bbox_xyxy", candidate_id),
        )
        _validate_content_grounded_candidate(grounded)
        selected.append(grounded)

    counts = {key: 0 for key in STRATA_KEYS}
    for spec in selected:
        counts[spec.stratum_id] += 1
    expected = {key: target_per_stratum for key in STRATA_KEYS}
    if counts != expected:
        raise ContentGroundingError(f"edit plan stratum counts {counts} do not match required {expected}")
    actual_quota_summary = {
        "stratum_counts": counts,
        "modification_type_counts": {
            key: sum(spec.modification_type == key for spec in selected) for key in MODIFICATION_TYPES
        },
        "mask_area_class_counts": {
            key: sum(spec.mask_area_class == key for spec in selected) for key in MASK_AREA_CLASSES
        },
    }
    if data.get("locked_quota_summary") != actual_quota_summary:
        raise ContentGroundingError(
            f"edit plan locked_quota_summary does not match selected candidates: {actual_quota_summary}"
        )
    return selected


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

    strata_definitions = [
        ("coco_sd2", "coco_2017", "stable_diffusion_2_inpainting"),
        ("coco_sdxl", "coco_2017", "sdxl_inpainting"),
        ("commons_sd2", "wikimedia_commons", "stable_diffusion_2_inpainting"),
        ("commons_sdxl", "wikimedia_commons", "sdxl_inpainting"),
    ]

    global_pool_idx = 0
    for stratum_id, source_origin, tool_key in strata_definitions:
        perm_mods, perm_masks = quota_layout(rng)

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


def build_content_grounded_mask(spec: CandidateSpec) -> tuple[Image.Image, float]:
    """Build the exact binary rectangle registered on the normalized canvas."""
    _validate_content_grounded_candidate(spec)
    x0, y0, x1, y1 = spec.mask_bbox_xyxy  # type: ignore[misc]
    mask_arr = np.zeros((TARGET_CANVAS_SIZE[1], TARGET_CANVAS_SIZE[0]), dtype=np.uint8)
    mask_arr[y0:y1, x0:x1] = 255
    return Image.fromarray(mask_arr, mode="L"), float(np.mean(mask_arr == 255))


def composite_generated_region(
    authentic_img: Image.Image,
    generated_img: Image.Image,
    mask_img: Image.Image,
) -> Image.Image:
    """Keep generated pixels only inside the registered binary request mask."""
    if authentic_img.size != generated_img.size or authentic_img.size != mask_img.size:
        raise GenerationContractError("Cannot composite images and mask with different sizes")
    if authentic_img.mode != "RGB" or generated_img.mode != "RGB" or mask_img.mode != "L":
        raise GenerationContractError("Compositing requires RGB authentic/generated images and an L mask")
    mask_arr = np.array(mask_img)
    if not set(np.unique(mask_arr)).issubset({0, 255}):
        raise GenerationContractError("Compositing requires a binary mask containing only 0 and 255")
    return Image.composite(generated_img, authentic_img, mask_img)


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
        raise GenerationContractError(
            f"Authentic image size/mode invalid: {authentic_img.size}, {authentic_img.mode}; expected {TARGET_CANVAS_SIZE}, RGB"
        )

    if edited_img.size != TARGET_CANVAS_SIZE or edited_img.mode != "RGB":
        raise GenerationContractError(
            f"Edited image size/mode invalid: {edited_img.size}, {edited_img.mode}; expected {TARGET_CANVAS_SIZE}, RGB"
        )

    if mask_img.size != TARGET_CANVAS_SIZE or mask_img.mode != "L":
        raise GenerationContractError(
            f"Mask size/mode invalid: {mask_img.size}, {mask_img.mode}; expected {TARGET_CANVAS_SIZE}, L"
        )

    if authentic_img.size[0] * authentic_img.size[1] > MAX_PIXEL_DECOMPRESSION_BOMB:
        return False, "Decompression bomb limit exceeded", 0.0

    mask_arr = np.array(mask_img)
    unique_vals = set(np.unique(mask_arr))
    if not unique_vals.issubset({0, 255}):
        return False, f"Mask contains non-binary values: {unique_vals}", 0.0

    mask_pixels = int(np.sum(mask_arr == 255))
    total_pixels = mask_arr.size
    ratio = float(mask_pixels / total_pixels)

    low, high = MASK_AREA_BOUNDS[mask_area_class]
    if not (low <= ratio <= high):
        return False, f"Mask ratio {ratio:.4f} outside bounds [{low}, {high}] for {mask_area_class}", ratio

    auth_arr = np.array(authentic_img, dtype=np.float32)
    edit_arr = np.array(edited_img, dtype=np.float32)

    if np.isnan(auth_arr).any() or np.isnan(edit_arr).any() or np.isinf(auth_arr).any() or np.isinf(edit_arr).any():
        return False, "Pixel arrays contain NaN or Inf values", ratio

    auth_std = float(np.std(auth_arr))
    edit_std = float(np.std(edit_arr))
    if auth_std < NON_BLANK_STD_THRESHOLD or edit_std < NON_BLANK_STD_THRESHOLD:
        return False, f"Degenerate solid or near-blank image detected (auth_std={auth_std:.2f}, edit_std={edit_std:.2f})", ratio

    masked_diff = np.abs(auth_arr - edit_arr)[mask_arr == 255]
    if len(masked_diff) == 0:
        return False, "Mask has zero foreground pixels", ratio

    mean_masked_diff = float(np.mean(masked_diff))
    if mean_masked_diff < MIN_MASKED_PIXEL_DELTA_L1:
        return False, f"Edited image exhibits near-zero change in masked region (mean_diff={mean_masked_diff:.2f})", ratio

    unmasked_diff = np.abs(auth_arr - edit_arr)[mask_arr == 0]
    mean_unmasked_diff = float(np.mean(unmasked_diff)) if len(unmasked_diff) else 0.0
    if mean_unmasked_diff > MAX_UNMASKED_PIXEL_DELTA_L1:
        return False, (
            f"Edited image changes pixels outside request mask "
            f"(mean_diff={mean_unmasked_diff:.2f}, max={MAX_UNMASKED_PIXEL_DELTA_L1:.2f})"
        ), ratio

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

    def describe(self) -> dict[str, Any]:
        return {"engine": "MockInpaintingEngine", "is_synthetic": True}

    def peak_vram_bytes(self) -> int | str:
        return "NOT_MEASURED"


INPAINTING_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "stable_diffusion_2_inpainting": {
        "tool_key": "stable_diffusion_2_inpainting",
        "name": "Stable Diffusion 2 Inpainting",
        "checkpoint": "sd2-community/stable-diffusion-2-inpainting",
        "pinned_revision": "5f74973cbb64c8568780732c17f43eb269d63a0d",
        "checkpoint_source_type": "community_mirror",
        "upstream_original_checkpoint": "stabilityai/stable-diffusion-2-inpainting",
        "pipeline_class_name": "StableDiffusionInpaintPipeline",
        "scheduler_class_name": "DDIMScheduler",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "required_configs": [
            "model_index.json",
            "unet/config.json",
            "scheduler/scheduler_config.json",
        ],
        "weight_head_file": "unet/diffusion_pytorch_model.fp16.safetensors",
        "weight_head_lfs_oid": "29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8",
        "weight_head_size": 1731933536,
        "component_lfs_hashes": {
            "unet/diffusion_pytorch_model.fp16.safetensors": "29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8",
            "512-inpainting-ema.safetensors": "b29e2ed9a8fe58e76f7e801bda091d23738bd74c1da3f339bcbe2d40922fcb60",
            "vae/diffusion_pytorch_model.fp16.safetensors": "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
            "text_encoder/model.fp16.safetensors": "681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668",
        },
        "extra_pipeline_kwargs": {"safety_checker": None},
        "target_height": 512,
        "target_width": 512,
        "license": "CreativeML OpenRAIL++",
    },
    "sdxl_inpainting": {
        "tool_key": "sdxl_inpainting",
        "name": "SDXL Inpainting 1.0",
        "checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pinned_revision": "115134f363124c53c7d878647567d04daf26e41e",
        "checkpoint_source_type": "official_diffusers",
        "upstream_original_checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pipeline_class_name": "StableDiffusionXLInpaintPipeline",
        "scheduler_class_name": "EulerDiscreteScheduler",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "required_configs": [
            "model_index.json",
            "unet/config.json",
            "scheduler/scheduler_config.json",
        ],
        "weight_head_file": "unet/diffusion_pytorch_model.fp16.safetensors",
        "weight_head_lfs_oid": "6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f",
        "weight_head_size": 5135178560,
        "component_lfs_hashes": {
            "unet/diffusion_pytorch_model.fp16.safetensors": "6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f",
        },
        "extra_pipeline_kwargs": {"variant": "fp16"},
        "target_height": 512,
        "target_width": 512,
        "license": "CreativeML OpenRAIL++",
    },
}


class _HeadRequest(urllib.request.Request):
    def get_method(self) -> str:
        return "HEAD"


def verify_model_access_preflight(
    tool_keys: Sequence[str] | None = None,
    timeout: float = 15.0,
    http_opener: Any = None,
) -> dict[str, dict[str, Any]]:
    """Preflight check verifying model repository metadata, configurations, and file access.

    Fails closed before any acquisition run directory is created.
    Distinguishes repository unavailable (deleted/gated/private/deprecated), 404 not found,
    forbidden (403), invalid token, and network errors.

    Resolves full revision SHA and returns mapping for bound model execution.
    """
    keys = list(tool_keys or INPAINTING_MODEL_REGISTRY.keys())
    for k in keys:
        if k not in INPAINTING_MODEL_REGISTRY:
            raise ModelPreflightError(
                f"Unknown inpainting model tool key: '{k}'",
                repo_id=k,
                step="registry_lookup",
                error_category="unknown_tool_key",
            )

    results: dict[str, dict[str, Any]] = {}
    opener = http_opener or urllib.request.build_opener()

    for tool_key in keys:
        reg = INPAINTING_MODEL_REGISTRY[tool_key]
        checkpoint = reg["checkpoint"]

        # Step 1: Query Hugging Face Model API Metadata
        meta_url = f"https://huggingface.co/api/models/{checkpoint}"
        req_meta = urllib.request.Request(meta_url, headers={"User-Agent": "forensics-web-lab/1.0"})
        try:
            with opener.open(req_meta, timeout=timeout) as resp:
                raw_meta = resp.read()
                meta_json = json.loads(raw_meta.decode("utf-8"))
        except urllib.error.HTTPError as e:
            status_code = e.code
            try:
                body_str = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_str = ""
            if status_code == 401:
                # Unauthenticated 401 on HF means repo is unavailable, gated, private, or deprecated
                raise ModelPreflightError(
                    f"Model repository '{checkpoint}' is unavailable (HTTP 401 Unauthorized, unauthenticated request). "
                    f"Repository may be gated, private, or deprecated upstream by owner. Raw response: {body_str[:120]}",
                    repo_id=checkpoint,
                    step="metadata_api",
                    error_category="repository_unavailable",
                    status_code=401,
                ) from e
            elif status_code == 404:
                raise ModelPreflightError(
                    f"Model repository '{checkpoint}' was not found (HTTP 404 Not Found). Raw response: {body_str[:120]}",
                    repo_id=checkpoint,
                    step="metadata_api",
                    error_category="repository_not_found",
                    status_code=404,
                ) from e
            elif status_code == 403:
                raise ModelPreflightError(
                    f"Access to model repository '{checkpoint}' is forbidden (HTTP 403 Forbidden). Raw response: {body_str[:120]}",
                    repo_id=checkpoint,
                    step="metadata_api",
                    error_category="forbidden",
                    status_code=403,
                ) from e
            else:
                raise ModelPreflightError(
                    f"HTTP error {status_code} while querying metadata for model '{checkpoint}': {e}. Raw response: {body_str[:120]}",
                    repo_id=checkpoint,
                    step="metadata_api",
                    error_category=f"http_{status_code}",
                    status_code=status_code,
                ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModelPreflightError(
                f"Network or socket error querying metadata for model '{checkpoint}': {e}",
                repo_id=checkpoint,
                step="metadata_api",
                error_category="network_error",
            ) from e
        except json.JSONDecodeError as e:
            raise ModelPreflightError(
                f"Failed to parse JSON metadata for model '{checkpoint}': {e}",
                repo_id=checkpoint,
                step="metadata_api",
                error_category="malformed_json",
            ) from e

        # Validate metadata payload
        if meta_json.get("gated"):
            raise ModelPreflightError(
                f"Model repository '{checkpoint}' is gated and requires authenticated license agreement.",
                repo_id=checkpoint,
                step="metadata_api",
                error_category="gated_repository",
            )
        if meta_json.get("private"):
            raise ModelPreflightError(
                f"Model repository '{checkpoint}' is private.",
                repo_id=checkpoint,
                step="metadata_api",
                error_category="private_repository",
            )

        resolved_revision = meta_json.get("sha")
        if not resolved_revision or not isinstance(resolved_revision, str) or len(resolved_revision) < 20:
            raise ModelPreflightError(
                f"Model repository '{checkpoint}' returned invalid revision SHA: {resolved_revision}",
                repo_id=checkpoint,
                step="metadata_api",
                error_category="invalid_revision",
            )

        # Step 2: Verify essential configuration files via raw endpoint
        verified_configs: list[str] = []
        for cfg in reg.get("required_configs", []):
            cfg_url = f"https://huggingface.co/{checkpoint}/raw/{resolved_revision}/{cfg}"
            req_cfg = urllib.request.Request(cfg_url, headers={"User-Agent": "forensics-web-lab/1.0"})
            try:
                with opener.open(req_cfg, timeout=timeout) as resp:
                    cfg_data = json.loads(resp.read().decode("utf-8"))
                    if not isinstance(cfg_data, dict):
                        raise ValueError(f"Config {cfg} is not a JSON object")
                    verified_configs.append(cfg)
            except Exception as e:
                raise ModelPreflightError(
                    f"Failed to retrieve or validate config file '{cfg}' in '{checkpoint}' at revision '{resolved_revision}': {e}",
                    repo_id=checkpoint,
                    step="config_access",
                    error_category="config_error",
                ) from e

        # Step 3: Verify weight file accessibility via HEAD request
        weight_file = reg.get("weight_head_file")
        if weight_file:
            weight_url = f"https://huggingface.co/{checkpoint}/resolve/{resolved_revision}/{weight_file}"
            req_weight = _HeadRequest(weight_url, headers={"User-Agent": "forensics-web-lab/1.0"})
            try:
                with opener.open(req_weight, timeout=timeout) as resp:
                    if resp.status not in (200, 302, 307):
                        raise RuntimeError(f"Unexpected HTTP status {resp.status}")
            except Exception as e:
                raise ModelPreflightError(
                    f"Failed to verify access to weight file '{weight_file}' in '{checkpoint}' at revision '{resolved_revision}': {e}",
                    repo_id=checkpoint,
                    step="weight_file_access",
                    error_category="weight_access_error",
                ) from e

        results[tool_key] = {
            "tool_key": tool_key,
            "checkpoint": checkpoint,
            "pinned_revision": reg["pinned_revision"],
            "resolved_revision": resolved_revision,
            "checkpoint_source_type": reg["checkpoint_source_type"],
            "upstream_original_checkpoint": reg["upstream_original_checkpoint"],
            "pipeline_class": reg["pipeline_class_name"],
            "scheduler_class": reg["scheduler_class_name"],
            "num_inference_steps": reg["num_inference_steps"],
            "guidance_scale": reg["guidance_scale"],
            "verified_configs": verified_configs,
            "weight_file_verified": weight_file,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    return results


class DiffusersInpaintingEngine:
    """Production GPU inpainting engine using HuggingFace Diffusers for SD2 and SDXL."""

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
            "engine": "DiffusersInpaintingEngine",
            "tool_key": tool_key,
            "checkpoint": checkpoint,
            "checkpoint_source_type": reg["checkpoint_source_type"],
            "upstream_original_checkpoint": reg["upstream_original_checkpoint"],
            "pinned_revision": reg["pinned_revision"],
            "resolved_revision": self.revision,
            "pipeline": pipeline_cls.__name__,
            "scheduler": scheduler_cls.__name__,
            "num_inference_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "dtype": str(self.dtype),
            "variant": extra.get("variant"),
            "target_height": TARGET_CANVAS_SIZE[1],
            "target_width": TARGET_CANVAS_SIZE[0],
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
        height: int = TARGET_CANVAS_SIZE[1],
        width: int = TARGET_CANVAS_SIZE[0],
    ) -> Image.Image:
        if authentic_img.size != TARGET_CANVAS_SIZE or authentic_img.mode != "RGB":
            raise GenerationContractError(
                f"Input authentic image size/mode invalid: {authentic_img.size}, {authentic_img.mode}; "
                f"expected {TARGET_CANVAS_SIZE}, RGB"
            )
        if mask_img.size != TARGET_CANVAS_SIZE or mask_img.mode not in ("L", "1"):
            raise GenerationContractError(
                f"Input mask image size/mode invalid: {mask_img.size}, {mask_img.mode}; "
                f"expected {TARGET_CANVAS_SIZE}, L"
            )

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
                f"Diffusers inpainting pipeline for tool '{self.tool_key}' produced invalid output "
                f"size/mode {result.size}, {result.mode}; expected ({width}, {height}), RGB"
            )
        return result

    def describe(self) -> dict[str, Any]:
        return dict(self.config)

    def peak_vram_bytes(self) -> int | str:
        if self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated())
        return "NOT_MEASURED"

    def unload(self) -> None:
        """Release GPU memory cleanly."""
        del self.pipeline
        self.pipeline = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


MANIFEST_FIELDS = [
    "source_id", "label", "label_id", "image_relpath", "image_sha256", "tool", "mask_relpath",
    "width", "height", "stratum", "origin_id", "license",
]


def _manifest_rows(receipt: AcquisitionReceipt) -> list[dict[str, Any]]:
    common = {
        "source_id": receipt.source_id,
        "width": TARGET_CANVAS_SIZE[0],
        "height": TARGET_CANVAS_SIZE[1],
        "stratum": receipt.stratum_id,
        "origin_id": receipt.origin_id,
        "license": receipt.license_name,
    }
    return [
        common | {"label": "authentic", "label_id": 0, "image_relpath": receipt.authentic_relpath,
                  "image_sha256": receipt.master_sha256, "tool": None, "mask_relpath": None},
        common | {"label": "ai_edited", "label_id": 1, "image_relpath": receipt.edited_relpath,
                  "image_sha256": receipt.edited_sha256, "tool": receipt.tool_key, "mask_relpath": receipt.mask_relpath},
    ]


def _check_run_binding(out_path: Path, run_binding: dict[str, Any], provenance_jsonl: Path) -> None:
    """Bind the output directory to exactly one run; refuse foreign or unbound artifacts."""
    binding_path = out_path / "run_binding.json"
    if binding_path.is_file():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != run_binding:
            diff = sorted(k for k in set(existing) | set(run_binding) if existing.get(k) != run_binding.get(k))
            raise RunBindingMismatchError(f"{out_path} is bound to another run (differs in {diff}).")
    elif provenance_jsonl.is_file() or (out_path / "run_receipt.json").is_file():
        raise RunBindingMismatchError(f"{out_path} holds unbound legacy artifacts; use a new output directory.")
    else:
        binding_path.write_text(json.dumps(run_binding, indent=2, sort_keys=True), encoding="utf-8")


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
    run_binding: dict[str, Any] | None = None,
    resolved_revisions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute independent cohort acquisition pipeline with fail-closed integrity.

    Modes:
    - 'production': Real photographic images only. Fails closed on download errors or synthetic sources.
    - 'fixture_test': Allows synthetic fixtures only if allow_synthetic=True. Emits provisional synthetic metadata.

    run_binding (run_id, git commit, protocol/catalog/plan hashes, mode, target) is written to
    run_binding.json; artifacts bound to anything else are refused. run_receipt.json is written
    only after the quota is met, so its presence proves completion of this bound run.
    """
    assert_detector_isolation()
    run_binding = dict(run_binding or {})
    run_id = str(run_binding.get("run_id", ""))

    if mode == "production" and allow_synthetic:
        raise SyntheticSourceProhibitedError(
            "allow_synthetic=True is forbidden when mode='production'! "
            "Production independent cohort must contain 100% verified real photographs."
        )

    # Candidate plan (validated before any artifact is written)
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
            if mode == "production":
                _validate_content_grounded_candidate(spec)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    provenance_jsonl = out_path / "provenance_ledger.jsonl"
    attempt_jsonl = out_path / "attempt_ledger.jsonl"
    _check_run_binding(out_path, run_binding, provenance_jsonl)
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Historical disjoint guards: namespaced source IDs from the committed source map;
    #    byte hashes only when the (Git-excluded) Option P manifest is present locally.
    hist_keys = load_historical_source_keys()
    hist_manifest = Path(historical_manifest_path or DEFAULT_HISTORICAL_MANIFEST)
    hist_hashes = load_historical_image_hashes(hist_manifest) if hist_manifest.is_file() else set()
    if not hist_hashes:
        print(f"NOTICE: historical byte-hash guard inactive ({hist_manifest} not present); disjointness rests on "
              "namespaced COCO IDs. Recorded in run_receipt.json historical_guard.")

    # 3. Setup default authentic provider
    if authentic_image_provider is None:
        if mode == "production":
            def real_downloader_provider(spec: CandidateSpec) -> tuple[Image.Image, bytes, str]:
                return download_authentic_image(
                    spec, hist_keys=hist_keys, hist_hashes=hist_hashes, allow_synthetic=False,
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

    # 5. Durable ledgers
    receipts: list[AcquisitionReceipt] = []
    completed_sources: set[str] = set()
    stratum_valid_counts: dict[str, int] = {k: 0 for k in STRATA_KEYS}
    mod_counts: dict[str, int] = {m: 0 for m in MODIFICATION_TYPES}
    mask_counts: dict[str, int] = {m: 0 for m in MASK_AREA_CLASSES}
    stratum_mod_counts: dict[str, dict[str, int]] = {k: {m: 0 for m in MODIFICATION_TYPES} for k in STRATA_KEYS}
    stratum_mask_counts: dict[str, dict[str, int]] = {k: {m: 0 for m in MASK_AREA_CLASSES} for k in STRATA_KEYS}
    manifest_rows: list[dict[str, Any]] = []

    def count_accepted(receipt: AcquisitionReceipt) -> None:
        receipts.append(receipt)
        completed_sources.add(receipt.source_id)
        stratum_valid_counts[receipt.stratum_id] += 1
        mod_counts[receipt.modification_type] += 1
        mask_counts[receipt.mask_area_class] += 1
        stratum_mod_counts[receipt.stratum_id][receipt.modification_type] += 1
        stratum_mask_counts[receipt.stratum_id][receipt.mask_area_class] += 1
        manifest_rows.extend(_manifest_rows(receipt))

    # Resume: accept only receipts of this run whose files match their recorded checksums.
    if resume and provenance_jsonl.is_file():
        print(f"Resuming from existing provenance ledger: {provenance_jsonl} ...")
        with provenance_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                receipt = AcquisitionReceipt(**json.loads(line))
                if receipt.run_id != run_id:
                    raise RunBindingMismatchError(
                        f"Ledger receipt {receipt.source_id} belongs to run '{receipt.run_id}', not '{run_id}'."
                    )

                auth_p = out_path / receipt.authentic_relpath
                edit_p = out_path / receipt.edited_relpath
                mask_p = out_path / receipt.mask_relpath

                if not (auth_p.is_file() and edit_p.is_file() and mask_p.is_file()):
                    raise TamperDetectedError(f"Missing file on disk for completed source: {receipt.source_id}")

                if _sha256_file(auth_p) != receipt.master_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on authentic master: {auth_p}")
                if _sha256_file(edit_p) != receipt.edited_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on edited master: {edit_p}")
                if _sha256_file(mask_p) != receipt.mask_sha256:
                    raise TamperDetectedError(f"SHA-256 tamper detected on mask: {mask_p}")

                count_accepted(receipt)

    # Attempts are one-shot authorizations. Resume may continue with candidates
    # that have no attempt record, but it must never retry a Technical QC failure,
    # generation error, or other previously recorded attempt.
    attempted_sources: set[str] = set()
    if resume and attempt_jsonl.is_file():
        with attempt_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                attempt = json.loads(line)
                if attempt.get("run_id") != run_id:
                    raise RunBindingMismatchError(
                        f"Attempt record {attempt.get('candidate_id')} belongs to run "
                        f"{attempt.get('run_id')!r}, not {run_id!r}."
                    )
                candidate_id = str(attempt.get("candidate_id", ""))
                if not candidate_id:
                    raise RunBindingMismatchError("Attempt ledger contains an empty candidate_id.")
                if candidate_id in attempted_sources:
                    raise RunBindingMismatchError(
                        f"Attempt ledger contains more than one attempt for {candidate_id}."
                    )
                attempted_sources.add(candidate_id)

    prov_writer_handle = provenance_jsonl.open("a", encoding="utf-8")
    attempt_writer_handle = attempt_jsonl.open("a", encoding="utf-8")
    log_writer_handle = (out_path / "acquisition.log").open("a", encoding="utf-8")

    def log_msg(msg: str) -> None:
        t_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{t_str}] {msg}"
        print(line, flush=True)
        try:
            log_writer_handle.write(line + "\n")
            log_writer_handle.flush()
        except Exception:
            pass

    log_msg(f"Starting acquisition run '{run_id}' (mode={mode}, target_per_stratum={target_per_stratum})")

    def log_attempt(record: dict[str, Any]) -> None:
        attempt_writer_handle.write(json.dumps(record) + "\n")
        attempt_writer_handle.flush()

    def log_receipt(receipt: AcquisitionReceipt) -> None:
        prov_writer_handle.write(json.dumps(asdict(receipt)) + "\n")
        prov_writer_handle.flush()

    strata_runtime: dict[str, dict[str, Any]] = {}
    enforce_quota = target_per_stratum == STRATUM_TARGET_PAIRS

    try:
        # 6. Execute by stratum sequentially (unload GPU models between tool strata)
        for stratum_id in STRATA_KEYS:
            stratum_candidates = candidates_by_stratum[stratum_id]
            if not stratum_candidates:
                continue

            stratum_t0 = time.perf_counter()
            tool_key = stratum_candidates[0].tool_key
            if engine_provider is not None:
                engine = engine_provider(tool_key)
            elif mode == "fixture_test" or allow_synthetic:
                engine = MockInpaintingEngine()
            elif mode == "production":
                resolved_rev = (
                    resolved_revisions.get(tool_key, {}).get("resolved_revision")
                    if resolved_revisions else None
                )
                engine = DiffusersInpaintingEngine(tool_key=tool_key, resolved_revision=resolved_rev)
            else:
                engine = MockInpaintingEngine()
            engine_config = engine.describe() if hasattr(engine, "describe") else {"engine": type(engine).__name__}

            for spec in stratum_candidates:
                if stratum_valid_counts[stratum_id] >= target_per_stratum:
                    break

                if spec.candidate_id in attempted_sources:
                    continue

                if enforce_quota:
                    # Enforce multi-dimensional quota preservation during replacement
                    if stratum_mod_counts[stratum_id][spec.modification_type] >= TARGET_MOD_PER_STRATUM[spec.modification_type]:
                        continue
                    if stratum_mask_counts[stratum_id][spec.mask_area_class] >= TARGET_MASK_PER_STRATUM[spec.mask_area_class]:
                        continue

                attempt_t0 = time.perf_counter()
                attempt_meta: dict[str, Any] = {
                    "run_id": run_id,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "candidate_id": spec.candidate_id,
                    "stratum_id": stratum_id,
                    "tool_key": spec.tool_key,
                    "modification_type": spec.modification_type,
                    "mask_area_class": spec.mask_area_class,
                    "origin_id": spec.origin_id,
                    "pool_index": spec.pool_index,
                    "prompt": spec.prompt,
                    "target_description": spec.target_description,
                    "placement_rationale": spec.placement_rationale,
                    "target_bbox_xyxy": spec.target_bbox_xyxy,
                    "mask_bbox_xyxy": spec.mask_bbox_xyxy,
                    "generation_seed": spec.generation_seed,
                }

                try:
                    # Step A: Download authentic image (fail-closed)
                    downloaded_at = datetime.now(timezone.utc).isoformat()
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
                    attempt_meta["raw_sha256"] = raw_sha256

                    auth_canvas = normalize_image_to_canvas(raw_auth_img)
                    if auth_canvas.size != TARGET_CANVAS_SIZE or auth_canvas.mode != "RGB":
                        raise GenerationContractError(
                            f"Input authentic canvas size/mode invalid: {auth_canvas.size}, {auth_canvas.mode}; "
                            f"expected {TARGET_CANVAS_SIZE}, RGB"
                        )

                    # Step B: Use the pre-registered content-grounded region in production.
                    if mode == "production":
                        mask_canvas, _target_ratio = build_content_grounded_mask(spec)
                    else:
                        mask_canvas, _target_ratio = synthesize_canonical_mask(
                            spec.mask_area_class,
                            seed=spec.generation_seed,
                        )
                    if mask_canvas.size != TARGET_CANVAS_SIZE or mask_canvas.mode not in ("L", "1"):
                        raise GenerationContractError(
                            f"Input mask canvas size/mode invalid: {mask_canvas.size}, {mask_canvas.mode}; "
                            f"expected {TARGET_CANVAS_SIZE}, L"
                        )

                    # Step C: Generate inpainting
                    generated_canvas = engine.inpaint(
                        auth_canvas,
                        mask_canvas,
                        prompt=spec.prompt,
                        seed=spec.generation_seed,
                    )
                    if generated_canvas.size != TARGET_CANVAS_SIZE or generated_canvas.mode != "RGB":
                        raise GenerationContractError(
                            f"Inpainting engine output size/mode invalid: {generated_canvas.size}, {generated_canvas.mode}; "
                            f"expected {TARGET_CANVAS_SIZE}, RGB"
                        )
                    edited_canvas = composite_generated_region(auth_canvas, generated_canvas, mask_canvas)

                    # Step D: Technical Quality Control
                    is_pass, fail_reason, actual_ratio = evaluate_technical_qc(
                        authentic_img=auth_canvas,
                        edited_img=edited_canvas,
                        mask_img=mask_canvas,
                        mask_area_class=spec.mask_area_class,
                        raw_bytes=raw_bytes,
                        historical_hashes=hist_hashes,
                    )

                    if not is_pass:
                        attempt_meta["status"] = "QC_FAILED"
                        attempt_meta["reason"] = fail_reason
                        attempt_meta["elapsed_seconds"] = round(time.perf_counter() - attempt_t0, 3)
                        log_attempt(attempt_meta)
                        log_msg(f"  [TECHNICAL QC REJECT] Candidate {spec.candidate_id}: {fail_reason}")
                        continue  # Stratum-preserving replacement: move to next candidate

                    # Step E: Save verified master files
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
                        master_sha256=_sha256_file(auth_path),
                        mask_sha256=_sha256_file(mask_path),
                        edited_sha256=_sha256_file(edit_path),
                        authentic_relpath=f"images/{auth_filename}",
                        edited_relpath=f"images/{edit_filename}",
                        mask_relpath=f"masks/{mask_filename}",
                        mask_pixel_ratio=actual_ratio,
                        technical_qc_status="PASS",
                        content_qc_status="PENDING_CONTENT_QC",
                        is_synthetic=spec.is_synthetic,
                        run_id=run_id,
                        source_keys=list(spec.source_keys),
                        creator_url=spec.creator_url,
                        license_url=spec.license_url,
                        usage_policy=spec.usage_policy,
                        date_captured=spec.date_captured,
                        download_rendition=spec.download_rendition,
                        downloaded_at_utc=downloaded_at,
                        raw_width=raw_auth_img.size[0],
                        raw_height=raw_auth_img.size[1],
                        engine_config=engine_config,
                        target_description=spec.target_description,
                        placement_rationale=spec.placement_rationale,
                        target_bbox_xyxy=spec.target_bbox_xyxy,
                        mask_bbox_xyxy=spec.mask_bbox_xyxy,
                    )

                    log_receipt(receipt)
                    count_accepted(receipt)

                    attempt_meta["status"] = "ACCEPTED"
                    attempt_meta["elapsed_seconds"] = round(time.perf_counter() - attempt_t0, 3)
                    log_attempt(attempt_meta)
                    log_msg(f"  [ACCEPTED] Candidate {spec.candidate_id} ({stratum_valid_counts[stratum_id]}/{target_per_stratum} in {stratum_id})")

                except GenerationContractError as e:
                    attempt_meta["status"] = "GENERATION_CONTRACT_ERROR"
                    attempt_meta["error"] = str(e)
                    attempt_meta["elapsed_seconds"] = round(time.perf_counter() - attempt_t0, 3)
                    log_attempt(attempt_meta)
                    fail_receipt = {
                        "run_id": run_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "FAILED",
                        "failure_type": "GENERATION_CONTRACT_ERROR",
                        "error_message": str(e),
                        "stratum_id": stratum_id,
                        "candidate_id": spec.candidate_id,
                        "tool_key": spec.tool_key,
                        "binding": run_binding,
                    }
                    (out_path / "failure_receipt.json").write_text(json.dumps(fail_receipt, indent=2), encoding="utf-8")
                    tb_str = traceback.format_exc()
                    log_msg(f"FATAL: Generation contract error on candidate {spec.candidate_id}: {e}\n{tb_str}")
                    raise
                except Exception as e:
                    attempt_meta["status"] = "ERROR"
                    attempt_meta["error"] = str(e)
                    attempt_meta["elapsed_seconds"] = round(time.perf_counter() - attempt_t0, 3)
                    log_attempt(attempt_meta)
                    log_msg(f"  [ERROR] Candidate {spec.candidate_id} error: {e}")
                    continue

            strata_runtime[stratum_id] = {
                "elapsed_seconds": round(time.perf_counter() - stratum_t0, 3),
                "peak_vram_bytes": engine.peak_vram_bytes() if hasattr(engine, "peak_vram_bytes") else "NOT_MEASURED",
                "engine": engine_config,
            }
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
        try:
            log_writer_handle.close()
        except Exception:
            pass

    # 7. Quota checks before anything claims completion
    total_valid = sum(stratum_valid_counts.values())
    if enforce_quota:
        if total_valid != TOTAL_TARGET_PAIRS or len(completed_sources) != TOTAL_TARGET_PAIRS:
            raise QuotaDeficitError(
                f"Cohort acquisition failed to meet target quota: {total_valid}/{TOTAL_TARGET_PAIRS} valid pairs."
            )
        expected_mod = {k: v * len(STRATA_KEYS) for k, v in TARGET_MOD_PER_STRATUM.items()}
        expected_mask = {k: v * len(STRATA_KEYS) for k, v in TARGET_MASK_PER_STRATUM.items()}
        if mod_counts != expected_mod:
            raise QuotaDeficitError(f"Modification type quotas skewed: {mod_counts}")
        if mask_counts != expected_mask:
            raise QuotaDeficitError(f"Mask area quotas skewed: {mask_counts}")

    # 8. Write manifest, checksum and (last) the run receipt
    manifest_csv = out_path / "manifest_independent_cohort.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = _sha256_file(manifest_csv)
    (out_path / "manifest_checksum.sha256").write_text(
        f"{manifest_sha}  manifest_independent_cohort.csv\n", encoding="utf-8"
    )

    status = (
        "PROVISIONAL_TECHNICAL_QC_PASS_PENDING_CONTENT_QC"
        if enforce_quota
        else "PARTIAL_PILOT_TECHNICAL_PASS"
    )
    result = {
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
    run_receipt = {
        "binding": run_binding,
        "status": status,
        "mode": mode,
        "target_per_stratum": target_per_stratum,
        "total_valid_pairs": total_valid,
        "stratum_counts": stratum_valid_counts,
        "modification_type_counts": mod_counts,
        "mask_area_counts": mask_counts,
        "manifest_sha256": manifest_sha,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "strata": strata_runtime,
        "model_preflight": resolved_revisions or {},
        "historical_guard": {
            "source_keys": len(hist_keys),
            "source_key_namespaces_checked": ["coco"],
            "flickr_photo_id_disjointness": "NOT_CHECKED (historical Flickr IDs not available)",
            "byte_hashes": len(hist_hashes),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "detector_calls": 0,
        "content_qc_status": "PENDING_CONTENT_QC",
    }
    (out_path / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2), encoding="utf-8")
    return result


def audit_acquisition_run(run_dir: Path | str, expected_binding: dict[str, Any]) -> dict[str, Any]:
    """Prove that run_dir holds a complete acquisition bound to expected_binding.

    Fails closed on: missing/foreign run receipt, any binding field mismatch, manifest
    checksum mismatch, ledger rows from another run, file hash mismatch, or a pair count
    different from the bound quota.
    """
    out = Path(run_dir)
    failure_path = out / "failure_receipt.json"
    if failure_path.is_file():
        fail_content = failure_path.read_text(encoding="utf-8")
        raise RunAuditError(f"Run terminated with failure receipt: {fail_content}")
    receipt_path = out / "run_receipt.json"
    if not receipt_path.is_file():
        raise RunAuditError(f"{receipt_path} missing: run did not complete (or predates run binding).")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bound = receipt.get("binding", {})
    mismatched = sorted(k for k in set(bound) | set(expected_binding) if bound.get(k) != expected_binding.get(k))
    if mismatched:
        raise RunAuditError(f"Run receipt binding differs from this session in: {mismatched}")
    binding_file = out / "run_binding.json"
    if not binding_file.is_file() or json.loads(binding_file.read_text(encoding="utf-8")) != expected_binding:
        raise RunAuditError("run_binding.json missing or differs from the run receipt binding.")

    manifest = out / "manifest_independent_cohort.csv"
    checksum = (out / "manifest_checksum.sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256_file(manifest)
    if not (actual == checksum == receipt["manifest_sha256"]):
        raise RunAuditError(f"Manifest sha256 mismatch: file={actual} checksum={checksum} receipt={receipt['manifest_sha256']}")

    rows = [json.loads(x) for x in (out / "provenance_ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        if r.get("run_id") != expected_binding.get("run_id"):
            raise RunAuditError(f"Ledger receipt {r.get('source_id')} has run_id {r.get('run_id')!r}.")
        for rel, key in (("authentic_relpath", "master_sha256"), ("edited_relpath", "edited_sha256"), ("mask_relpath", "mask_sha256")):
            p = out / r[rel]
            if not p.is_file() or _sha256_file(p) != r[key]:
                raise RunAuditError(f"sha256 mismatch or missing file: {p}")

    expected_pairs = int(expected_binding.get("target_per_stratum", 0)) * len(STRATA_KEYS)
    if not (len(rows) == receipt["total_valid_pairs"] == expected_pairs):
        raise RunAuditError(f"Pair count mismatch: ledger={len(rows)} receipt={receipt['total_valid_pairs']} expected={expected_pairs}")

    return {"status": "PASS", "run_id": expected_binding.get("run_id"), "pairs": len(rows),
            "manifest_sha256": actual, "run_status": receipt["status"]}
