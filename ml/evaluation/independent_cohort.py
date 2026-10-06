"""Independent Cohort Manifest Validation and Disjoint Guard for Phase 4C.7A.

Strictly enforces:
1. Binary classification scope: authentic = 0, ai_edited = 1.
2. Source-paired cohort structure (each source_id must have exactly one authentic and one ai_edited entry).
3. Zero-overlap source-disjoint guard against all 684 historical Option P sources
   (250 development_train + 91 inner_validation + 343 retired locked_test).
4. Prevention of locked-test recycling or renaming.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_MANIFEST = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"


class CohortValidationError(ValueError):
    """Raised when an independent cohort violates validation rules."""


class SourceOverlapError(CohortValidationError):
    """Raised when an independent cohort overlaps with historical development or locked-test sources."""


@dataclass(frozen=True)
class CohortSample:
    source_id: str
    label: str  # "authentic" or "ai_edited"
    label_id: int  # 0 or 1
    image_relpath: str
    image_sha256: str
    tool: str | None = None
    mask_relpath: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class CohortPair:
    source_id: str
    authentic: CohortSample
    ai_edited: CohortSample
    tool: str | None = None
    category: str | None = None
    license: str | None = None


def load_historical_source_ids(manifest_path: Path | str | None = None) -> set[str]:
    """Extract historical source IDs from the committed Option P manifest.

    Reads only metadata. Strictly avoids reading locked-test images or features.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_HISTORICAL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Historical manifest not found at {path}")

    source_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("source_id", "").strip()
            if sid:
                source_ids.add(sid)
    return source_ids


def load_historical_image_hashes(manifest_path: Path | str | None = None) -> set[str]:
    """Extract all historical image SHA-256 hashes from Option P manifest.

    Used by the disjoint guard to block exact duplicates or renamed historic images.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_HISTORICAL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Historical manifest not found at {path}")

    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            auth_sha = row.get("authentic_sha256", "").strip().lower()
            edit_sha = row.get("canonical_edit_sha256", "").strip().lower()
            if auth_sha:
                hashes.add(auth_sha)
            if edit_sha:
                hashes.add(edit_sha)
    return hashes


def load_historical_origin_ids(manifest_path: Path | str | None = None) -> set[str]:
    """Extract all historical instance/origin IDs from Option P manifest."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_HISTORICAL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Historical manifest not found at {path}")

    origins: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst_id = row.get("instance_id", "").strip()
            if inst_id:
                origins.add(inst_id)
    return origins


def validate_cohort_manifest(
    manifest_rows: Sequence[dict[str, Any]],
    historical_sources: set[str] | None = None,
    historical_hashes: set[str] | None = None,
    historical_origins: set[str] | None = None,
) -> list[CohortPair]:
    """Validate manifest rows for an independent validation cohort.

    Rules:
    - Must be non-empty.
    - Each source_id must appear exactly twice: once with label 'authentic' (0) and once with 'ai_edited' (1).
    - Source-Disjoint Guard: No source_id may belong to historical_sources.
    - Image-Disjoint Guard: No image_sha256 may belong to historical_hashes.
    - Origin-Disjoint Guard: No origin_id / instance_id may belong to historical_origins.
    - Resolution Consistency: authentic and ai_edited of the same pair must share the same resolution if specified.
    - Image paths and valid 64-char hex SHA-256 strings must be provided.
    """
    if not manifest_rows:
        raise CohortValidationError("Cohort manifest is empty.")

    # 1. Source ID Disjoint Guard
    if historical_sources is not None and len(historical_sources) > 0:
        incoming_sources = {str(r.get("source_id", "")).strip() for r in manifest_rows}
        overlap = incoming_sources & historical_sources
        if overlap:
            sample_overlap = sorted(list(overlap))[:5]
            raise SourceOverlapError(
                f"Contamination detected: {len(overlap)} sources overlap with historical development/locked-test "
                f"cohorts (e.g. {sample_overlap}). Independent cohort MUST be strictly source-disjoint."
            )

    # 2. Image SHA-256 Disjoint Guard (detects renamed historical files)
    if historical_hashes is not None and len(historical_hashes) > 0:
        incoming_hashes = {
            str(r.get("image_sha256") or r.get("sha256") or "").strip().lower()
            for r in manifest_rows
        }
        overlap_hashes = incoming_hashes & historical_hashes
        if overlap_hashes:
            sample_overlap_h = sorted(list(overlap_hashes))[:5]
            raise SourceOverlapError(
                f"Contamination detected: {len(overlap_hashes)} image SHA-256 hashes match historical Option P "
                f"images (e.g. {sample_overlap_h}). Renamed historic images are strictly forbidden."
            )

    # 3. Origin / Instance ID Disjoint Guard
    if historical_origins is not None and len(historical_origins) > 0:
        incoming_origins = {
            str(r.get("origin_id") or r.get("instance_id") or "").strip()
            for r in manifest_rows
            if r.get("origin_id") or r.get("instance_id")
        }
        overlap_origins = incoming_origins & historical_origins
        if overlap_origins:
            sample_overlap_o = sorted(list(overlap_origins))[:5]
            raise SourceOverlapError(
                f"Contamination detected: {len(overlap_origins)} origin IDs overlap with historical datasets "
                f"(e.g. {sample_overlap_o})."
            )

    by_source: dict[str, dict[str, CohortSample]] = {}
    pair_meta: dict[str, dict[str, Any]] = {}

    for idx, row in enumerate(manifest_rows):
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            raise CohortValidationError(f"Row {idx} missing required 'source_id'")

        label = str(row.get("label", "")).strip().lower()
        if label not in ("authentic", "ai_edited"):
            raise CohortValidationError(f"Row {idx} invalid label '{label}'; must be 'authentic' or 'ai_edited'")

        label_id = 0 if label == "authentic" else 1
        if "label_id" in row and int(row["label_id"]) != label_id:
            raise CohortValidationError(
                f"Row {idx} label '{label}' contradicts label_id {row['label_id']} (expected {label_id})"
            )

        img_path = str(row.get("image_relpath") or row.get("image_path") or "").strip()
        if not img_path:
            raise CohortValidationError(f"Row {idx} missing image path")

        sha256 = str(row.get("image_sha256") or row.get("sha256") or "").strip()
        if not sha256 or len(sha256) != 64:
            raise CohortValidationError(f"Row {idx} missing or invalid 64-char hex SHA-256")

        w_val = int(row["width"]) if ("width" in row and row["width"] is not None and str(row["width"]).isdigit()) else None
        h_val = int(row["height"]) if ("height" in row and row["height"] is not None and str(row["height"]).isdigit()) else None

        sample = CohortSample(
            source_id=source_id,
            label=label,
            label_id=label_id,
            image_relpath=img_path,
            image_sha256=sha256,
            tool=row.get("tool"),
            mask_relpath=row.get("mask_relpath"),
            width=w_val,
            height=h_val,
        )

        if source_id not in by_source:
            by_source[source_id] = {}
            pair_meta[source_id] = {
                "category": row.get("category"),
                "license": row.get("license"),
            }

        if label in by_source[source_id]:
            raise CohortValidationError(f"Duplicate entry for source '{source_id}' with label '{label}'")

        by_source[source_id][label] = sample

    pairs: list[CohortPair] = []
    for source_id, label_dict in sorted(by_source.items()):
        if "authentic" not in label_dict:
            raise CohortValidationError(f"Source '{source_id}' is missing required 'authentic' counterpart")
        if "ai_edited" not in label_dict:
            raise CohortValidationError(f"Source '{source_id}' is missing required 'ai_edited' counterpart")

        auth_s = label_dict["authentic"]
        edit_s = label_dict["ai_edited"]

        if (
            auth_s.width is not None
            and edit_s.width is not None
            and (auth_s.width != edit_s.width or auth_s.height != edit_s.height)
        ):
            raise CohortValidationError(
                f"Resolution mismatch for source '{source_id}': authentic is "
                f"{auth_s.width}x{auth_s.height}, ai_edited is {edit_s.width}x{edit_s.height}. "
                "Canvas dimensions must match."
            )

        meta = pair_meta[source_id]
        pairs.append(
            CohortPair(
                source_id=source_id,
                authentic=auth_s,
                ai_edited=edit_s,
                tool=edit_s.tool,
                category=meta.get("category"),
                license=meta.get("license"),
            )
        )

    return pairs


def generate_synthetic_planning_cohort(
    num_pairs: int = 50,
    prefix: str = "SYNTH_IND_",
) -> list[dict[str, Any]]:
    """Generate a valid, deterministic synthetic cohort manifest for planning and preflight tests.

    All samples are guaranteed source-disjoint from Option P historical sources.
    """
    rows: list[dict[str, Any]] = []
    tools = ["sd2_inpaint", "sdxl_inpaint", "firefly_sim", "flux_fill_sim"]

    for i in range(num_pairs):
        source_id = f"{prefix}{i:05d}"
        tool = tools[i % len(tools)]

        auth_content = f"synthetic_auth_{source_id}".encode("utf-8")
        edit_content = f"synthetic_edit_{source_id}_{tool}".encode("utf-8")
        auth_sha = hashlib.sha256(auth_content).hexdigest()
        edit_sha = hashlib.sha256(edit_content).hexdigest()

        rows.append(
            {
                "source_id": source_id,
                "label": "authentic",
                "label_id": 0,
                "image_relpath": f"data/synthetic_cohort/{source_id}_auth.png",
                "image_sha256": auth_sha,
                "category": f"cat_{i % 5}",
                "license": "SYNTHETIC_TEST_FIXTURE",
            }
        )
        rows.append(
            {
                "source_id": source_id,
                "label": "ai_edited",
                "label_id": 1,
                "image_relpath": f"data/synthetic_cohort/{source_id}_edit.png",
                "image_sha256": edit_sha,
                "tool": tool,
                "category": f"cat_{i % 5}",
                "license": "SYNTHETIC_TEST_FIXTURE",
            }
        )

    return rows
