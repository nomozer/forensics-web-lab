"""SYNTHETIC candidate-catalog fixture for Phase 4C.7B acquisition tests.

SYNTHETIC: every entry below is invented test data. Creator names, Flickr/Unsplash IDs,
URLs and timestamps are NOT real provenance and must never enter an official catalog.
The catalog is flagged ``is_synthetic: true`` so production acquisition rejects it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

SYNTHETIC_CHECKED_AT = "2026-01-01T00:00:00+00:00"

# Synthetic COCO IDs start far above the real COCO ID range so they never collide
# with the 684 historical Option P COCO IDs unless a test injects one on purpose.
_SYNTH_COCO_BASE = 900_000_000
_SYNTH_FLICKR_BASE = 990_000_000_000


def synthetic_coco_entry(i: int, license_code: str = "by", **overrides: Any) -> dict[str, Any]:
    coco_id = _SYNTH_COCO_BASE + i
    flickr_id = str(_SYNTH_FLICKR_BASE + i)
    entry = {
        "source_origin": "coco_2017",
        "acquisition_channel": "coco_2017_image_info",
        "coco_image_id": coco_id,
        "flickr_photo_id": flickr_id,
        "source_keys": [f"coco:{coco_id}", f"flickr:{flickr_id}"],
        "origin_id": f"coco:{coco_id}",
        "download_url": f"https://synthetic.invalid/coco/{coco_id}.jpg",
        "source_page_url": f"https://synthetic.invalid/flickr/photos/{flickr_id}/",
        "creator_name": f"SYNTHETIC Creator {i}",
        "creator_url": f"https://synthetic.invalid/flickr/people/{i}/",
        "creator_verification": "VERIFIED_FROM_SOURCE",
        "creator_evidence_url": f"https://synthetic.invalid/oembed/{flickr_id}",
        "license_name": f"SYNTHETIC CC {license_code.upper()} 2.0",
        "license_url": f"https://creativecommons.org/licenses/{license_code}/2.0/",
        "license_version": "2.0",
        "license_evidence_source": "https://synthetic.invalid/image_info_test2017.json",
        "date_captured": "2013-01-01 00:00:00",
        "date_captured_source": "SYNTHETIC",
        "published_date": None,
        "provenance_checked_at_utc": SYNTHETIC_CHECKED_AT,
        "download_rendition": "SYNTHETIC rendition",
    }
    entry.update(overrides)
    return entry


def build_synthetic_catalog(per_source: int = 220) -> dict[str, Any]:
    """Return an all-eligible SYNTHETIC catalog (both halves use the COCO-shaped entry)."""
    coco = [synthetic_coco_entry(i) for i in range(per_source)]
    second = []
    for i in range(per_source):
        e = synthetic_coco_entry(10_000 + i)
        e["source_origin"] = "unsplash_verified"
        second.append(e)
    return {
        "schema_version": "2.0.0",
        "evidence_class": "synthetic",
        "is_synthetic": True,
        "synthetic_note": "SYNTHETIC test catalog; not real provenance.",
        "coco_candidates": coco,
        "unsplash_candidates": second,
    }


def write_synthetic_catalog(path: Path, catalog: dict[str, Any] | None = None) -> Path:
    path.write_text(json.dumps(catalog or build_synthetic_catalog(), indent=1), encoding="utf-8")
    return path


FORBIDDEN_DETECTOR_MODULES = [
    "ml.evaluation.phase_4c2g_model",
    "ml.evaluation.locked_test_evaluator",
    "ml.evaluation.independent_evaluator",
    "ml.evaluation.independent_model_bindings",
]


@pytest.fixture(autouse=True)
def clean_detector_modules_isolation():
    """Ensure acquisition tests run in isolation with zero detector modules in sys.modules."""
    saved: dict[str, Any] = {}
    for mod in FORBIDDEN_DETECTOR_MODULES:
        if mod in sys.modules:
            saved[mod] = sys.modules.pop(mod)
    try:
        yield
    finally:
        for mod, val in saved.items():
            sys.modules[mod] = val
