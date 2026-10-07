#!/usr/bin/env python3
"""Build candidate catalog v2 with per-image verified provenance (Phase 4C.7B).

Run on a host that can reach images.cocodataset.org and www.flickr.com (e.g. Colab):
    python scripts/research/build_verified_candidate_catalog.py [--out PATH]
Offline eligibility audit of the superseded v1 catalog (no network):
    python scripts/research/build_verified_candidate_catalog.py --audit-legacy

COCO 2017 test images, in the order of the official image_info_test2017.json:
- COCO image ID, license (id, name, URL, version) and date_captured come from image_info.
- The Flickr photo ID is parsed from that file's flickr_url (never invented).
- Creator name/URL and the license currently shown on Flickr come from the Flickr oEmbed
  response for that photo; the request URL, response SHA-256 and check time are stored.
- Entries are evaluated with the pre-registered rules in
  ml.evaluation.independent_cohort_acquisition.evaluate_candidate_eligibility; excluded
  entries are kept in `excluded_candidates` with their reasons.

Unsplash: the previous source was the Unsplash Lite dataset (via a Hugging Face mirror).
Its terms do not cover exporting/sharing edited derivatives, so no Unsplash candidates are
produced; the Unsplash strata stay blocked until a source decision is recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort_acquisition import (  # noqa: E402
    CC_LICENSE_RE,
    DEFAULT_CATALOG_PATH,
    evaluate_candidate_eligibility,
    load_historical_source_keys,
)

COCO_IMAGE_INFO_URL = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
FLICKR_OEMBED = "https://www.flickr.com/services/oembed/?format=json&url="
FLICKR_URL_RE = re.compile(r"/(\d+)_[0-9a-f]+(?:_([a-z]))?\.jpg$")
BASE58 = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
USER_AGENT = "ForensicsWebLab-CatalogBuilder/2.0"


def flickr_short_url(photo_id: str) -> str:
    n, out = int(photo_id), ""
    while n:
        n, r = divmod(n, 58)
        out = BASE58[r] + out
    return f"https://flic.kr/p/{out}"


def _norm_license_url(url: str) -> str:
    m = CC_LICENSE_RE.search(url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else ""


def coco_entry(img: dict, licenses: dict[int, dict], oembed: dict | None, oembed_url: str,
               oembed_sha256: str, checked_at: str) -> dict:
    """Catalog entry built only from fields present in the source responses."""
    lic = licenses.get(img.get("license"), {})
    lic_url = lic.get("url", "")
    m_lic = CC_LICENSE_RE.search(lic_url)
    m_flickr = FLICKR_URL_RE.search(img.get("flickr_url", ""))
    flickr_id = m_flickr.group(1) if m_flickr else ""
    size = (m_flickr.group(2) if m_flickr else None) or "unknown"

    entry = {
        "source_origin": "coco_2017",
        "acquisition_channel": "coco_2017_image_info",
        "coco_image_id": img["id"],
        "flickr_photo_id": flickr_id or None,
        "source_keys": [f"coco:{img['id']}"] + ([f"flickr:{flickr_id}"] if flickr_id else []),
        "origin_id": f"coco:{img['id']}",
        "download_url": img.get("coco_url", ""),
        "flickr_url": img.get("flickr_url", ""),
        "license_id": img.get("license"),
        "license_name": lic.get("name", ""),
        "license_url": lic_url,
        "license_version": m_lic.group(2) if m_lic else "",
        "license_evidence_source": f"{COCO_IMAGE_INFO_URL} (images[].license -> licenses[])",
        "date_captured": img.get("date_captured") or None,
        "date_captured_source": "COCO image_info_test2017.json images[].date_captured" if img.get("date_captured") else None,
        "published_date": None,
        "provenance_checked_at_utc": checked_at,
        "download_rendition": (
            f"COCO-hosted copy of the Flickr size-'{size}' JPEG rendition ({img.get('width')}x{img.get('height')}); "
            "re-encoded by Flickr, not the original upload"
        ),
        "width": img.get("width"),
        "height": img.get("height"),
        "source_page_url": "",
        "creator_name": "",
        "creator_url": "",
        "creator_verification": "NOT_VERIFIED",
        "creator_evidence_url": "",
    }
    if oembed:
        page = oembed.get("web_page", "")
        author = (oembed.get("author_name") or "").strip()
        author_url = oembed.get("author_url", "")
        current = _norm_license_url(oembed.get("license_url", ""))
        entry.update(
            source_page_url=page,
            license_current_at_source=oembed.get("license_url", ""),
            creator_evidence_sha256=oembed_sha256,
        )
        if author and author_url.startswith("https://www.flickr.com/photos/") and flickr_id and flickr_id in page:
            entry.update(creator_name=author, creator_url=author_url, creator_evidence_url=oembed_url,
                         creator_verification="VERIFIED_FROM_SOURCE")
        if current != _norm_license_url(lic_url):  # includes "no CC license shown on Flickr now"
            # Pre-registered conservative rule: the license must agree at both sources.
            entry["license_url"] = ""
            entry["license_conflict"] = {"coco": lic_url, "flickr_current": oembed.get("license_url")}
    return entry


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def build(per_source: int = 220) -> dict:
    hist = load_historical_source_keys()
    info_bytes = _fetch(COCO_IMAGE_INFO_URL, timeout=120)
    info = json.loads(zipfile.ZipFile(io.BytesIO(info_bytes)).read("annotations/image_info_test2017.json"))
    licenses = {lic["id"]: lic for lic in info["licenses"]}

    eligible, excluded = [], []
    for img in info["images"]:
        if len(eligible) >= per_source:
            break
        checked_at = datetime.now(timezone.utc).isoformat()
        pre = coco_entry(img, licenses, None, "", "", checked_at)
        pre_reasons = set(evaluate_candidate_eligibility(pre, hist)["reasons"])
        # Skip the network call when the entry fails on source-independent grounds.
        blocking = pre_reasons - {"CREATOR_PLACEHOLDER", "CREATOR_UNVERIFIED", "PROVENANCE_FIELD_MISSING:source_page_url"}
        oembed, oembed_url, sha = None, "", ""
        if not blocking and pre["flickr_photo_id"]:
            oembed_url = FLICKR_OEMBED + urllib.parse.quote(flickr_short_url(pre["flickr_photo_id"]), safe="")
            try:
                raw = _fetch(oembed_url)
                oembed, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
            except Exception as e:  # deleted/private photo, network error: unverifiable -> excluded
                pre["oembed_error"] = f"{type(e).__name__}: {e}"
        entry = coco_entry(img, licenses, oembed, oembed_url, sha, checked_at) if oembed else pre
        result = evaluate_candidate_eligibility(entry, hist)
        if result["status"] == "EXCLUDED":
            excluded.append({"coco_image_id": img["id"], "reasons": result["reasons"], "entry": entry})
        else:
            eligible.append(entry)

    return {
        "schema_version": "2.0.0",
        "evidence_class": "verified_real_catalog",
        "is_synthetic": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": "research/evidence/phase-4c.7b/verified_candidate_catalog.json",
        "coco_image_info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "coco_candidates": eligible,
        "unsplash_candidates": [],
        "unsplash_status": "BLOCKED: Unsplash Lite dataset terms do not cover sharing edited derivatives; source decision pending",
        "excluded_candidates": excluded,
    }


LEGACY_CATALOG = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog.json"
LEGACY_AUDIT = REPO_ROOT / "research/evidence/phase-4c.7b/catalog_eligibility_audit.json"
# The v1 builder (commit dc00895) fetched Unsplash rows from the Hugging Face mirror
# 1aurent/unsplash-lite of the Unsplash Lite dataset; COCO rows from image_info_test2017.
LEGACY_CHANNELS = {"coco_2017": "coco_2017_image_info", "unsplash_verified": "unsplash_lite_dataset_hf_mirror"}


def audit_legacy_catalog() -> dict:
    """Apply the pre-registered rules to the v1 catalog without modifying it."""
    raw = LEGACY_CATALOG.read_bytes()
    catalog = json.loads(raw)
    hist = load_historical_source_keys()
    rows, reason_counts, license_counts = [], {}, {}
    for e in catalog["coco_candidates"] + catalog["unsplash_candidates"]:
        res = evaluate_candidate_eligibility(e | {"acquisition_channel": LEGACY_CHANNELS[e["source_origin"]]}, hist)
        for r in res["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        license_counts[e["license_name"]] = license_counts.get(e["license_name"], 0) + 1
        rows.append({"source_origin": e["source_origin"], "origin_id": e["origin_id"], "author_in_catalog": e["author"],
                     "license_name": e["license_name"], "status": res["status"], "reasons": res["reasons"]})
    return {
        "audited_catalog": "research/evidence/phase-4c.7b/verified_candidate_catalog.json",
        "audited_catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "catalog_status": "SUPERSEDED",
        "generated_by": "python scripts/research/build_verified_candidate_catalog.py --audit-legacy",
        "total": len(rows),
        "eligible": sum(r["status"] != "EXCLUDED" for r in rows),
        "excluded": sum(r["status"] == "EXCLUDED" for r in rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "findings": {
            "coco_author": "all 220 COCO authors are generated labels 'flickr_contributor_<coco_id>', not verified creators",
            "coco_published_date": "field holds COCO date_captured, not a publication date",
            "coco_origin_url": "points to the COCO-hosted copy, not the Flickr photo page; no Flickr photo ID recorded",
            "coco_nd": "91 entries are BY-ND or BY-NC-ND: no sharing of edited derivatives without separate permission",
            "unsplash_channel": "Unsplash Lite dataset (HF mirror): dataset terms grant download/store and internal ML "
                                "training only and prohibit disseminating/redistributing the data; editing and sharing "
                                "the cohort is not covered. The generic Unsplash License was assumed without checking "
                                "the dataset terms.",
            "unsplash_rendition": "download_url requests a processed rendition (auto=format, fit=crop, w=600, q=80): "
                                  "resized, re-compressed, format negotiated; not the original upload",
            "historical_disjointness": "v1 compared zero-padded COCO IDs to Option P source IDs (same COCO namespace); "
                                       "Unsplash IDs were compared to COCO IDs, which proves nothing",
        },
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--audit-legacy", action="store_true", help="Offline eligibility audit of the v1 catalog")
    args = parser.parse_args()
    if args.audit_legacy:
        audit = audit_legacy_catalog()
        LEGACY_AUDIT.write_text(json.dumps(audit, indent=1), encoding="utf-8")
        print(f"v1 catalog: {audit['eligible']}/{audit['total']} eligible; reasons {audit['reason_counts']}")
        return
    catalog = build()
    Path(args.out).write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    print(f"COCO eligible: {len(catalog['coco_candidates'])}/220, excluded: {len(catalog['excluded_candidates'])}, "
          f"Unsplash: {catalog['unsplash_status']}\nWrote {args.out}")


if __name__ == "__main__":
    main()
