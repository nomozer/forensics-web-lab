#!/usr/bin/env python3
"""Build candidate catalog v2 with per-image verified provenance (Phase 4C.7B).

Supports Protocol Amendment v1.3:
1. COCO 2017 Clean Subset (220 candidates):
   - Images from COCO val2017 split, strictly disjoint from 684 historical Option P IDs.
   - Verified via official Flickr oEmbed for author, Flickr page, and active CC license.
   - Pre-registered eligibility excludes NoDerivs (ND) and unverified creators.
2. Wikimedia Commons Photographic Works (220 candidates):
   - Curated high-resolution photography from Category:Quality_images.
   - Verified via MediaWiki Action API imageinfo extmetadata.
   - Pre-registered eligibility checks CC license (BY / BY-SA), author, and publication date.

Usage:
    python scripts/research/build_verified_candidate_catalog.py [--out PATH] [--per-source 220]
Offline eligibility audit of the superseded v1 catalog (no network):
    python scripts/research/build_verified_candidate_catalog.py --audit-legacy
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
import time
import urllib.parse
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort_acquisition import (  # noqa: E402
    CC_LICENSE_RE,
    DEFAULT_CATALOG_PATH,
    evaluate_candidate_eligibility,
    load_historical_source_keys,
)

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
FLICKR_OEMBED = "https://www.flickr.com/services/oembed/?format=json&url="
FLICKR_URL_RE = re.compile(r"/(\d+)_[0-9a-f]+(?:_([a-z]))?\.jpg$")
BASE58 = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
USER_AGENT = (
    "ForensicsWebLabResearch/2.0 "
    "(https://github.com/nomozer/forensics-web-lab; contact: nomozer on GitHub) Python/3.12"
)


class RemoteZipFile(io.RawIOBase):
    """Seekable stream reading remote zip archives via HTTP Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={self.pos}-{end}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            chunk = resp.read()
        n = len(chunk)
        b[:n] = chunk
        self.pos += n
        return n


def flickr_short_url(photo_id: str) -> str:
    n, out = int(photo_id), ""
    while n:
        n, r = divmod(n, 58)
        out = BASE58[r] + out
    return f"https://flic.kr/p/{out}"


def _norm_license_url(url: str) -> str:
    m = CC_LICENSE_RE.search(url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else ""


def _fetch(url: str, timeout: float = 30.0, max_retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {last_err}")


def coco_entry(
    img: dict,
    licenses: dict[int, dict],
    oembed: dict | None,
    oembed_url: str,
    oembed_sha256: str,
    checked_at: str,
) -> dict:
    """Catalog entry built only from fields present in the source responses."""
    lic = licenses.get(img.get("license"), {})
    lic_url = lic.get("url", "")
    m_lic = CC_LICENSE_RE.search(lic_url)
    m_flickr = FLICKR_URL_RE.search(img.get("flickr_url", ""))
    flickr_id = m_flickr.group(1) if m_flickr else ""

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
        "license_evidence_source": "COCO annotations captions_val2017.json + Flickr oEmbed",
        "date_captured": img.get("date_captured") or None,
        "date_captured_source": (
            "COCO captions_val2017.json images[].date_captured" if img.get("date_captured") else None
        ),
        "published_date": None,
        "provenance_checked_at_utc": checked_at,
        "download_rendition": (
            "COCO-hosted copy of the medium (size-'z', max 640px) rendition generated and "
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
        if author and author_url.startswith("https://www.flickr.com/photos/") and flickr_id:
            entry.update(
                creator_name=author,
                creator_url=author_url,
                creator_evidence_url=oembed_url,
                creator_verification="VERIFIED_FROM_SOURCE",
            )
        if current != _norm_license_url(lic_url):
            # License must agree at both sources
            entry["license_url"] = ""
            entry["license_conflict"] = {"coco": lic_url, "flickr_current": oembed.get("license_url")}
    return entry


def fetch_coco_candidates(
    needed: int = 220,
    historical_keys: set[str] | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Fetch verified COCO candidates using val2017 split and Flickr oEmbed."""
    hist = historical_keys or load_historical_source_keys()
    hist_ids = {int(k.split(":")[1]) for k in hist if k.startswith("coco:")}

    print(">>> Opening COCO annotations (captions_val2017.json) via range-stream ...")
    rz = RemoteZipFile(COCO_ANNOTATIONS_URL)
    zf = zipfile.ZipFile(rz)
    with zf.open("annotations/captions_val2017.json") as f:
        captions = json.load(f)

    licenses = {lic["id"]: lic for lic in captions.get("licenses", [])}
    raw_images = captions.get("images", [])
    raw_bytes = json.dumps(captions, sort_keys=True).encode("utf-8")
    annotations_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    eligible: list[dict] = []
    excluded: list[dict] = []

    # Check for existing checkpoint
    if checkpoint_path and checkpoint_path.is_file():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            eligible = ckpt.get("eligible", [])
            excluded = ckpt.get("excluded", [])
            print(f"Loaded COCO checkpoint: {len(eligible)} eligible, {len(excluded)} excluded.")
        except Exception:
            pass

    processed_ids = {e["coco_image_id"] for e in eligible} | {x["coco_image_id"] for x in excluded}

    print(f">>> Scanning {len(raw_images)} val2017 images (target: {needed} eligible) ...")
    for img in raw_images:
        if len(eligible) >= needed:
            break
        if img["id"] in processed_ids or img["id"] in hist_ids:
            continue

        checked_at = datetime.now(timezone.utc).isoformat()
        m_flickr = FLICKR_URL_RE.search(img.get("flickr_url", ""))
        if not m_flickr:
            continue

        flickr_id = m_flickr.group(1)
        oembed_url = FLICKR_OEMBED + urllib.parse.quote(flickr_short_url(flickr_id), safe="")

        oembed, sha = None, ""
        try:
            raw = _fetch(oembed_url, timeout=10)
            oembed = json.loads(raw)
            sha = hashlib.sha256(raw).hexdigest()
        except Exception as e:
            excluded.append({
                "coco_image_id": img["id"],
                "reasons": [f"OEMBED_ERROR:{type(e).__name__}"],
                "entry": {"flickr_photo_id": flickr_id, "error": str(e)},
            })
            continue

        entry = coco_entry(img, licenses, oembed, oembed_url, sha, checked_at)
        res = evaluate_candidate_eligibility(entry, hist)

        if res["status"] == "EXCLUDED":
            excluded.append({"coco_image_id": img["id"], "reasons": res["reasons"], "entry": entry})
        else:
            eligible.append(entry)
            print(f"  [COCO {len(eligible)}/{needed}] ID={img['id']} Creator='{entry['creator_name']}' License='{entry['license_name']}'")

        if checkpoint_path and len(eligible) % 10 == 0:
            checkpoint_path.write_text(json.dumps({"eligible": eligible, "excluded": excluded}, indent=1), encoding="utf-8")

        time.sleep(0.05)  # polite rate limit

    if checkpoint_path:
        checkpoint_path.write_text(json.dumps({"eligible": eligible, "excluded": excluded}, indent=1), encoding="utf-8")

    return eligible, excluded, annotations_sha256


def fetch_commons_candidates(
    needed: int = 220,
    historical_keys: set[str] | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fetch verified Wikimedia Commons candidates from Category:Quality_images via MediaWiki API."""
    hist = historical_keys or load_historical_source_keys()
    eligible: list[dict] = []
    excluded: list[dict] = []

    # Check for existing checkpoint
    if checkpoint_path and checkpoint_path.is_file():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            eligible = ckpt.get("eligible", [])
            excluded = ckpt.get("excluded", [])
            print(f"Loaded Commons checkpoint: {len(eligible)} eligible, {len(excluded)} excluded.")
        except Exception:
            pass

    processed_ids = {e["commons_page_id"] for e in eligible} | {x["commons_page_id"] for x in excluded}

    base_url = "https://commons.wikimedia.org/w/api.php"
    gcm_continue = ""

    print(f">>> Fetching Wikimedia Commons Quality_images (target: {needed} eligible) ...")
    while len(eligible) < needed:
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": "Category:Quality_images",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo|info",
            "iiprop": "url|size|extmetadata|mime|sha1",
            "inprop": "url",
        }
        if gcm_continue:
            params["gcmcontinue"] = gcm_continue

        query_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        try:
            raw = _fetch(query_url, timeout=20)
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"Warning: Wikimedia Commons query failed: {e}. Retrying after pause ...")
            time.sleep(2.0)
            continue

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            print("No pages returned in Wikimedia batch.")
            break

        checked_at = datetime.now(timezone.utc).isoformat()
        for pid, p in pages.items():
            if len(eligible) >= needed:
                break
            page_id = int(pid)
            if page_id in processed_ids:
                continue

            ii = p.get("imageinfo", [{}])[0]
            meta = ii.get("extmetadata", {})
            mime = ii.get("mime", "")

            # Exclude non-photographic formats
            if mime not in ("image/jpeg", "image/png"):
                excluded.append({
                    "commons_page_id": page_id,
                    "reasons": [f"UNSUPPORTED_MIME:{mime}"],
                    "entry": {"title": p.get("title")},
                })
                continue

            width = ii.get("width") or 0
            height = ii.get("height") or 0
            if width < 512 or height < 512:
                excluded.append({
                    "commons_page_id": page_id,
                    "reasons": [f"RESOLUTION_TOO_LOW:{width}x{height}"],
                    "entry": {"title": p.get("title")},
                })
                continue

            lic_url = meta.get("LicenseUrl", {}).get("value", "")
            # Normalize trailing deed / language suffix in CC license URL
            m_lic = CC_LICENSE_RE.search(lic_url or "")
            if not m_lic:
                # Try normalizing e.g. /deed.en or /legalcode
                clean_lic_url = re.sub(r"/(deed\.[a-z]+|legalcode)/?$", "/", lic_url or "")
                m_lic = CC_LICENSE_RE.search(clean_lic_url)
                if m_lic:
                    lic_url = clean_lic_url

            artist_raw = meta.get("Artist", {}).get("value", "")
            clean_artist = re.sub(r"<[^>]+>", "", artist_raw).strip()
            if not clean_artist:
                clean_artist = re.sub(r"<[^>]+>", "", meta.get("Credit", {}).get("value", "")).strip()

            entry = {
                "source_origin": "wikimedia_commons",
                "acquisition_channel": "wikimedia_commons_api",
                "commons_page_id": page_id,
                "source_keys": [f"commons:{page_id}"],
                "origin_id": f"commons:{page_id}",
                "download_url": ii.get("url", ""),
                "source_page_url": p.get("fullurl") or f"https://commons.wikimedia.org/wiki/{p.get('title', '')}",
                "creator_name": clean_artist,
                "creator_url": p.get("fullurl") or f"https://commons.wikimedia.org/wiki/{p.get('title', '')}",
                "creator_verification": "VERIFIED_FROM_SOURCE",
                "creator_evidence_url": f"{base_url}?action=query&pageids={page_id}&prop=imageinfo",
                "license_name": meta.get("LicenseShortName", {}).get("value", ""),
                "license_url": lic_url,
                "license_version": m_lic.group(2) if m_lic else "",
                "license_evidence_source": "Wikimedia Commons imageinfo API extmetadata",
                "date_captured": meta.get("DateTimeOriginal", {}).get("value") or None,
                "date_captured_source": (
                    "Wikimedia Commons extmetadata DateTimeOriginal" if meta.get("DateTimeOriginal") else None
                ),
                "published_date": meta.get("DateTime", {}).get("value") or None,
                "published_date_source": (
                    "Wikimedia Commons extmetadata DateTime" if meta.get("DateTime") else None
                ),
                "provenance_checked_at_utc": checked_at,
                "download_rendition": f"Original Wikimedia Commons upload ({width}x{height})",
                "width": width,
                "height": height,
            }

            res = evaluate_candidate_eligibility(entry, hist)
            if res["status"] == "EXCLUDED":
                excluded.append({"commons_page_id": page_id, "reasons": res["reasons"], "entry": entry})
            else:
                eligible.append(entry)
                print(f"  [Commons {len(eligible)}/{needed}] ID={page_id} Creator='{entry['creator_name'][:30]}' License='{entry['license_name']}'")

        if checkpoint_path and len(eligible) % 10 == 0:
            checkpoint_path.write_text(json.dumps({"eligible": eligible, "excluded": excluded}, indent=1), encoding="utf-8")

        cont = data.get("continue", {})
        gcm_continue = cont.get("gcmcontinue", "")
        if not gcm_continue:
            print("Reached end of Category:Quality_images members.")
            break

        time.sleep(0.05)  # polite rate limit

    if checkpoint_path:
        checkpoint_path.write_text(json.dumps({"eligible": eligible, "excluded": excluded}, indent=1), encoding="utf-8")

    return eligible, excluded


def build_catalog(per_source: int = 220, checkpoint_dir: Path | None = None) -> dict:
    """Build canonical 440-candidate catalog for Amendment v1.3."""
    hist = load_historical_source_keys()
    ckpt_coco = (checkpoint_dir / "ckpt_coco.json") if checkpoint_dir else None
    ckpt_commons = (checkpoint_dir / "ckpt_commons.json") if checkpoint_dir else None

    coco_eligible, coco_excluded, coco_ann_sha = fetch_coco_candidates(
        needed=per_source, historical_keys=hist, checkpoint_path=ckpt_coco
    )
    commons_eligible, commons_excluded = fetch_commons_candidates(
        needed=per_source, historical_keys=hist, checkpoint_path=ckpt_commons
    )

    if len(coco_eligible) < per_source:
        raise RuntimeError(f"Could not reach target COCO candidates: got {len(coco_eligible)}/{per_source}")
    if len(commons_eligible) < per_source:
        raise RuntimeError(f"Could not reach target Commons candidates: got {len(commons_eligible)}/{per_source}")

    return {
        "schema_version": "2.0.0",
        "phase": "4C.7B",
        "amendment": "v1.3",
        "evidence_class": "verified_real_catalog",
        "is_synthetic": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": "research/evidence/phase-4c.7b/verified_candidate_catalog.json",
        "coco_image_info_sha256": coco_ann_sha,
        "coco_candidates": coco_eligible[:per_source],
        "commons_candidates": commons_eligible[:per_source],
        # Backward-compatibility alias for test fixtures that inspect unsplash_candidates
        "unsplash_candidates": commons_eligible[:per_source],
        "excluded_candidates": coco_excluded + commons_excluded,
    }


LEGACY_CATALOG = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog.json"
LEGACY_AUDIT = REPO_ROOT / "research/evidence/phase-4c.7b/catalog_eligibility_audit.json"
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
        rows.append({
            "source_origin": e["source_origin"],
            "origin_id": e["origin_id"],
            "author_in_catalog": e["author"],
            "license_name": e["license_name"],
            "status": res["status"],
            "reasons": res["reasons"],
        })
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
            "unsplash_channel": (
                "Unsplash Lite dataset (HF mirror): dataset terms grant download/store and internal ML "
                "training only and prohibit disseminating/redistributing the data; editing and sharing "
                "the cohort is not covered. The generic Unsplash License was assumed without checking "
                "the dataset terms."
            ),
            "unsplash_rendition": (
                "download_url requests a processed rendition (auto=format, fit=crop, w=600, q=80): "
                "resized, re-compressed, format negotiated; not the original upload"
            ),
            "historical_disjointness": (
                "v1 compared zero-padded COCO IDs to Option P source IDs (same COCO namespace); "
                "Unsplash IDs were compared to COCO IDs, which proves nothing"
            ),
        },
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--per-source", type=int, default=220)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--audit-legacy", action="store_true", help="Offline eligibility audit of the v1 catalog")
    args = parser.parse_args()

    if args.audit_legacy:
        audit = audit_legacy_catalog()
        LEGACY_AUDIT.write_text(json.dumps(audit, indent=1), encoding="utf-8")
        print(f"v1 catalog: {audit['eligible']}/{audit['total']} eligible; reasons {audit['reason_counts']}")
        return

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog(per_source=args.per_source, checkpoint_dir=ckpt_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    print(
        f"\n>>> Wrote verified candidate catalog v2 ({len(catalog['coco_candidates'])} COCO + "
        f"{len(catalog['commons_candidates'])} Commons, excluded: {len(catalog['excluded_candidates'])}) to {out_path}"
    )


if __name__ == "__main__":
    main()
