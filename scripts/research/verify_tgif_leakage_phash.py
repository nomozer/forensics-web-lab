"""Perceptual Hash (dHash) Leakage Audit Runner for TGIF-Train-Clean-Subset N=400.

Phase 4C.7B trace - Pre-Evaluation Leakage Audit.
Strictly verifies:
1. Difference Hash (dHash, 64-bit integer, hash_size=8) on all 800 packaged images.
2. Cross-split disjointness vs Option P historical images (Hamming distance <= 3).
3. Cross-source near-duplicate detection within N=400 across different source IDs (Hamming distance <= 3).
4. Cross-split disjointness vs Phase 4C.7B pilot/diagnostic/calibration images (Hamming distance <= 3).
5. Outputs machine-readable audit receipt to research/evidence/phase-4c.7b/tgif_train_phash_leakage_audit_receipt.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time
from typing import Any
import zipfile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ZIP = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/tgif_train_clean_subset_package.zip"
DEFAULT_OPTION_P_CSV = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
DEFAULT_RECEIPT_OUT = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_phash_leakage_audit_receipt.json"
MAX_HAMMING_DISTANCE = 3


def compute_dhash_image(img: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for fast near-duplicate image detection."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    if hasattr(resized, "get_flattened_data"):
        pixels = list(resized.get_flattened_data())
    else:
        pixels = list(resized.getdata())

    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(pixel_left > pixel_right)

    decimal_val = 0
    for index, value in enumerate(diff):
        if value:
            decimal_val += 1 << index
    return decimal_val


def run_phash_leakage_audit(
    package_zip_path: Path = DEFAULT_PACKAGE_ZIP,
    option_p_csv_path: Path = DEFAULT_OPTION_P_CSV,
    receipt_out_path: Path = DEFAULT_RECEIPT_OUT,
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> dict[str, Any]:
    """Execute complete perceptual hash leakage verification."""
    print("=" * 70)
    print("TGIF-Train-Clean-Subset N=400 Perceptual Hash (pHash/dHash) Leakage Audit")
    print("=" * 70)

    if not package_zip_path.is_file():
        raise FileNotFoundError(f"Package ZIP not found: {package_zip_path}")
    if not option_p_csv_path.is_file():
        raise FileNotFoundError(f"Option P manifest not found: {option_p_csv_path}")

    # 1. Compute dHash for all 800 images in N=400 package
    print("\n--- Step 1: Computing dHash for N=400 Packaged Images (800 total) ---")
    n400_hashes: dict[str, int] = {}
    source_to_hashes: dict[str, dict[str, int]] = {}

    with zipfile.ZipFile(package_zip_path, "r") as zf:
        manifest_data = json.loads(zf.read("manifest_copy.json"))
        for cand in manifest_data["selected_candidates"]:
            sid = cand["source_id"]
            cat = cand["category"]
            raw = cand["raw_id"]
            auth_name = f"authentic/{cat}/{raw}_orig.png"
            edit_name = f"edited/{cat}/{raw}_sd2.png"

            with Image.open(io.BytesIO(zf.read(auth_name))) as img_a:
                h_a = compute_dhash_image(img_a)
            with Image.open(io.BytesIO(zf.read(edit_name))) as img_e:
                h_e = compute_dhash_image(img_e)

            n400_hashes[auth_name] = h_a
            n400_hashes[edit_name] = h_e
            source_to_hashes[sid] = {"authentic": h_a, "ai_edited": h_e}

    print(f"Computed dHash for {len(n400_hashes)} images across {len(source_to_hashes)} sources.")

    # 2. Compute dHash for Option P historical images
    print("\n--- Step 2: Computing dHash for Historical Option P Images ---")
    option_p_hashes: dict[str, int] = {}
    with open(option_p_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["source_id"]
            for key, p_rel in [("authentic", row["authentic_path"]), ("canonical_edit", row["canonical_edit_path"])]:
                p = REPO_ROOT / p_rel
                if p.is_file():
                    with Image.open(p) as img:
                        option_p_hashes[f"{sid}_{key}"] = compute_dhash_image(img)

    print(f"Computed dHash for {len(option_p_hashes)} Option P historical images.")

    # 3. Cross-split check: N=400 vs Option P
    print(f"\n--- Step 3: Checking Cross-Split Collisions vs Option P (Hamming <= {max_hamming}) ---")
    option_p_collisions = []
    comparisons_count_op = len(n400_hashes) * len(option_p_hashes)
    for n_name, n_h in n400_hashes.items():
        for op_name, op_h in option_p_hashes.items():
            dist = bin(n_h ^ op_h).count("1")
            if dist <= max_hamming:
                option_p_collisions.append({
                    "n400_image": n_name,
                    "option_p_image": op_name,
                    "hamming_distance": dist,
                })

    print(f"Option P comparisons: {comparisons_count_op:,} | Collisions: {len(option_p_collisions)}")

    # 4. Cross-source check within N=400
    print(f"\n--- Step 4: Checking Internal Cross-Source Collisions in N=400 (Hamming <= {max_hamming}) ---")
    cross_source_collisions = []
    sids = list(source_to_hashes.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            s1, s2 = sids[i], sids[j]
            for k1 in ["authentic", "ai_edited"]:
                for k2 in ["authentic", "ai_edited"]:
                    dist = bin(source_to_hashes[s1][k1] ^ source_to_hashes[s2][k2]).count("1")
                    if dist <= max_hamming:
                        cross_source_collisions.append({
                            "source_id_1": s1,
                            "source_id_2": s2,
                            "type_1": k1,
                            "type_2": k2,
                            "hamming_distance": dist,
                        })

    print(f"Internal cross-source collisions: {len(cross_source_collisions)}")

    # 5. Check vs Phase 4C.7B pilot images
    print(f"\n--- Step 5: Checking vs Phase 4C.7B Pilot/Diag/Calib Images (Hamming <= {max_hamming}) ---")
    phase_dir = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b"
    phase_pilot_hashes: dict[str, int] = {}
    for p in phase_dir.rglob("*.png"):
        if any(k in str(p) for k in ["pilot-", "diag-", "calib-"]):
            with Image.open(p) as img:
                phase_pilot_hashes[p.name] = compute_dhash_image(img)

    phase_collisions = []
    for n_name, n_h in n400_hashes.items():
        for p_name, p_h in phase_pilot_hashes.items():
            dist = bin(n_h ^ p_h).count("1")
            if dist <= max_hamming:
                phase_collisions.append({
                    "n400_image": n_name,
                    "pilot_image": p_name,
                    "hamming_distance": dist,
                })

    print(f"Phase 4C.7B pilot images checked: {len(phase_pilot_hashes)} | Collisions: {len(phase_collisions)}")

    # Determine audit status
    all_clear = (len(option_p_collisions) == 0 and len(cross_source_collisions) == 0 and len(phase_collisions) == 0)
    audit_status = "PASS" if all_clear else "COLLISION_DETECTED"

    receipt = {
        "schema_version": "1.0.0",
        "phase": "4C.7B",
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": f"PHASH_LEAKAGE_AUDIT_{audit_status}",
        "algorithm": "difference_hash_dhash_64bit",
        "hash_size": 8,
        "max_hamming_distance_threshold": max_hamming,
        "n400_package": {
            "path": str(package_zip_path.name),
            "total_images_hashed": len(n400_hashes),
            "total_sources_hashed": len(source_to_hashes),
        },
        "option_p_disjointness": {
            "status": "PASS" if len(option_p_collisions) == 0 else "FAIL",
            "historical_images_checked": len(option_p_hashes),
            "comparisons_count": comparisons_count_op,
            "collisions_count": len(option_p_collisions),
            "collisions": option_p_collisions,
        },
        "internal_n400_disjointness": {
            "status": "PASS" if len(cross_source_collisions) == 0 else "FAIL",
            "cross_source_pairs_checked": len(sids) * (len(sids) - 1) // 2 * 4,
            "collisions_count": len(cross_source_collisions),
            "collisions": cross_source_collisions,
        },
        "phase_4c7b_development_disjointness": {
            "status": "PASS" if len(phase_collisions) == 0 else "FAIL",
            "pilot_images_checked": len(phase_pilot_hashes),
            "collisions_count": len(phase_collisions),
            "collisions": phase_collisions,
        },
        "conclusion": (
            "ZERO perceptual hash duplicates detected across historical Option P, internal cross-sources, "
            "and Phase 4C.7B development images at Hamming distance threshold <= 3."
        ) if all_clear else "Collisions detected, review immediately.",
    }

    receipt_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"\nWrote pHash leakage audit receipt to: {receipt_out_path}")
    print("=" * 70)
    print(f"AUDIT STATUS: {receipt['status']}")
    print("=" * 70)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Audit TGIF Train N=400 pHash Leakage")
    parser.add_argument("--package-zip", type=Path, default=DEFAULT_PACKAGE_ZIP)
    parser.add_argument("--option-p-csv", type=Path, default=DEFAULT_OPTION_P_CSV)
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT_OUT)
    parser.add_argument("--max-hamming", type=int, default=MAX_HAMMING_DISTANCE)
    args = parser.parse_args()

    run_phash_leakage_audit(
        package_zip_path=args.package_zip,
        option_p_csv_path=args.option_p_csv,
        receipt_out_path=args.receipt_out,
        max_hamming=args.max_hamming,
    )


if __name__ == "__main__":
    main()
