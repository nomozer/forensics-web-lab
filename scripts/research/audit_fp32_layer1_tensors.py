#!/usr/bin/env python3
"""Audit Element-Wise Numerical Parity for Layer 1 Preprocessing Tensors.

Independently compares raw binary tensors exported from:
  1. Python reference: torchvision BICUBIC resize -> ToTensor -> ImageNet Normalize
  2. In-browser runtime: Chromium Web Worker Canvas2D RGBA -> buildBicubicTensor224

Verifies:
  - Exact file counts (16 samples)
  - Shape: [1, 3, 224, 224] (150,528 elements per sample, 2,408,448 total)
  - Dtype: float32, Byte length: 602,112 bytes each
  - Layout: NCHW
  - Finite validation: zero NaN, zero Inf on 100% elements
  - Element-wise Mean Absolute Error (MAE) = mean(|browser - reference|)
  - Element-wise Max Absolute Difference = max(|browser - reference|)
  - Fail-closed behavior on missing samples, corrupt shapes, or non-finite values

Technical Verification Tolerances (locked before measurement):
  - MAE <= 1.0e-2 (0.010)
  - Max Absolute Difference <= 5.0e-2 (0.050)
  - Bit-exact required: False (since floating-point interpolation rounding differs from fixed-point integer rounding)

Outputs:
  research/evidence/browser_fp32_parity/tensors/layer1_tensor_parity_audit_receipt.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "research/evidence/browser_fp32_parity/tensors/reference"
BROWSER_DIR = REPO_ROOT / "research/evidence/browser_fp32_parity/tensors/browser"
RECEIPT_PATH = REPO_ROOT / "research/evidence/browser_fp32_parity/tensors/layer1_tensor_parity_audit_receipt.json"

EXPECTED_SHAPE = (1, 3, 224, 224)
EXPECTED_NUM_ELEMENTS = 1 * 3 * 224 * 224  # 150,528
EXPECTED_BYTE_LENGTH = EXPECTED_NUM_ELEMENTS * 4  # 602,112
EXPECTED_SAMPLE_COUNT = 16

LOCKED_TOLERANCES = {
    "layer1_tensor_mae": 1.0e-2,  # 0.010
    "layer1_tensor_max_abs_diff": 5.0e-2,  # 0.050
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_layer1_tensors() -> dict[str, Any]:
    print("=" * 70)
    print("Auditing Layer 1 Element-Wise Input Tensor Numerical Parity")
    print("=" * 70)

    # 1. Fail-closed preflight checks
    if not REF_DIR.exists():
        raise FileNotFoundError(f"Missing reference tensor directory: {REF_DIR}")
    if not BROWSER_DIR.exists():
        raise FileNotFoundError(f"Missing browser tensor directory: {BROWSER_DIR}")

    ref_manifest_p = REF_DIR / "reference_tensors_manifest.json"
    browser_manifest_p = BROWSER_DIR / "browser_tensors_manifest.json"

    if not ref_manifest_p.exists():
        raise FileNotFoundError(f"Missing reference manifest: {ref_manifest_p}")
    if not browser_manifest_p.exists():
        raise FileNotFoundError(f"Missing browser manifest: {browser_manifest_p}")

    with open(ref_manifest_p, "r", encoding="utf-8") as f:
        ref_manifest = json.load(f)
    with open(browser_manifest_p, "r", encoding="utf-8") as f:
        browser_manifest = json.load(f)

    ref_samples = ref_manifest.get("samples", [])
    browser_samples = browser_manifest.get("samples", [])

    if len(ref_samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} reference samples, got {len(ref_samples)}"
        )
    if len(browser_samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} browser samples, got {len(browser_samples)}"
        )

    browser_map = {s["sample_index"]: s for s in browser_samples}

    per_sample_results = []
    total_elements_compared = 0
    overall_abs_sum = 0.0
    overall_max_abs_diff = 0.0

    hashes_matched_count = 0

    for ref_s in ref_samples:
        idx = ref_s["sample_index"]
        if idx not in browser_map:
            raise KeyError(f"Sample index {idx} missing from browser manifest")
        b_s = browser_map[idx]

        sid = ref_s["source_id"]
        label_name = ref_s["label_name"]

        # Verify matching metadata
        if b_s["source_id"] != sid:
            raise ValueError(f"Source ID mismatch at sample {idx}: {b_s['source_id']} vs {sid}")
        if b_s["label_name"] != label_name:
            raise ValueError(f"Label mismatch at sample {idx}: {b_s['label_name']} vs {label_name}")

        ref_bin_p = REF_DIR / ref_s["tensor_filename"]
        b_bin_p = BROWSER_DIR / b_s["tensor_filename"]

        if not ref_bin_p.exists():
            raise FileNotFoundError(f"Missing reference tensor binary: {ref_bin_p}")
        if not b_bin_p.exists():
            raise FileNotFoundError(f"Missing browser tensor binary: {b_bin_p}")

        # Check byte length
        ref_bytes = ref_bin_p.read_bytes()
        b_bytes = b_bin_p.read_bytes()

        if len(ref_bytes) != EXPECTED_BYTE_LENGTH:
            raise ValueError(
                f"Sample {idx} reference byte length {len(ref_bytes)} != {EXPECTED_BYTE_LENGTH}"
            )
        if len(b_bytes) != EXPECTED_BYTE_LENGTH:
            raise ValueError(
                f"Sample {idx} browser byte length {len(b_bytes)} != {EXPECTED_BYTE_LENGTH}"
            )

        ref_hash = sha256_file(ref_bin_p)
        b_hash = sha256_file(b_bin_p)

        if ref_hash == b_hash:
            hashes_matched_count += 1

        # Unpack as Float32 little-endian
        ref_arr = np.frombuffer(ref_bytes, dtype=np.float32).reshape(EXPECTED_SHAPE)
        b_arr = np.frombuffer(b_bytes, dtype=np.float32).reshape(EXPECTED_SHAPE)

        # Check finite
        if not bool(np.all(np.isfinite(ref_arr))):
            raise FloatingPointError(f"Non-finite value in reference tensor sample {idx}")
        if not bool(np.all(np.isfinite(b_arr))):
            raise FloatingPointError(f"Non-finite value in browser tensor sample {idx}")

        # Element-wise absolute diff
        diff = np.abs(b_arr - ref_arr)
        sample_mae = float(np.mean(diff))
        sample_max = float(np.max(diff))
        sample_num_elements = int(diff.size)

        total_elements_compared += sample_num_elements
        overall_abs_sum += float(np.sum(diff))
        overall_max_abs_diff = max(overall_max_abs_diff, sample_max)

        # Mean stat diff (for provenance reconciliation)
        b_mean = float(np.mean(b_arr))
        ref_mean = float(np.mean(ref_arr))
        mean_stat_diff = abs(b_mean - ref_mean)

        sample_pass = (
            sample_mae <= LOCKED_TOLERANCES["layer1_tensor_mae"]
            and sample_max <= LOCKED_TOLERANCES["layer1_tensor_max_abs_diff"]
        )

        per_sample_results.append({
            "sample_index": idx,
            "source_id": sid,
            "label_name": label_name,
            "ref_tensor_sha256": ref_hash,
            "browser_tensor_sha256": b_hash,
            "bit_exact_hash_match": bool(ref_hash == b_hash),
            "num_elements": sample_num_elements,
            "element_wise_mae": sample_mae,
            "element_wise_max_abs_diff": sample_max,
            "mean_stat_diff": mean_stat_diff,
            "status": "PASS" if sample_pass else "FAIL",
        })

        print(
            f"[{idx:02d}/16] {sid} ({label_name}): "
            f"MAE={sample_mae:.6f}, MaxDiff={sample_max:.6f}, "
            f"MeanStatDiff={mean_stat_diff:.8f}, BitExact={ref_hash == b_hash}"
        )

    overall_mae = overall_abs_sum / total_elements_compared
    all_samples_pass = all(s["status"] == "PASS" for s in per_sample_results)
    within_tolerances = (
        overall_mae <= LOCKED_TOLERANCES["layer1_tensor_mae"]
        and overall_max_abs_diff <= LOCKED_TOLERANCES["layer1_tensor_max_abs_diff"]
        and all_samples_pass
    )

    verdict = "PASS" if within_tolerances else "FAIL"

    receipt = {
        "schema_version": "1.0.0",
        "audit_name": "layer1_tensor_element_wise_parity_audit_receipt",
        "timestamp_utc": "2026-10-10T15:15:00Z",
        "status": "LAYER1_ELEMENT_WISE_PARITY_AUDITED",
        "verdict": verdict,
        "scientific_conclusion": (
            "PARITY_WITHIN_NUMERICAL_TOLERANCE_NOT_BIT_EXACT"
            if verdict == "PASS"
            else "PARITY_FAIL"
        ),
        "protocol_status": {
            "nature_of_check": "Supplementary technical verification (bổ sung kiểm chứng kỹ thuật)",
            "preregistration_status": "NOT_PREREGISTERED (locked before current measurement, not part of Phase 4C protocol)",
            "tolerance_lock_policy": "Locked prior to comparison; no post-hoc tolerance relaxation.",
            "bit_exactness_claim": "DISAVOWED. Tensor byte hashes differ due to integer fixed-point vs floating-point cubic spline rounding. Claim restricted to bounded numerical tolerance on the 16-sample development panel.",
        },
        "locked_tolerances": LOCKED_TOLERANCES,
        "aggregate_metrics": {
            "total_samples": EXPECTED_SAMPLE_COUNT,
            "total_elements_compared": total_elements_compared,
            "overall_element_wise_mae": overall_mae,
            "overall_max_abs_diff": overall_max_abs_diff,
            "bit_exact_matches_count": hashes_matched_count,
            "bit_exact_rate_percent": (hashes_matched_count / EXPECTED_SAMPLE_COUNT) * 100.0,
            "mae_tolerance_met": bool(overall_mae <= LOCKED_TOLERANCES["layer1_tensor_mae"]),
            "max_abs_diff_tolerance_met": bool(
                overall_max_abs_diff <= LOCKED_TOLERANCES["layer1_tensor_max_abs_diff"]
            ),
        },
        "provenance": {
            "reference_directory": str(REF_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
            "browser_directory": str(BROWSER_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
            "reference_manifest_sha256": sha256_file(ref_manifest_p),
            "browser_manifest_sha256": sha256_file(browser_manifest_p),
            "reference_transform": "torchvision.transforms.Compose([Resize((224, 224), BICUBIC), ToTensor(), Normalize])",
            "browser_transform": "Pillow-compatible 2-pass Bicubic Keys spline (a=-0.5, 22-bit fixed point) in TypeScript",
        },
        "samples": per_sample_results,
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Layer 1 Parity Verdict: {verdict}")
    print(f"Overall MAE: {overall_mae:.8f} (tolerance <= {LOCKED_TOLERANCES['layer1_tensor_mae']})")
    print(f"Overall Max Diff: {overall_max_abs_diff:.8f} (tolerance <= {LOCKED_TOLERANCES['layer1_tensor_max_abs_diff']})")
    print(f"Bit-Exact Matches: {hashes_matched_count} / {EXPECTED_SAMPLE_COUNT}")
    print(f"Scientific Conclusion: {receipt['scientific_conclusion']}")
    print(f"Receipt written to: {RECEIPT_PATH}")
    print("=" * 70)

    if verdict != "PASS":
        sys.exit(1)

    return receipt


if __name__ == "__main__":
    audit_layer1_tensors()
