#!/usr/bin/env python3
"""CLI runner for the calibrated late fusion development experiment.

Modes:
  preflight  verify protocol, manifest, 682 image hashes, weights and feature-cache
             availability; no fitting.
  pilot      outer fold 0 only (61 fits) to verify artifacts and runtime.
  full       all 5 outer folds (305 fits); verified completed folds are resumed.

Feature caches: pass --feature-cache-dir pointing at the visual/DSP ablation
output directory to reuse its verified `shared/visual_features.pt` and
`shared/dsp_features.pt` read-only (preferred: identical features make every
prediction paired with the ablation). Without it, caches are built under
--output-dir/shared exactly as the ablation built them.

Zero locked-test access: only the 341 development rows of the manifest load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ml.training.calibrated_late_fusion import (
    RECIPE_IDS,
    SCHEMA_VERSION,
    atomic_write_json,
    build_run_bindings,
    load_protocol,
    planned_budget,
    run_matrix,
    sha256_file,
    sha256_text_file,
    utc_now,
)
from ml.training.visual_dsp_ablation import (
    build_grouped_nested_folds,
    build_or_load_dsp_cache,
    build_or_load_visual_cache,
    load_development_pairs,
    public_fold_lock,
)

CACHE_FILES = (
    "shared/visual_features.pt",
    "shared/visual_features_receipt.json",
    "shared/dsp_features.pt",
    "shared/dsp_features_receipt.json",
)


def resolve_cache_root(feature_cache_dir: Path | None, output_dir: Path) -> tuple[Path, bool]:
    """Return (cache_root, reuse_only). Reused caches must already be complete."""
    if feature_cache_dir is None:
        return output_dir, False
    missing = [name for name in CACHE_FILES if not (feature_cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"--feature-cache-dir {feature_cache_dir} is missing {missing}; refusing to build "
            "caches inside another experiment's directory"
        )
    return feature_cache_dir, True


def verify_weights(weights_path: Path, protocol: dict[str, Any]) -> str:
    expected = protocol["artifact_bindings"]["pretrained_weights"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights missing: {weights_path}")
    actual = sha256_file(weights_path)
    if actual != expected["sha256"]:
        raise ValueError(f"Weights SHA mismatch: {actual} != {expected['sha256']}")
    return actual


def run_pipeline(
    *,
    mode: str,
    protocol_path: Path,
    manifest_path: Path,
    data_root: Path,
    weights_path: Path,
    output_dir: Path,
    feature_cache_dir: Path | None,
    device_name: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root, reuse_only = resolve_cache_root(feature_cache_dir, output_dir)
    caches_present = all((cache_root / name).is_file() for name in CACHE_FILES)
    if not caches_present:
        # Building the visual cache needs the exact pretrained backbone.
        verify_weights(weights_path, protocol)

    print(f"[{utc_now()}] Loading 341 development pairs and verifying 682 image hashes...")
    pairs = load_development_pairs(
        manifest_path=manifest_path,
        data_root=data_root,
        expected_manifest_sha256=protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"],
        verify_hashes=True,
    )
    source_ids = [pair.source_id for pair in pairs]
    cv = protocol["grouped_nested_cv"]
    folds = build_grouped_nested_folds(
        source_ids,
        outer_folds=cv["outer_folds"],
        inner_folds=cv["inner_folds"],
        outer_seed=cv["outer_seed"],
        inner_seed=cv["inner_seed"],
    )
    fold_lock = public_fold_lock(folds)

    if mode == "preflight":
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PREFLIGHT_PASS",
            "mode": mode,
            "sources": len(pairs),
            "samples": 2 * len(pairs),
            "recipes": list(RECIPE_IDS),
            "outer_folds": len(folds),
            "planned_budget": planned_budget(protocol),
            "manifest_sha256": sha256_file(manifest_path),
            "protocol_sha256": sha256_text_file(protocol_path),
            "weights_sha256": sha256_file(weights_path) if weights_path.is_file() else None,
            "feature_cache_root_reused": reuse_only,
            "feature_caches_present": caches_present,
            "fold_lock": fold_lock,
            "created_at_utc": utc_now(),
        }
        atomic_write_json(output_dir / "preflight_receipt.json", receipt)
        print(f"[{utc_now()}] PREFLIGHT PASS (caches present: {caches_present}, reused: {reuse_only})")
        return receipt

    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[{utc_now()}] Loading visual features from {cache_root / 'shared'} ...")
    visual, visual_ids, visual_receipt = build_or_load_visual_cache(
        pairs=pairs, weights_path=weights_path, output_root=cache_root, device=device
    )
    print(f"[{utc_now()}] Loading DSP features from {cache_root / 'shared'} ...")
    dsp, dsp_ids, dsp_receipt = build_or_load_dsp_cache(pairs=pairs, output_root=cache_root)
    if visual_ids != source_ids or dsp_ids != source_ids:
        raise ValueError("Feature cache source order differs from the development manifest order")

    bindings = build_run_bindings(
        protocol_path=protocol_path,
        data_origin="development_real",
        source_ids=source_ids,
        visual_features=visual,
        dsp_features=dsp,
        fold_lock=fold_lock,
        feature_cache_receipts={
            "visual_cache_sha256": visual_receipt.get("cache_sha256"),
            "dsp_cache_sha256": dsp_receipt.get("cache_sha256"),
            "reused_from_feature_cache_dir": reuse_only,
        },
    )
    outer_folds = (
        list(protocol["budget"]["pilot_outer_folds"]) if mode == "pilot" else list(range(len(folds)))
    )
    receipts = run_matrix(
        output_dir=output_dir,
        outer_folds=outer_folds,
        folds=folds,
        visual_features=visual,
        dsp_features=dsp,
        source_ids=source_ids,
        protocol=protocol,
        bindings=bindings,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "outer_folds": outer_folds,
        "total_fits": sum(r["fit_counts"]["total_fits"] for r in receipts),
        "unconverged_fits": sum(r["fit_counts"]["unconverged_fits"] for r in receipts),
        "fit_wall_seconds": sum(r["fit_wall_seconds"] for r in receipts),
        "folds": receipts,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / f"{mode}_summary.json", summary)
    print(
        f"[{utc_now()}] {mode.upper()} complete: {len(receipts)} outer fold(s), "
        f"{summary['total_fits']} fits, {summary['unconverged_fits']} unconverged"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrated late fusion runner (development only)")
    parser.add_argument("--mode", choices=["preflight", "pilot", "full"], default="preflight")
    parser.add_argument(
        "--protocol", type=Path, default=Path("ml/configs/calibrated_late_fusion_protocol.yaml")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/research/tgif/manifests/manifest_pilot_a_option_p.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/research/tgif"))
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("models/research/pretrained/mobilenet_v3_small-047dcff4.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/local-artifacts/calibrated_late_fusion"),
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Visual/DSP ablation output dir whose shared/ caches are reused read-only",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    result = run_pipeline(
        mode=args.mode,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        data_root=args.data_root,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        feature_cache_dir=args.feature_cache_dir,
        device_name=args.device,
    )
    print(json.dumps({"status": result.get("status", result.get("mode"))}))
    return result


if __name__ == "__main__":
    main()
