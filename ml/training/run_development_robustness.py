#!/usr/bin/env python3
"""CLI runner for the development robustness experiment (no fitting of any kind).

Modes:
  preflight  verify the protocol, the 682 development image hashes, the frozen
             4C.3B late-fusion and ablation early-fusion fold artifacts, fold
             bindings, sample alignment, the 4C.3B feature caches, and that cached
             features + rebuilt frozen models reproduce the stored predictions.
  pilot      technical check on outer fold 0 only: original condition, its
             reproduction gate, then the five transformed conditions.
  full       all 341 sources: original + gate first, then transformed conditions.
Completed conditions are resumed after hash verification; tampered artifacts fail.

All data paths come from the CLI. Outputs must live outside Git-tracked paths.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ml.training.calibrated_late_fusion import (
    feature_array_commitment,
    sha256_file,
    sha256_text_file,
    utc_now,
)
from ml.training.development_robustness import (
    CONDITION_IDS,
    DATA_ORIGIN_REAL,
    DATA_ORIGIN_SYNTHETIC,
    SCHEMA_VERSION,
    FeatureExtractor,
    MissingArtifactsError,
    ReferenceFeatureExtractor,
    compare_to_reference,
    load_frozen_artifacts,
    load_protocol,
    reference_metric_check,
    resolve_repo_path,
    run_scope,
    scope_samples,
    score_samples,
    verify_fold_binding,
)
from ml.training.visual_dsp_ablation import (
    build_grouped_nested_folds,
    build_or_load_dsp_cache,
    build_or_load_visual_cache,
    load_development_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILES = (
    "shared/visual_features.pt",
    "shared/visual_features_receipt.json",
    "shared/dsp_features.pt",
    "shared/dsp_features_receipt.json",
)


def check_output_dir(output_dir: Path, repo_root: Path = REPO_ROOT) -> None:
    """Refuse output inside the repository except under the git-ignored data/ tree."""
    resolved = output_dir.resolve()
    root = repo_root.resolve()
    if resolved == root or root in resolved.parents:
        data_root = root / "data"
        if not (resolved == data_root or data_root in resolved.parents):
            raise ValueError(
                f"Output directory {output_dir} is inside the repository; use a path outside Git "
                "or under data/research/local-artifacts/"
            )


def load_cached_features(ablation_dir: Path, pairs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    missing = [str(ablation_dir / name) for name in CACHE_FILES if not (ablation_dir / name).is_file()]
    if missing:
        raise MissingArtifactsError(missing)
    # Both loaders only read when receipt + cache exist (verified above); they verify
    # the receipt, the cache SHA-256 and the source order.
    visual, _, _ = build_or_load_visual_cache(
        pairs=pairs, weights_path=Path("unused"), output_root=ablation_dir, device=None  # type: ignore[arg-type]
    )
    dsp, _, _ = build_or_load_dsp_cache(pairs=pairs, output_root=ablation_dir)
    return visual, dsp


def cached_feature_gate(
    *, pairs: list[Any], folds: Any, frozen: Any, reference: dict, visual: np.ndarray, dsp: np.ndarray, tolerance: float
) -> dict[str, Any]:
    samples = scope_samples(pairs, folds, "full")
    rows = score_samples(
        condition="cached_original",
        samples=samples,
        visual=visual.reshape(-1, visual.shape[-1]),
        dsp=dsp.reshape(-1, dsp.shape[-1]),
        folds=folds,
        frozen=frozen,
    )
    return compare_to_reference(rows, reference, tolerance)


def run_pipeline(
    *,
    mode: str,
    protocol_path: Path,
    manifest_path: Path,
    data_root: Path,
    weights_path: Path | None,
    late_fusion_dir: Path,
    ablation_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    save_images: bool = True,
    batch_size: int = 32,
    repo_root: Path = REPO_ROOT,
    extractor_factory: Callable[[], FeatureExtractor] | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    check_output_dir(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"[{utc_now()}] Verifying frozen 4C.3B late-fusion and ablation early-fusion artifacts...")
    loaded = load_frozen_artifacts(protocol=protocol, late_fusion_dir=late_fusion_dir, ablation_dir=ablation_dir)
    bindings = loaded["late_fusion_bindings"]
    from ml.training.phase_4c2h_development import source_membership_commitment

    if bindings["source_membership_commitment"] != source_membership_commitment(source_ids):
        raise ValueError("Development manifest sources differ from the 4C.3B run")
    verify_fold_binding(folds=folds, bindings=bindings, reference=loaded["reference"])
    expected_samples = {f"{s}:{label}" for s in source_ids for label in ("authentic", "ai_edited")}
    for recipe in ("visual_calibrated", "early_fusion", "late_fusion_stacked"):
        if {sid for (r, sid) in loaded["reference"] if r == recipe} != expected_samples:
            raise ValueError(f"Reference predictions for {recipe} are not aligned with the 682 samples")

    visual, dsp = load_cached_features(ablation_dir, pairs)
    commitments = protocol["frozen_models"]["feature_caches"]
    if feature_array_commitment(visual) != commitments["visual_feature_commitment"] or (
        feature_array_commitment(dsp) != commitments["dsp_feature_commitment"]
    ):
        raise ValueError("4C.3B feature caches differ from the protocol commitments")
    gate_a = cached_feature_gate(
        pairs=pairs,
        folds=folds,
        frozen=loaded["frozen"],
        reference=loaded["reference"],
        visual=visual,
        dsp=dsp,
        tolerance=float(protocol["original_reproduction_gate"]["cached_feature_model_check"]["max_abs_probability_difference"]),
    )
    late_spec = protocol["frozen_models"]["late_fusion"]
    metric_gate = reference_metric_check(
        loaded["reference"],
        resolve_repo_path(repo_root, late_spec["reference_summary_path"]),
        late_spec["reference_summary_sha256"],
    )

    extractor: FeatureExtractor | None = None
    if mode != "preflight":
        if gate_a["status"] != "PASS" or metric_gate["status"] != "PASS":
            raise RuntimeError("Cached-feature model reproduction failed; robustness runs are blocked")
        extractor = (
            extractor_factory()
            if extractor_factory is not None
            else ReferenceFeatureExtractor(
                weights_path or Path("missing-weights"),
                device=device,
                expected_weights_sha256=protocol["artifact_bindings"]["pretrained_weights"]["sha256"],
            )
        )
    data_origin = (
        DATA_ORIGIN_REAL
        if bindings.get("data_origin") == DATA_ORIGIN_REAL and (extractor is None or extractor.is_reference_pipeline)
        else DATA_ORIGIN_SYNTHETIC
    )
    run_bindings = {
        "experiment_id": protocol["experiment_id"],
        "data_origin": data_origin,
        "protocol_sha256": sha256_text_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_membership_commitment": bindings["source_membership_commitment"],
        "late_fusion_run_manifest_sha256": late_spec["run_manifest_sha256"],
        "frozen_parameter_digests": [m.parameter_digest for m in loaded["frozen"]],
        "frozen_artifact_hashes": loaded["artifact_hashes"],
        "fold_lock_outer_test": [f["outer_test_commitment"] for f in bindings["fold_lock"]["outer_folds"]],
    }

    if mode == "preflight":
        status = "PREFLIGHT_PASS" if gate_a["status"] == "PASS" and metric_gate["status"] == "PASS" else "PREFLIGHT_FAIL"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "data_origin": data_origin,
            "sources": len(pairs),
            "samples": 2 * len(pairs),
            "conditions": list(CONDITION_IDS),
            "cached_feature_model_check": gate_a,
            "reference_metric_check": metric_gate,
            "weights_present": bool(weights_path and weights_path.is_file()),
            "run_bindings": run_bindings,
            "fits": 0,
            "created_at_utc": utc_now(),
        }
        (output_dir / "preflight_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[{utc_now()}] {status} ({data_origin})")
        return receipt

    assert extractor is not None
    samples = scope_samples(pairs, folds, mode)
    summary = run_scope(
        scope=mode,
        output_dir=output_dir,
        samples=samples,
        folds=folds,
        frozen=loaded["frozen"],
        reference=loaded["reference"],
        extractor=extractor,
        protocol=protocol,
        run_bindings=run_bindings,
        save_images=save_images,
        batch_size=batch_size,
    )
    print(f"[{utc_now()}] {mode.upper()} complete: {len(summary['conditions'])} conditions, 0 fits ({data_origin})")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Development robustness runner (frozen models, no fitting)")
    parser.add_argument("--mode", choices=["preflight", "pilot", "full"], default="preflight")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "ml/configs/development_robustness_protocol.yaml")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest_pilot_a_option_p.csv")
    parser.add_argument("--data-root", type=Path, required=True, help="directory the manifest paths are relative to")
    parser.add_argument("--weights-path", type=Path, default=None, help="mobilenet_v3_small-047dcff4.pth")
    parser.add_argument("--late-fusion-dir", type=Path, required=True, help="4C.3B calibrated_late_fusion output dir")
    parser.add_argument("--ablation-dir", type=Path, required=True, help="visual/DSP ablation output dir (fits/ and shared/)")
    parser.add_argument("--output-dir", type=Path, required=True, help="outside Git or under data/research/local-artifacts/")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-save-images", action="store_true", help="do not keep transformed image copies")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_pipeline(
        mode=args.mode,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        data_root=args.data_root,
        weights_path=args.weights_path,
        late_fusion_dir=args.late_fusion_dir,
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
        device=args.device,
        save_images=not args.no_save_images,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
