#!/usr/bin/env python3
"""CLI runner for the Phase 4C.6A controlled DSP-augmentation experiment.

Modes:
  preflight  0 fits. Verify protocol, manifest (development rows only, no image
             reads), 4C.3B late-fusion and ablation fold artifacts, fold lock,
             ablation feature caches, the 4C.4B robustness full scope bound in
             the protocol, condition features; rebuild visual_calibrated and
             late_fusion_stacked from the stored fold models and reproduce the
             stored 4C.3B (original) and 4C.4B (all 6 conditions) predictions.
  pilot      outer fold 0: baseline refit + gate, then the augmented DSP branch.
  full       all 5 outer folds; verified completed folds are resumed, never refit.

Every input path comes from the CLI. Outputs must live outside Git (or under the
git-ignored data/ tree). Missing artifacts stop the run before any fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ml.training import calibrated_late_fusion as clf
from ml.training import development_robustness as dr
from ml.training import dsp_augmentation as aug
from ml.training.phase_4c2h_development import source_membership_commitment
from ml.training.run_development_robustness import check_output_dir, load_cached_features
from ml.training.visual_dsp_ablation import (
    build_grouped_nested_folds,
    load_development_pairs,
    public_fold_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class GateError(RuntimeError):
    """A reproduction gate failed; the requested work is blocked."""


def _inside(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def _json_sha(value: Any) -> str:
    return clf.sha256_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))


def load_condition_features(scope_dir: Path, scope: dict[str, Any], source_ids: list[str]) -> tuple[dict, dict]:
    """Per-condition [sources, 2, dim] features from the verified robustness full scope."""
    expected_ids = [f"{s}:{label}" for s in source_ids for label in clf.LABEL_NAMES]
    visual_by, dsp_by = {}, {}
    for condition in aug.CONDITIONS:
        directory = dr.condition_directory(scope_dir, condition)
        ids = [r["sample_id"] for r in json.loads((directory / "images.json").read_text(encoding="utf-8"))]
        if ids != expected_ids:
            raise ValueError(f"Condition {condition!r}: sample order differs from the development manifest order")
        with np.load(directory / "features.npz", allow_pickle=False) as data:
            visual, dsp = data["visual"], data["dsp"]
        receipt = scope["receipts"][condition]
        if clf.feature_array_commitment(visual) != receipt["visual_feature_commitment"] or (
            clf.feature_array_commitment(dsp) != receipt["dsp_feature_commitment"]
        ):
            raise ValueError(f"Condition {condition!r}: features differ from the receipt commitments")
        if visual.shape != (len(ids), 576) or dsp.shape != (len(ids), 16) or visual.dtype != np.float32 or dsp.dtype != np.float32:
            raise ValueError(f"Condition {condition!r}: unexpected feature shape or dtype")
        if not (np.isfinite(visual).all() and np.isfinite(dsp).all()):
            raise ValueError(f"Condition {condition!r}: non-finite (NaN/Inf) features")
        visual_by[condition] = visual.reshape(len(source_ids), 2, -1)
        dsp_by[condition] = dsp.reshape(len(source_ids), 2, -1)
    return visual_by, dsp_by


def prepare(
    *,
    protocol_path: Path,
    manifest_path: Path,
    data_root: Path,
    ablation_dir: Path,
    late_fusion_dir: Path,
    robustness_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """All checks that need no fitting. Raises on the first missing or mismatched artifact."""
    for historical in (ablation_dir, late_fusion_dir, robustness_dir, data_root):
        if _inside(output_dir, historical):
            raise ValueError(f"Output dir {output_dir} lies inside historical input {historical}; refusing")
    check_output_dir(output_dir, REPO_ROOT)
    protocol = aug.load_protocol(protocol_path, REPO_ROOT)
    late_protocol = clf.load_protocol(REPO_ROOT / protocol["predecessors"]["late_fusion_protocol"]["path"])
    robustness_protocol = dr.load_protocol(REPO_ROOT / protocol["predecessors"]["robustness_protocol"]["path"])

    pairs = load_development_pairs(
        manifest_path=manifest_path,
        data_root=data_root,
        expected_manifest_sha256=robustness_protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"],
        verify_hashes=False,  # images are never read; features come from verified caches
    )
    source_ids = [p.source_id for p in pairs]
    if len(source_ids) != int(protocol["data_scope"]["expected_sources"]):
        raise ValueError(f"Expected {protocol['data_scope']['expected_sources']} development sources, found {len(source_ids)}")
    cv = protocol["grouped_nested_cv"]
    folds = build_grouped_nested_folds(
        source_ids, outer_folds=cv["outer_folds"], inner_folds=cv["inner_folds"],
        outer_seed=cv["outer_seed"], inner_seed=cv["inner_seed"],
    )

    loaded = dr.load_frozen_artifacts(protocol=robustness_protocol, late_fusion_dir=late_fusion_dir, ablation_dir=ablation_dir)
    late_bindings = loaded["late_fusion_bindings"]
    if late_bindings["source_membership_commitment"] != source_membership_commitment(source_ids):
        raise ValueError("Development manifest sources differ from the 4C.3B run")
    dr.verify_fold_binding(folds=folds, bindings=late_bindings, reference=loaded["reference"])

    visual, dsp = load_cached_features(ablation_dir, pairs)
    caches = robustness_protocol["frozen_models"]["feature_caches"]
    if clf.feature_array_commitment(visual) != caches["visual_feature_commitment"] or (
        clf.feature_array_commitment(dsp) != caches["dsp_feature_commitment"]
    ):
        raise ValueError("Ablation feature caches differ from the 4C.3B commitments")

    from scripts.research.analyze_development_robustness import load_full_scope

    scope = load_full_scope(robustness_dir)
    scope_dir = Path(robustness_dir).resolve() / "full"
    bound = protocol["predecessors"]["robustness_full_scope"]
    run_bindings = scope["bindings"]
    if run_bindings["protocol_sha256"] != protocol["predecessors"]["robustness_protocol"]["sha256"]:
        raise ValueError("Robustness run used a different robustness protocol")
    if _json_sha(run_bindings) != bound["run_bindings_sha256"]:
        raise ValueError("Robustness run bindings differ from the protocol binding")
    if clf.sha256_file(scope_dir / "original_gate.json") != bound["original_gate_sha256"]:
        raise ValueError("Robustness original gate differs from the protocol binding")
    for condition in aug.CONDITIONS:
        actual = clf.sha256_file(dr.condition_directory(scope_dir, condition) / "condition_receipt.json")
        if actual != bound["condition_receipts_sha256"][condition]:
            raise ValueError(f"Robustness condition receipt {condition!r} differs from the protocol binding")
    if [m.parameter_digest for m in loaded["frozen"]] != run_bindings["frozen_parameter_digests"]:
        raise ValueError("Robustness run used different frozen fold models")

    visual_by, dsp_by = load_condition_features(scope_dir, scope, source_ids)
    if clf.feature_array_commitment(visual_by["original"]) != caches["visual_feature_commitment"] or (
        clf.feature_array_commitment(dsp_by["original"]) != caches["dsp_feature_commitment"]
    ):
        raise ValueError("Robustness original-condition features differ from the 4C.3B caches")

    # Preflight reconstruction (0 fits): stored fold models + reused features vs stored predictions.
    tolerance = float(protocol["gates"]["preflight_reconstruction"]["max_abs_probability_difference"])
    samples = dr.scope_samples(pairs, folds, "full")
    by_condition = {}
    for condition in aug.CONDITIONS:
        rows = dr.score_samples(
            condition=condition, samples=samples, visual=visual_by[condition].reshape(-1, 576),
            dsp=dsp_by[condition].reshape(-1, 16), folds=folds, frozen=loaded["frozen"],
        )
        stored = {(r["recipe"], r["sample_id"]): r for r in scope["rows"][condition]}
        comparison = dr.compare_to_reference(rows, stored, tolerance)
        if condition == "original":
            historical = dr.compare_to_reference(rows, loaded["reference"], tolerance)
            comparison["vs_4c3b"] = historical
            if historical["status"] != "PASS":
                comparison["status"] = "FAIL"
        by_condition[condition] = comparison
    reconstruction = {
        "status": "PASS" if all(c["status"] == "PASS" for c in by_condition.values()) else "FAIL",
        "tolerance": tolerance,
        "conditions": by_condition,
    }
    if reconstruction["status"] != "PASS":
        raise GateError(f"Preflight reconstruction of historical predictions failed: {json.dumps(reconstruction)[:2000]}")

    data_origin = (
        "development_real"
        if run_bindings["data_origin"] == "development_real" and late_bindings["data_origin"] == "development_real"
        else "synthetic_only"
    )
    bindings = {
        "schema_version": aug.SCHEMA_VERSION,
        "experiment_id": aug.EXPERIMENT_ID,
        "data_origin": data_origin,
        "protocol_sha256": clf.sha256_text_file(protocol_path),
        "source_membership_commitment": source_membership_commitment(source_ids),
        "visual_feature_commitment": clf.feature_array_commitment(np.stack([visual_by[c] for c in aug.CONDITIONS])),
        "dsp_feature_commitment": clf.feature_array_commitment(np.stack([dsp_by[c] for c in aug.CONDITIONS])),
        "fold_lock": public_fold_lock(folds),
        "late_fusion_run_manifest_sha256": robustness_protocol["frozen_models"]["late_fusion"]["run_manifest_sha256"],
        "robustness_run_bindings_sha256": bound["run_bindings_sha256"],
        "manifest_sha256": clf.sha256_file(manifest_path),
    }
    return {
        "protocol": protocol,
        "late_protocol": late_protocol,
        "source_ids": source_ids,
        "folds": folds,
        "loaded": loaded,
        "scope": scope,
        "visual_by": visual_by,
        "dsp_by": dsp_by,
        "reconstruction": reconstruction,
        "bindings": bindings,
        "data_origin": data_origin,
    }


def fold_gate(result: dict[str, Any], ctx: dict[str, Any], outer_fold: int) -> dict[str, Any]:
    """Refitted baseline must equal the stored 4C.3B fold and reproduce 4C.3B/4C.4B predictions."""
    tolerance = float(ctx["protocol"]["gates"]["baseline_refit_reconstruction"]["max_abs_probability_difference"])
    model_equal = json.loads(json.dumps(result["baseline"]["model"])) == ctx["loaded"]["late_models"][outer_fold]
    worst = 0.0
    mismatches = 0
    names = {"visual_calibrated": "visual_calibrated", "late_fusion_original": "late_fusion_stacked"}
    for condition in aug.CONDITIONS:
        stored = {(r["recipe"], r["sample_id"]): r for r in ctx["scope"]["rows"][condition]}
        if condition == "original":
            stored = {**stored, **ctx["loaded"]["reference"]}
        for row in result["rows"]:
            if row["condition"] != condition or row["recipe"] not in names:
                continue
            ref = stored[(names[row["recipe"]], row["sample_id"])]
            worst = max(worst, abs(float(ref["probability_ai_edited"]) - row["probability_ai_edited"]))
            mismatches += int(int(ref["prediction"]) != row["prediction"])
    status = "PASS" if model_equal and worst <= tolerance and mismatches == 0 else "FAIL"
    return {
        "status": status,
        "baseline_model_equal": model_equal,
        "max_abs_probability_difference": worst,
        "decision_mismatches": mismatches,
        "tolerance": tolerance,
    }


def run_pipeline(
    *,
    mode: str,
    protocol_path: Path,
    manifest_path: Path,
    data_root: Path,
    ablation_dir: Path,
    late_fusion_dir: Path,
    robustness_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if mode not in ("preflight", "pilot", "full"):
        raise ValueError(f"Unknown mode {mode!r}")
    output_dir = Path(output_dir)
    ctx = prepare(
        protocol_path=Path(protocol_path), manifest_path=Path(manifest_path), data_root=Path(data_root),
        ablation_dir=Path(ablation_dir), late_fusion_dir=Path(late_fusion_dir), robustness_dir=Path(robustness_dir),
        output_dir=output_dir,
    )
    protocol, plan = ctx["protocol"], aug.planned_budget(ctx["protocol"], ctx["late_protocol"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "preflight":
        receipt = {
            "schema_version": aug.SCHEMA_VERSION,
            "status": "PREFLIGHT_PASS",
            "data_origin": ctx["data_origin"],
            "sources": len(ctx["source_ids"]),
            "samples": 2 * len(ctx["source_ids"]),
            "conditions": list(aug.CONDITIONS),
            "recipes": list(aug.RECIPE_IDS),
            "planned_budget": plan,
            "checks": {"reconstruction": ctx["reconstruction"]},
            "bindings": ctx["bindings"],
            "fits": 0,
            "locked_test_accesses": 0,
            "created_at_utc": clf.utc_now(),
        }
        clf.atomic_write_json(output_dir / "preflight_receipt.json", receipt)
        print(f"[{clf.utc_now()}] PREFLIGHT_PASS ({ctx['data_origin']}); planned fits {plan['total_fits']}")
        return receipt

    bindings = ctx["bindings"]
    clf.ensure_run_manifest(output_dir, bindings)
    outer_folds = list(protocol["budget"]["pilot_outer_folds"]) if mode == "pilot" else list(range(len(ctx["folds"])))
    receipts, reused, fits_this_run = [], [], 0
    for k in outer_folds:
        existing = clf.verify_fold_artifacts(output_dir, k, bindings)
        if existing is not None:
            if existing.get("gate", {}).get("status") != "PASS":
                raise ValueError(f"Existing outer fold {k} has no PASS baseline gate; refusing to resume it")
            receipts.append(existing)
            reused.append(k)
            print(f"[{clf.utc_now()}] outer_fold={k} RESUMED (verified)")
            continue
        result = aug.run_augmented_outer_fold(
            outer_fold=k, fold=ctx["folds"][k], visual_by_condition=ctx["visual_by"], dsp_by_condition=ctx["dsp_by"],
            source_ids=ctx["source_ids"], protocol=protocol, late_protocol=ctx["late_protocol"],
        )
        gate = fold_gate(result, ctx, k)
        if gate["status"] != "PASS":
            raise GateError(f"Outer fold {k}: baseline reconstruction gate failed {gate}; augmented branch not persisted")
        directory = clf.fold_directory(output_dir, k)
        clf.atomic_write_text(directory / clf.PREDICTIONS_NAME, aug.render_rows_csv(result["rows"]))
        clf.atomic_write_json(directory / clf.MODEL_NAME, result["model"])
        receipt = {
            "schema_version": aug.SCHEMA_VERSION,
            "status": "COMPLETED",
            "bindings": {key: bindings[key] for key in (
                "protocol_sha256", "source_membership_commitment", "visual_feature_commitment",
                "dsp_feature_commitment", "data_origin",
            )},
            **result["receipt"],
            "gate": gate,
            "predictions_sha256": clf.sha256_file(directory / clf.PREDICTIONS_NAME),
            "model_sha256": clf.sha256_file(directory / clf.MODEL_NAME),
            "created_at_utc": clf.utc_now(),
        }
        clf.atomic_write_json(directory / clf.FOLD_RECEIPT_NAME, receipt)
        fits_this_run += receipt["fit_counts"]["total_fits"]
        receipts.append(receipt)
        counts = receipt["fit_counts"]
        print(
            f"[{clf.utc_now()}] outer_fold={k} COMPLETED gate=PASS fits={counts['total_fits']} "
            f"C(dsp_aug)={result['receipt']['augmented_dsp']['best_C']} unconverged={counts['unconverged_fits']}"
        )
    total = sum(r["fit_counts"]["total_fits"] for r in receipts)
    expected = plan["pilot_fits"] if mode == "pilot" else plan["total_fits"]
    summary = {
        "schema_version": aug.SCHEMA_VERSION,
        "mode": mode,
        "data_origin": ctx["data_origin"],
        "outer_folds": outer_folds,
        "reused_folds": reused,
        "fits_this_run": fits_this_run,
        "total_fits_all_folds": total,
        "planned_fits": expected,
        "budget_check": "PASS" if total == expected else "FAIL",
        "unconverged_fits": sum(r["fit_counts"]["unconverged_fits"] for r in receipts),
        "fit_wall_seconds": sum(r["fit_wall_seconds"] for r in receipts),
        "folds": receipts,
        "completed_at_utc": clf.utc_now(),
    }
    clf.atomic_write_json(output_dir / f"{mode}_summary.json", summary)
    if summary["budget_check"] != "PASS":
        raise RuntimeError(f"Fit count {total} differs from the locked budget {expected}")
    print(f"[{clf.utc_now()}] {mode.upper()} complete: {len(receipts)} fold(s), {total} fits ({fits_this_run} this run)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4C.6A controlled DSP-augmentation runner (development only)")
    parser.add_argument("--mode", choices=["preflight", "pilot", "full"], default="preflight")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "ml/configs/dsp_augmentation_protocol.yaml")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest_pilot_a_option_p.csv")
    parser.add_argument("--data-root", type=Path, required=True, help="directory the manifest paths are relative to (images are not read)")
    parser.add_argument("--ablation-dir", type=Path, required=True, help="visual/DSP ablation output dir (fits/ and shared/)")
    parser.add_argument("--late-fusion-dir", type=Path, required=True, help="Phase 4C.3B calibrated_late_fusion output dir")
    parser.add_argument("--robustness-dir", type=Path, required=True, help="Phase 4C.4B robustness runner output dir (contains full/)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new dir outside Git or under data/research/local-artifacts/")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_pipeline(
        mode=args.mode, protocol_path=args.protocol, manifest_path=args.manifest, data_root=args.data_root,
        ablation_dir=args.ablation_dir, late_fusion_dir=args.late_fusion_dir, robustness_dir=args.robustness_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
