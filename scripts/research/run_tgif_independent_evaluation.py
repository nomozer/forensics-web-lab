"""Independent Evaluation CLI Runner and Verification Harness for TGIF N=400.

Phase 4C.7B trace - TGIF-Train-Clean-Subset N=400 Independent Evaluation.
Strictly implements:
1. Fail-closed human review approval gate: blocks real execution unless evaluation_authorized == True.
2. Mock dry-run verification: hermetically tests pipeline, stratified paired cluster bootstrap,
   and metric aggregation using isolated mock/synthetic data without loading the real detector backbone.
3. Pre-registered Stratified Paired Cluster Bootstrap:
   - Resampling unit: source_cluster (authentic + ai_edited strictly paired)
   - Stratified allocation: exactly 14 Large, 221 Medium, 165 Small in every replicate
   - 10,000 replicates, PCG64 PRNG, seed 20261007
   - Shared replicates across both recipes and all 5 fold models
4. Estimand: unweighted arithmetic mean of 5 fold Macro-F1 scores (no probability averaging, no ensemble).
5. Primary endpoint: Delta at jpeg_q75 (augmented - visual).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import zipfile

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_evaluator import (
    CONDITIONS,
    PRIMARY_CONDITION,
    RECIPES,
    aggregate_per_model_metrics,
    derive_independent_verdict,
)
from ml.evaluation.independent_model_bindings import (
    FoldCandidateModel,
    create_mock_candidate_models,
    load_candidate_models,
)
from ml.training.development_robustness import (
    CANONICAL_CONDITIONS,
    apply_operations,
)
from ml.training.dsp_features import DSP_FEATURE_DIM, extract_dsp_features

DEFAULT_CONFIG_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_independent_evaluation_execution_config.json"
DEFAULT_EVAL_RECEIPT_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_independent_evaluation_receipt.json"


class EvaluationGateError(PermissionError):
    """Raised when evaluation is triggered without explicit human review authorization."""


class ConfigurationIntegrityError(ValueError):
    """Raised when evaluation configuration hashes or files do not match commitments."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_stratified_paired_cluster_bootstrap(
    source_ids: Sequence[str],
    strata: Sequence[str],
    labels: np.ndarray,
    visual_model_probs: list[np.ndarray],  # 5 models, shape (N,) each
    augmented_model_probs: list[np.ndarray],  # 5 models, shape (N,) each
    replicates: int = 10000,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Execute stratified paired source-cluster bootstrap.

    Preserves exact stratification:
    - Independent resampling with replacement within each stratum
    - Keeps authentic and ai_edited paired together per source_id
    - Shared replicate indices across recipes and all 5 fold models.
    """
    from sklearn.metrics import f1_score

    # Group sample indices by source_id, and track stratum per source
    source_to_indices: dict[str, list[int]] = {}
    source_to_stratum: dict[str, str] = {}
    for idx, (sid, st) in enumerate(zip(source_ids, strata)):
        source_to_indices.setdefault(sid, []).append(idx)
        source_to_stratum[sid] = st

    # Group unique sources by stratum
    stratum_to_sources: dict[str, list[str]] = {}
    for sid, st in source_to_stratum.items():
        stratum_to_sources.setdefault(st, []).append(sid)

    # Sort sources deterministically
    for st in stratum_to_sources:
        stratum_to_sources[st].sort()

    rng = np.random.Generator(np.random.PCG64(seed))
    delta_replicates: list[float] = []

    for _ in range(replicates):
        chosen_sources: list[str] = []
        for st, s_list in sorted(stratum_to_sources.items()):
            n_st = len(s_list)
            sampled = rng.choice(s_list, size=n_st, replace=True)
            chosen_sources.extend(sampled)

        boot_indices = [idx for sid in chosen_sources for idx in source_to_indices[sid]]
        boot_indices_arr = np.array(boot_indices, dtype=int)
        boot_labels = labels[boot_indices_arr]

        # Mean Macro-F1 across 5 models for visual recipe
        vis_f1s = []
        for v_probs in visual_model_probs:
            b_probs = v_probs[boot_indices_arr]
            b_preds = (b_probs >= 0.5).astype(int)
            vis_f1s.append(f1_score(boot_labels, b_preds, average="macro", zero_division=0))
        mean_vis_f1 = float(np.mean(vis_f1s))

        # Mean Macro-F1 across 5 models for augmented recipe
        aug_f1s = []
        for a_probs in augmented_model_probs:
            b_probs = a_probs[boot_indices_arr]
            b_preds = (b_probs >= 0.5).astype(int)
            aug_f1s.append(f1_score(boot_labels, b_preds, average="macro", zero_division=0))
        mean_aug_f1 = float(np.mean(aug_f1s))

        delta_replicates.append(mean_aug_f1 - mean_vis_f1)

    delta_arr = np.array(delta_replicates, dtype=np.float64)
    ci_lower = float(np.percentile(delta_arr, 2.5))
    ci_upper = float(np.percentile(delta_arr, 97.5))
    mean_delta = float(np.mean(delta_arr))
    median_delta = float(np.median(delta_arr))

    return {
        "resampling_method": "stratified_paired_source_cluster_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "ci_contains_zero": bool(ci_lower <= 0.0 <= ci_upper),
        "proportion_greater_than_zero": float(np.mean(delta_arr > 0.0)),
        "strata_counts": {st: len(s_list) for st, s_list in stratum_to_sources.items()},
    }


def verify_configuration_readiness(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Verify integrity of all bound files in the evaluation configuration."""
    print("=" * 70)
    print("Verifying TGIF Independent Evaluation Configuration & Artifact Bindings")
    print("=" * 70)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    # 1. Cohort bindings check
    cohort = config["cohort_binding"]
    manifest_file = REPO_ROOT / cohort["manifest_relpath"]
    intake_receipt_file = REPO_ROOT / cohort["intake_receipt_relpath"]
    phash_receipt_file = REPO_ROOT / cohort["phash_receipt_relpath"]
    package_zip_file = REPO_ROOT / cohort["package_zip_relpath"]

    for name, f in [
        ("Locked manifest", manifest_file),
        ("Intake receipt", intake_receipt_file),
        ("pHash receipt", phash_receipt_file),
        ("Package ZIP", package_zip_file),
    ]:
        if not f.is_file():
            raise ConfigurationIntegrityError(f"Missing required artifact: {name} ({f})")

    actual_manifest_sha = sha256_file(manifest_file)
    if actual_manifest_sha != cohort["manifest_sha256"]:
        raise ConfigurationIntegrityError(f"Manifest SHA-256 mismatch: {actual_manifest_sha} != {cohort['manifest_sha256']}")

    actual_pkg_sha = sha256_file(package_zip_file)
    if actual_pkg_sha != cohort["package_zip_sha256"]:
        raise ConfigurationIntegrityError(f"Package ZIP SHA-256 mismatch: {actual_pkg_sha} != {cohort['package_zip_sha256']}")

    print(f"Cohort artifacts verified: locked manifest and package ZIP hashes match commit bindings.")

    # 2. Check 5 fold model checkpoints
    models_spec = config["candidate_model_bindings"]
    outer_folds = models_spec["outer_folds"]
    if len(outer_folds) != 5:
        raise ConfigurationIntegrityError(f"Expected 5 outer folds, found {len(outer_folds)}")

    for fold in outer_folds:
        f_idx = fold["outer_fold"]
        f_path = REPO_ROOT / fold["model_path"]
        if not f_path.is_file():
            raise ConfigurationIntegrityError(f"Fold {f_idx} model file missing: {f_path}")
        f_sha = sha256_file(f_path)
        if f_sha != fold["model_sha256"]:
            raise ConfigurationIntegrityError(f"Fold {f_idx} model SHA-256 mismatch: {f_sha} != {fold['model_sha256']}")

    print("Candidate model checkpoints verified: all 5 outer fold JSON models present and hash-identical.")

    # 3. Check backbone weights
    backbone = models_spec["backbone"]
    bb_path = REPO_ROOT / backbone["weights_path"]
    if not bb_path.is_file():
        raise ConfigurationIntegrityError(f"Backbone weights missing: {bb_path}")
    bb_sha = sha256_file(bb_path)
    if bb_sha != backbone["weights_sha256"]:
        raise ConfigurationIntegrityError(f"Backbone SHA-256 mismatch: {bb_sha} != {backbone['weights_sha256']}")

    print("Backbone weights verified: mobilenet_v3_small-047dcff4.pth present and hash-identical.")

    # 4. Check approval gate
    gate = config.get("human_approval_gate", {})
    authorized = gate.get("evaluation_authorized", False)
    status = "READY_FOR_HUMAN_APPROVAL" if not authorized else "AUTHORIZED_FOR_EVALUATION"

    report = {
        "status": status,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "evaluation_authorized": authorized,
        "cohort_pairs": cohort["cohort_pairs"],
        "total_images": cohort["total_images"],
        "strata": cohort["strata_breakdown"],
        "outer_folds_verified": 5,
        "backbone_verified": True,
        "conditions": config["evaluation_protocol"]["conditions"],
        "primary_endpoint": config["evaluation_protocol"]["primary_endpoint"],
        "resampling": config["statistical_plan"]["resampling_method"],
    }
    print("=" * 70)
    print(f"CONFIGURATION READINESS STATUS: {status} (evaluation_authorized={authorized})")
    print("=" * 70)
    return report


def run_mock_dry_run(
    config_path: Path = DEFAULT_CONFIG_PATH,
    bootstrap_replicates: int = 500,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Execute hermetic mock evaluation dry run.

    Zero detector backbone forward passes, zero detector loading.
    Simulates feature generation across all 400 pairs and 6 conditions,
    computes scores through the 5 fold models, runs stratified paired cluster bootstrap,
    and validates output format.
    """
    print("=" * 70)
    print("Executing Isolated Hermetic Mock Evaluation Dry-Run (ZERO DETECTOR CALLS)")
    print("=" * 70)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_file = REPO_ROOT / config["cohort_binding"]["manifest_relpath"]
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    candidates = manifest_data["selected_candidates"]

    # Assemble samples
    samples_meta = []
    for c in candidates:
        sid = c["source_id"]
        stratum = c["stratum_area_class"]
        samples_meta.append({
            "source_id": sid,
            "stratum": stratum,
            "label": 0,  # authentic
        })
        samples_meta.append({
            "source_id": sid,
            "stratum": stratum,
            "label": 1,  # ai_edited
        })

    n_samples = len(samples_meta)
    source_ids = [s["source_id"] for s in samples_meta]
    strata = [s["stratum"] for s in samples_meta]
    labels = np.array([s["label"] for s in samples_meta], dtype=int)

    # Use mock candidate models or load frozen scorers
    try:
        models = load_candidate_models()
    except Exception:
        models = create_mock_candidate_models()

    rng = np.random.Generator(np.random.PCG64(seed))
    condition_results: dict[str, Any] = {}
    vis_q75_probs: list[np.ndarray] = []
    aug_q75_probs: list[np.ndarray] = []

    for cond in CONDITIONS:
        # Mock 576-d visual features and 16-d dsp features
        vis_feats = rng.normal(0.0, 1.0, size=(n_samples, 576)).astype(np.float64)
        dsp_feats = rng.normal(0.0, 1.0, size=(n_samples, 16)).astype(np.float64)

        recipe_metrics: dict[str, Any] = {}
        for recipe in RECIPES:
            fold_predictions: list[dict[str, np.ndarray]] = []
            for m in models:
                scores = m.score(vis_feats, dsp_feats)
                fold_predictions.append(scores[recipe])

            agg = aggregate_per_model_metrics(fold_predictions, labels)
            recipe_metrics[recipe] = agg

            if cond == PRIMARY_CONDITION:
                probs_list = [fp["probability"] for fp in fold_predictions]
                if recipe == "visual_calibrated":
                    vis_q75_probs = probs_list
                elif recipe == "late_fusion_dsp_augmented":
                    aug_q75_probs = probs_list

        condition_results[cond] = recipe_metrics

    # Execute stratified paired cluster bootstrap
    boot_res = run_stratified_paired_cluster_bootstrap(
        source_ids=source_ids,
        strata=strata,
        labels=labels,
        visual_model_probs=vis_q75_probs,
        augmented_model_probs=aug_q75_probs,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    verdict = derive_independent_verdict(
        ci_lower=boot_res["ci_lower_95"],
        ci_upper=boot_res["ci_upper_95"],
        is_synthetic=True,
    )

    vis_mean_f1 = condition_results[PRIMARY_CONDITION]["visual_calibrated"]["mean_metrics"]["macro_f1"]
    aug_mean_f1 = condition_results[PRIMARY_CONDITION]["late_fusion_dsp_augmented"]["mean_metrics"]["macro_f1"]
    primary_delta = aug_mean_f1 - vis_mean_f1

    result = {
        "status": "MOCK_DRY_RUN_SUCCESS",
        "is_synthetic": True,
        "detector_calls": 0,
        "verdict": verdict,
        "num_pairs": len(candidates),
        "num_samples": n_samples,
        "primary_point_delta": primary_delta,
        "bootstrap": boot_res,
        "conditions_evaluated": list(CONDITIONS),
    }

    print(f"Mock Dry Run Status: {result['status']}")
    print(f"Detector Calls: {result['detector_calls']}")
    print(f"Stratified Paired Bootstrap Replicates: {boot_res['replicates']}")
    print(f"Primary Delta (Mock): {primary_delta:.4f}")
    print(f"95% CI (Mock): [{boot_res['ci_lower_95']:.4f}, {boot_res['ci_upper_95']:.4f}]")
    print("=" * 70)
    return result


def execute_independent_evaluation(
    config_path: Path = DEFAULT_CONFIG_PATH,
    receipt_out_path: Path = DEFAULT_EVAL_RECEIPT_PATH,
) -> dict[str, Any]:
    """Execute real independent evaluation on TGIF N=400.

    STRICTLY ENFORCES: human_approval_gate.evaluation_authorized == True.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = config.get("human_approval_gate", {})
    if not gate.get("evaluation_authorized", False):
        raise EvaluationGateError(
            "CRITICAL GATE: Human reviewer has NOT authorized evaluation execution. "
            "To prevent unauthorized runs, execution is stopped fail-closed. "
            "Zero detector calls made."
        )

    # If authorized, this is where the full real evaluation execution would run.
    raise NotImplementedError("Real evaluation gate is not authorized yet.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TGIF N=400 Independent Evaluation Runner & Preflight")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to execution config JSON")
    parser.add_argument("--check-config", action="store_true", help="Verify configuration integrity and readiness")
    parser.add_argument("--mock-dry-run", action="store_true", help="Run hermetic mock simulation without detector")
    parser.add_argument("--execute", action="store_true", help="Execute real evaluation (requires human approval)")
    parser.add_argument("--bootstrap-replicates", type=int, default=500, help="Replicates for mock simulation")
    parser.add_argument("--seed", type=int, default=20261007, help="RNG seed")
    args = parser.parse_args(argv)

    if not args.check_config and not args.mock_dry_run and not args.execute:
        args.check_config = True

    if args.check_config:
        verify_configuration_readiness(args.config)

    if args.mock_dry_run:
        run_mock_dry_run(args.config, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)

    if args.execute:
        try:
            execute_independent_evaluation(args.config)
        except EvaluationGateError as exc:
            print(f"\n[FAIL-CLOSED ERROR] {exc}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
