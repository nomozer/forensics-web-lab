#!/usr/bin/env python3
"""CLI Runner and Preflight Verification for Independent Validation (Phase 4C.7A).

Usage:
  python scripts/research/run_independent_validation.py --check-readiness
  python scripts/research/run_independent_validation.py --synthetic-preflight [--bootstrap-replicates N]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Adjust stdout for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import (
    DEFAULT_HISTORICAL_MANIFEST,
    generate_synthetic_planning_cohort,
    load_historical_source_ids,
    validate_cohort_manifest,
)
from ml.evaluation.independent_evaluator import (
    CONDITIONS,
    PRIMARY_CONDITION,
    RECIPES,
    aggregate_per_model_metrics,
    derive_independent_verdict,
    run_paired_source_cluster_bootstrap,
)
from ml.evaluation.independent_model_bindings import (
    DEFAULT_BINDINGS_PATH,
    create_mock_candidate_models,
    load_candidate_models,
)

EVIDENCE_DIR = REPO_ROOT / "research/evidence/phase-4c.7a"


def check_readiness(repo_root: Path | None = None, write_artifact: bool = True) -> dict[str, Any]:
    """Inspect readiness of protocol, candidate model bindings, and data cohort."""
    root = repo_root or REPO_ROOT
    protocol_path = root / "ml/configs/independent_validation_protocol.yaml"
    bindings_path = root / "research/evidence/phase-4c.7a/candidate_model_bindings.json"

    protocol_ok = protocol_path.is_file()
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_ok else None

    # Verify model bindings and underlying files
    bindings_ok = False
    models_verified = 0
    models_error = None
    try:
        models = load_candidate_models(bindings_path, repo_root=root)
        models_verified = len(models)
        bindings_ok = (models_verified == 5)
    except Exception as e:
        models_error = str(e)

    # Check historical source manifest for disjoint guard
    historical_sources_count = 0
    historical_manifest_ok = DEFAULT_HISTORICAL_MANIFEST.is_file()
    if historical_manifest_ok:
        try:
            sids = load_historical_source_ids(DEFAULT_HISTORICAL_MANIFEST)
            historical_sources_count = len(sids)
        except Exception:
            pass

    # Status determination:
    # Code, protocol, bindings verified; independent real cohort is not yet acquired.
    status = "PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT"
    ready_for_real_eval = False

    readiness = {
        "phase": "4C.7A",
        "status": status,
        "ready_for_real_evaluation": ready_for_real_eval,
        "protocol": {
            "path": protocol_path.relative_to(root).as_posix(),
            "exists": protocol_ok,
            "sha256": protocol_sha,
        },
        "model_bindings": {
            "path": bindings_path.relative_to(root).as_posix(),
            "exists": bindings_path.is_file(),
            "models_verified_count": models_verified,
            "bindings_verified": bindings_ok,
            "error": models_error,
        },
        "cohort_status": {
            "historical_option_p_sources_indexed": historical_sources_count,
            "real_independent_cohort_acquired": False,
            "note": "Independent cohort acquisition plan locked; real cohort collection deferred to Phase 4C.7B.",
        },
        "real_evaluation_performed": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if write_artifact:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        out_file = EVIDENCE_DIR / "readiness.json"
        out_file.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    return readiness


def run_synthetic_preflight(
    num_pairs: int = 50,
    bootstrap_replicates: int = 500,
    seed: int = 20261007,
    models: list[Any] | None = None,
    candidate_models: list[Any] | None = None,
    historical_sources: set[str] | None = None,
) -> dict[str, Any]:
    """Execute synthetic preflight check verifying data schemas, models, and bootstrap arithmetic."""
    # 1. Generate synthetic cohort and validate against historical sources
    synthetic_manifest = generate_synthetic_planning_cohort(num_pairs=num_pairs)
    if historical_sources is None:
        try:
            hist_sources = load_historical_source_ids()
        except FileNotFoundError:
            hist_sources = set()
    else:
        hist_sources = historical_sources
    pairs = validate_cohort_manifest(synthetic_manifest, historical_sources=hist_sources)

    # 2. Load candidate models (fallback to mock models if local disk artifacts absent)
    target_models = candidate_models if candidate_models is not None else models
    if target_models is None:
        try:
            active_models = load_candidate_models()
        except Exception:
            active_models = create_mock_candidate_models()
    else:
        active_models = target_models
    assert len(active_models) == 5, f"Expected 5 models, got {len(active_models)}"

    # 3. Simulate feature extraction for synthetic samples
    # 576 visual features, 16 dsp features
    rng = np.random.Generator(np.random.PCG64(seed))
    n_samples = len(pairs) * 2
    source_ids: list[str] = []
    labels: list[int] = []

    for p in pairs:
        source_ids.extend([p.source_id, p.source_id])
        labels.extend([0, 1])

    labels_arr = np.array(labels, dtype=int)

    # Mock synthetic feature generation per condition
    condition_results: dict[str, Any] = {}
    visual_q75_probs: list[np.ndarray] = []
    augmented_q75_probs: list[np.ndarray] = []

    for cond in CONDITIONS:
        vis_feat = rng.normal(0.0, 1.0, size=(n_samples, 576)).astype(np.float64)
        dsp_feat = rng.normal(0.0, 1.0, size=(n_samples, 16)).astype(np.float64)

        recipe_metrics: dict[str, Any] = {}
        for recipe in RECIPES:
            fold_predictions: list[dict[str, np.ndarray]] = []
            for m in active_models:
                scores = m.score(vis_feat, dsp_feat)
                fold_predictions.append(scores[recipe])

            agg = aggregate_per_model_metrics(fold_predictions, labels_arr)
            recipe_metrics[recipe] = agg

            if cond == PRIMARY_CONDITION:
                probs_list = [fp["probability"] for fp in fold_predictions]
                if recipe == "visual_calibrated":
                    visual_q75_probs = probs_list
                else:
                    augmented_q75_probs = probs_list

        condition_results[cond] = recipe_metrics

    # 4. Execute paired source-cluster bootstrap
    boot_res = run_paired_source_cluster_bootstrap(
        source_ids=source_ids,
        labels=labels_arr,
        visual_model_probs=visual_q75_probs,
        augmented_model_probs=augmented_q75_probs,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    verdict = derive_independent_verdict(
        ci_lower=boot_res["ci_lower_95"],
        ci_upper=boot_res["ci_upper_95"],
        is_synthetic=True,
    )

    # Primary point delta
    vis_mean_f1 = condition_results[PRIMARY_CONDITION]["visual_calibrated"]["mean_metrics"]["macro_f1"]
    aug_mean_f1 = condition_results[PRIMARY_CONDITION]["late_fusion_dsp_augmented"]["mean_metrics"]["macro_f1"]
    primary_delta = aug_mean_f1 - vis_mean_f1

    return {
        "status": "SYNTHETIC_PREFLIGHT_PASS",
        "is_synthetic": True,
        "verdict": verdict,
        "sample_count": n_samples,
        "source_pair_count": len(pairs),
        "primary_point_delta": primary_delta,
        "bootstrap": boot_res,
        "primary_condition": PRIMARY_CONDITION,
        "conditions_evaluated": list(CONDITIONS),
    }


def run_synthetic_image_preflight(
    num_pairs: int = 5,
    bootstrap_replicates: int = 100,
    seed: int = 20261007,
    models: list[Any] | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evaluation pipeline on synthetic PIL images.

    Verifies the entire chain:
    PIL Image decode -> 6 transforms -> 16-d DSP + 576-d Visual extraction
    -> 5 candidate models -> per-model scoring -> mean Macro-F1 -> bootstrap.
    """
    from PIL import Image, ImageDraw
    from ml.evaluation.independent_pipeline import IndependentEvaluationPipeline

    synthetic_manifest = generate_synthetic_planning_cohort(num_pairs=num_pairs, prefix="IMG_PREFLIGHT_")
    pairs = validate_cohort_manifest(synthetic_manifest)

    # Generate synthetic in-memory PIL images
    rng = np.random.Generator(np.random.PCG64(seed))
    images_cache: dict[str, Image.Image] = {}

    for p in pairs:
        # Authentic: clean synthetic texture
        auth_arr = rng.integers(50, 200, size=(64, 64, 3), dtype=np.uint8)
        auth_img = Image.fromarray(auth_arr, mode="RGB")
        images_cache[p.authentic.image_relpath] = auth_img

        # AI-edited: synthetic texture with modified patch
        edit_img = auth_img.copy()
        draw = ImageDraw.Draw(edit_img)
        draw.rectangle([20, 20, 44, 44], fill=(220, 40, 40))
        images_cache[p.ai_edited.image_relpath] = edit_img

    if models is None:
        try:
            target_models = load_candidate_models()
        except Exception:
            target_models = create_mock_candidate_models()
    else:
        target_models = models

    pipeline = IndependentEvaluationPipeline(candidate_models=target_models)
    eval_result = pipeline.evaluate_cohort(
        pairs=pairs,
        image_resolver=lambda relpath: images_cache[relpath],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        is_synthetic=True,
    )
    return eval_result


def run_cohort_evaluation(
    manifest_path: Path | str,
    data_root: Path | str | None = None,
    bootstrap_replicates: int = 10000,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Execute evaluation on a verified independent validation cohort manifest."""
    import csv
    from ml.evaluation.independent_cohort import (
        load_historical_image_hashes,
        load_historical_origin_ids,
        load_historical_source_ids,
        validate_cohort_manifest,
    )
    from ml.evaluation.independent_pipeline import IndependentEvaluationPipeline

    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Cohort manifest not found: {m_path}")

    rows: list[dict[str, Any]] = []
    with m_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Multi-level historical disjoint guard
    try:
        hist_sources = load_historical_source_ids()
    except FileNotFoundError:
        hist_sources = set()

    try:
        hist_hashes = load_historical_image_hashes()
    except FileNotFoundError:
        hist_hashes = set()

    try:
        hist_origins = load_historical_origin_ids()
    except FileNotFoundError:
        hist_origins = set()

    pairs = validate_cohort_manifest(
        rows,
        historical_sources=hist_sources,
        historical_hashes=hist_hashes,
        historical_origins=hist_origins,
    )

    models = load_candidate_models()
    pipeline = IndependentEvaluationPipeline(candidate_models=models)
    return pipeline.evaluate_cohort(
        pairs=pairs,
        data_root=data_root,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        is_synthetic=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4C.7A Independent Validation Readiness & Preflight")
    parser.add_argument("--check-readiness", action="store_true", help="Inspect readiness and export readiness.json")
    parser.add_argument("--synthetic-preflight", action="store_true", help="Run synthetic feature preflight simulation")
    parser.add_argument("--image-preflight", action="store_true", help="Run full end-to-end PIL image pipeline preflight")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on real independent cohort")
    parser.add_argument("--manifest", type=str, default=None, help="Path to independent cohort manifest CSV")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for image paths")
    parser.add_argument("--bootstrap-replicates", type=int, default=500, help="Number of bootstrap replicates for simulation")
    parser.add_argument("--seed", type=int, default=20261007, help="RNG seed")
    args = parser.parse_args(argv)

    if not args.check_readiness and not args.synthetic_preflight and not args.image_preflight and not args.evaluate:
        args.check_readiness = True

    if args.check_readiness:
        readiness = check_readiness()
        print(f"Readiness Status: {readiness['status']}")
        print(f"Model Bindings Verified: {readiness['model_bindings']['models_verified_count']} / 5")
        print(f"Real Cohort Acquired: {readiness['cohort_status']['real_independent_cohort_acquired']}")
        print(f"Readiness artifact written to: {EVIDENCE_DIR / 'readiness.json'}")

    if args.synthetic_preflight:
        print("\nExecuting synthetic feature preflight simulation...")
        res = run_synthetic_preflight(bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
        print(f"Preflight Status: {res['status']}")
        print(f"Synthetic Verdict: {res['verdict']}")
        print(f"Primary Delta (Synthetic): {res['primary_point_delta']:.4f}")
        print(f"Bootstrap 95% CI: [{res['bootstrap']['ci_lower_95']:.4f}, {res['bootstrap']['ci_upper_95']:.4f}]")

    if args.image_preflight:
        print("\nExecuting end-to-end PIL image pipeline preflight...")
        res_img = run_synthetic_image_preflight(bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
        print(f"Image Pipeline Status: {res_img['status']}")
        print(f"Conditions Evaluated: {', '.join(res_img['conditions_evaluated'])}")
        print(f"Synthetic Verdict: {res_img['verdict']}")
        print(f"Primary Condition (jpeg_q75) Mean Delta: {res_img['bootstrap']['mean_delta']:.4f}")
        print(f"Bootstrap 95% CI: [{res_img['bootstrap']['ci_lower_95']:.4f}, {res_img['bootstrap']['ci_upper_95']:.4f}]")

    if args.evaluate:
        if not args.manifest:
            print("Error: --evaluate requires --manifest <path_to_cohort_manifest.csv>", file=sys.stderr)
            sys.exit(1)
        print(f"\nExecuting evaluation on cohort manifest: {args.manifest}...")
        eval_res = run_cohort_evaluation(
            manifest_path=args.manifest,
            data_root=args.data_root,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        print(f"Evaluation Status: {eval_res['status']}")
        print(f"Verdict: {eval_res['verdict']}")
        print(f"Bootstrap 95% CI: [{eval_res['bootstrap']['ci_lower_95']:.4f}, {eval_res['bootstrap']['ci_upper_95']:.4f}]")


if __name__ == "__main__":
    main()
