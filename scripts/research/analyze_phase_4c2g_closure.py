#!/usr/bin/env python3
"""Read-only Phase 4C.2G closure and exploratory error analysis.

This program never loads a checkpoint, model, image, or manifest. It consumes only
the already-published evaluator outputs. Confirmatory quantities are reconciled;
all confusion/probability/source-error analyses are explicitly post-hoc and
exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.confirmatory_metrics import compute_confirmatory_descriptive_metrics


SEEDS = [42, 1337, 2025, 3407, 9001]
EXPLORATORY_CLASSIFICATION = (
    "EXPLORATORY_POST_HOC_LOCKED_TEST_ARTIFACT_ANALYSIS_NOT_CONFIRMATORY"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def linear_quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "population_sd": statistics.pstdev(materialized),
        "min": min(materialized),
        "q05": linear_quantile(materialized, 0.05),
        "q25": linear_quantile(materialized, 0.25),
        "median": linear_quantile(materialized, 0.50),
        "q75": linear_quantile(materialized, 0.75),
        "q95": linear_quantile(materialized, 0.95),
        "max": max(materialized),
    }


def probability_class_one(logits: list[float]) -> float:
    if len(logits) != 2 or not all(math.isfinite(float(value)) for value in logits):
        raise ValueError("Each prediction must contain two finite logits")
    delta = float(logits[0]) - float(logits[1])
    if delta >= 0:
        exp_neg = math.exp(-delta)
        return exp_neg / (1.0 + exp_neg)
    exp_pos = math.exp(delta)
    return 1.0 / (1.0 + exp_pos)


def prediction_class(logits: list[float]) -> int:
    return 0 if float(logits[0]) >= float(logits[1]) else 1


def verify_checksum_index(outputs_dir: Path, index_name: str) -> dict[str, Any]:
    index_path = outputs_dir / index_name
    index = read_json(index_path)
    failures: list[dict[str, Any]] = []
    for relative_path, expected in index["files"].items():
        target = outputs_dir / relative_path
        if not target.is_file():
            failures.append({"relative_path": relative_path, "reason": "missing"})
            continue
        actual_bytes = target.stat().st_size
        actual_sha256 = sha256_file(target)
        if actual_bytes != expected["bytes"] or actual_sha256 != expected["sha256"]:
            failures.append(
                {
                    "relative_path": relative_path,
                    "reason": "size_or_sha256_mismatch",
                    "actual_bytes": actual_bytes,
                    "actual_sha256": actual_sha256,
                }
            )
    if failures:
        raise ValueError(f"Checksum verification failed for {index_name}: {failures}")
    return {"index": index_name, "entries_verified": len(index["files"]), "status": "PASS"}


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.name)
    }


def assert_close(label: str, actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def analyze(
    *,
    outputs_dir: Path,
    wrapper_receipt_path: Path,
    output_dir: Path,
    expected_sources: int = 343,
    expected_samples: int = 686,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs_dir = outputs_dir.resolve(strict=True)
    wrapper_receipt_path = wrapper_receipt_path.resolve(strict=True)
    if output_dir.resolve() == outputs_dir:
        raise ValueError("Analysis output must not overwrite evaluator outputs")

    source_files = [path for path in outputs_dir.iterdir() if path.is_file()]
    before = snapshot_files(source_files + [wrapper_receipt_path])
    checksum_audit = [
        verify_checksum_index(outputs_dir, "checksums.json"),
        verify_checksum_index(outputs_dir, "session_checksums.json"),
    ]

    per_checkpoint_artifact = read_json(outputs_dir / "per_checkpoint_metrics.json")
    aggregate_artifact = read_json(outputs_dir / "aggregate_confirmatory_metrics.json")
    bootstrap_artifact = read_json(outputs_dir / "bootstrap_distribution_summary.json")
    decision_artifact = read_json(outputs_dir / "confirmatory_decision.json")
    execution_receipt = read_json(outputs_dir / "execution_receipt.json")
    wrapper_receipt = read_json(wrapper_receipt_path)

    canonical_identity: list[tuple[str, str, str, int, int]] | None = None
    source_labels: dict[str, set[int]] = defaultdict(set)
    all_probability_rows: list[dict[str, Any]] = []
    recomputed_metrics: dict[str, Any] = {}

    for seed in SEEDS:
        prediction_name = f"predictions_seed_{seed}.json"
        payload = read_json(outputs_dir / prediction_name)
        if payload.get("seed") != seed:
            raise ValueError(f"Prediction seed mismatch in {prediction_name}")
        records = list(payload.get("predictions", []))
        if len(records) != expected_samples:
            raise ValueError(f"{prediction_name} has {len(records)} records, expected {expected_samples}")
        records.sort(key=lambda row: int(row["sample_idx"]))
        identities = [
            (
                str(row["source_id"]),
                str(row["sample_id"]),
                str(row["relative_path"]),
                int(row["true_label"]),
                int(row["sample_idx"]),
            )
            for row in records
        ]
        if canonical_identity is None:
            canonical_identity = identities
        elif identities != canonical_identity:
            raise ValueError(f"Sample identity/order mismatch for seed {seed}")

        labels = np.asarray([int(row["true_label"]) for row in records], dtype=np.int64)
        logits = np.asarray([row["logits"] for row in records], dtype=np.float64)
        metrics = compute_confirmatory_descriptive_metrics(labels, logits)
        expected_metrics = per_checkpoint_artifact[str(seed)]["metrics"]
        for metric_name in (
            "macro_f1",
            "balanced_accuracy",
            "auroc",
            "brier_score",
            "ece",
            "citl",
            "signed_confidence_calibration_gap",
        ):
            assert_close(
                f"seed {seed} {metric_name}",
                metrics[metric_name],
                expected_metrics[metric_name],
            )
        if metrics["confusion_matrix"] != expected_metrics["confusion_matrix"]:
            raise ValueError(f"seed {seed} confusion matrix mismatch")

        recomputed_metrics[str(seed)] = metrics
        for row in records:
            label = int(row["true_label"])
            source_id = str(row["source_id"])
            source_labels[source_id].add(label)
            probability = probability_class_one(row["logits"])
            predicted = prediction_class(row["logits"])
            all_probability_rows.append(
                {
                    "seed": seed,
                    "source_id": source_id,
                    "sample_idx": int(row["sample_idx"]),
                    "true_label": label,
                    "predicted_label": predicted,
                    "probability_ai_edited": probability,
                    "correct": predicted == label,
                }
            )

    if canonical_identity is None:
        raise ValueError("No predictions were loaded")
    if len(source_labels) != expected_sources:
        raise ValueError(f"Found {len(source_labels)} sources, expected {expected_sources}")
    invalid_clusters = [source for source, labels in source_labels.items() if labels != {0, 1}]
    if invalid_clusters:
        raise ValueError("Every source must contain one authentic and one edited sample")

    macro_values = [recomputed_metrics[str(seed)]["macro_f1"] for seed in SEEDS]
    aggregate_mean = statistics.fmean(macro_values)
    assert_close("aggregate Macro-F1", aggregate_mean, aggregate_artifact["aggregate_macro_f1"])
    for field in (
        "bootstrap_replicates",
        "bootstrap_rng_seed",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_distribution_sha256_float64_le",
    ):
        if aggregate_artifact[field] != bootstrap_artifact[field]:
            raise ValueError(f"Aggregate/bootstrap cross-file mismatch for {field}")
    decision_expected = aggregate_artifact["ci_lower_95"] > 0.5
    if decision_artifact["success"] != decision_expected:
        raise ValueError("Decision success does not match the preregistered rule")
    if decision_artifact["verdict"] != aggregate_artifact["verdict"]:
        raise ValueError("Decision verdict does not match aggregate metrics")
    if execution_receipt.get("status") != "COMPLETED_VALID":
        raise ValueError("Execution receipt is not COMPLETED_VALID")

    wrapper_false_negative = (
        wrapper_receipt.get("controller_exit_code") == 0
        and wrapper_receipt.get("controller_completed") is True
        and wrapper_receipt.get("outputs_exist") is True
        and wrapper_receipt.get("confirmatory_result_exists") is False
        and (outputs_dir / "aggregate_confirmatory_metrics.json").is_file()
        and (outputs_dir / "confirmatory_decision.json").is_file()
    )
    if not wrapper_false_negative:
        raise ValueError("Wrapper false-negative contract could not be reconciled")

    pooled_confusion = Counter()
    per_seed_confusion = {}
    pair_outcomes_by_seed: dict[str, Counter[str]] = {}
    probability_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        matrix = recomputed_metrics[str(seed)]["confusion_matrix"]
        per_seed_confusion[str(seed)] = {
            "tn": matrix["tn"],
            "fp": matrix["fp"],
            "fn": matrix["fn"],
            "tp": matrix["tp"],
            "authentic_false_positive_rate": matrix["fp"] / (matrix["tn"] + matrix["fp"]),
            "edited_false_negative_rate": matrix["fn"] / (matrix["tp"] + matrix["fn"]),
        }
        pooled_confusion.update({key: int(matrix[key]) for key in ("tn", "fp", "fn", "tp")})
        rows = [row for row in all_probability_rows if row["seed"] == seed]
        probability_by_seed[str(seed)] = {
            "authentic_probability_ai_edited": distribution(
                row["probability_ai_edited"] for row in rows if row["true_label"] == 0
            ),
            "edited_probability_ai_edited": distribution(
                row["probability_ai_edited"] for row in rows if row["true_label"] == 1
            ),
            "separation_in_mean_probability": (
                statistics.fmean(row["probability_ai_edited"] for row in rows if row["true_label"] == 1)
                - statistics.fmean(row["probability_ai_edited"] for row in rows if row["true_label"] == 0)
            ),
        }
        by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_source[row["source_id"]][row["true_label"]] = row
        outcomes: Counter[str] = Counter()
        for pair in by_source.values():
            authentic_error = not pair[0]["correct"]
            edited_error = not pair[1]["correct"]
            if not authentic_error and not edited_error:
                outcomes["both_correct"] += 1
            elif authentic_error and not edited_error:
                outcomes["authentic_only_error"] += 1
            elif not authentic_error and edited_error:
                outcomes["edited_only_error"] += 1
            else:
                outcomes["both_error"] += 1
        pair_outcomes_by_seed[str(seed)] = outcomes

    probability_pooled = {
        "interpretation": "Pooled rows repeat each fixed sample across five checkpoints and are not independent.",
        "authentic_probability_ai_edited": distribution(
            row["probability_ai_edited"] for row in all_probability_rows if row["true_label"] == 0
        ),
        "edited_probability_ai_edited": distribution(
            row["probability_ai_edited"] for row in all_probability_rows if row["true_label"] == 1
        ),
        "correct_prediction_confidence": distribution(
            max(row["probability_ai_edited"], 1.0 - row["probability_ai_edited"])
            for row in all_probability_rows
            if row["correct"]
        ),
        "incorrect_prediction_confidence": distribution(
            max(row["probability_ai_edited"], 1.0 - row["probability_ai_edited"])
            for row in all_probability_rows
            if not row["correct"]
        ),
        "probability_bands": dict(
            sorted(
                Counter(
                    "[0.00,0.40)"
                    if row["probability_ai_edited"] < 0.40
                    else "[0.40,0.45)"
                    if row["probability_ai_edited"] < 0.45
                    else "[0.45,0.50)"
                    if row["probability_ai_edited"] < 0.50
                    else "[0.50,0.55)"
                    if row["probability_ai_edited"] < 0.55
                    else "[0.55,0.60)"
                    if row["probability_ai_edited"] < 0.60
                    else "[0.60,1.00]"
                    for row in all_probability_rows
                ).items()
            )
        ),
    }

    observations_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_probability_rows:
        observations_by_source[row["source_id"]].append(row)
    source_rows: list[dict[str, Any]] = []
    vote_patterns = {"authentic": Counter(), "ai_edited": Counter()}
    for rows in observations_by_source.values():
        authentic_rows = [row for row in rows if row["true_label"] == 0]
        edited_rows = [row for row in rows if row["true_label"] == 1]
        authentic_errors = sum(not row["correct"] for row in authentic_rows)
        edited_errors = sum(not row["correct"] for row in edited_rows)
        vote_patterns["authentic"][str(authentic_errors)] += 1
        vote_patterns["ai_edited"][str(edited_errors)] += 1
        source_rows.append(
            {
                "error_count_out_of_10": authentic_errors + edited_errors,
                "authentic_errors_out_of_5": authentic_errors,
                "edited_errors_out_of_5": edited_errors,
                "prediction_flip_samples": int(0 < authentic_errors < 5) + int(0 < edited_errors < 5),
                "mean_probability_ai_edited_for_authentic": statistics.fmean(
                    row["probability_ai_edited"] for row in authentic_rows
                ),
                "mean_probability_ai_edited_for_edited": statistics.fmean(
                    row["probability_ai_edited"] for row in edited_rows
                ),
            }
        )
    source_rows.sort(
        key=lambda row: (
            -row["error_count_out_of_10"],
            -row["prediction_flip_samples"],
            -row["mean_probability_ai_edited_for_authentic"],
            row["mean_probability_ai_edited_for_edited"],
        )
    )
    ranked_sources = [dict(source_rank=index + 1, **row) for index, row in enumerate(source_rows[:25])]

    source_error_analysis = {
        "analysis_classification": EXPLORATORY_CLASSIFICATION,
        "source_identifiers_published": False,
        "source_count": len(source_rows),
        "observations_per_source": 10,
        "source_error_count_histogram": dict(
            sorted(Counter(str(row["error_count_out_of_10"]) for row in source_rows).items(), key=lambda item: int(item[0]))
        ),
        "per_sample_error_votes_across_five_checkpoints": {
            "authentic_false_positive_votes": dict(sorted(vote_patterns["authentic"].items(), key=lambda item: int(item[0]))),
            "edited_false_negative_votes": dict(sorted(vote_patterns["ai_edited"].items(), key=lambda item: int(item[0]))),
        },
        "sources_with_any_error": sum(row["error_count_out_of_10"] > 0 for row in source_rows),
        "sources_with_prediction_instability": sum(row["prediction_flip_samples"] > 0 for row in source_rows),
        "sources_all_ten_predictions_correct": sum(row["error_count_out_of_10"] == 0 for row in source_rows),
        "sources_all_ten_predictions_wrong": sum(row["error_count_out_of_10"] == 10 for row in source_rows),
        "top_25_anonymous_hardest_sources": ranked_sources,
    }

    exploratory = {
        "schema_version": "1.0.0",
        "analysis_classification": EXPLORATORY_CLASSIFICATION,
        "scientific_guardrails": {
            "changes_confirmatory_verdict": False,
            "threshold_tuning_performed": False,
            "model_selection_performed": False,
            "inference_calls": 0,
            "training_runs": 0,
            "raw_source_sample_or_path_identifiers_published": False,
        },
        "confusion_analysis": {
            "per_checkpoint": per_seed_confusion,
            "pooled_descriptive_non_independent": {
                "tn": pooled_confusion["tn"],
                "fp": pooled_confusion["fp"],
                "fn": pooled_confusion["fn"],
                "tp": pooled_confusion["tp"],
                "authentic_false_positive_rate": pooled_confusion["fp"]
                / (pooled_confusion["tn"] + pooled_confusion["fp"]),
                "edited_false_negative_rate": pooled_confusion["fn"]
                / (pooled_confusion["tp"] + pooled_confusion["fn"]),
            },
            "paired_source_outcomes_per_checkpoint": {
                seed: dict(sorted(outcomes.items())) for seed, outcomes in pair_outcomes_by_seed.items()
            },
        },
        "probability_analysis": {
            "per_checkpoint": probability_by_seed,
            "pooled_descriptive_non_independent": probability_pooled,
        },
        "source_error_analysis": source_error_analysis,
    }

    after = snapshot_files(source_files + [wrapper_receipt_path])
    if before != after:
        raise RuntimeError("One or more evaluator input artifacts changed during read-only analysis")

    reconciliation = {
        "schema_version": "1.0.0",
        "result_classification": "CONFIRMATORY_RESULT_RECONCILIATION_NO_NEW_MODEL_EVALUATION",
        "checksum_audit": checksum_audit,
        "prediction_files_verified": 5,
        "samples_per_checkpoint": expected_samples,
        "sources": expected_sources,
        "sample_identity_and_order_equal_across_checkpoints": True,
        "per_checkpoint_metrics_recomputed_from_existing_logits": "PASS",
        "aggregate_macro_f1_recomputed": aggregate_mean,
        "aggregate_matches_published_result": True,
        "bootstrap_summary_cross_file_consistency": "PASS_NOT_RERUN",
        "ci_lower_95": aggregate_artifact["ci_lower_95"],
        "ci_upper_95": aggregate_artifact["ci_upper_95"],
        "confirmatory_verdict": decision_artifact["verdict"],
        "confirmatory_success": decision_artifact["success"],
        "wrapper_adjudication": {
            "outer_wrapper_false_negative": True,
            "controller_exit_code": wrapper_receipt["controller_exit_code"],
            "execution_receipt_status": execution_receipt["status"],
            "cause": "Wrapper checked non-contract confirmatory_result.json while the sealed controller published aggregate_confirmatory_metrics.json and confirmatory_decision.json.",
            "retry_performed": False,
            "scientific_result_affected": False,
        },
        "input_artifacts_unchanged": True,
        "new_inference_calls": 0,
        "new_training_runs": 0,
    }

    atomic_write_json(output_dir / "artifact_reconciliation.json", reconciliation)
    atomic_write_json(output_dir / "exploratory_error_analysis.json", exploratory)
    return reconciliation, exploratory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--wrapper-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sources", type=int, default=343)
    parser.add_argument("--expected-samples", type=int, default=686)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconciliation, exploratory = analyze(
        outputs_dir=args.outputs_dir,
        wrapper_receipt_path=args.wrapper_receipt,
        output_dir=args.output_dir,
        expected_sources=args.expected_sources,
        expected_samples=args.expected_samples,
    )
    print(
        json.dumps(
            {
                "status": "PHASE_4C2G_ARTIFACT_ANALYSIS_COMPLETE",
                "confirmatory_verdict": reconciliation["confirmatory_verdict"],
                "analysis_classification": exploratory["analysis_classification"],
                "new_inference_calls": 0,
                "new_training_runs": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
