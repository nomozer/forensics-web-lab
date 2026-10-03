"""Complete Phase 4C.2G locked-test confirmatory execution driver.

The module has no import-time data access. Formal execution requires an explicit
authorization artifact, fresh isolation receipt, OS-backed read-only proof,
five exact checkpoint bindings, and explicit locked-test paths.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ml.evaluation.confirmatory_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_RNG_SEED,
    UNINFORMATIVE_REFERENCE_THRESHOLD,
    compute_source_cluster_bootstrap_ci,
)
from ml.evaluation.locked_test_evaluator import AUTHORIZED_SEEDS, LockedTestEvaluator
from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest, sha256_file
from ml.evaluation.phase_4c2g_io import atomic_write_json
from ml.evaluation.phase_4c2g_model import (
    create_locked_test_data_loader,
    load_stage1_checkpoint,
)
from ml.evaluation.phase_4c2g_windows import (
    verify_isolation_receipt,
    verify_windows_read_only_receipt,
)


InferenceFunction = Callable[[int, Mapping[str, Any], Any], Mapping[str, Any]]


def make_stage1_inference_function(*, device: str) -> InferenceFunction:
    """Build real Stage 1 inference without fitting, tuning, or ensembling."""
    target_device = torch.device(device)

    def infer(
        seed: int,
        checkpoint_info: Mapping[str, Any],
        data_loader: Any,
    ) -> Mapping[str, Any]:
        loaded = load_stage1_checkpoint(
            checkpoint_path=checkpoint_info["path"],
            expected_sha256=str(checkpoint_info["sha256"]),
            expected_seed=seed,
            expected_bytes=int(checkpoint_info["byte_count"]),
            device=target_device,
        )
        records: list[dict[str, Any]] = []
        loaded.model.eval()
        with torch.no_grad():
            for inputs, labels, source_ids, sample_ids, relative_paths in data_loader:
                logits = loaded.model(inputs.to(target_device, non_blocking=False))
                logits_cpu = logits.detach().to("cpu", dtype=torch.float64).numpy()
                labels_cpu = labels.detach().to("cpu").numpy()
                for offset in range(len(labels_cpu)):
                    records.append(
                        {
                            "source_id": str(source_ids[offset]),
                            "sample_id": str(sample_ids[offset]),
                            "sample_idx": len(records),
                            "relative_path": str(relative_paths[offset]),
                            "true_label": int(labels_cpu[offset]),
                            "logits": [
                                float(logits_cpu[offset, 0]),
                                float(logits_cpu[offset, 1]),
                            ],
                        }
                    )
        return {
            "seed": seed,
            "protocol": "stage1_frozen_backbone_linear_probe",
            "sample_size": 250,
            "predictions": records,
            "calibration_fitted": False,
            "threshold_tuned": False,
            "probability_ensemble": False,
        }

    return infer


def run_five_checkpoint_orchestration(
    *,
    evaluator: LockedTestEvaluator,
    checkpoint_infos: Mapping[int, Mapping[str, Any]],
    data_loader: Any,
    inference_fn: InferenceFunction,
    session_id: str,
) -> list[dict[str, Any]]:
    """Run all and only the five preregistered checkpoints in canonical order."""
    actual_seeds = set(checkpoint_infos)
    expected_seeds = set(AUTHORIZED_SEEDS)
    if actual_seeds != expected_seeds:
        raise ValueError(
            "Checkpoint seed set mismatch: "
            f"expected {AUTHORIZED_SEEDS}, got {sorted(actual_seeds)}"
        )

    receipts: list[dict[str, Any]] = []
    for seed in AUTHORIZED_SEEDS:
        receipt = evaluator.execute_checkpoint_evaluation(
            seed=seed,
            checkpoint_info=dict(checkpoint_infos[seed]),
            inference_fn=inference_fn,
            data_loader=data_loader,
            session_id=session_id,
        )
        receipts.append(receipt)

    if evaluator.ledger.evaluation_attempts != 5:
        raise RuntimeError(
            "Five-checkpoint execution ended with an invalid attempt count: "
            f"{evaluator.ledger.evaluation_attempts}"
        )
    if evaluator.ledger.completed_model_evaluations != 5:
        raise RuntimeError(
            "Five-checkpoint execution ended with an invalid completion count: "
            f"{evaluator.ledger.completed_model_evaluations}"
        )
    return receipts


def _load_prediction_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("probability_ensemble") is not False:
        raise ValueError(f"Probability ensemble prohibition missing in {path}")
    if payload.get("calibration_fitted") is not False:
        raise ValueError(f"Calibration fitting prohibition missing in {path}")
    if payload.get("threshold_tuned") is not False:
        raise ValueError(f"Threshold tuning prohibition missing in {path}")
    return payload


def finalize_confirmatory_outputs(
    *,
    evaluator: LockedTestEvaluator,
    receipts: Sequence[Mapping[str, Any]],
    session_id: str,
    manifest_binding: Mapping[str, str],
    environment: Mapping[str, Any],
    n_bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Compute preregistered statistics and atomically seal all required outputs."""
    if [receipt.get("checkpoint_seed") for receipt in receipts] != AUTHORIZED_SEEDS:
        raise ValueError("Receipts are not in the exact preregistered checkpoint order.")
    if evaluator.ledger.evaluation_attempts != 5:
        raise RuntimeError("Cannot finalize unless exactly five attempts were reserved.")
    if evaluator.ledger.completed_model_evaluations != 5:
        raise RuntimeError("Cannot finalize unless exactly five evaluations completed.")

    output_dir = evaluator.output_dir
    predictions_per_checkpoint: dict[int, np.ndarray] = {}
    canonical_source_ids: list[str] | None = None
    canonical_y_true: np.ndarray | None = None
    per_checkpoint: dict[str, Any] = {}

    for receipt in receipts:
        seed = int(receipt["checkpoint_seed"])
        payload = _load_prediction_payload(output_dir / receipt["prediction_file"])
        rows = payload["predictions"]
        source_ids = [str(row["source_id"]) for row in rows]
        y_true = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
        logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
        if canonical_source_ids is None:
            canonical_source_ids = source_ids
            canonical_y_true = y_true
        elif source_ids != canonical_source_ids or not np.array_equal(
            y_true, canonical_y_true
        ):
            raise ValueError("Prediction cohort/order differs across checkpoints.")
        predictions_per_checkpoint[seed] = logits
        per_checkpoint[str(seed)] = {
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "prediction_file": receipt["prediction_file"],
            "prediction_file_sha256": receipt["prediction_file_sha256"],
            "metrics": receipt["metrics"],
        }

    if canonical_source_ids is None or canonical_y_true is None:
        raise RuntimeError("No predictions were available for finalization.")

    bootstrap = compute_source_cluster_bootstrap_ci(
        source_ids=canonical_source_ids,
        y_true=canonical_y_true,
        predictions_per_checkpoint=predictions_per_checkpoint,
        n_replicates=n_bootstrap_replicates,
        seed=BOOTSTRAP_RNG_SEED,
        is_synthetic_dry_run=False,
    )
    aggregate = {
        **bootstrap,
        "aggregation": "arithmetic_mean_of_5_macro_f1",
        "checkpoint_order": AUTHORIZED_SEEDS,
        "probability_ensemble": False,
        "best_seed_selection": False,
        "threshold_tuning": False,
        "calibration_fitting": False,
    }
    decision = {
        "reference": "balanced_binary_uninformative_reference",
        "reference_value": UNINFORMATIVE_REFERENCE_THRESHOLD,
        "rule": "ci_lower_95 > 0.5000",
        "ci_lower_95": bootstrap["ci_lower_95"],
        "success": bootstrap["ci_lower_95"] > UNINFORMATIVE_REFERENCE_THRESHOLD,
        "verdict": bootstrap["verdict"],
    }
    bootstrap_summary = {
        key: bootstrap[key]
        for key in (
            "bootstrap_replicates",
            "bootstrap_rng_seed",
            "bootstrap_generator",
            "resampling_unit",
            "observations_per_replicate",
            "sources_per_replicate",
            "bootstrap_distribution_sha256_float64_le",
            "bootstrap_distribution_mean",
            "bootstrap_distribution_min",
            "bootstrap_distribution_max",
            "ci_lower_95",
            "ci_upper_95",
        )
    }
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    execution_receipt = {
        "timestamp_utc": now,
        "session_id": session_id,
        "status": "COMPLETED_VALID",
        "checkpoint_order": AUTHORIZED_SEEDS,
        "evaluation_attempts": evaluator.ledger.evaluation_attempts,
        "completed_model_evaluations": evaluator.ledger.completed_model_evaluations,
        "completed_unsealing_sessions": evaluator.ledger.completed_unsealing_sessions,
        "manifest_binding": dict(manifest_binding),
        "tip_entry_hash": evaluator.ledger.tip_entry_hash,
    }
    environment_payload = {
        "timestamp_utc": now,
        "python": sys.version,
        "platform": platform.platform(),
        "network_bytes_transmitted_by_evaluator": 0,
        **dict(environment),
    }

    atomic_write_json(output_dir / "per_checkpoint_metrics.json", per_checkpoint)
    atomic_write_json(output_dir / "aggregate_confirmatory_metrics.json", aggregate)
    atomic_write_json(output_dir / "bootstrap_distribution_summary.json", bootstrap_summary)
    atomic_write_json(output_dir / "confirmatory_decision.json", decision)
    atomic_write_json(output_dir / "execution_receipt.json", execution_receipt)
    atomic_write_json(output_dir / "environment.json", environment_payload)

    evaluator.ledger.record_event(
        event_type="SESSION_CLOSE",
        session_id=session_id,
        metadata={
            "status": "COMPLETED_VALID",
            "confirmatory_verdict": decision["verdict"],
        },
    )
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "checksums.json" and not path.name.endswith(
            ".part"
        ):
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    checksums = {
        "timestamp_utc": now,
        "algorithm": "sha256",
        "self_digest_excluded": True,
        "files": files,
    }
    atomic_write_json(output_dir / "checksums.json", checksums)
    return {
        "per_checkpoint": per_checkpoint,
        "aggregate": aggregate,
        "bootstrap_summary": bootstrap_summary,
        "confirmatory_decision": decision,
        "execution_receipt": execution_receipt,
        "environment": environment_payload,
        "checksums": checksums,
    }


def _load_source_binding(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source binding root must be an object.")
    return payload


def verify_execution_authorization_binding(
    *,
    authorization: Mapping[str, Any],
    binding_path: Path | str,
    package_archive: Path | str,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check authorization against the external canonical package binding."""
    binding_file = Path(binding_path).resolve(strict=True)
    binding = json.loads(binding_file.read_text(encoding="utf-8"))
    if not isinstance(binding, dict):
        raise PermissionError("Execution authorization binding root must be an object.")

    manifest_commitment = binding.get("locked_test_manifest_commitment")
    if not isinstance(manifest_commitment, dict) or manifest_commitment.get(
        "status"
    ) != "COMMITTED_BEFORE_UNSEALING":
        raise PermissionError("BLOCKED_MANIFEST_COMMITMENT_ABSENT")

    archive = Path(package_archive).resolve(strict=True)
    archive_binding = binding.get("archive")
    if not isinstance(archive_binding, dict):
        raise PermissionError("External binding has no canonical archive record.")
    canonical_archive = {
        "filename": archive.name,
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
    }
    for field, actual in canonical_archive.items():
        if archive_binding.get(field) != actual:
            raise PermissionError(
                f"Canonical archive binding mismatch for archive.{field}."
            )

    source_components = source_binding.get("components")
    if binding.get("runtime_components") != source_components:
        raise PermissionError("External binding runtime_components mismatch source binding.")
    if binding.get("authorization_schema") != source_binding.get(
        "authorization_schema"
    ):
        raise PermissionError("External binding authorization_schema mismatch.")

    required_matches = {
        "sealed_package_sha256": archive_binding["sha256"],
        "sealed_package_bytes": archive_binding["bytes"],
        "evaluator_effective_commit": binding.get("effective_execution_commit"),
        "execution_package_commit": binding.get("execution_package_commit"),
        "locked_test_manifest_sha256": manifest_commitment.get("sha256"),
    }
    for field, expected in required_matches.items():
        if authorization.get(field) != expected:
            raise PermissionError(
                f"Authorization field {field!r} does not match external canonical binding."
            )

    if source_binding.get("effective_evaluator_commit") != binding.get(
        "effective_execution_commit"
    ):
        raise PermissionError("Source binding effective commit mismatch.")
    if source_binding.get("execution_package_commit") != binding.get(
        "execution_package_commit"
    ):
        raise PermissionError("Source binding package commit mismatch.")

    component_names = source_binding.get("authorization_component_names")
    if not isinstance(component_names, list) or not component_names:
        raise PermissionError("Source binding authorization component set is absent.")
    expected_component_hashes = {
        name: source_components[name]["sha256"] for name in component_names
    }
    if authorization.get("evaluator_component_hashes") != expected_component_hashes:
        raise PermissionError(
            "Authorization field 'evaluator_component_hashes' does not match external canonical binding."
        )

    return {
        "status": "EXECUTION_AUTHORIZATION_BINDING_VERIFIED",
        "binding_sha256": sha256_file(binding_file),
        "archive_sha256": archive_binding["sha256"],
        "archive_bytes": archive_binding["bytes"],
        "effective_execution_commit": binding["effective_execution_commit"],
        "locked_test_manifest_sha256": manifest_commitment["sha256"],
    }


def verify_execution_components(
    source_binding: Mapping[str, Any],
    *,
    package_root: Path,
) -> None:
    components = source_binding.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("Source binding has no execution components.")
    for name, metadata in components.items():
        relative_path = Path(str(metadata["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe component path for {name}: {relative_path}")
        component_path = (package_root / relative_path).resolve(strict=True)
        component_path.relative_to(package_root)
        actual_bytes = component_path.stat().st_size
        actual_sha = sha256_file(component_path)
        if actual_bytes != metadata["bytes"] or actual_sha != metadata["sha256"]:
            raise ValueError(
                f"Execution component binding mismatch for {name}: "
                f"bytes={actual_bytes}, sha256={actual_sha}"
            )


def execute_confirmatory_session(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the formal future session in the preregistered fail-closed order."""
    package_root = Path(__file__).resolve().parents[2]
    source_binding = _load_source_binding(args.source_binding)
    verify_execution_components(source_binding, package_root=package_root)

    authorization = LockedTestEvaluator.verify_human_authorization(
        args.authorization_file,
        schema_path=args.authorization_schema,
        source_binding_path=args.source_binding,
        bypass_git_checks=True,
    )
    package_archive = args.package_archive.resolve(strict=True)
    binding_verification = verify_execution_authorization_binding(
        authorization=authorization,
        binding_path=args.execution_authorization_binding,
        package_archive=package_archive,
        source_binding=source_binding,
    )
    expected_commit = binding_verification["effective_execution_commit"]
    package_sha256 = binding_verification["archive_sha256"]
    package_bytes = binding_verification["archive_bytes"]

    isolation = verify_isolation_receipt(
        args.isolation_receipt,
        expected_session_id=args.session_id,
    )
    read_only = verify_windows_read_only_receipt(
        args.read_only_receipt,
        locked_test_root=args.locked_test_root,
        expected_session_id=args.session_id,
    )

    evaluator = LockedTestEvaluator(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        effective_evaluator_commit=str(expected_commit),
    )
    if evaluator.ledger.entries:
        raise RuntimeError("Output ledger is not empty; silent session reuse is forbidden.")
    checkpoint_infos = evaluator.resolve_and_verify_checkpoints(args.checkpoint_dir)

    evaluator.ledger.record_event(
        event_type="PRE_READ_UNSEAL",
        session_id=args.session_id,
        metadata={
            "authorization_id": authorization["authorization_id"],
            "isolation_receipt_sha256": isolation["receipt_sha256"],
            "read_only_receipt_sha256": read_only["receipt_sha256"],
        },
        inc_unsealing_session=True,
    )
    try:
        manifest = load_locked_test_manifest(
            locked_test_root=args.locked_test_root,
            manifest_path=args.manifest,
            manifest_schema_path=args.manifest_schema,
            expected_manifest_sha256=binding_verification[
                "locked_test_manifest_sha256"
            ],
            expected_schema_sha256=authorization[
                "locked_test_manifest_schema_sha256"
            ],
            expected_split_seal=authorization["locked_test_split_seal"],
        )
        data_loader = create_locked_test_data_loader(
            manifest,
            batch_size=args.batch_size,
        )
        receipts = run_five_checkpoint_orchestration(
            evaluator=evaluator,
            checkpoint_infos=checkpoint_infos,
            data_loader=data_loader,
            inference_fn=make_stage1_inference_function(device=args.device),
            session_id=args.session_id,
        )
        return finalize_confirmatory_outputs(
            evaluator=evaluator,
            receipts=receipts,
            session_id=args.session_id,
            manifest_binding={
                "manifest_sha256": manifest.manifest_sha256,
                "schema_sha256": manifest.schema_sha256,
                "split_seal": manifest.split_seal,
            },
            environment={
                "device": args.device,
                "batch_size": args.batch_size,
                "authorization_id": authorization["authorization_id"],
                "effective_evaluator_commit": expected_commit,
                "sealed_package_sha256": package_sha256,
                "sealed_package_bytes": package_bytes,
                "execution_authorization_binding_sha256": binding_verification[
                    "binding_sha256"
                ],
                "isolation_receipt_sha256": isolation["receipt_sha256"],
                "read_only_receipt_sha256": read_only["receipt_sha256"],
                "scientific_result": True,
            },
        )
    except Exception as exc:
        evaluator.ledger.record_event(
            event_type="SESSION_INTERRUPTED",
            session_id=args.session_id,
            metadata={"error": str(exc), "human_adjudication_required": True},
        )
        raise


def execute_authorization_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate authorization/package bindings without touching locked-test paths."""
    package_root = Path(__file__).resolve().parents[2]
    source_binding = _load_source_binding(args.source_binding)
    verify_execution_components(source_binding, package_root=package_root)
    authorization = LockedTestEvaluator.verify_human_authorization(
        args.authorization_file,
        schema_path=args.authorization_schema,
        source_binding_path=args.source_binding,
        bypass_git_checks=True,
    )
    package_archive = args.package_archive.resolve(strict=True)
    binding_verification = verify_execution_authorization_binding(
        authorization=authorization,
        binding_path=args.execution_authorization_binding,
        package_archive=package_archive,
        source_binding=source_binding,
    )
    return {
        "status": "AUTHORIZATION_AND_FINAL_PACKAGE_BINDING_VERIFIED",
        "authorization_id": authorization["authorization_id"],
        "execution_authorization_binding_sha256": binding_verification[
            "binding_sha256"
        ],
        "locked_test_accesses": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4C.2G five-checkpoint confirmatory evaluator"
    )
    parser.add_argument("--authorization-file", type=Path, required=True)
    parser.add_argument("--package-archive", type=Path, required=True)
    parser.add_argument(
        "--execution-authorization-binding", type=Path, required=True
    )
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--authorization-schema", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--locked-test-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-schema", type=Path, required=True)
    parser.add_argument("--isolation-receipt", type=Path, required=True)
    parser.add_argument("--read-only-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--authorization-preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.authorization_preflight_only:
            result = execute_authorization_preflight(args)
        else:
            result = execute_confirmatory_session(args)
    except Exception as exc:
        print(f"[BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 20
    decision = result.get("confirmatory_decision", result)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
