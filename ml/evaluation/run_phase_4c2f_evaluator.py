"""
CLI entry point for Phase 4C.2F / Phase 4C.2 Locked-Test Confirmatory Evaluator.

Usage:
  # Synthetic dry-run (safe, zero locked-test access):
  python -m ml.evaluation.run_phase_4c2f_evaluator --dry-run-synthetic --output-dir /path/to/output

  # Formal unsealing (STRICTLY REQUIRES signed human authorization):
  python -m ml.evaluation.run_phase_4c2f_evaluator \\
      --authorization-file /path/to/HUMAN_UNSEALING_AUTHORIZATION.json \\
      --checkpoint-dir /path/to/checkpoints \\
      --locked-test-dir /path/to/locked_test \\
      --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from ml.evaluation.locked_test_evaluator import (
    EXPECTED_LOCKED_TEST_SAMPLES,
    EXPECTED_LOCKED_TEST_SOURCES,
    LockedTestEvaluator,
)


def generate_synthetic_mock_records() -> List[Dict[str, Any]]:
    """Generates synthetic mock records matching locked-test cardinality (343 sources, 686 samples)."""
    records: List[Dict[str, Any]] = []
    for s_idx in range(EXPECTED_LOCKED_TEST_SOURCES):
        source_id = f"synth_source_{s_idx:04d}"
        # Exactly two samples per source: one authentic (0), one ai_edited (1)
        records.append({
            "source_id": source_id,
            "sample_idx": s_idx * 2,
            "true_label": 0,
            "relative_path": f"synthetic/{source_id}_auth.png",
        })
        records.append({
            "source_id": source_id,
            "sample_idx": s_idx * 2 + 1,
            "true_label": 1,
            "relative_path": f"synthetic/{source_id}_edit.png",
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4C.2F Locked-Test Confirmatory Evaluator"
    )
    parser.add_argument(
        "--authorization-file",
        type=Path,
        default=None,
        help="Path to signed HUMAN_UNSEALING_AUTHORIZATION.json artifact (required for locked-test)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Path to directory containing resolved Stage 1 N=250 checkpoints",
    )
    parser.add_argument(
        "--locked-test-dir",
        type=Path,
        default=None,
        help="Path to read-only locked-test data root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory for access ledger, receipts, predictions, and summary",
    )
    parser.add_argument(
        "--dry-run-synthetic",
        action="store_true",
        help="Execute synthetic dry-run with mock data (0 locked-test access)",
    )

    args = parser.parse_args()

    evaluator = LockedTestEvaluator(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.dry_run_synthetic:
        print("[INFO] Executing synthetic dry-run (no locked-test access)...")
        mock_data = generate_synthetic_mock_records()
        summary = evaluator.run_synthetic_dry_run(mock_data)
        print(f"[SUCCESS] Synthetic dry-run completed successfully.")
        print(f"          Aggregate Macro-F1: {summary['bootstrap_results']['aggregate_macro_f1']:.4f}")
        print(f"          95% CI: [{summary['bootstrap_results']['ci_lower_95']:.4f}, {summary['bootstrap_results']['ci_upper_95']:.4f}]")
        print(f"          Verdict: {summary['bootstrap_results']['verdict']}")
        return 0

    # Formal execution path: STRICTLY FAIL-CLOSED
    print("[SECURITY] Verifying human unsealing authorization...")
    if not args.authorization_file:
        print(
            "[BLOCKED] Missing --authorization-file. "
            "Evaluator refuses to access locked-test without valid human authorization.",
            file=sys.stderr,
        )
        return 1

    try:
        auth_data = evaluator.verify_human_authorization(args.authorization_file)
        print(f"[SECURITY] Authorization verified. Approver: {auth_data.get('approver_id')}")
    except PermissionError as e:
        print(f"[BLOCKED] Authorization check failed: {e}", file=sys.stderr)
        return 1

    print("[SECURITY] Verifying network isolation...")
    try:
        evaluator.verify_network_isolation()
        print("[SECURITY] Network isolation confirmed.")
    except RuntimeError as e:
        print(f"[BLOCKED] Network isolation check failed: {e}", file=sys.stderr)
        return 2

    if not args.locked_test_dir or not args.locked_test_dir.is_dir():
        print(f"[BLOCKED] Invalid locked-test directory: {args.locked_test_dir}", file=sys.stderr)
        return 3

    print("[SECURITY] Verifying read-only mount...")
    try:
        evaluator.verify_read_only_mount(args.locked_test_dir)
        print("[SECURITY] Read-only data mount confirmed.")
    except RuntimeError as e:
        print(f"[BLOCKED] Read-only mount check failed: {e}", file=sys.stderr)
        return 4

    print("[SECURITY] Resolving and verifying candidate checkpoints...")
    try:
        cp_info = evaluator.resolve_and_verify_checkpoints(args.checkpoint_dir)
        print(f"[SECURITY] All {len(cp_info)} candidate checkpoints verified by SHA-256 and byte count.")
    except Exception as e:
        print(f"[BLOCKED] Checkpoint verification failed: {e}", file=sys.stderr)
        return 5

    print("[BLOCKED] Phase 4C.2F scope: Evaluator built and sealed, but locked-test execution is unauthorized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
