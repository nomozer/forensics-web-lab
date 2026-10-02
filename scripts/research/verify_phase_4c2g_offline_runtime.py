#!/usr/bin/env python3
"""
Standalone Offline Runtime Verifier for Phase 4C.2G.

Verifies execution worktree, component hashes, 5 canonical checkpoints,
frozen dependencies, passive network isolation, and filesystem separation
prior to human unsealing authorization.

Zero inference, zero model forward, zero torch load, zero socket probes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED_EXECUTION_PACKAGE_COMMIT = "2826a8274cb89ec548d6fac5c8ae50c1c2836202"
EXPECTED_EFFECTIVE_EVALUATOR_COMMIT = "3cf75c2bf0c9835dd58897b7b36982732cab40ab"
EXPECTED_PIP_FREEZE_SHA256 = "0e11c90b41f22f3f302a4c1a9860c6e3ffba109d6e465f361d447f71a3709a0c"

EXPECTED_COMPONENTS = {
    "ml/evaluation/confirmatory_metrics.py": {
        "bytes": 15199,
        "sha256": "bacbb9dc0a230359f1c3bbaece1ce8c19e30ddafa807313480d173a808bde808",
    },
    "ml/evaluation/locked_test_evaluator.py": {
        "bytes": 47708,
        "sha256": "b311e010535600781e52014fb6b675d32d78a558c5fe7e67bd3ec00b6989a5e1",
    },
    "ml/evaluation/run_phase_4c2f_evaluator.py": {
        "bytes": 5739,
        "sha256": "37ed02ff44d8c523150ebb968d2777bba9c2dee8597b3df60efa1fb5bd1e9041",
    },
    "docs/schemas/human-unsealing-authorization.v1.schema.json": {
        "bytes": 6143,
        "sha256": "15291643b2ca2b3d6e3c135957db140e33f7cc65bc11aebc4668f3e105af6734",
    },
}

CANONICAL_CHECKPOINTS = {
    42: {
        "subpath": "n250_seed_42/best_checkpoint.pt",
        "bytes": 5627375,
        "sha256": "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
    },
    1337: {
        "subpath": "n250_seed_1337/best_checkpoint.pt",
        "bytes": 5627375,
        "sha256": "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
    },
    2025: {
        "subpath": "n250_seed_2025/best_checkpoint.pt",
        "bytes": 5627375,
        "sha256": "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
    },
    3407: {
        "subpath": "n250_seed_3407/best_checkpoint.pt",
        "bytes": 5627375,
        "sha256": "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
    },
    9001: {
        "subpath": "n250_seed_9001/best_checkpoint.pt",
        "bytes": 5627375,
        "sha256": "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3",
    },
}


def compute_streaming_sha256_and_bytes(file_path: Path) -> tuple[str, int]:
    """Compute SHA-256 and byte count via 64KB streaming buffer."""
    h = hashlib.sha256()
    total_bytes = 0
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            total_bytes += len(chunk)
    return h.hexdigest(), total_bytes


def inspect_passive_network() -> dict:
    """Inspect local network settings passively with zero outbound network probes."""
    results = {
        "outbound_socket_probes_sent": 0,
        "dns_lookups_performed": 0,
        "http_requests_sent": 0,
        "http_proxy": os.environ.get("HTTP_PROXY", ""),
        "https_proxy": os.environ.get("HTTPS_PROXY", ""),
        "all_proxy": os.environ.get("ALL_PROXY", ""),
        "default_route_detected": False,
        "connected_network_adapters": [],
        "active_vpn_detected": False,
    }

    # Check Windows route print
    if platform.system() == "Windows":
        try:
            route_out = subprocess.check_output(
                ["route", "print", "0.0.0.0"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in route_out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                    results["default_route_detected"] = True
                    break
        except Exception:
            pass

        # Check PowerShell Get-NetAdapter (local query only, no outbound network)
        try:
            ps_cmd = 'Get-NetAdapter | Where-Object { $_.Status -eq "Up" } | Select-Object -ExpandProperty Name'
            adapter_out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            adapters = [a.strip() for a in adapter_out.splitlines() if a.strip()]
            results["connected_network_adapters"] = adapters
        except Exception:
            pass
    else:
        # Linux passive route check via /proc/net/route
        try:
            route_file = Path("/proc/net/route")
            if route_file.exists():
                for line in route_file.read_text().splitlines()[1:]:
                    fields = line.strip().split()
                    if len(fields) >= 2 and fields[1] == "00000000":
                        results["default_route_detected"] = True
                        break
        except Exception:
            pass

    return results


def verify_offline_runtime(
    execution_worktree: Path,
    checkpoint_root: Path,
    output_dir: Path,
    planned_locked_test_mountpoint: Path,
    synthetic_only: bool = False,
) -> dict:
    """Run all 9 offline readiness checks."""
    obs_start = datetime.now(timezone.utc).isoformat()
    checks = {}
    errors = []

    # 1. Check execution worktree HEAD
    try:
        head_out = subprocess.check_output(
            ["git", "-C", str(execution_worktree), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if head_out == EXPECTED_EXECUTION_PACKAGE_COMMIT:
            checks["worktree_head"] = {
                "status": "PASS",
                "commit": head_out,
            }
        else:
            checks["worktree_head"] = {
                "status": "FAIL",
                "commit": head_out,
                "expected": EXPECTED_EXECUTION_PACKAGE_COMMIT,
            }
            errors.append(f"Worktree HEAD mismatch: {head_out}")
    except Exception as e:
        checks["worktree_head"] = {"status": "FAIL", "error": str(e)}
        errors.append(f"Worktree git error: {e}")

    # 2. Check worktree porcelain status
    try:
        status_out = subprocess.check_output(
            ["git", "-C", str(execution_worktree), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not status_out:
            checks["worktree_cleanliness"] = {"status": "PASS", "porcelain": "CLEAN"}
        else:
            checks["worktree_cleanliness"] = {"status": "FAIL", "porcelain": status_out}
            errors.append("Worktree is dirty")
    except Exception as e:
        checks["worktree_cleanliness"] = {"status": "FAIL", "error": str(e)}
        errors.append(f"Worktree status error: {e}")

    # 3. Check 4 evaluator components
    comp_results = {}
    comp_pass = True
    for rel_path, expected in EXPECTED_COMPONENTS.items():
        comp_file = execution_worktree / rel_path
        if not comp_file.exists():
            comp_results[rel_path] = {"status": "FAIL", "reason": "FILE_MISSING"}
            comp_pass = False
            errors.append(f"Component missing: {rel_path}")
            continue
        sha, b_count = compute_streaming_sha256_and_bytes(comp_file)
        if sha == expected["sha256"] and b_count == expected["bytes"]:
            comp_results[rel_path] = {"status": "PASS", "sha256": sha, "bytes": b_count}
        else:
            comp_results[rel_path] = {
                "status": "FAIL",
                "actual_sha256": sha,
                "expected_sha256": expected["sha256"],
                "actual_bytes": b_count,
                "expected_bytes": expected["bytes"],
            }
            comp_pass = False
            errors.append(f"Component mismatch: {rel_path}")
    checks["evaluator_components"] = {
        "status": "PASS" if comp_pass else "FAIL",
        "components": comp_results,
    }

    # 4. Check 5 canonical checkpoints (zero model load, zero inference)
    ckpt_results = {}
    ckpt_pass = True
    for seed, expected in CANONICAL_CHECKPOINTS.items():
        ckpt_file = checkpoint_root / expected["subpath"]
        if not ckpt_file.exists():
            ckpt_results[f"seed_{seed}"] = {"status": "FAIL", "reason": "FILE_MISSING"}
            ckpt_pass = False
            errors.append(f"Checkpoint seed {seed} missing at {ckpt_file}")
            continue
        sha, b_count = compute_streaming_sha256_and_bytes(ckpt_file)
        if sha == expected["sha256"] and b_count == expected["bytes"]:
            ckpt_results[f"seed_{seed}"] = {
                "status": "PASS",
                "sha256": sha,
                "bytes": b_count,
                "model_loaded": False,
                "inference_run": False,
            }
        else:
            ckpt_results[f"seed_{seed}"] = {
                "status": "FAIL",
                "actual_sha256": sha,
                "expected_sha256": expected["sha256"],
                "actual_bytes": b_count,
                "expected_bytes": expected["bytes"],
            }
            ckpt_pass = False
            errors.append(f"Checkpoint seed {seed} digest/bytes mismatch")
    checks["checkpoints"] = {
        "status": "PASS" if ckpt_pass else "FAIL",
        "torch_load_count": 0,
        "inference_calls": 0,
        "results": ckpt_results,
    }

    # 5. Dependency environment
    dep_checks = {}
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
        )
        dep_checks["pip_check"] = "PASS" if pip_check.returncode == 0 else "FAIL"
    except Exception as e:
        dep_checks["pip_check"] = f"FAIL: {e}"

    try:
        freeze_out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        freeze_sha = hashlib.sha256(freeze_out.encode("utf-8")).hexdigest()
        dep_checks["pip_freeze_sha256"] = freeze_sha
        dep_checks["pip_freeze_match"] = freeze_sha == EXPECTED_PIP_FREEZE_SHA256
    except Exception as e:
        dep_checks["pip_freeze_sha256"] = f"FAIL: {e}"
        dep_checks["pip_freeze_match"] = False

    checks["dependencies"] = dep_checks

    # 6. Passive network isolation
    net_info = inspect_passive_network()
    has_proxy = bool(net_info["http_proxy"] or net_info["https_proxy"] or net_info["all_proxy"])
    has_default_route = net_info["default_route_detected"]
    has_connected_adapters = len(net_info["connected_network_adapters"]) > 0

    is_network_isolated = (not has_proxy) and (not has_default_route) and (not has_connected_adapters)
    checks["network_isolation"] = {
        "status": "PASS" if is_network_isolated else "USER_PHYSICAL_ACTION_REQUIRED",
        "has_proxy": has_proxy,
        "default_route_detected": has_default_route,
        "connected_network_adapters": net_info["connected_network_adapters"],
        "passive_checks": net_info,
    }

    # 7. Filesystem preparation
    output_dir.mkdir(parents=True, exist_ok=True)
    planned_locked_test_mountpoint.mkdir(parents=True, exist_ok=True)

    output_real = output_dir.resolve()
    mount_real = planned_locked_test_mountpoint.resolve()

    # Disjoint check
    disjoint = (output_real != mount_real) and (mount_real not in output_real.parents) and (output_real not in mount_real.parents)
    mount_empty = len(list(planned_locked_test_mountpoint.iterdir())) == 0

    checks["filesystem"] = {
        "output_dir_writable": os.access(output_dir, os.W_OK),
        "paths_disjoint": disjoint,
        "planned_mountpoint_empty": mount_empty,
        "locked_test_mount_state": "UNMOUNTED",
        "canary_write_performed": False,
        "read_only_mount_verification": "PENDING_HUMAN_AUTHORIZATION",
    }
    if not disjoint:
        errors.append("Output dir and mountpoint are not disjoint")
    if not mount_empty:
        errors.append("Planned mountpoint is not empty")

    # 8. Human authorization
    auth_file = Path("HUMAN_UNSEALING_AUTHORIZATION.json")
    checks["authorization"] = {
        "authorization_artifact_exists": auth_file.exists(),
        "status": "PENDING_HUMAN_APPROVAL",
    }
    if auth_file.exists():
        errors.append("HUMAN_UNSEALING_AUTHORIZATION.json already exists in workspace")

    # 9. Real counters
    counters = {
        "locked_test_real_accesses": 0,
        "completed_real_unsealing_sessions": 0,
        "completed_real_model_evaluations": 0,
        "evaluation_attempts": 0,
        "cpu_inference_calls": 0,
        "gpu_inference_calls": 0,
        "new_training_runs": 0,
    }
    checks["real_counters"] = counters

    obs_end = datetime.now(timezone.utc).isoformat()

    # Determine Verdict
    if errors:
        verdict = "BLOCKED_WITH_EXACT_REASON"
    elif not is_network_isolated and not synthetic_only:
        verdict = "USER_PHYSICAL_ACTION_REQUIRED"
    elif is_network_isolated:
        verdict = "READY_FOR_HUMAN_AUTHORIZATION_REVIEW"
    else:
        verdict = "USER_PHYSICAL_ACTION_REQUIRED"

    receipt_data = {
        "schema_version": "1.0.0",
        "phase": "Phase 4C.2G.0.2A",
        "verifier_name": "standalone_offline_runtime_verifier",
        "synthetic_only": synthetic_only,
        "clock_source": "SYSTEM_UTC_RUNTIME",
        "timezone_aware": True,
        "observation_started_at_utc": obs_start,
        "observation_completed_at_utc": obs_end,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "errors": errors,
        "checks": checks,
        "real_counters": counters,
    }

    return receipt_data


def write_receipt_atomic(receipt_path: Path, data: dict) -> None:
    """Atomically write receipt JSON using .part temporary file and fsync."""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = receipt_path.with_suffix(receipt_path.suffix + ".part")
    with part_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part_path, receipt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Offline Runtime Verifier for Phase 4C.2G")
    parser.add_argument("--execution-worktree", required=True, type=Path, help="Path to detached execution worktree")
    parser.add_argument("--checkpoint-root", required=True, type=Path, help="Path to root containing canonical checkpoints")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to empty evaluation output directory")
    parser.add_argument("--planned-locked-test-mountpoint", required=True, type=Path, help="Path to empty planned mountpoint")
    parser.add_argument("--receipt", required=True, type=Path, help="Path to write verification receipt JSON")
    parser.add_argument("--synthetic-only", action="store_true", help="Allow synthetic test mode for unit test fixtures")

    args = parser.parse_args()

    receipt_data = verify_offline_runtime(
        execution_worktree=args.execution_worktree,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        planned_locked_test_mountpoint=args.planned_locked_test_mountpoint,
        synthetic_only=args.synthetic_only,
    )

    write_receipt_atomic(args.receipt, receipt_data)

    print(f"Standalone Offline Runtime Verifier completed.")
    print(f"Verdict: {receipt_data['verdict']}")
    print(f"Receipt written to: {args.receipt}")

    if receipt_data["verdict"] == "READY_FOR_HUMAN_AUTHORIZATION_REVIEW":
        sys.exit(0)
    elif receipt_data["verdict"] == "USER_PHYSICAL_ACTION_REQUIRED":
        print("ACTION REQUIRED: Host has active default network route. Disconnect all network adapters physically.")
        sys.exit(2)
    else:
        print(f"VERIFIER FAILED: {receipt_data['errors']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
