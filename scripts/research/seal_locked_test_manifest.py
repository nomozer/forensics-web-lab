#!/usr/bin/env python3
"""Role-separated, evaluator-independent locked-test manifest custodian.

The real session is intentionally unavailable without an exact authorization,
fresh isolation/read-only receipts, and an empty external output directory.
This module never imports model/evaluator packages or image decoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import unicodedata
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence


CANONICAL_SPLIT_SEAL = (
    "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
)
INVENTORY_METADATA_NAME = "custodian_inventory.json"
MANIFEST_NAME = "locked_test_manifest.json"
SIDECAR_NAME = "locked_test_manifest.json.sha256"
RESERVATION_NAME = "custodian_access_reservation.json"
COMMITMENT_RECEIPT_NAME = "manifest_commitment_receipt.json"
INTERRUPTED_RECEIPT_NAME = "manifest_custodian_interrupted_receipt.json"
ALLOWED_READ_ONLY_EVIDENCE = {
    "windows_disk_is_read_only",
    "windows_cdrom_volume",
    "windows_read_only_virtual_disk",
    "synthetic_read_only_fixture",
}


def _normalize_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            unicodedata.normalize("NFC", str(key)): _normalize_json_strings(item)
            for key, item in value.items()
        }
    return value


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return the preregistered deterministic manifest representation."""
    normalized = _normalize_json_strings(dict(manifest))
    samples = normalized.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest samples must be an array.")
    normalized["samples"] = sorted(
        samples,
        key=lambda item: (
            str(item["unique_source_id"]),
            int(item["label_id"]),
            str(item["sample_id"]),
        ),
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite sealed output: {path}")
    part = path.with_name(path.name + ".part")
    if part.exists():
        raise RuntimeError(f"Interrupted partial output requires adjudication: {part}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with part.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(path, data)


def _load_regular_json(path: Path | str) -> tuple[Path, dict[str, Any]]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"JSON input must be a regular non-symlink file: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input root must be an object: {target}")
    return target, payload


def _safe_relative_path(value: str) -> PurePosixPath:
    normalized = unicodedata.normalize("NFC", value)
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if (
        not value
        or value != normalized
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or posix == PurePosixPath(".")
    ):
        raise ValueError(f"Unsafe or non-canonical relative_path: {value!r}")
    return posix


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def _inventory_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or _is_reparse(entry_stat):
                    raise ValueError(f"Symlink/reparse point forbidden: {entry.path}")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"Non-regular locked-test entry forbidden: {entry.path}")
                relative = path.relative_to(root).as_posix()
                if relative == INVENTORY_METADATA_NAME:
                    continue
                canonical = _safe_relative_path(relative).as_posix()
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                files[canonical] = resolved

    visit(root)
    return files


def _verify_runtime_receipts(
    *,
    locked_test_root: Path | str,
    session_id: str,
    read_only_receipt_path: Path | str,
    isolation_receipt_path: Path | str,
    allow_synthetic_read_only: bool,
) -> tuple[dict[str, Any], str]:
    read_only_path, read_only = _load_regular_json(read_only_receipt_path)
    isolation_path, isolation = _load_regular_json(isolation_receipt_path)
    evidence = read_only.get("evidence_kind")
    accepted = set(ALLOWED_READ_ONLY_EVIDENCE)
    if not allow_synthetic_read_only:
        accepted.discard("synthetic_read_only_fixture")
    expected_root = os.path.normcase(os.path.abspath(os.fspath(locked_test_root)))
    recorded_root = os.path.normcase(
        os.path.abspath(str(read_only.get("locked_test_root", "")))
    )
    read_only_valid = (
        read_only.get("session_id") == session_id
        and recorded_root == expected_root
        and read_only.get("status") == "READ_ONLY_VOLUME_VERIFIED"
        and read_only.get("os_enforced_read_only") is True
        and evidence in accepted
        and bool(read_only.get("backing_volume_unique_id"))
        and read_only.get("canary_writes_performed") == 0
    )
    if not read_only_valid:
        raise PermissionError("BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY")
    isolation_valid = (
        isolation.get("session_id") == session_id
        and isolation.get("isolation_verified") is True
        and isolation.get("proxy_enabled") is False
        and isolation.get("remaining_active_default_routes") == 0
        and isolation.get("active_egress_adapters") == []
        and isolation.get("active_vpn_route_owners") == []
        and isolation.get("unidentified_route_owners") == []
    )
    if not isolation_valid:
        raise PermissionError("BLOCKED_NETWORK_ISOLATION_RECEIPT_INVALID")
    proof = {
        "status": "READ_ONLY_VOLUME_VERIFIED",
        "os_enforced_read_only": True,
        "evidence_kind": evidence,
        "backing_volume_unique_id": read_only["backing_volume_unique_id"],
        "canary_writes_performed": 0,
        "receipt_sha256": sha256_file(read_only_path),
    }
    return proof, sha256_file(isolation_path)


def _validate_and_hash_inventory(
    *,
    root: Path,
    inventory: Mapping[str, Any],
    expected_split_seal: str,
    progress: dict[str, int],
) -> tuple[dict[str, Any], int, int]:
    if set(inventory) != {"schema_version", "dataset_id", "partition", "samples"}:
        raise ValueError("Inventory metadata has an unexpected top-level field set.")
    if inventory.get("schema_version") != "1.0.0" or inventory.get(
        "partition"
    ) != "locked_test":
        raise ValueError("Inventory schema_version/partition mismatch.")
    samples = inventory.get("samples")
    if not isinstance(samples, list) or len(samples) != 686:
        raise ValueError("Locked-test inventory must contain exactly 686 samples.")
    disk_files = _inventory_files(root)
    expected_paths: set[str] = set()
    sample_ids: set[str] = set()
    source_label_pairs: set[tuple[str, int]] = set()
    labels_by_source: dict[str, set[int]] = defaultdict(set)
    sealed_samples: list[dict[str, Any]] = []
    for raw in samples:
        if not isinstance(raw, dict):
            raise ValueError("Inventory sample must be an object.")
        required = {
            "sample_id",
            "unique_source_id",
            "relative_path",
            "expected_sha256",
            "label",
            "label_id",
            "partition",
        }
        if set(raw) != required:
            raise ValueError("Inventory sample has an unexpected field set.")
        sample_id = str(raw["sample_id"])
        source_id = str(raw["unique_source_id"])
        relative = _safe_relative_path(str(raw["relative_path"])).as_posix()
        label = str(raw["label"])
        label_id = int(raw["label_id"])
        if raw["partition"] != "locked_test":
            raise ValueError("Sample partition must be locked_test.")
        if (label, label_id) not in {("authentic", 0), ("ai_edited", 1)}:
            raise ValueError("Label mapping mismatch.")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if relative in expected_paths:
            raise ValueError(f"Duplicate relative_path: {relative}")
        pair = (source_id, label_id)
        if pair in source_label_pairs:
            raise ValueError(f"Duplicate source/label pair: {pair}")
        if relative not in disk_files:
            raise ValueError(f"Missing locked-test file: {relative}")
        digest = sha256_file(disk_files[relative])
        progress["custodian_files_hashed"] += 1
        expected_digest = str(raw["expected_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or digest != expected_digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        sample_ids.add(sample_id)
        expected_paths.add(relative)
        source_label_pairs.add(pair)
        labels_by_source[source_id].add(label_id)
        sealed_samples.append(
            {
                "sample_id": sample_id,
                "unique_source_id": source_id,
                "relative_path": relative,
                "sha256": digest,
                "label": label,
                "label_id": label_id,
                "partition": "locked_test",
            }
        )
    if set(disk_files) != expected_paths:
        raise ValueError(
            f"Locked-test file inventory mismatch: missing={sorted(expected_paths-set(disk_files))}, "
            f"extra={sorted(set(disk_files)-expected_paths)}"
        )
    if len(labels_by_source) != 343 or any(
        labels != {0, 1} for labels in labels_by_source.values()
    ):
        raise ValueError("Locked-test inventory must contain 343 exact source pairs.")
    split_payload = json.dumps(sorted(labels_by_source)).encode("utf-8")
    split_seal = hashlib.sha256(split_payload).hexdigest()
    if split_seal != expected_split_seal:
        raise ValueError(
            f"Locked source-ID split seal mismatch: expected {expected_split_seal}, got {split_seal}"
        )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": str(inventory["dataset_id"]),
        "partition": "locked_test",
        "samples": sealed_samples,
    }
    return manifest, len(labels_by_source), len(sealed_samples)


def seal_manifest_session(
    *,
    locked_test_root: Path | str,
    output_root: Path | str,
    session_id: str,
    read_only_receipt_path: Path | str,
    isolation_receipt_path: Path | str,
    manifest_schema_path: Path | str,
    expected_split_seal: str = CANONICAL_SPLIT_SEAL,
    allow_synthetic_read_only: bool = False,
    after_reservation: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Reserve one access session, hash exact bytes, and atomically seal outputs."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("SECOND_CUSTODIAN_SESSION_REJECTED_OR_ADJUDICATION_REQUIRED")
    schema_path = Path(manifest_schema_path)
    if not schema_path.is_file() or schema_path.is_symlink():
        raise ValueError("Manifest schema must be a regular non-symlink file.")
    read_only_proof, isolation_sha256 = _verify_runtime_receipts(
        locked_test_root=locked_test_root,
        session_id=session_id,
        read_only_receipt_path=read_only_receipt_path,
        isolation_receipt_path=isolation_receipt_path,
        allow_synthetic_read_only=allow_synthetic_read_only,
    )
    reserved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    reservation = {
        "event": "CUSTODIAN_ACCESS_RESERVED",
        "session_id": session_id,
        "reserved_at_utc": reserved_at,
        "counters": {
            "custodian_manifest_access_sessions": 1,
            "custodian_files_hashed": 0,
            "completed_real_unsealing_sessions": 0,
            "completed_real_model_evaluations": 0,
            "evaluation_attempts": 0,
        },
        "automatic_retry_permitted": False,
    }
    _atomic_json(output / RESERVATION_NAME, reservation)
    progress = {"custodian_files_hashed": 0}
    try:
        if after_reservation is not None:
            after_reservation()
        root = Path(locked_test_root).resolve(strict=True)
        inventory_path = root / INVENTORY_METADATA_NAME
        _, inventory = _load_regular_json(inventory_path)
        manifest, source_count, sample_count = _validate_and_hash_inventory(
            root=root,
            inventory=inventory,
            expected_split_seal=expected_split_seal,
            progress=progress,
        )
        manifest_bytes = canonical_manifest_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_write(output / MANIFEST_NAME, manifest_bytes)
        _atomic_write(output / SIDECAR_NAME, manifest_sha256.encode("ascii"))
        completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        counters = {
            "custodian_manifest_access_sessions": 1,
            "custodian_files_hashed": sample_count,
            "completed_real_unsealing_sessions": 0,
            "completed_real_model_evaluations": 0,
            "evaluation_attempts": 0,
        }
        receipt = {
            "manifest_sha256": manifest_sha256,
            "manifest_bytes": len(manifest_bytes),
            "manifest_schema_sha256": sha256_file(schema_path),
            "source_count": source_count,
            "sample_count": sample_count,
            "locked_test_split_seal": expected_split_seal,
            "custodian_tool_sha256": sha256_file(Path(__file__)),
            "session_id": session_id,
            "reserved_at_utc": reserved_at,
            "completed_at_utc": completed_at,
            "read_only_proof": read_only_proof,
            "network_isolation_receipt_sha256": isolation_sha256,
            "counters": counters,
            "verdict": "MANIFEST_COMMITMENT_SEALED",
        }
        _atomic_json(output / COMMITMENT_RECEIPT_NAME, receipt)
        return receipt
    except BaseException as exc:
        interrupted = {
            "schema_version": "1.0.0",
            "session_id": session_id,
            "event": "CUSTODIAN_ACCESS_INTERRUPTED",
            "interrupted_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "reason_type": type(exc).__name__,
            "counters": {
                **reservation["counters"],
                "custodian_files_hashed": progress["custodian_files_hashed"],
            },
            "automatic_retry_permitted": False,
            "required_action": "HUMAN_ADJUDICATION_REQUIRED",
        }
        try:
            _atomic_json(output / INTERRUPTED_RECEIPT_NAME, interrupted)
        except Exception:
            pass
        raise


def verify_custodian_authorization(
    *,
    authorization_file: Path | str,
    authorization_schema: Path | str,
    package_binding: Path | str,
    package_archive: Path | str,
) -> dict[str, Any]:
    """Validate structural authorization plus exact external package bindings."""
    import jsonschema

    _, authorization = _load_regular_json(authorization_file)
    schema_path, schema = _load_regular_json(authorization_schema)
    binding_path, binding = _load_regular_json(package_binding)
    archive = Path(package_archive)
    if not archive.is_file() or archive.is_symlink():
        raise PermissionError("Custodian package must be a regular non-symlink file.")
    jsonschema.validate(
        authorization,
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    if binding.get("status") != "READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL":
        raise PermissionError("Custodian package binding is not active.")
    archive_binding = binding.get("archive")
    if not isinstance(archive_binding, dict):
        raise PermissionError("Custodian package binding has no archive record.")
    actual_archive = {
        "filename": archive.name,
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
    }
    for field, actual in actual_archive.items():
        if archive_binding.get(field) != actual:
            raise PermissionError(f"Custodian archive binding mismatch: {field}")
    if binding.get("authorization_schema", {}).get("sha256") != sha256_file(
        schema_path
    ):
        raise PermissionError("Custodian authorization schema binding mismatch.")
    required = {
        "sealed_package_sha256": archive_binding["sha256"],
        "sealed_package_bytes": archive_binding["bytes"],
        "custodian_tool_commit": binding.get("effective_custodian_commit"),
        "custodian_package_commit": binding.get("custodian_package_commit"),
        "custodian_component_hashes": binding.get("authorization_component_hashes"),
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"Custodian authorization binding mismatch: {field}")
    return {
        "status": "CUSTODIAN_AUTHORIZATION_AND_PACKAGE_VERIFIED",
        "authorization_id": authorization["authorization_id"],
        "binding_sha256": sha256_file(binding_path),
        "custodian_manifest_access_sessions": 0,
        "completed_real_model_evaluations": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-preflight-only", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-schema", type=Path)
    parser.add_argument("--package-binding", type=Path)
    parser.add_argument("--package-archive", type=Path)
    parser.add_argument("--locked-test-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--read-only-receipt", type=Path)
    parser.add_argument("--isolation-receipt", type=Path)
    parser.add_argument("--manifest-schema", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.authorization_preflight_only:
        required = (
            args.authorization_file,
            args.authorization_schema,
            args.package_binding,
            args.package_archive,
        )
        if any(value is None for value in required):
            raise SystemExit("Authorization preflight arguments are incomplete.")
        result = verify_custodian_authorization(
            authorization_file=args.authorization_file,
            authorization_schema=args.authorization_schema,
            package_binding=args.package_binding,
            package_archive=args.package_archive,
        )
    else:
        required = (
            args.locked_test_root,
            args.output_root,
            args.session_id,
            args.read_only_receipt,
            args.isolation_receipt,
            args.manifest_schema,
        )
        if any(value is None for value in required):
            raise SystemExit("Manifest sealing arguments are incomplete.")
        result = seal_manifest_session(
            locked_test_root=args.locked_test_root,
            output_root=args.output_root,
            session_id=args.session_id,
            read_only_receipt_path=args.read_only_receipt,
            isolation_receipt_path=args.isolation_receipt,
            manifest_schema_path=args.manifest_schema,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
