"""Receipt validators for the future Windows authorized execution session."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from ml.evaluation.phase_4c2g_dataset import sha256_file


BLOCKED_READ_ONLY_REASON = "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY"
MAX_RECEIPT_AGE = datetime.timedelta(minutes=5)
MAX_FUTURE_SKEW = datetime.timedelta(seconds=60)
WINDOWS_OS_READ_ONLY_EVIDENCE = {
    "windows_disk_is_read_only",
    "windows_cdrom_volume",
    "windows_read_only_virtual_disk",
}


def _load_receipt(path: Path | str) -> tuple[Path, dict[str, Any]]:
    receipt_path = Path(path).resolve(strict=True)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RuntimeError(f"Receipt must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Malformed runtime receipt: {receipt_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime receipt root must be an object: {receipt_path}")
    return receipt_path, payload


def _verify_session_and_time(
    payload: dict[str, Any],
    *,
    expected_session_id: str,
    now_utc: datetime.datetime,
    receipt_kind: str,
) -> None:
    if payload.get("session_id") != expected_session_id:
        raise RuntimeError(
            f"{receipt_kind} session mismatch: expected {expected_session_id!r}, "
            f"got {payload.get('session_id')!r}"
        )
    try:
        verified_at = datetime.datetime.fromisoformat(
            str(payload["verified_at_utc"]).replace("Z", "+00:00")
        )
    except Exception as exc:
        raise RuntimeError(f"Invalid {receipt_kind} verified_at_utc: {exc}") from exc
    if verified_at.tzinfo is None or now_utc.tzinfo is None:
        raise RuntimeError(f"{receipt_kind} timestamps must be timezone-aware")
    age = now_utc - verified_at
    if age > MAX_RECEIPT_AGE:
        raise RuntimeError(
            f"stale {receipt_kind} receipt: age {age.total_seconds():.0f}s exceeds "
            f"{MAX_RECEIPT_AGE.total_seconds():.0f}s"
        )
    if age < -MAX_FUTURE_SKEW:
        raise RuntimeError(f"{receipt_kind} receipt timestamp is in the future")


def verify_isolation_receipt(
    receipt_path: Path | str,
    *,
    expected_session_id: str,
    now_utc: datetime.datetime | None = None,
) -> dict[str, Any]:
    path, payload = _load_receipt(receipt_path)
    now = now_utc or datetime.datetime.now(datetime.timezone.utc)
    _verify_session_and_time(
        payload,
        expected_session_id=expected_session_id,
        now_utc=now,
        receipt_kind="isolation",
    )
    required_zero_arrays = (
        "active_egress_adapters",
        "active_vpn_route_owners",
        "unidentified_route_owners",
    )
    valid = (
        payload.get("isolation_verified") is True
        and payload.get("proxy_enabled") is False
        and payload.get("remaining_active_default_routes") == 0
        and all(payload.get(field) == [] for field in required_zero_arrays)
    )
    if not valid:
        raise RuntimeError("Current-session network isolation receipt is not fail-closed clean.")
    return {
        **payload,
        "status": "FRESH_SESSION_ISOLATION_VERIFIED",
        "receipt_sha256": sha256_file(path),
        "outbound_probes_transmitted": 0,
    }


def verify_windows_read_only_receipt(
    receipt_path: Path | str,
    *,
    locked_test_root: Path | str,
    expected_session_id: str,
    now_utc: datetime.datetime | None = None,
) -> dict[str, Any]:
    path, payload = _load_receipt(receipt_path)
    now = now_utc or datetime.datetime.now(datetime.timezone.utc)
    _verify_session_and_time(
        payload,
        expected_session_id=expected_session_id,
        now_utc=now,
        receipt_kind="read-only",
    )
    root = Path(locked_test_root).resolve(strict=True)
    recorded_root = Path(str(payload.get("locked_test_root", ""))).resolve(strict=False)
    path_matches = str(root).casefold() == str(recorded_root).casefold()
    evidence_kind = payload.get("evidence_kind")
    backing_identity_present = bool(payload.get("backing_volume_unique_id"))
    if evidence_kind == "windows_disk_is_read_only":
        backing_identity_present = backing_identity_present and isinstance(
            payload.get("backing_disk_number"), int
        )
    valid = (
        path_matches
        and payload.get("status") == "READ_ONLY_VOLUME_VERIFIED"
        and payload.get("os_enforced_read_only") is True
        and evidence_kind in WINDOWS_OS_READ_ONLY_EVIDENCE
        and backing_identity_present
        and payload.get("canary_writes_performed") == 0
    )
    if not valid:
        raise RuntimeError(
            f"{BLOCKED_READ_ONLY_REASON}: backing volume/mount has no acceptable "
            "operating-system read-only evidence; folder ReadOnly attributes and "
            "write probes are not accepted."
        )
    return {**payload, "receipt_sha256": sha256_file(path)}
