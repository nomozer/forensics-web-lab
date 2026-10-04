"""Synthetic-only behavioral tests for the locked-test manifest custodian."""

from __future__ import annotations

import json
import hashlib
import copy
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_synthetic_custodian_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "locked-synthetic"
    output = tmp_path / "sealed-output"
    root.mkdir(parents=True)
    output.mkdir()
    samples = []
    source_ids = [f"synthetic-{index:03d}" for index in range(343)]
    for source_id in source_ids:
        for label, label_id in (("authentic", 0), ("ai_edited", 1)):
            relative_path = f"samples/{source_id}/{label}.bin"
            sample = root / Path(relative_path)
            sample.parent.mkdir(parents=True, exist_ok=True)
            sample.write_bytes(f"{source_id}:{label}".encode("utf-8"))
            samples.append(
                {
                    "sample_id": f"{source_id}-{label}",
                    "unique_source_id": source_id,
                    "relative_path": relative_path,
                    "expected_sha256": _sha256(sample),
                    "label": label,
                    "label_id": label_id,
                    "partition": "locked_test",
                }
            )
    inventory = {
        "schema_version": "1.0.0",
        "dataset_id": "synthetic-custodian",
        "partition": "locked_test",
        "samples": samples,
    }
    (root / "custodian_inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    split_payload = json.dumps(sorted(source_ids)).encode("utf-8")
    split_seal = hashlib.sha256(split_payload).hexdigest()
    session_id = "synthetic-custodian-session-001"
    read_only = tmp_path / "read-only-receipt.json"
    read_only.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "locked_test_root": str(root.resolve()),
                "status": "READ_ONLY_VOLUME_VERIFIED",
                "os_enforced_read_only": True,
                "evidence_kind": "synthetic_read_only_fixture",
                "backing_volume_unique_id": "synthetic-volume",
                "canary_writes_performed": 0,
            }
        ),
        encoding="utf-8",
    )
    isolation = tmp_path / "isolation-receipt.json"
    isolation.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "isolation_verified": True,
                "proxy_enabled": False,
                "remaining_active_default_routes": 0,
                "active_egress_adapters": [],
                "active_vpn_route_owners": [],
                "unidentified_route_owners": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "output": output,
        "session_id": session_id,
        "read_only": read_only,
        "isolation": isolation,
        "split_seal": split_seal,
    }


def test_canonical_manifest_bytes_are_deterministic_utf8_nfc_and_no_newline() -> None:
    from scripts.research.seal_locked_test_manifest import canonical_manifest_bytes

    samples = [
        {
            "sample_id": "source-002-ai_edited",
            "unique_source_id": "source-002",
            "relative_path": "samples/source-002/ai_edited.bin",
            "sha256": "b" * 64,
            "label": "ai_edited",
            "label_id": 1,
            "partition": "locked_test",
        },
        {
            "sample_id": "source-001-authentic",
            "unique_source_id": "source-001",
            "relative_path": "samples/source-001/cafe\u0301.bin",
            "sha256": "a" * 64,
            "label": "authentic",
            "label_id": 0,
            "partition": "locked_test",
        },
    ]
    first = {
        "partition": "locked_test",
        "samples": samples,
        "dataset_id": "synthetic-custodian",
        "schema_version": "1.0.0",
    }
    second = {**first, "samples": list(reversed(samples))}

    canonical = canonical_manifest_bytes(first)

    assert canonical == canonical_manifest_bytes(second)
    assert canonical == canonical.decode("utf-8").encode("utf-8")
    assert b"cafe\\u0301" not in canonical
    assert canonical.endswith(b"}") and not canonical.endswith(b"\n")
    decoded = json.loads(canonical)
    assert [item["sample_id"] for item in decoded["samples"]] == [
        "source-001-authentic",
        "source-002-ai_edited",
    ]


def test_synthetic_custodian_session_seals_manifest_and_public_receipt(
    tmp_path: Path,
) -> None:
    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    fixture = _build_synthetic_custodian_fixture(tmp_path)
    receipt = seal_manifest_session(
        locked_test_root=fixture["root"],
        output_root=fixture["output"],
        session_id=fixture["session_id"],
        read_only_receipt_path=fixture["read_only"],
        isolation_receipt_path=fixture["isolation"],
        manifest_schema_path=REPO_ROOT
        / "docs/schemas/locked-test-manifest.v1.schema.json",
        expected_split_seal=fixture["split_seal"],
        allow_synthetic_read_only=True,
    )

    output = fixture["output"]
    manifest_path = output / "locked_test_manifest.json"
    sidecar_path = output / "locked_test_manifest.json.sha256"
    assert receipt["verdict"] == "MANIFEST_COMMITMENT_SEALED"
    assert receipt["manifest_sha256"] == _sha256(manifest_path)
    assert sidecar_path.read_text(encoding="ascii") == receipt["manifest_sha256"]
    assert receipt["source_count"] == 343
    assert receipt["sample_count"] == 686
    assert receipt["counters"] == {
        "custodian_manifest_access_sessions": 1,
        "custodian_files_hashed": 686,
        "completed_real_unsealing_sessions": 0,
        "completed_real_model_evaluations": 0,
        "evaluation_attempts": 0,
    }
    public_receipt = json.loads(
        (output / "manifest_commitment_receipt.json").read_text(encoding="utf-8")
    )
    assert "samples" not in public_receipt
    assert not any(path.name.endswith(".part") for path in output.iterdir())


def test_custodian_rejects_malformed_inventory_and_records_interruption(
    tmp_path: Path,
) -> None:
    import pytest

    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    mutation_names = (
        "label_mapping",
        "missing_sample",
        "extra_sample",
        "duplicate_sample_id",
        "duplicate_path",
        "duplicate_source_label",
        "absolute_path",
        "traversal_path",
        "unc_path",
        "checksum",
        "split_seal",
        "extra_file",
    )
    for index, mutation in enumerate(mutation_names):
        case_root = tmp_path / f"case-{index:02d}"
        fixture = _build_synthetic_custodian_fixture(case_root)
        inventory_path = fixture["root"] / "custodian_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        samples = inventory["samples"]
        expected_split_seal = fixture["split_seal"]
        if mutation == "label_mapping":
            samples[0]["label_id"] = 1
        elif mutation == "missing_sample":
            samples.pop()
        elif mutation == "extra_sample":
            samples.append(copy.deepcopy(samples[-1]))
        elif mutation == "duplicate_sample_id":
            samples[1]["sample_id"] = samples[0]["sample_id"]
        elif mutation == "duplicate_path":
            samples[1]["relative_path"] = samples[0]["relative_path"]
        elif mutation == "duplicate_source_label":
            samples[2]["unique_source_id"] = samples[0]["unique_source_id"]
        elif mutation == "absolute_path":
            samples[0]["relative_path"] = "/absolute.bin"
        elif mutation == "traversal_path":
            samples[0]["relative_path"] = "../escape.bin"
        elif mutation == "unc_path":
            samples[0]["relative_path"] = r"\\server\share\escape.bin"
        elif mutation == "checksum":
            samples[0]["expected_sha256"] = "0" * 64
        elif mutation == "split_seal":
            expected_split_seal = "0" * 64
        elif mutation == "extra_file":
            (fixture["root"] / "unexpected.bin").write_bytes(b"unexpected")
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

        with pytest.raises(ValueError):
            seal_manifest_session(
                locked_test_root=fixture["root"],
                output_root=fixture["output"],
                session_id=fixture["session_id"],
                read_only_receipt_path=fixture["read_only"],
                isolation_receipt_path=fixture["isolation"],
                manifest_schema_path=REPO_ROOT
                / "docs/schemas/locked-test-manifest.v1.schema.json",
                expected_split_seal=expected_split_seal,
                allow_synthetic_read_only=True,
            )
        interrupted = json.loads(
            (
                fixture["output"] / "manifest_custodian_interrupted_receipt.json"
            ).read_text(encoding="utf-8")
        )
        assert interrupted["event"] == "CUSTODIAN_ACCESS_INTERRUPTED"
        assert interrupted["automatic_retry_permitted"] is False
        assert interrupted["counters"]["custodian_manifest_access_sessions"] == 1


def test_writable_volume_is_rejected_before_access_reservation(tmp_path: Path) -> None:
    import pytest

    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    session_id = "synthetic-writable-volume"
    read_only = tmp_path / "read-only.json"
    read_only.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "locked_test_root": str(tmp_path / "never-touched-locked-root"),
                "status": "READ_ONLY_VOLUME_VERIFIED",
                "os_enforced_read_only": False,
                "evidence_kind": "synthetic_read_only_fixture",
                "backing_volume_unique_id": "synthetic",
                "canary_writes_performed": 0,
            }
        ),
        encoding="utf-8",
    )
    isolation = tmp_path / "isolation.json"
    isolation.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "isolation_verified": True,
                "proxy_enabled": False,
                "remaining_active_default_routes": 0,
                "active_egress_adapters": [],
                "active_vpn_route_owners": [],
                "unidentified_route_owners": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with pytest.raises(
        PermissionError, match="BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY"
    ):
        seal_manifest_session(
            locked_test_root=tmp_path / "never-touched-locked-root",
            output_root=output,
            session_id=session_id,
            read_only_receipt_path=read_only,
            isolation_receipt_path=isolation,
            manifest_schema_path=REPO_ROOT
            / "docs/schemas/locked-test-manifest.v1.schema.json",
            allow_synthetic_read_only=True,
        )
    assert not (output / "custodian_access_reservation.json").exists()


def test_crash_after_reservation_records_access_and_forbids_retry(tmp_path: Path) -> None:
    import pytest

    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    fixture = _build_synthetic_custodian_fixture(tmp_path)

    def synthetic_crash() -> None:
        raise RuntimeError("synthetic crash after reservation")

    kwargs = {
        "locked_test_root": fixture["root"],
        "output_root": fixture["output"],
        "session_id": fixture["session_id"],
        "read_only_receipt_path": fixture["read_only"],
        "isolation_receipt_path": fixture["isolation"],
        "manifest_schema_path": REPO_ROOT
        / "docs/schemas/locked-test-manifest.v1.schema.json",
        "expected_split_seal": fixture["split_seal"],
        "allow_synthetic_read_only": True,
    }
    with pytest.raises(RuntimeError, match="synthetic crash after reservation"):
        seal_manifest_session(**kwargs, after_reservation=synthetic_crash)
    interrupted = json.loads(
        (
            fixture["output"] / "manifest_custodian_interrupted_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert interrupted["counters"]["custodian_manifest_access_sessions"] == 1
    assert interrupted["required_action"] == "HUMAN_ADJUDICATION_REQUIRED"
    with pytest.raises(RuntimeError, match="SECOND_CUSTODIAN_SESSION_REJECTED"):
        seal_manifest_session(**kwargs)


def test_sealer_has_no_model_image_decoder_or_network_imports() -> None:
    source = (
        REPO_ROOT / "scripts/research/seal_locked_test_manifest.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "import torch",
        "import torchvision",
        "from ml.evaluation",
        "import PIL",
        "from PIL",
        "import socket",
        "urllib",
        "requests",
        "http.client",
    )
    assert all(token not in source for token in forbidden)


def test_symlink_or_reparse_entry_is_rejected_without_following_it(
    tmp_path: Path,
) -> None:
    import pytest

    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    fixture = _build_synthetic_custodian_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"must not be hashed")
    link = fixture["root"] / "escape-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    with pytest.raises(ValueError, match="Symlink/reparse point forbidden"):
        seal_manifest_session(
            locked_test_root=fixture["root"],
            output_root=fixture["output"],
            session_id=fixture["session_id"],
            read_only_receipt_path=fixture["read_only"],
            isolation_receipt_path=fixture["isolation"],
            manifest_schema_path=REPO_ROOT
            / "docs/schemas/locked-test-manifest.v1.schema.json",
            expected_split_seal=fixture["split_seal"],
            allow_synthetic_read_only=True,
        )


def test_custodian_authorization_schema_locks_role_and_prohibitions() -> None:
    import jsonschema
    import pytest

    schema = json.loads(
        (
            REPO_ROOT
            / "docs/schemas/human-manifest-custodian-authorization.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    authorization = {
        "status": "AUTHORIZED",
        "authorization_id": "synthetic-schema-validation-only",
        "authorized_by": "synthetic role fixture",
        "role_separation_mode": "role_separated_automated_custodian_process",
        "authorized_at_utc": "2026-10-03T00:00:00Z",
        "authorization_purpose": "create_locked_test_manifest_commitment_only",
        "maximum_manifest_custodian_sealing_sessions": 1,
        "custodian_tool_commit": "a" * 40,
        "custodian_package_commit": "a" * 40,
        "custodian_component_hashes": {
            "manifest_sealer": "b" * 64,
            "windows_custodian_controller": "c" * 64,
            "custodian_authorization_schema": "d" * 64,
            "locked_test_manifest_schema": "e" * 64,
            "split_lock": "f" * 64,
        },
        "sealed_package_sha256": "1" * 64,
        "sealed_package_bytes": 1,
        "locked_test_split_seal": (
            "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
        ),
        "no_model_execution_acknowledgment": True,
        "no_metric_computation_acknowledgment": True,
        "no_data_modification_acknowledgment": True,
        "no_manifest_content_disclosure_acknowledgment": True,
        "data_integrity_access_not_model_evaluation_acknowledgment": True,
    }
    jsonschema.validate(authorization, schema, format_checker=jsonschema.FormatChecker())
    for field in (
        "no_model_execution_acknowledgment",
        "no_metric_computation_acknowledgment",
        "no_data_modification_acknowledgment",
        "no_manifest_content_disclosure_acknowledgment",
        "data_integrity_access_not_model_evaluation_acknowledgment",
    ):
        mutated = copy.deepcopy(authorization)
        mutated[field] = False
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(mutated, schema)


def test_windows_custodian_controller_contract_is_non_mutating_and_model_free() -> None:
    script = (
        REPO_ROOT
        / "scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ContractValidationOnly",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(completed.stdout)
    assert contract["verdict"] == "MANIFEST_CUSTODIAN_CONTRACT_VALID"
    assert contract["mutations_performed"] == 0
    assert contract["uac_or_session_invoked"] is False
    assert contract["evaluator_invocations"] == 0
    assert contract["ordered_steps"] == [
        "validate_custodian_authorization",
        "validate_exact_custodian_tool_and_package",
        "create_recovery_watchdog",
        "isolate_active_egress_adapters",
        "verify_fresh_same_session_isolation_receipt",
        "verify_os_backed_read_only_locked_test_storage",
        "run_manifest_sealer_once",
        "verify_atomic_commitment_receipt",
        "restore_network",
        "delete_and_read_back_watchdog",
        "stop_without_evaluator",
    ]
    source = script.read_text(encoding="utf-8")
    assert "run_phase_4c2g_confirmatory" not in source
    assert "execute_checkpoint_evaluation" not in source


def test_windows_custodian_formal_preflight_compares_component_key_counts(
    tmp_path: Path,
) -> None:
    import pytest

    if sys.platform != "win32":
        pytest.skip("Windows PowerShell controller regression")
    admin_probe = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "$i=[Security.Principal.WindowsIdentity]::GetCurrent();"
            "$p=New-Object Security.Principal.WindowsPrincipal($i);"
            "$p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if admin_probe.stdout.strip().lower() == "true":
        pytest.skip("Formal controller guard test must run without Administrator rights")

    schema = (
        REPO_ROOT
        / "docs/schemas/human-manifest-custodian-authorization.v1.schema.json"
    )
    archive = tmp_path / "synthetic-custodian-package.tar.gz"
    archive.write_bytes(b"synthetic package binding fixture")
    components = {
        "manifest_sealer": "b" * 64,
        "windows_custodian_controller": "c" * 64,
        "custodian_authorization_schema": "d" * 64,
        "locked_test_manifest_schema": "e" * 64,
        "split_lock": "f" * 64,
    }
    binding = {
        "status": "READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL",
        "effective_custodian_commit": "a" * 40,
        "custodian_package_commit": "a" * 40,
        "archive": {
            "filename": archive.name,
            "sha256": _sha256(archive),
            "bytes": archive.stat().st_size,
        },
        "authorization_schema": {"sha256": _sha256(schema)},
        "authorization_component_hashes": components,
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    authorization = {
        "status": "AUTHORIZED",
        "authorization_id": "synthetic-formal-preflight",
        "authorized_by": "synthetic fixture",
        "role_separation_mode": "role_separated_automated_custodian_process",
        "authorized_at_utc": "2026-10-03T00:00:00Z",
        "authorization_purpose": "create_locked_test_manifest_commitment_only",
        "maximum_manifest_custodian_sealing_sessions": 1,
        "custodian_tool_commit": "a" * 40,
        "custodian_package_commit": "a" * 40,
        "custodian_component_hashes": components,
        "sealed_package_sha256": _sha256(archive),
        "sealed_package_bytes": archive.stat().st_size,
        "locked_test_split_seal": (
            "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
        ),
        "no_model_execution_acknowledgment": True,
        "no_metric_computation_acknowledgment": True,
        "no_data_modification_acknowledgment": True,
        "no_manifest_content_disclosure_acknowledgment": True,
        "data_integrity_access_not_model_evaluation_acknowledgment": True,
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    output = tmp_path / "must-not-be-created"
    script = (
        REPO_ROOT
        / "scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1"
    )
    wrapper = tmp_path / "invoke-controller.ps1"
    escaped_script = str(script).replace("'", "''")
    wrapper.write_text(
        "Import-Module Microsoft.PowerShell.Utility\n"
        f"& '{escaped_script}' @args\n"
        "exit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-ElevatedWorker",
            "-PythonExe",
            sys.executable,
            "-AuthorizationFile",
            str(authorization_path),
            "-AuthorizationSchema",
            str(schema),
            "-PackageArchive",
            str(archive),
            "-PackageBinding",
            str(binding_path),
            "-SealerScript",
            str(REPO_ROOT / "scripts/research/seal_locked_test_manifest.py"),
            "-ManifestSchema",
            str(REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"),
            "-LockedTestRoot",
            str(tmp_path / "never-touched-locked-root"),
            "-OutputRoot",
            str(output),
            "-SessionId",
            "synthetic-formal-preflight",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "Elevated worker does not have Administrator privileges" in combined
    assert "Custodian component binding mismatch" not in combined
    assert not output.exists()


def test_custodian_package_member_allowlist_is_data_and_model_free() -> None:
    from scripts.research.build_phase4c2g_manifest_custodian_package import (
        CUSTODIAN_PACKAGE_MEMBERS,
    )

    assert set(CUSTODIAN_PACKAGE_MEMBERS) == {
        "scripts/research/seal_locked_test_manifest.py",
        "scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1",
        "docs/schemas/human-manifest-custodian-authorization.v1.schema.json",
        "docs/schemas/locked-test-manifest.v1.schema.json",
        "research/evidence/phase-4b.2/split-lock.json",
    }
    assert all(
        not member.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".pt", ".pth", ".onnx")
        )
        for member in CUSTODIAN_PACKAGE_MEMBERS
    )


def test_custodian_package_rebuild_is_deterministic_from_effective_git_objects(
    tmp_path: Path,
) -> None:
    from scripts.research.build_phase4c2g_manifest_custodian_package import (
        build_custodian_package,
    )

    source_binding = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.7/custodian_source_binding.json"
    )
    binding = json.loads(source_binding.read_text(encoding="utf-8"))
    kwargs = {
        "repo_root": REPO_ROOT,
        "effective_custodian_commit": binding["effective_custodian_commit"],
        "custodian_package_commit": binding["custodian_package_commit"],
        "source_binding_path": source_binding,
    }
    first = build_custodian_package(output_path=tmp_path / "first.tar.gz", **kwargs)
    second = build_custodian_package(output_path=tmp_path / "second.tar.gz", **kwargs)
    assert first.sha256 == second.sha256
    assert first.bytes == second.bytes
    assert first.member_count == 7


def test_custodian_source_binding_matches_effective_commit_git_objects() -> None:
    from scripts.research.build_phase4c2g_execution_package import (
        audit_git_commit_components,
    )

    binding = json.loads(
        (
            REPO_ROOT
            / "research/evidence/phase-4c.2g.0.7/custodian_source_binding.json"
        ).read_text(encoding="utf-8")
    )
    audit = audit_git_commit_components(
        repo_root=REPO_ROOT,
        commit=binding["effective_custodian_commit"],
        components=binding["components"],
        require_worktree_match=True,
    )
    assert len(audit) == 5
    assert all(item["bytes_match"] and item["sha256_match"] for item in audit.values())


def test_git_evidence_contains_no_manifest_or_per_sample_contents() -> None:
    evidence = REPO_ROOT / "research/evidence/phase-4c.2g.0.6"
    assert not (evidence / "locked_test_manifest.json").exists()
    assert not (evidence / "locked_test_manifest.json.sha256").exists()
    for path in evidence.iterdir():
        if path.suffix not in {".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert '"sample_id":' not in text
        assert '"unique_source_id":' not in text
        assert '"manifest_sha256":' not in text
